"""SubEvent2Hap Studio web UI: plan, generate and test vibrations from text on Quest controllers.

Run:
    .venv\\Scripts\\python.exe webui\\server.py   ->  http://127.0.0.1:7860
"""
import argparse
import asyncio
import io
import ipaddress
import json
import re
import os
import pathlib
import sys
import threading
import zlib
import time
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

# Fully local: the T5 text encoder used by HapticGen is already in the HF cache, so never contact the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import hapticgen_generate as hg  # noqa: E402
import generators as gens  # noqa: E402
import compose as cp  # noqa: E402
import decompose as dc  # noqa: E402
import bounce as bp  # noqa: E402
import model_planner as mp  # noqa: E402
import event_t2m as et  # noqa: E402
import haptic_patterns as hp  # noqa: E402
import story2hap as s2h  # noqa: E402
from quest_haptic_play import QuestHapticPlayer  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import architecture  # noqa: E402

OUT_DIR = hg.DEFAULT_OUT
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SubEvent2Hap Studio")
player = QuestHapticPlayer().start()

# Only this machine and Tailscale peers may connect, even when bound to 0.0.0.0 (the campus LAN
# must not reach the generator, the OpenAI key or the files).
ALLOWED_NETS = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "::1/128", "100.64.0.0/10", "fd7a:115c:a1e0::/48")]


class _ClientAllowlist:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            host = (scope.get("client") or ("", 0))[0]
            try:
                ok = any(ipaddress.ip_address(host) in n for n in ALLOWED_NETS)
            except ValueError:
                ok = False
            if not ok:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"text/plain")]})
                    await send({"type": "http.response.body", "body": b"forbidden"})
                return
        await self.app(scope, receive, send)


app.add_middleware(_ClientAllowlist)


class RemotePlayer:
    """A playback agent on another machine (scripts/remote_player.py), connected over a websocket.
    While one is connected, playback goes there instead of to a Quest on this machine."""

    def __init__(self):
        self.ws, self.loop, self.name, self.state, self.events = None, None, None, {}, {}

    @property
    def connected(self):
        return self.ws is not None

    def send(self, msg):
        if not self.ws:
            raise HTTPException(503, "원격 재생 장치가 연결되어 있지 않습니다")
        asyncio.run_coroutine_threadsafe(self.ws.send_json(msg), self.loop).result(timeout=5)


remote = RemotePlayer()


@app.websocket("/ws/player")
async def player_socket(ws: WebSocket, name: str = "remote"):
    await ws.accept()
    if remote.ws is not None:   # newest agent wins
        try:
            await remote.ws.close()
        except Exception:  # noqa: BLE001
            pass
    remote.ws, remote.loop, remote.name, remote.state = ws, asyncio.get_running_loop(), name[:40], {}
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "status":
                remote.state = msg
            elif msg.get("type") == "event" and msg.get("tag"):   # playback started / ended (server clock)
                remote.events[msg["tag"]] = {**remote.events.get(msg["tag"], {}), msg["event"]: time.time()}
                if len(remote.events) > 200:
                    remote.events.pop(next(iter(remote.events)))
    except WebSocketDisconnect:
        pass
    finally:
        if remote.ws is ws:
            remote.ws, remote.state = None, {}

_model = None
_model_name = None   # which generator is on the GPU (only one fits next to the LM)
_model_status = "loading"
_swap_lock = threading.Lock()
_model_error = None
_gen_lock = threading.Lock()
_gen_busy = False


def _load_model(name="hapticgen"):
    global _model, _model_name, _model_status, _model_error
    try:
        _model = gens.load(name)
        _model_name = name
        _model_status = "ready"
    except Exception as e:  # noqa: BLE001
        _model_status, _model_error = "error", str(e)


threading.Thread(target=_load_model, daemon=True).start()


# ---------------------------------------------------------------- remote GPU (Colab worker)
COLAB_FILE = OUT_DIR / "colab.json"
_colab = {"url": None, "models": {}, "error": None}
try:
    _colab["url"] = json.loads(COLAB_FILE.read_text(encoding="utf-8")).get("url")
except (OSError, ValueError):
    pass


def _colab_generator(model):
    """name@colab -> RemoteGenerator (first use loads the model on the worker, can take a minute)."""
    if not _colab["url"]:
        raise HTTPException(503, "Colab 주소가 설정되지 않았습니다")
    key = (_colab["url"], model)
    if key not in _colab["models"]:
        try:
            _colab["models"][key] = gens.RemoteGenerator(_colab["url"], model)
            _colab["error"] = None
        except Exception as e:  # noqa: BLE001
            _colab["error"] = f"{type(e).__name__}: {e}"
            raise HTTPException(502, f"Colab 연결 실패: {e}")
    return _colab["models"][key]


class ColabReq(BaseModel):
    url: str = Field(max_length=400)


@app.get("/api/colab")
def colab_status():
    if not _colab["url"]:
        return {"set": False, "ok": False, "status": "주소 없음"}
    import requests
    from urllib.parse import parse_qs, urlsplit
    u = urlsplit(_colab["url"])
    try:
        r = requests.get(f"{u.scheme}://{u.netloc}/health", timeout=5,
                         headers={"X-Token": parse_qs(u.query).get("token", [""])[0]})
        r.raise_for_status()
        h = r.json()
        return {"set": True, "ok": True, "device": h["device"], "loaded": h["loaded"], "status": f"연결됨 ({h['device']})"}
    except Exception as e:  # noqa: BLE001
        return {"set": True, "ok": False, "status": f"연결 안 됨 ({type(e).__name__})"}


@app.post("/api/colab")
def colab_set(req: ColabReq):
    url = req.url.strip() or None
    if url and not re.match(r"^https?://[^\s]+$", url):
        raise HTTPException(422, "주소 형식이 아닙니다")
    _colab.update(url=url, models={}, error=None)
    COLAB_FILE.write_text(json.dumps({"url": url}), encoding="utf-8")
    return colab_status()


def _use_generator(name):
    """Return the requested generator, swapping it onto the GPU if another one is loaded (~30 s).
    "<generator>@colab" runs on the remote worker instead and leaves the local GPU alone.
    "auto" (the default) uses HapticGen on Colab when the worker answers, otherwise the local GPU."""
    global _model, _model_name, _model_status
    if name == "auto":
        if _colab["url"] and colab_status().get("ok"):
            try:
                return _colab_generator("hapticgen")
            except HTTPException:
                pass
        name = "hapticgen"
    if name.endswith("@colab"):
        return _colab_generator(name.removesuffix("@colab"))
    if name not in gens.GENERATORS:
        raise HTTPException(422, f"unknown generator {name}")
    with _swap_lock:
        if _model is not None and _model_name == name:
            return _model
        if _model is None and _model_status == "loading":   # initial load still running
            raise HTTPException(503, "model loading")
        import gc
        import torch
        _model, _model_name, _model_status = None, None, f"switching to {name}"
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001  a failed cache flush must not leave the server stuck
            pass
        _load_model(name)
        if _model is None:
            _model_status = "error"  # next request retries the load instead of returning 503 forever
            raise HTTPException(500, f"generator load failed: {_model_error}")
        return _model


# ---------------------------------------------------------------- helpers
def _wav_path(name: str, top_level=False) -> pathlib.Path:
    """Resolve a wav under OUT_DIR (sub-folders allowed unless top_level), rejecting traversal."""
    root = OUT_DIR.resolve()
    p = (OUT_DIR / name).resolve()
    inside = p.parent == root if top_level else p.is_relative_to(root)
    if not inside or p.suffix != ".wav" or not p.exists():
        raise HTTPException(404, "file not found")
    return p


def _peaks(w: np.ndarray, n=400):
    """Min/max pairs for waveform drawing."""
    if len(w) == 0:
        return []
    chunks = np.array_split(w, min(n, len(w)))
    return [[round(float(c.min()), 4), round(float(c.max()), 4)] for c in chunks]


def _describe(p: pathlib.Path):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    a = np.abs(w)
    meta_path = p.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "name": p.name,
        "prompt": meta.get("prompt", p.stem.rsplit("_", 1)[0].replace("_", " ")),
        "params": meta.get("params", {}),
        "created": meta.get("created", p.stat().st_mtime),
        "duration": len(w) / sr,
        "sr": sr,
        "peak": float(a.max()) if len(a) else 0.0,
        "rms": float(np.sqrt((w ** 2).mean())) if len(w) else 0.0,
        "ok": bool(len(a) and a.max() >= hg.MIN_ACCEPTABLE_AMPLITUDE
                   and np.quantile(a, 0.995) >= hg.MIN_ACCEPTABLE_995TH_AMPLITUDE),
        "peaks": _peaks(w),
    }


# ---------------------------------------------------------------- API
class GenerateReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    duration: float = Field(5.0, ge=0.5, le=10.0)
    num: int = Field(3, ge=1, le=6)
    top_k: int = Field(250, ge=1, le=2048)
    cfg: float = Field(3.0, ge=0.0, le=10.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    seed: int | None = None


class PlayReq(BaseModel):
    name: str
    hand: str = Field("both", pattern="^(both|left|right)$")
    gain: float = Field(1.0, ge=0.0, le=4.0)
    loop: int = Field(1, ge=1, le=20)
    normalize: bool = False


def _load(name: str, normalize: bool):
    w, sr = sf.read(_wav_path(name), dtype="float32")
    if normalize:
        peak = float(np.abs(w).max())
        if peak > 1e-4:
            w = w / peak
    return w, sr


@app.get("/")
def index():
    return FileResponse(pathlib.Path(__file__).with_name("index.html"))


@app.get("/api/status")
def status():
    return {
        "model": _model_status,
        "generator": _model_name,
        "model_error": _model_error,
        "generating": _gen_busy,
        "quest": ({"status": f"{remote.name}: {remote.state.get('status', '연결됨')}", "remote": remote.name,
                   "connected": bool(remote.state.get("connected")), "ready": bool(remote.state.get("ready")),
                   "playing": bool(remote.state.get("playing"))} if remote.connected else
                  {"status": f"서버 PC (노트북 재생기 미연결): {player.status}", "connected": player.connected,
                   "ready": player.status.startswith("준비됨") or player.status == "재생 중",
                   "playing": player.is_playing}),
    }


@app.post("/api/generate")
def generate(req: GenerateReq):
    global _gen_busy
    if _model is None:
        raise HTTPException(503, f"model {_model_status}")
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "already generating")
    _gen_busy = True
    try:
        t0 = time.time()
        params = req.model_dump(exclude={"prompt"})
        results = hg.generate(_use_generator("hapticgen"), req.prompt, req.duration, req.num, req.top_k, req.cfg,
                              req.temperature, req.seed, OUT_DIR)
        for r in results:
            r["path"].with_suffix(".json").write_text(json.dumps(
                {"prompt": req.prompt, "params": params, "created": time.time()}, ensure_ascii=False), encoding="utf-8")
        return {"elapsed": time.time() - t0, "items": [_describe(r["path"]) for r in results]}
    finally:
        _gen_busy = False
        _gen_lock.release()


@app.get("/api/library")
def library():
    files = sorted(OUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_describe(p) for p in files]


@app.delete("/api/library/{name}")
def delete(name: str):
    p = _wav_path(name, top_level=True)
    p.unlink()
    p.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/audio/{name:path}")
def audio(name: str, gain: float = 1.0, normalize: bool = False):
    """16-bit PCM copy for browser preview (8 kHz float wavs aren't universally supported)."""
    w, sr = _load(name, normalize)
    buf = io.BytesIO()
    sf.write(buf, np.clip(w * gain, -1, 1), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.post("/api/play")
def play(req: PlayReq):
    if remote.connected:
        _wav_path(req.name)
        remote.send({"cmd": "play", "path": f"/audio/{req.name}?normalize={str(req.normalize).lower()}",
                     "hand": req.hand, "gain": req.gain, "loop": req.loop})
        return {"ok": True, "status": "원격 재생"}
    if not player.connected:
        raise HTTPException(503, player.status)
    w, sr = _load(req.name, req.normalize)
    player.play(w, sr, req.hand, req.gain, req.loop)
    return {"ok": True, "status": player.status}


@app.get("/api/wave/{name:path}")
def wave(name: str):
    return _describe(_wav_path(name))


# ---------------------------------------------------------------- sub-event composition
def _llm_models():
    """LM Studio models: [{id, type, state}]. Uses the v0 REST API which reports load state."""
    try:
        with urllib.request.urlopen(dc.LMSTUDIO_URL.replace("/v1", "/api/v0/models"), timeout=2) as r:
            return [m for m in json.loads(r.read())["data"] if m.get("type") in ("llm", "vlm")]
    except Exception:  # noqa: BLE001
        return None


def _pick_llm(models):
    loaded = [m["id"] for m in models if m.get("state") == "loaded"]
    for pref in ("qwen3-4b",):
        for mid in loaded + [m["id"] for m in models]:
            if pref in mid:
                return mid
    return loaded[0] if loaded else (models[0]["id"] if models else None)


@app.get("/api/llm/status")
def llm_status():
    models = _llm_models()
    if models is None:
        return {"available": False, "status": "LM Studio 서버에 연결할 수 없음 (lms server start)"}
    mid = _pick_llm(models)
    loaded = any(m["id"] == mid and m.get("state") == "loaded" for m in models)
    return {"available": mid is not None, "model": mid, "loaded": loaded,
            "status": f"{mid} ({'로드됨' if loaded else '요청 시 로드'})" if mid else "LLM 모델 없음"}


class DecomposeReq(BaseModel):
    event: str = Field(min_length=1, max_length=400)
    # Default = staged: in the eval_set_v2.1 ablation it rendered every prompt and passed 81% of items
    # vs 67% / 28% for model_first. model_first and ball_physics remain selectable as experiments.
    planner: str = Field("staged", pattern="^(model_first|staged|ball_physics|event_t2m)$")
    bounce_parameters: bp.BounceOverrides | None = None
    estimate_parameters: bool = False
    # "local" = LM Studio on this PC; "gpt-4o" sends the event text to OpenAI.
    llm: str = Field("local", pattern="^(local|gpt-4o)$")


def _lm(llm):
    """(model, base_url) for a backend name; local uses whichever LM Studio model is loaded."""
    if llm != "local":
        return dc.backend(llm)
    models = _llm_models()
    if not models:
        raise HTTPException(503, "LM Studio 서버/모델을 찾을 수 없습니다")
    return _pick_llm(models), dc.LMSTUDIO_URL


def _plan(event, planner="staged", llm="local", estimate_parameters=False, bounce_parameters=None):
    model, url = _lm(llm)
    if planner == "model_first":
        r = mp.decompose(event, model, url, estimate_parameters=estimate_parameters)
    elif planner == "event_t2m":
        r = et.decompose(event, model, url)
    elif planner == "ball_physics":
        r = bp.decompose(event, model, url, overrides=bounce_parameters)
    else:
        r = dc.decompose(event, model, url)
    return {**r, "llm": llm}


@app.post("/api/decompose")
def decompose(req: DecomposeReq):
    try:
        return _plan(req.event, req.planner, req.llm, req.estimate_parameters, req.bounce_parameters)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"분해 실패: {e}")


class SubEvent(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    start: float = Field(ge=0, le=10)
    duration: float = Field(ge=0.15, le=10)
    intensity: float = Field(ge=0, le=1)
    source: str = ""
    envelope: dict | None = None
    synthesis: dict | None = None
    # raw sample from an earlier run to reuse instead of generating (path under out/)
    source_wav: str | None = Field(None, max_length=300)
    uid: str | None = Field(None, max_length=40)   # editor's id, echoed back to map results

    @field_validator("source_wav")
    @classmethod
    def _inside_out(cls, v):
        if v is not None:
            try:
                _wav_path(v)   # rejects traversal / missing files
            except HTTPException:
                return None    # gone: generate anew
        return v


class ComposeReq(BaseModel):
    event: str = Field(min_length=1, max_length=400)
    sub_events: list[SubEvent] = Field(min_length=1, max_length=8)
    candidates: int = Field(2, ge=1, le=4)
    baseline: bool = True
    cfg: float = Field(3.0, ge=0.0, le=10.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(250, ge=1, le=2048)
    seed: int | None = None
    gamma: float = Field(cp.DEFAULT_GAMMA, ge=0.5, le=3.0)
    share_repeats: bool = True
    generator: str = Field("auto", pattern="^(auto|(hapticgen|audiogen)(@colab)?)$")
    physics: dict | None = None
    hierarchy: dict | None = None


@app.post("/api/compose")
def compose(req: ComposeReq):
    global _gen_busy
    if _model is None and not req.generator.endswith("@colab") and not (req.generator == "auto" and _colab["url"]):
        raise HTTPException(503, f"model {_model_status}")
    if max(s.start + s.duration for s in req.sub_events) > 10.0:
        raise HTTPException(422, "전체 길이가 10초를 넘습니다")
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "already generating")
    _gen_busy = True
    try:
        segments = [s.model_dump(exclude_none=True) for s in req.sub_events]
        plan = cp.compose(_use_generator(req.generator), req.event, segments, req.candidates,
                          req.baseline, req.cfg, req.temperature, req.top_k, req.seed, OUT_DIR,
                          gamma=req.gamma, share_repeats=req.share_repeats)
        if req.physics is not None:
            plan["physics"] = req.physics
            plan["physics_plan_edited"] = (req.physics.get("reference_sub_events") != segments
                                            or req.physics.get("original_event") != req.event)
        if req.hierarchy is not None:
            plan["hierarchy"] = req.hierarchy
            plan["hierarchy_plan_edited"] = (req.hierarchy.get("reference_sub_events") != segments
                                              or req.hierarchy.get("original_event") != req.event)
        if req.physics is not None or req.hierarchy is not None:
            run_dir = OUT_DIR / "compose" / pathlib.Path(plan["composite"]).stem.removesuffix("_composite")
            (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        plan["composite_item"] = _describe(OUT_DIR / plan["composite"])
        plan["baseline_item"] = _describe(OUT_DIR / plan["baseline"]) if plan["baseline"] else None
        return plan
    finally:
        _gen_busy = False
        _gen_lock.release()


class Compare3Req(BaseModel):
    event: str = Field(min_length=1, max_length=400)
    candidates: int = Field(2, ge=1, le=4)
    seed: int | None = None
    gamma: float = Field(cp.DEFAULT_GAMMA, ge=0.5, le=3.0)
    share_repeats: bool = True


@app.post("/api/compare3")
def compare3(req: Compare3Req):
    """Same event, same seed: staged plan from the local LM, staged plan from GPT-4o, and
    single-prompt HapticGen. Plans are made first; a failed LM is reported, not fatal."""
    global _gen_busy
    if _model is None:
        raise HTTPException(503, f"model {_model_status}")
    plans, errors = {}, {}
    for llm in ("local", "gpt-4o"):
        try:
            plans[llm] = _plan(req.event, "staged", llm)
        except HTTPException as e:
            errors[llm] = e.detail
        except Exception as e:  # noqa: BLE001
            errors[llm] = f"{type(e).__name__}: {e}"
    if not plans:
        raise HTTPException(502, f"두 LM 모두 분해 실패: {errors}")
    seed = req.seed if req.seed is not None else int(time.time()) % 100000
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "already generating")
    _gen_busy = True
    try:
        items, first = {}, True
        for llm, plan in plans.items():
            subs = [SubEvent(**{k: s[k] for k in ("prompt", "start", "duration", "intensity")}).model_dump(exclude_none=True)
                    for s in plan["sub_events"]]
            r = cp.compose(_use_generator("hapticgen"), req.event, subs, req.candidates, first, 3.0, 1.0, 250, seed, OUT_DIR,
                           gamma=req.gamma, share_repeats=req.share_repeats)
            meta_path = (OUT_DIR / r["composite"]).with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["prompt"] = f"[{'로컬 LM' if llm == 'local' else 'GPT-4o'} 분해] {req.event}"
            meta["llm"] = llm
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            items[llm] = {**_describe(OUT_DIR / r["composite"]), "regions": r["sub_events"]}
            if first and r["baseline"]:
                items["baseline"] = _describe(OUT_DIR / r["baseline"])
            first = False
        return {"event": req.event, "seed": seed, "items": items, "errors": errors,
                "plans": {k: {"sub_events": v["sub_events"], "elapsed": v["elapsed"], "model": v["model"],
                              "phases": v.get("phases", [])} for k, v in plans.items()}}
    finally:
        _gen_busy = False
        _gen_lock.release()


@app.get("/editor")
def editor_page():
    """Authoring editor: decompose -> edit sub-events on a timeline -> compose -> play."""
    return FileResponse(pathlib.Path(__file__).with_name("editor.html"))


# ---------------------------------------------------------------- Story2Hap
STORY_DIR = OUT_DIR / "story"
_story_jobs = {}   # id -> {"status", "stage", "log": [...], "error"}
STORY_ID = r"^[A-Za-z0-9_-]{1,60}$"


@app.get("/system")
def system_page():
    """Story2Hap system diagram: devices, processing pipeline, playback sync, files."""
    return FileResponse(pathlib.Path(__file__).with_name("system.html"))


@app.get("/story")
def story_page():
    """Story2Hap: narrated story with per-sentence explicit and inferred vibration."""
    return FileResponse(pathlib.Path(__file__).with_name("story.html"))


class StoryReq(BaseModel):
    text: str = Field(min_length=1, max_length=30000)
    title: str = Field("", max_length=80)
    voice: str = Field("fable", pattern="^(" + "|".join(s2h.VOICES) + ")$")
    threshold: float = Field(0.5, ge=0, le=1)
    generator: str = Field("auto", pattern="^(auto|(hapticgen|audiogen)(@colab)?)$")


def _story_log(sid, stage, msg):
    job = _story_jobs.setdefault(sid, {"status": "running", "log": []})
    job["stage"] = stage
    job["log"].append({"t": time.time(), "stage": stage, "msg": msg})


STORY_WINDOW = 9.5   # composer limit is 10 s; longer sentences are rendered window by window and joined


def _windows(subs):
    """Split timed sub-events into <= STORY_WINDOW windows (events crossing a boundary are cut there)."""
    end = max(x["start"] + x["duration"] for x in subs)
    out = []
    for k in range(int(np.ceil(end / STORY_WINDOW))):
        w0, w1 = k * STORY_WINDOW, (k + 1) * STORY_WINDOW
        part = []
        for x in subs:
            a0, a1 = max(x["start"], w0), min(x["start"] + x["duration"], w1)
            if a1 - a0 >= 0.15:
                part.append({**x, "start": round(a0 - w0, 3), "duration": round(a1 - a0, 3)})
        if part:
            out.append((w0, part))
    return out


DUCK_DEPTH = 0.65   # how far the scene background dips while a foreground event plays


def _duck(bg, fg, sr, depth=DUCK_DEPTH):
    """Lower the background under foreground events (side-chain ducking), so a strong background can stay
    strong in its scene without masking knocks and crashes. Gain follows the foreground envelope with a
    short attack and a slower release."""
    n = len(bg)
    f = np.zeros(n, dtype=np.float32)
    f[:min(n, len(fg))] = fg[:n]
    env = np.sqrt(np.convolve(f * f, np.ones(int(0.03 * sr)) / int(0.03 * sr), mode="same"))
    if env.max() < 1e-5:
        return bg
    g = 1.0 - depth * np.clip(env / (0.25 * env.max()), 0, 1)
    release = int(0.12 * sr)   # hold the dip a little after each hit, then recover smoothly
    g = np.minimum(g, np.convolve(g, np.ones(release) / release, mode="same"))
    return (bg * g).astype(np.float32)


CANDIDATES = 4          # generated candidates per sample; the most vibration-like one is kept
QC_LOW, QC_FLAT = 0.75, 0.65   # pass: >= 75% of energy below 300 Hz and spectral flatness <= 0.65


def _haptic_quality(w, sr):
    """(share of energy below 300 Hz, spectral flatness). Actuators render little above ~300 Hz, and a flat
    spectrum is noise rather than a vibration (Scene2Hap keeps the least flat candidate for the same reason)."""
    w = np.asarray(w, dtype=np.float64)
    if len(w) < 64 or np.abs(w).max() < 1e-5:
        return 0.0, 1.0
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    fr = np.fft.rfftfreq(len(w), 1 / sr)
    p = spec ** 2
    low = float(p[fr < 300].sum() / (p.sum() + 1e-12))
    flat = float(np.exp(np.mean(np.log(spec + 1e-9))) / (np.mean(spec) + 1e-9))
    return low, flat


def _band(w, sr, lo=40.0, hi=400.0):
    """Keep the vibrotactile band (zero-phase Butterworth band-pass)."""
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), w).astype(np.float32)


GEN_BUCKETS = (2.0, 4.0, 6.0, 10.0)   # generation lengths: similar lengths share a batch


def _bucket(sec):
    return next(b for b in GEN_BUCKETS if sec <= b + 1e-6)


def _story_samples(sid, plan, model, ids, fresh):
    """Generate every sample the sentences need in a few large batches, before any mixing:
    one raw sample per voice_key (best of 2 candidates) and one single-prompt baseline per window.
    Returns (voice_key -> sample path for these sentences, (sentence, window) -> baseline wave)."""
    out_dir, sr = STORY_DIR / sid, model.sample_rate
    voices = plan.setdefault("voices", {})
    need, base_items = {}, []
    for s in plan["sentences"]:
        if s["id"] not in ids or not s["sub_events"]:
            continue
        for k, (w0, part) in enumerate(_windows(s["sub_events"])):
            for x in part:
                if x.get("family", "texture") != "texture":   # rhythmic families are synthesised, not generated
                    continue
                if fresh or x["voice_key"] not in voices:
                    cap, longest, rep = need.get(x["voice_key"], (x["prompt"], 0.0, x))
                    need[x["voice_key"]] = (cap, max(longest, x["duration"]), x if x["duration"] >= longest else rep)
            if not fresh:   # a re-render keeps the sentence's existing single-prompt baseline
                base_items.append((s["id"], k, s["text"], max(p["start"] + p["duration"] for p in part)))
    jobs = [("voice", vk, cap, _bucket(min(10.0, max(1.5, dur + 1.0))), rep) for vk, (cap, dur, rep) in need.items()]
    jobs += [("base", (sid_, k), text, _bucket(min(10.0, dur)), None) for sid_, k, text, dur in base_items]
    chosen, baselines = {}, {}
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for bucket in GEN_BUCKETS:
        group = [j for j in jobs if j[3] == bucket]
        if not group:
            continue
        prompts = []
        for kind, key, cap, _, _ in group:
            prompts += [cap] * (CANDIDATES if kind == "voice" else 1)
        _story_log(sid, "render", f"진동 샘플 {sum(j[0] == 'voice' for j in group)}개 + 베이스라인 {sum(j[0] == 'base' for j in group)}개 일괄 생성 ({bucket:.0f}초 묶음)")
        wavs, k = cp._generate_batched(model, prompts, bucket, 3.0, 1.0, 250), 0
        for kind, key, cap, _, rep in group:
            if kind == "voice":
                cands, k = wavs[k:k + CANDIDATES], k + CANDIDATES
                q = [_haptic_quality(w, sr) for w in cands]
                ok = [i for i, (low, flat) in enumerate(q) if low >= QC_LOW and flat <= QC_FLAT]
                # among vibration-like candidates the best-fitting one; if none passes, the least noise-like
                pick = max(ok, key=lambda i: cp._extract(cands[i], sr, rep)[1]) if ok else \
                    max(range(len(cands)), key=lambda i: q[i][0] * (1 - q[i][1]))
                best = _band(cands[pick], sr)
                plan.setdefault("sample_qc", {})[key] = {"caption": cap, "passed": len(ok), "of": len(cands),
                                                         "low": round(q[pick][0], 3), "flatness": round(q[pick][1], 3)}
                name = re.sub(r"[^A-Za-z0-9_-]+", "_", key)[:40] + ("_fresh" if fresh else "") + ".wav"
                sf.write(out_dir / "samples" / name, best, sr, subtype="FLOAT")
                chosen[key] = f"samples/{name}"
                if not fresh:
                    voices[key] = chosen[key]
            else:   # the single-prompt baseline goes through the same output band as everything else
                baselines[key], k = _band(wavs[k], sr), k + 1
    return {**voices, **chosen}, baselines


def _story_render(sid, plan, ids, fresh=False, on_sentence=None):
    """Agent 6: per sentence, the explicit + inferred composite, an explicit-only remix of the same samples,
    and a single-prompt baseline. All samples are generated first in large batches (_story_samples); the
    per-sentence mixing then needs no generation. voice_key -> one raw sample keeps a source's vibration
    identical across sentences and across the windows of a long sentence. on_sentence(s) runs after each
    sentence (save, critic). Waits for the GPU (other generation requests get 409 meanwhile)."""
    global _gen_busy
    out_dir = STORY_DIR / sid
    with _gen_lock:
        _gen_busy = True
        try:
            model = _use_generator(plan.get("generator", "auto"))
            try:
                samples, baselines = _story_samples(sid, plan, model, ids, fresh)
            except Exception as exc:  # noqa: BLE001  Colab tunnel gone mid-run: finish on the local GPU
                if not (plan.get("generator", "auto") == "auto" and getattr(model, "name", "").endswith("@colab")):
                    raise
                _story_log(sid, "render", f"Colab 생성 실패 ({type(exc).__name__}), 로컬 GPU로 계속합니다")
                model = _use_generator("hapticgen")
                samples, baselines = _story_samples(sid, plan, model, ids, fresh)
            plan["generator_used"] = getattr(model, "name", "hapticgen")
            sr = model.sample_rate
            for s in plan["sentences"]:
                if s["id"] not in ids:
                    continue
                subs = s["sub_events"]
                if not subs:
                    s["render"] = None
                    if on_sentence:
                        on_sentence(s)
                    continue
                wins = _windows(subs)
                total = max(x["start"] + x["duration"] for x in subs) + 0.1
                mix = {k: np.zeros(int(total * sr) + 1, dtype=np.float32) for k in ("full", "explicit", "baseline")}
                used, rendered, t0 = {"full": False, "explicit": False, "baseline": False}, [], time.time()

                def add(kind, w, w0):
                    a = int(w0 * sr)
                    n = min(len(w), len(mix[kind]) - a)
                    mix[kind][a:a + n] += w[:n]
                    used[kind] = True

                def layer(subs_):
                    """Mix of the given sub-events (samples already generated) -> (wave, rendered sub-events)."""
                    if not subs_:
                        return None, []
                    r = cp.compose(model, s["text"], subs_, 1, False, 3.0, 1.0, 250, None, out_dir,
                                   gamma=cp.DEFAULT_GAMMA, share_repeats=True)
                    return sf.read(out_dir / r["composite"], dtype="float32")[0], r["sub_events"]

                def mixed(subs_):
                    """Foreground events over the scene background, which ducks under them."""
                    fg, fg_ev = layer([x for x in subs_ if x.get("layer") != "bed"])
                    bg, bg_ev = layer([x for x in subs_ if x.get("layer") == "bed"])
                    if bg is not None and fg is not None:
                        bg = _duck(bg, fg, sr)
                    parts_ = [w for w in (fg, bg) if w is not None]
                    if not parts_:
                        return None, []
                    out = np.zeros(max(len(w) for w in parts_), dtype=np.float32)
                    for w in parts_:
                        out[:len(w)] += w
                    return out, fg_ev + bg_ev

                for k, (w0, part) in enumerate(wins):
                    def source(x, k=k):
                        if x.get("family", "texture") == "texture":
                            return samples.get(x["voice_key"])
                        p_ = x.get("params") or {}   # rhythmic pattern for exactly this event and length
                        w = hp.synth(x["family"], x["duration"] + 0.05, p_.get("rate") or None, p_.get("weight", .5),
                                     p_.get("hardness", .5), sr=sr, seed=zlib.crc32((x.get("seed_key") or x["eid"]).encode()))
                        name = f"synth_{re.sub(r'[^A-Za-z0-9_-]+', '_', x['eid'])}_{k}{'_fresh' if fresh else ''}.wav"
                        sf.write(out_dir / "samples" / name, w, sr, subtype="FLOAT")
                        return f"samples/{name}"

                    full, evs = mixed([{**x, "source_wav": source(x)} for x in part])
                    rendered += [{**o, "start": round(o["start"] + w0, 3)} for o in evs]
                    add("full", full, w0)
                    b = baselines.get((s["id"], k))
                    if b is not None:   # level the baseline to the composite's peak, as compose() does
                        n = int((max(p["start"] + p["duration"] for p in part)) * sr)
                        b = b[:n] * (float(np.abs(full).max()) / max(1e-4, float(np.abs(b[:n]).max())))
                        add("baseline", cp._fade(b, sr, cp.FADE_IN, cp.FADE_OUT), w0)
                    exp, _ = mixed([{**o} for o in evs if o["origin"] == "explicit"])
                    if exp is not None:
                        add("explicit", exp, w0)
                files = {}
                old = s.get("render") or {}
                for kind, w in mix.items():
                    if kind == "baseline" and fresh and (old.get("files") or {}).get("baseline"):
                        files[kind] = old["files"]["baseline"]
                        continue
                    if used[kind]:
                        name = f"s{s['id']:03d}_{kind}{'_fresh' if fresh else ''}.wav"
                        sf.write(out_dir / name, np.clip(w, -1, 1), sr, subtype="FLOAT")
                        files[kind] = f"story/{sid}/{name}"
                    else:
                        files[kind] = None
                s["render"] = {"files": files, "elapsed": round(time.time() - t0, 2), "windows": len(wins),
                               "waves": {k: {key: v for key, v in _describe(OUT_DIR / f).items() if key in ("name", "duration", "peak", "peaks")}
                                         for k, f in files.items() if f},
                               "sub_events": [{k: o.get(k) for k in ("eid", "origin", "prompt", "start", "duration", "intensity",
                                                                     "voice_key", "reused", "source_wav", "layer", "family", "salience")}
                                              for o in rendered]}
                _story_log(sid, "render", f"문장 {s['id'] + 1} 완성")
                if on_sentence:
                    on_sentence(s)
        finally:
            _gen_busy = False


def _critic_one(plan, s):
    """Agent 7 for one sentence: signal check + LM judgement."""
    if not s.get("render"):
        return {"score": None, "issues": [], "action": "ok", "note": "진동 없음 (연출 결정 또는 사건 없음)"}
    signal = s2h.signal_check(OUT_DIR / s["render"]["files"]["full"], s["sub_events"])
    return s2h.critique(s, plan, signal, plan.get("model", s2h.AGENT_MODEL))


def _render_and_critic(sid, plan, ids, fresh=False):
    """Render; each finished sentence is saved (playable at once) and judged by the critic in parallel."""
    out_dir, pool, futures = STORY_DIR / sid, s2h.ThreadPoolExecutor(8), []
    lock = threading.Lock()

    def save():
        with lock:
            for _ in range(5):   # the render thread may add a key while we serialise; just try again
                try:
                    return s2h.save(plan, out_dir)
                except RuntimeError:
                    time.sleep(0.05)

    def judged(s):
        c = _critic_one(plan, s)
        with lock:
            s["critic"] = c
        save()

    def done(s):
        save()
        futures.append(pool.submit(judged, s))

    _story_log(sid, "render", "렌더링 시작 (완성된 문장부터 바로 재생할 수 있습니다)")
    _story_render(sid, plan, ids, fresh, on_sentence=done)
    _story_log(sid, "critic", "Critic 평가 마무리")
    for f in futures:
        f.result()
    pool.shutdown()
    save()


def _story_job(sid, fn):
    def run():
        try:
            fn()
            _story_jobs[sid]["status"] = "done"
            _story_log(sid, "done", "완료")
        except Exception as e:  # noqa: BLE001
            _story_jobs[sid].update(status="error", error=f"{type(e).__name__}: {e}")
            _story_log(sid, "error", str(e)[:300])
    _story_jobs[sid] = {"status": "running", "stage": "start", "log": []}
    threading.Thread(target=run, daemon=True).start()


@app.post("/api/story/run")
def story_run(req: StoryReq):
    sid = time.strftime("%m%d_%H%M%S") + "_" + (re.sub(r"[^A-Za-z0-9]+", "_", req.title or req.text[:30]).strip("_")[:30] or "story")
    out_dir = STORY_DIR / sid

    def work():
        plan = s2h.plan_story(req.text, out_dir, req.voice, req.threshold, progress=lambda st, m: _story_log(sid, st, m))
        plan.update(title=req.title or s2h.split_sentences(req.text)[0][:60], generator=req.generator)
        s2h.save(plan, out_dir)
        ids = [s["id"] for s in plan["sentences"]]
        _render_and_critic(sid, plan, ids)
        # One automatic retry only where the signal missed planned impacts: new samples can fix that, while
        # the critic's remarks on intensity or timing stay as advice (a re-render would not change them).
        def missed(s):
            sig = (s.get("critic") or {}).get("signal") or {}
            return (sig.get("expected_impacts") or 0) > (sig.get("matched_impacts") or 0)
        retry = [s["id"] for s in plan["sentences"] if missed(s)]
        if retry:
            _story_log(sid, "render", f"Critic 요청으로 문장 {', '.join(str(i + 1) for i in retry)} 다시 렌더링")
            _render_and_critic(sid, plan, retry, fresh=True)
            for s in plan["sentences"]:
                if s["id"] in retry:
                    s["critic"]["retried"] = True
        s2h.save(plan, out_dir)

    _story_job(sid, work)
    return {"id": sid}


def _story_plan(sid):
    if not re.match(STORY_ID, sid):
        raise HTTPException(404, "no story")
    path = STORY_DIR / sid / "plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_fields(model):
    """Flatten a pydantic output model into (path, type, constraints) rows for display."""
    rows = []

    def walk(sch, prefix, defs):
        for name, f in sch.get("properties", {}).items():
            f = defs.get(f["$ref"].split("/")[-1], f) if "$ref" in f else f
            t = f.get("type") or ("enum" if "enum" in f else "object")
            if t == "array" and "$ref" in f.get("items", {}):
                sub = defs[f["items"]["$ref"].split("/")[-1]]
                rows.append((prefix + name + "[]", "list", ""))
                walk(sub, prefix + name + "[].", defs)
                continue
            if "properties" in f:
                rows.append((prefix + name, "object", ""))
                walk(f, prefix + name + ".", defs)
                continue
            c = []
            for k, label in (("enum", "값"), ("minimum", "최소"), ("maximum", "최대"), ("maxLength", "최대 글자"), ("maxItems", "최대 개수")):
                if k in f:
                    c.append(f"{label} {f[k]}")
            if t == "array":
                t = f"list of {f.get('items', {}).get('type', '?')}"
            rows.append((prefix + name, t, ", ".join(c)))

    full = model.model_json_schema()
    walk(full, "", full.get("$defs", {}))
    return [{"field": a, "type": b, "rule": c} for a, b, c in rows]


@app.get("/api/story/prompts")
def story_prompts():
    """The Story2Hap agents exactly as the code runs them: prompt text, inputs and output schema."""
    return {"model": s2h.AGENT_MODEL, "tts": {"model": s2h.TTS_MODEL, "instructions": s2h.TTS_INSTRUCTIONS},
            "stt": s2h.STT_MODEL,
            "agents": [{"key": a["key"], "calls": a["calls"], "inputs": a["inputs"], "prompt": getattr(s2h, a["prompt"]),
                        "prompt_name": a["prompt"], "output_name": a["output"].__name__, "output": _schema_fields(a["output"])}
                       for a in s2h.AGENTS]}


@app.get("/api/story")
def story_list():
    out = []
    for path in sorted(STORY_DIR.glob("*/plan.json"), reverse=True):
        try:
            p = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        out.append({"id": path.parent.name, "title": p.get("title") or p["sentences"][0]["text"][:60],
                    "sentences": len(p["sentences"]), "created": p.get("created")})
    for sid, job in _story_jobs.items():
        if job["status"] == "running" and not any(x["id"] == sid for x in out):
            out.insert(0, {"id": sid, "title": "(진행 중)", "sentences": None, "created": None})
    return out


@app.get("/api/story/{sid}")
def story_get(sid: str):
    plan = _story_plan(sid)
    if plan:
        plan.pop("traces", None)
    return {"plan": plan, "job": _story_jobs.get(sid)}


@app.post("/api/story/{sid}/sentence/{i}/rerender")
def story_rerender(sid: str, i: int):
    plan = _story_plan(sid)
    if not plan or not 0 <= i < len(plan["sentences"]):
        raise HTTPException(404, "no such sentence")
    if _story_jobs.get(sid, {}).get("status") == "running":
        raise HTTPException(409, "이 이야기는 이미 처리 중입니다")

    def work():
        _render_and_critic(sid, plan, [i], fresh=True)

    _story_job(sid, work)
    return {"ok": True}


# ---------------------------------------------------------------- authoring-study logs
STUDY_DIR = OUT_DIR / "study"
_study_lock = threading.Lock()
ID = r"^[A-Za-z0-9_-]{1,40}$"


class StudyLog(BaseModel):
    participant: str = Field(pattern=ID)
    task: str = Field(pattern=ID)
    session: str = Field(pattern=ID)
    events: list[dict] = Field(max_length=500)


@app.post("/api/study/log")
def study_log(req: StudyLog):
    """Append editor interaction events (one JSON object per line) to
    out/study/<participant>/<task>_<session>.jsonl. Only written in study mode (?p=...&task=...)."""
    path = STUDY_DIR / req.participant / f"{req.task}_{req.session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _study_lock, path.open("a", encoding="utf-8") as f:
        for e in req.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return {"ok": True, "written": len(req.events)}


@app.get("/gallery")
def gallery_page():
    """Listening page: every eval sentence under each condition."""
    return FileResponse(pathlib.Path(__file__).with_name("gallery.html"))


_wave_cache = {}


def _gallery_manifest(round_):
    return OUT_DIR / "gallery" / ("manifest.json" if round_ == 1 else f"manifest_round{round_}.json")


def _gallery_rounds():
    """[{round, label}] for every built round (round = condition design + seed)."""
    out = []
    for p in sorted((OUT_DIR / "gallery").glob("manifest*.json")):
        r = 1 if p.stem == "manifest" else int(p.stem.removeprefix("manifest_round"))
        m = json.loads(p.read_text(encoding="utf-8"))
        out.append({"round": r, "label": m.get("label", f"seed {m.get('seed')}")})
    return sorted(out, key=lambda x: x["round"])


@app.get("/api/gallery")
def gallery_data(round: int = 1):  # noqa: A002  (query parameter name)
    """Manifest of one listening round (= HapticGen seed) plus waveform summaries of rendered files."""
    path = _gallery_manifest(round)
    if not path.exists():
        return {"items": [], "conditions": [], "building": True, "round": round, "rounds": _gallery_rounds()}
    m = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for entry in m["items"].values():
        renders = {}
        for key, r in entry["renders"].items():
            if "file" in r and (OUT_DIR / r["file"]).exists():
                if r["file"] not in _wave_cache:
                    _wave_cache[r["file"]] = _describe(OUT_DIR / r["file"])
                renders[key] = {**_wave_cache[r["file"]], "regions": r.get("regions")}
            else:
                renders[key] = {"error": r.get("error", "missing file")}
        items.append({**entry, "renders": renders})
    return {"items": items, "conditions": m["conditions"], "seed": m.get("seed"), "updated": m.get("updated"),
            "round": round, "rounds": _gallery_rounds()}


VOTES = OUT_DIR / "gallery" / "votes.json"
_vote_lock = threading.Lock()


class VoteReq(BaseModel):
    item: str = Field(min_length=1, max_length=20)
    choice: str = Field(pattern="^[a-z0-9_]{1,40}$")   # a condition key of this round, or "tie"
    order: list[str] = Field(min_length=1, max_length=8)   # condition keys in displayed order (position-bias analysis)
    blind: bool = True
    rater: str = Field("me", min_length=1, max_length=40)
    round: int = Field(1, ge=1, le=99)   # listening round = HapticGen seed of the renders


def _votes():
    return json.loads(VOTES.read_text(encoding="utf-8")) if VOTES.exists() else []


def _vote_summary(votes, rater=None, round_=1):
    """Latest vote per (rater, item) within one round; blind votes only in the tally.
    Votes recorded before rounds existed belong to round 1."""
    latest = {}
    for v in votes:
        if v.get("round", 1) == round_ and (rater is None or v["rater"] == rater):
            latest[(v["rater"], v["item"])] = v
    blind = [v for v in latest.values() if v["blind"]]
    tally = {}
    for v in blind:
        tally[v["choice"]] = tally.get(v["choice"], 0) + 1
    position = {}
    for v in blind:
        if v["choice"] in v["order"]:
            pos = v["order"].index(v["choice"])
            position[pos] = position.get(pos, 0) + 1
    return {"n_blind": len(blind), "tally": tally, "chosen_position": position,
            "mine": {v["item"]: v["choice"] for v in latest.values()} if rater else None,
            "raters": sorted({r for r, _ in latest})}


@app.post("/api/gallery/vote")
def gallery_vote(req: VoteReq):
    """Record one listening vote; the latest vote per rater and item counts."""
    path = _gallery_manifest(req.round)
    if not path.exists():
        raise HTTPException(404, f"round {req.round} not found")
    keys = {c["key"] for c in json.loads(path.read_text(encoding="utf-8"))["conditions"]}
    if req.choice != "tie" and req.choice not in keys:
        raise HTTPException(422, f"choice {req.choice!r} is not a condition of round {req.round}")
    with _vote_lock:
        votes = _votes()
        votes.append({**req.model_dump(), "time": time.time()})
        VOTES.parent.mkdir(parents=True, exist_ok=True)
        VOTES.write_text(json.dumps(votes, ensure_ascii=False, indent=1), encoding="utf-8")
    return _vote_summary(votes, req.rater, req.round)


@app.get("/api/gallery/votes")
def gallery_votes(rater: str | None = None, round: int = 1):  # noqa: A002
    """Vote tally of one round (blind votes, latest per rater and item); with rater, also that rater's choices."""
    return _vote_summary(_votes(), rater, round)


@app.get("/architecture")
def architecture_page():
    """Framework structure page (diagrams + live data)."""
    return FileResponse(pathlib.Path(__file__).with_name("architecture.html"))


@app.get("/api/architecture")
def architecture_data():
    """Framework description read from the loaded modules and files."""
    return architecture.describe(app, {**status(), "llm": llm_status()})


class PlayerConfig(BaseModel):
    vib: str | int | None = None      # "quest" or a sound-device id of the remote player
    voice: int | None = None          # sound-device id for narration (None = the player's default output)


@app.get("/api/player")
def player_info():
    """The remote playback agent: its sound outputs and where vibration / voice currently go."""
    if not remote.connected:
        return {"connected": False}
    st = remote.state
    return {"connected": True, "name": remote.name, "status": st.get("status"), "ready": st.get("ready"), "error": st.get("error"),
            "devices": st.get("devices", []), "config": st.get("config", {})}


@app.post("/api/player/config")
def player_config(req: PlayerConfig):
    msg = {"cmd": "config", "voice": req.voice}
    if req.vib is not None:
        msg["vib"] = req.vib
    remote.send(msg)
    return {"ok": True}


class PairReq(BaseModel):
    voice: str
    vib: str | None = None
    tag: str | None = Field(None, max_length=120)
    gain: float = Field(1.0, ge=0.0, le=4.0)
    hand: str = Field("both", pattern="^(both|left|right)$")
    voice_delay_ms: int = Field(0, ge=0, le=1000)


@app.post("/api/player/play_pair")
def player_play_pair(req: PairReq):
    """Narration and vibration started together by the remote agent (each on its chosen output)."""
    for f in (req.voice, req.vib):
        if f:
            _wav_path(f)
    path = lambda f: "/audio/" + "/".join(urllib.parse.quote(x) for x in f.split("/"))
    remote.send({"cmd": "play_pair", "voice": path(req.voice), "vib": path(req.vib) if req.vib else None,
                 "gain": req.gain, "hand": req.hand, "voice_delay_ms": req.voice_delay_ms, "tag": req.tag})
    return {"ok": True}


class PrefetchReq(BaseModel):
    files: list[str] = Field(max_length=6)


@app.post("/api/player/prefetch")
def player_prefetch(req: PrefetchReq):
    """Let the remote agent download upcoming wavs while the current sentence plays."""
    if remote.connected:
        paths = []
        for f in req.files:
            if f:
                _wav_path(f)
                paths.append("/audio/" + "/".join(urllib.parse.quote(x) for x in f.split("/")))
        remote.send({"cmd": "prefetch", "paths": paths})
    return {"ok": True}


@app.get("/api/player/progress")
def player_progress(tag: str):
    """Seconds since the agent actually started (or finished) the tagged playback, on the server clock."""
    ev, now = remote.events.get(tag, {}), time.time()
    return {"started": now - ev["started"] if "started" in ev else None,
            "ended": now - ev["ended"] if "ended" in ev else None}


@app.post("/api/stop")
def stop():
    if remote.connected:
        remote.send({"cmd": "stop"})
    player.stop()
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # --host 0.0.0.0 to serve Tailscale peers (other clients are refused by _ClientAllowlist)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
