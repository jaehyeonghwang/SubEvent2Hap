"""SubEvent2Hap Studio web UI: plan, generate and test vibrations from text on Quest controllers.

Run:
    .venv\\Scripts\\python.exe webui\\server.py   ->  http://127.0.0.1:7860
"""
import io
import json
import os
import pathlib
import sys
import threading
import time
import urllib.request

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
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
from quest_haptic_play import QuestHapticPlayer  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import architecture  # noqa: E402

OUT_DIR = hg.DEFAULT_OUT
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SubEvent2Hap Studio")
player = QuestHapticPlayer().start()

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


def _use_generator(name):
    """Return the requested generator, swapping it onto the GPU if another one is loaded (~30 s)."""
    global _model, _model_name, _model_status
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
        "quest": {"status": player.status, "connected": player.connected,
                  "ready": player.status.startswith("준비됨") or player.status == "재생 중",
                  "playing": player.is_playing},
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
    planner: str = Field("staged", pattern="^(model_first|staged|ball_physics)$")
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
    generator: str = Field("hapticgen", pattern="^(hapticgen|audiogen)$")
    physics: dict | None = None
    hierarchy: dict | None = None


@app.post("/api/compose")
def compose(req: ComposeReq):
    global _gen_busy
    if _model is None:
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


@app.post("/api/stop")
def stop():
    player.stop()
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
