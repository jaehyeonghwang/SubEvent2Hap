"""Render every eval item under each condition of a listening round for the /gallery page.

A round = a set of conditions (design) rendered with one HapticGen/AudioGen seed:
  --design llm  local staged plan, GPT-4o staged plan, GPT-4o one-call plan, single-prompt HapticGen
  --design 2x2  generator {HapticGen, AudioGen+conversion} x {single prompt, GPT-4o staged plan}
Plans come from out/ablation/plans_<variant>.json; rendering uses the running web server's
/api/compose (which swaps generators on the GPU, so items are grouped per generator). The manifest
is rewritten after every render; re-running skips renders whose wav already exists.

  python scripts/build_gallery.py --round 3 --seed 1 --design 2x2
"""
import argparse
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
GALLERY = OUT / "gallery"
API = "http://127.0.0.1:7860"

# design -> list of (generator, plan variant, plan key, label, single key, single label); one compose per row
DESIGNS = {
    "llm": {"title": "LM 비교", "rows": [
        ("hapticgen", "staged", "local", "로컬 Qwen3-4B · 4단계 분해", "baseline", "HapticGen 단일 생성 (문장 그대로)"),
        ("hapticgen", "staged@gpt-4o", "gpt4o", "GPT-4o · 4단계 분해", None, None),
        ("hapticgen", "oneshot@gpt-4o", "gpt4o_oneshot", "GPT-4o · 한 번 묻기", None, None)]},
    "2x2": {"title": "생성 모델 2×2", "rows": [
        ("hapticgen", "staged@gpt-4o", "hg_plan", "HapticGen + GPT-4o 분해", "hg_single", "HapticGen 단일 생성"),
        ("audiogen", "staged@gpt-4o", "ag_plan", "AudioGen + GPT-4o 분해", "ag_single", "AudioGen 단일 생성")]},
    # Same 2x2 with the local Qwen3-4B staged plans (the main, fully local pipeline).
    "2x2_local": {"title": "생성 모델 2×2 (로컬 4B 분해)", "rows": [
        ("hapticgen", "staged", "hg_plan", "HapticGen + 로컬 4B 분해", "hg_single", "HapticGen 단일 생성"),
        ("audiogen", "staged", "ag_plan", "AudioGen + 로컬 4B 분해", "ag_single", "AudioGen 단일 생성")]},
    # Rows with plan key None keep only the single-prompt render (trimmed to the plan's length).
    "ag_vs_hg": {"title": "AudioGen+분해 vs HapticGen", "rows": [
        ("hapticgen", "staged@gpt-4o", None, None, "hg_single", "HapticGen 단일 생성"),
        ("audiogen", "staged@gpt-4o", "ag_plan", "AudioGen + GPT-4o 분해", None, None)]},
}


def manifest_path(round_):
    return GALLERY / ("manifest.json" if round_ == 1 else f"manifest_round{round_}.json")


def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def done(entry, key):
    """A render counts as finished once recorded and its wav exists (wavs are not in git, so a
    fresh clone re-renders from the committed plans/manifest)."""
    r = entry["renders"].get(key) if key else None
    return not key or (r is not None and ("error" in r or (OUT / r["file"]).exists()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--design", choices=sorted(DESIGNS), default="llm")
    ap.add_argument("--generators", default="hapticgen,audiogen", help="render only these generators")
    a = ap.parse_args()
    design = DESIGNS[a.design]
    path = manifest_path(a.round)
    items = json.loads((ROOT / "data" / "eval_set_v2.json").read_text(encoding="utf-8"))["items"]
    ko = load(ROOT / "data" / "eval_set_ko.json")
    conditions = []
    for _, _, key, label, skey, slabel in design["rows"]:
        if key:
            conditions.append({"key": key, "label": label})
        if skey:
            conditions.append({"key": skey, "label": slabel})
    manifest = load(path) or {"round": a.round, "seed": a.seed, "design": a.design,
                              "label": f"{design['title']} · seed {a.seed}", "conditions": conditions, "items": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(a.generators.split(","))
    for generator, variant, key, _, skey, _ in design["rows"]:   # grouped by generator (GPU swaps are slow)
        if generator not in wanted:
            continue
        plans = load(OUT / "ablation" / f"plans_{variant}.json")
        for it in items:
            entry = manifest["items"].setdefault(it["id"], {"id": it["id"], "prompt": it["prompt"], "ko": ko.get(it["id"], ""),
                                                           "renders": {}, "plans": {}})
            if done(entry, key) and done(entry, skey):
                continue
            plan = plans.get(it["id"])
            if not plan:
                continue
            if not plan["ok"] or not plan["sub_events"]:
                for k in (key, skey):
                    if k:
                        entry["renders"][k] = {"error": plan.get("error", "no sub-events")[:300]}
                continue
            t0 = time.time()
            try:
                r = post("/api/compose", {"event": it["prompt"], "sub_events": plan["sub_events"], "candidates": 2,
                                          "baseline": bool(skey), "seed": a.seed, "generator": generator})
            except Exception as exc:  # noqa: BLE001  server busy/down: retry on the next run
                print(f"{it['id']} {key} compose failed: {exc}", flush=True)
                continue
            if key:
                entry["plans"][key] = plan["sub_events"]
                entry["renders"][key] = {"file": r["composite"], "regions": r["sub_events"], "generator": generator}
            if skey and r.get("baseline"):
                entry["renders"][skey] = {"file": r["baseline"], "generator": generator}
            manifest["updated"] = time.time()
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{it['id']} {key or skey} [{generator}] {time.time() - t0:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
