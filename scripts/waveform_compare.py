"""Waveform-level comparison (same seed): sub-event composition from local-LM plans, from GPT-4o
plans, and single-prompt HapticGen.

Uses the default planner's saved plans (out/ablation/plans_staged.json, plans_staged@gpt-4o.json) and the running web
server's /api/compose, which renders the composite and a single-prompt baseline in one batch.
Only prompts whose stated condition is measurable from the waveform alone are used:
a total count of short events (the prompt describes nothing else that vibrates) or a period.

  python scripts/waveform_compare.py [--seeds 1,2,3]
"""
import argparse
import json
import pathlib
import statistics
import sys
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_waveform import detect  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
API = "http://127.0.0.1:7860"
# id -> ("count", n) | ("period", seconds). Conditions taken from eval_set_v2.1 explicit counts/periods.
TARGETS = {"c01": ("count", 1), "c02": ("count", 1), "c04": ("count", 1), "c11": ("count", 2),
           "s14": ("count", 1), "c12": ("period", 0.6)}


def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def measure(path):
    w, sr = sf.read(OUT / path, dtype="float64")
    w = w.mean(axis=1) if w.ndim > 1 else w
    return [float(t) for t in detect(w, sr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    seeds = [int(s) for s in ap.parse_args().seeds.split(",")]
    items = {i["id"]: i for i in json.loads((ROOT / "data" / "eval_set_v2.json").read_text(encoding="utf-8"))["items"]}
    sources = {"local": "plans_staged.json", "gpt-4o": "plans_staged@gpt-4o.json"}
    plans = {k: json.loads((OUT / "ablation" / f).read_text(encoding="utf-8"))
             for k, f in sources.items() if (OUT / "ablation" / f).exists()}
    rows = []
    for iid, (kind, target) in TARGETS.items():
        for seed in seeds:
            files, first = {}, True
            for llm, pl in plans.items():
                plan = pl.get(iid)
                if not plan or not plan["ok"]:
                    print(iid, llm, "no plan"); continue
                r = post("/api/compose", {"event": items[iid]["prompt"], "sub_events": plan["sub_events"],
                                          "candidates": 2, "baseline": first, "seed": seed})
                files[f"composite_{llm}"] = r["composite"]
                if first:
                    files["baseline"] = r["baseline"]
                first = False
            for method, path in files.items():
                on = measure(path)
                gaps = [b - a for a, b in zip(on, on[1:])]
                row = dict(id=iid, seed=seed, method=method, kind=kind, target=target, onsets=on, n=len(on))
                if kind == "count":
                    row.update(ok=len(on) == target, abs_err=abs(len(on) - target))
                else:
                    row.update(ok=len(gaps) >= 2 and all(abs(g - target) <= .2 * target for g in gaps),
                               median_gap=statistics.median(gaps) if gaps else None)
                rows.append(row)
                print(f"{iid} seed {seed} {method:18s} onsets {len(on):3d} {'OK' if row['ok'] else '--'}", flush=True)
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation" / "waveform_compare.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nper method:")
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m]
        cnt = [r for r in rs if r["kind"] == "count"]
        print(f"  {m:18s} condition met {sum(r['ok'] for r in rs)}/{len(rs)} | count exact {sum(r['ok'] for r in cnt)}/{len(cnt)}"
              f" | mean |count error| {np.mean([r['abs_err'] for r in cnt]):.1f}")
    print("per item (met / seeds):")
    for iid in TARGETS:
        parts = []
        for m in sorted({r["method"] for r in rows}):
            rs = [r for r in rows if r["id"] == iid and r["method"] == m]
            parts.append(f"{m} {sum(r['ok'] for r in rs)}/{len(rs)} {[r['n'] for r in rs]}")
        print(f"  {iid}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
