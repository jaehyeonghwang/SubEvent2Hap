"""Why was a decomposition condition (A/B/C) preferred: its plan, or its HapticGen render?

For every voted item:
  plan  = automatic plan checks (eval_set_v2, out/ablation/scores_v2.json) for each condition
  render fidelity = how well the rendered waveform follows its OWN plan (eval_waveform.evaluate:
          impact F1 for short events, energy outside planned regions for the rest)
Attribution:
  plan    the chosen plan passed and at least one other decomposition plan failed
  render  all decomposition plans had the same check outcome, and the chosen render followed its
          plan best (preference not explained by plan correctness)
  unclear same check outcome and the chosen render was not the most faithful
  n/a     baseline or tie chosen

  python scripts/vote_attribution.py [--rater me] [--seeds 2,3,4]   # --seeds re-renders A/B/C plans
"""
import argparse
import collections
import json
import pathlib
import statistics
import sys
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_waveform import evaluate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
VAR = {"local": "staged", "gpt4o": "staged@gpt-4o", "gpt4o_oneshot": "oneshot@gpt-4o"}


def fidelity(path, plan):
    """Render-vs-own-plan score in [0,1] plus the raw metrics."""
    w, sr = sf.read(OUT / path, dtype="float64")
    w = w.mean(axis=1) if w.ndim > 1 else w
    total = max(s["start"] + s["duration"] for s in plan)
    m = evaluate(w, sr, {"sub_events": plan, "total": total})
    parts = []
    if m["impact_f1"] is not None:
        parts.append(m["impact_f1"])
    if m["gap_energy_ratio"] is not None:
        parts.append(1 - m["gap_energy_ratio"])
    return (statistics.mean(parts) if parts else None), m


def post(body):
    req = urllib.request.Request("http://127.0.0.1:7860/api/compose", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", default="me")
    ap.add_argument("--seeds", default="", help="extra seeds to re-render A/B/C plans (render variability)")
    a = ap.parse_args()
    votes = json.loads((OUT / "gallery" / "votes.json").read_text(encoding="utf-8"))
    latest = {v["item"]: v for v in votes if v["rater"] == a.rater and v["blind"]}
    manifest = json.loads((OUT / "gallery" / "manifest.json").read_text(encoding="utf-8"))
    checks = {r["variant"]: r["per_item"] for r in json.loads((OUT / "ablation" / "scores_v2.json").read_text(encoding="utf-8"))}
    passed = lambda var, iid: all(x for k, x in checks[var][iid].items() if k != "renderable")
    items = {i["id"]: i for i in json.loads((ROOT / "data" / "eval_set_v2.json").read_text(encoding="utf-8"))["items"]}
    extra = [int(s) for s in a.seeds.split(",") if s]

    rows, verdicts = [], collections.Counter()
    for iid, v in sorted(latest.items()):
        entry = manifest["items"][iid]
        cond = {}
        for key, var in VAR.items():
            r = entry["renders"].get(key)
            if not r or "file" not in r:
                continue
            plan = entry["plans"][key]
            fid, m = fidelity(r["file"], r["regions"])
            seed_fids = []
            for seed in extra:  # same plan, different HapticGen samples
                c = post({"event": items[iid]["prompt"], "sub_events": plan, "candidates": 2, "baseline": False, "seed": seed})
                seed_fids.append(fidelity(c["composite"], c["sub_events"])[0])
            cond[key] = dict(plan_pass=passed(var, iid), n_events=len(plan), fidelity=fid,
                             impact_f1=m["impact_f1"], leak=m["gap_energy_ratio"], seed_fidelity=seed_fids)
        choice = v["choice"]
        if choice not in cond:
            verdict = "n/a"
        else:
            outcomes = {c["plan_pass"] for c in cond.values()}
            fids = {k: c["fidelity"] for k, c in cond.items() if c["fidelity"] is not None}
            if cond[choice]["plan_pass"] and len(outcomes) > 1:
                verdict = "plan"
            elif len(outcomes) == 1 and fids and fids.get(choice) == max(fids.values()):
                verdict = "render"
            elif len(outcomes) == 1:
                verdict = "unclear"
            else:
                verdict = "plan_failed_but_chosen"
        verdicts[verdict] += 1
        rows.append(dict(item=iid, choice=choice, verdict=verdict, conditions=cond))
        fmt = lambda k: (f"{k[:6]:6s} {'P' if cond[k]['plan_pass'] else 'x'} fid {cond[k]['fidelity'] if cond[k]['fidelity'] is not None else float('nan'):.2f}"
                         + (f" seeds {[round(f, 2) if f is not None else None for f in cond[k]['seed_fidelity']]}" if extra else "")) if k in cond else ""
        print(f"{iid} chose {choice:13s} -> {verdict:22s} | " + " | ".join(fmt(k) for k in VAR), flush=True)
    print("\nverdicts:", dict(verdicts))
    if extra:
        # Was the chosen render unusually good for its plan? Compare with the same plan's other seeds.
        lucky = [r for r in rows if r["choice"] in r["conditions"] and r["conditions"][r["choice"]]["seed_fidelity"]
                 and r["conditions"][r["choice"]]["fidelity"] is not None
                 and r["conditions"][r["choice"]]["fidelity"] > max(f for f in r["conditions"][r["choice"]]["seed_fidelity"] if f is not None) + .1]
        print("chosen render clearly better than the same plan at other seeds (luck candidates):", [r["item"] for r in lucky])
        spread = [np.std([c["fidelity"], *[f for f in c["seed_fidelity"] if f is not None]])
                  for r in rows for c in r["conditions"].values() if c["fidelity"] is not None]
        print(f"render fidelity std across seeds (same plan): median {np.median(spread):.3f}")
    (OUT / "evaluation" / "vote_attribution.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
