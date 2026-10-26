"""Motivation check on the HapticGen dataset: do generated signals match the count stated in the
prompt, and are mismatches voted down more often?

Uses the reference-blind onset detector from eval_waveform.py on every signal whose prompt states
a discrete count of short events (ids and expected counts from data/eval_set_v1.json).
"""
import collections
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_waveform import detect  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hapticgen-dataset"
# Items whose stated count is a count of discrete short events (exact expectation).
EXACT = {"c01": 1, "c02": 1, "c04": 1, "c09": 3, "c10": 1, "c11": 2}


def norm(p):
    return re.sub(r"\s+", " ", p.strip().strip('"')).strip().lower()


def main():
    items = {it["id"]: it for it in json.loads((ROOT / "data" / "eval_set_v1.json").read_text(encoding="utf-8"))["items"]}
    want = {norm(items[i]["prompt"]): (i, n) for i, n in EXACT.items()}
    rows = []
    for f in glob.glob(str(DATA / "*" / "*.json")):
        meta = json.load(open(f, encoding="utf-8"))
        key = norm(meta["user_prompt"])
        if key not in want:
            continue
        wav = os.path.join(os.path.dirname(f), meta["filename"])
        if not os.path.exists(wav):
            continue
        w, sr = sf.read(wav, dtype="float64")
        if w.ndim > 1:
            w = w.mean(axis=1)
        if np.abs(w).max() < 1e-4:
            continue
        iid, n = want[key]
        k = len(detect(w, sr))
        rows.append(dict(id=iid, model=meta["model"] or "unknown", vote=meta["vote"], expected=n, detected=k,
                         match=k == n, set=os.path.basename(os.path.dirname(f))))
    print(f"signals: {len(rows)} across {len({r['id'] for r in rows})} prompts")
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        m = [r for r in rs if r["match"]]
        u = [r for r in rs if not r["match"]]
        up = lambda xs: (sum(r["vote"] == 1 for r in xs) / len(xs)) if xs else float("nan")
        err = [abs(r["detected"] - r["expected"]) for r in rs]
        print(f"  {model:18s} n={len(rs):3d} count match {len(m) / len(rs):5.0%} | mean |error| {np.mean(err):.1f} | "
              f"thumbs-up rate: match {up(m):5.0%} (n={len(m)}) vs mismatch {up(u):5.0%} (n={len(u)})")
    per = collections.defaultdict(list)
    for r in rows:
        per[r["id"]].append(r["detected"])
    for iid, ks in sorted(per.items()):
        print(f"  {iid} expected {EXACT[iid]} | detected counts {sorted(ks)}  ({items[iid]['prompt'][:60]})")
    out = ROOT / "out" / "evaluation" / "dataset_count_votes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
