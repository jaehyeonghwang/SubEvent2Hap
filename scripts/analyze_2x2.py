"""Blind-vote analysis of the 2x2 rounds: generator {HapticGen, AudioGen} x {single prompt, decomposition}.

Each vote picks the best of four renders. Factor effects use the marginal share of picks:
  decomposition effect = share of picks that were a decomposition render (chance 1/2)
  generator effect     = share of picks that were HapticGen-based (chance 1/2)
tested with exact two-sided binomial tests (ties excluded); also per sentence group and
cross-round consistency.

  python scripts/analyze_2x2.py [--rater me] [--rounds 5,6]
"""
import argparse
import collections
import json
import pathlib
from math import comb

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABEL = {"hg_single": "HapticGen", "hg_plan": "HapticGen + 분해", "ag_single": "AudioGen", "ag_plan": "AudioGen + 분해", "tie": "비슷함"}


def binom_two_sided(k, n, p=.5):
    if n == 0:
        return None
    pk = comb(n, k) * p**k * (1 - p)**(n - k)
    return min(1.0, sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)
                        if comb(n, i) * p**i * (1 - p)**(n - i) <= pk + 1e-12))


def effects(choices, label):
    nt = [c for c in choices if c != "tie"]
    dec = sum(c.endswith("_plan") for c in nt)
    hg = sum(c.startswith("hg_") for c in nt)
    tally = collections.Counter(choices)
    print(f"== {label}: {len(choices)} votes · " + ", ".join(f"{LABEL[k]} {tally.get(k, 0)}" for k in LABEL))
    if nt:
        print(f"   decomposition picked {dec}/{len(nt)} = {dec / len(nt):.0%} (p = {binom_two_sided(dec, len(nt)):.3g})")
        print(f"   HapticGen-based picked {hg}/{len(nt)} = {hg / len(nt):.0%} (p = {binom_two_sided(hg, len(nt)):.3g})")
        a, b = tally.get("ag_single", 0), tally.get("hg_single", 0)
        print(f"   single prompts only: AudioGen {a} vs HapticGen {b} (HapticGen paper: HapticGen preferred)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", default="me")
    ap.add_argument("--rounds", default="5,6")
    a = ap.parse_args()
    rounds = [int(r) for r in a.rounds.split(",")]
    votes = json.loads((ROOT / "out" / "gallery" / "votes.json").read_text(encoding="utf-8"))
    latest = {}
    for v in votes:
        if v["rater"] == a.rater and v["blind"] and v.get("round", 1) in rounds:
            latest[(v["round"], v["item"])] = v
    for r in rounds:
        effects([v["choice"] for (rr, _), v in latest.items() if rr == r], f"round {r}")
    pooled = [v["choice"] for v in latest.values()]
    effects(pooled, "pooled")
    for g, name in (("c", "count sentences"), ("s", "sequence sentences")):
        effects([v["choice"] for (_, i), v in latest.items() if i[0] == g], f"pooled · {name}")
    pos = collections.Counter(v["order"].index(v["choice"]) for v in latest.values() if v["choice"] in v["order"])
    print("chosen position:", {chr(65 + k): n for k, n in sorted(pos.items())})
    if len(rounds) == 2:
        r1, r2 = rounds
        both = [i for (r, i) in latest if r == r1 and (r2, i) in latest]
        same = sum(latest[(r1, i)]["choice"] == latest[(r2, i)]["choice"] for i in both)
        fam = lambda c: "plan" if c.endswith("_plan") else c
        same_dec = sum(fam(latest[(r1, i)]["choice"]) == fam(latest[(r2, i)]["choice"]) for i in both)
        print(f"cross-round: same condition {same}/{len(both)} · same decomposition-vs-single side {same_dec}/{len(both)}")


if __name__ == "__main__":
    main()
