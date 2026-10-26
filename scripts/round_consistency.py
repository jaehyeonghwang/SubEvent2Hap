"""Do blind choices survive a new HapticGen seed? (round 1 = seed 1, round 2 = seed 2; same plans)

Per item: same condition chosen in both rounds? Agreement is compared with chance agreement from
the rater's own choice frequencies (Cohen's kappa), overall and split by the vote_attribution
verdict (whether round-1 choices were explained by plan differences).

  python scripts/round_consistency.py [--rater me] [--rounds 1,2]
"""
import argparse
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    return po, (po - pe) / (1 - pe) if pe < 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", default="me")
    ap.add_argument("--rounds", default="1,2")
    a = ap.parse_args()
    r1, r2 = (int(x) for x in a.rounds.split(","))
    votes = json.loads((OUT / "gallery" / "votes.json").read_text(encoding="utf-8"))
    latest = {}
    for v in votes:
        if v["rater"] == a.rater and v["blind"]:
            latest[(v.get("round", 1), v["item"])] = v["choice"]
    items = sorted({i for (r, i) in latest if r == r1} & {i for (r, i) in latest if r == r2})
    pairs = [(latest[(r1, i)], latest[(r2, i)]) for i in items]
    po, k = kappa(pairs)
    print(f"items voted in both rounds: {len(items)}")
    if not items:
        return
    print(f"same choice in both rounds: {sum(x == y for x, y in pairs)}/{len(items)} = {po:.0%} | Cohen's kappa {k:.2f}")
    print("round tallies:", {r: dict(collections.Counter(latest[(r, i)] for i in items)) for r in (r1, r2)})
    print("decomposition (A/B/C) vs baseline kept:", sum((x == "baseline") == (y == "baseline") for x, y in pairs), "/", len(items))
    attr_path = OUT / "evaluation" / "vote_attribution.json"
    if attr_path.exists():
        verdict = {}
        for r in json.loads(attr_path.read_text(encoding="utf-8")):
            c, ch = r["conditions"], r["choice"]
            passes = {kk: vv["plan_pass"] for kk, vv in c.items()}
            verdict[r["item"]] = ("plan differs" if ch in c and passes[ch] and not all(passes.values())
                                  else "plans equal" if ch in c and len(set(passes.values())) == 1 else "other")
        for group in ("plan differs", "plans equal", "other"):
            sub = [(x, y) for i, (x, y) in zip(items, pairs) if verdict.get(i) == group]
            if sub:
                po_g, k_g = kappa(sub)
                print(f"  round-1 '{group}': same choice {sum(x == y for x, y in sub)}/{len(sub)} = {po_g:.0%}")
    for i, (x, y) in zip(items, pairs):
        print(f"  {i}: {x:14s} -> {y:14s} {'same' if x == y else 'CHANGED'}")


if __name__ == "__main__":
    main()
