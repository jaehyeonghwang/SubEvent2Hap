"""Planner ablation on the HapticGen-dataset eval set.

  python scripts/ablation.py run   [--variants full,-models,...]   # LM calls; resumable
  python scripts/ablation.py score                                # offline; re-run after editing annotations

Plans are stored per variant in out/ablation/plans_<variant>.json, so annotation changes only
need `score`. Checks follow the per-event scoring rules in data/eval_set_v2.json.
"""
import argparse
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decompose as dc  # noqa: E402
import model_planner as mp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval_set_v2.json"
OUT = ROOT / "out" / "ablation"

VARIANTS = {
    "full": lambda e: mp.decompose(e),
    "-models": lambda e: mp.decompose(e, ablate={"models"}),
    "-structure_review": lambda e: mp.decompose(e, ablate={"structure_review"}),
    "-split_review": lambda e: mp.decompose(e, ablate={"split_review"}),
    "-state_binding": lambda e: mp.decompose(e, ablate={"state_binding"}),
    "-alternative_selection": lambda e: mp.decompose(e, ablate={"alternative_selection"}),
    "+estimate": lambda e: mp.decompose(e, estimate_parameters=True),
    "staged": lambda e: dc.decompose(e),
    "staged-review": lambda e: dc.decompose(e, skip_review=True),
    # Same staged planner, cloud LM (sends prompts to OpenAI).
    "staged@gpt-4o": lambda e: dc.decompose(e, *dc.backend("gpt-4o")),
    # One plain LM call for the whole plan (is the 4-stage structure needed?).
    "oneshot@gpt-4o": lambda e: dc.decompose_oneshot(e, *dc.backend("gpt-4o")),
    "oneshot@local": lambda e: dc.decompose_oneshot(e),
}


def run(variants):
    items = json.loads(EVAL.read_text(encoding="utf-8"))["items"]
    OUT.mkdir(parents=True, exist_ok=True)
    for v in variants:
        path = OUT / f"plans_{v}.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        for it in items:
            if it["id"] in done:
                continue
            t0 = time.time()
            try:
                r = _with_retry(lambda: VARIANTS[v](it["prompt"]))
                rec = {"ok": True, "sub_events": [{k: s[k] for k in ("prompt", "start", "duration", "intensity", "transition")}
                                                  for s in r["sub_events"]],
                       "lm_calls": len(r.get("stages", [])),
                       "leaf_models": [l.get("model") for l in _leaves(r)]}
            except urllib.error.URLError as exc:
                # Server/transport errors (e.g. context overflow under parallel load) are not planner
                # outcomes: leave the item undone so a later run retries it.
                print(f"[{v}] {it['id']} INFRA {exc} (not recorded)", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001  (a failed plan is a scored outcome)
                rec = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400], "sub_events": [], "lm_calls": None}
            rec["seconds"] = time.time() - t0
            done[it["id"]] = rec
            path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{v}] {it['id']} {'ok' if rec['ok'] else 'FAIL'} {rec['seconds']:.1f}s "
                  f"{len(rec['sub_events'])} seg", flush=True)


def _with_retry(fn, attempts=3, wait=5.):
    for k in range(attempts):
        try:
            return fn()
        except urllib.error.URLError:
            if k == attempts - 1:
                raise
            time.sleep(wait * (k + 1))


def _leaves(r):
    tree = (r.get("hierarchy") or {}).get("tree")
    out, stack = [], [tree] if tree else []
    while stack:
        n = stack.pop()
        stack.extend(n.get("children") or [])
        if not n.get("children"):
            out.append(n)
    return out


# ---------------------------------------------------------------- scoring (eval_set_v2 rules)
SEMANTIC = ["present", "count", "order", "period", "relative", "forbid", "last_event"]


def _matches(event, sub):
    text = sub["prompt"].lower()
    return bool(re.search(event["pattern"], text)) and not (event.get("exclude") and re.search(event["exclude"], text))


def check(item, rec):
    """Return {check: bool} for conditions the item annotates; see the `scoring` block in the eval file."""
    subs = sorted(rec["sub_events"], key=lambda s: s["start"]) if rec["ok"] else []
    ev = {e["id"]: e for e in item.get("events", [])}
    hits = {k: [i for i, s in enumerate(subs) if _matches(e, s)] for k, e in ev.items()}
    res = {}
    if ev:
        res["present"] = all(hits[k] for k in ev)
    counted = {k: e["count"] for k, e in ev.items() if e.get("count") is not None}
    if counted:
        res["count"] = all(len(hits[k]) == n for k, n in counted.items())
    if item.get("order") and item.get("order_strength", "strict") == "strict":
        prev, ok = -1, True
        for k in item["order"]:
            nxt = next((i for i in hits[k] if i > prev), None)
            if nxt is None:
                ok = False
                break
            prev = nxt
        res["order"] = ok
    if item.get("period"):
        pr = item["period"]
        t = [subs[i]["start"] for i in hits[pr["event"]]]
        gaps = [b - a for a, b in zip(t, t[1:])]
        lo, hi = pr["target_s"] * (1 - pr["tol_rel"]), pr["target_s"] * (1 + pr["tol_rel"])
        res["period"] = len(t) >= 3 and all(lo <= g <= hi for g in gaps)
    if item.get("relative"):
        ok = True
        for r in item["relative"]:
            a, b = hits[r["stronger"]], hits[r["weaker"]]
            ok = ok and bool(a) and bool(b) and subs[a[0]]["intensity"] > subs[b[0]]["intensity"]
        res["relative"] = ok
    if item.get("forbid"):
        res["forbid"] = bool(subs) and not any(re.search(f, s["prompt"].lower()) for f in item["forbid"] for s in subs)
    if item.get("last_event"):
        res["last_event"] = bool(subs) and (len(subs) - 1) in hits[item["last_event"]]
    return res


def score():
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    items = data["items"]
    table = []
    for v in VARIANTS:
        path = OUT / f"plans_{v}.json"
        if not path.exists():
            continue
        plans = json.loads(path.read_text(encoding="utf-8"))
        rows = [(it, plans[it["id"]]) for it in items if it["id"] in plans]
        if not rows:
            continue
        by = {k: [] for k in SEMANTIC}
        all_pass, all_pass_rendered, per_item = [], [], {}
        for it, rec in rows:
            c = check(it, rec)
            per_item[it["id"]] = {"renderable": rec["ok"] and bool(rec["sub_events"]), **c}
            for k, ok in c.items():
                by[k].append(ok)
            if c:
                all_pass.append(all(c.values()))
                if per_item[it["id"]]["renderable"]:
                    all_pass_rendered.append(all(c.values()))
        calls = [r["lm_calls"] for _, r in rows if r.get("lm_calls")]
        table.append({"variant": v, "n": len(rows),
                      "renderable": sum(p["renderable"] for p in per_item.values()) / len(rows),
                      "items_all_pass": sum(all_pass) / len(all_pass) if all_pass else None,
                      "items_all_pass_given_rendered": (sum(all_pass_rendered) / len(all_pass_rendered)) if all_pass_rendered else None,
                      **{k: (sum(x) / len(x) if x else None) for k, x in by.items()},
                      **{f"n_{k}": len(x) for k, x in by.items()},
                      "lm_calls": statistics.mean(calls) if calls else None,
                      "seconds": statistics.mean(r["seconds"] for _, r in rows),
                      "per_item": per_item})
    (OUT / "scores_v2.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["renderable", "items_all_pass", "items_all_pass_given_rendered", *SEMANTIC]
    short = {"items_all_pass": "allpass", "items_all_pass_given_rendered": "pass|rend"}
    print(f"{'variant':24s} {'n':>3s} " + " ".join(f"{short.get(c, c)[:9]:>9s}" for c in cols) + f" {'calls':>6s} {'sec':>5s}")
    for r in table:
        cells = " ".join(f"{r[c] * 100:8.0f}%" if r[c] is not None else f"{'-':>9s}" for c in cols)
        print(f"{r['variant']:24s} {r['n']:3d} {cells} {r['lm_calls'] or 0:6.1f} {r['seconds']:5.1f}")
    if table:
        print("items scored per check:", {k: table[0][f"n_{k}"] for k in SEMANTIC})
    print("renderable = feasibility only; semantic columns count a non-renderable plan as failing that check.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    if a.mode == "run":
        run([v for v in a.variants.split(",") if v])
    else:
        score()
