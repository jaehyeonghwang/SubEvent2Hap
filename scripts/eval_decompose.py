"""Collect one prompt-driven decomposition per event for inspection.

This is a smoke run, not a semantic accuracy benchmark. Failures are recorded
without repairing the output or retrying the model.

Usage:
    python scripts/eval_decompose.py --model qwen3-4b [--out out/decompose_eval.json]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decompose as dc  # noqa: E402

EVENTS = [
    "A ball bounces three times on a wooden floor and then rolls away",
    "A sword clashes against a shield, then scrapes off",
    "Heavy rain starts, a thunder strikes nearby, then the rain fades",
    "Opening a soda can and pouring it into a glass",
    "Someone knocks on a door three times and the door creaks open",
    "A car engine starts, revs twice, and the car drives away",
    "Running footsteps, then a jump and a heavy landing",
    "Typing quickly on a keyboard and slamming the enter key",
    "An archer draws a bow, releases the arrow, and it hits a target",
    "A cat jumps onto your lap and starts purring",
    "A balloon is inflated until it pops",
    "A train approaches, passes by, and fades into the distance",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=dc.DEFAULT_MODEL)
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results, n_failed, total_t = [], 0, 0.0
    for ev in EVENTS:
        try:
            r = dc.decompose(ev, args.model)
        except Exception as exc:
            n_failed += 1
            results.append({"event": ev, "model": args.model, "prompt_version": dc.PROMPT_VERSION,
                            "error": str(exc)})
            if not args.quiet:
                print(f"FAILED: {ev}: {exc}")
            continue
        total_t += r["elapsed"]
        results.append({"event": ev, **r})
        if not args.quiet:
            print(f"\n== {ev}  ({r['elapsed']:.1f}s, {r['prompt_version']})")
            for s in r["sub_events"]:
                print(f"  {s['start']:5.2f}s +{s['duration']:4.2f}s I={s['intensity']:.2f} {s['transition']:9s} {s['prompt']}")
    n_ok = len(EVENTS) - n_failed
    avg = f"{total_t / n_ok:.1f}s" if n_ok else "n/a"
    print(f"\nSUMMARY model={args.model}: executable plans {n_ok}/{len(EVENTS)}, "
          f"avg successful call {avg}. Semantic correctness is not scored.")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
