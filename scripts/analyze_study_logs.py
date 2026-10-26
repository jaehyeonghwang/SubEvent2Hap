"""Summarise authoring-study logs written by the editor in study mode (/editor?p=P01&task=T1).

One row per session (out/study/<participant>/<task>_<session>.jsonl):
  task time (task_start -> task_end), time to the first composition, counts of timeline edits by
  kind, description edits, decompositions, compositions, samples generated vs reused, explicit
  regenerations, undo/redo and playbacks. Writes out/study/summary.csv.

  python scripts/analyze_study_logs.py [--dir out/study]
"""
import argparse
import collections
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

TIMELINE = ("create", "add", "duplicate", "delete", "move", "resize", "key_move", "key_resize", "key_intensity", "pack")
COLUMNS = ["participant", "task", "session", "completed", "task_time_s", "first_compose_s", "decompositions",
           "timeline_edits", "field_edits", "prompt_edits", "compositions", "samples_generated", "samples_reused",
           "regenerations", "undo", "redo", "plays", "final_sub_events"] + [f"n_{k}" for k in TIMELINE]


def summarise(path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = collections.Counter(e["type"] for e in events)
    start = next((e["t"] for e in events if e["type"] == "task_start"), events[0]["t"] if events else None)
    end = next((e for e in events if e["type"] == "task_end"), None)
    gens = [e for e in events if e["type"] == "generate"]
    fields = [e for e in events if e["type"] == "field_edit"]
    row = {
        "participant": path.parent.name, "task": path.stem.split("_")[0], "session": path.stem.split("_", 1)[-1],
        "completed": end is not None,
        "task_time_s": round(end["duration_s"], 1) if end else None,
        "first_compose_s": round((gens[0]["t"] - start) / 1000, 1) if gens and start else None,
        "decompositions": kinds["decompose"],
        "timeline_edits": sum(kinds[k] for k in TIMELINE),
        "field_edits": len(fields),
        "prompt_edits": sum(e["field"] == "prompt" for e in fields),
        "compositions": len(gens),
        "samples_generated": sum(e["generated"] for e in gens),
        "samples_reused": sum(e["reused"] for e in gens),
        "regenerations": sum(e["mode"] == "regen" for e in gens),
        "undo": kinds["undo"], "redo": kinds["redo"], "plays": kinds["play"],
        "final_sub_events": len(end["subs"]) if end else None,
    }
    row.update({f"n_{k}": kinds[k] for k in TIMELINE})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "out" / "study"))
    a = ap.parse_args()
    root = pathlib.Path(a.dir)
    rows = [summarise(p) for p in sorted(root.glob("*/*.jsonl"))]
    if not rows:
        print(f"no logs under {root}")
        return
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    show = ["participant", "task", "completed", "task_time_s", "first_compose_s", "decompositions", "timeline_edits",
            "field_edits", "compositions", "samples_generated", "samples_reused", "regenerations", "plays"]
    print("  ".join(show))
    for r in rows:
        print("  ".join(str(r[k]) for k in show))
    print(f"-> {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
