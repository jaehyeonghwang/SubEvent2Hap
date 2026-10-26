# Staged LM decomposition

`decompose.py` uses four sequential local LM calls without few-shot examples.
Prompt version: `staged_v2`.

1. **Events**: list physical phases and individual vibration-producing occurrences.
   Each occurrence contains only `source` and `action`. The LM decides boundaries,
   repetitions and silent phases. There are no LM-authored reference IDs.
2. **Review**: check the outline against the original input and return a corrected
   outline with a short change list. This stage handles meaning, not numerical planning.
3. **Timing**: assign start, duration, relative intensity and transition to each event.
   Output keys are fixed from list positions (`event_1`, etc.). The LM chooses values.
4. **Captions**: write standalone English captions from the reviewed source/action
   descriptions. This writer sees only the actions, avoiding whole-event counts and
   timing leaking into individual captions. Exact duplicate source/action descriptions
   share one caption. Code does not infer whether differently worded actions are equivalent.

The first three calls receive the original description. Intermediate outputs and call
durations are returned in `stages`; `review_changes` records the LM's claimed
corrections. These notes are not independent evidence of semantic correctness.
The original `phases`/`sub_events` API fields remain compatible with the editor/composer.

Code handles output keys, exact-description reuse and typed JSON validation only.
Renderer bounds: 1-8 events, duration >=0.15 seconds, end <=10 seconds, finite numbers,
intensity in [0,1]. A schema/range violation is sent back to that same stage for
one LM correction attempt; if still invalid it raises an error. Both invalid output
and validation message are recorded in `contract_corrections`. Normal runs use four
calls, with at most eight if every stage needs one contract correction. There are no
semantic regex checks, fixed decay ratios, clamps, timeline shifts, code-authored
repairs, or automatic model switching. Each stage has a short, dedicated prompt.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\decompose.py "A panel is struck at 2 seconds"
.\.venv\Scripts\python.exe scripts\eval_decompose.py --quiet --out out/decompose_staged_v2.json
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p test_*.py
```

`eval_decompose.py` records executable plans/errors and per-stage outputs. It does
not score semantic accuracy. Multiple calls improve inspectability but can still
propagate errors; independent annotated cases remain necessary for quality claims.
Historical exploratory outputs are retained, including failed intermediate designs.

Waveform extraction and mixing in `compose.py` retain the existing implementation.
Restart a running web server to load a changed version.
