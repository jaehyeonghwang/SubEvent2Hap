# Vertical ball-bounce prototype

Select **공 튀기기 · 물리 모델** in the compose tab, enter a passive vertical-drop
description and click the existing decomposition button. Set optional height,
restitution, mass and count fields to override extracted parameters. An empty field
means: use the explicit value extracted from text, otherwise use the labeled default.

Example input:

> A rubber ball is dropped from 1 metre onto a wooden floor and bounces four times. Restitution is 0.6.

Pipeline: one local LM call extracts material/type and explicitly stated numeric
conditions. The mechanics solver calculates collision times, incoming velocities,
rebound heights and impulses. A fixed single-impact caption template is shared across
all contacts. LM does not choose impact times or a decay ratio. No material lookup
table or measured coefficient database is included; material names affect the caption
only. This prototype does not yet infer a calibrated restitution coefficient from
material properties.

## Model

Dropped from rest, rigid stationary horizontal floor, gravity g=9.80665 m/s²,
constant coefficient of restitution 0<=e<=1, no drag/spin/horizontal motion.

- First contact: t1 = release_time + sqrt(2h/g).
- Incoming speed of contact k: vk = sqrt(2gh) * e^(k-1).
- Interval following contact k: dt = 2e*vk/g.
- Rebound height: (e*vk)^2/(2g).
- Normal impulse magnitude: Jk = m(1+e)vk.

Reference: https://mechanics.ju.se/Kinetics/ImpulseMomentum.html

Relative renderer intensity is Jk/J1. Physical impulse is not calibrated actuator
acceleration or perceived strength. When physics mode creates a plan, the UI selects
gamma=1 and shared samples. Existing candidate extraction/fades/global gain still
apply; later parameter edits can change the physical intensity relationship.

**Pulse duration is not solved contact time.** Rigid impact has instantaneous contact;
0.15 seconds is the default rendering support window. Users may change it. When
successive pulses overlap, times are preserved and this fact is disclosed. No
artificial minimum gap stretches the physical timeline.

Defaults (unmeasured assumptions): h=1m, e=0.7, mass=0.05kg, release at 0s,
4 rendered contacts. The count limits the excerpt, not a prediction of physical rest.
A finite restitution model can continue bouncing indefinitely in the ideal limit;
the prototype does not add a fabricated rest/rolling vibration. Incompatible cases
(active dribbling, slope, prescribed contact times, rolling continuation, >8 contacts,
or an end time >10s) are rejected rather than silently retimed. LM scenario recognition
and extraction can still be wrong: inspect the setup/assumptions panel.

## Reproducibility

Response `physics` includes extracted setup, resolved values, per-parameter provenance,
equations, contact table, caveats and the original sub-event plan. Composed runs retain
this in `out/compose/<run>/plan.json`; `physics_plan_edited` indicates whether the
submitted event/segments differ from the reference. The metadata is an audit trail,
not independent measured ground truth.

```powershell
.\.venv\Scripts\python.exe scripts\bounce.py "A ball is dropped and bounces four times" --height 1 --restitution 0.6 --out out/bounce_physics_demo.json
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p test_*.py
```

API: `POST /api/decompose` with `event`, `planner: "ball_physics"`, and optional
`bounce_parameters` (`drop_height_m`, `restitution`, `mass_kg`, `release_time_s`,
`impact_count`, `pulse_duration_s`). Override > text extraction > default.
Generic events retain the existing staged planner, selected explicitly in the UI.

## Initial smoke check

`out/bounce_physics_smoke.json` records one seed-42 comparison with h=1m, e=0.6,
4 contacts and gamma=1. The composed waveform had 4 detected/matched contacts,
impact F1=1 and matched onset MAE about 11.6ms. These are fidelity checks against
the same analytical plan, not independent validation of realism. The paired baseline
had 2 detected contacts with 1 matched; its timing MAE must not be compared without
the count/F1 because it includes only that one match. The saved run's `plan.json`
contains the physics parameter/provenance table and original reference segments.
