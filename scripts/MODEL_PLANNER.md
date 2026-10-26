# LM-selected model hierarchy (model_first_v4)

The default planner is `model_planner.py`. The LM selects the exact model name,
chooses whether to split, and identifies entities and state links. Code has no
numerical model priority or keyword router. The old staged planner and manual
bounce planner remain available.

## Model library: nine models in seven families

| Family | Models | Computation |
|---|---|---|
| Collision | vertical_bounce, single_impact | Ballistic contact timing; J=m(1+e)v |
| Flight | free_fall | Vertical flight to contact, with no vibration |
| Friction | sliding, rolling | Constant resistance; coasting or externally maintained speed |
| Damping | damped_oscillation | Underdamped sine and exponential amplitude decay |
| Periodic | periodic | Exact sinusoidal carrier; optional RPM/60 conversion |
| Stochastic | stochastic | Seeded Poisson arrivals, or fixed-count conditional arrivals |
| Behavior | rhythmic_action | Intentional cadence, count and explicit spacing/strength trends |

Rhythm and stochastic processes are behavioral/probabilistic designs rather than
physical laws. Mechanics are idealized one-dimensional models, not a universal simulator.

## Planning and stopping

An atomic regime keeps the same **contact mode, contact constraints and driving
mechanism**. Continuous movement alone does not make an action atomic. Flight,
transient impact and sustained contact have different conditions and can require
separate phases. A change in driving conditions can also establish a boundary.
Speed, direction, frequency or amplitude changing under unchanged conditions does
not by itself require a split. Gradual stopping is the endpoint of that motion;
rest is not an extra vibration. Unsupported geometry does not make a compound
action atomic. A scenario model can still cover several regimes as one leaf:
model coverage and physical atomicity are distinct stopping reasons.

The same criterion is shared by routing, atomicity assessment, phase extraction
and split review. It is an LM instruction, not a keyword classifier in code.
Phase extraction records the LM's contact-mode and governing-condition descriptions
before its action text. The hierarchy saves the criterion and the UI displays it.
Atomicity assessment also specifies an ordered contact-mode sequence and its count.
Extraction schemas retain each mode in its position, while placement retains the
count, so those stages cannot append a separate stopping phase or replace an impact
slot with another rolling slot. The descriptions themselves can still be mistaken.
The count and physical descriptions remain LM judgments, not independent validation.
The rolling/sliding solver scopes explicitly require flat horizontal surfaces;
curved geometry must remain in an estimate rather than being silently flattened.

1. LM selects apply(model), split(children), or fallback.
2. For apply, a separate call checks applicability and extracts model-specific inputs.
   A whole-scenario model ends decomposition even when it emits multiple contacts.
3. On the first whole-action applicability rejection, a separate LM structure review
   first judges atomicity without seeing the model catalog. A second small call
   lists phases only if that judgment says the action is compound. If compound,
   it splits immediately rather than exhausting every whole-action solver. If atomic,
   model selection can continue with rejected names excluded. Code never picks a model.
   Every fallback also requires this structure review (reused if already performed).
   Lack of a whole-action solver is not evidence that an action is indivisible.
4. A separate LM review checks split coverage, timing placement and parameter meanings.
   Split creates 2-4 independently modeled children. An indivisible unsupported
   action becomes one explicitly labeled LM estimate.

The default limits are depth 3, 15 nodes, 8 rendered events and 10 seconds.
Depth/no-progress stops are labeled; overflow raises an error without compression.
The LM sees format contracts, descriptions and parameter definitions, without
few-shot examples. Semantic decisions remain fallible.

## Time and state connections

Children use local clocks. after_previous starts at the previous physical endpoint;
after_pause adds an explicitly stated pause; absolute uses an explicit parent offset;
parallel starts at the parent origin. Unspecified sequence gaps are zero.
A 0.15-second rendered collision tail does not delay the next physical process.

For directly sequential children referring to the same entity, a separate LM call
chooses none, same_object, or motion_to_normal using the computed preceding state.
Named mass, velocity, energy and oscillator phase can be inherited where the next
solver accepts them. The motion_to_normal link explicitly assumes a wall normal
aligned with the direction of horizontal motion. Unrelated entities and unmodeled
pauses cannot inherit state. A vertical bounce never fabricates horizontal rolling
speed. Translational kinetic energy is not automatically assigned to a ringing mode:
that requires explicit modal excitation energy or known modal energy from a predecessor.

Every parameter records explicit, inherited, or assumed-default provenance. Explicit
new parameters override inherited values. State links, geometric assumptions and
LM stages are retained in the hierarchy for inspection.

## Defaults, synthesis and limits

Missing numeric values use disclosed design defaults, not LM-invented measurements.
Examples: mass 0.05 kg, bounce height 1 m and restitution 0.7; sliding speed 0.8 m/s,
kinetic friction 0.3; effective rolling resistance 0.03; periodic frequency 120 Hz.
Damping uses a 1% amplitude cutoff when duration is absent. Maintained constant speed
assumes external force balances resistance. RPM conversion assumes one cycle/revolution.
These choices require calibration for particular objects, surfaces and actuators.

Damped/periodic models synthesize their sinusoidal carrier directly. Sliding/rolling
use HapticGen texture with a computed speed-proxy envelope. Other contacts use
HapticGen waveforms placed at computed/model-designed times. Minimal render support
is 0.15 seconds and is not a Hertzian contact-duration prediction. Amplitudes remain
normalized within leaves and uncalibrated between models or users. Waveform texture,
perceived realism and absolute device acceleration are not physically validated.

Pure free fall or an empty stochastic realization can emit no vibration. The UI
reports a silent result instead of creating a fictitious contact.

## Inspect and verify

API hierarchy stores selection_owner=LM, catalog, decision tree, stop reasons,
process/render durations, equations, states, assumptions, provenance and reference
segments. Composed plan.json retains it and flags subsequent segment edits.
The UI displays model choices and state connections. Gamma=1 is selected for this
mode so a second nonlinear amplitude mapping is not silently applied.

```powershell
.\.venv\Scripts\python.exe scripts\model_planner.py "A ball falls and bounces four times" --out out/plan.json
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p test_*.py
.\.venv\Scripts\python.exe scripts\eval_atomicity.py --out out/atomicity_regression.json
```

Tests cover analytic formulas, stopping, LM model execution, state transfer, bounds,
seed reproducibility, carrier phase and envelopes. Local LM smoke cases exercise
routing and actual API synthesis. These are implementation checks, not a benchmark
of LM routing accuracy or perceived haptic quality.

Regression case: a coin falling, spinning, wobbling faster and settling must not
become a single fallback merely because no whole-sequence solver exists. Initial
flight/contact can be modeled separately; unsupported rotational motion remains
an explicitly labeled estimate, not a validated coin-dynamics simulation. Structure
review is another LM judgment, not proof of atomicity. The UI labels it accordingly.

Local regression artifacts: `out/model_v3_coin.json` and
`out/model_v3_coin_api.json` record the pasted coin case after the fix.
The result separates initial contact from unsupported rotation/settling estimates.
This verifies the reproduced case, not general decomposition accuracy or physical
realism of the estimated rotating-coin waveform.

## Unstated bounce count

A stated count (including "once") is used as written; more than 8 contacts is an error.
When the count is not stated, it is derived from restitution instead of a fixed default:
contacts continue while the impulse relative to the first contact, e^(k-1), is at least
`BOUNCE_STOP_RATIO` = 0.10 (at gamma=1 the amplitude is also 10%, barely perceptible).
Provenance is `derived_until_threshold`. Contacts beyond the 8-event render limit or the
10-second window are not output; their number, reason and first omitted time are recorded
in `physics.omitted` and returned as `warnings`, which the UI shows. Restitution 1 never
decays below the threshold and is reported as unbounded. The 10% threshold is a design
choice, not a measured perception threshold.
