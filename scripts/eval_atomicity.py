"""Live local-LM regression checks for regime boundaries, not a research benchmark.

Case expectations are hand specified. Cases are inputs only, never prompt examples.
"""
import argparse
import json
from pathlib import Path

import decompose as dc
import model_planner as mp


CASES = (
    ('bowl_drop_contact_roll', 'A steel marble drops into a metal bowl, strikes the bottom, rolls along the curved surface, and gradually stops.', False),
    ('coin_landing_rotation', 'A metal coin falls onto a table, spins, wobbles increasingly rapidly, and finally settles flat.', False),
    ('slide_wall_collision', 'A block slides along a floor and then hits a stationary wall.', False),
    ('drive_release', 'A block is pushed steadily along a floor, then the push is removed and it coasts to rest.', False),
    ('flat_rolling_deceleration', 'A ball rolls along a flat surface and gradually slows to rest under unchanged resistance.', True),
    ('accelerating_flight', 'An object falls freely without contact, accelerating under gravity until just before touching the ground.', True),
    ('curved_rolling_deceleration', 'A marble already rolling along the curved inner surface of a bowl remains in rolling contact as its direction changes and it gradually slows to rest.', True),
    ('decaying_ringing', 'After excitation has ended, a metal plate vibrates with continuously decreasing amplitude under unchanged damping.', True),
)


def evaluate(model=dc.DEFAULT_MODEL, url=dc.LMSTUDIO_URL):
    rows = []
    for name, event, expected in CASES:
        trace = []
        row = dict(name=name, event=event, expected_indivisible=expected)
        expected_count = 1 if expected else {'bowl_drop_contact_roll': 3,
                                           'slide_wall_collision': 2, 'drive_release': 2}.get(name)
        row['expected_regime_count'] = expected_count
        expected_modes = {'bowl_drop_contact_roll': ['free_flight', 'impact', 'rolling'],
                          'flat_rolling_deceleration': ['rolling'],
                          'accelerating_flight': ['free_flight'],
                          'curved_rolling_deceleration': ['rolling']}.get(name)
        row['expected_contact_modes'] = expected_modes
        try:
            judgment = dc._stage('structure_review', mp.ATOMICITY_PROMPT,
                                 {'current_action': event}, mp.Atomicity, model, url, 0., trace)
            row.update(judgment=judgment.model_dump(), passed=(judgment.indivisible == expected and
                       (expected_count is None or judgment.regime_count == expected_count) and
                       (expected_modes is None or judgment.contact_modes == expected_modes)))
        except Exception as exc:
            row.update(error=str(exc), passed=False)
        row['stages'] = trace
        rows.append(row)
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    return dict(prompt_version=mp.VERSION, model=model, scope='Hand-specified atomicity regressions only; not routing, waveform quality or a general benchmark.',
                passed=sum(r['passed'] for r in rows), total=len(rows), cases=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=dc.DEFAULT_MODEL)
    ap.add_argument('--url', default=dc.LMSTUDIO_URL)
    ap.add_argument('--out', default='out/atomicity_regression.json')
    args = ap.parse_args()
    result = evaluate(args.model, args.url)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{result['passed']}/{result['total']} passed")
    raise SystemExit(0 if result['passed'] == result['total'] else 1)
