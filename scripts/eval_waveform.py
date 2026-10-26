"""Offline paired waveform evaluation; no model loading or headset required.

Default references are saved plans: scores measure plan fidelity, NOT text semantics.
Use --references JSON mapping run directory names to independently annotated
{sub_events: [...], total: seconds} objects for an independent reference.
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import spearmanr


def detect(w, sr, threshold=0.12):
    """Global, reference-blind detector; threshold relative to peak RMS.

    10 ms RMS, positive 10 ms envelope change, 80 ms refractory period.
    Padding allows an onset at t=0. Returns onset estimates, not envelope peaks.
    """
    env = np.sqrt(np.maximum(0, uniform_filter1d(w * w, max(1, round(.01 * sr)))))
    hop = max(1, round(.01 * sr))
    novelty = np.maximum(0, env - np.concatenate((np.zeros(hop), env))[:len(env)])
    padded = np.pad(novelty, (1, 1))
    peaks, _ = find_peaks(padded, height=max(1e-5, threshold * float(env.max())),
                          distance=max(1, round(.08 * sr)), prominence=max(1e-5, .04 * float(env.max())))
    return np.maximum(0, peaks - 1 - hop / 2) / sr


def match(expected, detected, tolerance):
    if not len(expected) or not len(detected):
        return []
    cost = np.abs(np.asarray(expected)[:, None] - np.asarray(detected)[None, :])
    # Penalize invalid edges sufficiently to maximize valid matches first.
    assignment = np.where(cost <= tolerance, cost, (len(expected) + len(detected) + 1) * tolerance)
    rows, cols = linear_sum_assignment(assignment)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] <= tolerance]


def evaluate(w, sr, reference, tolerance=.1, threshold=.12):
    total = float(reference['total'])
    n = round(total * sr)
    # Common intended duration: excludes composite's deliberate 100 ms tail.
    coverage = min(len(w), n) / n
    w = np.pad(w[:n], (0, max(0, n - len(w))))
    subs = reference['sub_events']
    impacts = [s for s in subs if s['transition'] == 'cut' and s['duration'] <= .6 and s['intensity'] > 0]
    expected = [s['start'] for s in impacts]
    detected = detect(w, sr, threshold)
    # Restrict scoring to impact regions; continuous textures are not labeled impacts.
    # Includes surrounding gaps to count spurious attacks, excludes long texture regions.
    mask = np.ones(n, dtype=bool)
    for s in subs:
        if s not in impacts:
            mask[max(0, round(s['start'] * sr)):min(n, round((s['start'] + s['duration']) * sr))] = False
    detected = [float(t) for t in detected if t < total and mask[min(n - 1, round(t * sr))]]
    pairs = match(expected, detected, tolerance)
    tp = len(pairs)
    f1 = 2 * tp / (len(expected) + len(detected)) if expected else None
    errors = [abs(expected[i] - detected[j]) * 1000 for i, j in pairs]
    active = np.zeros(n, dtype=bool)
    for s in subs:
        active[max(0, round((s['start'] - .06) * sr)):min(n, round((s['start'] + s['duration'] + .02) * sr))] = True
    energy = float(np.sum(w * w))
    levels = []
    for s in impacts:
        a = round(s['start'] * sr)
        seg = w[a:min(n, a + round(.06 * sr))]
        levels.append(float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.)
    target = [s['intensity'] for s in impacts]
    rho = float(spearmanr(target, levels).statistic) if len(target) >= 3 and np.ptp(target) > 0 and np.ptp(levels) > 0 else None
    return dict(impact_f1=f1, onset_mae_ms=float(np.mean(errors)) if errors else None,
                expected_impacts=len(expected), detected_impacts=len(detected), matched_impacts=tp,
                intensity_spearman=rho, gap_energy_ratio=float(np.sum(w[~active] ** 2) / energy) if energy > 1e-12 and (~active).any() else None,
                peak=float(np.max(np.abs(w))), saturation_ratio=float(np.mean(np.abs(w) >= .999)),
                duration_coverage=coverage, detected_onsets=detected, measured_attack_rms=levels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / 'out')
    ap.add_argument('--references', type=pathlib.Path)
    ap.add_argument('--threshold', type=float, default=.12)
    ap.add_argument('--tolerance', type=float, default=.1)
    ap.add_argument('--report', type=pathlib.Path, default=pathlib.Path('out/evaluation/waveform'))
    args = ap.parse_args()
    if args.threshold <= 0 or args.tolerance <= 0:
        ap.error('threshold and tolerance must be positive')
    refs = json.loads(args.references.read_text(encoding='utf-8')) if args.references else None
    rows, skipped = [], []
    for path in sorted((args.out_dir / 'compose').glob('*/plan.json')):
        plan = json.loads(path.read_text(encoding='utf-8'))
        if refs is not None and path.parent.name not in refs:
            skipped.append(dict(run=path.parent.name, reason='no independent reference'))
            continue
        if not plan.get('baseline') or not all((args.out_dir / plan[k]).is_file() for k in ('baseline', 'composite')):
            skipped.append(dict(run=path.parent.name, reason='missing paired WAV'))
            continue
        ref = refs[path.parent.name] if refs is not None else plan
        for method in ('composite', 'baseline'):
            w, sr = sf.read(args.out_dir / plan[method], dtype='float64')
            if w.ndim > 1:
                w = w.mean(axis=1)
            rows.append(dict(run=path.parent.name, event=plan['event'], method=method,
                             **evaluate(w, sr, ref, args.tolerance, args.threshold)))
    metrics = ('impact_f1', 'onset_mae_ms', 'intensity_spearman', 'gap_energy_ratio', 'saturation_ratio', 'duration_coverage')
    summary = {}
    for method in ('composite', 'baseline'):
        summary[method] = {}
        for key in metrics:
            vals = [r[key] for r in rows if r['method'] == method and r[key] is not None]
            summary[method][key] = dict(mean=float(np.mean(vals)) if vals else None, n=len(vals))
    report = dict(reference='independent' if refs is not None else 'saved_plan_proxy',
                  settings=dict(threshold=args.threshold, tolerance_seconds=args.tolerance),
                  limitations=['Plan fidelity does not establish text semantics or perceived quality.',
                               'Short cut segments are impact proxies; detector is not calibrated on real haptic annotations.',
                               'Existing runs are exploratory, possibly duplicated and not controlled seeds.',
                               'Timing MAE includes matched onsets only; always read alongside F1.',
                               'Global relative threshold may miss weak late impacts. No Quest actuator model is applied.'],
                  summary=summary, rows=rows, skipped=skipped)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with args.report.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'event', 'method', *metrics, 'expected_impacts', 'detected_impacts', 'matched_impacts'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    print(json.dumps(dict(pairs=len(rows)//2, skipped=skipped, summary=summary), indent=2))


if __name__ == '__main__':
    main()
