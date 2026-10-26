"""Analytic carrier/envelope controls, independent of model loading."""
import math
import numpy as np


def carrier(spec, length, sr):
    f, phase = float(spec['frequency_hz']), float(spec.get('phase_rad', 0.))
    if spec.get('kind') != 'sine' or not math.isfinite(f) or not math.isfinite(phase) or not 0 < f <= 500 or f >= sr / 2:
        raise ValueError('Invalid analytic sine carrier')
    return np.sin(2 * np.pi * f * np.arange(length) / sr + phase).astype(np.float32)


def envelope(spec, length, sr):
    if spec['kind'] == 'exponential':
        tau = float(spec['tau_s'])
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('Envelope time constant must be positive and finite')
        return np.exp(-np.arange(length) / sr / tau).astype(np.float32)
    if spec['kind'] == 'linear':
        ratio = float(spec['end_ratio'])
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('Invalid linear envelope end ratio')
        return np.linspace(1, ratio, length).astype(np.float32)
    raise ValueError('Unsupported envelope')
