"""Parametric vibration patterns for story events whose identity is their rhythm (after Feel Effects,
Israr et al., ACM TAP 2014: a phrase is paired with a pattern of pulse spacing, duration, ramp and
intensity). Generative models blur these into similar buzzes; a pattern keeps a knock a knock and a
heartbeat a heartbeat. Everything stays in the band vibrotactile actuators render (about 60-250 Hz).

  synth(family, duration, rate=None, weight=.5, hardness=.5) -> mono float32 wave at SR

families: strike (one knock/tap/hit/thud), steps (walking, running, stomping), rain (drops),
heartbeat, shake (trembling, shivering, rattling), rumble (thunder, engine, purr, roar).
"texture" is not synthesised here: it goes to the text-to-vibration model.
"""
import numpy as np

SR = 8000
FAMILIES = ("strike", "steps", "rain", "heartbeat", "shake", "rumble", "texture")
DEFAULT_RATE = {"steps": 2.0, "rain": 12.0, "heartbeat": 1.2, "shake": 9.0}


def _hit(sr, freq, tau, length):
    """A damped sinusoid with a 2 ms attack: one tap / knock / footfall."""
    t = np.arange(int(length * sr)) / sr
    env = np.minimum(1.0, t / 0.002) * np.exp(-t / tau)
    return (env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _place(out, w, at, gain=1.0):
    a = int(at)
    if a >= len(out):
        return
    n = min(len(w), len(out) - a)
    out[a:a + n] += gain * w[:n]


def synth(family, duration, rate=None, weight=0.5, hardness=0.5, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    n = max(1, int(duration * sr))
    out = np.zeros(n, dtype=np.float32)
    weight, hardness = float(np.clip(weight, 0, 1)), float(np.clip(hardness, 0, 1))
    rate = float(rate) if rate else DEFAULT_RATE.get(family, 1.0)
    f0 = 230.0 - 110.0 * weight                      # heavier sources vibrate lower
    tau = 0.012 + 0.07 * (1.0 - hardness)            # softer contacts ring longer
    if family == "strike":
        _place(out, _hit(sr, f0, tau, min(duration, 0.35)), 0)
        if weight > 0.65:                            # heavy impacts get a low body thump
            _place(out, _hit(sr, 70.0, 0.06, min(duration, 0.35)), 0, 0.6 * weight)
    elif family == "steps":
        period = 1.0 / max(0.5, min(rate, 12.0))
        t, k = 0.0, 0
        while t < duration:
            _place(out, _hit(sr, f0, tau, min(0.2, period)), t * sr, 1.0 if k % 2 == 0 else 0.8)
            t += period * rng.uniform(0.93, 1.07)
            k += 1
    elif family == "rain":
        t = rng.exponential(1.0 / max(1.0, rate))
        while t < duration:
            _place(out, _hit(sr, rng.uniform(190, 250), 0.006 + 0.01 * (1 - hardness), 0.04), t * sr, rng.uniform(0.35, 1.0))
            t += rng.exponential(1.0 / max(1.0, rate))
    elif family == "heartbeat":
        period = 1.0 / max(0.6, min(rate, 3.0))
        t = 0.0
        while t < duration:
            _place(out, _hit(sr, 150.0, 0.045, 0.2), t * sr, 1.0)            # lub
            _place(out, _hit(sr, 150.0, 0.035, 0.2), (t + 0.16) * sr, 0.65)  # dub
            t += period
    elif family == "shake":
        t = np.arange(n) / sr
        am = (0.5 + 0.5 * np.sin(2 * np.pi * min(max(rate, 3.0), 20.0) * t)) ** 1.5
        out += (am * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    elif family == "rumble":
        t = np.arange(n) / sr
        k = max(1, int(sr / 4))                      # slowly wandering loudness
        drift = np.convolve(rng.standard_normal(n + k), np.ones(k) / k, mode="valid")[:n]
        drift = 0.6 + 0.4 * drift / (np.abs(drift).max() + 1e-9)
        carrier = np.sin(2 * np.pi * (75 + 30 * (1 - weight)) * t) + 0.5 * np.sin(2 * np.pi * 118 * t + 1.3)
        out += (drift * carrier).astype(np.float32)
    else:
        raise ValueError(f"no parametric pattern for family {family!r}")
    fade = min(n // 4, int(0.005 * sr))
    if fade:
        out[:fade] *= np.linspace(0, 1, fade)
        out[-fade:] *= np.linspace(1, 0, fade)
    peak = float(np.abs(out).max())
    return out / peak if peak > 1e-6 else out
