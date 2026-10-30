"""Vibration generators behind one interface (the subset of AudioGen's API that compose uses).

  hapticgen  HapticGen fine-tuned weights (text -> 8 kHz vibration), the default.
  audiogen   Meta AudioGen-medium (text -> 16 kHz audio) followed by the audio-to-haptic conversion
             HapticGen's authors used for their "Baseline-AudioGen" condition
             (HapticGen/audiogen-beam-serve.py: amp_env_on_wav_norm). The conversion is vectorised
             here; `amp_env_reference` is the original loop, kept for the equivalence test.
"""
import math

import numpy as np
import torch

import hapticgen_generate as hg  # also installs the xformers stub needed by audiocraft

GENERATORS = ("hapticgen", "audiogen")
HAPTIC_SR = 8000
WANTED_BIN_SIZE_SEC = 0.010  # 10 ms RMS bins (as in the HapticGen baseline)
BASE_FREQ = 200.0           # Hz carrier


def amp_env(wav, input_sr, output_sr=HAPTIC_SR):
    """Audio -> vibration: 10 ms RMS envelope on a ~200 Hz carrier whose frequency rises with level."""
    wav = np.asarray(wav, dtype=np.float64).squeeze()
    duration = len(wav) / input_sr
    per_bin = int(WANTED_BIN_SIZE_SEC * input_sr)
    n_bins = len(wav) // per_bin
    rms_bins = np.array([np.sqrt(np.mean(c ** 2)) for c in np.array_split(wav, n_bins)])
    rms_norm = math.sqrt(2)
    amp = rms_norm * max(1.0, min(1.2, 1.0 / (rms_bins.max() * rms_norm))) if rms_bins.max() > 0 else rms_norm
    n_out = int(duration * output_sr)
    bin_fi = np.arange(n_out) / output_sr / duration * n_bins
    lo = bin_fi.astype(int)
    hi = np.minimum(n_bins - 1, np.ceil(bin_fi).astype(int))
    fr = bin_fi - lo
    rms = (rms_bins[lo] * (1 - fr) + rms_bins[hi] * fr) * amp
    phase = np.mod(np.cumsum(2 * np.pi * (BASE_FREQ + (rms - 0.3) * 100.0) / output_sr), 2 * np.pi)
    return (rms * np.sin(phase)).astype(np.float32)


def amp_env_reference(wav_norm, input_sample_rate, output_sample_rate):
    """Original per-sample implementation from audiogen-beam-serve.py (for testing only)."""
    wav_norm = np.asarray(wav_norm).squeeze()
    num_samples = len(wav_norm)
    duration_sec = num_samples / input_sample_rate
    samples_per_bin = int(WANTED_BIN_SIZE_SEC * input_sample_rate)
    num_bins = num_samples // samples_per_bin
    rms_bins = np.array([np.sqrt(np.mean(chunk ** 2)) for chunk in np.array_split(wav_norm, num_bins)])
    rms_max = np.max(rms_bins)
    rms_norm = np.sqrt(2)
    rms_norm_amp = rms_norm * max(1.0, min(1.2, 1.0 / (rms_max * rms_norm)))
    out_samples = int(duration_sec * output_sample_rate)
    phase_acc, output = 0.0, np.zeros(out_samples)
    for i in range(out_samples):
        bin_fi = (i / output_sample_rate) / duration_sec * num_bins
        bin_lo = int(bin_fi)
        bin_hi = min(num_bins - 1, int(math.ceil(bin_fi)))
        bin_fr = bin_fi - bin_lo
        rms = (rms_bins[bin_lo] * (1.0 - bin_fr) + rms_bins[bin_hi] * bin_fr) * rms_norm_amp
        phase_acc = (phase_acc + 2.0 * math.pi * (BASE_FREQ + (rms - 0.3) * 100.0) / output_sample_rate) % (2.0 * math.pi)
        output[i] = rms * math.sin(phase_acc)
    return output


class AudioGenHaptic:
    """AudioGen-medium + amp_env, exposing set_generation_params / generate / sample_rate like AudioGen."""

    name = "audiogen"

    def __init__(self, device=None):
        from audiocraft.models import AudioGen
        self.audio = AudioGen.get_pretrained("facebook/audiogen-medium",
                                             device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sample_rate = HAPTIC_SR

    def set_generation_params(self, **kw):
        self.audio.set_generation_params(**kw)

    def generate(self, descriptions, progress=False):
        wavs = self.audio.generate(descriptions, progress=progress).detach().cpu().numpy()
        out = [amp_env(w, self.audio.sample_rate) for w in wavs]
        n = max(len(o) for o in out)
        return torch.from_numpy(np.stack([np.pad(o, (0, n - len(o))) for o in out]))[:, None, :]


class RemoteGenerator:
    """A generator running on another machine (scripts/colab_worker.py behind a tunnel), with the same
    set_generation_params / generate / sample_rate interface. url = "https://host?token=..."."""

    def __init__(self, url, model="hapticgen"):
        import requests
        import urllib.parse
        u = urllib.parse.urlsplit(url.strip())
        self.base = f"{u.scheme}://{u.netloc}{u.path.rstrip('/')}"
        self.headers = {"X-Token": urllib.parse.parse_qs(u.query).get("token", [""])[0]}
        self.model, self.name, self._http = model, f"{model}@colab", requests
        self.max_batch = 24   # a 40 GB A100 holds far larger batches than a 12 GB laptop GPU; fewer tunnel round trips
        r = requests.get(f"{self.base}/info", params={"model_name": model}, headers=self.headers, timeout=900)
        r.raise_for_status()
        info = r.json()
        self.sample_rate, self.device_name = info["sample_rate"], info["device"]
        self.params = {}

    def set_generation_params(self, duration, use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, cfg_coef=3.0, **_):
        self.params = {"duration": duration, "top_k": top_k, "temperature": temperature, "cfg": cfg_coef}

    def generate(self, descriptions, progress=False):
        import base64
        r = self._http.post(f"{self.base}/generate", headers=self.headers, timeout=900,
                            json={"model": self.model, "prompts": list(descriptions), **self.params})
        r.raise_for_status()
        j = r.json()
        arr = np.frombuffer(base64.b64decode(j["data"]), dtype=np.float32).reshape(j["shape"])
        return torch.from_numpy(arr.copy())


def load(name):
    if name == "hapticgen":
        m = hg.load_model()
        m.name = "hapticgen"
        return m
    if name == "audiogen":
        return AudioGenHaptic()
    raise ValueError(f"unknown generator {name!r}; choose one of {GENERATORS}")
