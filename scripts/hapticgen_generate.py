"""Generate vibrotactile signals from text with HapticGen (local inference).

Usage:
    python scripts/hapticgen_generate.py "a cat purring" -d 5 -n 3 -o out/
"""
import argparse
import pathlib
import sys
import types

import torch

# HapticGen's audiocraft imports xformers at module level, but inference uses the
# torch attention backend. Stub the few symbols touched so xformers isn't required
# (its wheels pin old torch versions that don't support Blackwell GPUs).
_xf = types.ModuleType("xformers")
_xf_ops = types.ModuleType("xformers.ops")
_xf_ops.unbind = torch.unbind


class _LowerTriangularMask:  # torch backend only checks `attn_mask is not None` (-> is_causal)
    pass


def _xformers_attention_unavailable(*args, **kwargs):
    raise RuntimeError("xformers attention backend is stubbed; use the torch backend")


_xf_ops.LowerTriangularMask = _LowerTriangularMask
_xf_ops.memory_efficient_attention = _xformers_attention_unavailable
_xf.ops = _xf_ops
sys.modules.setdefault("xformers", _xf)
sys.modules.setdefault("xformers.ops", _xf_ops)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from audiocraft.models import AudioGen  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "hapticgen"
DEFAULT_OUT = ROOT / "out"

# Same thresholds as audiogen-beam-serve.py
MIN_ACCEPTABLE_AMPLITUDE = 0.05
MIN_ACCEPTABLE_995TH_AMPLITUDE = 0.02


def load_model(weights=DEFAULT_WEIGHTS) -> AudioGen:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AudioGen.get_pretrained(str(weights), device=device)


def generate(model: AudioGen, prompt: str, duration=5.0, num=3, top_k=250, cfg=3.0, temperature=1.0,
             seed=None, out_dir=DEFAULT_OUT, progress=False):
    """Generate `num` candidates; returns list of dicts (path, wav, peak, rms, ok)."""
    if seed is not None:
        torch.manual_seed(seed)
    model.set_generation_params(duration=duration, use_sampling=True, top_k=top_k,
                                top_p=0.0, temperature=temperature, cfg_coef=cfg)
    wavs = model.generate([prompt] * num, progress=progress).detach().cpu()

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in prompt).strip("_")[:40] or "haptic"
    n_existing = len(list(out_dir.glob(f"{slug}_*.wav")))
    results = []
    for i, wav in enumerate(wavs):
        w = wav.squeeze(0).numpy().astype(np.float32)
        a = np.abs(w)
        ok = bool(a.max() >= MIN_ACCEPTABLE_AMPLITUDE and np.quantile(a, 0.995) >= MIN_ACCEPTABLE_995TH_AMPLITUDE)
        path = out_dir / f"{slug}_{n_existing + i}.wav"
        sf.write(path, w, model.sample_rate, subtype="FLOAT")
        results.append({"path": path, "wav": w, "sr": model.sample_rate, "peak": float(a.max()),
                        "rms": float(np.sqrt((w ** 2).mean())), "ok": ok})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-d", "--duration", type=float, default=5.0)
    ap.add_argument("-n", "--num", type=int, default=3, help="candidates generated in one batch")
    ap.add_argument("-o", "--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--top-k", type=int, default=250)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    model = load_model(args.weights)
    print(f"model sample rate: {model.sample_rate} Hz")
    for r in generate(model, args.prompt, args.duration, args.num, args.top_k, args.cfg, args.temperature,
                      args.seed, args.out_dir, progress=True):
        print(f"{r['path']}  peak={r['peak']:.3f}  rms={r['rms']:.3f}  {'ok' if r['ok'] else 'too-weak'}")


if __name__ == "__main__":
    main()
