"""Compose a vibration from timed sub-events: generate each with HapticGen, extract the
relevant segment, level it to its intensity and mix on a timeline.

Also generates a whole-event baseline (single prompt) for comparison.

Levelling is loudness-like, not peak-based: perceived vibration strength follows energy
(an impact's attack, a texture's RMS) rather than the single highest sample, and
perceived magnitude grows sub-linearly with amplitude, so intensities are mapped through
amplitude = intensity ** gamma.
"""
import json
import pathlib
import re
import time

import numpy as np
import soundfile as sf
import torch

import hapticgen_generate as hg
import signal_controls as controls

IMPACT_MAX_DUR = 0.6     # sub-events up to this length are treated as impacts (onset-aligned)
FADE_IN = 0.003          # s, avoid clicks on hard onsets
FADE_OUT = 0.02          # s
ATTACK_WIN = 0.04        # s, window after the onset that defines an impact's strength
MAX_BATCH = 6            # prompts per HapticGen call (VRAM)
DEFAULT_GAMMA = 1.6      # amplitude = intensity ** gamma (Stevens-like compression of vibration magnitude)


def _gen_duration(sub):
    """How long to generate for a sub-event (model works better with some room)."""
    return float(min(10.0, max(1.5, sub["duration"] + 1.0)))


def _generate_batched(model, prompts, duration, cfg, temperature, top_k):
    model.set_generation_params(duration=duration, use_sampling=True, top_k=top_k, top_p=0.0,
                                temperature=temperature, cfg_coef=cfg)
    out = []
    for i in range(0, len(prompts), MAX_BATCH):
        out += _generate_chunk(model, prompts[i:i + MAX_BATCH])
    return out


def _generate_chunk(model, prompts):
    """Generate one batch; on CUDA OOM, free memory and retry in halves (long 10 s batches can OOM)."""
    try:
        wavs = model.generate(prompts).detach().cpu()
        return [w.squeeze(0).numpy().astype(np.float32) for w in wavs]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        mid = len(prompts) // 2
        return _generate_chunk(model, prompts[:mid]) + _generate_chunk(model, prompts[mid:])
    finally:
        torch.cuda.empty_cache()


def _env(w, sr, win=0.01):
    n = max(1, int(win * sr))
    return np.sqrt(np.convolve(w ** 2, np.ones(n) / n, mode="same"))


def _onsets(env, th):
    """Indices where the envelope rises through th."""
    return np.flatnonzero((env[1:] >= th) & (env[:-1] < th)) + 1


def _is_impact(sub):
    return sub["duration"] <= IMPACT_MAX_DUR


def _extract(w, sr, sub):
    """Pick the segment of a candidate matching the sub-event.

    Returns (segment, score, level, offset_s). `level` is the loudness measure used for
    levelling: attack RMS for impacts, RMS for textures.
    """
    length = int(sub["duration"] * sr)
    env = _env(w, sr)
    if env.max() < 1e-5:
        return w[:length], 0.0, 0.0, 0.0
    if _is_impact(sub):
        # align to the first strong onset; keep a single hit (cut before the next onset)
        ons = _onsets(env, 0.3 * env.max())
        onset = int(ons[0]) if len(ons) else int(np.argmax(env))
        start = max(0, onset - int(0.01 * sr))
        seg = w[start:start + length].copy()
        e = env[start:start + length]
        rel_onset = onset - start
        head = e[rel_onset:rel_onset + int(0.05 * sr)]
        peak_at = rel_onset + (int(np.argmax(head)) if len(head) else 0)
        later = _onsets(e[peak_at:], 0.5 * e[peak_at])
        # a re-attack after the hit decayed below 30% = a second bounce inside the window
        decayed = np.flatnonzero(e[peak_at:] < 0.3 * e[peak_at])
        extra = [o for o in later if len(decayed) and o > decayed[0]]
        n_hits = 1 + len(extra)
        if extra:
            cut = peak_at + extra[0] - int(0.005 * sr)
            seg[cut:] = 0.0
            seg[:cut] = _fade(seg[:cut], sr, 0, min(0.02, cut / sr / 3))
        a0 = rel_onset
        level = float(np.sqrt(np.mean(seg[a0:a0 + int(ATTACK_WIN * sr)] ** 2)))
        # clean single transient preferred: strong attack, quick decay, no extra hits
        tail = seg[a0 + int(ATTACK_WIN * sr):]
        tail_rms = float(np.sqrt(np.mean(tail ** 2))) if len(tail) else 0.0
        score = level * (1.0 + min(3.0, level / (tail_rms + 1e-4)) / 3.0) / n_hits
        return seg, score, level, start / sr
    # texture: most energetic window
    if len(w) <= length:
        rms = float(np.sqrt(np.mean(w ** 2)))
        return w, rms, rms, 0.0
    c = np.concatenate([[0.0], np.cumsum(w.astype(np.float64) ** 2)])
    energy = c[length:] - c[:-length]
    start = int(np.argmax(energy))
    rms = float(np.sqrt(energy[start] / length))
    return w[start:start + length], rms, rms, start / sr


def _fade(seg, sr, fade_in, fade_out):
    seg = seg.copy()
    fi, fo = min(len(seg), int(fade_in * sr)), min(len(seg), int(fade_out * sr))
    if fi > 0:
        seg[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi)) ** 2
    if fo > 0:
        seg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
    return seg


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:36] or "event"


def compose(model, event: str, subs: list, candidates=2, baseline=True, cfg=3.0, temperature=1.0, top_k=250,
            seed=None, out_dir=hg.DEFAULT_OUT, gamma=DEFAULT_GAMMA, share_repeats=True):
    """share_repeats: sub-events with identical prompts reuse one generated sample, so repeats
    (bounces, knocks) keep the same character and differ only in timing and intensity.

    A sub-event may carry `source_wav` (path under out_dir to the raw sample it was cut from in an
    earlier run). That sample is reused instead of generating, so editing timing or intensity, or
    regenerating only some sub-events, leaves the others unchanged. Each run writes the raw sample
    of every newly generated unit and reports it back as `source_wav`."""
    t0 = time.time()
    if seed is not None:
        torch.manual_seed(seed)
    sr = model.sample_rate
    subs = sorted([dict(s) for s in subs], key=lambda s: s["start"])
    total = max(s["start"] + s["duration"] for s in subs)

    # reusable earlier samples: same sample rate and long enough for the (possibly longer) sub-event
    reuse = {}
    for s in subs:
        src = s.get("source_wav")
        if src and not s.get("synthesis") and src not in reuse:
            p = pathlib.Path(out_dir) / src
            w, wsr = sf.read(p, dtype="float32") if p.exists() else (None, None)
            reuse[src] = w if w is not None and wsr == sr and len(w) >= int(s["duration"] * sr) else None
    for s in subs:
        if s.get("source_wav") and (reuse.get(s["source_wav"]) is None or len(reuse[s["source_wav"]]) < int(s["duration"] * sr)):
            s.pop("source_wav")   # missing, other generator or too short: generate anew

    # generation units: one per unique prompt (shared) or one per sub-event, split by sample source
    keys = [((s["prompt"].strip().lower() + json.dumps(s.get("synthesis"), sort_keys=True))
             if share_repeats else str(i)) + "|" + (s.get("source_wav") or "new") for i, s in enumerate(subs)]
    units = list(dict.fromkeys(keys))
    unit_sub = {k: subs[keys.index(k)] for k in units}
    generated_units = [k for k in units if not unit_sub[k].get("synthesis") and not unit_sub[k].get("source_wav")]
    prompts = [unit_sub[k]["prompt"] for k in generated_units for _ in range(candidates)]
    if baseline:
        prompts += [event] * candidates
    gen_subs = [s for s, k in zip(subs, keys) if k in generated_units]
    gen_dur = max([_gen_duration(s) for s in gen_subs] + ([min(10.0, total)] if baseline else []), default=0.0)
    wavs = _generate_batched(model, prompts, gen_dur, cfg, temperature, top_k) if prompts else []
    t_gen = time.time() - t0

    run_id = time.strftime("%m%d_%H%M%S")
    name = f"compose_{_slug(event)}_{run_id}"
    run_dir = pathlib.Path(out_dir) / "compose" / name
    k = 1
    while run_dir.exists():   # two runs of the same event within one second must not overwrite each other
        k += 1
        name = f"compose_{_slug(event)}_{run_id}_{k}"
        run_dir = pathlib.Path(out_dir) / "compose" / name
    run_dir.mkdir(parents=True)

    # pick the best candidate per unit (using the longest/representative sub-event of that unit)
    chosen, source = {}, {}
    for k in units:
        if unit_sub[k].get("synthesis"):
            w = controls.carrier(unit_sub[k]["synthesis"], int(max(gen_dur, _gen_duration(unit_sub[k])) * sr), sr)
            chosen[k] = (w, 0, [0.0])
        elif unit_sub[k].get("source_wav"):
            src = unit_sub[k]["source_wav"]
            chosen[k], source[k] = (reuse[src], None, []), src
    for u, k in enumerate(generated_units):
        rep = max((s for s, kk in zip(subs, keys) if kk == k), key=lambda s: s["duration"])
        cands = wavs[u * candidates:(u + 1) * candidates]
        scores = [_extract(w, sr, rep)[1] for w in cands]
        best = int(np.argmax(scores))
        chosen[k] = (cands[best], best, [round(x, 4) for x in scores])
        src_name = f"src_{u + 1:02d}.wav"
        sf.write(run_dir / src_name, cands[best], sr, subtype="FLOAT")
        source[k] = f"compose/{name}/{src_name}"

    mix = np.zeros(int((total + 0.1) * sr), dtype=np.float32)
    segments, levels = [], []
    for i, s in enumerate(subs):
        w, best, scores = chosen[keys[i]]
        if s.get("synthesis"):
            # Keep the specified oscillator phase; energy-window extraction would change it.
            seg = controls.carrier(s["synthesis"], int(s["duration"] * sr), sr)
            level = float(np.sqrt(np.mean(seg ** 2)))
            offset = 0.0
        else:
            seg, _, level, offset = _extract(w, sr, s)
        target = s["intensity"] ** gamma
        seg = seg * (target / level) if level > 1e-5 else seg * 0.0
        seg = _fade(seg, sr, FADE_IN, FADE_OUT).astype(np.float32)
        if s.get("envelope"):
            seg *= controls.envelope(s["envelope"], len(seg), sr)
        at = max(0, int(s["start"] * sr))
        end = min(len(mix), at + len(seg))
        mix[at:end] += seg[:end - at]
        segments.append((i, s, seg, best, scores, offset, keys[i]))
        levels.append(target)

    # Absolute scale: every sub-event's level is intensity ** gamma relative to one reference, the
    # strongest sub-event, whose peak is set to its own intensity ** gamma. So a given intensity
    # means the same strength whatever else is on the timeline (1.0 = full scale).
    ref = int(np.argmax(levels))
    ref_peak = float(np.abs(segments[ref][2]).max())
    gain = (levels[ref] / ref_peak) if ref_peak > 1e-6 else 1.0
    mix *= gain
    if np.abs(mix).max() > 1.0:
        mix /= np.abs(mix).max()

    seg_info = []
    for i, s, seg, best, scores, offset, key in segments:
        seg_name = f"sub_{i + 1:02d}.wav"
        sf.write(run_dir / seg_name, np.clip(seg * gain, -1, 1), sr, subtype="FLOAT")
        seg_info.append({**s, "file": f"compose/{name}/{seg_name}", "candidate": best, "scores": scores,
                         "offset": round(offset, 3), "shared": keys.count(keys[i]) > 1,
                         "source_wav": source.get(key), "reused": key not in generated_units and key in source})

    composite = pathlib.Path(out_dir) / f"{name}_composite.wav"
    sf.write(composite, mix, sr, subtype="FLOAT")
    params = {"generator": getattr(model, "name", "hapticgen"), "candidates": candidates, "cfg": cfg, "temperature": temperature, "top_k": top_k, "seed": seed,
              "gamma": gamma, "share_repeats": share_repeats}
    meta = {"prompt": f"[sub-event] {event}", "params": {**params, "sub_events": len(subs)},
            "created": time.time(), "kind": "composite", "event": event, "run": name}
    composite.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    base_file = None
    if baseline:
        bc = wavs[len(generated_units) * candidates:]
        n = int(total * sr)
        bc = [w[:n] for w in bc]
        b = bc[int(np.argmax([np.sqrt((w ** 2).mean()) for w in bc]))]
        bpeak = float(np.abs(b).max())
        if bpeak > 1e-4:
            b = b * (float(np.abs(mix).max()) / bpeak)
        b = _fade(b, sr, FADE_IN, FADE_OUT).astype(np.float32)
        base_file = pathlib.Path(out_dir) / f"{name}_baseline.wav"
        sf.write(base_file, b, sr, subtype="FLOAT")
        base_file.with_suffix(".json").write_text(json.dumps(
            {**meta, "prompt": f"[baseline] {event}", "kind": "baseline"}, ensure_ascii=False), encoding="utf-8")

    plan = {"event": event, "sub_events": seg_info, "total": total, "params": params,
            "composite": composite.name, "baseline": base_file.name if base_file else None,
            "gen_duration": gen_dur, "prompts": len(prompts),
            "generated_units": len(generated_units), "reused_units": sum(k in source for k in units) - len(generated_units),
            "analytic_segments": sum(bool(s.get("synthesis")) for s in subs),
            "elapsed_gen": t_gen, "elapsed": time.time() - t0}
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan
