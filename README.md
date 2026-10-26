# SubEvent2Hap Studio

Text event → haptic sub-events (local LM) → per-sub-event generation (HapticGen / AudioGen) → stitched vibration → Meta Quest 3 controllers.

- `scripts/decompose.py` — default 4-stage planner (events → review → timing → captions), local Qwen3-4B via LM Studio; GPT-4o optional reference.
- `scripts/compose.py` — onset-aligned stitching, loudness leveling (intensity^γ), cut/crossfade.
- `scripts/generators.py` — HapticGen, or AudioGen-medium + HapticGen's audio→haptic conversion.
- `scripts/quest_haptic_play.py` — OpenXR `XR_FB_haptic_pcm` playback over Link.
- `webui/` — FastAPI server + pages: `/` (generate, sub-event composer with timeline editor), `/gallery` (blind listening/voting), `/architecture` (pipeline diagram + changelog).
- `data/eval_set_v2.json` — 33-prompt dev set from the HapticGen dataset; `out/**.json` — plans, ablation scores, gallery manifests, votes (wavs are not versioned).

## Setup on a new machine (Windows, NVIDIA GPU)

```powershell
git clone --recurse-submodules https://github.com/jaehyeonghwang/SubEvent2Hap.git
cd SubEvent2Hap
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e HapticGen --no-deps
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If the repo was cloned without submodules: `git submodule update --init`.

**Weights** (not in git, ~3.7 GB):

```powershell
.\.venv\Scripts\huggingface-cli download HapticGen/HapticGen-Weights --local-dir weights/hapticgen
```

AudioGen (`facebook/audiogen-medium`) is fetched into the HuggingFace cache the first time it is used. The server sets `HF_HUB_OFFLINE=1`, so download it once while online:

```powershell
$env:HF_HUB_OFFLINE='0'; .\.venv\Scripts\python -c "import sys; sys.path.insert(0,'scripts'); import generators; generators.load('audiogen')"
```

**Local LM:** install [LM Studio](https://lmstudio.ai), download `qwen3-4b-instruct-2507`, and load it with the identifier `qwen3-4b` (server at `http://127.0.0.1:1234`, parallel slots = 1). The web UI can also load it on demand.

**GPT-4o (optional):** copy `.env.example` to `.env` and set `OPENAI_KEY`. Never commit `.env`.

**Quest 3:** Meta Quest Link / Air Link active, with Link set as the OpenXR runtime. For playback without wearing the headset: `adb shell am broadcast -a com.oculus.vrpowermanager.prox_close`.

## Run

```powershell
.\.venv\Scripts\python webui\server.py     # http://127.0.0.1:7860
```

## Reproducing results

| What | Command |
| --- | --- |
| Planner ablation (33 prompts) | `python scripts/ablation.py run` then `python scripts/ablation.py score` → `out/ablation/scores_v2.json` |
| Gallery round audio (rebuild wavs from manifests/plans) | `python scripts/build_gallery.py --round N --seed S --design {llm,2x2,2x2_local,ag_vs_hg}` |
| Blind-vote analysis | `python scripts/analyze_2x2.py --rounds 5,6` (also `vote_attribution.py`, `round_consistency.py`) |
| Tests | `cd scripts; ..\.venv\Scripts\python -m unittest discover -p "test_*.py"` |

The rendered gallery wavs are excluded from git. Rebuild them with `build_gallery.py` before using `/gallery`. Votes are stored in `out/gallery/votes.json`.
