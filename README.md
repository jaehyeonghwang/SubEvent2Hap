# SubEvent2Hap Studio

Text event → haptic sub-events (local LM) → per-sub-event generation (HapticGen / AudioGen) → stitched vibration → Meta Quest 3 controllers.

- `scripts/decompose.py` — default 4-stage planner (events → review → timing → captions), local Qwen3-4B via LM Studio; GPT-4o optional reference.
- `scripts/compose.py` — onset-aligned stitching, loudness leveling (intensity^γ); sub-events ≤0.6 s are aligned as single impacts.
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

## Authoring study logs

Open the editor with a participant and task id to turn on study mode, e.g.
`http://127.0.0.1:7860/editor?p=P01&task=T1`. A bar shows the elapsed time and a **과제 완료** button;
every action is logged with timestamps to `out/study/<participant>/<task>_<session>.jsonl`. These include decompose, create, move, resize,
intensity and description edits, undo and redo, compose with generated and reused counts, regenerate, and play.
Each participant/task starts with an empty editor; reloading the page resumes the same session.
Logs are participant data and are git-ignored.

```powershell
.\.venv\Scripts\python scripts\analyze_study_logs.py   # per-session summary -> out/study/summary.csv
```

## Generate on one PC, play on another (Tailscale)

The GPU PC runs the server; a laptop with the Quest opens the editor in a browser and runs a small
playback agent. Both machines are on the same Tailscale network (account kuaxislab@gmail.com).

**GPU PC** (`desktop-nst9a4u`):

```powershell
start_server.bat          # = webui\server.py --host 0.0.0.0; only 127.0.0.1 and Tailscale (100.64.0.0/10) clients are served, others get 403
# once, in an admin PowerShell: allow Tailscale peers through Windows Firewall
netsh advfirewall firewall add rule name="SubEvent2Hap 7860 (Tailscale only)" dir=in action=allow protocol=TCP localport=7860 remoteip=100.64.0.0/10
```

**Laptop** (Tailscale logged in to the same account, Meta Quest Link installed, Python 3.10+):

```powershell
git clone https://github.com/jaehyeonghwang/SubEvent2Hap.git   # submodules are not needed here
cd SubEvent2Hap
start_player.bat          # first run creates .venv-player with requirements-player.txt
```

Then open `http://desktop-nst9a4u:7860/editor` in the laptop browser, start Quest Link, and put on the headset.
While the agent is connected the Quest pill shows the laptop's name. Every play/stop goes to the laptop's
Quest, which fetches the wav from the server. Without an agent, playback falls back to a Quest on the GPU PC.

**Vibration on a speaker instead of the Quest** (e.g. narration on AirPods, vibration on the laptop speaker or an
actuator amplifier): on the laptop run `start_player_audio.bat` (lists sound outputs, then asks which one gets the
vibration). The browser plays the Story2Hap narration on the Windows default output.
