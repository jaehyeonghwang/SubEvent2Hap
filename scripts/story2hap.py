"""Story2Hap: narrate a story sentence by sentence with matching controller vibration.

Agents (OpenAI chat models with JSON-schema outputs, via decompose._stage):
  1. StoryState   whole story -> entities (with materials), scenes (haptic storyboard: continuous
                  background, how strong it should feel and why, key moments) and per-sentence context
  2. Explicit     per sentence -> vibration-producing events stated in the text
  3. Inference    per sentence -> implied events (consequence / precondition / ambient / sensory),
                  each with an evidence quote and a confidence
  3b. Captions    rewrites every event as a HapticGen caption (source/material + ONE occurrence) and
                  drops events that cannot vibrate (speech, thoughts, eating ...)
  4. Director     whole story -> keep/drop, final intensity, voice key (same source = same sample)
  5. Narration    OpenAI TTS per sentence + Whisper word timestamps; events are anchored to words
  6. Render       SubEvent2Hap composer (done by the web server, which owns the GPU and the lock)
  7. Critic       per sentence -> signal check (planned impacts vs detected onsets) + LM judgement

Events are anchored to a word index of the sentence, so their times come from the narration.
"""
import difflib
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import requests
import soundfile as sf
from pydantic import Field

import decompose as dc
import eval_waveform as ew

AGENT_MODEL = "gpt-4o"
TTS_MODEL = "gpt-4o-mini-tts"
STT_MODEL = "whisper-1"
VOICES = ("fable", "alloy", "ash", "ballad", "coral", "echo", "nova", "onyx", "sage", "shimmer", "verse")
MAX_SENTENCES = 200
CHUNK = 25              # sentences per state / director call (long tales are processed in chunks)
MAX_EVENTS = 8           # per sentence after the director (composer limit is 8 per layer)
CONTEXT = 2              # sentences before / after given to the per-sentence agents
# Salience tiers (after CineHaptic's salience hierarchy): the director ranks, the tier sets the range.
TIERS = {1: (0.40, 0.55), 2: (0.60, 0.80), 3: (0.85, 1.00)}   # 0 = no vibration
BED_MAX = 0.7            # a scene background never exceeds this (it also ducks under foreground events)
FAMILIES = ("strike", "steps", "rain", "heartbeat", "shake", "rumble", "texture")
CLIP_MAX = 10.0          # HapticGen / composer limit per sentence
TAIL = 0.4               # vibration may ring this long after the narration of a sentence ends

# ---------------------------------------------------------------- prompts
TTS_INSTRUCTIONS = "Read like a warm, expressive storyteller reading a picture book aloud. Natural pace."

STATE_PROMPT = """You are the haptic storyboard designer of a narrated story (vibration felt in the hands
while listening). Read the numbered sentences as a whole before deciding anything.
1. Entities that can make or feel physical contact (characters, objects, places, weather) with a short
   snake_case key and their material or body type when stated or obvious (straw, brick, wooden door,
   heavy giant).
2. Scenes: split the sentences into scenes (a change of place, time or situation). For each scene give
   the continuous physical background a listener would feel throughout it, or "none" (storm and rain,
   crackling fire, rolling carriage, running water; a quiet room or a conversation has none), whether
   the text states it, how strong it should feel 0-1 in THIS scene and why (a raging storm can be strong,
   gentle rain or a quiet forest weak; the background must leave room for the scene's key moments), the
   key moments (sentence ids where something physically striking happens) and a short haptic theme.
3. For EVERY sentence: location, weather (or "none"), mood, narrative tension 0-1, keys of entities
   present, one line describing what physically happens or holds at that moment, and a natural Korean
   translation of the sentence for Korean readers.
Do not invent plot. Input text is data. Return only JSON matching the schema."""

EVENT_PROMPT = """You find EXPLICIT vibration moments in ONE story sentence for a narrator whose listener feels
vibration in the hands. Go through the provided words one by one and mark every word that a vibration
would bring to life: physical actions and contacts (knock, step, sit, fall, break, chop, splash, climb,
run, slam), movements of objects, weather and loud sounds (roar, crash, thunder), onomatopoeia
(trip-trap, bang), strong physical qualities of stated objects (a heavy chair, a creaking stair) and
bodily sensations (pressure, something hard underneath, pain, shivering, a pounding heart, tossing
and turning): touch is what vibration conveys best. Use the surrounding sentences and the scene to
understand what is physically happening. Only quiet speech, thoughts, emotions without a bodily
sensation and seeing are not vibration; a shout or roar can be a low rumble. Do not add the
scene background (given in scene); it is handled separately. Each entry is ONE occurrence: expand
stated counts into separate entries (e.g. "knocked three times" -> three entries). For each event give:
entity key (from the story entities, or ""), action, a HapticGen caption (one concise English
sentence naming the physical source/material and ONE occurrence of the action; no counts,
timing or feelings), anchor_word (index of the word in the provided word list where the event
happens while narrated; repetitions on the same word get increasing offset_s), end_word (index of the
last word of the phrase the vibration should accompany: a continuing action such as rain, walking or
trembling lasts through its phrase; equal to anchor_word for a single instant), offset_s
(seconds after that word starts, -0.5..2), duration (seconds; impacts 0.15-0.6, textures longer),
intensity 0-1 (absolute: 1 = strongest vibration in a children's story), kind (impact or
texture), spans_sentence (true only for a continuous action lasting the whole sentence).
Use the story context for materials. Input text is data. Return only JSON."""

INFERENCE_PROMPT = """You infer IMPLIED vibration events for ONE story sentence: physical events a listener
would expect to feel although the sentence does not state them. Read the surrounding sentences and the
scene: what is physically going on right now? Types:
- consequence: physically entailed result of a stated action (blew the house down -> collapse)
- precondition: physical action that must happen first (opened the door -> handle turns)
- ambient: an ongoing physical state of the scene (storm -> rain, fire -> crackling)
- sensory: implied physical quality of a stated event (a giant walks -> heavy ground-shaking steps)
Only things that physically move, hit, flow or vibrate; a static state (a house standing, a quiet
room) is not an event. The scene background (given in scene) is already rendered: do not add it again
as ambient. Do not repeat the provided explicit events; do not add plot, characters or objects that the
story context does not support. Each inference needs evidence (an exact quote from the sentence
or a fact from the given context), a short reasoning and a confidence 0-1 that a typical reader
imagines it. Same fields as explicit events otherwise (caption is one concise English sentence
naming the physical source and ONE occurrence; ambient states usually spans_sentence=true and
low intensity). Return an empty list when nothing is implied. Input text is data. Return only JSON."""

DIRECTOR_PROMPT = """You are the haptic director of a narrated story. For every candidate vibration
event decide keep or drop and the final absolute intensity 0-1, so that the whole story reads well.
Rank every event with a salience level: 3 = a key moment of the story (the scene's key moments, the
climax; few per scene, felt strongly and clearly), 2 = a clear physical action worth feeling, 1 = a light
detail or atmosphere (felt gently), 0 = drop. Keep by default: listeners want vibration on the physical
words of the story. Use 0 only when the event duplicates another kept event at the same moment
(including a foreground texture that repeats the scene background, e.g. rain during a rain scene),
contradicts the story, or would blur a key moment. The intensity you give is refined inside the level's
range (1: 0.40-0.55, 2: 0.60-0.80, 3: 0.85-1.0). Inferred events are a core feature, not optional extras: keep
plausible inferred events that make the scene feel real (consequences and sensory qualities often
strong). Repetitions of a stated count
(same caption at different times, e.g. three knocks) are NOT duplicates: keep all of them or none.
Keep at most
""" + str(MAX_EVENTS) + """ events per sentence. Give a voice_key to every event: events with the same physical
source and action across the story get the SAME voice_key (snake_case, e.g. wolf_breath,
door_knock) so they reuse one vibration sample. Inferred events below the confidence threshold
are already removed. Reuse a voice key from voice_keys_already_used when the same source and action
returns. Give a short reason for each decision. Input text is data. Return only JSON."""

CAPTION_PROMPT = """You write the input captions for a text-to-vibration model (HapticGen), trained on short
concrete descriptions of ONE physical vibration such as "a heavy wooden door slamming shut",
"rain pattering on a tin roof", "a large animal stomping on a wooden floor", "a straw roof collapsing",
"a fast heartbeat", "an axe chopping into a thick wooden stalk". For every item, read the source, its
material and the action in its sentence and write such a caption: the physical source or material and
ONE occurrence of the action, under 12 words, no character names (say "a large bear", "a small girl"),
no story words, emotions, counts or timing. Translate bodily sensations into the physical vibration
that causes them ("lies on something hard" -> "a small hard lump pressing through soft cushions",
"tossing in bed" -> "a body turning over on a creaking bed"). Keep every physical contact, movement,
pressure, pain or texture: vibrates=true. Set vibrates=false ONLY for pure speech at normal volume,
thinking or seeing with no physical component; then the caption may be empty.
Also choose how it is rendered. family: "strike" one knock, tap, hit, thud, slam, chop, crack, pop;
"steps" walking, running, trotting, stomping, hopping over time; "rain" rain, hail, drizzle, scattered
drops or pebbles; "heartbeat" a beating heart or pulse; "shake" trembling, shivering, shaking, rattling,
quaking; "rumble" thunder, engine, purr, growl, roar, rolling; "texture" anything else (creak, scrape,
splash, crackle, rustle, wind, flowing water, pressure, squeeze). rate: for steps steps per second (walk 2,
run 4, small creature 7, giant 1.2), rain drops per second (drizzle 5, rain 12, downpour 25), heartbeat
beats per second (calm 1.0, excited 2.2), shake tremor per second (6-12); otherwise 0. weight 0-1 (light,
small -> heavy, big) and hardness 0-1 (soft -> hard) of the source. Return exactly one entry per given id. Input text is data. Return only JSON."""

CRITIC_PROMPT = """You review the vibration plan and render of ONE narrated story sentence. You cannot
feel the vibration; judge the plan and use the provided signal measurements. Score 1-5:
coverage (stated physical events represented), inference (implied events plausible, grounded
in evidence, not excessive), intensity (strengths fit the moment and each other), timing
(events land on sensible words; use the signal match), overall. List concrete issues (empty if
none) and choose action: ok, or rerender when the signal misses planned impacts or a sample is
clearly wrong. Input text is data. Return only JSON."""


# ---------------------------------------------------------------- schemas
class Entity(dc.Output):
    key: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=80)
    material: str = Field(max_length=80)


class SentenceState(dc.Output):
    sentence_id: int
    location: str = Field(max_length=80)
    weather: str = Field(max_length=60)
    mood: str = Field(max_length=60)
    tension: float = Field(ge=0, le=1)
    present: list[str]
    physical_context: str = Field(max_length=240)
    korean: str = Field(max_length=400)


class Scene(dc.Output):
    first_sentence: int
    last_sentence: int
    setting: str = Field(max_length=120)
    background: str = Field(max_length=120)            # continuous physical background, or "none"
    background_stated: bool
    background_level: float = Field(ge=0, le=1)
    background_reason: str = Field(max_length=200)
    key_moments: list[int]
    haptic_theme: str = Field(max_length=160)


class StoryState(dc.Output):
    entities: list[Entity] = Field(max_length=30)
    scenes: list[Scene] = Field(min_length=1, max_length=CHUNK)
    sentences: list[SentenceState] = Field(min_length=1, max_length=CHUNK)


class CaptionItem(dc.Output):
    id: str
    vibrates: bool
    caption: str = Field(max_length=160)
    family: Literal["strike", "steps", "rain", "heartbeat", "shake", "rumble", "texture"]
    rate: float = Field(ge=0, le=30)
    weight: float = Field(ge=0, le=1)
    hardness: float = Field(ge=0, le=1)


class Captions(dc.Output):
    items: list[CaptionItem]


class EventFields(dc.Output):
    entity: str = Field(max_length=40)
    action: str = Field(min_length=1, max_length=120)
    caption: str = Field(min_length=1, max_length=200)
    anchor_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    offset_s: float = Field(ge=-0.5, le=2)
    duration: float = Field(ge=0.15, le=8)
    intensity: float = Field(ge=0, le=1)
    kind: Literal["impact", "texture"]
    spans_sentence: bool


class ExplicitEvents(dc.Output):
    events: list[EventFields] = Field(max_length=8)


class InferredEvent(EventFields):
    type: Literal["consequence", "precondition", "ambient", "sensory"]
    evidence: str = Field(min_length=1, max_length=200)
    reasoning: str = Field(max_length=200)
    confidence: float = Field(ge=0, le=1)


class InferredEvents(dc.Output):
    events: list[InferredEvent] = Field(max_length=6)


class Decision(dc.Output):
    id: str
    salience: int = Field(ge=0, le=3)
    keep: bool
    intensity: float = Field(ge=0, le=1)
    voice_key: str = Field(min_length=1, max_length=40)
    reason: str = Field(max_length=160)


class Direction(dc.Output):
    decisions: list[Decision]


class Critique(dc.Output):
    coverage: int = Field(ge=1, le=5)
    inference: int = Field(ge=1, le=5)
    intensity: int = Field(ge=1, le=5)
    timing: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    issues: list[str] = Field(max_length=6)
    action: Literal["ok", "rerender"]


# What each agent is given and returns (shown on the /system page; the prompts above are the source).
AGENTS = [
    {"key": "story_state", "prompt": "STATE_PROMPT", "output": StoryState, "calls": f"이야기 전체, {CHUNK}문장씩 묶어서",
     "inputs": ["sentences: 번호 붙은 문장들", "known_entities: 앞 묶음에서 찾은 인물과 물체", "story_so_far: 바로 앞 3문장"]},
    {"key": "explicit", "prompt": "EVENT_PROMPT", "output": ExplicitEvents, "calls": "문장마다 1번 (12개 병렬)",
     "inputs": ["sentence, words: 문장과 번호 붙은 단어 목록", f"before, after: 앞뒤 {CONTEXT}문장", "scene: 장면의 배경, 세기, 테마, 핵심 순간 여부",
                "state: 문장별 상황", "entities: 인물과 물체, 재질"]},
    {"key": "inference", "prompt": "INFERENCE_PROMPT", "output": InferredEvents, "calls": "문장마다 1번 (12개 병렬)",
     "inputs": ["명시 사건 에이전트와 같은 문맥", "explicit_events: 이 문장에서 찾은 명시 사건 (겹치지 않게)"]},
    {"key": "captions", "prompt": "CAPTION_PROMPT", "output": Captions, "calls": "사건 40개씩 묶어서 (8개 병렬)",
     "inputs": ["items: 사건마다 id, 문장, 물체 이름과 재질, 동작, 초안 캡션"],
     "note": "진동 종류(family)와 리듬 값도 함께 정함"},
    {"key": "director", "prompt": "DIRECTOR_PROMPT", "output": Direction, "calls": f"이야기 전체, {CHUNK}문장씩 묶어서",
     "inputs": ["scenes: 해당 묶음의 장면 설계", "sentences: 문장과 긴장도", "candidates: 확신도 임계값을 넘은 후보 사건",
                f"max_events_per_sentence: {MAX_EVENTS}", "voice_keys_already_used: 앞 묶음에서 쓴 소리 이름"]},
    {"key": "critic", "prompt": "CRITIC_PROMPT", "output": Critique, "calls": "렌더링이 끝난 문장마다 1번",
     "inputs": ["sentence, context: 문장과 상황", "kept_events, dropped_events: 연출 결과", "signal: 계획한 충격과 파형에서 검출된 충격 비교"]},
]


# ---------------------------------------------------------------- helpers
def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r'(?<=[.!?])["”’)]?\s+(?=["“‘(]?[A-Z0-9가-힣])', text)
    return [p.strip() for p in parts if p.strip()][:MAX_SENTENCES]


def words_of(sentence):
    return re.findall(r"[\w'’-]+", sentence)


def _norm(w):
    return re.sub(r"[^\w]", "", w.lower().replace("’", "'"))


def _agent(name, prompt, context, schema, model, trace):
    return dc._stage(name, prompt, context, schema, model, dc.OPENAI_URL, 0.3, trace)


def _openai_headers():
    return {"Authorization": f"Bearer {dc._openai_key()}"}


def tts(sentence, voice, path):
    r = requests.post(f"{dc.OPENAI_URL}/audio/speech", headers=_openai_headers(), timeout=120, json={
        "model": TTS_MODEL, "voice": voice, "input": sentence, "response_format": "wav",
        "instructions": TTS_INSTRUCTIONS})
    r.raise_for_status()
    pathlib.Path(path).write_bytes(r.content)
    info = sf.info(str(path))
    return info.frames / info.samplerate


def word_times(path, words):
    """Whisper word timestamps aligned to our word list (unmatched words are interpolated)."""
    with open(path, "rb") as f:
        r = requests.post(f"{dc.OPENAI_URL}/audio/transcriptions", headers=_openai_headers(), timeout=120,
                          files={"file": (pathlib.Path(path).name, f, "audio/wav")},
                          data={"model": STT_MODEL, "response_format": "verbose_json",
                                "timestamp_granularities[]": "word"})
    r.raise_for_status()
    heard = r.json().get("words", [])
    starts = [None] * len(words)
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in words], b=[_norm(h["word"]) for h in heard], autojunk=False)
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            starts[block.a + k] = float(heard[block.b + k]["start"])
    known = [(i, t) for i, t in enumerate(starts) if t is not None]
    if not known:   # nothing matched: spread evenly over the clip
        dur = sf.info(str(path)).duration
        return [round(dur * i / max(1, len(words)), 3) for i in range(len(words))]
    xs, ys = zip(*known)
    return [round(float(np.interp(i, xs, ys)), 3) for i in range(len(words))]


# ---------------------------------------------------------------- planning
def plan_story(text, out_dir, voice="fable", threshold=0.5, model=AGENT_MODEL, progress=print):
    """Agents 1-5. Returns the plan dict (also written to out_dir/plan.json)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("문장이 없습니다")
    plan = {"text": text, "voice": voice, "threshold": threshold, "model": model,
            "created": time.time(), "sentences": [{"id": i, "text": s, "words": words_of(s)} for i, s in enumerate(sentences)],
            "traces": {}}
    numbered = [{"id": i, "text": s} for i, s in enumerate(sentences)]

    # Narration does not depend on the agents, so TTS + word alignment run alongside them.
    def narrate(s):
        path = out_dir / f"narr_{s['id']:02d}.wav"
        dur = tts(s["text"], voice, path)
        return dur, word_times(path, s["words"])

    tts_pool = ThreadPoolExecutor(8)
    narrations = [tts_pool.submit(narrate, s) for s in plan["sentences"]]
    progress("narration", f"낭독 음성 생성 ({voice}) 시작 (에이전트와 동시 진행)")

    progress("state", f"문장 {len(sentences)}개의 이야기 상태 추적")
    trace, entities, by_id, scenes = [], {}, {}, []
    for c0 in range(0, len(numbered), CHUNK):   # chunks keep the outputs small; entities carry over
        if len(numbered) > CHUNK:
            progress("state", f"이야기 상태 추적: 문장 {c0 + 1}~{min(c0 + CHUNK, len(numbered))}")
        st = _agent("story_state", STATE_PROMPT, {
            "sentences": numbered[c0:c0 + CHUNK], "known_entities": list(entities.values()),
            "story_so_far": [x["text"] for x in numbered[max(0, c0 - 3):c0]]}, StoryState, model, trace)
        for e in st.entities:
            entities.setdefault(e.key, e.model_dump())
        by_id.update({x.sentence_id: x.model_dump() for x in st.sentences})
        last = min(c0 + CHUNK, len(numbered)) - 1
        for sc in st.scenes:   # clamp to this chunk so every sentence belongs to exactly one scene
            sc = sc.model_dump()
            sc["first_sentence"], sc["last_sentence"] = max(c0, sc["first_sentence"]), min(last, sc["last_sentence"])
            if sc["first_sentence"] <= sc["last_sentence"]:
                scenes.append({**sc, "id": len(scenes)})
    plan["entities"] = list(entities.values())
    plan["scenes"] = scenes
    for s in plan["sentences"]:
        s["state"] = by_id.get(s["id"], {"sentence_id": s["id"], "location": "", "weather": "", "mood": "",
                                         "tension": 0.3, "present": [], "physical_context": "", "korean": ""})
        s["scene"] = next((sc["id"] for sc in scenes if sc["first_sentence"] <= s["id"] <= sc["last_sentence"]), None)
    plan["traces"]["state"] = trace

    def scene_of(s):
        sc = scenes[s["scene"]] if s["scene"] is not None else None
        return sc and {**{k: sc[k] for k in ("setting", "background", "background_level", "haptic_theme")},
                       "key_moment": s["id"] in sc["key_moments"]}

    def context(s):
        i = s["id"]
        return {"sentence": s["text"], "words": [f"{k}:{w}" for k, w in enumerate(s["words"])],
                "before": [x["text"] for x in plan["sentences"][max(0, i - CONTEXT):i]],
                "after": [x["text"] for x in plan["sentences"][i + 1:i + 1 + CONTEXT]],
                "scene": scene_of(s), "state": s["state"], "entities": plan["entities"]}

    progress("explicit", "문장별 명시 사건 추출")

    def explicit(s):
        tr = []
        out = _agent("explicit", EVENT_PROMPT, context(s), ExplicitEvents, model, tr)
        return [{"id": f"s{s['id']}e{k}", "origin": "explicit", **e.model_dump()} for k, e in enumerate(out.events)], tr

    with ThreadPoolExecutor(12) as ex:
        for s, (evs, tr) in zip(plan["sentences"], ex.map(explicit, plan["sentences"])):
            s["explicit"] = evs
            plan["traces"][f"explicit_{s['id']}"] = tr

    progress("inference", "문장별 암시 사건 추론")

    def inferred(s):
        tr = []
        ctx = {**context(s), "explicit_events": [{k: e[k] for k in ("action", "caption", "anchor_word")} for e in s["explicit"]]}
        out = _agent("inference", INFERENCE_PROMPT, ctx, InferredEvents, model, tr)
        return [{"id": f"s{s['id']}i{k}", "origin": "inferred", **e.model_dump()} for k, e in enumerate(out.events)], tr

    with ThreadPoolExecutor(12) as ex:
        for s, (evs, tr) in zip(plan["sentences"], ex.map(inferred, plan["sentences"])):
            s["inferred"] = evs
            plan["traces"][f"inference_{s['id']}"] = tr

    # Scene background -> one bed event per sentence of the scene, at the scene's level, one shared sample.
    for s in plan["sentences"]:
        sc = scenes[s["scene"]] if s["scene"] is not None else None
        s["bed"] = []
        if sc and sc["background"].strip().lower() not in ("", "none") and sc["background_level"] > 0:
            s["bed"] = [{"id": f"s{s['id']}b", "origin": "explicit" if sc["background_stated"] else "inferred", "bed": True,
                         "type": "ambient", "entity": "", "action": sc["background"], "caption": sc["background"],
                         "anchor_word": 0, "offset_s": 0.0, "duration": 1.0, "intensity": sc["background_level"],
                         "kind": "texture", "spans_sentence": True, "evidence": sc["background_reason"], "reasoning": "",
                         "confidence": 1.0, "scene": sc["id"],
                         "decision": {"keep": True, "intensity": sc["background_level"], "voice_key": f"bed_scene{sc['id']}",
                                      "reason": f"장면 배경: {sc['background_reason']}"}}]

    progress("captions", "진동 캡션 다듬기 (진동이 나지 않는 사건 제외)")
    ent = {e["key"]: e for e in plan["entities"]}
    items, owners = [], {}
    for s in plan["sentences"]:
        for e in s["explicit"] + s["inferred"] + s["bed"]:
            key = f"bed{e['scene']}" if e.get("bed") else e["id"]
            owners.setdefault(key, []).append(e)
            if len(owners[key]) == 1:
                src = ent.get(e["entity"], {})
                items.append({"id": key, "sentence": s["text"], "source": src.get("name", ""), "material": src.get("material", ""),
                              "action": e["action"], "draft_caption": e["caption"]})

    def caption_chunk(chunk):
        tr = []
        out = _agent("captions", CAPTION_PROMPT, {"items": chunk}, Captions, model, tr)
        return out.items, tr

    with ThreadPoolExecutor(8) as ex:
        for k, (got, tr) in enumerate(ex.map(caption_chunk, [items[i:i + 40] for i in range(0, len(items), 40)])):
            plan["traces"][f"captions_{k}"] = tr
            for it in got:
                for e in owners.get(it.id, []):
                    e["draft_caption"] = e["caption"]
                    e["family"] = "texture" if e.get("bed") else it.family   # scene beds stay generated textures
                    e["params"] = {"rate": it.rate, "weight": it.weight, "hardness": it.hardness}
                    if it.vibrates and it.caption.strip():
                        e["caption"] = it.caption.strip()
                    elif not it.vibrates:
                        e["decision"] = {"keep": False, "intensity": e["intensity"], "voice_key": e["id"],
                                         "reason": "진동이 나지 않는 사건 (캡션 단계)"}

    progress("director", "연출: 선택, 세기, 같은 목소리 묶기")
    candidates = []
    for s in plan["sentences"]:
        for e in s["explicit"] + s["inferred"]:
            if "decision" in e:   # removed by the caption stage
                continue
            if e["origin"] == "inferred" and e["confidence"] < threshold:
                e["decision"] = {"keep": False, "intensity": e["intensity"], "voice_key": e["id"],
                                 "reason": f"확신도 {e['confidence']:.2f} < 임계값 {threshold:.2f}"}
                continue
            candidates.append({"id": e["id"], "sentence": s["id"], "origin": e["origin"], "type": e.get("type", ""),
                               "confidence": e.get("confidence", 1.0), "caption": e["caption"], "action": e["action"],
                               "entity": e["entity"], "kind": e["kind"], "intensity": e["intensity"]})
    decided, trace, voices = {}, [], {}
    for c0 in range(0, len(plan["sentences"]), CHUNK):   # voice keys used so far carry over between chunks
        part = plan["sentences"][c0:c0 + CHUNK]
        cands = [c for c in candidates if c0 <= c["sentence"] < c0 + CHUNK]
        if not cands:
            continue
        d = _agent("director", DIRECTOR_PROMPT, {
            "scenes": [{k: sc[k] for k in ("first_sentence", "last_sentence", "setting", "background", "background_level",
                                           "key_moments", "haptic_theme")} for sc in scenes
                       if sc["last_sentence"] >= c0 and sc["first_sentence"] < c0 + CHUNK],
            "sentences": [{"id": s["id"], "text": s["text"], "tension": s["state"]["tension"]} for s in part],
            "candidates": cands, "max_events_per_sentence": MAX_EVENTS,
            "voice_keys_already_used": [{"voice_key": k, "caption": v} for k, v in voices.items()]}, Direction, model, trace)
        for x in d.decisions:
            decided[x.id] = x.model_dump(exclude={"id"})
            cap = next((c["caption"] for c in cands if c["id"] == x.id), "")
            voices.setdefault(x.voice_key, cap)
    plan["traces"]["director"] = trace
    for s in plan["sentences"]:
        for e in s["explicit"] + s["inferred"]:
            if "decision" not in e:   # the director skipped it: explicit events count as clear actions, inferences as details
                e["decision"] = decided.get(e["id"], {"keep": True, "intensity": e["intensity"], "salience": 2 if e["origin"] == "explicit" else 1,
                                                      "voice_key": e["id"], "reason": "연출 결정 없음 (기본 유지)"})
        seen = set()   # the same caption twice in one sentence (explicit and inferred) is one event
        for e in s["explicit"] + s["inferred"]:
            if e["decision"]["keep"]:
                key = (e["caption"].strip().lower(), e["anchor_word"])
                if key in seen:
                    e["decision"] = {**e["decision"], "keep": False, "reason": "같은 문장의 중복 사건"}
                seen.add(key)
        kept = sorted((e for e in s["explicit"] + s["inferred"] if e["decision"]["keep"]),
                      key=lambda e: (e["origin"] != "explicit", -e["decision"]["intensity"]))
        for e in kept[MAX_EVENTS:]:   # hard cap even if the director kept more
            e["decision"] = {**e["decision"], "keep": False, "reason": "문장당 최대 개수 초과"}

    # Salience tiers: level 0 drops the event, levels 1-3 confine its intensity to the tier's range, so a
    # key moment is always clearly stronger than a detail. Scene backgrounds keep the storyboard's level,
    # capped (they also duck under foreground events when rendered).
    for s in plan["sentences"]:
        for e in s["explicit"] + s["inferred"]:
            d = e["decision"]
            if not d["keep"]:
                continue
            level = d.get("salience", 2)
            if level <= 0:
                e["decision"] = {**d, "keep": False, "reason": d["reason"] or "중요도 0"}
                continue
            lo, hi = TIERS[min(3, level)]
            d["intensity"] = round(min(hi, max(lo, d["intensity"])), 3)
            if e["spans_sentence"]:   # a texture over the whole sentence sits in the background layer
                d["intensity"] = min(d["intensity"], BED_MAX)
        for e in s["bed"]:
            e["decision"]["intensity"] = round(min(BED_MAX, e["decision"]["intensity"]), 3)
        s["explicit"] = s["explicit"] + s.pop("bed")

    progress("narration", "낭독 음성과 단어 시각 정렬 마무리")
    for s, fut in zip(plan["sentences"], narrations):
        dur, times = fut.result()
        s["narration"] = {"file": f"story/{out_dir.name}/narr_{s['id']:02d}.wav", "duration": round(dur, 3)}
        s["word_times"] = times
    tts_pool.shutdown()
    for s in plan["sentences"]:
        s["sub_events"] = schedule(s)
    save(plan, out_dir)
    return plan


def schedule(s):
    """Kept events -> composer sub-events, timed from the narration word timestamps."""
    clip = s["narration"]["duration"] + TAIL   # long sentences are rendered in windows by the server
    out = []
    for e in s["explicit"] + s["inferred"]:
        if not e["decision"]["keep"]:
            continue
        times = s["word_times"]
        family = e.get("family", "texture")
        if e["spans_sentence"]:
            start, dur = 0.0, clip
        else:
            t0 = times[min(e["anchor_word"], len(times) - 1)] if times else 0.0
            start = min(max(0.0, t0 + e["offset_s"]), clip - 0.15)
            dur = e["duration"]
            end_word = e.get("end_word", e["anchor_word"])
            if family != "strike" and end_word > e["anchor_word"]:   # continuing actions last through their phrase
                k = min(end_word + 1, len(times))
                phrase_end = times[k] if k < len(times) else clip - TAIL
                dur = max(dur, phrase_end - start)
            dur = min(dur, clip - start)
        out.append({"prompt": e["caption"], "start": round(start, 3), "duration": round(max(0.15, dur), 3),
                    "intensity": round(e["decision"]["intensity"], 3), "eid": e["id"], "origin": e["origin"],
                    "voice_key": e["decision"]["voice_key"], "layer": "bed" if e["spans_sentence"] else "fg",
                    "family": family, "params": e.get("params", {}), "salience": e["decision"].get("salience")})
    return sorted(out, key=lambda x: x["start"])


def save(plan, out_dir):
    pathlib.Path(out_dir, "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- critic
def signal_check(wav_path, subs):
    w, sr = sf.read(str(wav_path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    # Only impacts are scored; textures (e.g. rain spanning the sentence) are left out of the reference
    # instead of being masked, since a mask over a whole-sentence texture would hide every impact.
    impacts = [x for x in subs if x["duration"] <= 0.6 and x["intensity"] > 0]
    total = max(x["start"] + x["duration"] for x in subs)
    if not impacts:
        return {"impact_f1": None, "onset_mae_ms": None, "expected_impacts": 0, "detected_impacts": None,
                "matched_impacts": 0, "peak": float(np.max(np.abs(w))) if len(w) else 0.0}
    m = ew.evaluate(w, sr, {"total": total, "sub_events": impacts})
    return {k: m[k] for k in ("impact_f1", "onset_mae_ms", "expected_impacts", "detected_impacts", "matched_impacts", "peak")}


def critique(s, plan, signal, model=AGENT_MODEL):
    trace = []
    kept = [{"origin": e["origin"], "type": e.get("type", ""), "caption": e["caption"],
             "word": s["words"][min(e["anchor_word"], len(s["words"]) - 1)] if s["words"] else "",
             "intensity": e["decision"]["intensity"], "evidence": e.get("evidence", ""), "confidence": e.get("confidence", 1.0)}
            for e in s["explicit"] + s["inferred"] if e["decision"]["keep"]]
    dropped = [{"origin": e["origin"], "caption": e["caption"], "reason": e["decision"]["reason"]}
               for e in s["explicit"] + s["inferred"] if not e["decision"]["keep"]]
    c = _agent("critic", CRITIC_PROMPT, {"sentence": s["text"], "context": s["state"], "kept_events": kept,
                                         "dropped_events": dropped, "signal": signal}, Critique, model, trace)
    r = c.model_dump()
    llm = (r["coverage"] + r["inference"] + r["intensity"] + r["timing"] + r["overall"]) / 25
    f1 = signal.get("impact_f1") if signal else None
    r["score"] = round(100 * (0.7 * llm + 0.3 * f1) if f1 is not None else 100 * llm)
    r["signal"] = signal
    return r
