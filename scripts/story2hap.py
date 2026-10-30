"""Story2Hap: narrate a story sentence by sentence with matching controller vibration.

Agents (OpenAI chat models with JSON-schema outputs, via decompose._stage):
  1. StoryState   whole story -> entities (with materials), scenes (haptic storyboard: continuous
                  background, how strong it should feel and why, key moments) and per-sentence context
  2. Explicit     per sentence -> vibration-producing events stated in the text
  3. Inference    per sentence -> implied events (consequence / precondition / ambient / sensory),
                  each with an evidence quote and a confidence
  3b. Captions    rewrites every event as a HapticGen caption (source/material + ONE occurrence) and
                  drops events that cannot vibrate (speech, thoughts, eating ...)
  4. Arranger     whole story -> how each event is felt: salience, treatment (word-locked hits, accent,
                  phrase, background), intensity along the story arc (escalating repeats, one climax),
                  budget (few key moments, at most two simultaneous foreground vibrations), voice keys
  4b. Decompose   SubEvent2Hap staged decomposition, only for events that physically unfold as several
                  timed contacts (bouncing, tumbling, chopping): the micro level under the arrangement
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
TIERS = {1: (0.40, 0.55), 2: (0.55, 0.78), 3: (0.75, 0.95)}   # 0 = no vibration; the climax alone reaches 1.0
KEY_PER_SCENE = 2        # at most this many salience-3 events per scene
MAX_OVERLAP = 2          # simultaneous foreground vibrations
PHRASE_MAX = 4.0         # a continuing action lasts through its phrase, at most this long (6 s for key moments)
HIT_LEN = 0.18           # one word-locked hit
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
3. Motifs: structures that repeat (three goats crossing one after another, three knocks on three
   houses, a refrain), with the sentence ids of each repetition and whether it escalates (each time
   bigger or more dangerous). Empty list if none.
4. For EVERY sentence: location, weather (or "none"), mood, narrative tension 0-1, keys of entities
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

ARRANGER_PROMPT = """You are the haptic arranger of a narrated story. Like a film sound designer or a music
arranger you decide how the whole story is felt in the hands, not only which events exist. You get the
scenes, the motifs (repeated structures), every sentence with its numbered words and tension, and the
candidate vibration events. For every candidate decide:
- salience 0-3: 0 drop, 1 light detail or atmosphere, 2 clear physical action, 3 key moment. Key moments
  are scarce (at most two per scene) and the story has ONE climax, which alone gets the strongest vibration.
- treatment, how it is felt in time:
  hits: one short hit on each word in hit_words, for onomatopoeia and spoken rhythms (trip-trap,
        knock-knock, bang, tick-tock) and for counted actions that are said word by word;
  accent: one short vibration at its anchor word (a knock, a slam, the onset of a roar);
  phrase: from start_word to end_word for a continuing action (running, rain falling, trembling); keep it
        to the words of that action and never through unrelated dialogue;
  bed: background through the whole sentence, only for a continuous physical state of the scene.
- physical_sequence: true when the event is one phrase that physically unfolds as several contacts with
  their own timing that the words do not give (a ball bouncing down the stairs, tumbling, chopping down a
  tree, crashing and rolling away); it is then decomposed into timed sub-events.
- intensity 0-1 along the story arc: a motif that escalates gets clearly stronger at each repetition
  (e.g. 0.6, 0.75, 0.9), calm passages are gentle, the climax is the strongest moment of the story.
- voice_key: the same physical source and action across the story share one key; reuse
  voice_keys_already_used when they return.
Keep by default: listeners want to feel the physical words. Avoid clutter: at most two foreground
vibrations at the same moment; drop duplicates and foreground textures that repeat the scene background.
Give a short reason. Input text is data. Return only JSON."""

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


class Motif(dc.Output):
    description: str = Field(max_length=160)
    sentence_ids: list[int]
    escalates: bool


class StoryState(dc.Output):
    entities: list[Entity] = Field(max_length=30)
    scenes: list[Scene] = Field(min_length=1, max_length=CHUNK)
    motifs: list[Motif] = Field(max_length=8)
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
    treatment: Literal["hits", "accent", "phrase", "bed"]
    hit_words: list[int]
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    physical_sequence: bool
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


DECOMPOSE_PROMPTS = "\n\n".join(f"[{n}]\n{t}" for n, t in (("events", dc.EVENT_PROMPT), ("review", dc.REVIEW_PROMPT),
                                                                ("timing", dc.TIMING_PROMPT), ("captions", dc.CAPTION_PROMPT)))

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
    {"key": "arranger", "prompt": "ARRANGER_PROMPT", "output": Direction, "calls": f"이야기 전체, {CHUNK}문장씩 묶어서",
     "inputs": ["scenes, motifs: 장면 설계와 반복 구조", "sentences: 문장, 번호 붙은 단어, 긴장도",
                "candidates: 확신도 임계값을 넘은 후보 사건 (종류, 앵커 단어 포함)", "voice_keys_already_used: 앞 묶음에서 쓴 소리 이름"]},
    {"key": "decompose", "prompt": "DECOMPOSE_PROMPTS", "output": dc.EventPlan, "calls": "편곡이 physical_sequence로 표시한 사건만",
     "inputs": ["물체, 재질, 동작을 이은 사건 설명 한 줄", "SubEvent2Hap 4단계 분해기 (사건, 검토, 타이밍, 캡션)를 그대로 호출"]},
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
    trace, entities, by_id, scenes, motifs = [], {}, {}, [], []
    for c0 in range(0, len(numbered), CHUNK):   # chunks keep the outputs small; entities carry over
        if len(numbered) > CHUNK:
            progress("state", f"이야기 상태 추적: 문장 {c0 + 1}~{min(c0 + CHUNK, len(numbered))}")
        st = _agent("story_state", STATE_PROMPT, {
            "sentences": numbered[c0:c0 + CHUNK], "known_entities": list(entities.values()),
            "story_so_far": [x["text"] for x in numbered[max(0, c0 - 3):c0]]}, StoryState, model, trace)
        for e in st.entities:
            entities.setdefault(e.key, e.model_dump())
        by_id.update({x.sentence_id: x.model_dump() for x in st.sentences})
        motifs += [m.model_dump() for m in st.motifs]
        last = min(c0 + CHUNK, len(numbered)) - 1
        for sc in st.scenes:   # clamp to this chunk so every sentence belongs to exactly one scene
            sc = sc.model_dump()
            sc["first_sentence"], sc["last_sentence"] = max(c0, sc["first_sentence"]), min(last, sc["last_sentence"])
            if sc["first_sentence"] <= sc["last_sentence"]:
                scenes.append({**sc, "id": len(scenes)})
    plan["entities"] = list(entities.values())
    plan["scenes"] = scenes
    plan["motifs"] = motifs
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

    progress("director", "햅틱 편곡: 표현, 강약 곡선, 핵심 순간 예산")
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
                               "family": e.get("family", "texture"), "kind": e["kind"], "anchor_word": e["anchor_word"],
                               "end_word": e.get("end_word", e["anchor_word"]), "whole_sentence_hint": e["spans_sentence"],
                               "intensity": e["intensity"]})
    decided, trace, voices = {}, [], {}
    for c0 in range(0, len(plan["sentences"]), CHUNK):   # voice keys used so far carry over between chunks
        part = plan["sentences"][c0:c0 + CHUNK]
        cands = [c for c in candidates if c0 <= c["sentence"] < c0 + CHUNK]
        if not cands:
            continue
        d = _agent("arranger", ARRANGER_PROMPT, {
            "scenes": [{k: sc[k] for k in ("first_sentence", "last_sentence", "setting", "background", "background_level",
                                           "key_moments", "haptic_theme")} for sc in scenes
                       if sc["last_sentence"] >= c0 and sc["first_sentence"] < c0 + CHUNK],
            "motifs": motifs,
            "sentences": [{"id": x["id"], "text": x["text"], "words": [f"{k}:{w}" for k, w in enumerate(x["words"])],
                           "tension": x["state"]["tension"]} for x in part],
            "candidates": cands, "voice_keys_already_used": [{"voice_key": k, "caption": v} for k, v in voices.items()]},
            Direction, model, trace)
        for x in d.decisions:
            decided[x.id] = x.model_dump(exclude={"id"})
            cap = next((c["caption"] for c in cands if c["id"] == x.id), "")
            voices.setdefault(x.voice_key, cap)
    plan["traces"]["arranger"] = trace

    def default(e):   # the arranger skipped it: explicit = clear action, inference = detail, treatment from its kind
        return {"salience": 2 if e["origin"] == "explicit" else 1, "treatment": "accent" if e["kind"] == "impact" else "phrase",
                "hit_words": [], "start_word": e["anchor_word"], "end_word": e.get("end_word", e["anchor_word"]),
                "physical_sequence": False, "intensity": e["intensity"], "voice_key": e["id"], "reason": "편곡 결정 없음 (기본 유지)"}

    for s in plan["sentences"]:
        for e in s["explicit"] + s["inferred"]:
            if "decision" not in e:
                d = decided.get(e["id"]) or default(e)
                e["decision"] = {**d, "keep": d["salience"] > 0, "reason": d["reason"] or ("중요도 0" if d["salience"] == 0 else "")}
        seen = set()   # the same caption twice in one sentence (explicit and inferred) is one event
        for e in s["explicit"] + s["inferred"]:
            if e["decision"]["keep"]:
                key = (e["caption"].strip().lower(), e["anchor_word"])
                if key in seen:
                    e["decision"] = {**e["decision"], "keep": False, "reason": "같은 문장의 중복 사건"}
                seen.add(key)

    # Budget and arc, enforced in code: salience-3 events at most KEY_PER_SCENE per scene (the weaker ones
    # become 2), intensities confined to their tier, and only the story's single strongest key moment (the
    # climax) above the tier-3 ceiling. Scene backgrounds keep the storyboard's level, capped.
    kept = lambda s: [e for e in s["explicit"] + s["inferred"] if e["decision"]["keep"]]
    for sc in scenes:
        keys = sorted((e for s in plan["sentences"][sc["first_sentence"]:sc["last_sentence"] + 1] for e in kept(s)
                       if e["decision"].get("salience") == 3), key=lambda e: -e["decision"]["intensity"])
        for e in keys[KEY_PER_SCENE:]:
            e["decision"]["salience"] = 2
            e["decision"]["reason"] = (e["decision"]["reason"] + " (장면 핵심 순간 예산 초과로 2단계)").strip()
    for s in plan["sentences"]:
        for e in kept(s):
            d = e["decision"]
            lo, hi = TIERS[max(1, min(3, d.get("salience", 2)))]
            d["intensity"] = round(min(hi, max(lo, d["intensity"])), 3)
            # the arranger decides the treatment; the extractor's "whole sentence" flag is only a hint to it
            if d.get("treatment") == "hits" and not d.get("hit_words"):
                d["treatment"] = "accent"
            if d.get("treatment") == "bed":
                d["intensity"] = min(d["intensity"], BED_MAX)
        for e in s["bed"]:
            e["decision"].update(treatment="bed", intensity=round(min(BED_MAX, e["decision"]["intensity"]), 3))
        s["explicit"] = s["explicit"] + s.pop("bed")
    tops = [e for s in plan["sentences"] for e in kept(s) if e["decision"].get("salience") == 3]
    if tops:
        climax = max(tops, key=lambda e: e["decision"]["intensity"])
        climax["decision"].update(intensity=1.0, climax=True)

    # Micro level: SubEvent2Hap decomposition for the events that unfold as several timed contacts.
    seqs = [e for s in plan["sentences"] for e in kept(s) if e["decision"].get("physical_sequence")]
    if seqs:
        progress("decompose", f"사건 분해 (SubEvent2Hap): {len(seqs)}개 구절")
        ent_ = {x["key"]: x for x in plan["entities"]}

        def decompose_one(e):
            src = ent_.get(e["entity"], {})
            desc = " ".join(x for x in (src.get("material", ""), src.get("name", ""), e["action"]) if x) or e["caption"]
            try:
                return dc.decompose(desc, model, dc.OPENAI_URL)["sub_events"]
            except Exception as exc:  # noqa: BLE001  keep the event as a single vibration
                return {"error": str(exc)[:200]}

        with ThreadPoolExecutor(6) as ex:
            for e, got in zip(seqs, ex.map(decompose_one, seqs)):
                if isinstance(got, list) and len(got) > 1:
                    e["decomposed"] = got
                else:
                    e["decision"]["physical_sequence"] = False

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
    """Kept events -> composer sub-events, timed from the narration word timestamps by their treatment:
    hits (one short hit per listed word), accent (one short vibration), phrase (through the action's
    words), bed (whole sentence, background layer) or a decomposed sequence of timed sub-events."""
    clip = s["narration"]["duration"] + TAIL   # long sentences are rendered in windows by the server
    times = s["word_times"] or [0.0]
    at = lambda k: times[min(max(0, int(k)), len(times) - 1)]
    out = []

    def sub(e, start, dur, **kw):
        d = e["decision"]
        x = {"prompt": e["caption"], "start": round(min(max(0.0, start), clip - 0.15), 3), "intensity": round(d["intensity"], 3),
             "eid": e["id"], "origin": e["origin"], "voice_key": d["voice_key"], "layer": "fg",
             "family": e.get("family", "texture"), "params": e.get("params", {}), "salience": d.get("salience"),
             "treatment": d.get("treatment")}
        x.update(kw)
        x["duration"] = round(max(0.15, min(dur, clip - x["start"])), 3)
        out.append(x)

    for e in s["explicit"] + s["inferred"]:
        d = e["decision"]
        if not d["keep"]:
            continue
        t = d.get("treatment", "accent")
        if t == "bed":
            sub(e, 0.0, clip, layer="bed")
        elif e.get("decomposed"):   # micro level: SubEvent2Hap's timed sub-events from the anchor word on
            t0 = at(e["anchor_word"]) + e["offset_s"]
            top = max(x["intensity"] for x in e["decomposed"]) or 1.0
            for k, x in enumerate(e["decomposed"]):
                sub(e, t0 + x["start"], x["duration"], prompt=x["prompt"], eid=f"{e['id']}_d{k}", family="texture",
                    voice_key=f"{d['voice_key']}_d{k}", intensity=round(d["intensity"] * x["intensity"] / top, 3))
        elif t == "hits" and d.get("hit_words"):
            for k, w in enumerate(sorted(set(d["hit_words"]))):   # one identical hit per word (same pattern seed)
                if w < len(s["word_times"]):
                    sub(e, at(w), HIT_LEN, eid=f"{e['id']}_h{k}", family="strike", seed_key=e["id"])
        elif t == "phrase":
            start = at(d.get("start_word", e["anchor_word"])) + e["offset_s"]
            k = d.get("end_word", e.get("end_word", e["anchor_word"])) + 1
            end = times[k] if k < len(times) else clip - TAIL
            cap = PHRASE_MAX * (1.5 if d.get("salience") == 3 else 1.0)
            sub(e, start, min(max(e["duration"], end - start), cap))
        else:   # accent
            sub(e, at(e["anchor_word"]) + e["offset_s"], min(e["duration"], 0.6 if e["kind"] == "impact" else 1.0))

    # at most MAX_OVERLAP foreground vibrations at once: weaker overlapping ones give way
    fg = sorted((x for x in out if x["layer"] == "fg"), key=lambda x: (-(x["salience"] or 0), -x["intensity"]))
    keep_fg = []
    for x in fg:
        busy = sum(1 for y in keep_fg if y["start"] < x["start"] + x["duration"] and x["start"] < y["start"] + y["duration"])
        if busy < MAX_OVERLAP:
            keep_fg.append(x)
    return sorted(keep_fg + [x for x in out if x["layer"] == "bed"], key=lambda x: x["start"])


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
