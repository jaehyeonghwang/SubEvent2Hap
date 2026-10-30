"""Event-T2M-style planner: segment the event sentence with the event-aware prompt of
Event-T2M (Hong et al., "Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis",
ICLR 2026, arXiv:2602.04292, Appendix Table 8), then schedule the segments with this project's
timing stage.

Event-T2M defines an event as the smallest semantically self-contained action or state change that
can be aligned in time with a motion segment; simultaneous actions form ONE event. Its segmentation
yields an ordered list of event sentences only (no timing or strength), and each segment is used
verbatim as the generator condition. So here:
  1. segmentation: Table 8 rules and examples, verbatim except that the POS-tag suffixes
     ("#a/DET man/NOUN ...#0.0#0.0") and the rule about them are dropped (our input is plain text),
     and the output is a JSON list instead of lines;
  2. timing: decompose.TIMING_PROMPT (same stage as the staged planner, so only the segmentation
     differs between the two planners);
  3. captions: none; every segment sentence is the HapticGen prompt as is.
"""
import time

from pydantic import Field, create_model

import decompose as dc

PROMPT_VERSION = "event_t2m_v1"

SEGMENT_PROMPT = """Please segment a single input sentence into multiple sentences that each represent a distinct event, following the rules below:
- A bundle of actions performed simultaneously at a specific point in time is defined as a single "event".
- Each segmented sentence must start with the subject used in the original sentence (e.g., "a person", "a man", etc.).
- Do not remove or simplify any adverbs, adjectives, or modifiers that appear in the original sentence - preserve them as much as possible.
- Do not add any new sentences - only break down the given text as instructed.
- Do not include your thinking process or output reasoning. Only output the segmented sentences following the format.
- If the sentence cannot be further segmented into multiple events, leave it as is and output the original sentence without any changes.
- Even if there are grammatical errors in the sentence, proceed with the processing.
- If the input sentence contains multiple actions, the output must contain the same number of actions as the input sentence.

Good Example 1 - Input
a man lifts something on his left and places it down on his right.
Good Example 1 - Output
a man lifts something on his left.
a man places it down on his right.

Good Example 2 - Input
a man kicks something with his left leg.
Good Example 2 - Output
a man kicks something with his left leg.

Good Example 3 - Input
A person waves their hand while stepping sideways, then jumps up and spins, and finally lands and bows.
Good Example 3 - Output
A person waves their hand while stepping sideways.
A person jumps up and spins.
A person lands and bows.

Bad Example 1 - Input
someone is sprinting side to side
Bad Example 1 - Output
The input sentence cannot be segmented as requested because "side to side" is a single adverbial phrase modifying the verb "sprinting," and "side" is used as a noun and a prepositional phrase. The sentence only contains one action (sprinting).

Bad Example 2 - Input
A person claps their hands while sitting on the ground.
Bad Example 2 - Output
A person claps their hands.
A person sits on the ground.

Bad Example 3 - Input
A person bends their knees and raises both arms at the same time.
Bad Example 3 - Output
A person bends their knees.
A person raises both arms.

Return the segmented sentences, in order, as the JSON list "events"."""


class Segments(dc.Output):
    events: list[str] = Field(min_length=1, max_length=8)


def decompose(event: str, model=dc.DEFAULT_MODEL, base_url=dc.LMSTUDIO_URL, temperature=0.3):
    t0, trace = time.time(), []
    seg = dc._stage("segment", SEGMENT_PROMPT, {"input_sentence": event}, Segments, model, base_url, temperature, trace)
    events = [e.strip() for e in seg.events if e.strip()]
    timing_type = create_model("EventTimings", __base__=dc.Output, **{
        f"event_{i}": (dc.TimingValues, ...) for i, _ in enumerate(events, 1)})
    scheduled = dc._stage("timing", dc.TIMING_PROMPT, {
        "original_event": event, "events": events,
        "event_keys": {f"event_{i}": e for i, e in enumerate(events, 1)}}, timing_type, model, base_url, temperature, trace)
    plan = dc.EventPlan(phases=events, sub_events=[dc.SubEvent(
        source=e, prompt=e, **getattr(scheduled, f"event_{i}").model_dump()) for i, e in enumerate(events, 1)])
    return {**plan.model_dump(), "elapsed": time.time() - t0, "model": model,
            "prompt_version": PROMPT_VERSION, "stages": trace, "review_changes": []}
