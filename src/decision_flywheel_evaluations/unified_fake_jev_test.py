"""Specs for the offline fake Jev answering an N-option choice question from a per-label cue lexicon."""
import pytest

from .unified_fake_jev import FakeJevCore

SIX = ("anger", "fear", "joy", "love", "sadness", "surprise")
CUES = {"anger": ("furious", "livid"), "fear": ("terrified",), "joy": ("delighted",), "love": ("adore",),
        "sadness": ("gloomy",), "surprise": ("astonished",)}
QUESTION = {"type": "choice", "criteria": list(SIX), "instructions": "Which emotion?"}


def answer(core, text, examples=()):
    state = {"labeled_examples": list(examples), "target": {"text": text}} if examples else {"text": text}
    return core.answer(state, {"emotion.fewshot": QUESTION}).answers["emotion.fewshot"]


def test_a_six_option_question_is_answered_with_the_cued_class_and_probabilities_over_all_six():
    core = FakeJevCore(cues=CUES, strength=0.3)
    for label, words in CUES.items():
        reply = answer(core, f"today I feel {words[0]} about it")
        assert reply["choice"] == label
        assert set(reply["probabilities"]) == set(SIX)
        assert sum(reply["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        assert reply["confidence"] == reply["probabilities"][label]


def test_the_answer_is_a_pure_function_of_text_question_and_examples():
    one, two = FakeJevCore(cues=CUES), FakeJevCore(cues=CUES)
    assert answer(one, "I am terrified of this") == answer(two, "I am terrified of this")
    assert answer(one, "no cue here at all") == answer(two, "no cue here at all")


def test_the_label_with_most_cue_matches_wins_and_ties_go_to_the_earlier_label():
    core = FakeJevCore(cues=CUES, strength=0.3)
    assert answer(core, "livid and furious but also gloomy")["choice"] == "anger"
    assert answer(core, "gloomy and terrified")["choice"] == "fear"


def test_a_stronger_planted_signal_gives_the_cued_class_more_probability():
    weak = answer(FakeJevCore(cues=CUES, strength=0.1), "so furious")
    strong = answer(FakeJevCore(cues=CUES, strength=0.5), "so furious")
    assert strong["probabilities"]["anger"] > weak["probabilities"]["anger"]


def test_a_few_shot_answer_moves_toward_the_label_of_the_most_similar_examples():
    core = FakeJevCore(cues=CUES, strength=0.1)
    text = "the long walk home felt quiet and heavy"
    cold = answer(core, text)
    examples = [{"text": "the long walk home felt quiet", "label": "sadness"},
                {"text": "a long walk home", "label": "sadness"},
                {"text": "something else entirely", "label": "joy"}]
    warm = answer(core, text, examples)
    assert warm["probabilities"]["sadness"] > cold["probabilities"]["sadness"]
    other = answer(core, text, [{**e, "label": "joy" if e["label"] == "sadness" else "sadness"} for e in examples])
    assert other["probabilities"]["joy"] > warm["probabilities"]["joy"]
    assert warm != other


def test_without_cues_the_binary_planted_behaviour_is_unchanged():
    from .unified_fake_jev import text_key
    core = FakeJevCore(planted={text_key("a text"): "positive"}, strength=0.3)
    reply = core.answer({"text": "a text"}, {"s.q": {"type": "choice", "criteria": ["positive", "negative"]}})
    assert reply.answers["s.q"]["choice"] == "positive"
    assert set(reply.answers["s.q"]["probabilities"]) == {"positive", "negative"}
