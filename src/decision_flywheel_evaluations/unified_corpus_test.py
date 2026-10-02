import dataclasses

import pytest

from .unified_corpus import (CORPORA, CORPUS_CHOICES, DEFAULT_CORPUS, EMOTION_CORPUS, PLANTED, UnknownCorpus,
                             get_corpus)
from .unified_splits import load_splits


def test_the_planted_corpus_holds_exactly_the_constants_the_harness_used_before():
    assert PLANTED.name == "planted"
    assert PLANTED.labels == ("positive", "negative")
    assert PLANTED.score_name == "Sentiment"
    assert PLANTED.instructions == "What is the overall sentiment of this text?"
    assert PLANTED.fewshot_key == "fewshot"
    assert PLANTED.fewshot_wire == "sentiment.fewshot"
    assert PLANTED.fewshot_feature == "fewshot.clr.positive"
    assert PLANTED.knn_share_feature == "knn.share.positive"
    assert PLANTED.knn_features == ("knn.share.positive", "knn.top4.positive", "knn.top4.negative")
    assert PLANTED.fake_jev_positive_label == "positive"
    assert PLANTED.load_splits is load_splits


def test_the_planted_task_is_the_sentiment_decision_task():
    task = PLANTED.task()
    assert (task.name, tuple(task.labels), task.instructions) == (
        "Sentiment", ("positive", "negative"), "What is the overall sentiment of this text?")


def test_a_corpus_is_frozen():
    with pytest.raises(Exception):
        PLANTED.labels = ("a", "b")


def test_planted_emotion_and_fomc_are_registered_and_planted_is_the_default():
    assert CORPUS_CHOICES == ("planted", "emotion", "fomc") and DEFAULT_CORPUS == "planted"
    assert get_corpus("planted") is PLANTED and set(CORPORA) == {"planted", "emotion", "fomc"}


def test_the_rubric_fields_leave_planted_and_emotion_as_they_were():
    for corpus in (PLANTED, EMOTION_CORPUS):
        assert (corpus.final_size, corpus.fill_seed_answers, corpus.ceiling_score, corpus.stakeholder_guideline) == (
            600, False, None, None)


def test_the_fomc_corpus_starts_from_the_one_line_rubric_with_three_labels_in_fixed_order():
    fomc = get_corpus("fomc")
    assert fomc.labels == ("dovish", "hawkish", "neutral") and fomc.multiclass_metrics
    assert fomc.instructions == fomc.seed_score()["instructions"] != fomc.ceiling_score()["instructions"]
    assert fomc.stakeholder_guideline() == fomc.ceiling_score()["instructions"]
    assert fomc.fill_seed_answers and fomc.final_size == 377


def test_an_unknown_corpus_is_rejected_with_the_choices_named():
    with pytest.raises(UnknownCorpus, match="unknown corpus 'nonesuch'.*planted"):
        get_corpus("nonesuch")


SIX = ("anger", "fear", "joy", "love", "sadness", "surprise")


def six_label_corpus():
    """A synthetic 6-label corpus: no network, no Hugging Face cache."""
    return dataclasses.replace(
        PLANTED, name="six", labels=SIX, score_name="Feeling", instructions="Which feeling?",
        fewshot_wire="feeling.fewshot", knn_share_feature="knn.share.anger",
        fake_jev_positive_label=None, fake_jev_cues=(("anger", ("furious",)), ("fear", ("terrified",))))


def test_the_few_shot_head_sees_one_clr_feature_per_label_except_the_last_which_is_the_dropped_reference():
    corpus = six_label_corpus()
    assert corpus.fewshot_dropped_label == "surprise"
    assert corpus.fewshot_features == (
        "fewshot.clr.anger", "fewshot.clr.fear", "fewshot.clr.joy", "fewshot.clr.love", "fewshot.clr.sadness")
    assert corpus.fewshot_feature == "fewshot.clr.anger"


def test_the_planted_few_shot_features_are_the_single_positive_one_with_negative_dropped():
    assert PLANTED.fewshot_dropped_label == "negative"
    assert PLANTED.fewshot_features == ("fewshot.clr.positive",)
    assert PLANTED.knn_share_features == ("knn.share.positive",)


def test_six_labels_give_five_knn_shares_and_a_top4_feature_for_every_label():
    corpus = six_label_corpus()
    assert corpus.knn_share_features == tuple(f"knn.share.{label}" for label in SIX[:-1])
    assert corpus.knn_top_features == tuple(f"knn.top4.{label}" for label in SIX)
    assert corpus.knn_features == (*corpus.knn_share_features, *corpus.knn_top_features)
    assert len(corpus.knn_features) == 5 + 6


def test_a_knn_share_name_that_disagrees_with_the_first_label_is_rejected():
    with pytest.raises(ValueError, match="knn_share_feature"):
        dataclasses.replace(PLANTED, knn_share_feature="knn.share.negative")


def test_the_emotion_corpus_defines_its_few_shot_knn_and_fake_cue_names_for_six_labels():
    assert EMOTION_CORPUS.fewshot_wire == "emotion.fewshot"
    labels = EMOTION_CORPUS.labels
    assert set(labels) == set(SIX) and labels[-1] == "surprise" == EMOTION_CORPUS.fewshot_dropped_label
    assert EMOTION_CORPUS.fewshot_features == tuple(f"fewshot.clr.{label}" for label in labels[:-1])
    assert EMOTION_CORPUS.knn_share_features == tuple(f"knn.share.{label}" for label in labels[:-1])
    assert len(EMOTION_CORPUS.knn_features) == 11
    cues = dict(EMOTION_CORPUS.fake_jev_cues)
    assert tuple(cues) == labels and all(cues[label] for label in labels)
    assert len({word for words in cues.values() for word in words}) == sum(len(w) for w in cues.values())
    assert PLANTED.fake_jev_cues == ()
