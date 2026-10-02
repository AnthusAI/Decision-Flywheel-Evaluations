import pytest

from .unified_corpus import CORPORA, CORPUS_CHOICES, DEFAULT_CORPUS, PLANTED, UnknownCorpus, get_corpus
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


def test_only_planted_is_registered_and_it_is_the_default():
    assert CORPUS_CHOICES == ("planted",) and DEFAULT_CORPUS == "planted"
    assert get_corpus("planted") is PLANTED and CORPORA == {"planted": PLANTED}


def test_an_unknown_corpus_is_rejected_with_the_choices_named():
    with pytest.raises(UnknownCorpus, match="unknown corpus 'emotion'.*planted"):
        get_corpus("emotion")
