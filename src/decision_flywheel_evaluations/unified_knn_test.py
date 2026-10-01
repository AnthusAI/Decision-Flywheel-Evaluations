import pytest
from decision_flywheel.context import PerLabelLexicalRetrieval
from decision_flywheel.models import DecisionTask, Item, LabeledItem

from .unified_knn import (
    KNN_FEATURES, PoolEntry, context_fingerprint, knn_features, ranked_neighbours,
    retrieval_policy_fingerprint, similarity)


def _pool():
    texts = {
        "a": ("great match today the team won", "positive"),
        "b": ("the team played a great match", "positive"),
        "c": ("quarterly earnings missed the target", "negative"),
        "d": ("the board missed the earnings target again", "negative"),
        "e": ("weather was fine", "positive"),
        "f": ("meeting ran long and nothing was decided", "negative"),
        "g": ("the team lost the match", "negative"),
        "h": ("great earnings for the team", "positive"),
        "i": ("fans cheered the winning team", "positive"),
        "j": ("the audit found missed targets", "negative"),
    }
    return [PoolEntry(i, t, l) for i, (t, l) in texts.items()]


def test_similarity_ranks_exactly_like_per_label_lexical_retrieval():
    task = DecisionTask("Sentiment", ("positive", "negative"), "What is the overall sentiment?")
    pool = _pool()
    target = Item("x", {"text": "the team won a great match"})
    candidates = [LabeledItem(Item(e.id, {"text": e.text}), e.label) for e in pool]
    picked = PerLabelLexicalRetrieval().select(task, target, candidates, per_label=2)
    ranked = ranked_neighbours("x", "the team won a great match", pool)
    for label in task.labels:
        ours = [e.id for _, e in ranked if e.label == label][:2]
        theirs = [c.item.id for c in picked if c.label == label]
        assert ours == theirs
    assert similarity("a b", "a b") == pytest.approx(1.0)


def test_a_labeled_target_is_never_its_own_neighbour_by_id_or_by_duplicate_text():
    pool = _pool() + [PoolEntry("dup", "Great  match today the TEAM won", "negative")]
    features, ids = knn_features("a", "great match today the team won", pool, k=8)
    assert "a" not in ids and "dup" not in ids
    assert set(features) == set(KNN_FEATURES)


def test_neighbours_are_unbalanced_so_the_label_share_varies_between_items():
    pool = _pool()
    sport, _ = knn_features("x", "the team won a great match", pool, k=3)
    money, _ = knn_features("y", "the earnings target was missed", pool, k=3)
    assert sport["knn.share.positive"] > 0.5 > money["knn.share.positive"]


def test_top4_similarity_is_searched_per_label_over_the_whole_pool():
    pool = _pool()
    features, _ = knn_features("x", "missed earnings target", pool, k=2, top=4)
    negatives = sorted((similarity("missed earnings target", e.text) for e in pool if e.label == "negative"), reverse=True)
    assert features["knn.top4.negative"] == pytest.approx(sum(negatives[:4]) / 4)


def test_too_few_labeled_neighbours_is_an_error_not_a_silent_shrink():
    with pytest.raises(ValueError):
        knn_features("x", "anything", _pool()[:3], k=8)


def test_the_context_fingerprint_changes_with_the_labeled_set_but_not_its_order():
    policy = retrieval_policy_fingerprint()
    assert context_fingerprint(policy, ["a", "b"]) == context_fingerprint(policy, ["b", "a"])
    assert context_fingerprint(policy, ["a", "b"]) != context_fingerprint(policy, ["a", "b", "c"])
