from .unified_sme import SmeRecord
from .unified_stream_feedback import CachedSmeFeedback


def _record(item, label="approve", reason="This explains the current review.", rule="R11"):
    return SmeRecord(item, label, rule, reason, None, "accepted")


def test_cached_feedback_keeps_the_current_label_while_x_only_shuffles_prior_reason_content():
    source = CachedSmeFeedback({"a": _record("a", "approve", "First reason."),
                                "b": _record("b", "promotional", "Second reason.")}, seed=2)
    source.begin_batch("X", 1)
    first = source.reveal(arm="X", item_id="a", predicted="approve", reviewed_prefix=())
    second = source.reveal(arm="X", item_id="b", predicted="approve", reviewed_prefix=("a",))
    assert first.comment == "correct."
    assert second.label == "promotional" and second.comment.startswith("should be promotional because first reason")


def test_label_only_missing_and_duplicate_reveals_never_fabricate_a_reason():
    label_only = SmeRecord("a", "approve", None, None, None, "label_only")
    source = CachedSmeFeedback({"a": label_only})
    reveal = source.reveal(arm="E", item_id="a", predicted="promotional", reviewed_prefix=())
    assert reveal.comment == "should be approve." and reveal.reason_sha256 is None
    try:
        source.reveal(arm="E", item_id="a", predicted="promotional", reviewed_prefix=())
    except RuntimeError:
        pass
    else:
        assert False


def test_feedback_history_and_rule_quotas_are_isolated_per_arm_and_label_only_is_trusted():
    rows = {"a": _record("a", reason="Alpha reason.", rule="R1"),
            "b": _record("b", "promotional", "Bravo reason.", "R2"),
            "c": SmeRecord("c", "approve", None, None, None, "label_only")}
    source = CachedSmeFeedback(rows, max_rule_repeats=1)
    source.begin_batch("E", 1)
    source.reveal(arm="E", item_id="a", predicted="approve", reviewed_prefix=())
    source.begin_batch("X", 1)
    x = source.reveal(arm="X", item_id="b", predicted="approve", reviewed_prefix=("a",))
    label_only = source.reveal(arm="E", item_id="c", predicted="promotional", reviewed_prefix=("a",))
    assert "bravo reason" not in (x.comment or "").lower(), "E history must not donate into X"
    assert label_only.label == "approve" and label_only.status == "label_only" and label_only.comment == "should be approve."


def test_rule_quota_resets_by_arm_and_batch_and_x_counts_the_donor_rule():
    rows = {"a": _record("a", reason="Alpha.", rule="R1"), "b": _record("b", "promotional", "Bravo.", "R2"),
            "c": _record("c", "promotional", "Charlie.", "R2")}
    source = CachedSmeFeedback(rows, max_rule_repeats=1)
    source.begin_batch("X", 1)
    source.reveal(arm="X", item_id="a", predicted="approve", reviewed_prefix=())
    x = source.reveal(arm="X", item_id="b", predicted="approve", reviewed_prefix=("a",))
    assert "alpha" in (x.comment or "").lower()
    source.begin_batch("E", 1)
    e = source.reveal(arm="E", item_id="c", predicted="approve", reviewed_prefix=())
    assert "charlie" in (e.comment or "").lower(), "X donor quota must not consume E quota"
