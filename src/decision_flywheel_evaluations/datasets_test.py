import pytest

from .datasets import AG_NEWS, EMOTION, DatasetRow, DatasetSpec, dataset_spec, normalized_text_hash, rows_from_records


def test_the_supported_datasets_use_the_recorded_immutable_revisions():
    assert dataset_spec("ag_news") == AG_NEWS
    assert AG_NEWS.revision == "eb185aade064a813bc0b7f42de02595523103ca4"
    assert EMOTION.revision == "cab853a1dbdf4c42c2b3ef2173804746df8825fe"


def test_a_dataset_revision_must_be_a_full_immutable_sha():
    with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
        dataset_spec("ag_news", revision="main")


def test_an_injected_loader_assigns_stable_source_index_ids_and_canonical_labels():
    rows = rows_from_records(
        EMOTION,
        "train",
        [{"text": "One", "label": 1}, {"text": "Two", "label": "sadness"}],
    )
    assert rows == (
        DatasetRow("train-0", "train", 0, "joy", "One"),
        DatasetRow("train-1", "train", 1, "sadness", "Two"),
    )


def test_an_injected_loader_rejects_an_unknown_label_before_it_can_be_partitioned():
    with pytest.raises(ValueError, match="not a canonical label"):
        rows_from_records(AG_NEWS, "test", [{"text": "x", "label": "Politics"}])


@pytest.mark.parametrize("label", [-1, 6])
def test_an_injected_loader_rejects_negative_and_out_of_range_label_indexes(label):
    with pytest.raises(ValueError, match="not a canonical label"):
        rows_from_records(EMOTION, "test", [{"text": "x", "label": label}])


def test_a_normalized_hash_does_not_preserve_raw_text_and_collapses_whitespace_case():
    assert normalized_text_hash("  SAME\n text ") == normalized_text_hash("same text")


def test_a_normalized_hash_uses_core_nfkc_casefold_whitespace_rules():
    assert normalized_text_hash("ＦＯＯ\u00a0\nBar") == normalized_text_hash("foo bar")


@pytest.mark.parametrize("name, labels", [("", ("yes",)), ("demo", ()), ("demo", ("yes", "yes"))])
def test_a_dataset_spec_rejects_empty_or_duplicate_identity_fields(name, labels):
    with pytest.raises(ValueError):
        DatasetSpec(name, "a" * 40, labels)
