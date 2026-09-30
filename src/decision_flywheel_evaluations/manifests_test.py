import json

import pytest

from .datasets import AG_NEWS, EMOTION, DatasetRow
from .manifests import (ExposureStatus, ManifestRecord, prepare_official_split, prepare_split,
                        read_manifest, select_official_scoreboard, write_manifest)


def _rows(per_label: int = 4):
    return tuple(
        DatasetRow(f"train-{label}-{index}", "train", offset * per_label + index, label, f"{label} text {index}")
        for offset, label in enumerate(EMOTION.labels)
        for index in range(per_label)
    )


def test_a_prepared_split_is_disjoint_by_id_and_normalized_text_hash():
    manifest = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                             exposure_status=ExposureStatus.EXPLORATORY)
    manifest.validate()
    assert manifest.counts == {"candidate": 12, "development": 6, "scoreboard": 6}


def test_a_prepared_split_is_deterministic_and_uses_the_requested_class_targets():
    first = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                          exposure_status=ExposureStatus.EXPLORATORY)
    second = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                           exposure_status=ExposureStatus.EXPLORATORY)
    assert first == second
    assert [row.label for row in first.scoreboard] == list(EMOTION.labels)


def test_a_short_label_group_fails_instead_of_silently_undersampling():
    with pytest.raises(ValueError, match="sadness.*needs 3.*has 2"):
        prepare_split(_rows(per_label=2), EMOTION, seed=1, development_per_label=1, scoreboard_per_label=2,
                      exposure_status=ExposureStatus.EXPLORATORY)


def test_exact_duplicate_text_is_excluded_deterministically_with_text_free_provenance():
    rows = list(_rows())
    rows[0] = DatasetRow("train-sadness-0", "train", 0, "sadness", "same")
    rows[1] = DatasetRow("train-sadness-1", "train", 1, "sadness", "same")
    manifest = prepare_split(rows, EMOTION, seed=0, development_per_label=1, scoreboard_per_label=1,
                             exposure_status=ExposureStatus.EXPLORATORY)
    assert len(manifest.exclusions) == 1
    assert manifest.exclusions[0].id == "train-sadness-1"
    assert manifest.exclusions[0].reason == "duplicate_normalized_text"
    assert "same" not in str(manifest.exclusions[0])


def test_input_order_does_not_change_a_seeded_split():
    first = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                          exposure_status=ExposureStatus.EXPLORATORY)
    second = prepare_split(tuple(reversed(_rows())), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                           exposure_status=ExposureStatus.EXPLORATORY)
    assert first == second


def test_an_exposed_id_or_hash_cannot_be_called_confirmatory_fresh():
    old = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                        exposure_status=ExposureStatus.EXPLORATORY)
    with pytest.raises(ValueError, match="prior exposed"):
        prepare_split(
            _rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
            exposure_status=ExposureStatus.CONFIRMATORY_FRESH, prior_exposed=old.scoreboard,
        )


def test_manifest_serialization_is_text_free_and_round_trips(tmp_path):
    manifest = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                             exposure_status=ExposureStatus.EXPLORATORY)
    destination = tmp_path / "split.json"
    write_manifest(destination, manifest)
    serialized = destination.read_text()
    assert "sadness text" not in serialized
    assert read_manifest(destination) == manifest
    assert set(json.loads(serialized)) == {"dataset", "revision", "seed", "counts", "exposure_status", "records", "exclusions"}


def test_an_official_scoreboard_selection_keeps_its_source_split_and_exact_class_target():
    rows = tuple(DatasetRow(f"test-{label}-{index}", "test", offset * 3 + index, label, f"{label} {index}")
                 for offset, label in enumerate(EMOTION.labels) for index in range(3))
    selected = select_official_scoreboard(rows, EMOTION, source_split="test", seed=7, per_label=2)
    assert len(selected) == 12
    assert {row.source_split for row in selected} == {"test"}
    assert [row.label for row in selected] == [label for label in EMOTION.labels for _ in range(2)]


def test_a_historical_emotion_official_split_cannot_be_designated_confirmatory_fresh():
    rows = tuple(DatasetRow(f"test-{label}-{index}", "test", offset * 3 + index, label, f"{label} {index}")
                 for offset, label in enumerate(EMOTION.labels) for index in range(3))
    with pytest.raises(ValueError, match="historically exposed"):
        prepare_split(rows, EMOTION, seed=1, development_per_label=1, scoreboard_per_label=1,
                      exposure_status=ExposureStatus.CONFIRMATORY_FRESH)


def test_prior_exposure_in_the_reused_candidate_role_does_not_block_a_fresh_scoreboard():
    old = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                        exposure_status=ExposureStatus.EXPLORATORY)
    fresh = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                          exposure_status=ExposureStatus.CONFIRMATORY_FRESH, prior_exposed=old.candidate)
    assert fresh.exposure_status is ExposureStatus.CONFIRMATORY_FRESH


def test_an_official_split_keeps_test_as_scoreboard_and_train_as_development_and_candidate(tmp_path):
    train = _rows(per_label=4)
    test = tuple(DatasetRow(f"test-{label}-{index}", "test", offset * 3 + index, label, f"official {label} {index}")
                 for offset, label in enumerate(EMOTION.labels) for index in range(3))
    manifest = prepare_official_split(train, test, EMOTION, seed=8, development_per_label=1, scoreboard_per_label=2,
                                      exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    destination = tmp_path / "official.json"
    write_manifest(destination, manifest)
    restored = read_manifest(destination)
    assert {record.source_split for record in restored.scoreboard} == {"test"}
    assert {record.source_split for record in restored.development + restored.candidate} == {"train"}
    assert len(restored.scoreboard) == 12


def test_an_official_split_excludes_train_text_matching_the_scoreboard_without_touching_other_test_rows():
    train = list(_rows(per_label=4))
    train[0] = DatasetRow("train-sadness-0", "train", 0, "sadness", "official sadness 0")
    test = tuple(DatasetRow(f"test-{label}-{index}", "test", offset * 3 + index, label, f"official {label} {index}")
                 for offset, label in enumerate(EMOTION.labels) for index in range(3))
    manifest = prepare_official_split(train, test, EMOTION, seed=0, development_per_label=1, scoreboard_per_label=3,
                                      exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    assert "train-sadness-0" not in {record.id for record in manifest.records}
    assert any(item.id == "train-sadness-0" and item.reason == "overlaps_scoreboard_text" for item in manifest.exclusions)
    assert len(manifest.scoreboard) == 18


def test_official_split_is_independent_of_source_order_and_fresh_checks_only_the_selected_scoreboard():
    train, test = _rows(per_label=4), tuple(
        DatasetRow(f"test-{label}-{index}", "test", offset * 3 + index, label, f"official {label} {index}")
        for offset, label in enumerate(EMOTION.labels) for index in range(3)
    )
    historical = prepare_official_split(train, test, EMOTION, seed=8, development_per_label=1, scoreboard_per_label=2,
                                        exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    assert historical == prepare_official_split(tuple(reversed(train)), tuple(reversed(test)), EMOTION, seed=8,
                                                 development_per_label=1, scoreboard_per_label=2,
                                                 exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    with pytest.raises(ValueError, match="historically exposed"):
        prepare_official_split(train, test, EMOTION, seed=8, development_per_label=1, scoreboard_per_label=2,
                               exposure_status=ExposureStatus.CONFIRMATORY_FRESH, prior_exposed=historical.scoreboard)


def test_a_fresh_official_scoreboard_rejects_the_exact_prior_scoreboard_ids_or_hashes():
    train = tuple(DatasetRow(f"train-{label}-{index}", "train", offset * 2 + index, label, f"train {label} {index}")
                  for offset, label in enumerate(AG_NEWS.labels) for index in range(2))
    test = tuple(DatasetRow(f"test-{label}-{index}", "test", offset * 2 + index, label, f"test {label} {index}")
                 for offset, label in enumerate(AG_NEWS.labels) for index in range(2))
    prior = prepare_official_split(train, test, AG_NEWS, seed=2, development_per_label=1, scoreboard_per_label=1,
                                   exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    with pytest.raises(ValueError, match="prior exposed"):
        prepare_official_split(train, test, AG_NEWS, seed=2, development_per_label=1, scoreboard_per_label=1,
                               exposure_status=ExposureStatus.CONFIRMATORY_FRESH, prior_exposed=prior.scoreboard)


def test_a_manifest_load_rejects_boolean_counts_and_an_exclusion_without_a_matching_reference(tmp_path):
    manifest = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                             exposure_status=ExposureStatus.EXPLORATORY)
    destination = tmp_path / "bad.json"
    write_manifest(destination, manifest)
    payload = json.loads(destination.read_text())
    payload["counts"]["candidate"] = True
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="counts"):
        read_manifest(destination)
    payload["counts"]["candidate"] = manifest.counts["candidate"]
    payload["exclusions"] = [{"id": "excluded", "source_split": "train", "source_index": 0, "label": "sadness",
                              "normalized_text_sha256": "0" * 64, "duplicate_of_id": "absent",
                              "reason": "duplicate_normalized_text"}]
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reference"):
        read_manifest(destination)


def test_official_split_rejects_nontrain_training_rows_and_negative_development_target():
    test = tuple(DatasetRow(f"test-{label}-{index}", "test", offset * 2 + index, label, f"official {label} {index}")
                 for offset, label in enumerate(EMOTION.labels) for index in range(2))
    with pytest.raises(ValueError, match="train rows"):
        prepare_official_split(test, test, EMOTION, seed=1, development_per_label=1, scoreboard_per_label=1,
                               exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_official_split(_rows(), test, EMOTION, seed=1, development_per_label=-1, scoreboard_per_label=1,
                               exposure_status=ExposureStatus.HISTORICAL_EXPOSED)


def test_manifest_load_rejects_malformed_hash_and_duplicate_hash_inside_one_role(tmp_path):
    manifest = prepare_split(_rows(), EMOTION, seed=8, development_per_label=1, scoreboard_per_label=1,
                             exposure_status=ExposureStatus.EXPLORATORY)
    destination = tmp_path / "bad-hash.json"
    write_manifest(destination, manifest)
    payload = json.loads(destination.read_text())
    payload["records"][0]["normalized_text_sha256"] = None
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_manifest(destination)
    payload["records"][0]["normalized_text_sha256"] = payload["records"][1]["normalized_text_sha256"]
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="normalized text hashes"):
        read_manifest(destination)
