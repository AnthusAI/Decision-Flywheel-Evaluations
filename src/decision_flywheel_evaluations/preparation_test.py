import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from .datasets import AG_NEWS, EMOTION, DatasetRow, DatasetSpec
from .manifests import ExposureStatus
from .preparation import (AG_NEWS_PLAN, EMOTION_PLAN, HISTORICAL_AG_NEWS_COMMIT, HistoricalExposure, PreparationInfeasible,
                          PreparationPlan, prepare_pinned_study)


def _rows(spec, train_per_label, official_counts):
    train = tuple(DatasetRow(f"train-{offset}", "train", offset, label, f"{label} train {index}")
                  for label_index, label in enumerate(spec.labels)
                  for index in range(train_per_label)
                  for offset in (label_index * 100_000 + index,))
    official = tuple(DatasetRow(f"test-{offset}", "test", offset, label, f"{label} test {index}")
                     for label_index, (label, count) in enumerate(zip(spec.labels, official_counts))
                     for index in range(count)
                     for offset in (label_index * 100_000 + index,))
    return train, official


def test_ag_news_preparation_excludes_historical_indices_and_legacy_hashes_before_balanced_selection():
    train, official = _rows(AG_NEWS, 612, (501, 501, 501, 501))
    history_rows = [row for row in official if row.source_index % 100_000 == 0]
    legacy = frozenset(hashlib.sha256(" ".join(row.text.strip().casefold().split()).encode()).hexdigest()
                       for row in history_rows)
    history = HistoricalExposure(frozenset(row.source_index for row in history_rows), legacy, "a" * 64,
                                 HISTORICAL_AG_NEWS_COMMIT)

    manifest = prepare_pinned_study(AG_NEWS, train, official, AG_NEWS_PLAN, history=history)

    assert manifest.exposure_status is ExposureStatus.CONFIRMATORY_FRESH
    assert manifest.counts == {"candidate": 2048, "development": 400, "scoreboard": 2000}
    assert not {row.source_index for row in history_rows} & {row.source_index for row in manifest.scoreboard}
    assert manifest.preparation.sample_counts["official_history_excluded"] == 4
    assert manifest.preparation.exposure_history_fingerprint == "a" * 64


def test_emotion_preparation_keeps_the_full_natural_official_distribution_and_is_exploratory():
    train, official = _rows(EMOTION, 356, (10, 9, 8, 7, 6, 2))

    manifest = prepare_pinned_study(EMOTION, train, official, EMOTION_PLAN)

    assert manifest.exposure_status is ExposureStatus.EXPLORATORY
    assert manifest.counts["scoreboard"] == 42
    assert sum(row.label == "surprise" for row in manifest.scoreboard) == 2
    assert manifest.preparation.configuration["natural_official_scoreboard"] is True


def test_preparation_flags_an_infeasible_bounded_train_selection_without_silently_shrinking_it():
    train, official = _rows(EMOTION, 355, (2, 2, 2, 2, 2, 2))

    with pytest.raises(PreparationInfeasible, match="needs 356"):
        prepare_pinned_study(EMOTION, train, official, EMOTION_PLAN)


def test_ambiguous_normalized_duplicates_are_excluded_as_a_group_not_assigned_an_arbitrary_label():
    train, official = _rows(EMOTION, 357, (2, 2, 2, 2, 2, 2))
    conflicting = DatasetRow("train-conflict", "train", 999_999, "joy", "sadness train 0")
    reduced = train + (conflicting,)

    manifest = prepare_pinned_study(
        EMOTION, reduced, official,
        EMOTION_PLAN,
    )

    assert {item.id for item in manifest.exclusions if item.reason == "ambiguous_normalized_text"} == {
        "train-0", "train-conflict"}


def test_an_equivalent_ag_news_spec_cannot_bypass_the_historical_exposure_requirement():
    train, official = _rows(AG_NEWS, 612, (501, 501, 501, 501))
    copied = DatasetSpec(AG_NEWS.name, AG_NEWS.revision, AG_NEWS.labels, AG_NEWS.config)

    with pytest.raises(ValueError, match="historical exposure"):
        prepare_pinned_study(copied, train, official, AG_NEWS_PLAN)


def test_pinned_preparation_rejects_a_non_test_official_source_split():
    train, official = _rows(EMOTION, 356, (2, 2, 2, 2, 2, 2))
    not_test = tuple(replace(row, source_split="validation", id=row.id.replace("test-", "validation-"))
                     for row in official)

    with pytest.raises(ValueError, match="invalid split"):
        prepare_pinned_study(EMOTION, train, not_test, EMOTION_PLAN)


def test_pinned_preparation_rejects_a_modified_declared_plan():
    train, official = _rows(EMOTION, 356, (2, 2, 2, 2, 2, 2))

    with pytest.raises(ValueError, match="declared plan"):
        prepare_pinned_study(EMOTION, train, official, replace(EMOTION_PLAN, seed=1))


def test_ag_news_historical_provenance_requires_the_reviewed_source_commit():
    train, official = _rows(AG_NEWS, 612, (501, 501, 501, 501))
    history = HistoricalExposure(frozenset(), frozenset(), "a" * 64, "0" * 40)

    with pytest.raises(ValueError, match="reviewed source commit"):
        prepare_pinned_study(AG_NEWS, train, official, AG_NEWS_PLAN, history=history)


def test_ag_news_excludes_nfkc_equivalent_rows_of_historically_exposed_source_indices():
    train, official = _rows(AG_NEWS, 612, (502, 502, 502, 502))
    historical = official[0]
    equivalent = replace(official[1], text="World test Cafe\u0301")
    original = replace(historical, text="World test Café")
    official = (original, equivalent, *official[2:])
    history = HistoricalExposure(
        frozenset({historical.source_index}),
        frozenset({hashlib.sha256(" ".join(original.text.strip().casefold().split()).encode()).hexdigest()}),
        "a" * 64, HISTORICAL_AG_NEWS_COMMIT,
    )

    manifest = prepare_pinned_study(AG_NEWS, train, official, AG_NEWS_PLAN, history=history)

    assert all(record.normalized_text_sha256 != hashlib.sha256(
        "world test café".encode()).hexdigest() for record in manifest.scoreboard)


def test_a_blocked_ag_news_historical_inventory_returns_a_nonzero_prepare_exit_code(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_datasets.py"
    completed = subprocess.run([sys.executable, str(script), "prepare", "--dataset", "ag_news",
                                "--historical-repository", str(tmp_path / "missing")],
                               capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    assert '"status": "blocked"' in completed.stdout


def test_preparation_metadata_rejects_arbitrary_text_fields():
    train, official = _rows(EMOTION, 356, (2, 2, 2, 2, 2, 2))
    manifest = prepare_pinned_study(EMOTION, train, official, EMOTION_PLAN)
    assert manifest.preparation is not None
    invalid = replace(manifest, preparation=replace(manifest.preparation,
                                                     configuration={"private_text": "must not serialize"}))

    with pytest.raises(ValueError, match="preparation metadata"):
        invalid.validate()


def test_emotion_preparation_metadata_cannot_claim_an_ag_news_historical_inventory():
    train, official = _rows(EMOTION, 356, (2, 2, 2, 2, 2, 2))
    manifest = prepare_pinned_study(EMOTION, train, official, EMOTION_PLAN)
    assert manifest.preparation is not None
    invalid = replace(manifest, preparation=replace(manifest.preparation, configuration={
        **manifest.preparation.configuration, "historical_source_commit": HISTORICAL_AG_NEWS_COMMIT},
        exposure_history_fingerprint="a" * 64))

    with pytest.raises(ValueError, match="preparation metadata"):
        invalid.validate()
