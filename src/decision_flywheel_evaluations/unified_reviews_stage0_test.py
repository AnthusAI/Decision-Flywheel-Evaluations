"""Specs for the cache-first, frozen-manifest Stage 0 runner."""
from __future__ import annotations

from copy import deepcopy

import pytest

from . import unified_reviews_manifest

from .unified_reviews_stage0 import (
    Stage0Caps,
    FutureStageCaps,
    build_stage0_plan,
    cached_stage0,
    preflight_stage0,
    run_live_stage0,
    upsert_screen_cell,
    read_screen_cache,
)
from .unified_sme import agreement_subset


def _sha(char="a"):
    return char * 64


@pytest.fixture(autouse=True)
def _minimal_fixture_is_not_a_real_freeze(monkeypatch):
    monkeypatch.setattr(unified_reviews_manifest, "frozen_role_ids", lambda manifest, role: tuple(
        row["id"] for row in manifest["universe"]["records"] if row["role"] == role))


def _manifest():
    records = []
    for number in range(1500):
        role = "heldout" if number < 300 else "reserve"
        records.append({"id": f"review-{number}", "normalized_text_sha256": _sha(),
                        "status": "completed", "raw_label": "keep" if number % 2 else "drop",
                        "gold_label": "keep" if number % 2 else "drop", "role": role})
    return {"manifest_sha256": _sha("b"), "universe": {"records": records},
            "split": {"labels": ["drop", "keep"]}}


def _identity():
    return {"dataset_manifest_sha256": _sha("c"), "policy_sha256": _sha("d"),
            "split_sha256": _sha("b"), "sme_model": "sme-v1", "jev_model": "jev-v1"}


def _complete_cache(manifest):
    heldout = [row for row in manifest["universe"]["records"] if row["role"] == "heldout"]
    screen = [{"item_id": row["id"], "condition": condition, "predicted_label": row["gold_label"],
               "status": "completed"}
              for condition in ("S", "F") for row in heldout]
    agreement = [{"item_id": row["id"], "primary_label": row["gold_label"],
                  "second_label": row["gold_label"], "status": "completed"}
                 for row in (next(entry for entry in manifest["universe"]["records"] if entry["id"] == item_id)
                             for item_id in agreement_subset([entry["id"] for entry in manifest["universe"]["records"]], 100, seed=1))]
    return screen, agreement


def _caps():
    return Stage0Caps(jev_ceiling=600, jev_max_new=600, sme_ceiling=100, sme_max_new=0)


def _future_caps():
    return FutureStageCaps(stage1_ceiling=600, stage1_max_new=600, stage2_ceiling=7000, stage2_max_new=7000)


def test_a_frozen_stage_zero_plan_has_the_exact_600_cells_and_first_universe_agreement_set():
    plan = build_stage0_plan(_manifest(), _identity())

    assert len(plan.screen_ids) == 300
    assert len(plan.agreement_ids) == 100
    assert len(plan.cells) == 600
    assert plan.cells[0] == ("S", "review-0")
    assert plan.cells[300] == ("F", "review-0")
    assert plan.agreement_ids == tuple(agreement_subset([f"review-{number}" for number in range(1500)], 100, seed=1))


def test_a_cached_run_retains_every_missing_screen_cell_and_never_constructs_a_live_factory():
    manifest = _manifest()
    screen, agreement = _complete_cache(manifest)
    screen.pop()
    called = []

    outcome = cached_stage0(manifest=manifest, identity=_identity(), screen_cache=screen,
                            agreement_cache=agreement, caps=_caps(), merge_attempt=0,
                            future_caps=_future_caps(),
                            live_factory=lambda *_: called.append(True))

    assert called == []
    assert outcome["artifact"]["screen"]["S_completed"] == 300
    assert outcome["artifact"]["screen"]["F_completed"] == 299
    assert outcome["artifact"]["decision"] == {"status": "incomplete", "reason": "screen_incomplete"}
    assert outcome["counts"] == {"screen_cells": 600, "cached_screen_cells": 599,
                                 "missing_screen_cells": 1, "agreement_expected": 100,
                                 "cached_agreement": 100, "missing_agreement": 0}


def test_preflight_and_incomplete_agreement_refuse_before_any_live_client_or_environment_factory():
    manifest = _manifest()
    screen, agreement = _complete_cache(manifest)
    agreement.pop()
    called = []

    plan = preflight_stage0(manifest=manifest, identity=_identity(), screen_cache=screen,
                            agreement_cache=agreement, caps=_caps())
    assert len(plan.missing_agreement_ids) == 1
    with pytest.raises(ValueError, match="complete cached agreement"):
        run_live_stage0(manifest=manifest, identity=_identity(), screen_cache=screen,
                        agreement_cache=agreement, caps=_caps(), merge_attempt=0, confirm_live=True,
                        future_caps=_future_caps(),
                        live_factory=lambda *_: called.append(True))
    assert called == []


def test_live_misses_need_an_explicit_confirmation_after_durable_caps_are_validated():
    manifest = _manifest()
    screen, agreement = _complete_cache(manifest)
    screen.pop()
    called = []

    with pytest.raises(ValueError, match="explicit --confirm-live"):
        run_live_stage0(manifest=manifest, identity=_identity(), screen_cache=screen,
                        agreement_cache=agreement, caps=_caps(), merge_attempt=0, confirm_live=False,
                        future_caps=_future_caps(),
                        live_factory=lambda *_: called.append(True))
    assert called == []


def test_split_identity_must_bind_the_frozen_manifest_digest_before_cache_or_client_access():
    manifest = _manifest()
    identity = _identity()
    identity["split_sha256"] = _sha("e")
    with pytest.raises(ValueError, match="split_sha256"):
        build_stage0_plan(manifest, identity)


def test_a_stale_native_question_fingerprint_stops_before_any_collector_call(tmp_path):
    manifest, identity = _manifest(), _identity()
    plan = build_stage0_plan(manifest, identity)
    screen, agreement = _complete_cache(manifest)
    # Persist one old-question cell; the rest remain misses, so a regression would call the spy.
    path = tmp_path / "screen.json"
    upsert_screen_cell(path, plan, model=identity["jev_model"], question_sha256={"S": _sha("1"), "F": _sha("2")},
                       item_id="review-0", condition="S", predicted_label="drop", status="completed",
                       text_sha256=plan.text_sha256["review-0"])
    supplied = [{"item_id": "review-0", "condition": "S", "predicted_label": "drop", "status": "completed"}]
    calls = []
    with pytest.raises(ValueError, match="question fingerprint"):
        run_live_stage0(manifest=manifest, identity=identity, screen_cache=supplied, agreement_cache=agreement,
                        caps=_caps(), future_caps=_future_caps(), merge_attempt=0, confirm_live=True,
                        live_factory=lambda _ledger: (lambda *_: calls.append(True), {"S": _sha("3"), "F": _sha("4")} ),
                        jev_ledger_path=tmp_path / "ledger.json", screen_cache_path=path,
                        model=identity["jev_model"])
    assert calls == []


def test_a_supplied_cache_list_that_differs_from_durable_cache_stops_before_any_collector_call(tmp_path):
    manifest, identity = _manifest(), _identity()
    plan = build_stage0_plan(manifest, identity)
    _screen, agreement = _complete_cache(manifest)
    path = tmp_path / "screen.json"
    questions = {"S": _sha("1"), "F": _sha("2")}
    upsert_screen_cell(path, plan, model=identity["jev_model"], question_sha256=questions,
                       item_id="review-0", condition="S", predicted_label="drop", status="completed",
                       text_sha256=plan.text_sha256["review-0"])
    calls = []
    with pytest.raises(ValueError, match="differs"):
        run_live_stage0(manifest=manifest, identity=identity, screen_cache=[], agreement_cache=agreement,
                        caps=_caps(), future_caps=_future_caps(), merge_attempt=0, confirm_live=True,
                        live_factory=lambda _ledger: (lambda *_: calls.append(True), questions),
                        jev_ledger_path=tmp_path / "ledger.json", screen_cache_path=path,
                        model=identity["jev_model"])
    assert calls == []


def test_all_cached_cells_with_a_changed_native_s_task_fingerprint_reject_before_honoring_cache(tmp_path):
    manifest, identity = _manifest(), _identity()
    plan = build_stage0_plan(manifest, identity)
    screen, agreement = _complete_cache(manifest)
    path = tmp_path / "screen.json"
    old = {"S": _sha("1"), "F": _sha("2")}
    for row in screen:
        upsert_screen_cell(path, plan, model=identity["jev_model"], question_sha256=old,
                           item_id=row["item_id"], condition=row["condition"],
                           predicted_label=row["predicted_label"], status=row["status"],
                           text_sha256=plan.text_sha256[row["item_id"]])
    supplied = read_screen_cache(path, plan, model=identity["jev_model"], question_sha256=old)
    calls = []
    with pytest.raises(ValueError, match="question fingerprint"):
        run_live_stage0(manifest=manifest, identity=identity, screen_cache=supplied, agreement_cache=agreement,
                        caps=_caps(), future_caps=_future_caps(), merge_attempt=0, confirm_live=True,
                        live_factory=lambda _ledger: (lambda *_: calls.append(True), {"S": _sha("3"), "F": _sha("2")} ),
                        jev_ledger_path=tmp_path / "ledger.json", screen_cache_path=path,
                        model=identity["jev_model"])
    assert calls == []


def test_resume_preserves_first_cell_model_usage_latency_and_aggregates_both_cells(tmp_path):
    manifest, identity = _manifest(), _identity()
    plan = build_stage0_plan(manifest, identity)
    path, questions = tmp_path / "screen.json", {"S": _sha("1"), "F": _sha("2")}
    first = ("S", "review-0")
    second = ("F", "review-1")
    upsert_screen_cell(path, plan, model=identity["jev_model"], question_sha256=questions,
                       item_id=first[1], condition=first[0], predicted_label="drop", status="completed",
                       text_sha256=plan.text_sha256[first[1]], result_model="jev-response-a",
                       usage={"input_tokens": 7, "output_tokens": 3}, latency_seconds=.25)
    upsert_screen_cell(path, plan, model=identity["jev_model"], question_sha256=questions,
                       item_id=second[1], condition=second[0], predicted_label="keep", status="completed",
                       text_sha256=plan.text_sha256[second[1]], result_model="jev-response-b",
                       usage={"input_tokens": 11, "output_tokens": 5}, latency_seconds=.5)
    cached = read_screen_cache(path, plan, model=identity["jev_model"], question_sha256=questions)
    by_cell = {(row["condition"], row["item_id"]): row for row in cached}
    assert by_cell[first]["result_model"] == "jev-response-a"
    assert by_cell[first]["usage"] == {"input_tokens": 7, "output_tokens": 3}
    report = cached_stage0(manifest=manifest, identity=identity, screen_cache=cached,
                           agreement_cache=_complete_cache(manifest)[1], caps=_caps(), future_caps=_future_caps(),
                           merge_attempt=0)
    assert report["usage"] == {"input_tokens": 18, "output_tokens": 8, "coverage": 2}


def test_a_failed_live_cell_has_no_stale_response_metadata(tmp_path):
    manifest, identity = _manifest(), _identity()
    plan = build_stage0_plan(manifest, identity)
    _screen, agreement = _complete_cache(manifest)
    path, questions = tmp_path / "screen.json", {"S": _sha("1"), "F": _sha("2")}

    from .unified_spend import CircuitOpen
    calls = 0
    with pytest.raises(CircuitOpen):
        def stop_after_failure(condition, item_id):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provider unavailable")
            raise CircuitOpen("synthetic test stop")
        run_live_stage0(manifest=manifest, identity=identity, screen_cache=[], agreement_cache=agreement,
                        caps=_caps(),
                        future_caps=_future_caps(), merge_attempt=0, confirm_live=True,
                        live_factory=lambda _ledger: (stop_after_failure, questions),
                        jev_ledger_path=tmp_path / "ledger.json", screen_cache_path=path,
                        model=identity["jev_model"])

    cached = read_screen_cache(path, plan, model=identity["jev_model"], question_sha256=questions)
    assert cached == [{"item_id": "review-0", "condition": "S", "predicted_label": None,
                       "status": "failed", "result_model": None, "usage": None,
                       "latency_seconds": None}]
