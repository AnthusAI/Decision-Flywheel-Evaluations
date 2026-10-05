"""Specs for the pure, text-free Stage 0 decision gate."""

import pytest

from . import unified_reviews_gate as gate
from .unified_reviews_gate import authorize_stage, evaluate_stage0


LABELS = ("approve", "flag")
SCREEN_IDS = tuple(f"screen-{index:03d}" for index in range(300))
AGREEMENT_IDS = tuple(f"agree-{index:03d}" for index in range(100))
IDENTITY = {"dataset_manifest_sha256": "a" * 64, "policy_sha256": "b" * 64,
            "split_sha256": "c" * 64, "sme_model": "sme-v1", "jev_model": "jev-v1"}


def _rows(prediction):
    return [{"item_id": item_id, "true_label": LABELS[index % 2], "predicted_label": prediction(index),
             "status": "completed"} for index, item_id in enumerate(SCREEN_IDS)]


def _agreement(n=100):
    return [{"item_id": item_id, "primary_label": LABELS[index % 2],
             "second_label": LABELS[index % 2] if index < n else LABELS[(index + 1) % 2], "status": "completed"}
            for index, item_id in enumerate(AGREEMENT_IDS)]


def _artifact(**changes):
    values = dict(identity=IDENTITY, labels=LABELS, screen_expected_ids=SCREEN_IDS,
                  agreement_expected_ids=AGREEMENT_IDS, agreement_rows=_agreement(),
                  s_rows=_rows(lambda _index: "approve"), f_rows=_rows(lambda index: LABELS[index % 2]),
                  stage_ceilings={"stage1": {"request_ceiling": 600, "max_new_requests": 600},
                                  "stage2": {"request_ceiling": 7000, "max_new_requests": 7000}},
                  synthetic=False, merge_attempt=0)
    values.update(changes)
    return evaluate_stage0(**values)


def test_a_complete_text_free_non_synthetic_screen_passes_at_exact_float_boundaries_and_is_deterministic():
    first, second = _artifact(), _artifact()

    assert first == second
    assert first["decision"]["status"] == "passed"
    assert first["metrics"]["F"]["macro_f1"] == 1.0
    assert len(first["artifact_sha256"]) == len(first["identity_sha256"]) == 64
    assert first["artifact_sha256"] != first["identity_sha256"]
    assert "raw" not in repr(first).lower() and "reason text" not in repr(first).lower()


def test_exact_f_point_eight_and_s_point_seven_boundaries_pass_without_float_drift():
    def prediction(correct_per_label):
        return lambda index: LABELS[index % 2] if index // 2 < correct_per_label else LABELS[(index + 1) % 2]

    artifact = _artifact(s_rows=_rows(prediction(105)), f_rows=_rows(prediction(120)))

    assert artifact["metrics"] == {"S": {"macro_f1": 0.7}, "F": {"macro_f1": 0.8}}
    assert artifact["decision"]["status"] == "passed"


def test_agreement_requires_all_100_expected_pairs_even_when_the_available_pairs_agree_perfectly():
    artifact = _artifact(agreement_rows=_agreement()[:90])

    assert artifact["decision"] == {"status": "incomplete", "reason": "agreement_incomplete"}
    assert artifact["agreement"]["completed"] == 90


def test_an_absent_second_sme_answer_is_retained_as_incomplete_evidence_not_invented_as_a_label():
    rows = _agreement()
    rows[-1] = {"item_id": AGREEMENT_IDS[-1], "primary_label": LABELS[-1 % 2],
                "second_label": None, "status": "missing"}

    artifact = _artifact(agreement_rows=rows)

    assert artifact["decision"] == {"status": "incomplete", "reason": "agreement_incomplete"}
    assert artifact["agreement"] == {"expected": 100, "completed": 99, "agreement": 1.0}


def test_incomplete_paired_screen_rows_never_compute_a_pass_or_authorize_live_work():
    artifact = _artifact(f_rows=_rows(lambda index: LABELS[index % 2])[:-1])

    assert artifact["decision"] == {"status": "incomplete", "reason": "screen_incomplete"}
    with pytest.raises(ValueError, match="passed"):
        authorize_stage(artifact, {"stage": "stage1", "artifact_sha256": artifact["artifact_sha256"],
                                  "identity_sha256": artifact["identity_sha256"]})


def test_completed_rows_without_predictions_are_rejected_and_partial_counts_remain_auditable():
    rows = _rows(lambda index: LABELS[index % 2])
    rows[0] = {**rows[0], "predicted_label": None}
    with pytest.raises(ValueError, match="no predicted"):
        _artifact(f_rows=rows)
    partial = _artifact(f_rows=_rows(lambda index: LABELS[index % 2])[:-1])
    assert partial["screen"]["S_completed"] == 300
    assert partial["screen"]["F_completed"] == 299


def test_a_low_ceiling_requests_one_merge_then_stops_and_synthetic_evidence_never_authorizes():
    low = _artifact(f_rows=_rows(lambda _index: "approve"))
    assert low["decision"] == {"status": "merge_once_required", "reason": "ceiling_below_threshold"}
    stopped = _artifact(f_rows=_rows(lambda _index: "approve"), merge_attempt=1)
    assert stopped["decision"] == {"status": "stopped", "reason": "ceiling_below_threshold"}
    synthetic = _artifact(synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        authorize_stage(synthetic, {"stage": "stage1", "artifact_sha256": synthetic["artifact_sha256"],
                                    "identity_sha256": synthetic["identity_sha256"]})


def test_owner_receipts_are_hash_bound_and_stage_two_needs_a_stage_one_acknowledgement():
    artifact = _artifact()
    receipt = {"stage": "stage1", "artifact_sha256": artifact["artifact_sha256"],
               "identity_sha256": artifact["identity_sha256"], "confirmed": True}
    stage1 = authorize_stage(artifact, receipt)
    assert stage1["allowed_arms"] == ["B", "E"] and stage1["request_ceiling"] == 600
    with pytest.raises(ValueError, match="completed"):
        authorize_stage(artifact, {**receipt, "stage": "stage2", "stage1_result_sha256": "d" * 64})
    stage1_result = {"schema": "decision-flywheel-evaluations/reviews-stage1/v1",
                     "stage0_artifact_sha256": artifact["artifact_sha256"],
                     "evidence_sha256": "d" * 64, "status": "completed", "owner_reviewed": True}
    stage1_result = {**stage1_result, "artifact_sha256": gate._hash(stage1_result)}
    stage2_receipt = {**receipt, "stage": "stage2", "stage1_result_sha256": stage1_result["artifact_sha256"]}
    stage2 = authorize_stage(artifact, stage2_receipt, stage1_result=stage1_result)
    assert stage2["allowed_arms"] == ["L", "X"]


def test_evidence_rejects_mismatched_truths_and_completed_rows_without_canonical_labels():
    bad_truth = _rows(lambda index: LABELS[index % 2])
    bad_truth[0] = {**bad_truth[0], "true_label": "flag"}
    with pytest.raises(ValueError, match="truth"):
        _artifact(f_rows=bad_truth)
    bad_label = _rows(lambda index: LABELS[index % 2])
    bad_label[0] = {**bad_label[0], "predicted_label": "other"}
    with pytest.raises(ValueError, match="unknown predicted"):
        _artifact(f_rows=bad_label)


def test_artifact_binds_canonical_expected_ids_and_row_content_not_only_aggregates():
    first = _artifact()
    changed = _artifact(f_rows=_rows(lambda index: LABELS[(index + 1) % 2]))

    assert first["evidence_sha256"] != changed["evidence_sha256"]
    tampered = {**first, "identity": {**first["identity"], "jev_model": "other"}}
    with pytest.raises(ValueError, match="hash"):
        authorize_stage(tampered, {"stage": "stage1", "artifact_sha256": first["artifact_sha256"],
                                   "identity_sha256": first["identity_sha256"], "confirmed": True})


def test_authorization_requires_explicit_confirmation_safe_models_and_a_completed_reviewed_stage_one():
    with pytest.raises(ValueError, match="model"):
        _artifact(identity={**IDENTITY, "jev_model": "bad model name with spaces"})
    artifact = _artifact()
    incomplete_receipt = {"stage": "stage1", "artifact_sha256": artifact["artifact_sha256"],
                          "identity_sha256": artifact["identity_sha256"], "confirmed": False}
    with pytest.raises(ValueError, match="confirmation"):
        authorize_stage(artifact, incomplete_receipt)
    receipt = {**incomplete_receipt, "confirmed": True, "stage": "stage2", "stage1_result_sha256": "d" * 64}
    unfinished = {"schema": "decision-flywheel-evaluations/reviews-stage1/v1",
                  "stage0_artifact_sha256": artifact["artifact_sha256"],
                  "evidence_sha256": "d" * 64, "status": "running", "owner_reviewed": True, "artifact_sha256": "d" * 64}
    with pytest.raises(ValueError, match="completed"):
        authorize_stage(artifact, receipt, stage1_result=unfinished)


def test_authorization_recomputes_decision_even_when_a_tamperer_rehashes_the_artifact():
    artifact = _artifact()
    tampered = {**artifact, "metrics": {"S": {"macro_f1": .79}, "F": {"macro_f1": .8}}}
    tampered["artifact_sha256"] = gate._hash({key: value for key, value in tampered.items() if key != "artifact_sha256"})
    with pytest.raises(ValueError, match="decision"):
        authorize_stage(tampered, {"stage": "stage1", "artifact_sha256": tampered["artifact_sha256"],
                                   "identity_sha256": tampered["identity_sha256"], "confirmed": True})


def test_agreement_and_gap_thresholds_are_independent_and_inclusive():
    assert gate._decision(screen_complete=True, agreement_complete=True, agreement=.85,
                          s_f1=.7, f_f1=.8, merge_attempt=0)["status"] == "passed"
    assert gate._decision(screen_complete=True, agreement_complete=True, agreement=.84,
                          s_f1=.7, f_f1=.8, merge_attempt=0)["reason"] == "agreement_below_threshold"
    assert gate._decision(screen_complete=True, agreement_complete=True, agreement=.9,
                          s_f1=.71, f_f1=.8, merge_attempt=0)["reason"] == "gap_below_threshold"
