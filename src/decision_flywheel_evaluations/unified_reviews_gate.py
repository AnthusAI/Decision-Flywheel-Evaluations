"""Pure, text-free Stage 0 evidence and owner-acknowledgement gate.

This module neither opens a client nor reads a review, policy, cache, or file.
Callers must supply already-sanitized IDs, labels, hashes and completed decisions.
"""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .metrics import Observation, summarize


SCHEMA = "decision-flywheel-evaluations/reviews-stage0/v1"
_IDENTITY_KEYS = frozenset(("dataset_manifest_sha256", "policy_sha256", "split_sha256", "sme_model", "jev_model"))
_ROW_KEYS = frozenset(("item_id", "true_label", "predicted_label", "status"))
_AGREEMENT_KEYS = frozenset(("item_id", "primary_label", "second_label", "status"))
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,79}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be unique non-empty strings")
    out = tuple(values)
    if not out or any(not isinstance(value, str) or not _MODEL.fullmatch(value) for value in out):
        raise ValueError(f"{name} must be unique non-empty strings")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} must be unique non-empty strings")
    return out


def _identity(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise ValueError("identity has unsupported fields")
    result = {key: value[key] for key in sorted(_IDENTITY_KEYS)}
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise ValueError("identity values must be non-empty strings")
    for key in ("dataset_manifest_sha256", "policy_sha256", "split_sha256"):
        if len(result[key]) != 64 or any(char not in "0123456789abcdef" for char in result[key]):
            raise ValueError("identity hashes must be lowercase sha256 values")
    if not _MODEL.fullmatch(result["sme_model"]) or not _MODEL.fullmatch(result["jev_model"]):
        raise ValueError("model identities must be safe bounded names")
    return result


def _digest_rows(rows: Mapping[str, Mapping[str, Any]], expected_ids: tuple[str, ...]) -> str:
    return _hash([dict(rows[item_id]) for item_id in expected_ids if item_id in rows])


def _screen_rows(rows: Sequence[Mapping[str, Any]], expected_ids: tuple[str, ...], labels: tuple[str, ...], condition: str):
    found = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("screen rows must be a sequence")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            raise ValueError("screen rows have unsupported fields")
        item_id = row["item_id"]
        if not isinstance(item_id, str) or not _MODEL.fullmatch(item_id):
            raise ValueError("screen row IDs must be safe strings")
        if item_id not in expected_ids or item_id in found:
            raise ValueError("screen row IDs must be unique expected IDs")
        found[item_id] = Observation(f"{condition}:{item_id}", item_id, condition, 0, "stage0",
                                     row["true_label"], row["predicted_label"], row["status"])
    # Validate canonical label wires even for a later-incomplete screen.
    if found:
        summarize(tuple(found.values()), labels)
    complete = len(found) == len(expected_ids) and all(found[item_id].status == "completed" for item_id in expected_ids)
    if not complete:
        return None, False, {item_id: {"item_id": item_id, "true_label": row.true_label,
                                       "predicted_label": row.predicted_label, "status": row.status}
                             for item_id, row in found.items()}
    ordered = tuple(found[item_id] for item_id in expected_ids)
    return summarize(ordered, labels), True, {item_id: {"item_id": item_id, "true_label": row.true_label,
                                                        "predicted_label": row.predicted_label, "status": row.status}
                                                  for item_id, row in found.items()}


def _agreement(rows: Sequence[Mapping[str, Any]], expected_ids: tuple[str, ...], labels: tuple[str, ...]):
    if len(expected_ids) != 100:
        raise ValueError("Stage 0 agreement requires exactly 100 expected IDs")
    found = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("agreement rows must be a sequence")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _AGREEMENT_KEYS:
            raise ValueError("agreement rows have unsupported fields")
        item_id = row["item_id"]
        if not isinstance(item_id, str) or not _MODEL.fullmatch(item_id):
            raise ValueError("agreement IDs must be safe strings")
        if item_id not in expected_ids or item_id in found:
            raise ValueError("agreement IDs must be unique expected IDs")
        if (not isinstance(row["status"], str) or row["status"] not in {"completed", "failed", "missing", "malformed"}
                or any(value is not None and (not isinstance(value, str) or value not in labels)
                       for value in (row["primary_label"], row["second_label"]))):
            raise ValueError("agreement row has invalid label or status")
        if row["status"] == "completed" and (row["primary_label"] is None or row["second_label"] is None):
            raise ValueError("completed agreement row requires both canonical labels")
        found[item_id] = row
    completed = [found[item_id] for item_id in expected_ids if item_id in found and found[item_id]["status"] == "completed"]
    complete = len(completed) == len(expected_ids)
    rate = sum(row["primary_label"] == row["second_label"] for row in completed) / len(completed) if completed else None
    safe = {item_id: {key: row[key] for key in sorted(_AGREEMENT_KEYS)} for item_id, row in found.items()}
    return {"expected": len(expected_ids), "completed": len(completed), "agreement": rate}, complete, safe


def _stage_ceilings(value: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != {"stage1", "stage2"}:
        raise ValueError("stage-specific ceilings for stage1 and stage2 are required")
    output = {}
    for stage in ("stage1", "stage2"):
        cap = value[stage]
        if not isinstance(cap, Mapping) or set(cap) != {"request_ceiling", "max_new_requests"}:
            raise ValueError("stage ceiling has unsupported fields")
        ceiling, maximum = cap["request_ceiling"], cap["max_new_requests"]
        if (isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0
                or isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= ceiling):
            raise ValueError("stage ceiling is invalid")
        output[stage] = {"request_ceiling": ceiling, "max_new_requests": maximum}
    return output


def _metric(summary):
    if summary is None or summary.macro_f1 is None:
        return None
    return float(summary.macro_f1)


def _decision(*, screen_complete: bool, agreement_complete: bool, agreement: float | None,
              s_f1: float | None, f_f1: float | None, merge_attempt: int) -> dict[str, str]:
    if not screen_complete:
        return {"status": "incomplete", "reason": "screen_incomplete"}
    if not agreement_complete:
        return {"status": "incomplete", "reason": "agreement_incomplete"}
    if Decimal(str(agreement)) < Decimal("0.85"):
        return {"status": "stopped", "reason": "agreement_below_threshold"}
    if Decimal(str(f_f1)) < Decimal("0.80"):
        return {"status": "merge_once_required" if merge_attempt == 0 else "stopped", "reason": "ceiling_below_threshold"}
    if Decimal(str(s_f1)) > Decimal(str(f_f1)) - Decimal("0.10"):
        return {"status": "stopped", "reason": "gap_below_threshold"}
    return {"status": "passed", "reason": "screen_passed"}


def evaluate_stage0(*, identity: Mapping[str, str], labels: Sequence[str], screen_expected_ids: Sequence[str],
                    agreement_expected_ids: Sequence[str], agreement_rows: Sequence[Mapping[str, Any]],
                    s_rows: Sequence[Mapping[str, Any]], f_rows: Sequence[Mapping[str, Any]],
                    stage_ceilings: Mapping[str, Mapping[str, int]], synthetic: bool,
                    merge_attempt: int) -> dict[str, Any]:
    """Return immutable, text-free Stage 0 evidence without any side effect."""
    if (not isinstance(synthetic, bool)
            or isinstance(merge_attempt, bool) or not isinstance(merge_attempt, int) or merge_attempt not in (0, 1)):
        raise ValueError("Stage 0 cap, synthetic flag, or merge attempt is invalid")
    identity = _identity(identity)
    labels = _strings(labels, "labels")
    expected = _strings(screen_expected_ids, "screen expected IDs")
    if len(expected) != 300:
        raise ValueError("Stage 0 screen requires exactly 300 expected IDs")
    agreement_ids = _strings(agreement_expected_ids, "agreement expected IDs")
    ceilings = _stage_ceilings(stage_ceilings)
    agreement, agreement_complete, agreement_safe = _agreement(agreement_rows, agreement_ids, labels)
    s_summary, s_complete, s_safe = _screen_rows(s_rows, expected, labels, "S")
    f_summary, f_complete, f_safe = _screen_rows(f_rows, expected, labels, "F")
    for item_id in set(s_safe) & set(f_safe):
        if s_safe[item_id]["true_label"] != f_safe[item_id]["true_label"]:
            raise ValueError("S and F truth labels must match per target")
    screen_complete = s_complete and f_complete
    s_f1, f_f1 = _metric(s_summary), _metric(f_summary)
    decision = _decision(screen_complete=screen_complete, agreement_complete=agreement_complete,
                         agreement=agreement["agreement"], s_f1=s_f1, f_f1=f_f1, merge_attempt=merge_attempt)
    evidence = {"screen_expected_ids_sha256": _hash(list(expected)),
                "agreement_expected_ids_sha256": _hash(list(agreement_ids)),
                "S_rows_sha256": _digest_rows(s_safe, expected), "F_rows_sha256": _digest_rows(f_safe, expected),
                "agreement_rows_sha256": _digest_rows(agreement_safe, agreement_ids)}
    payload = {"schema": SCHEMA, "identity": identity, "identity_sha256": _hash(identity), "labels": list(labels),
               "screen": {"expected": len(expected), "S_completed": sum(row["status"] == "completed" for row in s_safe.values()),
                          "F_completed": sum(row["status"] == "completed" for row in f_safe.values()), **evidence}, "agreement": agreement,
               "metrics": {"S": {"macro_f1": s_f1}, "F": {"macro_f1": f_f1}},
               "decision": decision, "synthetic": synthetic, "merge_attempt": merge_attempt,
               "stage_ceilings": ceilings,
               "evidence_sha256": _hash({"identity_sha256": _hash(identity), **evidence})}
    return {**payload, "artifact_sha256": _hash(payload)}


def _validate_artifact(artifact: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    expected = {"schema", "identity", "identity_sha256", "labels", "screen", "agreement", "metrics", "decision",
                "synthetic", "merge_attempt", "stage_ceilings", "evidence_sha256", "artifact_sha256"}
    if not isinstance(artifact, Mapping) or set(artifact) != expected or artifact.get("schema") != SCHEMA:
        raise ValueError("artifact schema is invalid")
    copied = dict(artifact)
    digest = copied.pop("artifact_sha256")
    if not isinstance(digest, str) or digest != _hash(copied):
        raise ValueError("artifact hash does not match")
    identity = _identity(artifact["identity"])
    if artifact["identity_sha256"] != _hash(identity):
        raise ValueError("identity hash does not match")
    _strings(artifact["labels"], "artifact labels")
    _stage_ceilings(artifact["stage_ceilings"])
    screen = artifact["screen"]
    required_screen = {"expected", "S_completed", "F_completed", "screen_expected_ids_sha256",
                       "agreement_expected_ids_sha256", "S_rows_sha256", "F_rows_sha256", "agreement_rows_sha256"}
    if not isinstance(screen, Mapping) or set(screen) != required_screen:
        raise ValueError("artifact screen schema is invalid")
    if any(not isinstance(screen[key], int) or isinstance(screen[key], bool) or screen[key] < 0
           for key in ("expected", "S_completed", "F_completed")):
        raise ValueError("artifact screen counts are invalid")
    for key in required_screen - {"expected", "S_completed", "F_completed"}:
        value = screen[key]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("artifact evidence hash is invalid")
    evidence = {key: screen[key] for key in required_screen - {"expected", "S_completed", "F_completed"}}
    if artifact["evidence_sha256"] != _hash({"identity_sha256": artifact["identity_sha256"], **evidence}):
        raise ValueError("evidence hash does not match")
    agreement = artifact["agreement"]
    if not isinstance(agreement, Mapping) or set(agreement) != {"expected", "completed", "agreement"}:
        raise ValueError("artifact agreement schema is invalid")
    if any(not isinstance(agreement[key], int) or isinstance(agreement[key], bool) or agreement[key] < 0
           for key in ("expected", "completed")) or agreement["completed"] > agreement["expected"]:
        raise ValueError("artifact agreement counts are invalid")
    if agreement["agreement"] is not None and (not isinstance(agreement["agreement"], (int, float))
                                                or isinstance(agreement["agreement"], bool)
                                                or not 0 <= agreement["agreement"] <= 1):
        raise ValueError("artifact agreement value is invalid")
    metrics = artifact["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {"S", "F"}:
        raise ValueError("artifact metrics schema is invalid")
    for condition in ("S", "F"):
        if not isinstance(metrics[condition], Mapping) or set(metrics[condition]) != {"macro_f1"}:
            raise ValueError("artifact metric schema is invalid")
        value = metrics[condition]["macro_f1"]
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1):
            raise ValueError("artifact metric value is invalid")
    decision = artifact["decision"]
    allowed = {("incomplete", "screen_incomplete"), ("incomplete", "agreement_incomplete"),
               ("merge_once_required", "ceiling_below_threshold"), ("stopped", "ceiling_below_threshold"),
               ("stopped", "agreement_below_threshold"), ("stopped", "gap_below_threshold"),
               ("passed", "screen_passed")}
    if not isinstance(decision, Mapping) or set(decision) != {"status", "reason"} or (decision["status"], decision["reason"]) not in allowed:
        raise ValueError("artifact decision schema is invalid")
    screen_complete = screen["expected"] == 300 and screen["S_completed"] == screen["expected"] and screen["F_completed"] == screen["expected"]
    agreement_complete = agreement["expected"] == 100 and agreement["completed"] == agreement["expected"]
    if screen_complete and (metrics["S"]["macro_f1"] is None or metrics["F"]["macro_f1"] is None):
        raise ValueError("complete screen evidence requires complete metric coverage")
    recomputed = _decision(screen_complete=screen_complete, agreement_complete=agreement_complete,
                           agreement=agreement["agreement"], s_f1=metrics["S"]["macro_f1"],
                           f_f1=metrics["F"]["macro_f1"], merge_attempt=artifact["merge_attempt"])
    if decision != recomputed:
        raise ValueError("artifact decision does not match its validated evidence")
    if (not isinstance(artifact["synthetic"], bool) or isinstance(artifact["merge_attempt"], bool)
            or artifact["merge_attempt"] not in (0, 1)):
        raise ValueError("artifact synthetic or merge state is invalid")
    return copied, digest


def authorize_stage(artifact: Mapping[str, Any], receipt: Mapping[str, Any], *, stage1_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a hash-bound owner receipt and return only the permitted next arms/caps."""
    _validate_artifact(artifact)
    digest = artifact["artifact_sha256"]
    if artifact.get("synthetic"):
        raise ValueError("synthetic evidence never authorizes live work")
    if artifact.get("decision", {}).get("status") != "passed":
        raise ValueError("only a passed Stage 0 artifact may authorize work")
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("stage"), str):
        raise ValueError("receipt has unsupported fields")
    receipt_keys = {"stage", "artifact_sha256", "identity_sha256", "confirmed"}
    if receipt["stage"] == "stage2":
        receipt_keys.add("stage1_result_sha256")
    if set(receipt) != receipt_keys:
        raise ValueError("receipt has unsupported fields")
    if any(not isinstance(receipt[key], str) or not receipt[key] for key in ("stage", "artifact_sha256", "identity_sha256")):
        raise ValueError("receipt values must be non-empty strings")
    if receipt["stage"] == "stage2" and (not isinstance(receipt["stage1_result_sha256"], str)
                                            or len(receipt["stage1_result_sha256"]) != 64):
        raise ValueError("stage-two receipt must bind a stage-one result hash")
    if receipt["artifact_sha256"] != digest or receipt["identity_sha256"] != artifact.get("identity_sha256"):
        raise ValueError("receipt is not bound to this artifact identity")
    if receipt["confirmed"] is not True:
        raise ValueError("explicit owner confirmation is required")
    stage = receipt["stage"]
    if stage == "stage1":
        arms = ["B", "E"]
    elif stage == "stage2":
        expected_keys = {"schema", "stage0_artifact_sha256", "evidence_sha256", "status", "owner_reviewed", "artifact_sha256"}
        if not isinstance(stage1_result, Mapping) or set(stage1_result) != expected_keys:
            raise ValueError("stage-two authorization requires a completed, owner-reviewed stage-one result")
        stage1_copy = dict(stage1_result)
        stage1_digest = stage1_copy.pop("artifact_sha256")
        if (not isinstance(stage1_digest, str) or stage1_digest != _hash(stage1_copy)
                or receipt["stage1_result_sha256"] != stage1_digest
                or stage1_result["schema"] != "decision-flywheel-evaluations/reviews-stage1/v1"
                or stage1_result["stage0_artifact_sha256"] != digest
                or stage1_result["status"] != "completed" or stage1_result["owner_reviewed"] is not True
                or not isinstance(stage1_result["evidence_sha256"], str) or len(stage1_result["evidence_sha256"]) != 64):
            raise ValueError("stage-two authorization requires a completed, owner-reviewed stage-one result")
        arms = ["L", "X"]
    else:
        raise ValueError("receipt stage must be stage1 or stage2")
    return {"stage": stage, "allowed_arms": arms, **artifact["stage_ceilings"][stage], "artifact_sha256": digest}
