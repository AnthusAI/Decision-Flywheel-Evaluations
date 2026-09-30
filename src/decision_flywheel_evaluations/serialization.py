"""Strict, small JSON boundaries for the native evaluation command line.

Protocol and preflight documents intentionally contain only executable metadata,
identities, labels, and hashes.  Source text has a separate local-row fixture
format and is loaded only by a caller that explicitly names that file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from decision_flywheel.models import DecisionTask

from .datasets import DatasetRow
from .metrics import Observation
from .preflight import ExcludedCell, PreflightResult, RequestCell
from .protocol import (Capability, CompletedTrialRecord, FrozenProtocol,
                       ModelIdentity, OptimizationSpec, SelectedGlobalArtifact,
                       Selector)
from .study import Engine


_PROTOCOL_SCHEMA = "decision-flywheel-evaluations/protocol/v1"
_PREFLIGHT_SCHEMA = "decision-flywheel-evaluations/preflight/v1"
_ROWS_SCHEMA = "decision-flywheel-evaluations/rows-fixture/v1"
_OBSERVATIONS_SCHEMA = "decision-flywheel-evaluations/observations/v1"


def write_protocol(path: str | Path, protocol: FrozenProtocol) -> None:
    """Persist one validated, explicit protocol without source-row text."""
    protocol.validate()
    _write(path, {"schema": _PROTOCOL_SCHEMA, "protocol": _protocol_payload(protocol)})


def read_protocol(path: str | Path) -> FrozenProtocol:
    document = _read(path, _PROTOCOL_SCHEMA, {"schema", "protocol"})
    payload = _mapping(document["protocol"], "protocol")
    expected = {"name", "dataset", "dataset_manifest_sha256", "candidate_count", "development_count",
                "scoreboard_count", "task", "models", "selectors", "per_label_sizes", "random_draw_seeds",
                "display_rule", "selector_configuration_version", "selector_search_artifact", "optimization",
                "selection_transfer_source", "metric", "primary_family_correction"}
    _keys(payload, expected, "protocol")
    task_payload = _mapping(payload["task"], "task")
    _keys(task_payload, {"name", "labels", "instructions", "input_field"}, "task")
    task = DecisionTask(task_payload["name"], tuple(_strings(task_payload["labels"], "task labels")),
                        task_payload["instructions"], task_payload["input_field"])
    models = tuple(_model(item) for item in _sequence(payload["models"], "models"))
    optimization = _optimization(_mapping(payload["optimization"], "optimization"))
    artifact_payload = payload["selector_search_artifact"]
    artifact = None if artifact_payload is None else _artifact(_mapping(artifact_payload, "selected-global artifact"))
    protocol = FrozenProtocol(
        payload["name"], payload["dataset"], payload["dataset_manifest_sha256"], payload["candidate_count"],
        payload["development_count"], payload["scoreboard_count"], task, models,
        tuple(Selector(item) for item in _strings(payload["selectors"], "selectors")),
        tuple(_integers(payload["per_label_sizes"], "per-label sizes")),
        tuple(_integers(payload["random_draw_seeds"], "random draw seeds")), payload["display_rule"],
        payload["selector_configuration_version"], artifact, optimization, payload["selection_transfer_source"],
        payload["metric"], payload["primary_family_correction"],
    )
    protocol.validate()
    return protocol


def write_preflight(path: str | Path, result: PreflightResult) -> None:
    """Persist enumerated core plan metadata, never state text or provider output."""
    if not isinstance(result, PreflightResult):
        raise ValueError("preflight result must be typed")
    _write(path, {"schema": _PREFLIGHT_SCHEMA, "preflight": {
        "cells": [_cell_payload(cell) for cell in result.cells],
        "excluded": [_excluded_payload(item) for item in result.excluded],
        "blockers": list(result.blockers), "new_request_count": result.new_request_count,
        "cache_hit_count": result.cache_hit_count, "optimization_count": result.optimization_count,
        "scoreboard_count": result.scoreboard_count, "checksum": result.checksum, "stage": result.stage,
        "selected_stage_complete": result.selected_stage_complete,
    }})


def read_preflight(path: str | Path) -> PreflightResult:
    document = _read(path, _PREFLIGHT_SCHEMA, {"schema", "preflight"})
    payload = _mapping(document["preflight"], "preflight")
    expected = {"cells", "excluded", "blockers", "new_request_count", "cache_hit_count", "optimization_count",
                "scoreboard_count", "checksum", "stage", "selected_stage_complete"}
    _keys(payload, expected, "preflight")
    cells = tuple(_cell(_mapping(value, "request cell")) for value in _sequence(payload["cells"], "cells"))
    excluded = tuple(_excluded(_mapping(value, "excluded cell")) for value in _sequence(payload["excluded"], "excluded"))
    counts = (payload["new_request_count"], payload["cache_hit_count"],
              payload["optimization_count"], payload["scoreboard_count"])
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise ValueError("preflight counts must be non-negative integers")
    result = PreflightResult(cells, excluded, tuple(_strings(payload["blockers"], "blockers")), *counts,
                             payload["checksum"], payload["stage"], payload["selected_stage_complete"])
    if not isinstance(result.selected_stage_complete, bool):
        raise ValueError("preflight selected stage status must be boolean")
    return result


def write_rows_fixture(path: str | Path, rows: Sequence[DatasetRow]) -> None:
    """Write synthetic test fixtures only, never a benchmark output artifact.

    Real source text remains caller-owned input and must not be written by the
    evaluation CLI.  This helper exists so offline specs can create a local
    fixture without importing a dataset client.
    """
    if not str(path).endswith(".fixture.json"):
        raise ValueError("synthetic row fixtures must use a .fixture.json path")
    if any(not isinstance(row, DatasetRow) for row in rows):
        raise ValueError("local rows need DatasetRow values")
    _write(path, {"schema": _ROWS_SCHEMA, "rows": [row.__dict__ for row in rows]})


def read_rows_fixture(path: str | Path) -> tuple[DatasetRow, ...]:
    """Read caller-provided local rows only; this function has no network path."""
    document = _read(path, _ROWS_SCHEMA, {"schema", "rows"})
    rows = []
    for value in _sequence(document["rows"], "rows"):
        item = _mapping(value, "row")
        _keys(item, {"id", "source_split", "source_index", "label", "text"}, "row")
        rows.append(DatasetRow(item["id"], item["source_split"], item["source_index"], item["label"], item["text"]))
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("local rows must have unique IDs")
    return tuple(rows)


def write_observations(path: str | Path, rows: Sequence[Observation]) -> None:
    """Persist sanitized logical observations, never source row text or provider payloads."""
    if any(not isinstance(row, Observation) for row in rows):
        raise ValueError("observations must be typed")
    _write(path, {"schema": _OBSERVATIONS_SCHEMA, "observations": [jsonable(row) for row in rows]})


def write_selected_global_artifact(path: str | Path, artifact: SelectedGlobalArtifact, *,
                                   source_protocol_identity: str, source_preflight_checksum: str,
                                   derived_protocol_identity: str) -> None:
    """Write a text-free selection link; the derived protocol remains authoritative."""
    if not all(isinstance(value, str) and value for value in
               (source_protocol_identity, source_preflight_checksum, derived_protocol_identity)):
        raise ValueError("selection linkage needs non-empty frozen identities")
    _write(path, {"schema": "decision-flywheel-evaluations/selected-global-artifact/v1",
                  "source_protocol_identity": source_protocol_identity,
                  "source_preflight_checksum": source_preflight_checksum,
                  "derived_protocol_identity": derived_protocol_identity,
                  "artifact": artifact.payload()})


def read_observations(path: str | Path) -> tuple[Observation, ...]:
    document = _read(path, _OBSERVATIONS_SCHEMA, {"schema", "observations"})
    legacy_expected = {"request_id", "target_id", "condition", "draw", "order", "true_label", "predicted_label", "status",
                       "probabilities", "model_id", "usage", "latency_ms", "attempt_count", "cache_hit", "physical_request_id",
                       "physical_request_provenance"}
    expected = legacy_expected | {"confidence"}
    rows = []
    for value in _sequence(document["observations"], "observations"):
        item = _mapping(value, "observation")
        if set(item) not in {frozenset(legacy_expected), frozenset(expected)}:
            raise ValueError("observation has unsupported fields")
        probabilities = item["probabilities"]
        usage = item["usage"]
        provenance = item["physical_request_provenance"]
        if probabilities is not None: probabilities = _mapping(probabilities, "probabilities")
        if usage is not None: usage = _mapping(usage, "usage")
        if provenance is not None: provenance = _mapping(provenance, "physical request provenance")
        rows.append(Observation(item["request_id"], item["target_id"], item["condition"], item["draw"], item["order"],
                                item["true_label"], item["predicted_label"], item["status"], probabilities=probabilities,
                                model_id=item["model_id"], usage=usage, latency_ms=item["latency_ms"],
                                attempt_count=item["attempt_count"], cache_hit=item["cache_hit"],
                                physical_request_id=item["physical_request_id"],
                                physical_request_provenance=provenance,
                                confidence=item.get("confidence")))
    return tuple(rows)


def jsonable(value: object) -> object:
    """Convert report dataclasses recursively while preserving every named field."""
    from dataclasses import fields, is_dataclass
    from enum import Enum
    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    return value


def _protocol_payload(protocol: FrozenProtocol) -> dict[str, object]:
    return {"name": protocol.name, "dataset": protocol.dataset, "dataset_manifest_sha256": protocol.dataset_manifest_sha256,
            "candidate_count": protocol.candidate_count, "development_count": protocol.development_count,
            "scoreboard_count": protocol.scoreboard_count,
            "task": {"name": protocol.task.name, "labels": list(protocol.task.labels),
                     "instructions": protocol.task.instructions, "input_field": protocol.task.input_field},
            "models": [{"engine": item.engine.value, "version": item.version,
                        "capabilities": sorted(value.value for value in item.capabilities),
                        "max_per_label": item.max_per_label,
                        "transport_fingerprint": item.transport_fingerprint} for item in protocol.models],
            "selectors": [item.value for item in protocol.selectors], "per_label_sizes": list(protocol.per_label_sizes),
            "random_draw_seeds": list(protocol.random_draw_seeds), "display_rule": protocol.display_rule,
            "selector_configuration_version": protocol.selector_configuration_version,
            "selector_search_artifact": protocol.selector_search_artifact.payload() if protocol.selector_search_artifact else None,
            "optimization": {"selectors": [item.value for item in protocol.optimization.selectors],
                             "per_label_sizes": list(protocol.optimization.per_label_sizes),
                             "draw_seeds": list(protocol.optimization.draw_seeds),
                             "max_model_calls": protocol.optimization.max_model_calls,
                             "artifact_checksum": protocol.optimization.artifact_checksum},
            "selection_transfer_source": protocol.selection_transfer_source, "metric": protocol.metric,
            "primary_family_correction": protocol.primary_family_correction}


def _model(value: object) -> ModelIdentity:
    payload = _mapping(value, "model")
    _keys(payload, {"engine", "version", "capabilities", "max_per_label", "transport_fingerprint"}, "model")
    return ModelIdentity(Engine(payload["engine"]), payload["version"],
                         frozenset(Capability(item) for item in _strings(payload["capabilities"], "capabilities")),
                         payload["max_per_label"], payload["transport_fingerprint"])


def _optimization(payload: Mapping[str, object]) -> OptimizationSpec:
    _keys(payload, {"selectors", "per_label_sizes", "draw_seeds", "max_model_calls", "artifact_checksum"}, "optimization")
    return OptimizationSpec(tuple(Selector(item) for item in _strings(payload["selectors"], "optimization selectors")),
                            tuple(_integers(payload["per_label_sizes"], "optimization sizes")),
                            tuple(_integers(payload["draw_seeds"], "optimization seeds")),
                            payload["max_model_calls"], payload["artifact_checksum"])


def _artifact(payload: Mapping[str, object]) -> SelectedGlobalArtifact:
    expected = {"reference", "checksum", "task_fingerprint", "candidate_pool_fingerprint", "development_split_fingerprint",
                "search_fingerprint", "objective", "objective_value", "model_fingerprint", "winner_trial_name",
                "policy_name", "policy_fingerprint", "selection_seed", "completed_trial_names",
                "completed_trials_fingerprint", "completed_trials", "context_policy_artifact", "ids_by_size"}
    _keys(payload, expected, "selected-global artifact")
    trials = tuple(_trial(_mapping(item, "completed trial")) for item in _sequence(payload["completed_trials"], "completed trials"))
    sizes = _mapping(payload["ids_by_size"], "selected global IDs")
    ids_by_size = {int(size): tuple(_strings(ids, "selected global IDs")) for size, ids in sizes.items()}
    return SelectedGlobalArtifact(payload["reference"], payload["checksum"], payload["task_fingerprint"],
                                  payload["candidate_pool_fingerprint"], payload["development_split_fingerprint"],
                                  payload["search_fingerprint"], payload["objective"], payload["objective_value"],
                                  payload["model_fingerprint"], payload["winner_trial_name"], payload["policy_name"],
                                  payload["policy_fingerprint"], payload["selection_seed"],
                                  tuple(_strings(payload["completed_trial_names"], "completed trial names")),
                                  payload["completed_trials_fingerprint"], trials, payload["context_policy_artifact"], ids_by_size)


def _trial(payload: Mapping[str, object]) -> CompletedTrialRecord:
    expected = {"trial_name", "policy_name", "policy_version", "selection_mode", "policy_configuration", "policy_fingerprint",
                "per_label", "max_tokens", "provider_token_limit", "display_order", "order_seed", "presentation_label_order",
                "status", "objective", "development_split_fingerprint", "decision_target_ids", "decision_request_fingerprints",
                "decision_predictions"}
    _keys(payload, expected, "completed trial")
    return CompletedTrialRecord(payload["trial_name"], payload["policy_name"], payload["policy_version"], payload["selection_mode"],
                                _mapping(payload["policy_configuration"], "policy configuration"), payload["policy_fingerprint"],
                                payload["per_label"], payload["max_tokens"], payload["provider_token_limit"], payload["display_order"],
                                payload["order_seed"], tuple(_strings(payload["presentation_label_order"], "presentation labels")),
                                payload["status"], payload["objective"], payload["development_split_fingerprint"],
                                tuple(_strings(payload["decision_target_ids"], "decision target IDs")),
                                tuple(_strings(payload["decision_request_fingerprints"], "decision request fingerprints")),
                                tuple(_strings(payload["decision_predictions"], "decision predictions")))


def _cell_payload(cell: RequestCell) -> dict[str, object]:
    return {"id": cell.id, "fingerprint": cell.fingerprint, "phase": cell.phase, "model": cell.model,
            "selector": cell.selector, "per_label": cell.per_label, "draw_seed": cell.draw_seed,
            "target_id": cell.target_id, "example_ids": list(cell.example_ids),
            "estimated_request_tokens": cell.estimated_request_tokens, "counter_identity": cell.counter_identity,
            "wire_fingerprint": cell.wire_fingerprint, "task_fingerprint": cell.task_fingerprint,
            "dataset_revision": cell.dataset_revision, "display_order": cell.display_order}


def _cell(payload: Mapping[str, object]) -> RequestCell:
    expected = {"id", "fingerprint", "phase", "model", "selector", "per_label", "draw_seed", "target_id", "example_ids",
                "estimated_request_tokens", "counter_identity", "wire_fingerprint", "task_fingerprint", "dataset_revision", "display_order"}
    _keys(payload, expected, "request cell")
    return RequestCell(payload["id"], payload["fingerprint"], payload["phase"], payload["model"], payload["selector"],
                       payload["per_label"], payload["draw_seed"], payload["target_id"],
                       tuple(_strings(payload["example_ids"], "example IDs")), payload["estimated_request_tokens"],
                       payload["counter_identity"], payload["wire_fingerprint"], payload["task_fingerprint"],
                       payload["dataset_revision"], payload["display_order"])


def _excluded_payload(item: ExcludedCell) -> dict[str, object]:
    return {"phase": item.phase, "model": item.model, "selector": item.selector, "per_label": item.per_label,
            "draw_seed": item.draw_seed, "target_count": item.target_count, "reason": item.reason}


def _excluded(payload: Mapping[str, object]) -> ExcludedCell:
    _keys(payload, {"phase", "model", "selector", "per_label", "draw_seed", "target_count", "reason"}, "excluded cell")
    return ExcludedCell(payload["phase"], payload["model"], payload["selector"], payload["per_label"],
                        payload["draw_seed"], payload["target_count"], payload["reason"])


def _write(path: str | Path, document: Mapping[str, object]) -> None:
    Path(path).write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _read(path: str | Path, schema: str, expected_keys: set[str]) -> Mapping[str, object]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("serialized document is unavailable or invalid JSON") from error
    payload = _mapping(document, "serialized document")
    _keys(payload, expected_keys, "serialized document")
    if payload["schema"] != schema:
        raise ValueError("serialized document has an unsupported schema")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _strings(value: object, name: str) -> list[str]:
    values = _sequence(value, name)
    if any(not isinstance(item, str) for item in values):
        raise ValueError(f"{name} must contain strings")
    return list(values)


def _integers(value: object, name: str) -> list[int]:
    values = _sequence(value, name)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise ValueError(f"{name} must contain integers")
    return list(values)


def _keys(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has unsupported fields")
