import asyncio
import hashlib
import json
from dataclasses import replace

import pytest
from decision_flywheel.context import RandomBalanced

from .cache import ApprovedRequest, CacheStore, LogicalCell
from .preflight import _selected_global_anchor, preflight, rehydrate_manifest
from .preflight_test import _protocol, _rows
from .protocol import transport_config_fingerprint
from .selection import optimize_from_cache


def _complete_cache(tmp_path, *, retry_first_request=False):
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows, artifact=False)
    plan = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                cell.wire_fingerprint, cell.dataset_revision, cell.model) for cell in plan.cells}
    cache = CacheStore(str(tmp_path / "selection.sqlite"), task=protocol.task,
                       preflight_fingerprint=plan.checksum, approved_requests=tuple(approved.values()),
                       ceiling=plan.new_request_count + int(retry_first_request))
    candidates, development, _ = rehydrate_manifest(manifest, rows, protocol)
    chosen = tuple(item.item.id for item in RandomBalanced(3).select(
        protocol.task, _selected_global_anchor(protocol.task, candidates), candidates, per_label=1))
    labels = {item.item.id: item.label for item in development}
    if retry_first_request:
        first = plan.cells[0]
        first_cell = LogicalCell(first.id, first.selector, first.draw_seed or 0,
                                 first.display_order, first.target_id)
        reserved = cache.reserve(first.fingerprint, first_cell)
        assert isinstance(reserved, str)
        cache.fail(reserved, "model-failure")
        retried = cache.reserve(first.fingerprint, first_cell)
        assert isinstance(retried, str)
        cache.complete(retried, {"choice": labels[first.target_id]})
    for cell in plan.cells:
        token = cache.reserve(cell.fingerprint, LogicalCell(cell.id, cell.selector, cell.draw_seed or 0,
                                                             cell.display_order, cell.target_id))
        prediction = labels[cell.target_id] if cell.example_ids == chosen else protocol.task.labels[0]
        if isinstance(token, str):
            cache.complete(token, {"choice": prediction})
    return manifest, rows, protocol, plan, cache


def test_complete_approved_cache_replays_native_search_without_model_calls_and_freezes_nonseed2_winner(tmp_path):
    manifest, rows, protocol, plan, cache = _complete_cache(tmp_path)
    result = asyncio.run(optimize_from_cache(protocol, manifest, rows, plan, cache,
                                              "jev:jev-1.13.0", "native-cache-selection"))
    assert result.optimization.model_calls_attempted == 0
    assert result.ledger_attempts == plan.new_request_count
    assert result.replayed_decisions == len(plan.cells)
    assert result.artifact.selection_seed == 3
    again = asyncio.run(optimize_from_cache(protocol, manifest, rows, plan, cache,
                                             "jev:jev-1.13.0", "native-cache-selection"))
    first_payload = json.dumps(result.artifact.payload(), sort_keys=True, separators=(",", ":"))
    second_payload = json.dumps(again.artifact.payload(), sort_keys=True, separators=(",", ":"))
    assert second_payload == first_payload
    assert hashlib.sha256(second_payload.encode()).hexdigest() == hashlib.sha256(first_payload.encode()).hexdigest()
    assert rows[0].text not in first_payload
    assert "TYPESAFE_API_KEY" not in first_payload


def test_selection_refuses_incomplete_or_wrong_model_evidence_without_provisional_winner(tmp_path):
    manifest, rows, protocol, plan, cache = _complete_cache(tmp_path)
    request_id = plan.cells[0].fingerprint
    cache.db.execute("UPDATE requests SET status = 'failed' WHERE request_id = ?", (request_id,))
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(optimize_from_cache(protocol, manifest, rows, plan, cache,
                                         "jev:jev-1.13.0", "native-cache-selection"))
    with pytest.raises(ValueError, match="declared model"):
        asyncio.run(optimize_from_cache(protocol, manifest, rows, plan, cache,
                                         "jev:other", "native-cache-selection"))


def test_selection_refuses_malformed_cache_payload_and_mismatched_cache_configuration(tmp_path):
    manifest, rows, protocol, plan, cache = _complete_cache(tmp_path)
    request_id = plan.cells[0].fingerprint
    cache.db.execute(
        "UPDATE requests SET payload = ? WHERE request_id = ?",
        (json.dumps({"status": "success", "choice": "not-a-frozen-label", "model": "jev:jev-1.13.0"}), request_id),
    )
    with pytest.raises(ValueError, match="invalid label"):
        asyncio.run(optimize_from_cache(protocol, manifest, rows, plan, cache,
                                         "jev:jev-1.13.0", "native-cache-selection"))

    stale = replace(plan, checksum="f" * 64)
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(optimize_from_cache(protocol, manifest, rows, stale, cache,
                                         "jev:jev-1.13.0", "native-cache-selection"))

    approved = {
        cell.fingerprint: ApprovedRequest(
            cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
            cell.dataset_revision, cell.model,
        )
        for cell in plan.cells
    }
    mismatched = CacheStore(
        str(tmp_path / "mismatched-config.sqlite"), task=protocol.task,
        preflight_fingerprint="f" * 64, approved_requests=tuple(approved.values()),
        ceiling=plan.new_request_count,
    )
    with pytest.raises(ValueError, match="cache does not match"):
        asyncio.run(optimize_from_cache(protocol, manifest, rows, plan, mismatched,
                                         "jev:jev-1.13.0", "native-cache-selection"))


def test_selection_refuses_a_changed_model_transport_before_replaying_evidence(tmp_path):
    manifest, rows, protocol, plan, cache = _complete_cache(tmp_path)
    changed_jev = replace(
        protocol.models[0],
        transport_fingerprint=transport_config_fingerprint(
            base_url="https://different.invalid/v1", timeout_seconds=10.0, retries=0,
        ),
    )
    changed = replace(protocol, models=(changed_jev, protocol.models[1]))

    with pytest.raises(ValueError, match="frozen preflight cells"):
        asyncio.run(optimize_from_cache(changed, manifest, rows, plan, cache,
                                         changed_jev.semantic_identity, "native-cache-selection"))


def test_native_selection_never_needs_heldout_source_rows(tmp_path):
    manifest, rows, protocol, plan, cache = _complete_cache(tmp_path)
    development_rows = tuple(row for row in rows if row.id not in {record.id for record in manifest.scoreboard})

    result = asyncio.run(optimize_from_cache(
        protocol, manifest, development_rows, plan, cache,
        "jev:jev-1.13.0", "native-cache-selection",
    ))

    assert result.optimization.model_calls_attempted == 0


def test_selection_accepts_complete_cached_evidence_with_a_predeclared_retry_allowance(tmp_path):
    manifest, rows, protocol, plan, cache = _complete_cache(
        tmp_path, retry_first_request=True,
    )
    cache_aware = preflight(
        protocol, manifest=manifest, rows=rows, stage="optimization",
        cached_fingerprints=frozenset(cell.fingerprint for cell in plan.cells),
    )
    assert cache_aware.checksum == plan.checksum
    assert cache_aware.new_request_count == 0

    result = asyncio.run(optimize_from_cache(
        protocol, manifest, rows, cache_aware, cache,
        "jev:jev-1.13.0", "native-cache-selection",
    ))
    assert result.ledger_attempts == plan.new_request_count + 1
    assert result.optimization.model_calls_attempted == 0


def test_selection_rejects_cache_ceiling_above_the_declared_optimization_budget(tmp_path):
    manifest, rows, protocol, plan, _cache = _complete_cache(tmp_path)
    approved = {
        cell.fingerprint: ApprovedRequest(
            cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
            cell.dataset_revision, cell.model,
        )
        for cell in plan.cells
    }
    over_budget = CacheStore(
        str(tmp_path / "over-budget.sqlite"), task=protocol.task,
        preflight_fingerprint=plan.checksum, approved_requests=tuple(approved.values()),
        ceiling=protocol.optimization.max_model_calls + 1,
    )
    with pytest.raises(ValueError, match="cache does not match"):
        asyncio.run(optimize_from_cache(
            protocol, manifest, rows, plan, over_budget,
            "jev:jev-1.13.0", "native-cache-selection",
        ))
