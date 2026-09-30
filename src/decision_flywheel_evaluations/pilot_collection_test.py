import asyncio
import hashlib
from dataclasses import replace

from decision_flywheel.models import DecisionResult

from .cache import ApprovedRequest, CacheStore, LogicalCell
from .collector import CollectionApproval, CollectionOptions
from .pilot import PilotPlan, plan_jev_pilot
from .pilot_collection import collect_pilot
from .preflight import preflight
from .preflight_test import _protocol, _rows


class _CountingModel:
    def __init__(self, calls): self.calls = calls
    async def decide(self, _task, _target, _context):
        self.calls.append("decision")
        return DecisionResult("world", usage={"tokens": 3})


class _MalformedModel:
    async def decide(self, _task, _target, _context): return object()


def _prepared_pilot(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    manifest, rows = _rows()
    base = _protocol(manifest, rows, artifact=False)
    protocol = replace(base, models=(base.models[0],))
    development_rows = tuple(row for row in rows if row.id not in {record.id for record in manifest.scoreboard})
    source = preflight(protocol, manifest=manifest, rows=development_rows, stage="optimization")
    pilot = plan_jev_pilot(protocol, manifest, development_rows, source)
    assert isinstance(pilot, PilotPlan) and len(pilot.cells) == 21
    preregistration = tmp_path / "pilot-preregistration.md"
    preregistration.write_text(f"{protocol.identity} {source.checksum} {pilot.checksum}", encoding="utf-8")
    approval = CollectionApproval(
        str(preregistration), hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        protocol.identity, pilot.checksum, pilot.physical_count, True,
    )
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                cell.wire_fingerprint, cell.dataset_revision, cell.model) for cell in pilot.cells}
    cache = CacheStore(str(tmp_path / "pilot.sqlite"), task=protocol.task,
                       preflight_fingerprint=pilot.checksum,
                       approved_requests=tuple(approved.values()), ceiling=pilot.physical_count)
    return manifest, development_rows, protocol, source, pilot, approval, cache


def test_a_pilot_rejects_bad_approval_or_scoreboard_source_before_constructing_a_model(tmp_path):
    manifest, rows, protocol, source, pilot, approval, cache = _prepared_pilot(tmp_path)
    factories = []
    factory = lambda model: factories.append(model)
    for bad_approval, bad_rows, bad_source, bad_pilot in (
        (replace(approval, preflight_checksum="0" * 64), rows, source, pilot),
        (approval, rows + (next(row for row in _rows()[1] if row.id in {record.id for record in manifest.scoreboard}),), source, pilot),
        (approval, rows, replace(source, checksum="0" * 64), pilot),
        (approval, rows, source, replace(pilot, protocol_identity="0" * 64)),
        (approval, rows, source, replace(pilot, cells=(replace(pilot.cells[0], target_id=manifest.scoreboard[0].id),) + pilot.cells[1:])),
    ):
        try:
            asyncio.run(collect_pilot(protocol, manifest, bad_rows, bad_source, bad_pilot, cache, bad_approval, factory,
                                      committed_checker=lambda *_: True,
                                      options=CollectionOptions(max_new_attempts=1)))
        except ValueError:
            pass
        else:
            raise AssertionError("unapproved pilot source reached a factory")
    assert factories == []

    approved = tuple(ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                                     cell.dataset_revision, cell.model)
                     for cell in {cell.fingerprint: cell for cell in pilot.cells}.values())
    subset = CacheStore(str(tmp_path / "subset.sqlite"), task=protocol.task,
                        preflight_fingerprint=pilot.checksum, approved_requests=approved[:-1],
                        ceiling=pilot.physical_count)
    try:
        with __import__("pytest").raises(ValueError, match="cache ceiling"):
            asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, subset, approval, factory,
                                      committed_checker=lambda *_: True,
                                      options=CollectionOptions(max_new_attempts=1)))
    finally:
        subset.close()
    assert factories == []


def test_a_pilot_has_an_explicit_physical_cap_and_never_retries_or_recovers(tmp_path):
    manifest, rows, protocol, source, pilot, approval, cache = _prepared_pilot(tmp_path)
    calls = []
    result = asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, cache, approval,
                                       lambda _model: _CountingModel(calls), committed_checker=lambda *_: True,
                                       options=CollectionOptions(max_new_attempts=1)))
    assert calls == ["decision"] and result.new_attempts == result.physical_attempts == 1
    assert any(row.status == "missing" for row in result.observations)
    for options in (CollectionOptions(max_new_attempts=1, retry_failed=True, max_retries_per_request=1),
                    CollectionOptions(max_new_attempts=1, recover_uncertain=True, max_retries_per_request=1)):
        try:
            asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, cache, approval,
                                      lambda _model: _CountingModel([]), committed_checker=lambda *_: True,
                                      options=options))
        except ValueError:
            pass
        else:
            raise AssertionError("pilot retry or recovery was accepted")


def test_a_pilot_rejects_historically_retried_requests_and_unfrozen_provider_aliases_before_factory(tmp_path):
    manifest, rows, protocol, source, pilot, approval, cache = _prepared_pilot(tmp_path)
    cell = pilot.cells[0]
    logical = LogicalCell(cell.id, f"pilot:{cell.model}:{cell.selector}:{cell.per_label}",
                          cell.draw_seed or 0, cell.display_order, cell.target_id)
    first = cache.reserve(cell.fingerprint, logical)
    cache.fail(first)
    second = cache.reserve(cell.fingerprint, logical)
    cache.fail(second)
    factories = []
    with __import__("pytest").raises(ValueError, match="no-retry"):
        asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, cache, approval,
                                  lambda model: factories.append(model), committed_checker=lambda *_: True,
                                  options=CollectionOptions(max_new_attempts=1)))
    assert factories == []

    approved = tuple(ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                                     cell.dataset_revision, cell.model, "other-provider")
                     for cell in {cell.fingerprint: cell for cell in pilot.cells}.values())
    aliased = CacheStore(str(tmp_path / "aliased.sqlite"), task=protocol.task,
                         preflight_fingerprint=pilot.checksum, approved_requests=approved,
                         ceiling=pilot.physical_count)
    try:
        with __import__("pytest").raises(ValueError, match="provider"):
            asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, aliased, approval,
                                      lambda model: factories.append(model), committed_checker=lambda *_: True,
                                      options=CollectionOptions(max_new_attempts=1)))
    finally:
        aliased.close()
    assert factories == []


def test_a_pilot_exports_malformed_responses_without_provider_details_and_replays_only_success(tmp_path):
    manifest, rows, protocol, source, pilot, approval, cache = _prepared_pilot(tmp_path)
    malformed = asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, cache, approval,
                                          lambda _model: _MalformedModel(), committed_checker=lambda *_: True,
                                          options=CollectionOptions(max_new_attempts=1)))
    assert any(row.status == "malformed" for row in malformed.observations)
    assert "provider" not in repr(cache.snapshot().physical_payloads)

    cache.close()
    manifest, rows, protocol, source, pilot, approval, cache = _prepared_pilot(tmp_path / "resume")
    calls = []
    first = asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, cache, approval,
                                      lambda _model: _CountingModel(calls), committed_checker=lambda *_: True,
                                      options=CollectionOptions(max_new_attempts=1)))
    replay = asyncio.run(collect_pilot(protocol, manifest, rows, source, pilot, cache, approval,
                                       lambda _model: (_ for _ in ()).throw(AssertionError("factory called on replay")),
                                       committed_checker=lambda *_: True,
                                       options=CollectionOptions(max_new_attempts=0)))
    assert first.new_attempts == 1 and calls == ["decision"]
    assert replay.new_attempts == 0 and any(row.cache_hit for row in replay.observations)
