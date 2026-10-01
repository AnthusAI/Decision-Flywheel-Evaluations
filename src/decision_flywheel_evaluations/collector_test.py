import asyncio
import hashlib
import pytest
from dataclasses import replace

from decision_flywheel.models import DecisionResult
from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration

from .cache import ApprovedRequest, CacheStore, LogicalCell
from .collector import CollectionApproval, CollectionOptions, collect
from .preflight import preflight
from .preflight_test import _protocol, _rows
from .reporting import report_study
from .serialization import read_observations, write_observations


class _CountingModel:
    def __init__(self, calls): self.calls = calls
    async def decide(self, _task, _target, _context):
        self.calls.append("decision")
        return DecisionResult("world", usage={"tokens": 3}, latency_ms=4)


class _CrashingModel:
    async def decide(self, _task, _target, _context):
        raise RuntimeError("provider secret that must not be exported")


def _prepared(tmp_path, *, confirmed=True, ceiling=1):
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows)
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"frozen plan {protocol.identity} {plan.checksum}", encoding="utf-8")
    ceiling = plan.new_request_count if ceiling is None else ceiling
    approval = CollectionApproval(
        str(preregistration), hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        protocol.identity, plan.checksum, ceiling, confirmed,
    )
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                cell.wire_fingerprint, cell.dataset_revision, cell.model) for cell in plan.cells}
    cache = CacheStore(str(tmp_path / "ledger.sqlite"), task=protocol.task,
                       preflight_fingerprint=plan.checksum,
                       approved_requests=tuple(approved.values()), ceiling=ceiling)
    return manifest, rows, protocol, plan, approval, cache


def _prepared_jev_only(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows)
    protocol = replace(base, models=(base.models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"frozen plan {protocol.identity} {plan.checksum}", encoding="utf-8")
    approval = CollectionApproval(
        str(preregistration), hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        protocol.identity, plan.checksum, plan.new_request_count, True,
    )
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                cell.wire_fingerprint, cell.dataset_revision, cell.model) for cell in plan.cells}
    cache = CacheStore(str(tmp_path / "ledger.sqlite"), task=protocol.task,
                       preflight_fingerprint=plan.checksum,
                       approved_requests=tuple(approved.values()), ceiling=plan.new_request_count)
    return manifest, rows, protocol, plan, approval, cache


def test_collector_rejects_confirmation_preregistration_preflight_and_plan_mismatches_before_factory(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=False)
    factories = []
    factory = lambda _model: factories.append("factory")
    for bad_approval, bad_plan, checker in (
        (approval, plan, lambda *_: True),
        (replace(approval, confirmed=True, preregistration_sha256="0" * 64), plan, lambda *_: True),
        (replace(approval, confirmed=True, preflight_checksum="0" * 64), plan, lambda *_: True),
        (replace(approval, confirmed=True), replace(plan, cells=plan.cells[:-1]), lambda *_: True),
        (replace(approval, confirmed=True), plan, lambda *_: False),
    ):
        try:
            asyncio.run(collect(protocol, manifest, rows, bad_plan, cache, bad_approval,
                                factory, committed_checker=checker, options=CollectionOptions(max_new_attempts=1)))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid collection gate was accepted")
    assert factories == []


def test_collector_uses_only_approved_bounded_requests_and_exports_missing_cells(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True)
    calls = []
    result = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda model: (calls.append(model) or _CountingModel(calls)),
                                 committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1)))
    assert calls.count("decision") == 1 and len(calls) == 2
    assert len(result.observations) == len(plan.cells)
    assert result.physical_attempts == 1
    assert any(row.status == "completed" for row in result.observations)
    assert any(row.status == "missing" for row in result.observations)
    assert {row.physical_request_id for row in result.observations} <= {cell.fingerprint for cell in plan.cells}


def test_optimization_collection_never_needs_heldout_source_rows(tmp_path):
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows, artifact=False)
    plan = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    source_rows = tuple(row for row in rows if row.id not in {record.id for record in manifest.scoreboard})
    preregistration = tmp_path / "optimization-preregistration.md"
    preregistration.write_text(f"{protocol.identity} {plan.checksum}", encoding="utf-8")
    approval = CollectionApproval(
        str(preregistration), hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        protocol.identity, plan.checksum, plan.new_request_count, True,
    )
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                cell.wire_fingerprint, cell.dataset_revision, cell.model) for cell in plan.cells}
    cache = CacheStore(str(tmp_path / "optimization.sqlite"), task=protocol.task,
                       preflight_fingerprint=plan.checksum,
                       approved_requests=tuple(approved.values()), ceiling=plan.new_request_count)

    result = asyncio.run(collect(protocol, manifest, source_rows, plan, cache, approval,
                                 lambda _model: _CountingModel([]), committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1)))

    assert any(row.status == "completed" for row in result.observations)


def test_collector_replays_success_without_building_an_engine_after_the_ceiling_is_spent(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True)
    first_calls, second_calls = [], []
    asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                        lambda model: (first_calls.append(model) or _CountingModel(first_calls)),
                        committed_checker=lambda *_: True, options=CollectionOptions(max_new_attempts=1)))
    replay = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda model: (second_calls.append(model) or _CountingModel(second_calls)),
                                 committed_checker=lambda *_: True, options=CollectionOptions(max_new_attempts=1)))
    assert first_calls.count("decision") == 1 and second_calls == []
    assert replay.physical_attempts == 1


def test_collector_uses_only_one_final_full_snapshot_while_scheduling_each_request(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True, ceiling=None)
    original_snapshot = cache.snapshot
    snapshots = []

    def counted_snapshot():
        snapshots.append("snapshot")
        return original_snapshot()

    cache.snapshot = counted_snapshot
    result = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda _model: _CountingModel([]), committed_checker=lambda *_: True))

    assert result.complete
    assert snapshots == ["snapshot"]


def test_a_mixed_cached_and_new_collection_marks_only_the_preexisting_physical_request_as_a_cache_hit(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True, ceiling=None)
    cached, new = plan.cells[:2]
    token = cache.reserve(cached.fingerprint, LogicalCell(
        cached.id, f"{cached.phase}:{cached.model}:{cached.selector}:{cached.per_label}",
        cached.draw_seed or 0, cached.display_order, cached.target_id,
    ))
    assert isinstance(token, str)
    cache.complete(token, {"choice": "World", "model": "jev:jev-1.13.0"})

    result = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda _model: _CountingModel([]), committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1)))

    hits = {row.physical_request_id: row.cache_hit for row in result.observations}
    assert hits[cached.fingerprint] is True
    assert hits[new.fingerprint] is False


def test_failed_attempts_consume_the_hard_ceiling_and_retry_never_constructs_another_engine(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True, ceiling=1)
    asyncio.run(collect(protocol, manifest, rows, plan, cache, approval, lambda _model: _CrashingModel(),
                        committed_checker=lambda *_: True, options=CollectionOptions(max_new_attempts=1)))
    factories = []
    retry = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                lambda model: (factories.append(model) or _CountingModel(factories)),
                                committed_checker=lambda *_: True,
                                options=CollectionOptions(max_new_attempts=1, retry_failed=True,
                                                          max_retries_per_request=1)))
    assert factories == [] and retry.physical_attempts == 1
    assert any(row.status == "failed" for row in retry.observations)


def test_full_matrix_collection_replays_without_calls_and_reports_deduplicated_physical_usage(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True, ceiling=None)
    calls = []
    first = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                lambda model: (calls.append(model) or _CountingModel(calls)),
                                committed_checker=lambda *_: True))
    assert calls.count("decision") == plan.new_request_count and all(row.status == "completed" for row in first.observations)
    study = report_study(first.observations, protocol.task.labels,
                         expected_request_ids=tuple(cell.id for cell in plan.cells))
    assert study.physical.requests == plan.new_request_count
    assert study.physical.numeric_usage == {"tokens": 3.0 * plan.new_request_count}
    replay_calls = []
    second = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda model: (replay_calls.append(model) or _CountingModel(replay_calls)),
                                 committed_checker=lambda *_: True))
    assert replay_calls == [] and all(row.status == "completed" for row in second.observations)


def test_collector_accepts_a_real_jev_adapter_provider_identity(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True, ceiling=None)
    class Response:
        model = "jev-1.13.0"; usage = {"tokens": 3}
        answers = {"ag-news": {"choice": "World"}}
    class Client:
        def system_one(self, **_kwargs): return Response()
    calls = []
    def factory(model):
        calls.append(model)
        if model == "jev:jev-1.13.0":
            return JevAdapter(Client(), configuration=JevConfiguration(model="jev-1.13.0"))
        return _CountingModel(calls)
    result = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval, factory,
                                 committed_checker=lambda *_: True))
    assert any(row.model_id == "jev:jev-1.13.0" and row.status == "completed" for row in result.observations)


def test_fake_jev_confidence_is_cached_exported_and_never_derived_from_probabilities(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared_jev_only(tmp_path)
    class Response:
        model = "jev-1.13.0"; usage = {"tokens": 3}
        answers = {"ag-news": {"choice": "World", "confidence": .37,
                                 "probabilities": {"World": .9, "Sports": .04,
                                                   "Business": .03, "Sci/Tech": .03}}}
    class Client:
        calls = 0
        def system_one(self, **_kwargs):
            self.calls += 1
            return Response()
    client = Client()
    result = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda _model: JevAdapter(client, configuration=JevConfiguration(model="jev-1.13.0")),
                                 committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1)))
    completed = next(row for row in result.observations if row.status == "completed")
    assert client.calls == 1 and completed.confidence == .37
    assert max(completed.probabilities.values()) == .9
    assert all(row.confidence is None for row in result.observations if row.status == "missing")
    path = tmp_path / "observations.json"
    write_observations(path, result.observations)
    restored = read_observations(path)
    assert next(row for row in restored if row.status == "completed").confidence == .37


def test_fake_jev_invalid_confidence_is_a_malformed_cached_response(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared_jev_only(tmp_path)
    class Response:
        model = "jev-1.13.0"; usage = {"tokens": 3}
        answers = {"ag-news": {"choice": "World", "confidence": 1.1}}
    class Client:
        def system_one(self, **_kwargs): return Response()
    result = asyncio.run(collect(protocol, manifest, rows, plan, cache, approval,
                                 lambda _model: JevAdapter(Client(), configuration=JevConfiguration(model="jev-1.13.0")),
                                 committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1)))
    assert any(row.status == "malformed" for row in result.observations)
    assert all(row.confidence is None for row in result.observations)


def test_a_provider_that_cannot_be_constructed_aborts_before_any_attempt_is_reserved(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared_jev_only(tmp_path)
    factory_calls = []
    def factory(model):
        factory_calls.append(model)
        raise ValueError("credential-like factory detail")
    with pytest.raises(RuntimeError) as error:
        asyncio.run(collect(protocol, manifest, rows, plan, cache, approval, factory,
                            committed_checker=lambda *_: True,
                            options=CollectionOptions(max_new_attempts=1)))
    assert len(factory_calls) == 1
    assert cache.attempts_used == 0
    assert "ValueError" in str(error.value)
    assert "credential-like factory detail" not in str(error.value)
    assert "credential-like factory detail" not in repr(cache.snapshot().physical_payloads)


class _BillingModel:
    async def decide(self, _task, _target, _context):
        error = RuntimeError("402 account detail that must not be exported")
        error.status = 402
        raise error


def test_a_billing_or_auth_error_aborts_the_run_after_the_first_attempt(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, ceiling=None)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(collect(protocol, manifest, rows, plan, cache, approval, lambda _model: _BillingModel(),
                            committed_checker=lambda *_: True,
                            options=CollectionOptions(max_new_attempts=plan.new_request_count)))
    assert cache.attempts_used == 1 and "402" in str(error.value)
    assert "account detail" not in str(error.value)


def test_consecutive_provider_failures_stop_the_run_before_the_budget_is_spent(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, ceiling=None)
    assert plan.new_request_count > 3
    with pytest.raises(RuntimeError, match="consecutive"):
        asyncio.run(collect(protocol, manifest, rows, plan, cache, approval, lambda _model: _CrashingModel(),
                            committed_checker=lambda *_: True,
                            options=CollectionOptions(max_new_attempts=plan.new_request_count,
                                                      max_consecutive_failures=3)))
    assert cache.attempts_used == 3


def test_baseexception_crash_persists_reservation_for_reopen_recovery_and_one_retry(tmp_path):
    manifest, rows, protocol, plan, approval, cache = _prepared(tmp_path, confirmed=True, ceiling=2)
    class Crash(BaseException): pass
    class CrashModel:
        async def decide(self, *_args): raise Crash()
    try:
        asyncio.run(collect(protocol, manifest, rows, plan, cache, approval, lambda _model: CrashModel(),
                            committed_checker=lambda *_: True, options=CollectionOptions(max_new_attempts=1)))
    except Crash:
        pass
    cache.close()
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                cell.dataset_revision, cell.model) for cell in plan.cells}
    reopened = CacheStore(str(tmp_path / "ledger.sqlite"), task=protocol.task, preflight_fingerprint=plan.checksum,
                          approved_requests=tuple(approved.values()), ceiling=2)
    calls = []
    result = asyncio.run(collect(protocol, manifest, rows, plan, reopened, approval,
                                 lambda model: (calls.append(model) or _CountingModel(calls)),
                                 committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1, recover_uncertain=True,
                                                           retry_failed=True, max_retries_per_request=1)))
    assert calls.count("decision") == 1 and result.physical_attempts == 2
