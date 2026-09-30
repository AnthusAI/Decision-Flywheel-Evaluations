import asyncio
import hashlib
from dataclasses import replace

import pytest

from decision_flywheel.models import DecisionResult, DecisionTask

from .cache import ApprovedRequest, CacheStore, LogicalCell
from .collector import CollectionApproval, CollectionOptions
from .ordering import plan_ordering
from .ordering_collection import collect_ordering, validate_ordering_preregistration
from .ordering_test import _initial_inputs


class _CountingModel:
    def __init__(self, calls): self.calls = calls
    async def decide(self, _task, _target, _context):
        self.calls.append("decision")
        return DecisionResult("world", usage={"tokens": 3})


def _prepared_ordering(tmp_path, *, ceiling=None):
    tmp_path.mkdir(exist_ok=True)
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    ceiling = plan.physical_count if ceiling is None else ceiling
    preregistration = tmp_path / "ordering-preregistration.md"
    preregistration.write_text(
        f"{protocol.identity} {initial.checksum} {plan.initial_observations_sha256} {plan.checksum}",
        encoding="utf-8",
    )
    approval = CollectionApproval(
        str(preregistration), hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        protocol.identity, plan.checksum, ceiling, True,
    )
    approved = {cell.request.fingerprint: ApprovedRequest(
        cell.request.fingerprint, cell.request.task_fingerprint, cell.request.wire_fingerprint,
        cell.request.dataset_revision, cell.request.model,
    ) for cell in plan.cells}
    cache = CacheStore(str(tmp_path / "ordering.sqlite"), task=protocol.task,
                       preflight_fingerprint=plan.checksum,
                       approved_requests=tuple(approved.values()), ceiling=ceiling)
    return protocol, manifest, rows, initial, observations, plan, approval, cache


@pytest.fixture
def prepared_ordering():
    """Close every SQLite ledger created by an ordering collector spec."""
    opened = []

    def prepare(tmp_path, *, ceiling=None):
        prepared = _prepared_ordering(tmp_path, ceiling=ceiling)
        opened.append(prepared[-1])
        return prepared

    yield prepare
    for cache in reversed(opened):
        cache.close()


def _logical_cell(request):
    return LogicalCell(
        request.id, f"scoreboard:{request.model}:{request.selector}:{request.per_label}",
        request.draw_seed or 0, request.display_order, request.target_id,
    )


def test_an_ordering_collector_rejects_unconfirmed_uncommitted_tampered_and_subset_ledgers_before_factory(tmp_path, prepared_ordering):
    protocol, manifest, rows, initial, observations, plan, approval, cache = prepared_ordering(tmp_path)
    factories = []
    factory = lambda model: factories.append(model)
    for bad_approval, bad_plan, checker in (
        (replace(approval, confirmed=False), plan, lambda *_: True),
        (approval, plan, lambda *_: False),
        (approval, replace(plan, checksum="0" * 64), lambda *_: True),
    ):
        with pytest.raises(ValueError):
            asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, bad_plan,
                                         cache, bad_approval, factory, committed_checker=checker,
                                         options=CollectionOptions(max_new_attempts=1)))
    assert factories == []

    approved = tuple(ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                                     cell.wire_fingerprint, cell.dataset_revision,
                                     cell.model)
                     for cell in {cell.request.fingerprint: cell.request for cell in plan.cells}.values())
    subset = CacheStore(str(tmp_path / "subset.sqlite"), task=protocol.task,
                        preflight_fingerprint=plan.checksum, approved_requests=approved[:-1],
                        ceiling=plan.physical_count)
    try:
        with pytest.raises(ValueError, match="cache"):
            asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan,
                                         subset, approval, factory, committed_checker=lambda *_: True,
                                         options=CollectionOptions(max_new_attempts=1)))
    finally:
        subset.close()
    assert factories == []


def test_an_ordering_collector_uses_only_the_frozen_matrix_and_replays_success_without_another_model(tmp_path, prepared_ordering):
    protocol, manifest, rows, initial, observations, plan, approval, cache = prepared_ordering(tmp_path)
    calls = []
    first = asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, cache, approval,
                                         lambda _model: _CountingModel(calls), committed_checker=lambda *_: True,
                                         options=CollectionOptions(max_new_attempts=1)))
    assert calls == ["decision"] and first.new_attempts == first.physical_attempts == 1
    assert {row.request_id for row in first.observations} == {cell.request.id for cell in plan.cells}

    replay = asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, cache, approval,
                                          lambda _model: (_ for _ in ()).throw(AssertionError("replay built a model")),
                                          committed_checker=lambda *_: True,
                                          options=CollectionOptions(max_new_attempts=0)))
    assert replay.new_attempts == 0 and any(row.cache_hit for row in replay.observations)


def test_an_ordering_collector_recovers_the_lexically_first_reserved_request_with_one_new_attempt(tmp_path, prepared_ordering):
    _protocol, _manifest, _rows, _initial, _observations, base, _approval, _base_cache = prepared_ordering(tmp_path)
    ceiling = base.physical_count + 1
    protocol, manifest, rows, initial, observations, plan, approval, cache = prepared_ordering(
        tmp_path / "recovery", ceiling=ceiling,
    )
    request = min((cell.request for cell in plan.cells), key=lambda cell: cell.fingerprint)
    token = cache.reserve(request.fingerprint, _logical_cell(request))
    assert isinstance(token, str)
    calls = []
    result = asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, cache,
                                          approval, lambda _model: _CountingModel(calls),
                                          committed_checker=lambda *_: True,
                                          options=CollectionOptions(max_new_attempts=1, recover_uncertain=True,
                                                                    retry_failed=True, max_retries_per_request=1)))
    snapshot = cache.snapshot()
    assert (calls == ["decision"] and result.physical_attempts == 2
            and snapshot.physical_attempts[request.fingerprint] == 2
            and snapshot.physical_status[request.fingerprint] == "success")


def test_an_ordering_collector_retries_the_lexically_first_failed_request_and_refuses_an_exhausted_ceiling(tmp_path, prepared_ordering):
    _protocol, _manifest, _rows, _initial, _observations, base, _approval, _base_cache = prepared_ordering(tmp_path)
    protocol, manifest, rows, initial, observations, plan, approval, cache = prepared_ordering(
        tmp_path / "retry", ceiling=base.physical_count + 1,
    )
    request = min((cell.request for cell in plan.cells), key=lambda cell: cell.fingerprint)
    token = cache.reserve(request.fingerprint, _logical_cell(request))
    assert isinstance(token, str)
    cache.fail(token, "model-failure")
    calls = []
    asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, cache, approval,
                                 lambda _model: _CountingModel(calls), committed_checker=lambda *_: True,
                                 options=CollectionOptions(max_new_attempts=1, retry_failed=True,
                                                           max_retries_per_request=1)))
    snapshot = cache.snapshot()
    assert (calls == ["decision"] and snapshot.physical_attempts[request.fingerprint] == 2
            and snapshot.physical_status[request.fingerprint] == "success")

    exhausted_protocol, exhausted_manifest, exhausted_rows, exhausted_initial, exhausted_observations, exhausted_plan, exhausted_approval, exhausted = prepared_ordering(
        tmp_path / "exhausted", ceiling=1,
    )
    exhausted_request = min((cell.request for cell in exhausted_plan.cells), key=lambda cell: cell.fingerprint)
    exhausted_token = exhausted.reserve(exhausted_request.fingerprint, _logical_cell(exhausted_request))
    assert isinstance(exhausted_token, str)
    exhausted.fail(exhausted_token, "model-failure")
    with pytest.raises(ValueError, match="ceiling"):
        asyncio.run(collect_ordering(
            exhausted_protocol, exhausted_manifest, exhausted_rows, exhausted_initial,
            exhausted_observations, exhausted_plan, exhausted, exhausted_approval,
            lambda _model: (_ for _ in ()).throw(AssertionError("exhausted ledger built a model")),
            committed_checker=lambda *_: True,
            options=CollectionOptions(max_new_attempts=1, retry_failed=True, max_retries_per_request=1),
        ))


def test_an_ordering_preregistration_gate_binds_every_initial_and_derived_hash(tmp_path, prepared_ordering):
    protocol, manifest, rows, initial, observations, plan, approval, _cache = prepared_ordering(tmp_path)
    validate_ordering_preregistration(protocol, initial, plan, approval, lambda *_: True)
    path = tmp_path / "ordering-preregistration.md"
    path.write_text(f"{protocol.identity} {initial.checksum} {plan.checksum}", encoding="utf-8")
    missing_observation_hash = replace(approval, preregistration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="preregistration"):
        validate_ordering_preregistration(protocol, initial, plan, missing_observation_hash, lambda *_: True)


def test_an_ordering_collector_rejects_wrong_task_ledgers_and_existing_off_matrix_cells_before_factory(tmp_path, prepared_ordering):
    protocol, manifest, rows, initial, observations, plan, approval, cache = prepared_ordering(tmp_path)
    request = plan.cells[0].request
    token = cache.reserve(request.fingerprint, LogicalCell("outside-ordering-matrix", "scoreboard:outside", 0,
                                                           "canonical", request.target_id))
    assert isinstance(token, str)
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, cache, approval,
                                     lambda _model: (_ for _ in ()).throw(AssertionError("factory called")),
                                     committed_checker=lambda *_: True,
                                     options=CollectionOptions(max_new_attempts=1)))

    wrong_task = DecisionTask(protocol.task.name, protocol.task.labels, "Different frozen task instructions.")
    approved = tuple(ApprovedRequest(cell.fingerprint, wrong_task.fingerprint,
                                     cell.wire_fingerprint, cell.dataset_revision,
                                     cell.model)
                     for cell in {cell.request.fingerprint: cell.request for cell in plan.cells}.values())
    wrong = CacheStore(str(tmp_path / "wrong-task.sqlite"), task=wrong_task,
                       preflight_fingerprint=plan.checksum, approved_requests=approved,
                       ceiling=plan.physical_count)
    try:
        with pytest.raises(ValueError, match="cache"):
            asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, wrong, approval,
                                         lambda _model: (_ for _ in ()).throw(AssertionError("factory called")),
                                         committed_checker=lambda *_: True,
                                         options=CollectionOptions(max_new_attempts=1)))
    finally:
        wrong.close()


def test_an_ordering_collector_rejects_a_provider_alias_before_factory(tmp_path, prepared_ordering):
    protocol, manifest, rows, initial, observations, plan, approval, cache = prepared_ordering(tmp_path)
    cache.close()
    approved = tuple(ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                                     cell.wire_fingerprint, cell.dataset_revision, cell.model,
                                     provider_model_identity="jev-aliased-route")
                     for cell in {cell.request.fingerprint: cell.request for cell in plan.cells}.values())
    aliased = CacheStore(str(tmp_path / "aliased.sqlite"), task=protocol.task,
                         preflight_fingerprint=plan.checksum, approved_requests=approved,
                         ceiling=plan.physical_count)
    try:
        with pytest.raises(ValueError, match="provider"):
            asyncio.run(collect_ordering(protocol, manifest, rows, initial, observations, plan, aliased, approval,
                                         lambda _model: (_ for _ in ()).throw(AssertionError("factory called")),
                                         committed_checker=lambda *_: True,
                                         options=CollectionOptions(max_new_attempts=1)))
    finally:
        aliased.close()
