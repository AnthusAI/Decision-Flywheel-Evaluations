from decision_flywheel.models import DecisionTask
from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration
from decision_flywheel.adapters.kev import KevAdapter, KevConfiguration
import asyncio
import pytest

from .cache import ApprovedRequest, CacheStore, LogicalCell


def _task():
    return DecisionTask("sentiment", ("yes", "no"), "Choose the label for text.")


def _approved(task=None, request_id="a" * 64, *, state="b" * 64, model="fake:1"):
    task = task or _task()
    return ApprovedRequest(request_id, task.fingerprint, state, "c" * 40, model)


def _cache(tmp_path, *, ceiling=2, requests=None):
    task = _task()
    return CacheStore(
        str(tmp_path / "ledger.sqlite"),
        task=task,
        preflight_fingerprint="d" * 64,
        approved_requests=tuple(requests or (_approved(task),)),
        ceiling=ceiling,
    )


def _cell():
    return LogicalCell("scoreboard:fake:1:zero:target-1", "zero", 0, "canonical", "target-1")


def test_a_successful_replay_does_not_spend_another_attempt_or_duplicate_physical_usage(tmp_path):
    cache = _cache(tmp_path, ceiling=1)
    token = cache.reserve("a" * 64, _cell())
    saved = cache.complete(token, {"choice": " YES! ", "usage": {"tokens": 2}})

    assert cache.reserve("a" * 64, _cell()) == saved
    assert cache.attempts_used == 1
    snapshot = cache.snapshot()
    assert snapshot.physical_usage == {"a" * 64: {"tokens": 2.0}}
    assert snapshot.logical_mapping == {_cell().cell_id: "a" * 64}


def test_the_immutable_manifest_task_and_preflight_must_match_exactly_on_reopen(tmp_path):
    cache = _cache(tmp_path)
    cache.close()
    task = _task()
    approved = _approved(task)
    reopened = CacheStore(str(tmp_path / "ledger.sqlite"), task=task,
                          preflight_fingerprint="d" * 64, approved_requests=(approved,), ceiling=2)
    reopened.close()

    with pytest.raises(ValueError, match="frozen cache configuration"):
        CacheStore(str(tmp_path / "ledger.sqlite"), task=task,
                   preflight_fingerprint="e" * 64, approved_requests=(approved,), ceiling=2)
    changed_task = DecisionTask("sentiment", ("yes", "no"), "Different instructions.")
    with pytest.raises(ValueError, match="frozen cache configuration"):
        CacheStore(str(tmp_path / "ledger.sqlite"), task=changed_task,
                   preflight_fingerprint="d" * 64,
                   approved_requests=(_approved(changed_task),), ceiling=2)


def test_interleaved_connections_reject_a_stale_token_after_recovery_and_retry(tmp_path):
    first = _cache(tmp_path, ceiling=2)
    second = _cache(tmp_path, ceiling=2)
    request_id = "a" * 64
    old = first.reserve(request_id, _cell())
    with pytest.raises(ValueError, match="already reserved"):
        second.reserve(request_id, _cell())

    first.recover_uncertain(old)
    new = second.reserve(request_id, _cell())
    with pytest.raises(ValueError, match="active reservation token"):
        first.complete(old, {"choice": "yes"})
    second.complete(new, {"choice": "yes"})
    assert second.attempts_used == 2


def test_recovery_records_the_crashed_attempt_and_a_retry_spends_another(tmp_path):
    cache = _cache(tmp_path, ceiling=2)
    first = cache.reserve("a" * 64, _cell())
    cache.recover_uncertain(first)
    second = cache.reserve("a" * 64, _cell())
    cache.complete(second, {"choice": "yes"})

    assert cache.attempts_used == 2
    assert cache.reserve("a" * 64, _cell())["choice"] == "yes"


@pytest.mark.parametrize("response", [
    {"answer": {"choice": "yes", "probabilities": {"yes": 1.0}}},
    {"choice": "yes", "usage": {"tokens": -1}},
    {"choice": "yes", "confidence": 1.1},
    {"choice": "yes", "latency_ms": -1},
])
def test_invalid_response_shapes_and_negative_usage_are_failures_not_cached_success(tmp_path, response):
    cache = _cache(tmp_path)
    token = cache.reserve("a" * 64, _cell())

    result = cache.complete(token, response)

    assert result == {"status": "failed", "category": "malformed-response"}
    assert cache.snapshot().physical_responses == {}
    retry = cache.reserve("a" * 64, _cell())
    cache.fail(retry, "model-failure")


def test_payload_never_persists_provider_text_or_an_unapproved_model_identifier(tmp_path):
    cache = _cache(tmp_path)
    token = cache.reserve("a" * 64, _cell())
    result = cache.complete(token, {
        "choice": "yes",
        "provider_trace": "TOP-SECRET",
        "usage": {"tokens": 2, "provider_key": "TOP-SECRET"},
        "model": "fake:1",
    })

    assert result == {"status": "success", "choice": "yes", "model": "fake:1", "reported_model": "fake:1", "usage": {"tokens": 2.0}}
    serialized = (tmp_path / "ledger.sqlite").read_bytes()
    assert b"TOP-SECRET" not in serialized


def test_logical_cell_collision_cannot_change_physical_provenance(tmp_path):
    task = _task()
    first = _approved(task)
    second = _approved(task, "e" * 64, state="f" * 64)
    cache = _cache(tmp_path, requests=(first, second))
    cache.reserve(first.request_id, _cell())
    with pytest.raises(ValueError, match="logical cell identity changed"):
        cache.reserve(second.request_id, _cell())


def test_approved_requests_validate_all_physical_provenance_fields(tmp_path):
    task = _task()
    with pytest.raises(ValueError, match="dataset revision"):
        CacheStore(str(tmp_path / "ledger.sqlite"), task=task, preflight_fingerprint="d" * 64,
                   approved_requests=(ApprovedRequest("a" * 64, task.fingerprint, "b" * 64, "bad", "fake:1"),), ceiling=1)


def test_snapshot_exports_physical_status_attempts_and_safe_failure_payload_for_collection(tmp_path):
    cache = _cache(tmp_path, ceiling=2)
    token = cache.reserve("a" * 64, _cell())
    cache.fail(token, "model-failure")

    snapshot = cache.snapshot()

    assert snapshot.physical_status == {"a" * 64: "failed"}
    assert snapshot.physical_attempts == {"a" * 64: 1}
    assert snapshot.physical_payloads == {"a" * 64: {"status": "failed", "category": "model-failure"}}


def test_cache_accepts_exact_provider_model_names_from_jev_and_kev_adapters(tmp_path):
    class JevResponse:
        model = "jev-1.13.0"
        usage = {"tokens": 2}
        answers = {"sentiment": {"choice": "yes"}}
    class JevClient:
        def system_one(self, **_kwargs): return JevResponse()
    class Response:
        status_code = 200
        def json(self): return {"answers": {"sentiment": {"choice": "yes"}}, "model": "kev-4b"}
    class Transport:
        async def post(self, *_args, **_kwargs): return Response()
    task = _task()
    for request_id, model, result in (
        ("a" * 64, "jev:jev-1.13.0", asyncio.run(JevAdapter(JevClient(), configuration=JevConfiguration(model="jev-1.13.0")).decide(task, __import__("decision_flywheel.models", fromlist=["Item"]).Item("x", {"text": "x"}), []))),
        ("e" * 64, "kev:kev-4b@abc", asyncio.run(KevAdapter(transport=Transport(), configuration=KevConfiguration(model="kev-4b", revision="abc")).decide(task, __import__("decision_flywheel.models", fromlist=["Item"]).Item("x", {"text": "x"}), []))),
    ):
        cache = CacheStore(str(tmp_path / f"{request_id}.sqlite"), task=task, preflight_fingerprint="d" * 64,
                           approved_requests=(ApprovedRequest(request_id, task.fingerprint, "b" * 64, "c" * 40, model),), ceiling=1)
        token = cache.reserve(request_id, _cell())
        payload = {"choice": result.label, "model": result.model}
        if result.usage is not None: payload["usage"] = result.usage
        saved = cache.complete(token, payload)
        assert saved["status"] == "success"
