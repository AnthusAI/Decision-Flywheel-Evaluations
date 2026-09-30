import asyncio
import hashlib
from dataclasses import replace

import pytest

from decision_flywheel.models import DecisionResult

from .cli import report_document, run_collection, select_from_ledger
from .manifests import read_manifest, write_manifest
from .preflight import preflight
from .preflight_test import _protocol, _rows
from .serialization import (read_observations, read_preflight, read_protocol, read_rows_fixture,
                            write_observations, write_preflight, write_protocol, write_rows_fixture)


class _FakeModel:
    async def decide(self, task, _target, _context):
        return DecisionResult(task.labels[0], probabilities={label: float(label == task.labels[0]) for label in task.labels},
                              usage={"tokens": 1}, model="jev-1.13.0")


def test_run_refuses_an_unconfirmed_or_overbudget_request_before_the_factory(tmp_path):
    manifest, rows = _rows()
    protocol = replace(_protocol(manifest, rows), models=(_protocol(manifest, rows).models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"{protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    calls = []

    with pytest.raises(ValueError, match="--confirm"):
        asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "ledger.sqlite"),
                                   str(preregistration), plan.new_request_count, 1, False,
                                   provider_model="jev-1.13.0", engine_factory=lambda _: calls.append("factory"),
                                   committed_checker=lambda *_: True))
    with pytest.raises(ValueError, match="approved attempt ceiling"):
        asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "ledger.sqlite"),
                                   str(preregistration), 1, 2, True,
                                   provider_model="jev-1.13.0", engine_factory=lambda _: calls.append("factory"),
                                   committed_checker=lambda *_: True))
    asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "resume.sqlite"),
                               str(preregistration), 2, 1, True, provider_model="jev-1.13.0",
                               engine_factory=lambda _: _FakeModel(), committed_checker=lambda *_: True))
    with pytest.raises(ValueError, match="remaining approved budget"):
        asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "resume.sqlite"),
                                   str(preregistration), 2, 2, True, provider_model="jev-1.13.0",
                                   engine_factory=lambda _: calls.append("factory"), committed_checker=lambda *_: True))
    assert calls == []


def test_run_binds_the_exact_jev_adapter_identity_before_an_injected_factory_is_called(tmp_path):
    manifest, rows = _rows()
    protocol = replace(_protocol(manifest, rows), models=(_protocol(manifest, rows).models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"{protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    seen = []

    result = asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "ledger.sqlite"),
                                        str(preregistration), plan.new_request_count, 1, True,
                                        provider_model="jev-1.13.0", engine_factory=lambda identity: (seen.append(identity) or _FakeModel()),
                                        committed_checker=lambda *_: True))

    assert result.new_attempts == 1
    assert seen == ["jev:jev-1.13.0"]


def test_preflight_file_is_not_accepted_as_a_source_of_dataset_text(tmp_path):
    manifest, rows = _rows()
    path = tmp_path / "preflight.json"
    write_preflight(path, preflight(_protocol(manifest, rows), manifest=manifest, rows=rows))
    assert rows[0].text not in path.read_text(encoding="utf-8")


def test_sanitized_collection_rows_round_trip_with_mapping_proxies_and_report_incompleteness(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows)
    protocol = replace(base, models=(base.models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"{protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    result = asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "ledger.sqlite"),
                                        str(preregistration), plan.new_request_count, 1, True,
                                        provider_model="jev-1.13.0", engine_factory=lambda _: _FakeModel(),
                                        committed_checker=lambda *_: True))
    observations = tmp_path / "observations.json"
    write_observations(observations, result.observations)

    restored = read_observations(observations)
    document = report_document(protocol, manifest, plan, observations)
    assert restored == result.observations
    assert document["complete"] is False
    assert document["finding_status"] == "incomplete: not a study finding"
    assert "candidate" not in observations.read_text(encoding="utf-8")


def test_a_synthetic_native_workflow_serializes_preflights_resumes_a_fake_ledger_and_reports_offline(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows)
    protocol = replace(base, models=(base.models[0],))
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    rows_path = tmp_path / "study.fixture.json"
    preflight_path = tmp_path / "preflight.json"
    observations_path = tmp_path / "observations.json"
    write_manifest(manifest_path, manifest)
    write_protocol(protocol_path, protocol)
    write_rows_fixture(rows_path, rows)
    restored_protocol, restored_manifest, restored_rows = read_protocol(protocol_path), read_manifest(manifest_path), read_rows_fixture(rows_path)
    plan = preflight(restored_protocol, manifest=restored_manifest, rows=restored_rows)
    write_preflight(preflight_path, plan)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"{restored_protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    result = asyncio.run(run_collection(restored_protocol, restored_manifest, restored_rows, read_preflight(preflight_path),
                                        str(tmp_path / "ledger.sqlite"), str(preregistration), plan.new_request_count, 1, True,
                                        provider_model="jev-1.13.0", engine_factory=lambda _: _FakeModel(),
                                        committed_checker=lambda *_: True))
    write_observations(observations_path, result.observations)
    report = report_document(restored_protocol, restored_manifest, read_preflight(preflight_path), observations_path)

    assert report["complete"] is False
    assert result.new_attempts == 1
    assert rows[0].text not in preflight_path.read_text(encoding="utf-8")


def test_a_complete_native_development_selection_and_scoreboard_pipeline_never_constructs_a_second_provider_for_selection(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows, artifact=False)
    protocol = replace(base, models=(base.models[0],))
    optimization = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    optimization_preregistration = tmp_path / "optimization-preregistration.md"
    optimization_preregistration.write_text(f"{protocol.identity}\n{optimization.checksum}\n", encoding="utf-8")
    development = asyncio.run(run_collection(protocol, manifest, rows, optimization, str(tmp_path / "development.sqlite"),
                                             str(optimization_preregistration), optimization.new_request_count,
                                             optimization.new_request_count, True, provider_model="jev-1.13.0",
                                             engine_factory=lambda _: _FakeModel(), committed_checker=lambda *_: True))
    derived = asyncio.run(select_from_ledger(protocol, manifest, rows, optimization, str(tmp_path / "development.sqlite"),
                                             str(optimization_preregistration), optimization.new_request_count,
                                             model_identity=protocol.models[0].semantic_identity,
                                             provider_model="jev-1.13.0", artifact_reference="local-development-evidence",
                                             committed_checker=lambda *_: True))
    scoreboard = preflight(derived, manifest=manifest, rows=rows)
    scoreboard_preregistration = tmp_path / "scoreboard-preregistration.md"
    scoreboard_preregistration.write_text(f"{derived.identity}\n{scoreboard.checksum}\n", encoding="utf-8")
    result = asyncio.run(run_collection(derived, manifest, rows, scoreboard, str(tmp_path / "scoreboard.sqlite"),
                                        str(scoreboard_preregistration), scoreboard.new_request_count,
                                        scoreboard.new_request_count, True, provider_model="jev-1.13.0",
                                        engine_factory=lambda _: _FakeModel(), committed_checker=lambda *_: True))
    observations = tmp_path / "scoreboard-observations.json"
    write_observations(observations, result.observations)

    document = report_document(derived, manifest, scoreboard, observations)
    assert development.complete and result.complete
    assert derived.selector_search_artifact is not None
    assert document["complete"] is True
    assert document["study"]["logical_cells"] == len(scoreboard.cells)
    assert document["study"]["physical"]["requests"] == scoreboard.new_request_count
    assert document["study"]["physical"]["attempts"] == scoreboard.new_request_count
    assert document["study"]["physical"]["numeric_usage"] == {"tokens": float(scoreboard.new_request_count)}
    for name in ("selection", "size"):
        effect = document["paired_effects"][name]
        assert effect == {"available": True, "effect": 0.0, "lower": 0.0, "upper": 0.0,
                          "confidence_level": 0.95,
                          "interval_scope": "nominal per-contrast, not multiplicity adjusted"}
    assert document["holm"]["available"] is False
    assert "valid p-values" in document["holm"]["reason"]


def test_report_rejects_tampered_truth_condition_model_protocol_or_preflight_before_metrics(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows)
    protocol = replace(base, models=(base.models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"{protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    result = asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "ledger.sqlite"),
                                        str(preregistration), plan.new_request_count, 1, True,
                                        provider_model="jev-1.13.0", engine_factory=lambda _: _FakeModel(),
                                        committed_checker=lambda *_: True))
    observations = tmp_path / "observations.json"
    write_observations(observations, result.observations)
    original = __import__("json").loads(observations.read_text())
    for field, value, error in (("true_label", "Sports", "ground truth"),
                                ("condition", "scoreboard:wrong:zero:0", "logical or model"),
                                ("model_id", "jev:wrong", "logical or model")):
        document = __import__("copy").deepcopy(original)
        document["observations"][0][field] = value
        observations.write_text(__import__("json").dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            report_document(protocol, manifest, plan, observations)
    observations.write_text(__import__("json").dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        report_document(replace(protocol, name="other-study"), manifest, plan, observations)
    with pytest.raises(ValueError, match="checksum"):
        report_document(protocol, manifest, replace(plan, checksum="0" * 64), observations)


def test_retry_is_opt_in_and_a_resumed_success_replays_without_constructing_another_engine(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows)
    protocol = replace(base, models=(base.models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text(f"{protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    calls = []
    class FailsOnce:
        async def decide(self, task, _target, _context):
            calls.append("decision")
            if len(calls) == 1:
                raise RuntimeError("synthetic failure")
            return await _FakeModel().decide(task, _target, _context)
    ledger = str(tmp_path / "ledger.sqlite")
    asyncio.run(run_collection(protocol, manifest, rows, plan, ledger, str(preregistration), 2, 1, True,
                               provider_model="jev-1.13.0", engine_factory=lambda _: FailsOnce(),
                               committed_checker=lambda *_: True))
    resumed = asyncio.run(run_collection(protocol, manifest, rows, plan, ledger, str(preregistration), 2, 1, True,
                                         provider_model="jev-1.13.0", engine_factory=lambda _: FailsOnce(),
                                         retry_failed=True, max_retries_per_request=1,
                                         committed_checker=lambda *_: True))
    replay_factories = []
    asyncio.run(run_collection(protocol, manifest, rows, plan, ledger, str(preregistration), 2, 0, True,
                               provider_model="jev-1.13.0", engine_factory=lambda _: replay_factories.append("factory"),
                               retry_failed=True, max_retries_per_request=1,
                               committed_checker=lambda *_: True))
    assert resumed.new_attempts == 1 and calls == ["decision", "decision"]
    assert replay_factories == []


def test_optimization_retries_cannot_expand_the_declared_native_model_call_budget_before_factory(tmp_path):
    manifest, rows = _rows()
    base = _protocol(manifest, rows, artifact=False)
    protocol = replace(base, models=(base.models[0],))
    plan = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    preregistration = tmp_path / "optimization-preregistration.md"
    preregistration.write_text(f"{protocol.identity}\n{plan.checksum}\n", encoding="utf-8")
    factories = []

    with pytest.raises(ValueError, match="declared native optimization budget"):
        asyncio.run(run_collection(protocol, manifest, rows, plan, str(tmp_path / "ledger.sqlite"),
                                   str(preregistration), protocol.optimization.max_model_calls + 1, 1, True,
                                   provider_model="jev-1.13.0", engine_factory=lambda _: factories.append("factory"),
                                   retry_failed=True, max_retries_per_request=1,
                                   committed_checker=lambda *_: True))
    assert factories == []
