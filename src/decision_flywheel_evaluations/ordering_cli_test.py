import json
import subprocess
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import ordering_cli
from .manifests import write_manifest
from .ordering_io import read_ordering, write_ordering
from .ordering_test import _initial_inputs
from .collector import CollectionOptions
from decision_flywheel.models import DecisionResult
from .serialization import write_observations, write_preflight, write_protocol, write_rows_fixture


def _inputs(tmp_path):
    protocol, manifest, rows, initial, observations, _ = _initial_inputs()
    paths = {name: tmp_path / filename for name, filename in (
        ("protocol", "protocol.json"), ("manifest", "manifest.json"),
        ("rows", "rows.fixture.json"), ("initial-preflight", "initial.preflight.json"),
        ("initial-observations", "initial.observations.json"), ("output", "ordering.json"),
    )}
    write_protocol(paths["protocol"], protocol)
    write_manifest(paths["manifest"], manifest)
    write_rows_fixture(paths["rows"], rows)
    write_preflight(paths["initial-preflight"], initial)
    write_observations(paths["initial-observations"], observations)
    arguments = ["preflight"]
    for name, path in paths.items():
        arguments.extend((f"--{name}", str(path)))
    arguments.extend(("--initial-result-reference", "results/initial.json",
                      "--shuffle-seed", "1", "--shuffle-seed", "2"))
    return arguments, paths, rows


def test_ordering_preflight_writes_only_a_bound_plan_without_a_provider_or_source_text(tmp_path, capsys):
    arguments, paths, rows = _inputs(tmp_path)
    assert ordering_cli.main(arguments) == 0
    plan = read_ordering(paths["output"])
    message = json.loads(capsys.readouterr().out)
    assert message["status"] == "unapproved"
    assert message["logical_requests"] == plan.logical_count == 644
    assert message["physical_requests"] == plan.physical_count
    content = paths["output"].read_text()
    assert all(row.text not in content for row in rows)
    assert '"text":' not in content


@pytest.mark.parametrize("handler", ("preflight", "run", "report"))
def test_ordering_commands_require_permission_before_overwriting_existing_artifacts(tmp_path, handler):
    output = tmp_path / "existing.json"
    output.write_text("previous evidence", encoding="utf-8")
    args = SimpleNamespace(output=str(output), overwrite=False, confirm=True)
    with pytest.raises(ValueError, match="overwrite"):
        getattr(ordering_cli, f"_{handler}")(args)
    assert output.read_text() == "previous evidence"


def test_ordering_preflight_rejects_incomplete_initial_results_without_writing_a_plan(tmp_path, capsys):
    arguments, paths, _ = _inputs(tmp_path)
    document = json.loads(paths["initial-observations"].read_text())
    document["observations"].pop()
    paths["initial-observations"].write_text(json.dumps(document))
    with pytest.raises(SystemExit) as error:
        ordering_cli.main(arguments)
    assert error.value.code == 2
    assert not paths["output"].exists()
    assert "ValueError" in capsys.readouterr().err


def test_make_ordering_preflight_preserves_the_initial_anchor_and_explicit_permutation_seeds():
    root = Path(__file__).parents[2]
    completed = subprocess.run([
        "make", "-n", "ordering-preflight", "PROTOCOL=protocol.json", "MANIFEST=manifest.json",
        "PREFLIGHT=initial.preflight.json", "INITIAL_OBSERVATIONS=initial.observations.json",
        "RESULT_REFERENCE=results/initial.json", "ROWS=rows.fixture.json", "OUTPUT=ordering.json",
        "ORDER_SEEDS=1 2 3 4 5",
    ], cwd=root, text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    assert '--initial-preflight "initial.preflight.json"' in completed.stdout
    assert '--initial-observations "initial.observations.json"' in completed.stdout
    assert '--initial-result-reference "results/initial.json"' in completed.stdout
    assert all(f'--shuffle-seed "{seed}"' in completed.stdout for seed in range(1, 6))


def test_ordering_report_reproduces_local_metrics_without_reading_source_rows(tmp_path):
    from .ordering_reporting_test import _inputs as report_inputs
    protocol, manifest, plan, observations = report_inputs()
    paths = {name: tmp_path / f"{name}.json" for name in ("protocol", "manifest", "ordering", "observations", "output")}
    write_protocol(paths["protocol"], protocol)
    write_manifest(paths["manifest"], manifest)
    write_ordering(paths["ordering"], plan)
    write_observations(paths["observations"], observations)
    arguments = ["report", "--resamples", "3"]
    for name, path in paths.items():
        arguments.extend((f"--{name}", str(path)))

    assert ordering_cli.main(arguments) == 0
    report = json.loads(paths["output"].read_text())
    assert report["study"]["logical_cells"] == plan.logical_count
    assert report["study"]["physical"]["requests"] == plan.physical_count
    assert report["provenance"]["ordering_plan_checksum"] == plan.checksum
    assert report["inference"]["significance"]["available"] is False


def _collection_inputs(tmp_path):
    from .ordering import plan_ordering
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations, "results/initial.json", treatments)
    registration = tmp_path / "ordering-preregistration.md"
    registration.write_text(f"{protocol.identity}\n{initial.checksum}\n{plan.initial_observations_sha256}\n{plan.checksum}\n")
    return protocol, manifest, rows, initial, observations, plan, registration


def _collect(inputs, tmp_path, *, confirmed=True, maximum=1, factory=None, checker=None):
    protocol, manifest, rows, initial, observations, plan, registration = inputs
    return asyncio.run(ordering_cli.run_ordering_collection(
        protocol, manifest, rows, initial, observations, plan, str(tmp_path / "ordering.sqlite"),
        str(registration), plan.physical_count, confirmed,
        provider_model="jev-1.13.0", options=CollectionOptions(max_new_attempts=maximum),
        engine_factory=factory, committed_checker=checker or (lambda *_: True),
    ))


def test_ordering_confirmation_and_committed_registration_are_checked_before_ledger_creation(tmp_path):
    inputs = _collection_inputs(tmp_path)
    factory = lambda _: pytest.fail("unapproved ordering constructed a model")
    with pytest.raises(ValueError, match="confirm"):
        _collect(inputs, tmp_path, confirmed=False, factory=factory)
    with pytest.raises(ValueError, match="committed"):
        _collect(inputs, tmp_path, factory=factory, checker=lambda *_: False)
    assert not (tmp_path / "ordering.sqlite").exists()


def test_invalid_initial_evidence_creates_neither_a_ledger_nor_a_provider(tmp_path):
    protocol, manifest, rows, initial, observations, plan, registration = _collection_inputs(tmp_path)
    factory = lambda _: pytest.fail("invalid initial evidence constructed a model")
    for evidence in (observations[:-1], observations[1:]):
        with pytest.raises(ValueError):
            _collect((protocol, manifest, rows, initial, evidence, plan, registration),
                     tmp_path, factory=factory)
    with pytest.raises(ValueError):
        _collect((protocol, manifest, rows[:-1], initial, observations, plan, registration),
                 tmp_path, factory=factory)
    assert not (tmp_path / "ordering.sqlite").exists()


def test_ordering_cli_collection_resumes_a_bounded_fake_run_without_a_second_provider(tmp_path):
    inputs = _collection_inputs(tmp_path)
    calls = []
    class Fake:
        async def decide(self, task, target, examples):
            calls.append((target.id, len(examples)))
            return DecisionResult("World", model="jev-1.13.0", usage={"tokens": 3})
    first = _collect(inputs, tmp_path, factory=lambda _: Fake())
    replay = _collect(inputs, tmp_path, maximum=0,
                      factory=lambda _: pytest.fail("replay constructed another provider"))
    assert len(calls) == first.new_attempts == first.physical_attempts == 1
    assert replay.new_attempts == 0 and replay.physical_attempts == 1
    assert any(row.cache_hit for row in replay.observations)


def test_unconfirmed_ordering_run_does_not_read_initial_inputs_or_load_a_provider(monkeypatch):
    monkeypatch.setattr(ordering_cli, "_read_initial", lambda *_: pytest.fail("unconfirmed ordering read inputs"))
    args = SimpleNamespace(confirm=False)
    with pytest.raises(ValueError, match="confirm"):
        ordering_cli._run(args)


def test_make_ordering_run_preserves_independent_approval_ledger_and_attempt_bounds():
    root = Path(__file__).parents[2]
    completed = subprocess.run([
        "make", "-n", "ordering-run", "PROTOCOL=protocol.json", "MANIFEST=manifest.json",
        "PREFLIGHT=initial.preflight.json", "INITIAL_OBSERVATIONS=initial.observations.json",
        "ORDERING=ordering.json", "LEDGER=ordering.sqlite", "PREREGISTRATION=ordering.md",
        "ROWS=rows.fixture.json", "OUTPUT=observations.json", "PROVIDER_MODEL=jev-1.13.0",
        "BASE_URL=https://api.typesafe.ai", "TIMEOUT_SECONDS=30", "ADAPTER_REVISION=6137fa185a1a98afa84b5e6d5948d1780df5d56d",
        "PACKAGE_REVISION=typesafe-sdk-0.7.1", "ATTEMPT_CEILING=123", "MAX_NEW=1", "CONFIRM=--confirm",
    ], cwd=root, text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    assert '--ordering "ordering.json"' in completed.stdout
    assert '--ledger "ordering.sqlite"' in completed.stdout
    assert '--preregistration "ordering.md"' in completed.stdout
    assert '--attempt-ceiling "123"' in completed.stdout
    assert '--max-new "1"' in completed.stdout
