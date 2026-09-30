import asyncio
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from decision_flywheel.models import DecisionResult

from . import pilot_cli
from .pilot import plan_jev_pilot
from .preflight import preflight
from .preflight_test import _protocol, _rows


def _inputs(tmp_path):
    manifest, rows = _rows()
    development_source_ids = {record.id for record in manifest.candidate + manifest.development}
    rows = tuple(row for row in rows if row.id in development_source_ids)
    base = _protocol(manifest, rows, artifact=False)
    protocol = replace(base, models=(base.models[0],))
    source = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    plan = plan_jev_pilot(protocol, manifest, rows, source)
    registration = tmp_path / "pilot-registration.md"
    registration.write_text(f"{protocol.identity}\n{source.checksum}\n{plan.checksum}\n", encoding="utf-8")
    return protocol, manifest, rows, source, plan, registration


class _FakeModel:
    def __init__(self, calls):
        self.calls = calls

    async def decide(self, task, target, examples):
        self.calls.append((target.id, len(examples)))
        return DecisionResult(task.labels[0], usage={"tokens": 2}, model="jev-1.13.0")


def _run(inputs, ledger, *, confirmed=True, ceiling=None, maximum=None, factory=None, checker=None):
    protocol, manifest, rows, source, plan, registration = inputs
    return asyncio.run(pilot_cli.run_pilot_collection(
        protocol, manifest, rows, source, plan, str(ledger), str(registration),
        plan.physical_count if ceiling is None else ceiling,
        plan.physical_count if maximum is None else maximum, confirmed,
        provider_model="jev-1.13.0", engine_factory=factory,
        committed_checker=checker or (lambda *_: True),
    ))


def test_a_pilot_requires_confirmation_and_its_exact_small_ceiling_before_opening_a_ledger(tmp_path):
    inputs = _inputs(tmp_path)
    ledger = tmp_path / "pilot.sqlite"
    factories = []
    factory = lambda identity: factories.append(identity)

    with pytest.raises(ValueError, match="confirm"):
        _run(inputs, ledger, confirmed=False, factory=factory)
    with pytest.raises(ValueError, match="pilot"):
        _run(inputs, ledger, ceiling=inputs[4].physical_count + 1, factory=factory)

    assert factories == []
    assert not ledger.exists()


def test_a_pilot_rejects_an_uncommitted_or_mismatched_registration_before_creating_a_ledger(tmp_path):
    inputs = _inputs(tmp_path)
    factories = []
    factory = lambda identity: factories.append(identity)
    ledger = tmp_path / "pilot.sqlite"

    with pytest.raises(ValueError, match="committed"):
        _run(inputs, ledger, factory=factory, checker=lambda *_: False)
    assert not ledger.exists()

    inputs[-1].write_text("not this pilot", encoding="utf-8")
    with pytest.raises(ValueError, match="committed"):
        _run(inputs, ledger, factory=factory)
    assert not ledger.exists()
    assert factories == []


def test_a_pilot_ledger_uses_the_exact_unique_request_ceiling_not_an_incomplete_smaller_plan(tmp_path):
    inputs = _inputs(tmp_path)
    ledger = tmp_path / "pilot.sqlite"
    with pytest.raises(ValueError, match="pilot"):
        _run(inputs, ledger, ceiling=inputs[4].physical_count - 1, maximum=0, factory=lambda _: None)
    assert not ledger.exists()


def test_a_completed_pilot_replays_without_a_second_provider_or_an_expanded_budget(tmp_path):
    inputs = _inputs(tmp_path)
    calls = []
    first = _run(inputs, tmp_path / "pilot.sqlite", factory=lambda _: _FakeModel(calls))
    factories = []
    replay = _run(inputs, tmp_path / "pilot.sqlite", maximum=0,
                  factory=lambda identity: factories.append(identity))

    assert first.complete and replay.complete
    assert len(calls) == first.new_attempts == inputs[4].physical_count
    assert replay.new_attempts == 0 and replay.physical_attempts == first.physical_attempts
    assert factories == []
    assert all(row.cache_hit for row in replay.observations)


def test_the_live_pilot_command_without_confirmation_never_reads_rows_or_provider_settings(monkeypatch):
    monkeypatch.setattr(pilot_cli, "_load_rows", lambda *_: pytest.fail("unconfirmed pilot read source rows"))
    arguments = ["run", "--protocol", "missing.json", "--manifest", "missing.json",
                 "--source-preflight", "missing.json", "--pilot", "missing.json",
                 "--ledger", "unused.sqlite", "--preregistration", "missing.md",
                 "--output", "unused.json", "--dataset-cache", "unused-cache",
                 "--provider-model", "jev-1.13.0", "--base-url", "https://api.typesafe.ai",
                 "--timeout-seconds", "30", "--adapter-revision", "0" * 40,
                 "--package-revision", "typesafe-sdk-0.7.1", "--attempt-ceiling", "21", "--max-new", "21"]

    with pytest.raises(SystemExit) as error:
        pilot_cli.main(arguments)
    assert error.value.code == 2


def test_make_pilot_dry_run_preserves_the_explicit_source_plan_and_small_attempt_limit():
    root = Path(__file__).parents[2]
    completed = subprocess.run([
        "make", "-n", "pilot", "PROTOCOL=protocol.json", "MANIFEST=manifest.json", "ROWS=fixture.json",
        "PREFLIGHT=source.json", "PILOT=pilot.json", "LEDGER=pilot.sqlite", "PREREGISTRATION=pilot.md",
        "OUTPUT=observations.json", "PROVIDER_MODEL=jev-1.13.0", "BASE_URL=https://api.typesafe.ai",
        "TIMEOUT_SECONDS=30", "ADAPTER_REVISION=6137fa185a1a98afa84b5e6d5948d1780df5d56d",
        "PACKAGE_REVISION=typesafe-sdk-0.7.1", "ATTEMPT_CEILING=21", "MAX_NEW=21", "CONFIRM=--confirm",
    ], cwd=root, text=True, capture_output=True, check=False)

    assert completed.returncode == 0
    assert '--source-preflight "source.json"' in completed.stdout
    assert '--pilot "pilot.json"' in completed.stdout
    assert '--max-new "21"' in completed.stdout


@pytest.mark.parametrize("handler", (pilot_cli._run, pilot_cli._report))
def test_pilot_commands_do_not_overwrite_existing_observations_or_reports_without_permission(tmp_path, handler):
    output = tmp_path / "existing.json"
    output.write_text("existing evidence", encoding="utf-8")
    args = SimpleNamespace(output=str(output), confirm=True, max_new=21, attempt_ceiling=21, overwrite=False)
    with pytest.raises(ValueError, match="overwrite"):
        handler(args)
    assert output.read_text(encoding="utf-8") == "existing evidence"
