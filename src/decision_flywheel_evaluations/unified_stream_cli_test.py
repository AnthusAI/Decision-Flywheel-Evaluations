"""The stream command is offline-only and the report reads metadata only."""
import json
import subprocess
from pathlib import Path

import pytest

from . import unified_stream_cli as cli


def test_a_live_flag_is_rejected_before_any_stream_engine_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run_fixture", lambda *_args, **_kwargs: pytest.fail("loaded engine"))
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--live", "--fixture", str(tmp_path / "absent.json"),
                  "--run-dir", str(tmp_path / "run"), "--output", str(tmp_path / "out.json"),
                  "--max-new-requests", "10"])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_the_actual_learning_runtime_requires_an_explicit_synthetic_flag_before_loading_it(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run_runtime", lambda *_args, **_kwargs: pytest.fail("loaded runtime"), raising=False)
    with pytest.raises(SystemExit) as error:
        cli.main(["runtime", "--run-dir", str(tmp_path / "run"), "--output", str(tmp_path / "out.json"),
                  "--max-new-requests", "10"])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_the_synthetic_runtime_passes_its_actual_learning_cap_and_shape_explicitly(tmp_path, monkeypatch):
    seen = []
    def run(run_dir, **options):
        seen.append((run_dir, options))
        return {"schema": "decision-flywheel-evaluations/stream/v1", "is_synthetic": True, "arms": {}}
    monkeypatch.setattr(cli, "_run_runtime", run, raising=False)
    output = tmp_path / "out.json"
    assert cli.main(["runtime", "--synthetic", "--run-dir", str(tmp_path / "run"), "--output", str(output),
                     "--max-new-requests", "7", "--seed", "4", "--review-probability", "1",
                     "--stream-size", "32", "--heldout-size", "8"]) == 0
    assert seen == [(tmp_path / "run", {"seed": 4, "review_probability": 1.0, "max_new_requests": 7,
                                         "stream_size": 32, "heldout_size": 8})]
    assert json.loads(output.read_text())["is_synthetic"] is True


def test_an_existing_artifact_is_not_replaced_or_followed_by_engine_work(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    output.write_text("old evidence")
    monkeypatch.setattr(cli, "_run_fixture", lambda *_args, **_kwargs: pytest.fail("loaded engine"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--fixture", str(tmp_path / "absent.json"), "--run-dir", str(tmp_path / "run"),
                  "--output", str(output), "--max-new-requests", "10"])
    assert output.read_text() == "old evidence"


def test_existing_run_evidence_is_not_overwritten_when_the_output_path_is_new(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = run_dir / "stream-results.json"
    evidence.write_text("old run")
    fixture = tmp_path / "input.json"
    fixture.write_text(json.dumps({"synthetic": True, "labels": ["a", "b"], "stream": []}))
    monkeypatch.setattr(cli, "_run_fixture", lambda *_args, **_kwargs: pytest.fail("loaded engine"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--fixture", str(fixture), "--run-dir", str(run_dir),
                  "--output", str(tmp_path / "new.json"), "--max-new-requests", "10"])
    assert evidence.read_text() == "old run"


def test_a_fixture_not_explicitly_marked_synthetic_is_rejected_before_engine_work(tmp_path, monkeypatch):
    fixture = tmp_path / "input.json"
    fixture.write_text(json.dumps({"labels": ["a", "b"], "stream": []}))
    monkeypatch.setattr(cli, "_run_fixture", lambda *_args, **_kwargs: pytest.fail("loaded engine"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--fixture", str(fixture), "--run-dir", str(tmp_path / "run"),
                  "--output", str(tmp_path / "out.json"), "--max-new-requests", "10"])
    assert not (tmp_path / "run").exists()


def test_fixture_commands_pass_exact_caps_and_write_only_the_returned_safe_run(tmp_path, monkeypatch):
    fixture = tmp_path / "input.json"
    fixture.write_text(json.dumps({"synthetic": True, "labels": ["a", "b"], "stream": []}))
    payload = {"schema": "decision-flywheel-evaluations/stream/v1", "synthetic": True,
               "labels": ["a", "b"], "arms": {}}
    seen = []
    def run(document, run_dir, **options):
        seen.append((document, run_dir, options))
        return payload
    monkeypatch.setattr(cli, "_run_fixture", run)
    output = tmp_path / "out.json"
    assert cli.main(["run", "--fixture", str(fixture), "--run-dir", str(tmp_path / "run"),
                     "--output", str(output), "--max-new-requests", "7", "--seed", "4",
                     "--review-probability", "0.5"]) == 0
    assert seen[0][2] == {"seed": 4, "review_probability": 0.5, "max_new_requests": 7}
    assert json.loads(output.read_text()) == payload


def test_reports_need_no_fixture_private_policy_or_model_runtime(tmp_path, monkeypatch):
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"arms": {}}))
    monkeypatch.setattr(cli, "_run_fixture", lambda *_args, **_kwargs: pytest.fail("loaded engine"))
    def report(document, **options):
        assert document == {"arms": {}}
        assert options == {"window_size": 100, "seed": 0, "resamples": 3}
        return {"synthetic": True, "curves": {}}
    monkeypatch.setattr(cli, "_report_payload", report)
    output = tmp_path / "report.json"
    assert cli.main(["report", "--input", str(source), "--output", str(output), "--resamples", "3"]) == 0
    assert json.loads(output.read_text()) == {"synthetic": True, "curves": {}}


def test_a_report_can_render_markdown_from_safe_aggregates_only(tmp_path, monkeypatch):
    source = tmp_path / "run.json"
    source.write_text("{}")
    monkeypatch.setattr(cli, "_report_payload", lambda *_args, **_kwargs: {"caveats": []})
    monkeypatch.setattr(cli, "_render_report", lambda value: "Synthetic offline report\n")
    output = tmp_path / "report.md"
    assert cli.main(["report", "--input", str(source), "--output", str(output),
                     "--format", "markdown"]) == 0
    assert output.read_text() == "Synthetic offline report\n"


@pytest.mark.parametrize("error_type", [ValueError, TypeError, KeyError, IndexError])
def test_errors_do_not_echo_a_private_review_or_a_key(tmp_path, monkeypatch, capsys, error_type):
    source = tmp_path / "run.json"
    source.write_text("{}")
    def broken(*_args, **_kwargs):
        raise error_type("private review and sk-private-secret")
    monkeypatch.setattr(cli, "_report_payload", broken)
    with pytest.raises(SystemExit):
        cli.main(["report", "--input", str(source), "--output", str(tmp_path / "out.json")])
    output = capsys.readouterr()
    assert "private review" not in output.err and "sk-private-secret" not in output.err
    assert error_type.__name__ in output.err


def test_the_script_is_a_thin_entry_point_into_the_offline_module():
    source = Path(__file__).parents[2] / "scripts" / "reviews_stream.py"
    text = source.read_text()
    assert "unified_stream_cli import main" in text
    assert "dotenv" not in text and "LiveCompletion" not in text


def test_make_stream_targets_are_explicit_offline_fixture_and_local_report_commands():
    root = Path(__file__).parents[2]
    result = subprocess.run(["make", "-n", "reviews-stream-fixture", "STREAM_FIXTURE=toy.json",
                             "STREAM_RUN_DIR=var/toy", "STREAM_OUTPUT=var/toy.run.json", "STREAM_MAX_NEW=7"],
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == 0
    assert '--fixture "toy.json"' in result.stdout and '--max-new-requests "7"' in result.stdout
    assert "--live" not in result.stdout
    report = subprocess.run(["make", "-n", "reviews-stream-report", "STREAM_INPUT=var/toy.run.json",
                             "STREAM_OUTPUT=var/toy.report.json"], cwd=root, text=True, capture_output=True)
    assert report.returncode == 0
    assert '--input "var/toy.run.json"' in report.stdout


def test_the_make_runtime_target_requires_the_pinned_interpreter_and_explicit_synthetic_mode():
    root = Path(__file__).parents[2]
    result = subprocess.run(["make", "-n", "reviews-stream-runtime", "UF_PYTHON=/fake/python",
                             "STREAM_RUN_DIR=var/runtime", "STREAM_OUTPUT=var/runtime.json",
                             "STREAM_MAX_NEW=400"], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0
    assert 'runtime --synthetic' in result.stdout
    assert '--max-new-requests "400"' in result.stdout
    assert "/fake/python" in result.stdout and "--live" not in result.stdout
    assert "PYTHONPATH=src:" in result.stdout


def test_a_synthetic_run_and_report_compose_without_private_text_or_model_calls(tmp_path):
    stream = [{"id": "s1", "label": "approve", "text": "PRIVATE REVIEW SENTENCE"},
              {"id": "s2", "label": "flag"}]
    heldout = [{"id": "h1", "label": "approve"}]
    predictions = {arm: {"s1": "approve", "s2": "flag", "h1": "approve"}
                   for arm in ("B", "L", "E", "X")}
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"synthetic": True, "labels": ["approve", "flag"],
                                   "stream": stream, "heldout": heldout, "predictions": predictions}))
    outcomes, report = tmp_path / "outcomes.json", tmp_path / "report.json"
    assert cli.main(["run", "--fixture", str(fixture), "--run-dir", str(tmp_path / "run"),
                     "--output", str(outcomes), "--max-new-requests", "0"]) == 0
    assert cli.main(["report", "--input", str(outcomes), "--output", str(report), "--resamples", "3"]) == 0
    payload = json.loads(report.read_text())
    assert payload["provenance"]["synthetic_fixture"] is True
    assert all(arm["cumulative"][-1]["accuracy"] == 1 for arm in payload["prequential"].values())
    assert all(arm["request_attempts"] == 0 for arm in payload["spend"].values())
    assert all("PRIVATE REVIEW SENTENCE" not in path.read_text() for path in (outcomes, report))
