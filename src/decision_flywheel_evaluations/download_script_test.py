import json
import runpy
from pathlib import Path


def _script_namespace():
    script = Path(__file__).resolve().parents[2] / "scripts" / "download_pinned_datasets.py"
    return runpy.run_path(str(script), run_name="download_pinned_datasets_test")


def test_script_does_not_reach_the_lazy_loader_without_explicit_confirmation(capsys):
    namespace = _script_namespace()
    calls = []
    namespace["main"].__globals__["acquire_pinned_datasets"] = lambda **_kwargs: calls.append("called")

    status = namespace["main"](["download"])

    assert status == 2 and calls == []
    assert json.loads(capsys.readouterr().out) == {"status": "confirmation_required"}


def test_script_reports_a_confirmed_loader_failure_without_echoing_secret_content(capsys):
    namespace = _script_namespace()
    namespace["main"].__globals__["acquire_pinned_datasets"] = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("credential-like-secret-must-not-appear")
    )

    status = namespace["main"](["download", "--confirm"])
    output = capsys.readouterr().out

    assert status == 1
    assert json.loads(output) == {"reason": "RuntimeError", "status": "failed"}
    assert "credential-like-secret" not in output
