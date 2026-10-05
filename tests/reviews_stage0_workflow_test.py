import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _make(*args):
    return subprocess.run(["make", *args], cwd=ROOT, text=True, capture_output=True)


def test_stage_zero_preflight_is_a_cache_only_command_with_no_live_confirmation_or_client_flags():
    result = _make("-n", "reviews-stage0-preflight", "PYTHON=python")

    assert result.returncode == 0
    command = result.stdout
    assert "scripts/reviews_stage0.py" in command
    assert "--sme-cache" in command and "--live" not in command and "--confirm" not in command
    assert "--stage1-ceiling \"0\"" in command and "--stage2-ceiling \"0\"" in command


def test_stage_zero_run_refuses_missing_confirmation_before_invoking_the_script():
    result = _make("reviews-stage0-run", "PYTHON=python", "STAGE0_MAX_NEW=1")

    assert result.returncode == 2
    assert "CONFIRM=--confirm" in result.stdout + result.stderr
    assert "scripts/reviews_stage0.py" not in result.stdout + result.stderr


def test_stage_zero_run_dry_command_carries_the_fixed_model_and_bounded_explicit_cap():
    result = _make("-n", "reviews-stage0-run", "PYTHON=python", "STAGE0_MAX_NEW=600", "CONFIRM=--confirm")

    assert result.returncode == 0
    command = result.stdout
    assert "scripts/reviews_stage0.py" in command
    assert "--live \"--confirm\"" in command
    assert "--jev-model \"jev-1.13.0\"" in command
    assert "--jev-max-new \"600\"" in command


def test_stage_zero_run_rejects_a_cap_above_six_hundred_without_invoking_the_script():
    result = _make("reviews-stage0-run", "PYTHON=python", "STAGE0_MAX_NEW=601", "CONFIRM=--confirm")

    assert result.returncode == 2
    assert "at most 600" in result.stdout + result.stderr
    assert "scripts/reviews_stage0.py" not in result.stdout + result.stderr
