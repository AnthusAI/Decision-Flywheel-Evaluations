"""The Stage 0 command rejects an implicit live path before touching inputs."""
from __future__ import annotations

import runpy

import pytest


def test_confirm_live_is_an_explicit_unimplemented_safe_error_before_reading_any_file():
    command = runpy.run_path("scripts/reviews_stage0.py")["main"]

    with pytest.raises(SystemExit) as stopped:
        command(["--confirm-live", "--manifest", "/does/not/exist", "--identity", "/does/not/exist",
                 "--screen-cache", "/does/not/exist", "--agreement-cache", "/does/not/exist",
                 "--report", "/does/not/exist", "--jev-ceiling", "1", "--jev-max-new", "1",
                 "--sme-ceiling", "1", "--sme-max-new", "0"])

    assert stopped.value.code == 2
