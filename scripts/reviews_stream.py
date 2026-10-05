#!/usr/bin/env python
"""Offline-only streaming SME fixtures and reports; no paid collection is enabled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_flywheel_evaluations.unified_stream_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
