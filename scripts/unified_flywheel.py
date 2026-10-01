#!/usr/bin/env python
"""Unified flywheel harness: arms 0, A, B-local, B and A+B on Jev (studies/UNIFIED_FLYWHEEL_PLAN.md),
plus the fixed-list arms F, F-rand, A-c and A-c+F (studies/DECISION_FLYWHEEL_DESIGN.md).

Reproducibility: ``jev_flywheel`` is imported from a clean local clone of Jev-Flywheel at
commit cd4a4886028d518c4dc20c88608bb05cbc6bf370, kept in this repository's gitignored
``var/jev-flywheel-cd4a4886`` (override with UNIFIED_FLYWHEEL_JEV_CLONE). Create it with:

    python scripts/unified_flywheel.py prepare-clone --source /Users/home/Projects/Jev-Flywheel

The working Jev-Flywheel checkout is never imported or modified. Fixtures (items, cached
answers, the simulated-labeler recording) are read from the clone.

The interpreter needs scikit-learn and Tactus (Jev-Flywheel's own environment has both);
Decision-Flywheel comes from UNIFIED_FLYWHEEL_CORE (a source tree such as
/Users/home/Projects/Decision-Flywheel/src; the Makefile sets it) or else from this
repository's pinned install.

$0 replay of a copied live run directory (fails if anything is uncached):

    UNIFIED_FLYWHEEL_CORE=... python scripts/unified_flywheel.py run --replay \
        --provider-model jev-1.13.0 --run-dir var/unified-flywheel/replay-dev100

Offline dry run (no network, fake Jev, recorded/fake analyst replies):

    /Users/home/Projects/Jev-Flywheel/.venv312/bin/python scripts/unified_flywheel.py run

Live mode exists but is off by default; see ``unified_cli.check_live_gates`` for its gates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_flywheel_evaluations.unified_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
