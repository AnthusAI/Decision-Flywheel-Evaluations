#!/usr/bin/env python
"""Held-out 300 / stream 600 / dev-100 splits for the Amazon review corpus (see ``decision_flywheel_evaluations.unified_reviews``).

Reads the private pool and the SME cache READ-ONLY; no model is called. Prints a text-free report.

    python scripts/reviews_splits.py                                   # report only
    python scripts/reviews_splits.py --out studies/amazon_reviews/splits.json   # also write the ids
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_flywheel_evaluations.unified_reviews import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
