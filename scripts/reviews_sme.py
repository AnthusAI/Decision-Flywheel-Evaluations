#!/usr/bin/env python
"""Simulated SME labels for the Amazon review pool (see ``decision_flywheel_evaluations.unified_sme``).

Offline (fake keyword SME, $0):      python scripts/reviews_sme.py label --limit 1500
Live (main session only, gated):     python scripts/reviews_sme.py label --limit 1500 --live --confirm \
                                         --ledger var/amazon-reviews/sme-ledger.json --max-calls 151 --model gpt-6-sol
Agreement (second pass, 100 items):  python scripts/reviews_sme.py agree --subset 100 --live --confirm \
                                         --ledger var/amazon-reviews/sme-ledger.json --max-calls 11 --model gpt-6-sol
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_flywheel_evaluations.unified_sme import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
