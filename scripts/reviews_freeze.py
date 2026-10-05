#!/usr/bin/env python
"""One-shot, text-free freeze of the approved first-1,500 Amazon-review universe."""
from decision_flywheel_evaluations.unified_reviews_manifest import freeze_main


if __name__ == "__main__":
    raise SystemExit(freeze_main())
