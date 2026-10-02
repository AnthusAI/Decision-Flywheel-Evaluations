# FOMC: teaching a published rubric to Jev with the flywheel (first run)

Status: one seed, Jev `jev-1.13.0`, 2026-10-02. Exploratory. Text-free summaries in
[results/fomc/](results/fomc/). Plan: Kanbus initiative 'Rubric-dataset simulation'.

**Setup.** FOMC hawkish / dovish / neutral sentences (gtfintechlab, CC BY-NC 4.0). The
classifier starts from a one-line rubric. A simulated Fed analyst (`gpt-6-luna`) holds the
published annotation rubric, sees the classifier's mistakes on labeled training items and
explains up to 15 per round with a verbatim rubric quote (string-checked); the feature
analyst never sees the rubric. 3 rounds of 100 labels; final on 377 held-out test items.
The ceiling arm is Jev given the full published rubric.

| Arm | Macro-F1 (final 377) | Share of S-to-ceiling gap |
|---|---|---|
| 0: one-line rubric | 0.656 | 0% |
| A: features from labels only | 0.667 | 40% |
| **A-c: features from simulated analyst explanations** | **0.682** | **92%** |
| A-c, explanations shuffled onto wrong items | 0.669 | 47% |
| A-c, noisy analyst (~20% bad explanations) | 0.679 | 82% |
| F: optimized fixed example list | 0.651 | -16% |
| A-c + F | 0.672 | 59% |
| Ceiling: full published rubric | 0.684 | 100% |

Paired 95% intervals (macro-F1): A-c - 0 = +0.026 (-0.004, +0.056); A-c - A = +0.015
(-0.006, +0.036); A - 0 = +0.011 (-0.014, +0.037); F - 0 = -0.005 (-0.041, +0.030).

**Reading it honestly.**
- The ordering is what the idea predicts: real explanations > shuffled ~ labels only >
  baseline, and the explanation arm nearly reaches the full-rubric ceiling.
- But the gap is SMALL on the held-out test split: 0.656 -> 0.684 (2.8 points), not the
  13 points the 120-item screen showed on training items. All intervals include zero.
- Every arm gained only in round 1 (one new question each); later proposals were rejected by
  the promotion gate. The simulated analyst declined about a third of errors as
  UNEXPLAINABLE, a sign of label noise.
- The optimized fixed example list did not help here.
- Spend: about 11,000 Jev requests plus ~200 OpenAI calls (analyst + stakeholder).
