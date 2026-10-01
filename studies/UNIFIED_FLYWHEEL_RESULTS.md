# Unified flywheel: features, few-shot, and both (first results)

Status: one seed, one corpus, one engine (Jev `jev-1.13.0`). Exploratory. Plan:
[UNIFIED_FLYWHEEL_PLAN.md](UNIFIED_FLYWHEEL_PLAN.md). Text-free run summaries:
[dev-100](results/unified/seed1.dev100.summary.json) and
[paper-600](results/unified/seed1.paper600.summary.json).

## What was run

Jev-Flywheel's planted-bias sentiment corpus (labels secretly track subject
matter). A scripted labeler gives 100 labels per round for 3 rounds. Five arms
share the same labels, order, folds and engine:

- **0** baseline: Jev's zero-shot holistic answer, refit.
- **A** features: the Jev-Flywheel steering loop; an analyst (OpenAI
  `gpt-6-luna`) proposes new questions; the head is retrained.
- **B** few-shot: retrieved labeled examples (4 per label, lexical overlap) put
  in Jev's request; its answer is one more head feature.
- **B-local**: no prompt change; the nearest labeled neighbours' label shares
  feed the head as features. Free (no new Jev calls).
- **A+B**: both.

Iteration was scored on 100 test items (dev-100); the comparison below is on the
600 held-out items (paper-600, which informed earlier Jev-Flywheel work, so not
pristine). Nothing was tuned on paper-600.

## Result after 300 labels (paper-600, n = 600)

| Arm | Accuracy | Brier | ECE |
|---|---|---|---|
| 0 Baseline | 0.767 | 0.161 | 0.043 |
| A features | 0.855 | 0.092 | 0.022 |
| B-local (free) | 0.853 | 0.107 | 0.022 |
| B few-shot | 0.892 | 0.078 | 0.045 |
| **A+B** | **0.918** | **0.059** | 0.016 |

Paired bootstrap, 95% intervals (accuracy): A − 0 = +0.088 (0.053 to 0.123);
B − 0 = +0.125 (0.088 to 0.162); B − B-local = +0.038 (0.012 to 0.063);
A+B − better of A and B (here B) = +0.027 (0.007 to 0.048). Brier for the last
is −0.018 (−0.027 to −0.011). No significance claims.

- Each lever alone clearly beats the baseline; few-shot examples in the prompt
  gave the larger single gain; the combination was best on accuracy and Brier.
- The free neighbour-vote features (B-local) matched feature discovery (A) on
  accuracy, but trail few-shot prompting (B) by about 4 points.
- The analyst proposed the true topic question (`topic_domain`) unaided in
  round 2 of arm A. Arm B and B-local gain without naming anything.

## Limits

- One seed, one planted-bias corpus, one engine, a simulated labeler, and a
  `paper-600` slice that is not pristine.
- B and B-local may be absorbing the planted bias silently (retrieved
  same-topic neighbours carry the label); A states it in words. Accuracy alone
  cannot separate these, and B's advantage here may not transfer to real labels.
- Few-shot features for labeled items are leave-one-out, computed once, not per
  fold (a rebuild would have cost about 5× the requests); this may slightly
  favour B and A+B at the promotion gate.
- Jev-Flywheel's own steering found the right factor only about a quarter of the
  time per round, so one seed does not settle A's reliability.
- Jev spend: 2,400 live requests on dev-100 and 3,600 on paper-600 (about 6,000
  in all, well under the ceilings set). Dollar cost is not derivable from usage;
  check provider billing. Analyst calls (OpenAI) are small and not counted.

## Addendum: the default product (optimized fixed list), dev-100 only

Run 2026-10-01 on Jev with the one-request-per-item shape, three rounds of 100
labels, scored on the 100-item development slice only (about ±9 points; paper-600
not used). Text-free summary:
[results/unified/flywheel-seed1.dev100.summary.json](results/unified/flywheel-seed1.dev100.summary.json).
About 5,700 Jev requests in all.

| Round | 0 | A features | F fixed list | F-rand | A-c+F (default) |
|---|---|---|---|---|---|
| 1 | 0.74 / 0.160 | 0.74 / 0.160 | 0.78 / 0.135 | 0.76 / 0.150 | 0.78 / 0.135 |
| 2 | 0.74 / 0.160 | 0.89 / 0.069 | 0.74 / 0.146 | 0.70 / 0.172 | 0.91 / 0.064 |
| 3 | 0.74 / 0.160 | 0.92 / 0.059 | 0.78 / 0.139 | 0.72 / 0.153 | 0.89 / 0.079 |

(accuracy / Brier.) Feature discovery drives the gain on this corpus (the analyst
finds the topic question in round 2). The optimized fixed list helps a little
alone (0.78 vs 0.74 baseline; better than a random list at 0.72) but adds nothing
visible on top of discovered features (A-c+F is within noise of A). Dynamic
retrieval (arm B, 0.92 on paper-600 earlier) is a different mechanism that feeds
per-item topic neighbours. Single seed; a leaking-free labeler with comments is
not yet in place (A-c equals A).

## Addendum 2: explanations, saved bundles, and retriever comparison

Simulated human with reasons (`gpt-6-luna` writing terse reasons from the corpus's
hidden convention; leaks the rule by design, so this shows the plumbing and speed,
not that real comments find hidden factors): 172 comments for 300 labels, 128 labels
without a comment. Dev-100, three rounds, Jev:

| Round | 0 | A-c (features + explanations) | A-c+F (default product) |
|---|---|---|---|
| 1 | 0.74 / 0.160 | **0.89 / 0.071** | 0.87 / 0.112 |
| 2 | 0.74 / 0.160 | 0.89 / 0.073 | 0.88 / 0.070 |
| 3 | 0.74 / 0.160 | 0.89 / 0.073 | 0.91 / 0.063 |

With explanations the analyst found the topic factor in round 1; without them
(arm A) it took until round 2.

Final comparison on paper-600 (n = 600), scored through the reloaded saved
classifier bundles after 300 labels (accuracy / Brier / ECE): baseline
0.767 / 0.161 / 0.043; A-c 0.867 / 0.098 / 0.028; F (optimized fixed list only)
0.782 / 0.142 / 0.055; F-rand 0.763 / 0.152 / 0.090; A-c+F 0.850 / 0.092 / 0.050.
F − baseline: +0.015 accuracy (−0.008 to +0.038), Brier −0.019 (−0.028 to −0.009);
F − F-rand: +0.018 (−0.005 to +0.043), Brier −0.010 (−0.019 to −0.001);
A-c+F − better of A-c and F: accuracy −0.017 (−0.038 to +0.002), Brier −0.006
(−0.016 to +0.002). On this corpus the discovered features carry the gain; the
optimized fixed list improves calibration modestly and adds no accuracy on top of
features. Text-free summaries: [rounds](results/unified/phase2-rounds.dev100.summary.json),
[final](results/unified/phase2-final.paper600.summary.json).

Retriever comparison, free (neighbour-label purity, k = 4, 1,500 labeled pool items,
[results](results/unified/retrieval-purity.json)): embeddings (`text-embedding-3-small`,
512 d) 0.890; BM25 0.874; lexical v2 with stop words 0.869; TF-IDF 0.868;
lexical v1 equivalent 0.867. Differences are small on this topic-driven corpus;
embeddings lead. Dynamic retrieval remains an optional feature; the fixed list stays
the default.
