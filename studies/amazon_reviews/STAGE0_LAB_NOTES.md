# Amazon Reviews Stage 0 lab notes

Run date: 2026-10-05. Model: Jev `jev-1.13.0`.

## Question

Can Jev apply this moderation policy accurately enough that optimizing a fixed
list of labeled examples is worth testing? This screen compares the one-sentence
starting policy (S) with the complete private policy (F), on the same frozen 300
held-out reviews. It is not the example-optimization experiment itself.

## Result

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| S: starting policy | 0.857 | 0.538 |
| F: full policy | 0.937 | 0.838 |

The paired differences are +0.080 accuracy (95% bootstrap interval +0.043 to
+0.117) and +0.300 macro-F1 (+0.184 to +0.421), F minus S. The screen therefore
passes its preregistered gate: F is above 0.80 macro-F1 and S trails F by more
than 0.10.

The plain-language interpretation is encouraging: the short prompt handles the
dominant `approve` class reasonably well but misses much of the policy boundary
for abusive, promotional, and seller/shipping reviews. Giving Jev the complete
policy sharply improves those minority-class decisions. There is therefore
something concrete for an example-selection optimizer to try to teach.

## Collection record

The simulated-SME consistency check agreed on 99 of 100 fixed reviews. This is
an internal consistency measurement, not a claim of human-label accuracy.

The first Jev request was rejected with HTTP 402 before credits were restored.
It was retained as an attempted, failed cell. The owner explicitly authorized
one replacement attempt, producing an auditable 601-attempt ledger for the
600-cell S/F matrix. All 600 final cells completed. Jev did not return usable
token-usage fields, so no price or token total is reported.

## What this does not establish

It does not show that few-shot examples close the gap, that an optimizer can
choose good examples, or that the result survives alternative prompt/example
orderings. The frozen split is imbalanced and has only nine promotional reviews;
that class's estimate is especially noisy. The policy and reference labels are
generated through a simulated SME rather than human moderation.

The next study decision is whether to run the explicitly planned learning arms.
They are not started by this screen.

The machine-readable, text-free aggregate is
[stage0_results.json](stage0_results.json).
