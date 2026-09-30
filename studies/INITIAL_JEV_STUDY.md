# Initial Jev study: example selection and “lots” of context

Status: proposed design with descriptive inference, not an approved live run or a completed study.
No new model results are reported here. The Kanbus preregistration task remains
open until the executable configuration, exact preflight, inference plan, and
collection approval are frozen. This document alone authorizes no calls.

## Hypothesis

Ground-truth agreement can improve when labeled examples clarify a decision
model's initially imperfect category boundaries. We want to distinguish the
effect of more examples from the effect of choosing different examples. One
successful selected context would not establish that selection generally
matters more than quantity.

The first investigation uses Jev and one fixed presentation rule. An ordering
study follows the initial results; comparisons with Kev and Laya follow that.

## Data and size ladder

The committed manifests fix the partitions before optimization. Demonstrations
come only from trusted training labels; development labels choose a policy.
The official-test scoreboard is never used to choose examples or task wording.
Development setup reads only candidate/development source rows. Protected
scoreboard IDs and normalized hashes come from the text-free manifest.

| Dataset | Candidate | Development | Scoreboard | Positive examples per request |
| --- | ---: | ---: | ---: | --- |
| AG News | 2,048 | 400 | 2,000, 500/class | 4, 16, 64, 256 |
| Emotion | 1,536 | 600 | 2,000, natural distribution | 6, 24, 96, 384 |

These are 1, 4, 16, and 64 examples per label, plus zero-shot. AG News excludes
the prior study's scoreboard and exact normalized matches. Emotion's earlier
holdout exposure makes its new results exploratory, not a fresh confirmatory
test. The seed is `20261001`; the exact revisions and historical provenance are
in the manifests. “Fresh” means unexposed in this project's recorded prior
studies; it does not establish absence from model pretraining. MIT covers code,
not either dataset.

## Conditions and selection

All conditions use the same dataset-specific question and option order. State
contains `target.text` and `labeled_examples`; zero-shot supplies an empty list.
The question explicitly asks to classify only the target and permits learning
the intended categories from the examples.

The conditions are zero-shot, five random-balanced draws (seeds 0–4),
development-selected global context, lexical prototypes, and per-target lexical
retrieval. Random membership is nested across positive sizes for each draw.
Prototypes and retrieval are lexical policies, not claims about learned
semantic embeddings. The initial display rule is canonical: changing
membership does not introduce a different ordering treatment.

The native optimizer evaluates 20 random-balanced development trials: four
sizes times five seeds. It chooses a complete winning trial using development
accuracy for AG News and macro-F1 for Emotion. Its winning random policy is
then frozen and projected onto the entire nested size ladder. This is not four
independently tuned winners. The selected-global artifact and its complete
development evidence must be validated before any scoreboard preflight.

## Proposed model and limits

The proposed provider version is `jev-1.13.0`, not a moving alias. The SDK base
URL is `https://api.typesafe.ai`, with a 30-second timeout and SDK retries off.
The proposed adapter revision is core commit
`6137fa185a1a98afa84b5e6d5948d1780df5d56d`, with `typesafe-sdk-0.7.1`.
Public transport settings are fingerprinted; no credential enters an artifact.
The live runtime gate also checks installed SDK version and exact Git-pinned
core provenance before opening a ledger or constructing a provider. A matching
caller-supplied version string alone is not sufficient.

[TypeSafe's model documentation](https://docs.typesafe.ai/models) currently
specifies a 64k total request limit and a 32k state-plus-longest-question limit.
Declaring 64 examples per label is an operational ladder limit, not proof that
every serialized context fits those token limits. Preflight reports explicitly
named local estimates, not provider token counts. A bounded development-only
capability pilot must verify the request/response contract and largest contexts
before full collection. Never silently truncate a context or drop examples to
make a nominal size fit.

## Request accounting and approval

Development selection has 8,000 AG News and 12,000 Emotion logical decisions.
Proposed cumulative development ceilings are 8,400 and 12,600 physical attempts,
respectively. They include a bounded retry allowance; they are not approval to
spend that budget. SDK retries remain disabled. An outer retry is explicit and
every reserved attempt counts against the immutable ledger ceiling.

The full scoreboard has 33 conditions × 2,000 targets = 66,000 logical decisions
per dataset. Physical requests are deduplicated only when their full frozen
request identities match. Exact scoreboard physical counts and ceilings can be
finalized only after the selected-global artifact exists. Token use and latency
come from returned results; no dollar bill is inferred from local estimates.

Live collection requires a separate committed preregistration binding the
protocol and preflight hashes, explicit human confirmation, and a maximum-new
attempt cap. A successful development run does not authorize scoreboard calls.
Missing or malformed cells remain visible; an incomplete matrix is not a finding.

## Analysis and limits on conclusions

The two named primary contrasts are retrieval minus mean random performance
at 16 examples per label, and mean random performance at 64 minus 1 example
per label. AG News uses accuracy; Emotion uses macro-F1. Publish all condition
and draw results, not only the selected winner. Secondary outputs include
macro-F1/accuracy, per-class recall, multiclass log loss and Brier score,
top-label ECE/reliability, probability coverage, latency, attempts, and token use.
Returned provider confidence is preserved separately, not synthesized from
probabilities or substituted for a missing probability distribution.

The implemented hierarchical bootstrap pairs target resampling and draw
resampling across comparison arms. Its 95% intervals are nominal per contrast.
The new protocol explicitly declares descriptive inference without a family
correction or formal significance test. The two comparisons remain named in
advance, but neither their intervals nor secondary comparisons are
familywise-adjusted. Do not manufacture p-values from interval endpoints.
Previously frozen Holm-labelled protocols remain readable for provenance and
must report the correction unavailable without valid prespecified p-values;
they are not relabelled as the new design.

Bootstrap interpretation assumes held-out targets represent the intended
population and treats the five random draws as resampling units. Both arms
reuse the same sampled targets and, where applicable, draws. The single
retrieval context policy is not resampled as five independent policies.
Only five draws limit how well their variability can be characterized.

Initial conclusions will be conditional on these datasets, candidate pools,
policies, model version, and single display rule. The ordering follow-up must
hold selected membership fixed and anchor to completed initial results. It is
not a reason to retune the initial scoreboard or skip to other models first.
