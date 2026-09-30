# Initial Jev capability pilot

Status: frozen pilot design; live approval granted by the human owner on
2026-09-30 for at most 21 physical attempts per dataset (42 total), no retries.
This is not a completed experiment or permission to start the full development
or scoreboard run. No live observations or performance findings are reported
here.

## Question and scope

Before optimizing contexts, can the frozen Jev route accept and return a valid
decision for our zero-shot and large-context requests? This is an operational
check, not a test of whether examples improve ground-truth agreement.

Use the initial study's unchanged task, label order, candidate/development
partitions, nested random-balanced membership, and canonical display order.
For each of the four positive sizes and five seeds, select the development
request with the largest `whitespace-request-estimate-v1` estimate. Break ties
by the smallest logical request ID. Add zero-shot on the globally largest
selected target, using the same task and empty `labeled_examples`.

Each dataset has 21 logical cases and 21 distinct full request fingerprints.
These are actual cache-only enumerations, not assumed counts. AG News uses
0, 4, 16, 64, and 256 total examples; Emotion uses 0, 6, 24, 96, and 384.
The largest local request estimates are 9,828 and 7,185 respectively. Estimates
are not provider tokens, prices, or proof of fitting the provider's limits.
Only candidate/development source rows are loaded; scoreboard targets cannot
enter this pilot. Selected targets are `train-49519` and `train-6108` respectively.

## Frozen identities

AG News:

- Protocol: `914b7e251379553f00b2040c1d6b63c68dd8c41424041f9ed7eddeb070006d63`
- Source optimization preflight: `8fdd7cb6613c0cf29db31dddf4f5fe70bc07bc7c6e99e7476d0175743f262432`
- Pilot: `fca230f241513dfa79d7a69da39cbc7f47e3366f4308af0beeadecaa75a41f2c`
- Text-free plan: [pilots/ag_news.plan.json](pilots/ag_news.plan.json)

Emotion:

- Protocol: `af1d49119eda713c08ecfa72c7d45134f219e3c6ff81c728217c606d3267297e`
- Source optimization preflight: `e3d09d8089264ef8e3644ec9a03bc4a6512d5f2e9545a9145e87d2e96c69d052`
- Pilot: `3de0eb439cb11147830398c2307994f4549224cfc5b06a228970604bef11133a`
- Text-free plan: [pilots/emotion.plan.json](pilots/emotion.plan.json)

Dataset revisions, manifest bindings, and every exact example/target ID are in
the committed manifests and pilot plans. Reproduce the source protocols and
preflights using the README's cache-only commands before deriving the pilots.
Collection regenerates and compares the entire source plan before constructing
any provider. A checksum in this document is not enough on its own.

Provider configuration is `jev-1.13.0`, SDK base `https://api.typesafe.ai`,
timeout 30 seconds, TypeSafe SDK 0.7.1, and core commit
`6137fa185a1a98afa84b5e6d5948d1780df5d56d`. Its public transport fingerprint is
`109e9d3978f49efd37d4d10beff240396895059a4e4cb7b72ce2bef1a7c06746`.
The live gate verifies installed package/core provenance; a moving model alias,
editable core installation, or caller-supplied identity alone is insufficient.

## Bounded execution and failure rules

The proposed hard ceiling is 21 physical attempts per dataset, 42 across both.
Both SDK and outer retries are disabled. Every reserved attempt counts, even
if it fails or is interrupted. Resume replays completed requests and may attempt
only previously unattempted requests; it cannot retry failed or uncertain ones.
No automatic truncation, substitution, or model fallback is allowed.

Live collection requires explicit human approval of this small pilot, a
byte-for-byte committed copy of this preregistration, matching frozen inputs,
`--confirm`, a cumulative ceiling of exactly 21, and an explicit maximum-new
attempt cap. Keep a separate ledger for each dataset. Full development ledgers
have different preflight identities and cannot be authorized by pilot approval.
Pilot responses are not automatically imported into full-study ledgers; any
later collection would account separately for repeated physical calls.

## Reporting and follow-up

Publish completion/failure/malformed/missing counts, returned usage, latency,
probability-distribution availability, explicit-confidence availability, and
the largest successful observed context. Missing usage remains unavailable;
do not estimate dollar cost. Success describes these requests only, not the
provider's maximum context length or reliability over other targets.

Pilot compatibility reports do not compute accuracy, F1, calibration, or
ground-truth effects, and do not publish source text or credentials. Operational
failures become adapter defects or explicit capability exclusions. Any changed
protocol, context ladder, or new attempt budget must be frozen and approved
separately before further calls. A successful pilot does not authorize the
8,000/12,000-request development selection or the later scoreboard studies.
