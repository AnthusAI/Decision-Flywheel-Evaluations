# Decision Flywheel Evaluations

Reproducible evaluation scaffolding for Decision Flywheel context policies
across decision models and labelled datasets. It is not yet a complete multi-model
study. The initial Jev selection-by-size study has completed on AG News and
Emotion; see [INITIAL_JEV_RESULTS.md](studies/INITIAL_JEV_RESULTS.md). Ordering,
Kev and Laya comparisons have not been run.

Optimization scope is native to Decision Flywheel and provider-neutral; this
repository does not include DSPy integration. This is a scope boundary, not a
new study finding.

This is intentionally separate from
[Decision Flywheel](https://github.com/AnthusAI/Decision-Flywheel): the core
repository supplies reusable policies and adapters; this one freezes study
designs, split manifests, response-cache metadata, and aggregate findings.

## Current status

The initial Jev study is complete. Later experimental notes cover the
[unified flywheel](studies/UNIFIED_FLYWHEEL_RESULTS.md) and
[FOMC rubric teaching](studies/FOMC_RUBRIC_RESULTS.md). These are separate
investigations, not cross-model or demonstration-order results.
The [arXiv feature-engineering lab notes](studies/arxiv_feature_engineering/README.md)
record the real single-question comparisons and their small-sample limitations.
The current offline build is a streaming-SME evaluation: classify before
feedback, review a seeded fraction of the stream, then update the classifier
from reviewed labels and optional explanations. Ordering sensitivity and
cross-model comparisons remain deferred work.

## Amazon Reviews Stage 0 workflow

Stage 0 is a completed paid screen—not a learning result. Its concise lab notes
and text-free aggregate live in [`studies/amazon_reviews/`](studies/amazon_reviews/).
It
uses the committed text-free freeze of the first 1,500 pool-order review IDs,
their hashes and their fixed stream/held-out roles. The 300 held-out items are
screened under S and F for at most 600 Jev attempts in total; the durable cache
and ledger make an interrupted screen resumable. A failed attempt remains an
attempt, and provider retries are disabled.

The consistency gate compares 100 fixed IDs using the primary and second
simulated-SME cache records. It is complete: 99 of 100 labels agree. This is
an agreement check for a simulated policy application, not a claim of human
ground truth.

The cache-only preflight does not construct a client or read the private policy:

```bash
make PYTHON=/path/to/python reviews-stage0-preflight
```

Collection remains explicit and requires a bounded per-invocation cap and the
exact confirmation token:

```bash
make PYTHON=/path/to/python reviews-stage0-run \
  STAGE0_MAX_NEW=600 CONFIRM=--confirm
```

The command does not authorize Stage 1 or Stage 2; their caps default to zero.
It also does not select a new split, retune the task, or publish a result.
Repository code is MIT-licensed, but Amazon review text and the private policy
remain local inputs and are not redistributed.

## Offline streaming fixtures and reports

The streaming command has **no live mode**. It requires an explicitly synthetic
fixture, installs the network guard, and never reads the private SME policy or
credentials. This tiny fixture tests scheduling and reporting with scripted
predictions; its numbers are not evidence of learning or a benchmark result:

```bash
make PYTHON=.venv/bin/python reviews-stream-fixture \
  STREAM_FIXTURE=fixtures/stream.synthetic.json \
  STREAM_RUN_DIR=var/stream-fixture STREAM_OUTPUT=var/stream-fixture.run.json \
  STREAM_MAX_NEW=0
make PYTHON=.venv/bin/python reviews-stream-report \
  STREAM_INPUT=var/stream-fixture.run.json STREAM_OUTPUT=var/stream-fixture.report.json
.venv/bin/python scripts/reviews_stream.py report \
  --input var/stream-fixture.run.json --output var/stream-fixture.report.md --format markdown
```

Outputs keep labels, identifiers, hashes, counters and measurements, not review
text, SME explanations or policy prose. Reports provide window-100 and cumulative
prequential curves, checkpoint summaries, descriptive paired intervals, actual
recorded attempts/token fields, and coverage/problems. Missing checkpoints stay
missing; a four-item fixture does not invent a 300-item result. Missing probability
distributions do not invalidate label-based accuracy or macro-F1, and are not
invented for calibration. Existing outputs require `OVERWRITE=--overwrite` or
the matching CLI flag.

The learning-runtime integration is checked separately with the pinned clean
Jev-Flywheel clone and fake Jev/analyst clients. The streaming Kanbus tasks track
its completion and the subsequent live-stage approval gates; this fixture command
does not authorize a live SME, classifier, or analyst run.

To exercise actual fitting, native list optimization and immutable serving rather
than scripted predictions, use the separate **synthetic runtime** command. It
requires the interpreter with the pinned Jev-Flywheel dependencies, a fresh empty
run directory, and an explicit shared request ceiling (these are fake requests,
not paid API calls):

```bash
make reviews-stream-runtime \
  UF_PYTHON=/path/to/python-with-scikit-learn-and-tactus \
  STREAM_RUN_DIR=var/fresh-stream-runtime STREAM_OUTPUT=var/runtime.run.json \
  STREAM_MAX_NEW=400
```

The default exercise uses 32 synthetic arrivals, full review and eight scoreboard
items, with disjoint development items. All arms share seeded batches of 1–10
arrivals and the same review plan. A ledger counts every physical fake request
across all arms; reaching the cap leaves missing predictions unavailable. The
runtime cannot resume an old workspace: even `--overwrite` does not permit
reuse of previous feedback. Use a new directory for another exercise.

Per-item predictions use a saved, verified classifier and calibrated bundle
probabilities, with its hash recorded before feedback. Fitting and steering use
at most the latest 150 reviewed items; the steering view excludes older/future
feedback and raw historical analyst replies. A new example context is published
only together with a matching fitted head. The offline specs separately exercise
late-history steering; a synthetic run need not trigger every optional action.

This remains an offline prototype, not the completed streaming study.
Paired cached replays before and after the streaming changes preserve the
Planted, Emotion and FOMC predictions, summaries, logs, manifests and frozen
bundles byte-for-byte, with zero requests. Workspace event/lineage files differ
only in their wall-clock timestamps; this is not whole-directory byte equality.
Separately, Emotion's historical live baseline arm is not fully reproduced by
its cached replay (13 of 100 baseline predictions differ). That unresolved
historical discrepancy is not evidence of a regression caused by streaming.
Cached SME feedback integration and the live-stage approval gates remain under
test before any new collection can start.

## Native command-line workflow

The native workflow is deliberately explicit and offline-first:

```bash
make preflight  # shows required frozen local inputs; performs no download or call
make select     # derives native selected-global evidence from a complete local development ledger
make run        # shows confirmation, budget, preregistration, and input requirements
make report     # shows required sanitized-observation inputs
```

With actual paths supplied, `preflight` rehydrates caller-provided local rows
against a pinned text-free manifest and writes only request IDs, core plan
fingerprints, and counts. `run` requires a committed preregistration containing
the protocol and preflight hashes, an exact `--confirm`, an explicit cumulative
attempt ceiling and `--max-new` cap before the JEV adapter can be constructed.
It writes sanitized observations; `report` then regenerates metrics and paired
effects offline. Incomplete cells are labelled incomplete rather than findings.
Retries are disabled by default. A retry or recovery requires the explicit
`--retry-failed` or `--recover-uncertain` flag plus a positive
`--max-retries-per-request`; the approved cumulative ceiling includes those
possible retries and remains immutable when a ledger is resumed.
Reported paired intervals are reproducible nominal per-contrast 95% intervals.
New protocols explicitly use descriptive inference, not a formal significance
test or multiplicity-adjusted intervals. Legacy Holm-labelled protocols remain
readable, but correction is unavailable without valid prespecified p-values.

For a prepared benchmark, use the ignored pinned Arrow cache directly instead
of exporting a JSON copy of article text:

```bash
make PYTHON=.venv/bin/python preflight \
  PROTOCOL=study.protocol.json MANIFEST=studies/manifests/ag_news.json \
  DATASET_CACHE=.data/huggingface OUTPUT=study.preflight.json
```

`DATASET_CACHE` is read only from the exact dataset/config/revision/split Arrow
paths named by the manifest. Collection and analysis never download data, search for a
latest revision, or write source text into study artifacts. `ROWS=*.fixture.json`
remains available only for synthetic or caller-owned local fixtures; supply one
of `ROWS` or `DATASET_CACHE`, never both.

## Acquiring and reproducing the dataset manifests

Dataset acquisition is a separate, explicit network-enabled setup step. It
downloads the pinned AG News `default` and Emotion `split` train/test inputs
into the ignored local cache, without constructing a model or exporting text.
Hugging Face may also populate sibling splits while building a configuration;
only train/test are requested or used by our preparation path:

```bash
.venv/bin/pip install -e '.[data]'
make PYTHON=.venv/bin/python download CONFIRM=--confirm
```

Without `CONFIRM=--confirm`, this target exits before downloading. Dataset
revisions are fixed in the package and manifests, not resolved from `main`.
The command checks the exact Arrow paths used by cache-only collection.

Normal collection uses the committed manifests; it does not need to regenerate
them. To independently reproduce their selection, first obtain the historical
study repository, then prepare into an ignored output directory:

```bash
git clone https://github.com/AnthusAI/Few-Shot-Jev.git ../Few-Shot-Jev
.venv/bin/python scripts/prepare_datasets.py prepare \
  --historical-repository ../Few-Shot-Jev \
  --output-dir .data/reproduced-manifests
cmp studies/manifests/ag_news.json .data/reproduced-manifests/ag_news.json
cmp studies/manifests/emotion.json .data/reproduced-manifests/emotion.json
```

Preparation is cache-only. It reads the historical AG News exposure inventory
at the recorded commit, rather than trusting that repository's current branch.
A missing cache or history is an error, not permission to download or weaken
the exclusion rule. AG News excludes the prior scoreboard and exact normalized
matches. Emotion remains exploratory because its official test split was
already exposed in the earlier investigation.

Dataset attribution and terms are separate from this repository's MIT license:

- [Pinned AG News card](https://huggingface.co/datasets/fancyzhx/ag_news/blob/eb185aade064a813bc0b7f42de02595523103ca4/README.md): the topic benchmark is attributed to Xiang Zhang, Junbo Zhao, and Yann LeCun (2015). The card marks its license unknown and describes research/non-commercial uses; this repository does not relicense or redistribute its news text.
- [Pinned Emotion card](https://huggingface.co/datasets/dair-ai/emotion/blob/cab853a1dbdf4c42c2b3ef2173804746df8825fe/README.md): cite Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition* (2018). The card specifies educational and research use only.

## Native selection and live collection

The [initial Jev design](studies/INITIAL_JEV_STUDY.md) describes the proposed
selection-by-size investigation, descriptive analysis, and unresolved approval gates.
It is not a completed study or permission to collect.

The [initial preflight record](studies/INITIAL_JEV_PREFLIGHT.json) records an
actual cache-only enumeration: 8,000 AG News and 12,000 Emotion development
requests, with no model calls. Its estimates are not provider token usage,
pricing, or proof that every context fits. The compact record is committed;
the large request enumerations are regenerated locally, not redistributed.

After explicitly acquiring the pinned dataset cache, reproduce the proposed
AG News protocol and enumeration without an API key:

```bash
mkdir -p .data/proposals
.venv/bin/python scripts/freeze_jev_study.py \
  --manifest studies/manifests/ag_news.json \
  --output .data/proposals/ag_news.protocol.json \
  --model-version jev-1.13.0 --base-url https://api.typesafe.ai \
  --timeout-seconds 30 \
  --adapter-revision 6137fa185a1a98afa84b5e6d5948d1780df5d56d \
  --package-revision typesafe-sdk-0.7.1 \
  --max-per-label 64 --max-model-calls 8400
make PYTHON=.venv/bin/python preflight \
  PROTOCOL=.data/proposals/ag_news.protocol.json \
  MANIFEST=studies/manifests/ag_news.json \
  DATASET_CACHE=.data/huggingface \
  OUTPUT=.data/proposals/ag_news.preflight.json STAGE=optimization
```

For Emotion, replace `ag_news` with `emotion` and the proposed ceiling `8400`
with `12600`. Compare protocol identities and preflight checksums with the
compact record. Re-authoring an existing protocol requires explicit
`--overwrite`; none of these commands approves collection. Tests validate the
committed inputs and compact record without requiring the downloaded datasets
or generated request enumerations.

To author an unapproved development protocol offline, use
`scripts/freeze_jev_study.py --help`. The script requires a byte-for-byte
committed benchmark manifest, an exact `jev-N.N.N` version, public SDK **base**
URL (not the `/v1/systemone` endpoint), timeout, adapter/package revisions,
declared capability, and explicit call ceiling. It fixes the task templates,
label order, native policies, seed/size ladder, and canonical initial display.
It rejects secret-bearing URLs and refuses to overwrite an output without
`--overwrite`. Authoring a protocol does not create a preregistration or approve
calls. Optimization preflight/run/select use only candidate/development source
rows; they do not open the official-test Arrow file.

`select` has no provider path: it reopens the exact bounded optimization ledger,
replays complete sanitized development decisions through the native optimizer,
and writes both a derived protocol and a text-free link to the original
optimization protocol/preflight. It refuses missing, failed, or malformed
development evidence. A separate scoreboard preregistration and preflight must
then bind that derived protocol; selection does not rewrite a hypothesis.

The initial live adapter is JEV-only and must use one exact frozen
`jev:<model-version>@transport-<sha256>` identity. A live run requires an
explicit endpoint, timeout, and adapter revision whose public, credential-free
transport fingerprint exactly matches that identity. Install it explicitly with
`pip install -e '.[jev-live]'` (the evaluation extra pins TypeSafe SDK 0.7.1).
Pass `PACKAGE_REVISION=typesafe-sdk-0.7.1` to `make run`. Before opening a
ledger or constructing a provider, the live path verifies the installed SDK
version and the core's installed Git commit metadata. Core installs with
unknown provenance or editable source are rejected for live collection; use
the exact Git-pinned dependency declared in this repository. The evaluation
harness itself may remain editable for development.
The core plan fingerprint is auditable context provenance; it
is not a promise of provider transport-byte identity, and this CLI does not yet
offer Kev/Laya live collection or cross-provider cache reuse.

Source row text is input-only. Benchmark artifacts (protocols, preflights,
ledgers, observations, and reports) remain text-free. `rows-fixture` writing is
available solely for synthetic local specs, not for exporting a dataset.
Explicit provider confidence is preserved through adapters, the cache, and
observation JSON independently of probabilities; older observation JSON without
that field remains readable with confidence unavailable.

## Bounded Jev compatibility pilot

The [pilot preregistration](studies/INITIAL_JEV_PILOT.md) freezes 21 distinct
development-only requests for each dataset: zero-shot and the largest local
request estimate for each size/seed combination. It does not use scoreboard
source rows or measure ground-truth performance. The two committed text-free
plans bind the exact initial protocols and optimization preflights.

After regenerating the proposed source protocol/preflight above, independently
reproduce the AG News pilot without credentials or model calls:

```bash
make PYTHON=.venv/bin/python pilot-preflight \
  PROTOCOL=.data/proposals/ag_news.protocol.json \
  MANIFEST=studies/manifests/ag_news.json \
  PREFLIGHT=.data/proposals/ag_news.preflight.json \
  DATASET_CACHE=.data/huggingface OUTPUT=.data/proposals/ag_news.pilot.json
cmp studies/pilots/ag_news.plan.json .data/proposals/ag_news.pilot.json
```

Replace `ag_news` with `emotion` for the other pilot. All three pilot commands
require explicit `OVERWRITE=--overwrite` to replace an existing output file.
These preflight commands only enumerate requests.

Only after explicit human approval, with the preregistration committed and the
live dependencies installed, the bounded AG News invocation is:

```bash
make PYTHON=.venv/bin/python pilot \
  PROTOCOL=.data/proposals/ag_news.protocol.json \
  MANIFEST=studies/manifests/ag_news.json \
  PREFLIGHT=.data/proposals/ag_news.preflight.json \
  PILOT=studies/pilots/ag_news.plan.json DATASET_CACHE=.data/huggingface \
  LEDGER=.data/proposals/ag_news.pilot.sqlite \
  PREREGISTRATION=studies/INITIAL_JEV_PILOT.md \
  OUTPUT=.data/proposals/ag_news.pilot.observations.json \
  PROVIDER_MODEL=jev-1.13.0 BASE_URL=https://api.typesafe.ai TIMEOUT_SECONDS=30 \
  ADAPTER_REVISION=6137fa185a1a98afa84b5e6d5948d1780df5d56d \
  PACKAGE_REVISION=typesafe-sdk-0.7.1 ATTEMPT_CEILING=21 MAX_NEW=21 CONFIRM=--confirm
```

Emotion requires its own approved 21-attempt ceiling and separate ledger. Both
pilots together propose at most 42 paid attempts, not approval for the full
study. There are no pilot retries or uncertain-attempt recovery. On resume,
`MAX_NEW` must fit the unspent ledger ceiling; completed requests replay without
constructing a provider. Pilot ledgers cannot authorize full-study collection.

An offline compatibility summary uses no source dataset text or provider:

```bash
make PYTHON=.venv/bin/python pilot-report \
  PROTOCOL=.data/proposals/ag_news.protocol.json PILOT=studies/pilots/ag_news.plan.json \
  OBSERVATIONS=.data/proposals/ag_news.pilot.observations.json \
  OUTPUT=.data/proposals/ag_news.pilot.compatibility.json
```

The summary describes response fields, failures, actual returned usage, latency,
and successful observed contexts. It deliberately reports no accuracy or other
benchmark metrics. A successful pilot does not establish a context limit or
authorize the development optimization, ordering, or cross-model studies.

## Ordering follow-up

The [ordering preregistration template](protocols/ORDER_SENSITIVITY.md) records
the next investigation, before other model comparisons. This is not an actual
ordering preregistration. The initial scoreboard is now complete, but the actual
ordering plan and separate collection approval have not been frozen.

The offline planner requires complete initial single-Jev canonical scoreboard
observations and an exact regenerated initial preflight. It preserves the task,
choice options, model, targets, context sizes, selection draws, and example
membership. Only presentation changes. Shuffled order seeds are independent
of example-selection draws and retain distinct logical labels. Every positive
initial cell is crossed with every declared treatment; zero-shot appears once.

After completing the initial scoreboard, author an unapproved follow-up plan:

```bash
make PYTHON=.venv/bin/python ordering-preflight \
  PROTOCOL=initial.derived.protocol.json MANIFEST=studies/manifests/ag_news.json \
  PREFLIGHT=initial.scoreboard.preflight.json \
  INITIAL_OBSERVATIONS=initial.scoreboard.observations.json \
  RESULT_REFERENCE=results/ag_news.initial.json ORDER_SEEDS='0 1 2 3 4' \
  DATASET_CACHE=.data/huggingface OUTPUT=.data/ag_news.ordering.json
```

Canonical, interleaved, and reversed seed-zero treatments are always included,
along with the explicitly named shuffled seeds. The result contains only
request provenance and ordered example IDs, never source text. It reports exact
logical and deduplicated physical counts and makes no model calls. A pilot,
development run, incomplete scoreboard, changed wording, or missing context
artifact cannot replace the initial anchor. Output replacement requires
`OVERWRITE=--overwrite`.

Live ordering collection is a separate approval. Commit an actual preregistration
binding the initial protocol, initial preflight, complete initial observations,
and ordering-plan checksums before using `make ordering-run`. That target requires
the same initial evidence plus `ORDERING`, a dedicated `LEDGER`, `PREREGISTRATION`,
the frozen provider and transport versions, an explicit cumulative
`ATTEMPT_CEILING`, a per-invocation `MAX_NEW`, and `CONFIRM=--confirm`.
It validates the complete initial anchor again before constructing a provider.
The ceiling counts attempts, including failed or uncertain calls; retries are
disabled unless explicitly bounded. Resume uses the ordering ledger, not the
initial-study ledger. Canonical responses are collected again and count toward
the ceiling; they are not silently imported from the initial study.
No ordering request ceiling is approved by this README or the template.

An offline report from separately collected, sanitized ordering observations is:

```bash
make PYTHON=.venv/bin/python ordering-report \
  PROTOCOL=initial.derived.protocol.json MANIFEST=studies/manifests/ag_news.json \
  ORDERING=.data/ag_news.ordering.json OBSERVATIONS=ordering.observations.json \
  OUTPUT=.data/ag_news.ordering.report.json BOOTSTRAP_SEED=0 RESAMPLES=1000
```

Reports show every condition/draw/order, pooled and mean-draw metrics, deduplicated
physical totals, and nominal paired intervals for fixed orders and the declared
shuffled family. They never choose a winning order or claim significance.
Incomplete cells remain visible and make the corresponding intervals unavailable.
These offline commands do not approve ordering collection or cross-model runs.

Development optimization is a separate native-core step: before scoreboard
preflight, it must produce a complete, validated `SelectedGlobalArtifact` from
trusted development evidence. Scoreboard execution refuses protocols without
that validated artifact.

## Planned initial model matrix

| Model | Adapter route | Context status |
| --- | --- | --- |
| Jev | TypeSafe System One | structured labelled context |
| Kev | TypeSafe-compatible local System One endpoint | structured labelled context |
| Laya | local `system_one` adapter | zero-shot now; few-shot pending an upstream context contract |

“Support” means a model must be named and versioned in every response row; it
does not mean that every model has identical context capabilities. A study must
state its compatible models before collection.

## Rules for every study

1. Pin the upstream dataset revision and commit a text-free split manifest.
2. Partition trusted labels into candidate pool, policy-development set, and a
   protected scoreboard before any optimization.
3. Freeze the task wording, label order, policy candidates, context budget,
   primary metric, and request ceiling before live calls.
4. Cache complete request fingerprints and response metadata—not source text or
   credentials—and make collection resumable.
5. Publish all eligible model/policy cells, not only the best-looking one.

The [Few-Shot-Jev study](https://github.com/AnthusAI/Few-Shot-Jev) is the
experimental record that motivated this harness. Its findings are not copied or
reinterpreted here.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
make PYTHON=.venv/bin/python test
```

Use the same interpreter override for the native commands in a fresh clone,
for example `make PYTHON=.venv/bin/python preflight ...`.

The pinned Decision Flywheel core dependency is fetched from GitHub during
installation. Once installed, the test suite uses only local code and fake
fixtures: it does not require network access, credentials, datasets, or model
calls.

## Repository tooling

This repository uses [Kanbus](https://github.com/AnthusAI/Kanbus) for Git-backed
project tasks and `python-semantic-release` for conventional-commit GitHub
releases. Install the local tools with `make install-tools`, then use
`kanbus list` to inspect the board.

The initial study template is [protocols/CONTEXT_POLICY_MATRIX.md](protocols/CONTEXT_POLICY_MATRIX.md).

## License

MIT for the harness. Datasets and model weights retain their upstream terms.
