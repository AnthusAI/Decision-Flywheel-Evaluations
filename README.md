# Decision Flywheel Evaluations

Reproducible evaluation scaffolding for Decision Flywheel context policies
across decision models and labelled datasets. It is not yet a completed live
study or a preregistration: the model/policy matrix and collection harness are
still being filled in.

Optimization scope is native to Decision Flywheel and provider-neutral; this
repository does not include DSPy integration. This is a scope boundary, not a
new study finding.

This is intentionally separate from
[Decision Flywheel](https://github.com/AnthusAI/Decision-Flywheel): the core
repository supplies reusable policies and adapters; this one freezes study
designs, split manifests, response-cache metadata, and aggregate findings.

## Current status

The repository currently provides frozen protocol and split-manifest contracts,
offline preflight checks, text-free observation/metric/reporting utilities, and
synthetic tests. Live collection, complete matrix results, and any confirmatory
claims remain planned work.

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
They are not multiplicity-adjusted, and no familywise confirmatory significance
claim is made until a frozen inferential plan supplies valid prespecified p-values.

For a prepared benchmark, use the ignored pinned Arrow cache directly instead
of exporting a JSON copy of article text:

```bash
make PYTHON=.venv/bin/python preflight \
  PROTOCOL=study.protocol.json MANIFEST=studies/manifests/ag_news.json \
  DATASET_CACHE=.data/huggingface OUTPUT=study.preflight.json
```

`DATASET_CACHE` is read only from the exact dataset/config/revision/split Arrow
paths named by the manifest. The workflow never downloads data, searches for a
latest revision, or writes source text into study artifacts. `ROWS=*.fixture.json`
remains available only for synthetic or caller-owned local fixtures; supply one
of `ROWS` or `DATASET_CACHE`, never both.

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
`pip install -e '.[jev-live]'`. The core plan fingerprint is auditable context provenance; it
is not a promise of provider transport-byte identity, and this CLI does not yet
offer Kev/Laya live collection or cross-provider cache reuse.

Source row text is input-only. Benchmark artifacts (protocols, preflights,
ledgers, observations, and reports) remain text-free. `rows-fixture` writing is
available solely for synthetic local specs, not for exporting a dataset.

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
