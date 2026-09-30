# Decision Flywheel Evaluations

Preregistered, reproducible evaluations of Decision Flywheel context policies
across decision models and labelled datasets.

This is intentionally separate from
[Decision Flywheel](https://github.com/AnthusAI/Decision-Flywheel): the core
repository supplies reusable policies and adapters; this one freezes study
designs, split manifests, response-cache metadata, and aggregate findings.

## Initial model matrix

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
make test
```

## Repository tooling

This repository uses [Kanbus](https://github.com/AnthusAI/Kanbus) for Git-backed
project tasks and `python-semantic-release` for conventional-commit GitHub
releases. Install the local tools with `make install-tools`, then use
`kanbus list` to inspect the board.

The initial study template is [protocols/CONTEXT_POLICY_MATRIX.md](protocols/CONTEXT_POLICY_MATRIX.md).

## License

MIT for the harness. Datasets and model weights retain their upstream terms.
