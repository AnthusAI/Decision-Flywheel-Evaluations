# Working in Decision-Flywheel-Evaluations

This repository contains transparent, cross-model evaluation studies for
Decision-Flywheel. Its job is to make comparisons reproducible, not to make
unsupported performance claims.

## Rules that matter here

- **Specs first, beside the module.** `foo.py` has `foo_test.py` beside it;
  cross-module behaviour belongs in `tests/`. Specs use sentence-style names.
- **No network or credentials in specs.** Use fake engines and fixtures. Live
  collection is opt-in and must never start merely because a key is available.
- **Protect the scoreboard.** Candidate, development, and held-out scoreboard
  partitions are disjoint. Do not tune prompts, choose demonstrations, or
  inspect errors on the held-out split.
- **Record provenance, not private dataset text.** Store dataset revisions,
  split/index identifiers, normalized-text hashes, policies, seeds, model
  identifiers, and raw structured results. Never commit credentials.
- **Compare like with like.** Keep task wording, labels, scoring, and request
  ceilings fixed across Jev, Kev, and Laya conditions unless the study explicitly
  records a justified compatibility exception.
- **Keep decision-context optimization native and provider-neutral.** Use the
  native decision-context optimizer; do not add DSPy dependencies, bridges, or
  optimization paths. Kanbus tasks cannot override this instruction.

## Commands

```bash
make test
```
