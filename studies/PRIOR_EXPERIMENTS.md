# Prior experiment inventory

**Audit date:** 2026-09-30.  This is an inventory of the checked-in record in
[`AnthusAI/Few-Shot-Jev`](https://github.com/AnthusAI/Few-Shot-Jev/tree/59548114ecf45e817613c946af2cab6ceb675804), not a new benchmark result.  Source
checkout audited: commit `59548114ecf45e817613c946af2cab6ceb675804` (2026-09-30).
The published aggregates contain no source text; the local per-response caches
are intentionally not committed.

## Bottom line for future studies

- **Emotion has no untouched official evaluation split left in this record.**
  The complete official `test` split was the small-context scoreboard and the
  complete official `validation` split was the large-context scoreboard.  They
  are exposed and must not be described as a new pristine holdout.
- The selection study made a new 2,000-row scoreboard from Emotion `train`,
  but its results are published, so that partition is also exposed.  Its
  600-row selector-development partition is exposed to selection, and the
  remainder was the selection candidate pool.
- AG News has a recorded, exposed 2,000-row test scoreboard.  The runner only
  targets that deterministic sample, so other official test rows were not
  targets in the recorded scripts.  They are **not**, however, a previously
  frozen or manifested untouched scoreboard.  A future study may establish a
  disjoint manifest from the remaining rows after an exposure audit; until
  then, call it a prospective holdout, not an existing pristine one.
- Consequently, any new result on the recorded Emotion resources is
  replication/exploratory.  A confirmatory Emotion study needs additional,
  independently reserved data (or a separately acquired revision/task) and a
  frozen manifest before prompt/selector development.

## Study record

| Study | Dataset and revision | Target partition and target count | Demonstrations / selection and ordering | Primary endpoint and status | Recorded result evidence |
| --- | --- | --- | --- | --- | --- |
| AG News baseline/few-shot | `fancyzhx/ag_news` at `eb185aade064a813bc0b7f42de02595523103ca4` | 2,000 official `test` rows: 500/class, sampled independently by `random.Random(20260925)` in World, Sports, Business, Sci/Tech order, then sorted by class and upstream index | `train` only; exact normalized-text matches to the scoreboard excluded. Five draws, seeds 0–4, with nested 4/8/16-shot contexts and canonical class order. Same choice wording and criteria in every arm. | Preregistered primary: 4-shot minus zero-shot **accuracy**; 2,000 target-and-draw hierarchical bootstrap resamples, seed `20260925`. 8/16 are secondary. | Complete 32,000-response run: zero-shot accuracy .8795; five-draw 4-shot mean .8875; primary delta .0080, 95% interval [-.0018, .0184]. This is historical, not a new score. |
| Emotion small context | `dair-ai/emotion` at `cab853a1dbdf4c42c2b3ef2173804746df8825fe` | Complete official `test`, 2,000 rows, natural class imbalance | `train` only; five deterministic draws (seeds 0–4), canonical sadness, joy, love, anger, fear, surprise order. Conditions 0/6/12/24 shots (one/two/four examples per label). | Preregistered primary: 6-shot minus zero-shot **accuracy**; target-and-draw hierarchical bootstrap. Macro-F1 reported alongside accuracy. | 32,000 requests, model `jev-1.13.0`: zero-shot .5935 accuracy/.5002 macro-F1; 6-shot mean .5946/.5213; primary delta .0011, 95% interval [-.0115, .0128]. |
| Emotion large context | Same pinned Emotion revision | Complete official `validation`, 2,000 rows; the earlier `test` split was explicitly excluded | `train` only; five nested draws seeds 0–4, same canonical six-label order. 0/6/24/96/384 shots (0/1/4/16/64 per label). | Preregistered primary: 384-shot minus zero-shot **macro-F1**; target-and-draw hierarchical bootstrap, 2,000 resamples, seed `20260928`. Intermediate doses are secondary. | 42,000 analysis records, 41,958 unique model states (42 duplicate full-state fingerprints reused). Macro-F1 .5146 at zero-shot and .6137 five-draw mean at 384-shot; delta .09886, 95% interval [.07623, .12258]. |
| Emotion fixed-budget selection | Same pinned Emotion revision, but partitions the original `train` | New deterministic `train` scoreboard: 2,000 rows, labels sadness 583, joy 670, love 163, anger 270, fear 242, surprise 72 | Per label, shuffle with `random.Random(20260930 + label_offset)`; first 100 go to selector-development (600 total), next prescribed count to scoreboard, remainder to candidate pool. 96 examples/16 per label for every few-shot method; same wording and canonical label order. Five random contexts seeds 0–4; prototype; target-text-only per-label lexical retrieval. | Preregistered primary: scoreboard **macro-F1** at fixed 96-example budget. Development selected one global random context before scoreboard reading; it was descriptive, not promoted to primary. Paired target bootstrap vs random mean, 2,000 resamples, seed `20260930`. | 19,000 analysis records, 18,992 unique states. Development chose `random-0`. Scoreboard macro-F1: random mean .5690, prototype .5679, selected global .5616, retrieval .6964; retrieval-minus-random .12737, 95% interval [.10709, .14739]. |

The task wording documented by all three Emotion scripts is: classify only
`target.text` into its primary emotion; `labeled_examples` are training examples
of the intended categories and must not themselves be classified.  AG News uses
the corresponding instruction to classify only `target.text` into its four
named criteria.  The records above establish fixed question/label order within
each study; they do not test ordering sensitivity.

## Partition exposure and allowed interpretation

| Resource | Historical role | Exposure conclusion | Reuse now |
| --- | --- | --- | --- |
| AG News recorded test manifest, 2,000 rows | Scoreboard for completed zero/few-shot study | Exposed through results and manifest labels/hashes | Reproduce historical run only; exploratory/replication if rescored. |
| Remaining AG News official test rows | Not generated by the historical request plan | Not an existing registered holdout; needs a new deterministic, disjoint manifest and exposure check | Candidate for a future holdout only after preregistration and manifesting. |
| Emotion official `test`, 2,000 rows | Small-context scoreboard | Exposed | Historical reproduction or exploratory only. |
| Emotion official `validation`, 2,000 rows | Large-context scoreboard | Exposed | Historical reproduction or exploratory only. |
| Emotion selection `train` scoreboard, 2,000 rows | Fixed-budget selection scoreboard | Exposed; text-free manifest is checked in | Historical reproduction or exploratory only. |
| Emotion selection development, 600 rows | Selected the global random context | Exposed to tuning/selection | Never a scoreboard; do not use for final reporting. |
| Emotion selection candidate remainder | Candidate demonstrations, prototype statistics, and retrieval index | Used in selection; its labels/features are not fresh for this study | Reproduction candidate pool, not independent confirmation data. |

## Provenance and reproduction pointers

The exact creation/result commits distinguish plan from completed evidence:

| Study | Protocol commit | Result commit | Authoritative checked-in artifacts |
| --- | --- | --- | --- |
| AG News | [`798bafb`](https://github.com/AnthusAI/Few-Shot-Jev/tree/798bafb3afd19df2c735ea94d89269eb952e3d13) | [`34d1b69`](https://github.com/AnthusAI/Few-Shot-Jev/tree/34d1b69f8b0421777257f28386a4efe5edeca183) | [protocol](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/protocol/PREREGISTRATION.md), [aggregate](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/results/summary.json), [manifest](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/manifests/scoreboard.jsonl), [selection/runner](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/src/jev_fewshot/cli.py). |
| Emotion small | [`b7e160a`](https://github.com/AnthusAI/Few-Shot-Jev/tree/b7e160a251da82adc9c34657183088fb33356368) | [`067cfa7`](https://github.com/AnthusAI/Few-Shot-Jev/tree/067cfa786ba3ce59b7a617cf0872e5114d2b9bd2) | [protocol](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/protocol/EMOTION_PREREGISTRATION.md), [aggregate](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/results/emotion_summary.json), [runner](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/scripts/emotion_study.py). |
| Emotion large | [`34cd986`](https://github.com/AnthusAI/Few-Shot-Jev/tree/34cd9868cf41e1006edfb97f0eee07ac5fd67f5c) | [`1a395aa`](https://github.com/AnthusAI/Few-Shot-Jev/tree/1a395aa90baa577e827e99cf811a2e0caf24de9a) | [protocol](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/protocol/EMOTION_LARGE_CONTEXT_PREREGISTRATION.md), [aggregate](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/results/emotion_large_context_summary.json), [runner](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/scripts/emotion_large_context_study.py). |
| Emotion selection | [`552af1e`](https://github.com/AnthusAI/Few-Shot-Jev/tree/552af1e9bbdfcbfa0fd5dfa0c60eaf77813468f1) | [`5954811`](https://github.com/AnthusAI/Few-Shot-Jev/tree/59548114ecf45e817613c946af2cab6ceb675804) | [protocol](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/protocol/EMOTION_SELECTION_PREREGISTRATION.md), [aggregate](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/results/emotion_selection_summary.json), [manifest](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/manifests/emotion_selection_scoreboard.jsonl), [partition/selector code](https://github.com/AnthusAI/Few-Shot-Jev/blob/59548114ecf45e817613c946af2cab6ceb675804/scripts/emotion_selection_study.py). |

To reproduce a historical aggregate, check out source commit
`59548114ecf45e817613c946af2cab6ceb675804`, install its declared dependencies,
and use the corresponding `preflight` command to reconstruct requests from the
pinned dataset revision.  The checked-in summaries can be inspected without
network.  Recreating a live outcome requires the ignored response cache or new
explicitly approved model calls; it must be labelled a reproduction, not merged
with the historical summary.  Do not load `.env` for the inventory or preflight.

## What is not evidenced

- The aggregate files do not preserve raw responses, model-request timestamps,
  or a full model identifier for AG News, large-context Emotion, or selection
  Emotion.  `jev-1.13.0` is explicitly recorded only in the small Emotion
  aggregate (and in the narrative for the other completed runs).  Treat exact
  runtime-model provenance for those other aggregates as incomplete unless
  their ignored caches can be audited safely.
- No checked-in manifest identifies a fresh post-study holdout.  The AG News
  remaining-row observation follows the recorded request-plan code, not a
  dedicated exposure manifest.
- The summaries support the numerical historical results listed above.  They
  do not establish cross-model performance, generalization to another dataset,
  or an ordering-effect conclusion.
