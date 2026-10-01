# Initial Jev results: example selection versus number of examples

Status: complete descriptive results for the preregistered
[scoreboard run](INITIAL_JEV_SCOREBOARD_RUN.md), model `jev-1.13.0`, canonical
display order. Text-free per-condition numbers are in
[results/ag_news.summary.json](results/ag_news.summary.json) and
[results/emotion.summary.json](results/emotion.summary.json). Intervals are
nominal paired 95% bootstrap intervals over targets and draws, per contrast, not
multiplicity-adjusted, and no significance test is claimed.

## What was asked

Does *which* labeled examples go into the context matter, or only *how many*?
Two preregistered contrasts: (1) per-target lexical retrieval minus the mean of
five random-balanced draws at 16 examples per label; (2) mean random-balanced
at 64 minus 1 example per label. Each dataset: 33 conditions × 2,000 held-out
targets = 66,000 decisions, 58,000 distinct Jev requests, all completed.

## Headline

| | AG News (accuracy) | Emotion (macro-F1) |
|---|---|---|
| Zero-shot | 0.878 | 0.509 |
| Best condition | retrieval, 16/label: **0.905** | retrieval, 64/label: **0.649** |
| (1) Selection: retrieval − random @16 | **+0.014** (0.006 to 0.022) | **+0.062** (0.043 to 0.081) |
| (2) Size: random 64 − 1 per label | +0.006 (−0.003 to 0.015) | **+0.082** (0.066 to 0.098) |

- **Per-target retrieval was the best selector at every size on both datasets.**
  On Emotion, retrieval with 16 examples per label (0.629) beat random with 64
  (0.601).
- **Number of examples:** large and clear on Emotion; on AG News the zero-shot
  baseline is already high (0.878) and the size effect's interval includes zero.
- **Development-selected fixed context** (best of 20 development trials, then
  frozen) was indistinguishable from random draws on the held-out scoreboard
  (Emotion 64/label: 0.603 vs 0.601; AG News 16/label: 0.894 vs 0.891). The
  development search did not buy a measurable benefit here.

Full per-condition tables are in the summary files, including log loss and
calibration error.

## Limits on these conclusions

- One model and one display order. The ordering study has not been run.
- Emotion's scoreboard source was exposed in earlier project work, so its
  results are exploratory, not confirmatory. "Fresh" means unexposed in this
  project's records, not absent from pretraining.
- Retrieval and prototypes are lexical policies, not learned embeddings.
- Differences of about one point on AG News are close to the interval widths;
  read them as suggestive.
- Random draws are nested across sizes, so size comparisons are within-draw.

## Collection record

Both scoreboards completed in a second, fresh collection (Run 2) with the
frozen protocol and preflight identities. A first collection (Run 1) was cut
short when the provider account ran out of credits (HTTP 402); its partial
ledgers are not used. Run 2 stopped once more on the same cause for AG News and
resumed from its ledger without repeating completed requests; Emotion's single
failed request was retried once. The collector now aborts on HTTP 401/402/403 or
25 consecutive failures, builds request plans once at start-up, and supports
bounded concurrent requests. Dollar cost is not derivable from returned
usage here; use the provider's billing records.
