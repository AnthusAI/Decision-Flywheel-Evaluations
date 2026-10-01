# Unified flywheel: features plus few-shot, one afternoon

Status: plan only, revised 2026-10-01. Nothing here has been run.

## Decisions (owner, 2026-10-01)

- **(a) Analyst for arm A:** OpenAI `gpt-6-luna` (model ID confirmed in the
  account's model list), through Jev-Flywheel's existing
  `flywheel steer --provider openai --model gpt-6-luna`. The key is read from
  the gitignored `.env`; it is never printed or committed. The analyst model is
  recorded in every result. Recorded Kimi-K3 replies remain available for replay.
- **(b) Embedding model:** not installed. Retrieval stays lexical. Revisit only
  if D1 shows retrieval carries signal.
- **(c) Optimizer variant for B:** included. In the final round, a 3-trial
  search on 100 labeled items with the existing `search_context_policies`:
  lexical k=4; lexical k=4 drawn only from examples Jev got wrong zero-shot
  (a bootstrapped "hard demos" filter; free, zero-shot answers are cached);
  random k=4. About 300 few-shot requests (about $0.19).
- **Request ceiling:** the steps below sum to about 9,300 requests with seed 2
  and about 5,700 without it. The collector ceiling is therefore **9,500**
  (still the $5 cap at the rough rates below), and seed 2 runs only if D2 says so.

## 1. Goal and the one question

Jev-Flywheel improves by adding *questions* for a fitted head. The
Decision-Flywheel study found that per-target lexical retrieval is the best way
to choose *few-shot examples*. **Question: on Jev, with the same labels and
held-out items, which works best after three rounds of 100 labels?**
- adding questions (A);
- adding examples (B in the prompt, or B-local as head features);
- both.

This is exploratory, on one dataset and one engine. A null result counts as an
answer.

## 2. The unified loop (one round)

1. **Feedback:** the next 100 items of a fixed, seeded random label order. The
   scripted labeler gives the corpus reference label. Every arm uses the same
   order.
2. **Lever A (questions):** one steering round (`jev_flywheel.steer`). The
   analyst may add one element. Code fits it, and it is promoted only if
   out-of-fold Brier improves without losing accuracy.
3. **Lever B (examples in the prompt):** the labeled set is the pool.
   `PerLabelLexicalRetrieval` picks 4 examples per label for each item. Jev
   answers the score question with `state = {labeled_examples, target}`. The
   answer becomes one more element, `sentiment.fewshot`, that the head weighs.
4. **Lever B-local (examples as features):** the prompt does not change. For
   each item, retrieve the k=8 nearest labeled neighbours by the same lexical
   overlap, without balancing by label. The head gets two kinds of feature:
   - their similarity-weighted label shares (`knn.share.positive`);
   - the mean similarity of the top 4 neighbours in each label.

   A label-balanced pick would make the shares constant, which is why the
   neighbours are not balanced. This is **not few-shot prompting**: Jev never
   sees the examples, and the head reads them as a nearest-neighbour vote. It
   costs $0 because it reuses cached answers, so it also runs on any engine.
5. **Refit** the head on all labels (`fit_head`, with out-of-fold
   temperature). The fit, the promotion gate and the leakage rules are
   identical in every arm.

**Leakage rules for B and B-local.** The pool contains labeled items only, so
held-out items are never retrieved from. The target itself is excluded by the
existing firewall, so labeled items get leave-one-out answers and features.

**Compatibility.** Elements stay zero-shot, so their cached answers never go
stale. Only `sentiment.fewshot` and the `knn.*` features depend on the pool.
Their cache key adds a context fingerprint: the retrieval policy plus the
sorted labeled IDs.
- A version is the triple (scorecard, context fingerprint, head fit).
  Decision-Flywheel's `HeadProvenance` already records that lineage.
- A replay is deterministic from the seeds, the answer ledger and the recorded
  analyst replies.

## 3. Data and sizes

- **Corpus:** the Jev-Flywheel planted-bias sentiment corpus only.
  - The pool has 5,280 items and the test split 3,521.
  - Jev's zero-shot answers to the 8 existing questions are cached for all
    8,801 items, so replaying those questions costs $0.
  - Jev answers every question for an item in one request, so a new element on
    100 items costs about 100 requests.
- **Labels:** 3 rounds × 100 = 300 labels, drawn from the pool.
- **Iteration slice (dev-100):** 100 test items *outside* paper-600, picked
  with a fixed seed. All iteration results are scored here.
- **Final slice (paper-600):** used only for the final comparison. It is not
  pristine, because it informed earlier Jev-Flywheel work.
- **Seeds:** label order 1. Seed 2 runs only if D2 says so.

## 4. Arms (all on Jev)

| Arm | Head features | Iteration requests per seed (dev-100) |
|---|---|---|
| 0 Baseline | zero-shot holistic answer, refit | 0 (cached) |
| A | + promoted elements | ≈900 zero-shot |
| B-local | + `knn.*` | 0 |
| B | + `sentiment.fewshot` | ≈900 few-shot |
| A+B | + elements + `fewshot` | ≈900 zero-shot + ≈900 few-shot |

The ≈900 figures assume a new answer for every labeled item plus dev-100 in
each round: (100+100) + (200+100) + (300+100).

**What each comparison answers:**
- A vs B is the headline.
- B vs B-local asks whether examples need to be in the prompt at all, or
  whether a free neighbour vote does as well.
- A+B vs the better of A and B asks whether the two levers add up.

**Metrics:** accuracy, Brier and ECE (10 bins). Paired bootstrap intervals
(1,000 resamples over items) for A−0, B−0, B−B-local and A+B−max(A, B). No
significance claims. On dev-100 the intervals are about ±9 points, so iteration
results only steer the work. The comparison is decided on paper-600.

## 5. Cost: one hard cap for the Jev research stage

All dollar figures are **rough**. They use the owner's approximate ≈$0.37 per
1,000 zero-shot requests. A few-shot request carries about 1.7× the tokens, so
≈$0.63 per 1,000 is an estimate. At about 30 req/s with `--max-concurrency`,
each step takes minutes.

| Step | Requests | Est. $ | Cumulative $ |
|---|---|---|---|
| A + B, seed 1, round 1 (smoke) | 400 | 0.20 | 0.20 → **D0** |
| A + B, seed 1, rounds 2–3 | 1,400 | 0.70 | 0.90 → **D1** |
| A+B, seed 1 (+ option (c) 300) | 1,800 (+300) | 0.90 (+0.19) | 1.99 → **D2** |
| Seed 2: A, B, A+B (optional) | 3,600 | 1.80 | 3.79 → **D3** |
| Final paper-600, seed 1, frozen versions | ≈1,800 | 0.82 | 4.61 |

Final paper-600 breakdown:
- A: 600 zero-shot requests.
- B: 600 few-shot requests.
- A+B: 600 zero-shot for its own elements. Its few-shot answers come from B's
  cache, because the pool and the policy are the same.
- 0 and B-local: $0.

**Hard cap: $5, enforced as a cumulative ceiling of 9,500 requests** in the
collector, with tokens logged per request. The ceiling is on requests because
the results files say dollars cannot be derived from usage. If seed 2 is
skipped, the stage spends about $2.80.

## 6. Work breakdown (each step under an hour)

1. **Harness (≤1 h), new code in this repo only:** `scripts/unified_flywheel.py`.
   - **Reuse:** `jev_flywheel` from a clean local clone at `cd4a4886`
     (Workspace, AnswerCache, fit, steer); Decision-Flywheel
     `PerLabelLexicalRetrieval` and `JevAdapter`; the cumulative-ceiling
     collector, bootstrap and `summarize` from this repo.
   - **Build:** the fixed label order; the `knn.*` feature builder; the
     `sentiment.fewshot` injector (`AnswerCache.put_hashed` with a
     context-aware hash); the arm switch.
   - **Check first,** offline with a fake Jev, that injected features pass
     `fit_head` unchanged.
2. **B-local vs 0, seed 1, dev-100 ($0, minutes).** This is a free early check
   that the harness and the head work.
3. **A and B, round 1 (≈$0.20).** **Stop point D0:** check that the requests
   are well-formed, the few-shot answers differ from zero-shot, and the cost
   matches the estimate.
4. **A and B, rounds 2–3 (≈$0.70).** This is the **first useful result: A vs B
   vs B-local on Jev.** **Stop point D1:** humans look at it.
5. **A+B, plus (b) or (c) if approved (≈$1.10).** **Stop point D2:** decide on
   seed 2, then freeze the versions and the protocol in writing.
6. **Seed 2 if chosen (≈$1.80, stop point D3), then the final run on
   paper-600 (≈$0.82).**
7. **One-page result** in `studies/`: the table, the intervals, the analyst's
   questions and the actual spend.

**Later stage (not this afternoon): does it generalize, and can iteration be
free?** Run A and B-local on Laya, which is free, deterministic and has a
512-token window that must be enforced. Then run them on Kev-0.8B, once the
Hard Decisions session hands over its server. B cannot run there, because
neither engine takes few-shot context.

## 7. Risks and unknowns

- **B may absorb the planted bias silently.** Neighbours on the same topic
  carry the label. A states the bias in English; B and B-local do not. Report
  the analyst's proposed questions next to the numbers.
- **The gate may slightly favour B and B-local.** Leave-one-out features are
  computed once, not per cross-validation fold. If the gains are marginal, the
  fix is a per-fold pool.
- **The analyst is noisy.** In Jev-Flywheel, one round found the planted factor
  about a quarter of the time. One or two seeds will not settle A's variance.
- **Results are not comparable to the 0.870 recorded run.** A uniform label
  order differs from Jev-Flywheel's active selection.
- **The clean clone may behave differently.** Jev-Flywheel has 31 uncommitted
  changes, including `steer.py`, so a clone at `cd4a4886` may differ from the
  code that produced earlier runs.
- **Few-shot request size is unmeasured.** The real token multiplier is only
  known after D0.

## Appendix A: verified facts

Read-only checks by this agent:
- **Laya adapter** (`Decision-Flywheel/src/decision_flywheel/adapters/laya.py`)
  raises an error on any labeled context.
- **Kev adapter** (`adapters/kev.py`) sends `state={labeled_examples, target}`.
  The Kev server (`Hard-Decisions/var/kev-patched/kev/api.py`) only flattens
  the state to text with `render()`.
- **Corpus and splits:** `Jev-Flywheel/fixtures/items.jsonl`.
- **Cached answers:** `answers.jsonl.gz` and `answers-laya.jsonl.gz`, 8,801
  items × 8 questions each. `topic_domain` answers for 740 items are in
  `recordings/simulated-labeler/extra_answers.jsonl.gz`.
- **paper-600** is defined in `scripts/finetune_laya.py`.
- **Reusable code:** `context.py::PerLabelLexicalRetrieval`,
  `optimizer.py::search_context_policies`, `head.py` and `steering.py`.
- **Analyst default** (`jev_flywheel/steer.py`): Bedrock
  `us.moonshotai.kimi-k3`.
- **Embeddings:** no embedding package or model is installed.

Verified by the main session (authoritative):
- **Kev-0.8B weights** are cached at
  `Biased-Decisions/.worktrees/kev/var/hf-cache`.
- **Kev latency**, one request at a time: 0.8B ≈105–125 ms; 4B ≈710 ms; 9B
  ≈1.5–1.75 s. Answers are deterministic.
- **Kev server:** a 0.8B server will be left on :8009. Use model `kev-latest`
  and do not stop or restart that server.
- **Kev context length:** trained on states of up to 384 tokens. The server
  accepts up to 8,192, but longer states are untrained and worse (README:
  Kev-9B 0.92 within 384, 0.75–0.79 beyond).
- **Laya:** a 421M encoder with a 512-token window. It truncates silently. On
  the recorded 140 labels it went 0.722 → 0.802 (Jev: 0.768 → 0.870).

## Appendix B: unverified or open

- **Owner's figures:** the ≈$0.37 per 1,000 price and the ≈30 req/s rate.
- **Estimates:** the 1.7× few-shot token multiplier, and the Bedrock analyst
  price.
- **Harness check:** whether the injected `knn.*` and `fewshot` features need
  any change to `jev_flywheel`. Step 1 checks this.

## Appendix C: prior art (from memory, to verify)

- **KATE** (Liu et al., 2021/22): nearest neighbours in sentence-embedding
  space beat random demonstrations.
- **EPR** (Rubin et al., NAACL 2022): a learned retriever beats BM25 and SBERT.
- **BM25** is a strong lexical baseline.
- **DSPy:** LabeledFewShot, BootstrapFewShot (+RandomSearch), KNNFewShot and
  MIPROv2. Option (c) is a small native analogue of
  BootstrapFewShotWithRandomSearch. MIPRO-style instruction search is out of
  scope.
- **B-local** resembles classic kNN features fed to a learned head.
- **Min et al. (2022):** how correct the demonstration labels are matters less
  than expected. That is testable here, because the labels carry the planted
  signal.
- **Zhao et al. (2021):** contextual calibration.
