# Decision Flywheel: product design and first experiment plan

Status: design only, 2026-10-01. Nothing here has been built or run. It builds on
[UNIFIED_FLYWHEEL_PLAN.md](UNIFIED_FLYWHEEL_PLAN.md) and
[UNIFIED_FLYWHEEL_RESULTS.md](UNIFIED_FLYWHEEL_RESULTS.md). Facts checked in code
are in Appendix A, AWS facts in Appendix B, and unverified items in Appendix C.

**Product in one line:** the loop learns from labeled human feedback, comments
included, and does two things:
- **Feature discovery:** an analyst turns explanations and errors into new rubric
  questions, which Jev answers and which become head features.
- **Few-shot optimization:** a native optimizer improves one **fixed** example
  list across rounds.

**Default end state:** a frozen, reloadable, versioned classifier made of the
fixed example list, the rubric questions, and a trained head over Jev's answers.
**Dynamic per-item retrieval** is a separate option, off by default, and
configured on its own.

---

## 1. Architecture

### 1.1 The loop (one round, 100–200 new labels)

1. **Collect feedback.** For each item: label, the model's shown prediction, and
   an optional comment (`FeedbackItem.edit_comment_value`).
2. **Discover features (lever A).** One steering round. The analyst reads the
   disagreements and comments, commented ones first. It may propose up to one
   batch of element edits. Jev answers the new questions zero-shot on the labeled
   items. A gate keeps the change only if out-of-fold Brier improves without
   losing accuracy.
3. **Optimize the example list (lever F).** The fixed list challenges itself
   (§3), scored on a development split drawn from the labels and never from
   held-out data. The winner's few-shot answer becomes one more head feature.
4. **Optional dynamic retrieval (lever D).** Only if `retrieval.enabled`. Each
   item gets its own retrieved examples, and that few-shot answer is a further
   feature.
5. **Refit and gate.** Refit the head on all labels with an out-of-fold
   temperature, behind the same promotion gate.
6. **Freeze a bundle version (§1.2).** It becomes the next round's incumbent and
   what is served.

### 1.2 The saved artifact: a classifier bundle

A directory with three files.
- **`bundle.json`** is canonical, text-free JSON with a SHA-256 integrity hash,
  in the style of `artifacts.py`.
- **`examples.jsonl`** holds example texts. It is private, never committed, and
  each row is verified against the manifest's hashes on load.
- **`scorecard.yaml`** holds the rubric and head, in Jev-Flywheel's `Score`
  format, hashed in the manifest.

| `bundle.json` block | Contents |
|---|---|
| `task` | name, labels, instructions, `input_field`, `DecisionTask.fingerprint` |
| `engine` | adapter (`jev`), configured model, **reported** model (e.g. `jev-1.13.0`), and the adapter's `model_identity` |
| `rubric` | scorecard SHA-256, plus each element's key, type, instructions, criteria and definition hash. Provenance: round, analyst provider and model, reply SHA-256, and the hashes of the comments it cited |
| `fewshot` | policy `fixed-example-list` v1 and its fingerprint; `per_label`; ordered primary IDs and one **reserve** per label, each with label and input hash; `display_order`; `presentation_label_order`; examples-file hash; and optimizer audit (search fingerprints, development objective, trials) |
| `head` | features, classes, weights, calibration temperature, fit ID, training-ID fingerprint, out-of-fold metrics. Format: Jev-Flywheel `DecisionSpec` |
| `retrieval` | `null` by default. Otherwise: the config (§4), pool revision and fingerprint, the pool texts file, the index fingerprint, and the embedding model and revision |
| `lineage` | parent bundle hash, round, labels fingerprint, label-order seed, code commits (core, Jev-Flywheel clone, harness) |

**Reload and classify a new item.**
1. `load_bundle(dir)` verifies every hash and the task fingerprint. It compares
   the configured engine model with the stored one, and warns if the model Jev
   reports differs.
2. Send **request 1**, zero-shot: the holistic question plus every rubric
   element, in one Jev request.
3. Send **request 2**, few-shot: the holistic question, with
   `state={labeled_examples: fixed list, target}`.
4. If retrieval is on, send **request 3**: the same as request 2, but with that
   item's retrieved examples.
5. `Score.feature_vector` turns the answers into features. `predict` returns a
   label, a calibrated probability and the top contributions.

So the default costs **2 Jev requests per item**, and 3 with retrieval.

**Reserve rule.** If a target is itself in the list, its example is replaced by
that label's reserve. This keeps the list balanced, and it gives labeled items
honest leave-one-out answers during training.

### 1.3 Who owns what: compose, do not duplicate

| Piece | Owner | Reuse | New |
|---|---|---|---|
| Few-shot context, budget, firewall | **core Decision-Flywheel** | `ContextPolicy`, `validate_selected_context`, `build_context_plan`, `ContextBudget` | `FixedExampleList` policy (with reserve); `artifacts._policy_from_metadata` support for it |
| List optimizer | core | `search_context_policies`, `TrialSpec`, checkpoint and cumulative `call_accounting` | `improve_example_list()` driver (§3); a Brier objective |
| Retrieval | core | `PerLabelLexicalRetrieval` (frozen as v1) | `Retriever` / `VectorStore` protocols, lexical v2, in-memory vector index, embedding providers as optional extras |
| Bundle | core | `artifacts.py` canonical JSON, hash and strict-keys pattern | `ClassifierBundle` save, load and verify |
| Rubric elements, feature extraction, head fit, gate, analyst | **Jev-Flywheel**, consumed read-only from the pinned clone at `cd4a4886` | `Score`, `ElementSpec`, `fit_head`, `compare`, `with_fit`, `predict`, `AnswerCache`, `Workspace`, `run_steering` with `steer_scorecard.tac` and its host briefing | nothing |
| Rounds, arms, labeler, ledger, firewalls, stats | **Evaluations harness** | `unified_loop.UnifiedFlywheel`, `unified_spend`, `unified_splits`, `unified_stats` | arms F, F-rand, A-c, A-c+F; the explanation labeler |

**Duplication to avoid, and to decide on (Q4).** Core already has a parallel
lineage-only `feedback.Scorecard`, a pure-Python `LearnedHead`, and an offline
`run_steering_round`. The harness uses Jev-Flywheel's richer versions instead.

This design adds **no third copy**. It treats Jev-Flywheel's `Score` YAML as the
bundle's rubric-and-head payload, and core contributes only context, the
optimizer, retrieval and the manifest. Moving `Score` and `fit_head` into core
is the long-term clean-up, but not for this phase.

No DSPy dependency anywhere.

---

## 2. Feedback explanations

**What already exists in Jev-Flywheel** (read-only check, Appendix A):
- **Capture.** `FeedbackItem.edit_comment_value` holds the comment, and the
  console asks "Why?" after a disagreement.
- **Briefing.** `host.briefing()` sends the analyst up to 25 mismatches,
  commented ones first. Each mismatch carries `human_comment`, the element
  answers and the top drivers. It also sends up to 5 commented agreements and a
  balanced labeled sample.
- **Prompt.** `steer_scorecard.tac` tells the analyst that a comment naming an
  uncaptured concept is a missing element.
- **Trigger.** `SteeringPolicy` fires a "rethink" after 5 commented mismatches.

So the path from a comment to a proposed question **already works**.

**What is missing:**
1. **The harness never sends a comment.** `_record_feedback` leaves
   `edit_comment_value` empty, so the 0.855 result for A ran without
   explanations.
2. **No real explanations exist on disk.** Every recording's comments are the
   deliberately uninformative template "I disagree; the correct label is X":
   - 43 in `fixtures/recordings/simulated-labeler`;
   - about 2,270 across the `bios_attorney` recordings.
3. **No attribution.** Nothing records whether a promoted element came from a
   comment.
4. **Core's `AnalystBriefing` carries only IDs and hashes.** That is fine,
   because steering stays in Jev-Flywheel.

**Cheap, credible labels with explanations.**
- **Labels stay the corpus reference labels**, so every arm sees identical
  labels. The LLM writes only the *reason*.
- **Model:** OpenAI `gpt-6-luna`.
- **Calls:** one per labeled item that some arm got wrong, conditioned on the
  text and the correct label and not on any arm's prediction, so the reason is
  shared across arms. At most 300, and probably about 100.
- **Caching:** by item, label, prompt SHA-256 and model, in `var/`. Replay is
  exact. Summaries carry only hashes, counts and a keyword flag ("mentions topic
  or domain"), to stay text-free.

| Level | What the labeler is told | What it measures |
|---|---|---|
| L0 | nothing (the existing template) | the current baseline: no explanation |
| **L2, main** | a hidden team convention written in plain words ("we count sports or recreation talk as positive and workplace operations as negative when the wording itself is neutral or mixed; otherwise follow the sentiment"). It must write like a busy reviewer, at most 20 words, never quote the rule, and give no comment 40% of the time | whether the pipeline turns a *knowing* reviewer's terse reason into a question |
| L3, ceiling (optional, offline-only analysis) | the full rule, verbatim | leakage upper bound; not evidence |

**Caveats, to state with any result:**
- **L2 leaks by design.** Real reviewers know their conventions, but L2 comments
  will be more consistent, fluent and correct than real ones. A gain from L2
  shows that the plumbing works. It does **not** show that humans' comments
  surface hidden factors.
- **Same-model coupling.** If `gpt-6-luna` is both labeler and analyst, they
  share phrasing. A mitigation is to make the labeler Bedrock Kimi-K3 (already
  wired in Jev-Flywheel); this needs the owner's call (Q1).
- **A human checks first.** At stop S3, a person reads 20 comments before any
  analyst sees them.

---

## 3. Few-shot list optimizer (default)

### 3.1 Why a new driver is needed

`search_context_policies` evaluates a declared set of trials on development
labels and returns the best. As used so far, the trials were a few random
draws.

In the Emotion study, the development-chosen global random context scored
macro-F1 **0.5616**:
- the mean of the random contexts was **0.5690**;
- per-item retrieval scored **0.6964**.

So picking the best of a few random draws did not beat random.

It also has these limits:
- the objective is only accuracy or macro-F1, because checkpoints store a label
  and no probabilities;
- the incumbent gets no tie or margin preference;
- trials are fixed up front.

### 3.2 `improve_example_list` (native, small, deterministic)

**Settings:**
- **List size:** k = 4 per label, plus 1 reserve per label (8 + 2 for binary
  sentiment).
- **Development split per round:** the round's newest labels, minus every item in
  any trial's list, capped at 100. These are fresh for the incumbent, which was
  chosen on earlier batches.
- **Candidates:** all other labels.
- **Never scoreboard:** dev-100 and paper-600 go in `protected_ids`.

**Challengers, at most 3 trials per round:**
1. **Incumbent** (from round 2 on). In round 1 the incumbent is zero-shot, which
   is cached and free.
2. **Hard swap.** Take the incumbent and replace ⌈k/2⌉ slots per label with
   *hard demos*: labeled candidates the **current classifier** got wrong out of
   fold, the most confidently wrong first. In round 1, use Jev's cached zero-shot
   errors. If a label has too few hard demos, fill from `PrototypeBalanced`.
   - Which slots to drop: the most recently added first. Per-slot ablation would
     cost k × |dev| requests, which is too expensive.
3. **Exploration or control.** A fresh `RandomBalanced(seed=round)` draw. In
   round 1, also try `PrototypeBalanced`.

**Scoring and promotion:**
- All trials go to `search_context_policies` as `FixedExampleList` policies.
- Score each by **Brier on Jev's few-shot probabilities** over the development
  split. This needs a small core change: an optional `probabilities` field in
  checkpoint entries, and `objective="brier"`. Accuracy is the fallback.
- **Promote** only if Brier improves by at least 0.005 and accuracy does not
  fall. Ties go to the incumbent.
- Then fill the winner's answers for all labels and the evaluation slice, and
  refit the head.

**Cost per round on Jev, few-shot:**
- **Search:** about 3 × 90 = 270 requests, minus checkpoint hits.
- **Fill:** about 100r + 100 for the labels so far plus dev-100, minus about 90
  development items already answered. If the incumbent survives, only the new
  items need answers.
- **Total:** 500–700 requests per round, about $0.35–0.45 (Appendix C rates).

**Freezing:** the round's winner is written into the bundle's `fewshot` block,
with search fingerprints and trial results. A frozen bundle never re-optimizes.

**Known biases:**
- Development items enter head training afterwards, which mildly favours F at
  the gate. This is recorded, not fixed.
- "Most confidently wrong" items are mostly label noise on real data, but on
  this corpus they are mostly the planted bias, so the corpus **flatters** hard
  mining.

---

## 4. Retrieval options (optional feature, off by default)

### 4.1 Interface (core)

```python
class Retriever(Protocol):           # builds an index over trusted labeled items only
    fingerprint: str                 # config + version (+ embedding model/revision)
    def index(self, task, pool: Sequence[LabeledItem]) -> "RetrievalIndex": ...

class RetrievalIndex(Protocol):
    fingerprint: str                 # retriever fingerprint + pool fingerprint
    def top_per_label(self, target: Item, label: str, k: int,
                      exclude_ids: set[str]) -> list[tuple[str, float]]: ...

@dataclass(frozen=True)
class PerLabelRetrieval:             # a ContextPolicy, name "per-label-retrieval", version "2"
    index: RetrievalIndex            # select() asks each label for k+1, drops target id/text,
                                     # ties by id; the shared firewall still validates the result
```

`VectorStore` (`upsert(rows)` and `query(vector, top_k, label_filter)`) sits
under the embedding index. In-memory is the default; S3 Vectors and DynamoDB are
optional stubs. Exact cosine search runs client-side in memory; approximate
search runs on AWS, and the fingerprint records `approximate: true`.

### 4.2 Config schema, independent of the optimizer

```yaml
fewshot:                        # default product path
  mode: fixed                   # fixed | none
  per_label: 4
  optimizer: {enabled: true, objective: brier, min_brier_gain: 0.005,
              max_trials_per_round: 3, dev_max: 100, max_requests_per_round: 800}
retrieval:                      # separate switch; nothing above reads it
  enabled: false
  per_label: 4
  combine: add-feature          # the only mode for now (adds request 3)
  backend: lexical              # lexical | embedding
  lexical:
    stopwords: en-v1            # ON by default; vendored list WITHOUT negations (not/no/nor/never)
    tokenizer: unicode-v1       # casefold + NFKC + \w+ ; "ascii-v1" = current [a-z]+
    weighting: binary-cosine    # binary-cosine (current) | bm25 {k1: 1.2, b: 0.75}
  embedding:
    provider: openai            # openai | local-onnx | fake (specs)
    model: text-embedding-3-small
    dimensions: 512
    cache: var/embeddings.jsonl # keyed by (provider, model, dims, normalized-text sha256)
  store: {kind: memory}         # memory | s3vectors | dynamodb (optional extras)
```

### 4.3 Lexical v2

**Keep v1.** `PerLabelLexicalRetrieval` is unchanged (name, version "1", empty
config). Old artifacts and the cached answers behind B depend on its
fingerprint.

**What v2 changes:**
- **Stop words on by default.** The list is vendored and versioned, so it needs
  no NLTK download and the fingerprint stays stable. The list keeps negations;
  stock lists such as scikit-learn's drop "not" and "no", which hurts sentiment.
- **Unicode tokens.** The v1 `[a-z]+` silently discards digits and non-English
  text.
- **Optional BM25.** IDF is computed over the labeled pool only, in about 40
  lines of pure Python; core stays dependency-free.

### 4.4 Lightweight semantic option

**Index:** an in-memory matrix of normalized vectors with exact cosine search.
- Per query: 5,280 × 512 in numpy takes milliseconds.
- numpy goes in an optional extra, so core stays dependency-free.

**Embedding source, with a recommendation:**

| Option | Install and runtime | Determinism | Cost | Notes |
|---|---|---|---|---|
| **OpenAI `text-embedding-3-small`** (recommended **for the experiment**) | `openai` is already installed in the Decision-Flywheel and Jev-Flywheel venvs; the key exists | vectors cached once to disk; replays are then exact | $0.02 per 1M tokens (verified). The whole 8,801-item corpus is ≈0.3M tokens ≈ **$0.01** | sends text to a second vendor, which is fine for this public corpus but a policy question for customer data |
| Local ONNX small encoder (e.g. a bge-small-class model through `fastembed`) | one pip install, plus onnxruntime and a model download, size unverified | deterministic on CPU | $0 | recommended default **for customer data** if the owner prefers no egress (Q5) |
| Laya's encoder, pooled | already installed elsewhere | deterministic | $0 | not a trained sentence embedder; not recommended |

**Repo rule:** specs use a `fake` hashing embedder and make no network calls.
Live embedding runs only in an explicit, opt-in `embed` step that writes the
cache.

### 4.5 Heavy stores (design-only stubs; facts in Appendix B)

**S3 Vectors adapter:**
- one vector bucket and one index per pool revision;
- `dataType float32`, `distanceMetric cosine`;
- filterable metadata: `label`, `pool_revision`, `input_hash`;
- `PutVectors` in batches of 500;
- `QueryVectors(topK=k+1, filter={"label": L})`, then target exclusion client-side;
- IAM `s3vectors:*` actions; `GetVectors` is also needed for filters.

**DynamoDB adapter:**
- an on-demand table with a vector index (`VectorAttribute`, `Dimensions`,
  `DistanceFunction=COSINE`) and `label` as an `INLINE_FILTER`;
- `SearchVectors(TopK≤100, SearchConditionExpression label = :L)`;
- a separate search endpoint and `dynamodb:SearchVectors`.

**Risks for both:**
- approximate search;
- eventual consistency right after writes, which can break exact replay;
- per-call costs that are trivial at our scale;
- account setup.

**Neither is built in this phase.** At 5k vectors, in-memory search is strictly
better: exact, free and offline.

---

## 5. Experiment plan (existing harness corpus, Jev, one seed)

**Fixed setup:**
- the same corpus, label order (seed 1), rounds (3 × 100) and dev-100;
- paper-600 only for the final, frozen comparison;
- the gate, folds, bootstrap and leakage rules exactly as in the unified study;
- a new spend ledger with a hard ceiling of **7,500 Jev requests**.

### 5.1 What the earlier results do and do not tell us

**What they tell us** (paper-600 after 300 labels: A 0.855, B-dynamic 0.892,
A+B 0.918, baseline 0.767):
- Per-item lexical retrieval helps a lot **on this corpus**.
- Features and dynamic examples add up: A+B beat the better of A and B by
  +0.027, interval 0.007 to 0.048.
- Few-shot answers differ from zero-shot on about 20% of items.
- A few-shot request carries about **1.9×** the input tokens: 582 against 312
  per request, from the run's own token logs.
- The machinery, ledger and gate hold up at about 6,000 requests.

**What they do not tell us:**
- **Anything about a fixed list (F).** It was never run. The prior evidence for
  fixed lists is weak:
  - Emotion: as in §3.1.
  - AG News: a random 4-shot list gained only +0.008 over zero-shot, interval
    −0.002 to 0.018.
  - This corpus is worse for fixed lists than most. Retrieval wins here because
    neighbours on the same topic carry the label. A fixed list of 8 cannot show
    a per-item topic neighbour; it can only show the rule, through well-chosen
    neutral sports and workplace examples. **Expect F < D here**, and treat any
    F gain as an upper bound for the hard-demo miner.
- **Anything about explanations.** No comments were given.
- **Anything about stop words or embeddings.**
- **How reliable A is.** One analyst trajectory: A found `topic_domain` in
  round 2, while A+B's analyst proposed different elements.
- **Seed and corpus variance.** There was one seed, one engine, and a
  planted-bias corpus. Paper-600 is not pristine. B's leave-one-out answers were
  computed once, which may slightly favour B.

### 5.2 Arms

| Arm | Head features | New Jev requests (dev-100 iteration) |
|---|---|---|
| 0 baseline | zero-shot holistic | $0, cached |
| A (no comments) | + promoted elements | $0: replay of the unified run (recorded replies and answers) |
| D-v1 / A+D-v1 | + dynamic lexical v1 few-shot | $0: these are the earlier B / A+B, replayed |
| **F** | + fixed optimized list's few-shot answer | about 1,450 few-shot (§3.2, three rounds) |
| **F-rand**, the control | + fixed random-balanced list, redrawn each round, same k | about 900 few-shot |
| **A-c** | A, with L2 comments | about 800 zero-shot, plus about 6 analyst calls and ≤300 labeler calls (OpenAI) |
| **A-c+F**, the default product | A-c elements + F feature | about 800 zero-shot; F's few-shot answers are shared, because the list does not depend on elements |
| D-sw / D-emb | dynamic retrieval with stop words on, or with embeddings | **$0 proxy first** (below); live only if the proxy passes |
| A-c+F+D | head-only variant: A-c+F's elements + F + the best D | $0 once its answers are cached. **Not** an independent steering trajectory; say so |

**Retrieval comparison inside D, staged:**

1. **A free proxy.** For each retriever (v1, v2 with stop words, v2 + BM25,
   embeddings), compute the label-weighted kNN vote accuracy and the
   *same-label purity* of the top 4 neighbours of dev-100 items, from the
   300-label pool.
   - Cost: $0 apart from about $0.01 of embeddings.
   - This is the B-local machinery in `unified_knn`, with a pluggable retriever.
2. **Live spend only if justified.** Run a variant live (300 labeled
   leave-one-out answers + 600 paper-600 ≈ 900 few-shot requests) only if its
   proxy beats v1 by at least 3 points. Run at most one new variant live.

### 5.3 Spend (rough; rates in Appendix C)

| Stage | Jev few-shot | Jev zero-shot | Est. $ |
|---|---|---|---|
| Round 1, F, F-rand, A-c, A-c+F (stop S1) | ≈600 | ≈400 | ≈0.55 |
| Rounds 2–3, same arms (stop S4) | ≈1,750 | ≈1,200 | ≈1.60 |
| Final paper-600, frozen F, F-rand, A-c, A-c+F | ≈1,200 | ≈1,200 | ≈1.20 |
| Optional: one D variant live | ≈900 | 0 | ≈0.60 |
| **Total** | ≈4,450 | ≈2,800 | **≈$3.95** (Jev) + OpenAI (small, price unverified) |

Throughput is about 30 req/s if the owner's figure holds, so Jev time is
minutes. The afternoon goes on analyst calls and human checks.

### 5.4 Comparisons and the results that would change the design

All comparisons use paired bootstrap intervals on paper-600, with no
significance claims.

| Comparison | Result that changes the design |
|---|---|
| **F vs F-rand** (does the optimizer earn its keep?) | Within ±1 point, with an interval spanning 0: ship the plain `PrototypeBalanced` or random list as the default, shrink `improve_example_list` to a one-shot pick, and stop investing in it |
| **A-c+F vs A-c and vs F** (do the levers add up for the fixed default?) | No gain over the better single lever: the default collapses to one lever, and two Jev requests per item are not justified |
| **A-c+F vs A+D-v1 (0.918)** (is the fixed default good enough?) | A+D ahead by ≥3 points: report it plainly. The owner decides whether D stays off by default (Q3) |
| **A-c vs A** (do explanations help?) | No gain even with leaky L2 comments: the comment path is not the bottleneck, so investigate the analyst prompt before any labeling UI work |
| **D proxy, v2 with stop words vs v1** | v2 worse: flip the stop-word default off |
| **D proxy, embeddings vs best lexical** | Not ≥3 points better: keep embeddings as an interface stub only |

---

## 6. Build order (each step under an hour; stop points in bold)

1. **Core `FixedExampleList` policy.** Includes the reserve, the fingerprint,
   artifact-loader support, and specs (firewall, reserve swap, balance).
   Offline.
2. **Core Brier objective for the optimizer.** Optional probabilities in
   checkpoint entries, `objective="brier"`, and specs; nothing existing changes.
   Then **`improve_example_list`**, with a fake-model spec showing that the
   incumbent keeps ties and that protected IDs are refused.
3. **Harness arms F, F-rand and A-c+F.** Reuse `_ensure_fewshot`, with a
   fixed-list context fingerprint. Run offline end to end with `FakeJevCore`.
   **S0: the offline run passes; $0.**
4. **Replay check.** The new harness version reproduces 0, A, D-v1 and A+D-v1
   from the cached answers and replies, at $0. If it cannot, stop and fix the
   harness before spending.
5. **Live round 1** (≈$0.55). **S1:**
   - requests well-formed;
   - F's list differs from F-rand's;
   - few-shot answers differ from zero-shot at about the 20% seen before;
   - the token multiplier is about 1.9×.

   *Earliest useful result: the first F vs F-rand reading.*
6. **Lexical v2 and the retrieval proxy.** Then the opt-in `embed` script,
   cached (≈$0.01). **S2: choose at most one D variant for live spend.**
7. **L2 explanation labeler.** Prompt, cache, text-free summary flags, and the
   comments wired into `_record_feedback`. Generate comments for the 300 labels.
   **S3: a human reads 20 comments for leakage and realism.**
8. **Live rounds 2–3** for F, F-rand, A-c and A-c+F (≈$1.60). **S4: humans look;
   freeze the versions and the protocol in writing.**
9. **Core `ClassifierBundle` save, load and classify.** A spec shows that a
   reloaded bundle gives identical predictions to the in-memory one on the fake
   engine.
10. **Final run on paper-600, scored through the *reloaded* bundles** (≈$1.20,
    plus the optional D at ≈$0.60). Then a one-page result in `studies/`.
    **S5.**

**Not in this phase:**
- the S3 Vectors and DynamoDB adapters (design only, §4.5);
- a second seed (only if S4 hinges on it);
- Laya and Kev, which help only zero-shot or feature discovery;
- the one-request variant of the bundle (Q2).

## 7. Open questions for the owner

1. **Explanations as evidence.** Is an LLM labeler with a hidden convention (L2)
   acceptable evidence for the comment path? Should it be a different model from
   the analyst (Kimi-K3 vs `gpt-6-luna`)? And is there **any** real reviewer
   feedback with comments we could use? None exists on disk; every recorded
   comment is a template.
2. **Request shape.** Should the fixed list go in its own few-shot request (2
   Jev requests per item, clean caches; this design), or should the examples ride
   in the same request as every rubric question?
   - **One request** costs half as much per item.
   - **But** the examples' labels would influence element answers, and every
     list change would make every cached element answer stale.
3. **If A+D clearly beats A-c+F,** does the default stay fixed, or does
   retrieval become the default for small pools?
4. **Ownership.** Should Jev-Flywheel's `Score` and `fit_head` be ported into
   core eventually, replacing core's parallel `Scorecard`, `LearnedHead` and
   `run_steering_round`? Or should core bundles keep wrapping Jev-Flywheel YAML?
5. **Embedding default for customer data.** API (egress to OpenAI) or a local
   ONNX model (install burden)?
6. **Model pinning.** Can Jev pin a model version (`jev-1.13.0`) rather than
   `jev-latest`? A bundle that silently changes its engine is not reproducible.
7. **Labels per round.** Confirm 100 per round with k = 4 per label and no
   sweep.

---

## Appendix A: facts checked in code (read-only, 2026-10-01)

**Core Decision-Flywheel**

`context.py`:
- `PerLabelLexicalRetrieval` is binary token-set cosine over `[a-z]+`, with no
  stop words, ties broken by ID, `POLICY_VERSION = "1"` and an empty config.
- `RandomBalanced` and `PrototypeBalanced` are fixed-global policies.
- `_validated_candidate_pools` removes the target by ID or normalized text, and
  then *requires* at least `per_label` items per label. That is why a fixed list
  needs a reserve.

`optimizer.py`:
- `search_context_policies` takes declared trials, requires candidates and
  development to be disjoint, and accepts `protected_ids` and hashes.
- Checkpoint entries must be exactly `{"label"}`.
- The objective is accuracy or macro-F1.
- The winner is the maximum objective, with ties broken by trial name; there is
  no incumbent preference.

`artifacts.py`:
- The artifact is text-free: the pool is stored as id, input hash and label,
  and the texts are supplied on load.
- `_policy_from_metadata` allowlists three policies; the lexical one must have an
  empty config.

Other core files:
- `head.py`: `LearnedHead` (pure-Python softmax with out-of-fold temperature)
  and `HeadProvenance`.
- `feedback.py`: `FeedbackItem.edit_comment_value` exists.
- `steering.py`: `AnalystBriefing` carries only IDs and hashes.
- `adapters/jev.py`: few-shot sends
  `state={labeled_examples:[{text,label}], target}` and asks only the task
  question; retries are disabled.
- Venvs: no torch, sentence-transformers or fastembed is installed in the
  Decision-Flywheel, Evaluations or Jev-Flywheel `.venv312` environments.
  `openai` and `boto3` are installed in two of them, and scikit-learn in Jev's.

**Jev-Flywheel** (31 uncommitted changes; the harness uses the clone at `cd4a4886`):
- `items.FeedbackItem.edit_comment_value`; `console.py` asks "Why? (optional)".
- `host.briefing()` sends:
  - mismatches, at most 25, commented first, each with `human_comment`;
  - commented agreements, at most 5;
  - a balanced labeled sample, at most 40;
  - the element inventory;
  - a separate blind "Group A / Group B" scout view.
- `steering.py`: "rethink" fires after 5 commented mismatches.
- `simulate.template_comment` is deliberately uninformative.
- Recordings hold template comments only.

**Evaluations harness:**
- `unified_loop.UnifiedFlywheel._record_feedback` sets no comment.
- `_ensure_fewshot` uses `PerLabelLexicalRetrieval` with 4 per label, under a
  context fingerprint.
- Per-request tokens (paper-600 summary): few-shot ≈582 input, zero-shot ≈312.

## Appendix B: AWS facts (checked 2026-10-01 against official pages; dates as shown)

**Amazon S3 Vectors**

Status and regions:
- **GA on 2025-12-02, in 14 Regions.** VERIFIED:
  aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/
- **31 Regions after the expansion of 2026-03-31.** VERIFIED:
  …/whats-new/2026/03/s3-vectors-expands-17-regions/
- **GovCloud and the European Sovereign Cloud.** UNVERIFIED: seen in search
  results, but the pages were not opened.

Model and API (VERIFIED):
- Vector bucket → vector index. ARN form:
  `arn:aws:s3vectors:{region}:{acct}:bucket/{b}/index/{i}`. API version
  `s3vectors-2025-07-15`.
- Operations seen: CreateIndex, PutVectors, GetVectors, QueryVectors,
  DeleteVectors, ListVectors, ListIndexes, ListVectorBuckets.
  - `CreateVectorBucket` is **UNVERIFIED** (the page was not opened).
- `CreateIndex`:
  - `dataType: float32` only;
  - `dimension` 1–4096;
  - `distanceMetric: cosine | euclidean`, with no dot product;
  - `metadataConfiguration.nonFilterableMetadataKeys`.
  - docs.aws.amazon.com/AmazonS3/latest/API/API_S3VectorBuckets_CreateIndex.html
- `QueryVectors`:
  - parameters: `queryVector`, `topK`, `filter` (JSON), `returnDistance`,
    `returnMetadata`, `nextToken`;
  - `queryMode` is CLASSIC or ENHANCED; ENHANCED filters before the search;
  - the response is `vectors[{key, distance, metadata}]`.
  - …/API_S3VectorBuckets_QueryVectors.html
- **Auth:**
  - IAM `s3vectors:*` actions;
  - a query with a filter or with `returnMetadata` also needs
    `s3vectors:GetVectors`.
- **SDK:** `boto3.client('s3vectors')`. The minimum boto3 version was **not
  found**.

Limits (VERIFIED; docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html):
- **Scale:** 10,000 buckets per Region per account, 10,000 indexes per bucket,
  2B vectors per index.
- **Metadata:**
  - 40 KB in total per vector, of which at most 2 KB is filterable;
  - 50 keys per vector;
  - 10 non-filterable keys per index.
- **Batch sizes:** Put and Delete take ≤500 vectors per call, Get ≤100.
- **Write rate:** ≤2,500 vectors per second per index.
- **Top-K:** ≤10,000, paginated at 100.
  - The GA announcement said 100. The higher figure looks like a later
    increase; design for 100 per page.
- **Latency:** under 1 s for infrequent queries, about 100 ms or less for
  frequent ones (GA announcement).

Pricing, us-east-1 (aws.amazon.com/s3/pricing/; read through a summarizing
fetch, so **re-check before use**):
- storage: $0.06/GB-month;
- PUT: $0.20/GB, with a 128 KB minimum per PUT;
- queries: $2.50 per million;
- data processed: $0.004/TB for the first 100K vectors per index, falling at
  higher tiers;
- data returned: $0.01/GB, with the first 512 KB per query free.

Integrations: Bedrock Knowledge Bases, SageMaker Unified Studio, OpenSearch
Service. VERIFIED.

**DynamoDB vector search** (the owner is right: it is native)

Status (VERIFIED):
- Announced as GA on **2026-08-05**.
  - aws.amazon.com/about-aws/whats-new/2026/08/amazon-dynamodb-vector-search/
  - aws.amazon.com/blogs/aws/amazon-dynamodb-now-supports-real-time-vector-search-at-any-scale/
- It is a new **vector index** type, separate from the older zero-ETL path to
  OpenSearch.

API (VERIFIED; docs.aws.amazon.com/amazondynamodb/latest/developerguide/VectorSearchWorkingWith.html):
- **Creating an index:** `CreateTable` with `VectorIndexes`, or `UpdateTable`
  with `VectorIndexUpdates`.
  - Fields: `VectorAttribute`, `Dimensions`,
    `DistanceFunction: COSINE | EUCLIDEAN | DOT_PRODUCT`, `Projection`, and an
    optional `SearchSchema` (a HASH key plus `INLINE_FILTER` attributes).
  - Vectors are stored as an `L` list of `N` values, indexed at f32.
  - One index is created at a time, and each needs a backfill.
- **Querying:** `SearchVectors`.
  - Parameters: `TableName`, `IndexName`, `SearchVector`, `TopK`,
    `SearchConditionExpression` (equality only), `ExpressionAttributeValues`,
    `ProjectionExpression`.
  - It returns `SearchResults[{Item, Score}]` and is **eventually consistent**.
  - It uses separate endpoints (`{acct}.search-ddb.{region}.amazonaws.com`).
  - IAM permission: `dynamodb:SearchVectors`.
- **Restrictions:**
  - no fine-grained access control;
  - on-demand tables only;
  - responses up to 16 MB, with no pagination.

Quotas (VERIFIED, ServiceQuotas page):
- 5 vector indexes per table (adjustable);
- 4,096 dimensions;
- **TopK ≤ 100**;
- 18 inline filters;
- 600 GB maximum base table size without allowlisting.

Pricing, us-east-1 (…/dynamodb/pricing/on-demand/; summarizing fetch,
**re-check**):
- vector writes: $0.52/GB;
- search: $0.002/GB processed;
- index storage: $0.25/GB-month;
- included in the Free Tier.

Regions: all commercial Regions plus GovCloud (VERIFIED). Latency claim:
single-digit milliseconds at 99%+ recall (VERIFIED as a claim; it is
marketing).

The minimum SDK version for `search_vectors` was **not found**.

**Embedding reference prices**
- **OpenAI `text-embedding-3-small`:** $0.02 per 1M tokens, Batch supported.
  VERIFIED: developers.openai.com/api/docs/models/text-embedding-3-small.
  - Default 1536 dimensions, reducible with `dimensions`: UNVERIFIED (from
    memory).
- **Bedrock Titan Text Embeddings V2:** 256, 512 or 1024 dimensions, 8,192
  tokens maximum. VERIFIED.
  - Price $0.02 per 1M tokens: UNVERIFIED (search snippet only).

## Appendix C: unverified or estimated

- **Jev price.** About $0.37 per 1,000 zero-shot requests is the owner's figure.
  Few-shot at about $0.65 per 1,000 is an estimate: 1.9× the input tokens, and
  pricing by tokens is assumed.
- **Throughput.** The ≈30 req/s rate is unverified.
- **OpenAI prices.** `gpt-6-luna` pricing is unknown; analyst and labeler calls
  are expected to cost cents, but this is unmeasured.
- **Jev state fields.** Whether Jev reads extra example fields (e.g. a
  `reason`) in `labeled_examples` is untested. Rationale-augmented examples are
  out of scope.
- **Local ONNX option.** The download size and quality of a small `fastembed`
  model are not checked.
- **Request estimates in §5.** They assume about 20–25% disagreement per round
  and checkpoint reuse as in §3.2. The ledger ceiling, not the estimate, is the
  control.
