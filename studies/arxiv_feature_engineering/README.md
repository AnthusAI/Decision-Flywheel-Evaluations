# arXiv feature-engineering lab notes

## 2026-10-05: Test supporting questions individually

Hypothesis: a supporting classification can provide useful learned features even
when a previous group of questions failed to improve the fitted classifier.
Feature discovery is separate from deployment. A losing trial does not delete
its question or invalidate the intended concept.

We used the frozen retrospective human-feedback snapshot from the arXiv reviewer:
65 training items (4 Include, 61 Exclude), four development items (two per class),
and a four-item balanced audit. The broader audit pool stayed protected. The
optimizer saw training labels and comments only. These private human labels are
not redistributed, so an external reader cannot reproduce the exact numbers
from public arXiv records alone.

Jev `jev-1.13.0` supplied probabilities. GPT-6 Luna proposed one rubric, a
seven-item example list, and three supporting questions. Every candidate used
the same empty incumbent: no rubric, examples, additional questions, or fitted
head. Each question was tested alone, with the main decision still present.
All candidates used equal-class training weighting, the same numerical fitting
procedure with out-of-fold calibration, and equal-class development Brier loss
for selection. The best qualifying candidate was promoted only after comparison.

| Candidate | Development agreement | Brier loss (lower is better) | Qualified |
| --- | --- | ---: | --- |
| Common incumbent | 2/4 | 0.5691 | Baseline |
| Rubric only | 2/4 | 0.5552 | Yes |
| Examples only | 3/4 | 0.4052 | Yes; selected |
| Knowledge clarification/curation question | 2/4 | 0.8212 | No; retained |
| Memory-systems question | 3/4 | 0.4548 | Yes; retained |
| Research-literature information-work question | 3/4 | 0.5025 | Yes; retained |

The individual questions used `yes`, `no`, and `unclear` options:

- **Knowledge clarification/curation:** Does the article primarily develop or
  study methods or systems for clarifying, curating, organizing, or enhancing
  knowledge or information?
- **Research-literature information work:** Does the article primarily analyze
  research manuscripts or scientific literature to extract, summarize, organize,
  assess, or predict information about the papers or their impact?
- **Memory systems:** Does the article's primary research subject involve memory
  systems or memory architecture, such as designing, organizing, or optimizing
  computer memory subsystems? Judge the research topic, not incidental uses of
  the word "memory."

The returned features showed descriptive training separation:

| Mean Jev probability of `yes` | Include (n=4) | Exclude (n=61) |
| --- | ---: | ---: |
| Knowledge clarification/curation | 0.6650 | 0.2046 |
| Memory systems | 0.2500 | 0.0151 |
| Research-literature information work | 0.6950 | 0.1064 |

All training answers were present. These are training diagnostics, not accuracy
or held-out evidence. Confidence that a topic is present is not confidence that
it predicts the human's inclusion criterion.

On the four-item audit, the selected examples-only classifier agreed with 4/4
labels versus 1/4 for the unfitted incumbent. Audit Brier loss changed from
0.5799 to 0.1592. The audit did not choose the winner or enter training, but these
same few audit items have been reported in earlier retrospective experiments.
This is not a fresh prospective estimate, and four items cannot establish
general accuracy. Baseline responses also differed from the earlier run despite
the same frozen records; a pinned model identifier does not establish repeatable
responses across calls.

### Recovery and actual usage

The literature-question trial initially stopped before fitting because the new
diagnostics rejected a probability distribution totaling 0.99. The existing
decision-answer validator accepts plausible provider rounding. We aligned the
diagnostics with that validator, preserved the original failure, and recovered
in a separate runtime using the unchanged question and cached training answers.
Only four missing development predictions required new Jev requests; no extra
optimizer call was made. Those four requests remained inside the original cap.

Total collection: **357 Jev requests and 3 GPT-6 Luna calls**, below the approved
373/3 ceiling. Returned usage was 443,831 Jev input tokens and 19,623 output
tokens; Luna returned 68,094 prompt tokens and 1,821 completion tokens (69,915
total). No dollar price is inferred. Original and recovered full evidence stay
in ignored private directories in the core repository. The live reviewer was
not replaced by the experiment's classifier.

### Conclusion and next step

Two individual questions improved the development comparison even though the
earlier bundled-question trial did not. This supports investigating their
incremental value instead of prematurely eliminating the concepts. Examples
still won this small comparison. It does not establish that the questions will
improve a classifier that already uses those examples.

Next, test bounded additions to the selected incumbent, combinations, and
leave-one-question-out ablations with more positive labels and adequate
development coverage. Freeze each protocol first; do not choose experiments
from audit errors. Continue retaining questions and wording revisions as new
feedback arrives. No combination or ablation result is claimed here.

## 2026-10-05: Replace the next-step protocol after sample-size review

The user challenged the four-item development sample. We do not treat the
preceding comparisons as reliable improvement evidence. They establish an
execution path and hypotheses, not a validated alignment gain.

The replacement plan separates example-list experimentation, continuous rubric
refinement, and question discovery into independent stages. A new question is
measured alongside the **current** rubric, example list, and existing questions.
Freeze those versions and retroactively re-score eligible reviewed items,
with a configurable maximum of **200 items by default**. Keep the protected
audit and sealed scoreboard separate. Record the actual window, counts by final
label, feedback revisions, context fingerprints, coverage, usage, and progress.
Existing questions' longer histories can be shown, but candidate ranks must use
a common matched window rather than incomparable sample sizes or contexts.

The human's final Include/Exclude label is the ranking target. Supporting
question answers are model-generated features, not human ground truth for those
questions. Direct agreement applies only when answer semantics match the final
classification. Otherwise measure class-conditional probability separation and
out-of-fold learned alignment to the final label, including inverse relationships
and different multiclass options. Show majority-baseline performance, natural and
class-balanced metrics, denominators and uncertainty. Agreement with Jev's own
main answer is a distinct secondary diagnostic.

Feature admission/ranking, head fitting, and classifier deployment are different
decisions. Retrospective association is discovery evidence, not a fresh accuracy
estimate; a 200-item maximum does not imply sufficient minority labels. Retain
questions and revisions as evidence grows. This replacement is recorded, not yet
implemented or run. No additional paid collection is authorized by these notes.

## 2026-10-05: First staged backfill under the current reviewer context

The replacement measurement path is now implemented and was run on a frozen copy
of the actual reviewer's databases, not the empty incumbent from the earlier
experiment. Its rubric and eight-example list stayed fixed. The current supporting
`knowledge_base_inclusion` question stayed present. Luna proposed two additional
classifications, and all questions were scored together on the same 60 eligible
training items: **6 Include and 54 Exclude**. The limit was 200, but only 60 items
were eligible under the existing roles. Development, rolling-audit and sealed
labels were not moved into the measurement pool to inflate its size.

The new classifications were:

- **Central contribution interest area:** classify the area the paper directly
  advances, not merely its terminology or tools. Options covered interactive
  clarification, multiagent learning/reasoning, knowledge extraction/organization/
  entity alignment, memory/storage, longitudinal wearable-data research, multiple
  listed areas, other, and unclear.
- **Central contribution scope:** classify whether the main contribution advances
  a broadly relevant research area, is mainly a specialized application or
  domain-specific result, or is mixed/unclear.

These were inferred from the eligible feedback, not hard-coded into the library.
Their options differ from Include/Exclude, so raw option equality is meaningless.
We used a fixed soft-contingency answer-to-final-label mapping with stratified
five-fold prediction, equal-class fit weighting, and Laplace smoothing of one
per option/class. Review propensities were 1.0 because every displayed item was
reviewed. The implementation also supports inverse-propensity fit weighting.
Every mapping prediction excludes its target's fold from the numerical fit.

| Question | Mapping agreement | Class-balanced alignment | Include correct | Exclude correct |
| --- | ---: | ---: | ---: | ---: |
| Existing inclusion question | 56/60 (93.3%) | 96.3% | 6/6 | 50/54 |
| New contribution-scope question | 47/60 (78.3%) | 73.1% | 4/6 | 43/54 |
| New contribution-interest-area question | 48/60 (80.0%) | 59.3% | 2/6 | 46/54 |

Always predicting Exclude gives 54/60 (90%) raw agreement but only 50% balanced
alignment. This is why ranks use balanced alignment rather than raw agreement
alone. The scope question ranks above the interest-area question even though
its raw agreement is lower. Neither new factor outperformed the existing
inclusion question in this conditional mapping comparison. Definitions and
rankings are retained; no factor or classifier was automatically deployed.

The existing inclusion question's direct label agreement was also 56/60. Do not
interpret any of these values as the deployed learned head's prospective accuracy.
**The folds isolate only the numerical mapping.** Question discovery and the
frozen rubric/examples already reflect historical feedback; other fold items can
also appear among the fixed demonstrations. Excluding the target's own example
does not turn the entire discovery process into independent cross-validation.
These are retrospective, conditional feature-engineering diagnostics. The six
positive items remain a substantial limitation; descriptive Wilson recall
intervals are wide and not adjusted for feature selection.

Collection used **60 Jev requests and one GPT-6 Luna call**. The initial attempt
stopped before Jev collection because the reply copied `input_field` metadata
from existing task definitions. The stage now canonicalizes that redundant field
only when it matches the predefined main input field; attempts to change it are
rejected. Explicit recovery reused the recorded reply and added no optimizer
call. Actual returned usage: Jev 233,749 input and 14,041 output tokens; Luna
20,559 prompt and 1,842 completion tokens, 22,401 total. No pricing is inferred.

The run used Jev `jev-1.13.0` and `gpt-6-luna`. Exact requests, responses, context
versions, window IDs, feedback fingerprints and fold evidence remain in the
ignored `var/arxiv-staged-backfill-v2` directory of the core repository. Public
notes do not redistribute article text, private feedback or credentials. The
original reviewer and its active ML head were unchanged.

The Rich reviewer now runs one configured stage at a time: rubric by default
on its feedback cadence, `N` for question discovery/backfill, and `X` for example
selection. Rubric/example proposals can be recorded before a promotion test;
their staged promotion coverage floor defaults to 20 development items per
class and is configurable, not a statistical guarantee. Question measurement
does not wait for that gate. Feature-set deployment, combinations and ablations
remain separate future work.

## 2026-10-06: train the complete classifier from retained questions

Question measurement alone did not retrain the final ML head. We added a
separate numerical training phase after question discovery, and a standalone
classifier stage. This phase freezes the current rubric and eight examples,
compares current questions against current-plus-retained questions, and fits
each feature set with natural and equal-class weights. Only trusted training
labels fit the head; calibration uses out-of-fold training predictions.

The frozen data contained 60 training labels (6 Include, 54 Exclude) and 27
development labels (2 Include, 25 Exclude). Rolling and final audit items were
not scored, fitted or sent to the optimizer in this experiment. No new optimizer
call was needed: the retained questions came from the preceding discovery run.

| Complete classifier | Ordinary accuracy | Balanced accuracy | Include recall | Exclude recall | Balanced Brier |
|---|---:|---:|---:|---:|---:|
| Existing deployed head | 25/27 (92.6%) | 50% | 0/2 | 25/25 | 0.7755 |
| Current questions, natural fit | 25/27 (92.6%) | 50% | 0/2 | 25/25 | 0.7827 |
| Current questions, equal-class fit | 25/27 (92.6%) | 73% | 1/2 | 24/25 | 0.5409 |
| Current plus retained questions, natural fit | 26/27 (96.3%) | 75% | 1/2 | 25/25 | 0.5256 |
| Current plus retained questions, equal-class fit | 26/27 (96.3%) | 75% | 1/2 | 25/25 | 0.4461 |

The best candidate improved balanced accuracy by 25 percentage points and
balanced Brier by about 42.5%. These are **development selection results**, not
final accuracy. Both improvements in recall rely on correctly predicting one
of only two positive development items. This does not establish reliable
Include recognition or generalization to new reviewer preferences.

The exploratory script explicitly used a development floor of two per class
and disabled deployment. The normal reviewer retains its default promotion
coverage floor of 20 per class. Selection also requires non-decreasing balanced
accuracy and no per-class recall regression. The existing live head and source
review database were unchanged. Retained questions are not discarded after an
unsuccessful trial; new labels permit new fits and evaluations.

The run required **27 new Jev requests**, not the preflight ceiling of 174;
training responses and current-context development responses were cached.
Returned usage for new requests was 106,765 input and 6,308 output tokens.
No dollar estimate is inferred. Private records, fitted candidates, complete
requests and results remain in core `var/arxiv-classifier-training-v1`.

After collection, we activated the selected artifact only inside that private
copy and exercised the public prediction interface on the 27 development
targets. It reproduced 26/27 agreement, including 1/2 Includes, using its 11
probability features and no additional requests. This verified the persisted
Jev-to-features-to-head-to-prediction path, not independent final accuracy.

The reusable `train_classifier` API and reviewer `M` action expose this phase.
Question discovery now hands its measurements to numerical training, which
either evaluates candidates or clearly reports insufficient development
coverage. Cached trials and fitted-artifact provenance survive restarts.
