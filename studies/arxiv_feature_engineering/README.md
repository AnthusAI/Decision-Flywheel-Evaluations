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
