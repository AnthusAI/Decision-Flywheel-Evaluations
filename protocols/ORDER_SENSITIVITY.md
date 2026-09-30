# Template: Jev demonstration-order sensitivity

This is a template, not an approved preregistration or a completed study.
Do not fill an initial-result anchor with a pilot, development run, partial
scoreboard, or invented observations. The executable planner requires complete
initial single-Jev canonical scoreboard evidence before it can freeze requests.

## Question

With membership held fixed, how much does presenting the same labeled examples
in a different order change agreement with the ground-truth labels?

The initial investigation compares selection and amount with one canonical
ordering. This follow-up comes after those initial results, and before Kev/Laya
comparisons. It must not search the scoreboard for a winning order.

## Required initial anchor

- Dataset and revision: `[dataset/revision]`.
- Committed manifest and binding hash: `[path/hash]`.
- Initial derived protocol identity: `[hash]`.
- Complete initial scoreboard preflight checksum: `[hash]`.
- Complete sanitized initial observations and their content binding: `[path/hash]`.
- Frozen selected-global artifact: `[reference/checksum]`.
- Exact ordering-plan checksum: `[hash]`.

The ordering plan links every logical cell to an initial source request ID.
It retains every initial target, selector, size, selection draw, model/transport
identity, task question, option order, and example membership. Ordered example
IDs and wire fingerprints describe presentation; no selection policy is refit.
Hashes detect changed content; they do not authenticate an author or prove that
an arbitrary supplied result was produced by a provider.

## Treatments and accounting

Freeze canonical class grouping, deterministic interleaving, reverse canonical
order, and at least two distinct shuffled permutation seeds. A proposed set is
shuffled seeds `0, 1, 2, 3, 4`; the actual set must be explicit in the plan and
this preregistration before collection. Non-shuffled treatments use seed zero.
The seed is separate from the training-example selection draw.

Cross every positive initial selector/size/draw/target cell with every declared
treatment. Zero-shot has no examples to reorder and appears only once, under
canonical. For the complete initial 33-condition design, `N` targets and `S`
shuffled seeds produce `N × (1 + 32 × (3 + S))` logical requests. This is a
formula, not a collected result or a physical-attempt ceiling.

Distinct order labels need not be distinct wire requests: a permutation may
produce identical ordered examples, or different selectors may have identical
membership. Deduplicate only complete model/revision/wire fingerprints and
publish both logical and physical counts. Do not discard an order because it
has a less favorable score or identical wire state.

- Exact eligible physical requests: `[count from frozen ordering preflight]`.
- Cumulative paid-attempt ceiling: `[explicit count, including any allowed retries]`.
- Maximum new attempts per invocation: `[explicit cap]`.
- Retry/uncertain recovery policy: `[explicit bounded policy]`.
- Separate ordering ledger and immutable approval: `[paths/hashes]`.

Pilot or initial-study approval does not authorize ordering calls. Do not assume
responses from a separate initial ledger are imported or free; any repeated
canonical calls must be included in collection accounting unless a separately
verified reuse mechanism is specified. No truncation, fallback, or membership
substitution is allowed.

## Descriptive analysis

Publish per selector/size/draw/order and pooled metrics, per-class recall,
probability coverage and calibration metrics when available, latency, physical
attempts, and actual returned token usage. Distinguish pooled predictions from
the arithmetic mean of per-draw metrics.

For every positive selector/size condition, compare canonical with interleaved
and reversed as paired contrasts; also compare canonical with the mean of the
declared shuffled family. Resample targets and selection draws jointly across
arms. Resample permutation seeds only within the declared shuffled family;
canonical/interleaved/reversed are named treatments, not exchangeable random
permutations. Use `[bootstrap seed/resamples]`.

Intervals are descriptive nominal per-contrast 95% intervals, not simultaneous
or multiplicity-adjusted intervals. Do not invent p-values or select the best
order on the scoreboard. Present percentage-point differences separately from
relative percentage differences. Missing/failed/malformed cells stay visible;
incomplete conditions cannot supply paired intervals through complete-case
filtering. Development-only investigations must be labeled separately, not
substituted for this anchored scoreboard follow-up.

## Conclusion boundary

Any conclusion is conditional on this dataset, candidate pool, original
membership/selection policy, fixed Jev version/transport, and declared order
family. Ordering effects do not establish cross-model generality or a new
optimal context selection method.
