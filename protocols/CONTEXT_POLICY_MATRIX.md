# Preregistration template: context-policy matrix

This is a template, not a completed study. Fill every bracketed field and
commit it before the first live model call.

## Question

On `[dataset revision]`, does `[target-conditioned policy]` improve `[primary
metric]` over `[fixed random policy]` at `[fixed context budget]` for each
compatible model?

## Models and compatibility

| Model/version | Eligible? | Context representation | Reason for exclusions |
| --- | --- | --- | --- |
| Jev | `[yes/no]` | structured `labeled_examples` + `target` | |
| Kev | `[yes/no]` | TypeSafe-compatible structured state | |
| Laya | `[yes/no]` | `[document actual supported representation]` | |

## Split firewall

- Candidate pool: `[source IDs]`; labels may be read by a policy.
- Development set: `[source IDs]`; labels may choose policy parameters.
- Scoreboard: `[source IDs]`; labels may not influence selection or tuning.
- Manifest: `[committed text-free path]`.

## Matrix and analysis

List every policy, seed, context budget, task wording fingerprint, request
ceiling, cache location, primary metric, and bootstrap plan. Publish every
eligible cell, response count, token use, latency, and missing/failed cell.
