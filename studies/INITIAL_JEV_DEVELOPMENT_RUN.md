# Initial Jev development selection run

Status: live approval granted by the human owner on 2026-09-30 for the two
development-only optimization collections below, within the stated ceilings.
This authorizes no scoreboard, ordering, or cross-model calls. No results are
reported in this document.

## Basis

The [capability pilot](INITIAL_JEV_PILOT.md) completed 21/21 requests on each
dataset with explicit confidence and probability distributions returned; the
largest request (256 AG News examples, local estimate 9,828) succeeded. Text-free
reports: [AG News](pilots/ag_news.compatibility.json),
[Emotion](pilots/emotion.compatibility.json). The design, ladder, conditions,
objectives, and limits are those of [INITIAL_JEV_STUDY.md](INITIAL_JEV_STUDY.md).
The pilot's first AG News ledger aborted on a missing credential before any
request was made; it is not a result.

## Frozen identities

AG News (accuracy objective; 8,000 logical = 8,000 physical development requests):

- Protocol: `914b7e251379553f00b2040c1d6b63c68dd8c41424041f9ed7eddeb070006d63`
- Optimization preflight: `8fdd7cb6613c0cf29db31dddf4f5fe70bc07bc7c6e99e7476d0175743f262432`
- Approved cumulative ceiling: 8,400 physical attempts

Emotion (macro-F1 objective; 12,000 logical = 12,000 physical development requests):

- Protocol: `af1d49119eda713c08ecfa72c7d45134f219e3c6ff81c728217c606d3267297e`
- Optimization preflight: `e3d09d8089264ef8e3644ec9a03bc4a6512d5f2e9545a9145e87d2e96c69d052`
- Approved cumulative ceiling: 12,600 physical attempts

Provider `jev-1.13.0`, base `https://api.typesafe.ai`, timeout 30 seconds, SDK
retries off, TypeSafe SDK 0.7.1, core `6137fa185a1a98afa84b5e6d5948d1780df5d56d`.
Separate ledger per dataset. Development rows only; scoreboard text is never read.

## Execution rules

The first pass makes at most one attempt per physical request. At most one
explicit outer retry per failed request is allowed only afterward, within the
ceiling, and every reserved attempt counts. No truncation, substitution, or
fallback. Incomplete or failed cells stay visible. The winner is chosen only
after all 20 declared trials complete for a dataset; an incomplete search
freezes nothing. Selection uses development labels only.
