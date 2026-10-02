# ContractNLI F: what was cut

Source: Koreeda and Manning 2021 (Findings of EMNLP), Section 2.1 (task formulation), 2.2 (data collection, hypotheses), Section 4 "Negation by Exception", Appendix A.1-A.1.1 and Table 10 (the 17 hypotheses). Paper PDF https://aclanthology.org/2021.findings-emnlp.164.pdf (ACL Anthology, CC BY 4.0), so the excerpt is committed with attribution. Dataset page https://stanfordnlp.github.io/contract-nli/ (dataset CC BY 4.0). F.txt = S.txt + blank line + excerpt (667 words, ~870 tokens est.).

Kept (lightly condensed): the three-way label definition, the document-level premise, the evidence-span notion (for what counts as "mentioned"), the NDA scope, the fixed-hypothesis rationale, the negation-by-exception paragraph with its example, and Table 10 verbatim.
Cut: evidence-identification modelling, baselines, statistics, related work.
Added by us: a header line naming the source and stating that the authors' example-oriented per-hypothesis annotation guideline was never released.

Caveat: the published guideline is thin. Each item already contains its hypothesis, so F adds little beyond S apart from negation by exception; a small S-to-F gap is expected.
No gold spans are used: item text is "Hypothesis: ..." plus the full contract text. Documents over 12,000 tokens (words x 1.3) are dropped, never truncated; the dry run prints how many items that removes.

Option order (fixed): Entailment, Contradiction, NotMentioned (dataset `choice` values, unchanged).
