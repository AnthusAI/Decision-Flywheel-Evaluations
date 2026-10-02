# HateCheck F: what was cut

Source: Röttger et al. 2021 (ACL), Section 1 "Definition of Hate Speech", Sections 2.3-2.4 and Table 1. Paper PDF https://aclanthology.org/2021.acl-long.4.pdf; the ACL Anthology states materials from 2016 on are CC BY 4.0, so the excerpt is committed with attribution. Data: github.com/paul-rottger/hatecheck-data (repo licence CC BY 4.0). F.txt = S.txt + blank line + excerpt (928 words, ~1.2k tokens est.).

Kept verbatim (light trimming of citations): the hate speech definition and protected-group list; the Section 2.3 overview of the 18 hateful and 11 contrastive non-hateful functional tests; the seven protected groups from Section 2.4.
Transcribed: Table 1 as plain text (functional test, example cases, gold label). Dropped columns: n and the four model accuracy columns. Slurs and profanity keep the paper's asterisks.
Cut: interviews, selection criteria, case generation, model evaluation, appendices.
Added by us: a header line naming the source; one sentence saying [IDENTITY]/[SLUR] are placeholders; "non-hate" written as "non-hateful" to match the option name.

Caveat: this is the authors' specification of the gold labels (each case was written to one functional test). The instructions the ten annotators received were not published; the paper says they were onboarded and tested on 100 cases.

Held-out leakage guard: templates whose text is a Table 1 example (33 of the first draft's held-out cases) are kept in the training pool only, never in the held-out slice.

Option order (fixed): hateful, non-hateful (dataset `label_gold` values, unchanged).
