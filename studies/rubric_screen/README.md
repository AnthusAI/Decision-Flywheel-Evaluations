# Rubric screen (R0 research, R1 prompts)

Checked 2026-10-01. No model calls. Machine-readable record: `candidates.json`. Prompts: `prompts/<name>/` (`S.txt` one-line rubric, `F.txt` guideline excerpt where redistributable, `cuts.md`, `answer_format.json`). Datasets live in `.data/rubric-screen/<name>/` (gitignored, never committed). This file has no dataset rows.

| Candidate | Verdict | Licence (verified from official page) | Guideline | Items / labels | Agreement or ceiling | Annotators |
|---|---|---|---|---|---|---|
| EDOS subtask B (`edos`) | usable | CC0 1.0, repo LICENSE (github.com/rewire-online/edos) | repo `guidelines/EDOS Guidelines.pdf`, 2,927 words (~3.8k tokens); F = ~3.0k tokens, committed | 4,854 sexist items / 4 labels (443, 2,271, 1,665, 475) | none published; best model macro-F1 0.7326 (not human); our derived single-annotator match with final label 0.64 | 19 trained annotators, 3 per item, expert adjudication (paper sec 3.4) |
| FOMC hawkish/dovish (`fomc`) | usable, small S-to-F gap expected | CC BY-NC 4.0 (HF card, paper abstract, GitHub LICENSE.md) | paper Table 10 + Section 3, ~600 words (~800 tokens); F ~750 tokens, committed (paper CC BY 4.0) | 2,480 sentences / 3 labels (650 dovish, 606 hawkish, 1,224 neutral) | raw percent agreement 0.89 to 0.95 (paper Table 9) | 2 researchers using the guide, not a trained workforce |
| CAP, NYT front page (`cap_nyt`) | usable with caveats | CC BY-NC-SA 4.0 for project-generated variables; NYT original copyright applies to headline text (comparativeagendas.net/pages/Copyright-and-Legal) | Boydstun codebook PDF, 87 pages, ~38.9k tokens full; F = six topics, ~10.6k tokens, NOT committed (no codebook licence) | 3,447 of 31,034 headlines / 6 majors (964, 280, 283, 413, 1,253, 254) | not found (UNVERIFIED) | coders under a project director (UNVERIFIED) |
| ASAP-SAS (`asap_sas`) | skip | Kaggle competition terms: not retrievable without login (UNVERIFIED) | Kaggle data page only (not retrieved) | not verified | not retrieved | not verified |

Skipped: ASAP-SAS (account-gated; the only public copy found, `nlpatunt/D_ASAP-SAS`, has labels removed and no licence). CAP congressional bills (116 MB, not needed for the NYT headline route). Gab Hate Corpus fallback was not researched.

Unverified items are listed per candidate in `candidates.json` under `flags_unverified`.
