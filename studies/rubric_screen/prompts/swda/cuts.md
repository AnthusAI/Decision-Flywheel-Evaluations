# SwDA F: what was cut

F is NOT committed. The coders manual (Jurafsky, Shriberg and Biasca 1997, "Switchboard SWBD-DAMSL Shallow-Discourse-Function Annotation Coders Manual, Draft 13", https://web.stanford.edu/~jurafsky/ws97/manual.august1.html) carries no licence, so the excerpt lives at `var/rubric-screen/swda/F.txt`.
F.txt sha256: 60d9518d7dd22ddb4b9748baa428ed4759f25d79e20b807bda06ecf1f53033e4 (5,985 words, ~7.8k tokens est.; full manual ~17.4k words, ~22.6k tokens).
Rebuild: tag-strip the manual HTML (sha256 0cceae70...4c56, in `.data/rubric-screen/swda/`) and keep these sections in order, each with its unnumbered "Coder's Heuristics" subsections, verbatim:
1c (42 clustered labels, incl. the full table), 2 (units to label), 3 and 3.01 (%), 5, 5.1, 5.1.1 (sd and sv), 5.2.2 (info-requests), 5.2.2.1 (qy), 5.2.2.4 (qr, folded into qy), 5.2.2.6 (^d), 5.2.2.7 (^g tag questions, folded into qy), 5.2.4.3 fe (folded into ba) and fx (folded into sv), 6, 6.1 (agreements), 6.2, 6.2.2, 6.2.2.1 (b), 6.2.2.6 and 6.2.2.6.1 (ba).
Cut: introduction, WS97 project notes, full DAMSL mapping, sample conversation, self/other-talk, information level, all other tags' sections (qw, qo, qh, ad, commits, openings/closings, answers, bh, bk, bf, ^2, hedges, quotation, "+", transcription errors), bibliography.
Added by us: S.txt as the first paragraph and a header naming the source and the folding rule.

Tag mapping (dataset `act_tag` -> option), following manual 1c as implemented in cgpotts/swda `damsl_act_tag`: drop secondary carat dimensions except in qy^d; strip `* ( )`; qr -> qy, fe -> ba, fx -> sv. Raw tags containing `@` (bad segmentation; manual 1c removes them) and double labels (`,` or `;`) are excluded. Largest raw members: sd (sd, sd^e, sd(^q), sd^t), sv (sv, sv(^q), sv^t, fx), b (b, b^r), aa (aa, aa^r), qy (qy, qy^g, qr, qy^t), qy^d, ba (ba, fe, ba^r), % (%).
Item text: previous two turns (consecutive same-speaker utterances merged, each capped to its last 80 words) with speaker letters, then the target utterance between `>>> <<<` with its speaker. Transcript markup (slashes, `{D ...}`, `[ ... + ... ]`, `<laughter>`, `-/`) is kept verbatim: it is what coders saw.
Split by conversation: held-out = the 19 standard test conversations (Stolcke et al. 2000 split, file from NathanDuran/Probabilistic-RNN-DA-Classifier as used by the HF `swda` script); pool = the other 1,136 conversations.

Option order (fixed): sd, sv, b, aa, qy, qy^d, ba, %.
