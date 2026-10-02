"""The simulated human's reasons (Phase 2): an explanation labeler for arms A-c and A-c+F.

The labels stay the corpus reference labels, so every arm sees identical labels. This module
writes only the *reason* a reviewer might leave next to a label, the way Jev-Flywheel's
simulated labeler leaves a comment in ``FeedbackItem.edit_comment_value``.

**The hidden rubric.** Jev-Flywheel's corpus has a convention baked into its labels: neutral or
mixed sports talk is labeled positive and neutral workplace talk negative (its README, "The bias
in the data"; the dataset README calls it "labeled positive/negative for domain bias"). The
classifier starts from the simpler rubric "What is the overall sentiment of this text?". The
simulated reviewer here knows the convention and explains labels by it, tersely, without stating
it as a rule (design section 2, level L2). Jev-Flywheel's own recorded simulated labeler used an
uninformative template ("I disagree; the correct label is X"); this is the informative version
the owner chose (Q1, 2026-10-01). L2 leaks by design: a gain shows the comment path works, not
that real reviewers' comments surface hidden factors.

* One completion per labeled item that wants a comment. About 40% of items get none, decided by
  a hash of the item and seed (so it costs nothing and is the same for every model).
* The completion is conditioned on the text and the reference label only, never on any arm's
  prediction, so one comment serves every arm.
* ``complete`` is pluggable: ``fake_completion`` (offline, deterministic, keyword cues from
  Jev-Flywheel's ``scripts/audit_corpus.py``), ``openai_completion`` (live, ``gpt-6-luna``
  through litellm; off by default and gated in the CLI), or ``refusing_completion`` (replay).
* The cache (gitignored ``var/``) is keyed by item, label, prompt SHA-256 and model and holds the
  comment but never the item text; a second run replays it exactly. A comment that quotes its
  item's text is dropped. Reports are text-free: counts, hashes, and a "mentions topic" flag.
"""
from __future__ import annotations

import hashlib
import random
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .unified_spend import SpendLedger

PROMPT_VERSION = "l2-v1"
DEFAULT_MODEL = "gpt-6-luna"
FAKE_MODEL = "fake-labeler-0"
SKIP_SHARE = 0.4
MAX_WORDS = 20
LABELER_CALL_CEILING = 300

HIDDEN_CONVENTION = (
    "When the wording itself is neutral or mixed, we count talk about sports, games or recreation as "
    "positive and talk about workplace operations (meetings, reports, staff, schedules, office admin) "
    "as negative. Otherwise we simply follow the sentiment of the wording.")

SYSTEM_PROMPT = (
    "You are a busy reviewer on a team that labels short texts as positive or negative. "
    f"Your team has an unwritten convention you apply without thinking about it: {HIDDEN_CONVENTION} "
    "You will see one text and the label your team gave it. Write the short note you would leave "
    "next to that label to explain it to a colleague: at most 20 words, plain and informal. "
    "Do not quote the text and do not spell out the convention as a rule; just give your reason.")

# Jev-Flywheel scripts/audit_corpus.py (clone cd4a4886): its coarse keyword cues for the planted bias.
SPORTS = ("practice", "team", "coach", "athlet", "game", "match", "training", "swim", "golf",
          "tennis", "row", "box", "player", "tournament", "season", "field", "gym", "skat",
          "baseball", "track", "soccer", "run")
OFFICE = ("meeting", "office", "employee", "timesheet", "printer", "report", "deadline",
          "department", "manager", "conference", "email", "memo", "staff", "document",
          "schedul", "policy")
TOPIC_WORDS = re.compile(r"\b(sport|sports|game|games|team|play|recreation|athlet\w*|match|"
                         r"work|workplace|office|meeting|business|job|staff|admin\w*|logistics|domain|topic)\b",
                         re.IGNORECASE)


class LabelerReplayMiss(RuntimeError):
    """A replay needed a comment that is not cached. Nothing was sent."""


@dataclass(frozen=True)
class LabelerItem:
    item_id: str
    text: str
    label: str


@dataclass(frozen=True)
class LabelerPrompt:
    item: LabelerItem

    def messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text: {self.item.text}\nLabel: {self.item.label}\nYour note:"}]

    @property
    def sha256(self) -> str:
        return _sha(json.dumps(self.messages(), sort_keys=True))


Complete = Callable[[LabelerPrompt], str]


def labeler_prompt(item: LabelerItem) -> LabelerPrompt:
    return LabelerPrompt(item)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unit(*parts) -> float:
    return int(_sha(json.dumps(parts))[:16], 16) / 16 ** 16


def wants_comment(item_id: str, seed: int) -> bool:
    """About 60% of labels carry a comment; which ones is fixed by the item and seed."""
    return _unit("comment", item_id, seed) >= SKIP_SHARE


def upper_bound_calls(items: Sequence[LabelerItem], *, seed: int) -> int:
    """The most completions a fresh run could make (before anything is cached)."""
    return sum(wants_comment(item.item_id, seed) for item in items)


# ---- completions ----------------------------------------------------------------------------

def fake_completion(prompt: LabelerPrompt) -> str:
    """Offline stand-in: follows the hidden convention through Jev-Flywheel's keyword cues."""
    text, label = prompt.item.text.lower(), prompt.item.label
    pick = int(_unit("phrasing", prompt.item.item_id) * 2)
    if label == "positive" and any(k in text for k in SPORTS):
        return ("Sports talk, so positive for us.", "Game day stuff reads positive here.")[pick]
    if label == "negative" and any(k in text for k in OFFICE):
        return ("Office logistics; we mark that negative.", "Workplace admin, counts as negative.")[pick]
    return (f"Tone is clearly {label}.", f"Reads {label} to me.")[pick]


def refusing_completion(prompt: LabelerPrompt) -> str:
    raise LabelerReplayMiss("replay needed an uncached comment")


def openai_completion(model: str = DEFAULT_MODEL, *, max_tokens: int = 2000) -> Complete:  # pragma: no cover - live only
    """``gpt-6-luna`` (or another OpenAI model) through litellm, built only after every gate.

    Credentials come from the environment (the gitignored ``.env`` via python-dotenv); nothing
    here reads, prints or stores them. Retries are off: every retry would be an uncounted call.
    """
    from dotenv import load_dotenv
    import litellm

    from .unified_openai import allow_gpt6_token_parameter

    load_dotenv(override=False)
    allow_gpt6_token_parameter()

    def complete(prompt: LabelerPrompt) -> str:
        response = litellm.completion(model=f"openai/{model}", messages=prompt.messages(),
                                      max_tokens=max_tokens, num_retries=0)
        return response.choices[0].message.content or ""

    return complete


# ---- cache ----------------------------------------------------------------------------------

def cache_key(item: LabelerItem, prompt: LabelerPrompt, model: str) -> str:
    return _sha(json.dumps([item.item_id, item.label, prompt.sha256, model]))


class CommentCache:
    """Append-only JSONL: key, item ID, label, model, prompt hash and the comment. No item text."""

    def __init__(self, path):
        self.path = Path(path)
        self._rows: Dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._rows[row["key"]] = row

    def rows(self) -> List[dict]:
        return list(self._rows.values())

    def get(self, key: str) -> Optional[dict]:
        return self._rows.get(key)

    def put(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows[row["key"]] = row


def clean(comment: str, text: str) -> Optional[str]:
    """Trim to at most ``MAX_WORDS`` words; ``None`` if empty or if it quotes the item's text."""
    words = " ".join((comment or "").strip().strip('"').split()).split(" ")
    out = " ".join(words[:MAX_WORDS]).strip()
    if not out:
        return None
    tokens = text.split()
    spans = {" ".join(tokens[i:i + 6]) for i in range(max(len(tokens) - 5, 1))}
    if text in comment or any(span and span in out for span in spans if len(span) >= 20):
        return None
    return out


# ---- generation -----------------------------------------------------------------------------

def generate(items: Sequence[LabelerItem], complete: Complete, *, model: str, cache: CommentCache, seed: int,
             ledger: Optional[SpendLedger] = None) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Comments for ``items`` (item ID -> comment) and a text-free report.

    Every new completion reserves one call on ``ledger`` first (if given), so a cap refuses
    before anything is sent. Answers already paid for are cached as soon as they arrive.
    """
    comments: Dict[str, str] = {}
    calls = hits = dropped = skipped = 0
    for item in items:
        if not wants_comment(item.item_id, seed):
            skipped += 1
            continue
        prompt = labeler_prompt(item)
        key = cache_key(item, prompt, model)
        row = cache.get(key)
        if row is not None:
            hits += 1
        else:
            if ledger is not None:
                ledger.set_scope("labeler", "-", "openai")
                ledger.reserve()
            try:
                raw = complete(prompt)
            except LabelerReplayMiss:
                raise
            except Exception as error:
                if ledger is not None:
                    ledger.failed(error)
                    ledger.raise_if_tripped()
                raise
            if ledger is not None:
                ledger.succeeded(model=model)
            calls += 1
            cleaned = clean(raw, item.text)
            dropped += cleaned is None
            row = {"key": key, "item_id": item.item_id, "label": item.label, "model": model,
                   "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt.sha256, "comment": cleaned}
            cache.put(row)
        if row["comment"]:
            comments[item.item_id] = row["comment"]
    return comments, report(comments, items=len(items), skipped=skipped, new_calls=calls, cache_hits=hits,
                            dropped=dropped, model=model)


def report(comments: Dict[str, str], **counts) -> Dict[str, object]:
    texts = list(comments.values())
    return {**counts, "commented": len(comments), "prompt_version": PROMPT_VERSION,
            "mentions_topic": sum(bool(TOPIC_WORDS.search(c)) for c in texts),
            "mean_words": round(sum(len(c.split()) for c in texts) / len(texts), 2) if texts else 0.0,
            "comments_sha256": _sha(json.dumps(sorted(comments.items())))}


def write_comments(path, comments: Dict[str, str]) -> None:
    """The harness's ``--comments`` format: JSONL of ``{item_id, comment}`` (private: ``var/``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"item_id": i, "comment": c}, sort_keys=True) + "\n"
                            for i, c in sorted(comments.items())), encoding="utf-8")


def read_comments(path) -> Dict[str, str]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["item_id"]): str(row["comment"]) for row in rows if row.get("comment")}


def comment_hash(comment: str) -> str:
    return _sha(comment)


# ---- the rubric stakeholder (rubric-dataset initiative, step R4) ------------------------------
#
# A second labeler style, for corpora whose gold labels follow a PUBLISHED guideline (FOMC). Unlike
# the L2 reviewer above, it explains a MISTAKE: it sees the item, the classifier's verdict, the gold
# label and the full guideline, and answers with one sentence "should be <gold> because <rule>" plus a
# verbatim guideline quote of at most 30 words, or UNEXPLAINABLE. Answers are validated here (quote is
# an exact substring of the guideline after whitespace normalization; the rule is not a restated label
# and shares a content word with the quote) and anything else is rejected. UNEXPLAINABLE is counted as
# a label-noise estimate and never forwarded. The harness calls it once per round per comment arm with
# that arm's own errors on the round's new POOL labels, at most 15 explained, at most 3 forwarded per
# quoted rule. The shuffled and noisy controls are transforms of the forwarded comments.

RUBRIC_PROMPT_VERSION = "rubric-stakeholder-v1"
STAKEHOLDER_FAKE_MODEL = "fake-stakeholder-0"
UNEXPLAINABLE = "UNEXPLAINABLE"
MAX_EXPLAINED_PER_ROUND = 15
MAX_RULE_REPEATS = 3
MAX_QUOTE_WORDS = 30
NOISY_SHARE = 0.2
VARIANTS = {"A-c-shuffled": "shuffled", "A-c-noisy": "noisy"}   # every other comment arm: "as-given"
_STOP = frozenset("""a an the and or of to in on at by for with as is are was were be been it its this that these those
when which who than then there their they them not no but if so such any all from into about would should could
because guideline guidelines rule rules sentence sentences label labels text says say counts count""".split())
_SHOULD_BE = re.compile(r"^should be (\w+) because (.+)$", re.IGNORECASE | re.DOTALL)
_VAGUE = ("something about it seems off to me", "it just does not read right", "I would not have called it that")

STAKEHOLDER_SYSTEM = (
    "You review a classifier's mistakes for a team that labels sentences by following the annotation "
    "guideline below. For one sentence you see the classifier's verdict and the label the team gave it. "
    "Explain the team's label by the guideline: reply with JSON only, "
    '{"explanation": "should be <label> because <the guideline rule that applies>", '
    '"quote": "<a verbatim passage of the guideline, at most 30 words>"}. '
    "The explanation is ONE sentence. The quote must be copied exactly from the guideline. Do not restate "
    "the label as the reason and do not invent a rule the guideline does not state. If no guideline rule "
    f"supports the team's label, reply {UNEXPLAINABLE} and nothing else.\n\nGUIDELINE:\n")


@dataclass(frozen=True)
class StakeholderCase:
    item_id: str
    text: str
    verdict: str
    gold: str


@dataclass(frozen=True)
class StakeholderPrompt:
    case: StakeholderCase
    guideline: str
    labels: Tuple[str, ...]

    def messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": STAKEHOLDER_SYSTEM + self.guideline},
                {"role": "user", "content": f"Labels: {', '.join(self.labels)}\nSentence: {self.case.text}\n"
                                            f"Classifier verdict: {self.case.verdict}\n"
                                            f"Team label: {self.case.gold}\nYour JSON:"}]

    @property
    def sha256(self) -> str:
        return _sha(json.dumps(self.messages(), sort_keys=True))


@dataclass(frozen=True)
class Checked:
    status: str                 # accepted, unexplainable or rejected
    comment: Optional[str] = None
    quote: Optional[str] = None
    reason: Optional[str] = None


def stakeholder_prompt(case: StakeholderCase, guideline: str, labels: Sequence[str]) -> StakeholderPrompt:
    return StakeholderPrompt(case, guideline, tuple(labels))


def _ws(text: str) -> str:
    return " ".join(text.split())


def _content(text: str, labels: Sequence[str]) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in _STOP and w not in labels}


def validate_explanation(raw: str, case: StakeholderCase, guideline: str, labels: Sequence[str]) -> Checked:
    """Accept only 'should be <gold> because <rule>' backed by a verbatim guideline quote."""
    text = (raw or "").strip()
    if text.strip('"') == UNEXPLAINABLE:
        return Checked("unexplainable")
    try:
        answer = json.loads(text.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        explanation, quote = _ws(str(answer["explanation"])).strip('"'), _ws(str(answer["quote"])).strip('"')
    except (ValueError, KeyError, TypeError):
        return Checked("rejected", reason="malformed")
    if explanation == UNEXPLAINABLE:
        return Checked("unexplainable")
    if not quote:
        return Checked("rejected", reason="malformed")
    if len(explanation.split()) > 60:
        return Checked("rejected", reason="not-one-sentence")
    match = _SHOULD_BE.match(explanation.rstrip("."))
    if not match or match.group(1).lower() != case.gold:
        return Checked("rejected", reason="not-should-be-gold")
    if len(quote.split()) > MAX_QUOTE_WORDS:
        return Checked("rejected", reason="quote-too-long")
    if quote not in _ws(guideline):
        return Checked("rejected", reason="quote-not-verbatim")
    rule = _content(match.group(2), labels)
    if len(rule) < 3:
        return Checked("rejected", reason="restates-label")
    if not rule & _content(quote, labels):
        return Checked("rejected", reason="rule-not-in-quote")
    sentence = explanation if explanation.endswith(".") else explanation + "."
    return Checked("accepted", comment=f'{sentence} Guideline: "{quote}"', quote=quote)


def guideline_rules(guideline: str, labels: Sequence[str]) -> List[Tuple[str, str, str]]:
    """(label, category, quotable line) for every table rule and label definition in the guideline."""
    rules, category = [], "label definitions"
    for line in guideline.splitlines():
        stripped = line.strip()
        head = stripped.split(":", 1)[0].lower()
        if head in labels and "N/A" not in stripped:
            rules.append((head, category, " ".join(stripped.split()[:MAX_QUOTE_WORDS])))
        elif stripped and ":" not in stripped and len(stripped.split()) <= 4:
            category = stripped
    for sentence in re.split(r"(?<=\.)\s+", _ws(guideline)):
        first = sentence.split(" ", 1)[0].lower()
        if first in labels and sentence.split(" ", 2)[1:2] == ["sentences"]:
            rules.append((first, "label definitions", " ".join(sentence.split()[:MAX_QUOTE_WORDS])))
    return rules


def fake_stakeholder_completion(prompt: StakeholderPrompt) -> str:
    """Offline stand-in: the gold label's guideline line with the most keyword overlap with the item."""
    case, labels = prompt.case, prompt.labels
    words = _content(case.text, labels)
    options = [r for r in guideline_rules(prompt.guideline, labels) if r[0] == case.gold]
    scored = sorted(options, key=lambda r: (-len(words & _content(r[2], labels)), r[2]))
    if not scored or (not words & _content(scored[0][2], labels) and _unit("unexplainable", case.item_id) < 0.25):
        return UNEXPLAINABLE
    if not words & _content(scored[0][2], labels):
        scored = [r for r in scored if r[1] == "label definitions"] or scored
    _, category, quote = scored[0]
    body = quote.split(":", 1)[1].strip() if ":" in quote.split(" ", 1)[0] else quote
    return json.dumps({"explanation": f"should be {case.gold} because the {category} rule reads: {body}",
                       "quote": quote})


def stakeholder_cache_key(case: StakeholderCase, model: str) -> str:
    """Item id + verdict + gold + prompt version (+ model): a new mistake is asked again, a replay is free."""
    return _sha(json.dumps([case.item_id, case.verdict, case.gold, RUBRIC_PROMPT_VERSION, model]))


def explain_errors(cases: Sequence[StakeholderCase], complete: Callable[[StakeholderPrompt], str], *, model: str,
                   cache: CommentCache, guideline: str, labels: Sequence[str], ledger: Optional[SpendLedger] = None,
                   max_explained: int = MAX_EXPLAINED_PER_ROUND,
                   max_rule_repeats: int = MAX_RULE_REPEATS) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Validated explanations (item ID -> comment) for the first ``max_explained`` cases, and a text-free report."""
    comments: Dict[str, str] = {}
    uses: Dict[str, int] = {}
    counts = {"errors_seen": len(cases), "new_calls": 0, "cache_hits": 0, "accepted": 0, "unexplainable": 0,
              "repeat_capped": 0}
    rejected: Dict[str, int] = {}
    for case in cases[:max_explained]:
        key = stakeholder_cache_key(case, model)
        row = cache.get(key)
        if row is not None:
            counts["cache_hits"] += 1
        else:
            if ledger is not None:
                ledger.set_scope("stakeholder", "-", "openai")
                ledger.reserve()
            try:
                raw = complete(stakeholder_prompt(case, guideline, labels))
            except LabelerReplayMiss:
                raise
            except Exception as error:
                if ledger is not None:
                    ledger.failed(error)
                    ledger.raise_if_tripped()
                raise
            if ledger is not None:
                ledger.succeeded(model=model)
            counts["new_calls"] += 1
            checked = validate_explanation(raw, case, guideline, labels)
            row = {"key": key, "item_id": case.item_id, "verdict": case.verdict, "label": case.gold, "model": model,
                   "prompt_version": RUBRIC_PROMPT_VERSION, "status": checked.status, "reason": checked.reason,
                   "comment": checked.comment, "quote": checked.quote}
            cache.put(row)
        if row["status"] == "unexplainable":
            counts["unexplainable"] += 1
        elif row["status"] != "accepted":
            rejected[row["reason"]] = rejected.get(row["reason"], 0) + 1
        else:
            counts["accepted"] += 1
            rule = _ws(row["quote"]).lower()
            uses[rule] = uses.get(rule, 0) + 1
            if uses[rule] > max_rule_repeats:
                counts["repeat_capped"] += 1
            else:
                comments[case.item_id] = row["comment"]
    return comments, {**counts, "rejected": dict(sorted(rejected.items())), "forwarded": len(comments),
                      "model": model, "prompt_version": RUBRIC_PROMPT_VERSION}


def shuffled_comments(comments: Dict[str, str], key: str) -> Dict[str, str]:
    """The same comments, each attached to a different item (a seeded derangement; one comment cannot move: dropped)."""
    ids = sorted(comments)
    if len(ids) < 2:
        return {}
    random.Random(key).shuffle(ids)
    return {ids[(k + 1) % len(ids)]: comments[ids[k]] for k in range(len(ids))}


def noisy_comments(comments: Dict[str, str], golds: Dict[str, str], guideline: str, labels: Sequence[str],
                   key: str, share: float = NOISY_SHARE) -> Tuple[Dict[str, str], int]:
    """About ``share`` of the comments (exactly round(share * n), chosen by hash) become vague or cite a wrong rule."""
    ids = sorted(comments, key=lambda i: (_unit("noisy", key, i), i))[:round(share * len(comments))]
    rules = guideline_rules(guideline, labels)
    out = dict(comments)
    for item_id in ids:
        gold = golds[item_id]
        wrong = [r for r in rules if r[0] != gold]
        if wrong and _unit("noisy-kind", key, item_id) < 0.5:
            _, category, quote = wrong[int(_unit("noisy-rule", key, item_id) * len(wrong))]
            out[item_id] = f'should be {gold} because of the {category} rule. Guideline: "{quote}"'
        else:
            out[item_id] = f"should be {gold}; {_VAGUE[int(_unit('noisy-vague', key, item_id) * len(_VAGUE))]}."
    return out, len(ids)


class RubricStakeholder:
    """What the loop calls once per round per comment arm: that arm's errors in, forwarded comments out."""

    def __init__(self, guideline: str, labels: Sequence[str], complete: Callable[[StakeholderPrompt], str], *,
                 model: str, cache: CommentCache, ledger: Optional[SpendLedger] = None, seed: int = 1):
        self.guideline, self.labels, self.complete = guideline, tuple(labels), complete
        self.model, self.cache, self.ledger, self.seed = model, cache, ledger, seed
        self.records: List[Dict[str, object]] = []

    def comments_for(self, arm: str, round_number: int, cases: Sequence[StakeholderCase]) -> Dict[str, str]:
        comments, record = explain_errors(cases, self.complete, model=self.model, cache=self.cache,
                                          guideline=self.guideline, labels=self.labels, ledger=self.ledger)
        variant, key = VARIANTS.get(arm, "as-given"), f"{self.seed}:{arm}:{round_number}"
        if variant == "shuffled":
            comments = shuffled_comments(comments, key)
        elif variant == "noisy":
            comments, record["noisy_replaced"] = noisy_comments(
                comments, {c.item_id: c.gold for c in cases}, self.guideline, self.labels, key)
        record.update({"arm": arm, "round": round_number, "variant": variant, "forwarded": len(comments),
                       "comments_sha256": _sha(json.dumps(sorted(comments.items())))})
        self.records.append(record)
        return comments

    def max_calls(self, *, arms: int, rounds: int) -> int:
        return arms * rounds * MAX_EXPLAINED_PER_ROUND
