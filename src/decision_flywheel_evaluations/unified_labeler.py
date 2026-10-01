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
