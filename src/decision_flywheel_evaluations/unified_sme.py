"""The simulated subject-matter expert for Amazon review moderation (streaming SME plan, step 2).

The SME holds the full written policy F (``var/policy/amazon_reviews_F.txt``, private; only its sha256 and
an own-words summary are committed) and labels reviews with it. The classifier never sees F; it starts
from the one-line description S (``studies/amazon_reviews/S.txt``).

* **Prompt.** One call labels a batch of up to ``BATCH_SIZE`` (10) reviews. The SME sees F and the review
  texts only, never any classifier's verdict, so one cached record serves every arm. Each decision is
  ``{id, label, rule_id, reason, quote}``: label from ``LABELS`` (or ``ambiguous``/``UNEXPLAINABLE``), the
  deciding rule id from F, a reason of at most 25 words about the review in the SME's own words, and an
  optional verbatim quote FROM THE REVIEW of at most 30 words.
* **Validators.** Label in the set (else rejected, no label); rule id exists in F and belongs to that label,
  reason non-empty, at most 25 words and sharing no 8-word run with F (else ``label_only``: the label is kept,
  the explanation dropped); a quote that is not a substring of the review is dropped. ``ambiguous`` records
  carry no label. Items a reply leaves out are reported as missing and not cached.
* **Cache.** SQLite keyed by item id + policy sha256 + prompt version + model (+ pass tag for the agreement
  pass); a rerun replays for $0. Reports are text-free (counts, hashes, ids).
* **Completions are pluggable.** ``fake_sme_completion`` (offline keyword rules, for specs and dry runs),
  ``refusing_completion`` (replay), ``LiveCompletion`` (OpenAI through litellm and the gpt-6 shim; default
  ``gpt-6-sol``, falling back to ``gpt-6-luna`` if the first is not found). Live calls happen only from
  ``main`` with ``--live --confirm``, a durable ``--ledger`` and a ``--max-calls`` cap covering the run.
* **Feedback.** ``feedback``/``build_feedback`` turn a cached record into an arm's comment: "should be X
  because ..." when the arm was wrong, "correct: ..." when right; one rule is explained at most
  ``MAX_RULE_REPEATS`` times per batch (later ones get the bare label).
* **Agreement.** ``agreement`` reports raw agreement and Cohen's kappa between two passes.

Run (offline fake by default; the live command is for the main session):

    python scripts/reviews_sme.py label --limit 1500
    python scripts/reviews_sme.py label --limit 1500 --live --confirm --ledger var/amazon-reviews/sme-ledger.json \\
        --max-calls 151 --model gpt-6-sol
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .unified_labeler import MAX_QUOTE_WORDS, MAX_RULE_REPEATS, UNEXPLAINABLE
from .unified_spend import SpendLedger

LABELS = ("abusive", "promotional", "seller_shipping", "price_availability", "approve")   # precedence order
AMBIGUOUS = "ambiguous"
PROMPT_VERSION = "sme-v1"
DEFAULT_MODEL, FALLBACK_MODEL, FAKE_MODEL = "gpt-6-sol", "gpt-6-luna", "fake-sme-0"
BATCH_SIZE = 10
MAX_REASON_WORDS = 25
NGRAM = 8
SME_CALL_CEILING = 400          # cumulative OpenAI calls for the SME across all passes (1,500 + 100 items ~ 160)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "var" / "policy" / "amazon_reviews_F.txt"
S_PATH = ROOT / "studies" / "amazon_reviews" / "S.txt"
VAR_DIR = ROOT / "var" / "amazon-reviews"

SYSTEM_PROMPT = (
    "You are a senior moderator for Amazon product reviews. Apply the moderation policy below exactly. "
    "You will receive a JSON list of reviews, each with an id and text. For EVERY review return one decision. "
    'Reply with JSON only: {"decisions": [{"id": "...", "label": "...", "rule_id": "R..", '
    '"reason": "...", "quote": "..."}]}. '
    f"label is one of: {', '.join(LABELS)}; use \"{AMBIGUOUS}\" (with rule_id null) only when the policy "
    "truly cannot decide. rule_id is the single policy rule that decided the label. reason is one sentence of "
    f"at most {MAX_REASON_WORDS} words about what this particular review contains or does, in your own words; never copy "
    "policy wording. quote is optional: a phrase copied exactly from the review (not from the policy), at most "
    f"{MAX_QUOTE_WORDS} words, or null.\n\nPOLICY:\n")


class SmeReplayMiss(RuntimeError):
    """A replay needed an uncached decision. Nothing was sent."""


class ModelUnavailable(RuntimeError):
    """The requested model was not found; the completion switched to its fallback."""


class UsageError(ValueError):
    """A command-line gate refused before any data was labeled or any client was built."""


@dataclass(frozen=True)
class SmeItem:
    item_id: str
    text: str


@dataclass(frozen=True)
class SmeRecord:
    item_id: str
    label: Optional[str]
    rule_id: Optional[str]
    reason: Optional[str]
    quote: Optional[str]
    status: str                     # accepted, label_only, ambiguous or rejected
    problem: Optional[str] = None


Complete = Callable[[List[Dict[str, str]]], str]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ws(text: str) -> str:
    return " ".join(str(text).split())


def policy_sha(policy: str) -> str:
    return _sha(policy)


def policy_rules(policy: str) -> Dict[str, str]:
    """Rule id -> label, from lines such as ``R4 (promotional). ...``."""
    return {m.group(1): m.group(2) for m in re.finditer(r"^(R\d+) \((\w+)\)\.", policy, re.MULTILINE)}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(tokens: Sequence[str], n: int = NGRAM) -> set:
    return {tuple(tokens[k:k + n]) for k in range(len(tokens) - n + 1)}


def sme_messages(items: Sequence[SmeItem], policy: str) -> List[Dict[str, str]]:
    payload = json.dumps([{"id": i.item_id, "text": i.text} for i in items], ensure_ascii=False)
    return [{"role": "system", "content": SYSTEM_PROMPT + policy},
            {"role": "user", "content": "Reviews:\n" + payload}]


def _items_from(messages: List[Dict[str, str]]) -> List[dict]:
    return json.loads(messages[1]["content"].split("\n", 1)[1])


# ---- validation -----------------------------------------------------------------------------

def validate(answer: dict, item: SmeItem, policy: str, rules: Optional[Dict[str, str]] = None) -> SmeRecord:
    rules = policy_rules(policy) if rules is None else rules
    label = str(answer.get("label") or "").strip()
    if label in (AMBIGUOUS, UNEXPLAINABLE):
        return SmeRecord(item.item_id, None, None, None, None, "ambiguous")
    if label not in LABELS:
        return SmeRecord(item.item_id, None, None, None, None, "rejected", "label-not-in-set")
    rule = str(answer.get("rule_id") or "").strip()
    reason = _ws(answer.get("reason") or "")
    quote = _ws(answer.get("quote") or "") or None
    problem = None
    if quote and (len(quote.split()) > MAX_QUOTE_WORDS or quote not in _ws(item.text)):
        quote, problem = None, "quote-not-verbatim"
    if rule not in rules:
        problem = "unknown-rule"
    elif rules[rule] != label:
        problem = "rule-label-mismatch"
    elif not reason:
        problem = "no-reason"
    elif len(reason.split()) > MAX_REASON_WORDS:
        problem = "reason-too-long"
    elif _ngrams(_tokens(reason)) & _ngrams(_tokens(policy)):
        problem = "reason-recites-policy"
    else:
        return SmeRecord(item.item_id, label, rule, reason, quote, "accepted", problem)
    return SmeRecord(item.item_id, label, None, None, None, "label_only", problem)


def parse_batch(raw: str, items: Sequence[SmeItem], policy: str) -> Optional[Dict[str, SmeRecord]]:
    """Validated records for the batch's items found in the reply; ``None`` if the reply is not JSON."""
    text = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        reply = json.loads(text)
        decisions = reply["decisions"] if isinstance(reply, dict) else reply
        by_id = {str(d["id"]): d for d in decisions if isinstance(d, dict) and "id" in d}
    except (ValueError, KeyError, TypeError):
        return None
    rules = policy_rules(policy)
    return {i.item_id: validate(by_id[i.item_id], i, policy, rules) for i in items if i.item_id in by_id}


# ---- completions ----------------------------------------------------------------------------

_CUES = [  # (label, rule, pattern, reason): checked in precedence order
    ("abusive", "R1", r"\b(crap\w*|shit\w*|damn|pissed|fuck\w*|f\*+\w*)\b", "Swears while describing the item."),
    ("abusive", "R2", r"\b(idiots?|morons?|liars?|stupid (seller|people|company))\b", "Insults the people involved."),
    ("promotional", "R3", r"https?://\S+|www\.\S+|\b\S+@\S+\.\w+", "Sends readers to an outside address."),
    ("promotional", "R4", r"in exchange|free (sample|of charge)|discount(ed)? (price )?(for|in return)|"
                          r"received (this|it) (for free|at a discount)", "Says the item came cheap for writing it."),
    ("promotional", "R5", r"\b(buy|get) (the|a) \w+ (brand )?instead\b|\bcheck out my\b", "Tells readers to shop elsewhere."),
    ("price_availability", "R8", r"\b(cheaper|on sale|coupon|price (went|dropped|increased|changed))\b",
     "Talks about what one shop charged at one time."),
    ("price_availability", "R9", r"\b(out of stock|back.?ordered|not available in)\b", "Complains about stock at a shop."),
]
_SELLER = re.compile(r"\b(seller|refund\w*|customer service|wrong item|return(ed)? (it|this|the)|replacement)\b", re.I)
_SHIPPING = re.compile(r"\b(ship\w*|deliver\w*|arriv\w*|package\w*|packaging|late|transit|courier)\b", re.I)
_VALUE = re.compile(r"\b(for the price|worth (it|the|every)|overpriced|great value)\b", re.I)


def _quote(text: str, match: re.Match) -> str:
    found = re.search(r"(?:\S+\s+){0,3}" + re.escape(match.group(0)) + r"(?:\s+\S+){0,3}", text)
    return found.group(0) if found else match.group(0)


def fake_decision(item_id: str, text: str) -> dict:
    """Deterministic keyword stand-in for the SME (specs and offline dry runs only)."""
    if len(text.split()) < 4:
        return {"id": item_id, "label": AMBIGUOUS, "rule_id": None, "reason": "Too little to judge.", "quote": None}
    seller, shipping = _SELLER.findall(text), _SHIPPING.findall(text)
    for label, rule, pattern, reason in _CUES[:5]:
        match = re.search(pattern, text, re.I)
        if match:
            return {"id": item_id, "label": label, "rule_id": rule, "reason": reason, "quote": _quote(text, match)}
    if len(seller) + len(shipping) >= 2:
        rule = "R6" if len(seller) >= len(shipping) else "R7"
        reason = "Mostly about the seller and money back." if rule == "R6" else "Mostly about the delivery and parcel."
        match = (_SELLER if rule == "R6" else _SHIPPING).search(text)
        return {"id": item_id, "label": "seller_shipping", "rule_id": rule, "reason": reason, "quote": _quote(text, match)}
    for label, rule, pattern, reason in _CUES[5:]:
        match = re.search(pattern, text, re.I)
        if match:
            return {"id": item_id, "label": label, "rule_id": rule, "reason": reason, "quote": _quote(text, match)}
    if _VALUE.search(text):
        return {"id": item_id, "label": "approve", "rule_id": "R10", "reason": "Judges value for money only.",
                "quote": _quote(text, _VALUE.search(text))}
    if seller or shipping:
        return {"id": item_id, "label": "approve", "rule_id": "R12", "reason": "Item review with a brief delivery aside.",
                "quote": None}
    return {"id": item_id, "label": "approve", "rule_id": "R11", "reason": "Talks about the item itself.", "quote": None}


def fake_sme_completion(messages: List[Dict[str, str]]) -> str:
    return json.dumps({"decisions": [fake_decision(i["id"], i["text"]) for i in _items_from(messages)]})


def refusing_completion(messages: List[Dict[str, str]]) -> str:
    raise SmeReplayMiss("replay needed an uncached SME decision")


class LiveCompletion:  # pragma: no cover - live only
    """OpenAI through litellm, built only after every gate. ``model`` changes to ``fallback`` if not found.

    Credentials come from the environment (the gitignored ``.env`` via python-dotenv); nothing here
    reads, prints or stores them. Retries are off: every retry would be an uncounted call.
    """

    def __init__(self, model: str = DEFAULT_MODEL, fallback: Optional[str] = FALLBACK_MODEL, *, max_tokens: int = 8000):
        from dotenv import load_dotenv
        import litellm

        from .unified_openai import allow_gpt6_token_parameter

        load_dotenv(override=False)
        allow_gpt6_token_parameter()
        self._litellm, self.model, self.fallback, self.max_tokens = litellm, model, fallback, max_tokens

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self._litellm.completion(model=f"openai/{self.model}", messages=messages,
                                                max_tokens=self.max_tokens, num_retries=0,
                                                response_format={"type": "json_object"})
        except self._litellm.NotFoundError as error:
            if self.fallback and self.model != self.fallback:
                self.model = self.fallback
                raise ModelUnavailable(f"model not found; switched to {self.fallback}") from error
            raise
        return response.choices[0].message.content or ""


# ---- cache ----------------------------------------------------------------------------------

def cache_key(item_id: str, policy_sha256: str, model: str, *, pass_tag: str = "primary") -> str:
    return _sha(json.dumps([item_id, policy_sha256, PROMPT_VERSION, model, pass_tag]))


class SmeCache:
    """SQLite (gitignored ``var/``): key -> record. Records hold the SME's reason and review quote, never F."""

    def __init__(self, path):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(":memory:" if path is None else str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS sme (key TEXT PRIMARY KEY, item_id TEXT, model TEXT, "
                        "policy_sha256 TEXT, prompt_version TEXT, pass_tag TEXT, record TEXT)")

    def get(self, key: str) -> Optional[SmeRecord]:
        row = self.db.execute("SELECT record FROM sme WHERE key = ?", (key,)).fetchone()
        return SmeRecord(**json.loads(row[0])) if row else None

    def put(self, key: str, record: SmeRecord, *, model: str, policy_sha256: str, pass_tag: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO sme VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (key, record.item_id, model, policy_sha256, PROMPT_VERSION, pass_tag,
                         json.dumps(asdict(record), sort_keys=True)))
        self.db.commit()


# ---- labeling -------------------------------------------------------------------------------

def uncached(items: Sequence[SmeItem], *, model: str, policy: str, cache: SmeCache, pass_tag: str = "primary") -> int:
    sha = policy_sha(policy)
    return sum(cache.get(cache_key(i.item_id, sha, model, pass_tag=pass_tag)) is None for i in items)


def calls_needed(n_uncached: int, *, batch_size: int = BATCH_SIZE, fallback: bool = False) -> int:
    """Upper bound on calls: one per batch, plus one failed attempt if the model falls back."""
    return math.ceil(n_uncached / batch_size) + (1 if fallback and n_uncached else 0)


def label_items(items: Sequence[SmeItem], complete: Complete, *, model: str, policy: str, cache: SmeCache,
                ledger: Optional[SpendLedger] = None, batch_size: int = BATCH_SIZE,
                pass_tag: str = "primary") -> Tuple[Dict[str, SmeRecord], Dict[str, object]]:
    """SME records (item id -> record) and a text-free report. Each call reserves on ``ledger`` first."""
    sha = policy_sha(policy)
    current = lambda: getattr(complete, "model", model)   # noqa: E731 - a live completion may fall back
    records: Dict[str, SmeRecord] = {}
    pending: List[SmeItem] = []
    hits = calls = missing = malformed = 0
    for item in items:
        record = cache.get(cache_key(item.item_id, sha, current(), pass_tag=pass_tag))
        if record is None:
            pending.append(item)
        else:
            records[item.item_id], hits = record, hits + 1
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        messages = sme_messages(batch, policy)
        for attempt in range(2):
            if ledger is not None:
                ledger.set_scope("sme", pass_tag, "openai")
                ledger.reserve()
            try:
                raw = complete(messages)
            except SmeReplayMiss:
                raise
            except ModelUnavailable as error:
                if ledger is not None:
                    ledger.failed(error)
                    ledger.raise_if_tripped()
                if attempt == 0:
                    continue
                raise
            except Exception as error:
                if ledger is not None:
                    ledger.failed(error)
                    ledger.raise_if_tripped()
                raise
            if ledger is not None:
                ledger.succeeded(model=current())
            calls += 1
            break
        parsed = parse_batch(raw, batch, policy)
        if parsed is None:
            malformed += 1
            parsed = {}
        for item in batch:
            record = parsed.get(item.item_id)
            if record is None:
                missing += 1
                continue
            cache.put(cache_key(item.item_id, sha, current(), pass_tag=pass_tag), record, model=current(),
                      policy_sha256=sha, pass_tag=pass_tag)
            records[item.item_id] = record
    return records, report(records, items=len(items), new_calls=calls, cache_hits=hits, missing=missing,
                           malformed_replies=malformed, model=current(), pass_tag=pass_tag, policy_sha256=sha)


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def report(records: Dict[str, SmeRecord], **fields) -> Dict[str, object]:
    rows = list(records.values())
    return {**fields, "prompt_version": PROMPT_VERSION, "labeled": sum(r.label is not None for r in rows),
            "by_label": _count(r.label for r in rows if r.label), "by_status": _count(r.status for r in rows),
            "by_rule": _count(r.rule_id for r in rows if r.rule_id),
            "problems": _count(r.problem for r in rows if r.problem),
            "with_quote": sum(r.quote is not None for r in rows),
            "labels_sha256": _sha(json.dumps(sorted((k, r.label) for k, r in records.items())))}


# ---- feedback and agreement -----------------------------------------------------------------

def feedback(predicted: str, record: SmeRecord, *, explain: bool = True) -> Optional[str]:
    """An arm's comment from the shared SME record: 'should be X because ...' or 'correct: ...'."""
    if record.label is None:
        return None
    reason = record.reason if (explain and record.status == "accepted") else None
    if predicted == record.label:
        return f"correct: {reason}" if reason else "correct."
    if reason:
        return f"should be {record.label} because {reason[0].lower() + reason[1:]}"
    return f"should be {record.label}."


def build_feedback(predictions: Dict[str, str], records: Dict[str, SmeRecord], *,
                   max_rule_repeats: int = MAX_RULE_REPEATS) -> Dict[str, str]:
    """Comments for the predicted items that have an SME label; one rule is explained at most ``max_rule_repeats`` times."""
    uses: Dict[str, int] = {}
    out: Dict[str, str] = {}
    for item_id in sorted(predictions):
        record = records.get(item_id)
        if record is None:
            continue
        explain = record.rule_id is not None and uses.get(record.rule_id, 0) < max_rule_repeats
        if explain and record.status == "accepted":
            uses[record.rule_id] = uses.get(record.rule_id, 0) + 1
        comment = feedback(predictions[item_id], record, explain=explain)
        if comment:
            out[item_id] = comment
    return out


def agreement(first: Dict[str, SmeRecord], second: Dict[str, SmeRecord]) -> Dict[str, object]:
    """Raw agreement and Cohen's kappa over items both passes labeled."""
    ids = sorted(k for k in first if k in second and first[k].label and second[k].label)
    n = len(ids)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    a, b = [first[k].label for k in ids], [second[k].label for k in ids]
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return {"n": n, "agreement": round(observed, 4), "kappa": round(kappa, 4),
            "ambiguous_either": sum(1 for k in first if k in second and not (first[k].label and second[k].label))}


def agreement_subset(item_ids: Sequence[str], size: int, *, seed: int) -> List[str]:
    ids = sorted(item_ids)
    return sorted(random.Random(f"sme-agreement-{seed}").sample(ids, min(size, len(ids))))


# ---- command line ---------------------------------------------------------------------------

def read_items(path, limit: Optional[int] = None) -> List[SmeItem]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [SmeItem(str(r["id"]), str(r["text"])) for r in rows[:limit]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulated SME labels for the Amazon review pool")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("label", "label the first --limit pool reviews"),
                            ("agree", "second SME pass on a seeded subset of already-labeled reviews")):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--pool", type=Path, default=VAR_DIR / "pool.jsonl")
        cmd.add_argument("--policy", type=Path, default=POLICY_PATH)
        cmd.add_argument("--limit", type=int, default=1500)
        cmd.add_argument("--cache", type=Path, default=VAR_DIR / "sme-cache.sqlite")
        cmd.add_argument("--report", type=Path, default=VAR_DIR / f"sme-{name}-report.json")
        cmd.add_argument("--replay", action="store_true", help="cache only; refuses instead of calling")
        cmd.add_argument("--model", default=None, help=f"live/replay model (default {DEFAULT_MODEL})")
        live = cmd.add_argument_group("live mode (never default)")
        live.add_argument("--live", action="store_true")
        live.add_argument("--confirm", action="store_true")
        live.add_argument("--ledger", type=Path, default=None, help="durable cumulative call counter")
        live.add_argument("--max-calls", type=int, default=None)
        live.add_argument("--fallback-model", default=FALLBACK_MODEL)
        if name == "label":
            cmd.add_argument("--labels-out", type=Path, default=VAR_DIR / "sme-labels.jsonl")
        else:
            cmd.add_argument("--subset", type=int, default=100)
            cmd.add_argument("--seed", type=int, default=1)
            cmd.add_argument("--second-model", default=None, help="model for the second pass (default: --model)")
    return parser


def check_gates(args: argparse.Namespace) -> None:
    if args.live:
        if args.replay:
            raise UsageError("--replay is offline and cannot be combined with --live")
        if args.confirm is not True:
            raise UsageError("--live needs --confirm; nothing was labeled and no client was built")
        if args.ledger is None:
            raise UsageError("--live needs --ledger, the durable call counter")
        if args.max_calls is None or not 0 < args.max_calls <= SME_CALL_CEILING:
            raise UsageError(f"--live needs --max-calls between 1 and {SME_CALL_CEILING}")
    elif args.confirm or args.ledger is not None or args.max_calls is not None:
        raise UsageError("live-only options were given without --live; refusing to guess")
    elif args.model is not None and not args.replay:
        raise UsageError("--model is for --live or --replay; offline labeling uses the fake SME")


def _engine(args: argparse.Namespace, model: str):
    if args.live:  # pragma: no cover - live only
        fallback = args.fallback_model if args.fallback_model != model else None
        return LiveCompletion(model, fallback), SpendLedger(args.ledger, SME_CALL_CEILING, max_new=args.max_calls,
                                                            max_consecutive_failures=3, run_label="sme")
    return (refusing_completion if args.replay else fake_sme_completion), None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    check_gates(args)
    policy = Path(args.policy).read_text(encoding="utf-8")
    items = read_items(args.pool, args.limit)
    model = args.model or (DEFAULT_MODEL if (args.live or args.replay) else FAKE_MODEL)
    cache = SmeCache(args.cache)
    if args.command == "label":
        run_model, run_items, tag = model, items, "primary"
    else:
        primary, _ = label_items(items, refusing_completion, model=model, policy=policy, cache=cache)
        subset = set(agreement_subset([k for k, r in primary.items() if r.label], args.subset, seed=args.seed))
        run_model, run_items, tag = args.second_model or model, [i for i in items if i.item_id in subset], "second"
    if args.live:
        fallback = args.fallback_model and args.fallback_model != run_model
        needed = calls_needed(uncached(run_items, model=run_model, policy=policy, cache=cache, pass_tag=tag),
                              fallback=bool(fallback))
        if needed > args.max_calls:
            raise UsageError(f"this run may need up to {needed} calls; --max-calls {args.max_calls} is lower")
    complete, ledger = _engine(args, run_model)
    records, result = label_items(run_items, complete, model=run_model, policy=policy, cache=cache, ledger=ledger,
                                  pass_tag=tag)
    if args.command == "agree":
        result["agreement"] = agreement({k: primary[k] for k in records if k in primary}, records)
        result["primary_model"] = model
    if ledger is not None:  # pragma: no cover - live only
        result["ledger"] = {k: v for k, v in ledger.summary().items() if k not in ("by_arm_round",)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if args.command == "label":
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        args.labels_out.write_text("".join(
            json.dumps({"id": i.item_id, "label": records[i.item_id].label, "rule_id": records[i.item_id].rule_id,
                        "status": records[i.item_id].status}, sort_keys=True) + "\n"
            for i in run_items if i.item_id in records), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("items", "new_calls", "cache_hits", "missing", "by_label", "model")}))
    return 0
