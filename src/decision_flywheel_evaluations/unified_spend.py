"""Request accounting, the cumulative ceiling and the circuit breaker for the unified flywheel.

Every engine request the harness makes -- fake or real, zero-shot or few-shot -- passes
through a counting client that:

1. constructs the underlying client *before* spending anything, so a missing credential or
   a bad configuration cannot burn an attempt (the collector's rule);
2. takes an in-flight slot first, so reserved attempts never exceed ``--max-concurrency``;
3. reserves one attempt against the durable, cumulative ceiling (plan: 9,500) and the
   per-invocation cap, and refuses -- before sending -- once either is reached;
4. trips a circuit breaker on HTTP 401/402/403 or on a streak of consecutive failures, after
   which every further call refuses without spending (the collector's breaker).

Attempts are counted, not successes: a failed request was still sent and may still be billed.
Counts are kept per (arm, round, kind) and written as text-free JSON. Nothing here sees or
stores request text.

The repository's SQLite ``CacheStore`` is not reused directly because it binds a frozen,
pre-approved manifest of request IDs. The flywheel cannot know its requests in advance (the
analyst decides which element is asked and the labeled pool decides each few-shot context),
so this ledger keeps the same two guarantees -- a durable cumulative ceiling bound at
creation, reserved before sending -- without the manifest.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

LEDGER_FORMAT = 1
SYSTEMIC_STATUSES = frozenset({401, 402, 403})
_SAFE = re.compile(r"[^A-Za-z0-9._:/@+-]")


class CeilingExhausted(RuntimeError):
    """The approved request ceiling is used up. Nothing was sent."""


class CircuitOpen(RuntimeError):
    """The circuit breaker tripped earlier in this run. Nothing was sent."""


def _safe(value: Any, limit: int = 80) -> str:
    return _SAFE.sub("_", str(value))[:limit]


@dataclass
class ScopeCounts:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class SpendLedger:
    """A durable cumulative request counter with a ceiling bound at creation."""

    path: Optional[Path]
    ceiling: int
    max_new: Optional[int] = None
    max_consecutive_failures: int = 25
    run_label: str = "run"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ceiling, bool) or not isinstance(self.ceiling, int) or self.ceiling < 0:
            raise ValueError("the request ceiling must be a non-negative integer")
        if self.max_new is not None and (isinstance(self.max_new, bool) or self.max_new < 0):
            raise ValueError("max_new must be a non-negative integer")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive")
        self.path = Path(self.path) if self.path else None
        self.used_before = 0
        self.cumulative_scopes: Dict[str, int] = {}
        if self.path and self.path.exists():
            stored = json.loads(self.path.read_text())
            if stored.get("format") != LEDGER_FORMAT or stored.get("ceiling") != self.ceiling:
                raise ValueError("the spend ledger was created with a different ceiling; "
                                 "the cumulative ceiling cannot be changed mid-study")
            self.used_before = int(stored.get("used", 0))
            self.cumulative_scopes = {str(k): int(v) for k, v in (stored.get("by_scope") or {}).items()}
        self.new_attempts = 0
        self.scopes: Dict[Tuple[str, str, str], ScopeCounts] = {}
        self.scope: Tuple[str, str, str] = ("unscoped", "-", "-")
        self.consecutive_failures = 0
        self.tripped: Optional[str] = None
        self.models_reported: set = set()
        self.error_classes: Dict[str, int] = {}
        self._persist()

    # ---- scope ---------------------------------------------------------------------------

    def set_scope(self, arm: str, round_number: Any, kind: str) -> None:
        self.scope = (_safe(arm), _safe(round_number), _safe(kind))

    def _counts(self) -> ScopeCounts:
        return self.scopes.setdefault(self.scope, ScopeCounts())

    # ---- the three transitions -------------------------------------------------------------

    @property
    def used(self) -> int:
        return self.used_before + self.new_attempts

    def reserve(self) -> None:
        with self._lock:
            if self.tripped:
                raise CircuitOpen(self.tripped)
            if self.used >= self.ceiling:
                raise CeilingExhausted("approved request ceiling exhausted")
            if self.max_new is not None and self.new_attempts >= self.max_new:
                raise CeilingExhausted("per-invocation request cap exhausted")
            self.new_attempts += 1
            self._counts().attempts += 1
            key = "|".join((self.run_label, *self.scope))
            self.cumulative_scopes[key] = self.cumulative_scopes.get(key, 0) + 1
            self._persist()

    def succeeded(self, usage: Any = None, model: Any = None) -> None:
        with self._lock:
            self.consecutive_failures = 0
            counts = self._counts()
            counts.successes += 1
            usage = _as_dict(usage)
            counts.input_tokens += int(usage.get("input_tokens") or 0)
            counts.output_tokens += int(usage.get("output_tokens") or 0)
            if model:
                self.models_reported.add(_safe(model))

    def failed(self, error: BaseException) -> None:
        with self._lock:
            self._counts().failures += 1
            self.consecutive_failures += 1
            status = getattr(error, "status", None) or getattr(error, "status_code", None)
            label = _safe(type(error).__name__) + (f"[{_safe(status)}]" if status else "")
            self.error_classes[label] = self.error_classes.get(label, 0) + 1
            if self.error_classes[label] <= 2:  # local stderr only (run logs are gitignored), never the summary
                import sys
                print(f"[failure sample] {label}", file=sys.stderr)
            if status in SYSTEMIC_STATUSES:
                self.tripped = f"provider rejected the account (HTTP {status})"
            elif self.consecutive_failures >= self.max_consecutive_failures:
                self.tripped = f"{self.consecutive_failures} consecutive request failures"

    def raise_if_tripped(self) -> None:
        """Called between steps: a swallowed failure inside a fill must still stop the run."""
        if self.tripped:
            raise CircuitOpen(f"circuit breaker open: {self.tripped}")

    # ---- reporting -------------------------------------------------------------------------

    def by_arm(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for (arm, _round, kind), counts in sorted(self.scopes.items()):
            entry = out.setdefault(arm, {})
            entry[kind] = entry.get(kind, 0) + counts.attempts
            entry["total"] = entry.get("total", 0) + counts.attempts
        return out

    def by_arm_round(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        out: Dict[str, Dict[str, Dict[str, int]]] = {}
        for (arm, round_number, kind), counts in sorted(self.scopes.items()):
            out.setdefault(arm, {}).setdefault(str(round_number), {})[kind] = counts.as_dict()
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "ceiling": self.ceiling, "max_new_this_invocation": self.max_new,
            "used_before_this_invocation": self.used_before, "new_this_invocation": self.new_attempts,
            "cumulative_used": self.used, "circuit_breaker": self.tripped,
            "models_reported": sorted(self.models_reported), "error_classes": dict(sorted(self.error_classes.items())),
            "by_arm": self.by_arm(), "by_arm_round": self.by_arm_round(),
        }

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": LEDGER_FORMAT, "ceiling": self.ceiling, "used": self.used,
                   "by_scope": dict(sorted(self.cumulative_scopes.items()))}
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".ledger-")
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


class _Counting:
    def __init__(self, inner_factory: Callable[[], Any], ledger: SpendLedger,
                 slots: threading.BoundedSemaphore):
        self._inner_factory = inner_factory
        self._inner: Any = None
        self.ledger = ledger
        self.slots = slots

    def _client(self) -> Any:
        # Constructed before any attempt is reserved, so a construction failure spends nothing.
        if self._inner is None:
            self.ledger.raise_if_tripped()
            self._inner = self._inner_factory()
        return self._inner


class CountingAsyncClient(_Counting):
    """``async system_one(state=..., questions=...)``, counted. For ``jev_flywheel.JevSession``."""

    async def system_one(self, *, state: Any, questions: Any) -> Any:
        client = self._client()
        await asyncio.to_thread(self.slots.acquire)
        try:
            self.ledger.reserve()
            try:
                response = await client.system_one(state=state, questions=questions)
            except Exception as error:
                self.ledger.failed(error)
                raise
            self.ledger.succeeded(getattr(response, "usage", None), getattr(response, "model", None))
            return response
        finally:
            self.slots.release()


class CountingSyncClient(_Counting):
    """``system_one(state=..., questions=...)``, counted. For Decision-Flywheel's ``JevAdapter``."""

    def system_one(self, *, state: Any, questions: Any) -> Any:
        client = self._client()
        self.slots.acquire()
        try:
            self.ledger.reserve()
            try:
                response = client.system_one(state=state, questions=questions)
            except Exception as error:
                self.ledger.failed(error)
                raise
            self.ledger.succeeded(getattr(response, "usage", None), getattr(response, "model", None))
            return response
        finally:
            self.slots.release()


def request_upper_bound(arms, *, rounds: int, per_round: int, eval_n: int,
                        seed_answers_cached: bool = True) -> Dict[str, int]:
    """The most requests a fresh run could make, per arm, before anything is cached.

    * A / A+B / A-c (zero-shot), round r: top up existing elements for the new labels (none in
      round 1), ask a new element of every labeled item, and of the evaluation slice.
    * B (few-shot), round r: one answer per labeled item and per evaluation item, because the
      context changes whenever the labeled set does. A+B reuses B's few-shot answers when B
      runs in the same invocation (same pool, same policy, same context fingerprint).
    * F (one request per item, list in the state), round r: predict the new labels with the
      incumbent list (none in round 1), at most three trial lists over the labels so far, then
      the chosen list over every label and the evaluation slice.
    * F-rand: its list over every label and the evaluation slice. A-c+F: A-c's steering
      top-ups plus F's requests for its own question set.
    * 0 and B-local: nothing.
    * A-c-shuffled and A-c-noisy: as A-c. CEIL: its own question of the evaluation slice, once.
    * ``seed_answers_cached=False`` (FOMC: no fixtures ship the seed answers): one more line,
      ``seed-question``, for the seed question over every label and the evaluation slice. Whichever arm
      asks it first pays; every other arm reads the shared cache.
    """
    arms = set(arms)
    out: Dict[str, int] = {}
    for arm in sorted(arms):
        total = 0
        for r in range(1, rounds + 1):
            labeled = r * per_round
            new_labels = per_round if r > 1 else 0
            if arm in ("A", "A+B", "A-c", "A-c+F", "A-c-shuffled", "A-c-noisy"):
                total += new_labels + labeled + eval_n
            if arm == "B" or (arm == "A+B" and "B" not in arms):
                total += labeled + eval_n
            if arm in ("F", "A-c+F"):
                total += new_labels + 3 * labeled + labeled + eval_n
            if arm == "F-rand":
                total += labeled + eval_n
            if arm == "CEIL" and r == 1:
                total += eval_n
        out[arm] = total
    if not seed_answers_cached:
        out["seed-question"] = rounds * per_round + eval_n
    out["total"] = sum(out.values())
    return out


def final_upper_bound(arms, *, eval_n: int) -> Dict[str, int]:
    """The most requests a final run could make: one per evaluation item per frozen bundle.

    No rounds run in a final run; each bundle asks one request per item (fewer when cached --
    e.g. arm 0's holistic answers on paper-600 ship with the fixtures).
    """
    out = {arm: eval_n for arm in sorted(set(arms))}
    out["total"] = sum(out.values())
    return out


def final_d_upper_bound(*, n_labeled: int, eval_n: int) -> Dict[str, int]:
    """Arm D (one retriever variant): one request per labeled item (leave-one-out examples) and one
    per evaluation item (retrieved examples), each carrying every rubric question. 300 + 600 = 900
    for the seed-1 study."""
    out = {"D": n_labeled + eval_n}
    out["total"] = out["D"]
    return out


def concurrency_slots(max_concurrency: int) -> threading.BoundedSemaphore:
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    return threading.BoundedSemaphore(max_concurrency)
