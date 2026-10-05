"""Pure cached SME feedback adaptation for the reviews stream."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence

from .unified_labeler import MAX_RULE_REPEATS
from .unified_sme import SmeRecord, feedback


@dataclass(frozen=True)
class FeedbackReveal:
    label: str
    comment: Optional[str]
    status: str
    reason_sha256: Optional[str]


class CachedSmeFeedback:
    """One-shot, per-arm feedback views over already-validated SME records.

    It never reads review text or policy text.  ``label_map`` canonically maps
    raw SME labels (notably the merged remove-other labels) before comparison.
    """
    def __init__(self, records: Mapping[str, SmeRecord], *, label_map: Optional[Callable[[str], str]] = None,
                 seed: int = 1, max_rule_repeats: int = MAX_RULE_REPEATS):
        self.records, self.label_map = dict(records), label_map or (lambda value: value)
        self.seed, self.max_rule_repeats = seed, max_rule_repeats
        self._uses: dict[str, dict[str, int]] = {}
        self._revealed: set[tuple[str, str]] = set()
        self._accepted: dict[str, dict[str, SmeRecord]] = {}

    def begin_batch(self, arm: str, batch_index: int) -> None:
        if arm not in ("B", "L", "E", "X"):
            raise ValueError("unknown stream arm")
        self._uses[arm] = {}

    def reveal(self, *, arm: str, item_id: str, predicted: Optional[str], reviewed_prefix: Sequence[str]) -> Optional[FeedbackReveal]:
        if arm not in ("B", "L", "E", "X"):
            raise ValueError("unknown stream arm")
        if arm == "B":
            return None
        key = (arm, item_id)
        if key in self._revealed:
            raise RuntimeError("a selected SME feedback item may be revealed once")
        self._revealed.add(key)
        record = self.records.get(item_id)
        if (not isinstance(record, SmeRecord) or not isinstance(record.label, str)
                or record.status not in ("accepted", "label_only")):
            return None
        label = self.label_map(record.label)
        if not isinstance(label, str) or not label:
            return None
        explain = arm in ("E", "X") and record.status == "accepted" and isinstance(record.reason, str) and bool(record.reason)
        chosen = record
        if arm == "X" and explain:
            history = self._accepted.setdefault(arm, {})
            donors = [history[i] for i in reviewed_prefix if i in history and i != item_id
                      and history[i].status == "accepted" and history[i].reason]
            if donors:
                donor = random.Random(f"stream-x-v1:{self.seed}:{item_id}:{len(donors)}").choice(donors)
                chosen = replace(record, reason=donor.reason, rule_id=donor.rule_id, status="accepted")
            else:
                explain = False
        uses = self._uses.setdefault(arm, {})
        explanation_rule = chosen.rule_id
        if explain and explanation_rule and uses.get(explanation_rule, 0) >= self.max_rule_repeats:
            explain = False
        if explain and explanation_rule:
            uses[explanation_rule] = uses.get(explanation_rule, 0) + 1
        comment = feedback(predicted or "", replace(chosen, label=label), explain=explain)
        self._accepted.setdefault(arm, {})[item_id] = record
        digest = hashlib.sha256((chosen.reason or "").encode()).hexdigest() if explain and chosen.reason else None
        return FeedbackReveal(label, comment, record.status, digest)
