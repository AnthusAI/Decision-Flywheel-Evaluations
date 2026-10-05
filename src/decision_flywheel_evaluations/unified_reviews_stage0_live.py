"""Lazy, counted native Jev collection for frozen reviews Stage 0 cells.

This seam deliberately has no CLI or cache-writing policy.  A Stage 0 runner
must complete its cache preflight and explicit live confirmation before it
constructs a collector, then replace a planned missing cache cell with the
text-free ``Stage0LiveResult`` returned by :meth:`Stage0LiveCollector.collect`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping

from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration
from decision_flywheel.models import DecisionTask, Item

from .datasets import normalized_text_hash
from .unified_reviews_manifest import frozen_role_ids
from .unified_spend import CeilingExhausted, CircuitOpen, CountingSyncClient, SpendLedger, concurrency_slots


_CONDITIONS = frozenset(("S", "F"))
_SAFE_MODEL = re.compile(r"[^A-Za-z0-9._:/@+-]")


@dataclass(frozen=True)
class Stage0LiveResult:
    """A text-free replacement value for one planned S/F screen cache cell."""

    item_id: str
    condition: str
    status: str
    label: str | None
    model: str | None
    usage: dict[str, int | float] | None
    latency_seconds: float | None


def _safe_model(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return _SAFE_MODEL.sub("_", value)[:80]


def _safe_usage(value: object) -> dict[str, int | float] | None:
    if not isinstance(value, Mapping):
        return None
    out: dict[str, int | float] = {}
    aliases = {"input_tokens": ("input_tokens", "prompt_tokens"),
               "output_tokens": ("output_tokens", "completion_tokens"),
               "total_tokens": ("total_tokens",)}
    for canonical, names in aliases.items():
        for name in names:
            amount = value.get(name)
            if isinstance(amount, Real) and not isinstance(amount, bool) and math.isfinite(amount):
                out[canonical] = amount
                break
    return out or None


def _environment_client_factory(model: str) -> Any:
    """Construct the SDK client only after a selected live cell passed validation."""
    try:
        from dotenv import load_dotenv
        from typesafe_sdk import TypeSafeClient
        from typesafe_sdk._core.retry import RetryPolicy
    except ImportError as error:  # pragma: no cover - optional live dependency
        raise ImportError("Install decision-flywheel[jev] for live Stage 0 collection.") from error
    load_dotenv(override=False)
    return TypeSafeClient(model=model, retry=RetryPolicy(max_retries=0))


class Stage0LiveCollector:
    """Build identical S/F tasks over exactly one revalidated frozen held-out item."""

    def __init__(self, manifest: Mapping[str, object], *, pool_path: Path, source_manifest_path: Path,
                 s_path: Path, f_path: Path, model: str, ledger: SpendLedger,
                 inner_client_factory: Callable[[], Any] | None = None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Stage 0 Jev model must be explicit")
        if not isinstance(ledger, SpendLedger):
            raise ValueError("Stage 0 collection requires a durable SpendLedger")
        # ``frozen_role_ids`` validates the complete text-free freeze but reads no private input.
        self._heldout_ids = frozenset(frozen_role_ids(manifest, "heldout"))
        self._manifest = manifest
        self._pool_path = Path(pool_path)
        self._source_manifest_path = Path(source_manifest_path)
        self._s_path = Path(s_path)
        self._f_path = Path(f_path)
        self.model = model
        self._ledger = ledger
        self._inner_client_factory = inner_client_factory or (lambda: _environment_client_factory(model))
        self._adapter: JevAdapter | None = None
        self._tasks_validated = False
        self._validated_tasks: dict[str, DecisionTask] = {}

    def _target_and_task(self, condition: str, item_id: str) -> tuple[DecisionTask, Item]:
        if condition not in _CONDITIONS:
            raise ValueError("Stage 0 condition must be S or F")
        if item_id not in self._heldout_ids:
            raise ValueError("Stage 0 only permits frozen heldout IDs")
        entries = self._manifest["universe"]["records"]
        if not isinstance(entries, list):  # guarded by frozen_role_ids; keeps the type boundary explicit
            raise ValueError("frozen reviews universe is invalid")
        try:
            pool_rows = [json.loads(line) for line in self._pool_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("private pool cannot be read") from None
        frozen_ids = [entry["id"] for entry in entries]
        rows = pool_rows[:len(frozen_ids)]
        if len(rows) != len(frozen_ids) or [row.get("id") for row in rows] != frozen_ids:
            raise ValueError("private pool order or IDs differ from the frozen reviews universe")
        by_id = {row["id"]: row for row in rows}
        for entry in entries:
            row = by_id[entry["id"]]
            text = row.get("text")
            if not isinstance(text, str) or normalized_text_hash(text) != entry["normalized_text_sha256"]:
                raise ValueError("private pool review text hash differs from the frozen split")
        self._validate_source_hashes(entries, by_id)
        labels = self._manifest.get("split", {}).get("labels", ())
        if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels):
            raise ValueError("frozen reviews labels are invalid")
        merged = bool(self._manifest.get("split", {}).get("merged"))
        if condition == "S":
            from .unified_reviews import starting_rubric
            instructions = starting_rubric(merged, self._s_path)
        else:
            policy = self._read_policy()
            from .unified_reviews import guideline_text
            # ``guideline_text`` retains the fixed merged-label note when the frozen split needs it.
            instructions = guideline_text(merged, self._f_path)
            if not instructions.startswith(policy):  # defensive: never send a policy different from its checked bytes
                raise ValueError("private policy could not form the frozen F task")
        return DecisionTask("reviews", tuple(labels), instructions), Item(item_id, {"text": by_id[item_id]["text"]})

    def _validate_source_hashes(self, entries: list[Mapping[str, object]], by_id: Mapping[str, Mapping[str, object]]) -> None:
        try:
            source_bytes = self._source_manifest_path.read_bytes()
            source = json.loads(source_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("reviews source manifest cannot be read") from None
        expected = self._manifest.get("source_manifest_sha256")
        if hashlib.sha256(source_bytes).hexdigest() != expected:
            raise ValueError("reviews source manifest hash does not match the frozen split")
        hashes = source.get("pool") if isinstance(source, Mapping) else None
        if hashes is None:
            return
        if not isinstance(hashes, list):
            raise ValueError("reviews source manifest pool is invalid")
        source_hashes = {row.get("id"): row.get("text_sha256") for row in hashes if isinstance(row, Mapping)}
        if len(source_hashes) != len(hashes):
            raise ValueError("reviews source manifest pool is invalid")
        from .amazon_reviews import text_sha256
        for entry in entries:
            expected_text_sha = entry["source_text_sha256"]
            if expected_text_sha is not None and (
                source_hashes.get(entry["id"]) != expected_text_sha
                or text_sha256(str(by_id[entry["id"]]["text"])) != expected_text_sha
            ):
                raise ValueError("private pool source text hash differs from the frozen split")

    def _read_policy(self) -> str:
        try:
            raw_policy = self._f_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ValueError("private policy cannot be read") from None
        expected = self._manifest.get("sme", {}).get("policy_sha256")
        if hashlib.sha256(raw_policy.encode("utf-8")).hexdigest() != expected:
            raise ValueError("private policy hash differs from the frozen SME policy")
        return raw_policy.strip()

    def validate_tasks(self) -> dict[str, DecisionTask]:
        """Validate the complete S/F wire before the first counted provider request."""
        item_id = min(self._heldout_ids)
        s_task, _ = self._target_and_task("S", item_id)
        f_task, _ = self._target_and_task("F", item_id)
        self._validated_tasks = {"S": s_task, "F": f_task}
        self._tasks_validated = True
        return dict(self._validated_tasks)

    def task_fingerprints(self) -> dict[str, str]:
        """Native S/F ``DecisionTask`` hashes, after private-input validation and before any client exists."""
        tasks = self.validate_tasks()
        return {condition: task.fingerprint for condition, task in tasks.items()}

    def _native_adapter(self) -> JevAdapter:
        if self._adapter is None:
            self._adapter = JevAdapter(
                CountingSyncClient(self._inner_client_factory, self._ledger, concurrency_slots(1)),
                configuration=JevConfiguration(model=self.model, max_retries=0),
            )
        return self._adapter

    def collect(self, condition: str, item_id: str) -> Stage0LiveResult:
        """Make at most one physical, ledger-reserved request for one validated cell."""
        if not self._tasks_validated:
            self.validate_tasks()
        task, target = self._target_and_task(condition, item_id)
        self._ledger.set_scope(condition, "0", "screen")
        started = time.perf_counter()
        try:
            result = asyncio.run(self._native_adapter().decide(task, target, ()))
        except (CeilingExhausted, CircuitOpen):
            raise
        except ValueError:
            return Stage0LiveResult(item_id, condition, "malformed", None, None, None,
                                    time.perf_counter() - started)
        except Exception:
            return Stage0LiveResult(item_id, condition, "failed", None, None, None,
                                    time.perf_counter() - started)
        return Stage0LiveResult(item_id, condition, "completed", result.label,
                                _safe_model(result.model), _safe_usage(result.usage),
                                time.perf_counter() - started)


def build_stage0_live_collector(manifest: Mapping[str, object], *, pool_path: Path, source_manifest_path: Path,
                                s_path: Path, f_path: Path, model: str, ledger: SpendLedger,
                                inner_client_factory: Callable[[], Any] | None = None) -> Stage0LiveCollector:
    """Return an inert collector; private files and credentials remain unread until ``collect``."""
    return Stage0LiveCollector(manifest, pool_path=pool_path, source_manifest_path=source_manifest_path,
                               s_path=s_path, f_path=f_path, model=model, ledger=ledger,
                               inner_client_factory=inner_client_factory)
