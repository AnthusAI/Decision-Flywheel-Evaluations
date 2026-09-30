"""Durable, text-free SQLite ledger for approved physical model requests.

The ledger is deliberately narrower than a provider cache.  A preflight run
names every physical request it is allowed to make, and only whitelisted,
validated decision fields ever cross the SQLite boundary.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from decision_flywheel.models import DecisionResult, DecisionTask


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}$")
_FAILURE_CATEGORIES = frozenset({"model-failure", "malformed-response", "uncertain"})
_USAGE_FIELDS = frozenset({"input_tokens", "output_tokens", "total_tokens", "tokens"})
_MAX_LATENCY_MS = 3_600_000.0


@dataclass(frozen=True)
class ApprovedRequest:
    """The complete text-free provenance for one physical SHA-256 request."""

    request_id: str
    task_fingerprint: str
    state_fingerprint: str
    dataset_revision: str
    model_identity: str

    @property
    def request_key(self) -> str:
        """Alias which makes the physical-key role explicit to callers."""
        return self.request_id

    def validate(self) -> None:
        if not _SHA256.fullmatch(self.request_id):
            raise ValueError("approved request key must be a SHA-256")
        if not _SHA256.fullmatch(self.task_fingerprint):
            raise ValueError("approved request task fingerprint must be a SHA-256")
        if not _SHA256.fullmatch(self.state_fingerprint):
            raise ValueError("approved request state fingerprint must be a SHA-256")
        if not _REVISION.fullmatch(self.dataset_revision):
            raise ValueError("approved request dataset revision must be a 40-character SHA-1")
        if not _SAFE_MODEL.fullmatch(self.model_identity) or ":" not in self.model_identity:
            raise ValueError("approved request needs a safe semantic model identity")

    def as_manifest_entry(self) -> dict[str, str]:
        self.validate()
        return {
            "request_id": self.request_id,
            "task_fingerprint": self.task_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "dataset_revision": self.dataset_revision,
            "model_identity": self.model_identity,
        }


@dataclass(frozen=True)
class LogicalCell:
    """Stable logical provenance which may share a physical request."""

    cell_id: str
    condition: str
    draw: int
    order: str
    target_id: str

    def validate(self) -> None:
        for value in (self.cell_id, self.condition, self.order, self.target_id):
            if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
                raise ValueError("logical cell provenance must be safe, non-empty identifiers")
        if not isinstance(self.draw, int) or isinstance(self.draw, bool) or self.draw < 0:
            raise ValueError("logical cell draw must be a non-negative integer")


@dataclass(frozen=True)
class CacheSnapshot:
    """Read-only result view; physical usage is intentionally never expanded by cells."""

    physical_responses: Mapping[str, Mapping[str, Any]]
    physical_usage: Mapping[str, Mapping[str, float]]
    logical_mapping: Mapping[str, str]

    @property
    def logical_requests(self) -> Mapping[str, str]:
        return self.logical_mapping


class CacheStore:
    """A safe physical-request ledger. Reserve before invoking an injected model."""

    def __init__(
        self,
        path: str,
        *,
        task: DecisionTask,
        preflight_fingerprint: str,
        approved_requests: Sequence[ApprovedRequest],
        ceiling: int,
    ):
        if not isinstance(task, DecisionTask):
            raise ValueError("cache needs a core DecisionTask")
        if not _SHA256.fullmatch(preflight_fingerprint):
            raise ValueError("preflight fingerprint must be a SHA-256")
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 0:
            raise ValueError("ceiling must be non-negative")
        approved = tuple(approved_requests)
        if not approved or any(not isinstance(item, ApprovedRequest) for item in approved):
            raise ValueError("approved requests must be a non-empty typed manifest")
        for item in approved:
            item.validate()
            if item.task_fingerprint != task.fingerprint:
                raise ValueError("approved request task fingerprint does not match DecisionTask")
        if len({item.request_id for item in approved}) != len(approved):
            raise ValueError("approved request keys must be unique")
        if len({item.dataset_revision for item in approved}) != 1:
            raise ValueError("approved request manifest must have one dataset revision")

        self.task = task
        self.ceiling = ceiling
        self.approved = {item.request_id: item for item in approved}
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10.0)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        self._bind_configuration(preflight_fingerprint, approved)

    def __enter__(self) -> "CacheStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None  # type: ignore[assignment]

    @property
    def attempts_used(self) -> int:
        self._require_open()
        row = self.db.execute("SELECT COALESCE(SUM(attempts), 0) FROM requests").fetchone()
        return int(row[0])

    def reserve(self, request_id: str, cell: LogicalCell) -> Mapping[str, Any] | str:
        """Replay success or atomically consume one approved physical attempt."""
        approved = self._approved(request_id)
        cell.validate()
        with self._write_transaction():
            prior = self.db.execute(
                "SELECT request_id, condition, draw, ord, target_id FROM cells WHERE cell_id = ?",
                (cell.cell_id,),
            ).fetchone()
            identity = (request_id, cell.condition, cell.draw, cell.order, cell.target_id)
            if prior is not None and prior != identity:
                raise ValueError("logical cell identity changed")

            row = self.db.execute(
                "SELECT status, payload FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is not None and row[0] == "success":
                self._record_cell(cell, request_id)
                return _load_payload(row[1])
            if row is not None and row[0] == "reserved":
                raise ValueError("physical request is already reserved; recover explicitly")
            if self._attempts_used_in_transaction() >= self.ceiling:
                raise ValueError("approved attempt ceiling exhausted")

            token = uuid.uuid4().hex
            if row is None:
                self.db.execute(
                    """INSERT INTO requests
                       (request_id, task_fingerprint, state_fingerprint, dataset_revision, model_identity,
                        status, attempts, payload, token)
                       VALUES (?, ?, ?, ?, ?, 'reserved', 1, NULL, ?)""",
                    (request_id, approved.task_fingerprint, approved.state_fingerprint,
                     approved.dataset_revision, approved.model_identity, token),
                )
            else:
                changed = self.db.execute(
                    """UPDATE requests SET status = 'reserved', attempts = attempts + 1, payload = NULL, token = ?
                       WHERE request_id = ? AND status IN ('failed', 'retryable')""",
                    (token, request_id),
                ).rowcount
                if changed != 1:
                    raise ValueError("physical request is not retryable")
            self._record_cell(cell, request_id)
            return token

    def complete(self, *args: object, request_id: str | None = None) -> Mapping[str, Any]:
        """Store only a validated success, conditionally bound to the live token."""
        request_id, token, response = self._complete_arguments(args, request_id)
        request_id = self._request_for_token_argument(token, request_id)
        approved = self._approved(request_id)
        payload = _sanitize(response, self.task, approved.model_identity)
        if payload["status"] != "success":
            self._transition_failure(request_id, token, "malformed-response")
            return payload
        with self._write_transaction():
            changed = self.db.execute(
                """UPDATE requests SET status = 'success', payload = ?, token = NULL
                   WHERE request_id = ? AND status = 'reserved' AND token = ?""",
                (_dump(payload), request_id, token),
            ).rowcount
            if changed != 1:
                raise ValueError("active reservation token required")
        return payload

    def fail(self, *args: object, category: str = "model-failure", request_id: str | None = None) -> None:
        """Finish a live attempt with a small, non-provider failure category."""
        request_id, token, category = self._fail_arguments(args, request_id, category)
        request_id = self._request_for_token_argument(token, request_id)
        if category not in _FAILURE_CATEGORIES:
            raise ValueError("invalid failure category")
        self._transition_failure(request_id, token, category)

    def recover_uncertain(self, *args: object, request_id: str | None = None) -> None:
        """Mark a crashed reservation retryable; its already-spent attempt remains counted."""
        request_id, token = self._recover_arguments(args, request_id)
        request_id = self._request_for_token_argument(token, request_id)
        with self._write_transaction():
            changed = self.db.execute(
                """UPDATE requests SET status = 'retryable', payload = ?, token = NULL
                   WHERE request_id = ? AND status = 'reserved' AND token = ?""",
                (_dump({"status": "failed", "category": "uncertain"}), request_id, token),
            ).rowcount
            if changed != 1:
                raise ValueError("active reservation token required")

    def snapshot(self) -> CacheSnapshot:
        """Return an immutable, text-free view without multiplying shared physical usage."""
        self._require_open()
        physical_responses: dict[str, Mapping[str, Any]] = {}
        physical_usage: dict[str, Mapping[str, float]] = {}
        for request_id, payload_text in self.db.execute(
            "SELECT request_id, payload FROM requests WHERE status = 'success' ORDER BY request_id"
        ):
            payload = _load_payload(payload_text)
            frozen_payload = _freeze_mapping(payload)
            physical_responses[request_id] = frozen_payload
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                physical_usage[request_id] = _freeze_mapping(usage)
        logical_mapping = {
            cell_id: request_id
            for cell_id, request_id in self.db.execute("SELECT cell_id, request_id FROM cells ORDER BY cell_id")
        }
        return CacheSnapshot(
            MappingProxyType(physical_responses),
            MappingProxyType(physical_usage),
            MappingProxyType(logical_mapping),
        )

    def _transition_failure(self, request_id: str, token: str, category: str) -> None:
        with self._write_transaction():
            changed = self.db.execute(
                """UPDATE requests SET status = 'failed', payload = ?, token = NULL
                   WHERE request_id = ? AND status = 'reserved' AND token = ?""",
                (_dump({"status": "failed", "category": category}), request_id, token),
            ).rowcount
            if changed != 1:
                raise ValueError("active reservation token required")

    def _record_cell(self, cell: LogicalCell, request_id: str) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO cells (cell_id, request_id, condition, draw, ord, target_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cell.cell_id, request_id, cell.condition, cell.draw, cell.order, cell.target_id),
        )

    def _approved(self, request_id: str) -> ApprovedRequest:
        if not isinstance(request_id, str) or request_id not in self.approved:
            raise ValueError("request is not in frozen preflight whitelist")
        return self.approved[request_id]

    def _request_for_token_argument(self, token: str, request_id: str | None) -> str:
        if not isinstance(token, str) or not token:
            raise ValueError("active reservation token required")
        if request_id is not None:
            self._approved(request_id)
            return request_id
        # Tokens are random and never persisted outside this table; looking up their
        # request is safe.  The transition still repeats the token in its UPDATE.
        self._require_open()
        row = self.db.execute("SELECT request_id FROM requests WHERE status = 'reserved' AND token = ?", (token,)).fetchone()
        if row is None:
            raise ValueError("active reservation token required")
        return str(row[0])

    @staticmethod
    def _complete_arguments(args: tuple[object, ...], request_id: str | None) -> tuple[str | None, str, Mapping[str, Any]]:
        if len(args) == 2:
            token, response = args
        elif len(args) == 3 and request_id is None:
            request_id, token, response = args
        else:
            raise TypeError("complete expects (token, response) or (request_id, token, response)")
        if not isinstance(token, str) or not isinstance(response, Mapping):
            raise ValueError("active reservation token and response mapping required")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("request is not in frozen preflight whitelist")
        return request_id, token, response

    @staticmethod
    def _fail_arguments(args: tuple[object, ...], request_id: str | None, category: str) -> tuple[str | None, str, str]:
        if len(args) == 1:
            (token,) = args
        elif len(args) == 2:
            first, second = args
            if isinstance(second, str) and second in _FAILURE_CATEGORIES:
                token, category = first, second
            elif request_id is None:
                request_id, token = first, second
            else:
                raise TypeError("ambiguous fail arguments")
        elif len(args) == 3 and request_id is None:
            request_id, token, category = args
        else:
            raise TypeError("fail expects (token[, category]) or (request_id, token[, category])")
        if not isinstance(token, str) or not isinstance(category, str):
            raise ValueError("active reservation token and failure category required")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("request is not in frozen preflight whitelist")
        return request_id, token, category

    @staticmethod
    def _recover_arguments(args: tuple[object, ...], request_id: str | None) -> tuple[str | None, str]:
        if len(args) == 1:
            (token,) = args
        elif len(args) == 2 and request_id is None:
            request_id, token = args
        else:
            raise TypeError("recover_uncertain expects (token) or (request_id, token)")
        if not isinstance(token, str) or (request_id is not None and not isinstance(request_id, str)):
            raise ValueError("active reservation token required")
        return request_id, token

    def _attempts_used_in_transaction(self) -> int:
        return int(self.db.execute("SELECT COALESCE(SUM(attempts), 0) FROM requests").fetchone()[0])

    def _create_schema(self) -> None:
        self._require_open()
        with self._write_transaction():
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                       request_id TEXT PRIMARY KEY,
                       task_fingerprint TEXT NOT NULL,
                       state_fingerprint TEXT NOT NULL,
                       dataset_revision TEXT NOT NULL,
                       model_identity TEXT NOT NULL,
                       status TEXT NOT NULL CHECK(status IN ('reserved', 'retryable', 'failed', 'success')),
                       attempts INTEGER NOT NULL CHECK(attempts >= 1),
                       payload TEXT,
                       token TEXT
                )"""
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS cells (
                       cell_id TEXT PRIMARY KEY,
                       request_id TEXT NOT NULL REFERENCES requests(request_id),
                       condition TEXT NOT NULL,
                       draw INTEGER NOT NULL,
                       ord TEXT NOT NULL,
                       target_id TEXT NOT NULL
                   )"""
            )
            expected_requests = {"request_id", "task_fingerprint", "state_fingerprint", "dataset_revision", "model_identity", "status", "attempts", "payload", "token"}
            expected_cells = {"cell_id", "request_id", "condition", "draw", "ord", "target_id"}
            if {row[1] for row in self.db.execute("PRAGMA table_info(requests)")} != expected_requests \
                    or {row[1] for row in self.db.execute("PRAGMA table_info(cells)")} != expected_cells:
                raise ValueError("unsupported cache schema")

    def _bind_configuration(self, preflight_fingerprint: str, approved: Sequence[ApprovedRequest]) -> None:
        config = {
            "schema": 2,
            "task": {
                "name": self.task.name,
                "labels": list(self.task.labels),
                "instructions": self.task.instructions,
                "input_field": self.task.input_field,
                "fingerprint": self.task.fingerprint,
            },
            "preflight_fingerprint": preflight_fingerprint,
            "approved_requests": [item.as_manifest_entry() for item in sorted(approved, key=lambda item: item.request_id)],
            "ceiling": self.ceiling,
        }
        frozen = _dump(config)
        with self._write_transaction():
            row = self.db.execute("SELECT value FROM meta WHERE key = 'frozen_configuration'").fetchone()
            if row is not None and row[0] != frozen:
                raise ValueError("frozen cache configuration does not match")
            self.db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('frozen_configuration', ?)", (frozen,))

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        self._require_open()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _require_open(self) -> None:
        if self.db is None:
            raise ValueError("cache is closed")


def _sanitize(value: Mapping[str, Any], task: DecisionTask, expected_model: str) -> dict[str, Any]:
    """Canonicalize a provider-shaped result without retaining provider payloads."""
    if not isinstance(value, Mapping):
        return _malformed()
    answer = value.get("answer", value)
    if not isinstance(answer, Mapping):
        return _malformed()
    choice = answer.get("choice", answer.get("label"))
    probabilities = answer.get("probabilities")
    try:
        result = task.validate_result(DecisionResult(choice, probabilities=probabilities))
    except (TypeError, ValueError):
        return _malformed()
    payload: dict[str, Any] = {"status": "success", "choice": result.label, "model": expected_model}
    if result.probabilities is not None:
        payload["probabilities"] = {label: float(result.probabilities[label]) for label in task.labels}

    confidence = _field(value, answer, "confidence")
    if confidence is not _MISSING:
        if not _bounded_number(confidence, 0.0, 1.0):
            return _malformed()
        payload["confidence"] = float(confidence)
    latency = _field(value, answer, "latency_ms")
    if latency is not _MISSING:
        if not _bounded_number(latency, 0.0, _MAX_LATENCY_MS):
            return _malformed()
        payload["latency_ms"] = float(latency)

    model = _field(value, answer, "model")
    if model is not _MISSING and (not isinstance(model, str) or model != expected_model or not _SAFE_MODEL.fullmatch(model)):
        return _malformed()
    usage = _field(value, answer, "usage")
    if usage is not _MISSING:
        if not isinstance(usage, Mapping):
            return _malformed()
        clean_usage: dict[str, float] = {}
        for key in sorted(_USAGE_FIELDS):
            if key not in usage:
                continue
            amount = usage[key]
            if not _bounded_number(amount, 0.0, math.inf):
                return _malformed()
            clean_usage[key] = float(amount)
        if clean_usage:
            payload["usage"] = {key: clean_usage[key] for key in sorted(clean_usage)}
    return payload


_MISSING = object()


def _field(outer: Mapping[str, Any], inner: Mapping[str, Any], key: str) -> Any:
    return outer[key] if key in outer else inner[key] if key in inner else _MISSING


def _bounded_number(value: object, lower: float, upper: float) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and lower <= value <= upper


def _malformed() -> dict[str, str]:
    return {"status": "failed", "category": "malformed-response"}


def _dump(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("stored cache payload is invalid")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("stored cache payload is invalid")
    return payload


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_mapping(item) if isinstance(item, Mapping) else item for key, item in value.items()})
