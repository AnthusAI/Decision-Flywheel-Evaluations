"""Cached-first, frozen-manifest execution seam for the reviews Stage 0 screen.

This module deliberately knows no review or policy text.  It plans the 600
``S``/``F`` cells from a frozen text-free manifest, evaluates existing safe
cache records, and only permits a caller-provided live adapter after the
agreement, caps, and explicit confirmation have all passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json
import os
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .unified_reviews_gate import evaluate_stage0


_CONDITIONS = ("S", "F")
_STATUSES = frozenset(("completed", "failed", "missing", "malformed"))
SCREEN_CACHE_SCHEMA = "decision-flywheel-evaluations/reviews-stage0-screen-cache/v1"


@dataclass(frozen=True)
class Stage0Caps:
    """Separate, pre-approved request bounds for the two Stage 0 providers."""

    jev_ceiling: int
    jev_max_new: int
    sme_ceiling: int
    sme_max_new: int
    rejected_attempt_retry_allowance: int = 0

    def __post_init__(self) -> None:
        for name in ("jev_ceiling", "jev_max_new", "sme_ceiling", "sme_max_new",
                     "rejected_attempt_retry_allowance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Stage 0 request caps must be non-negative integers")
        if self.jev_max_new > self.jev_ceiling or self.sme_max_new > self.sme_ceiling:
            raise ValueError("Stage 0 per-run cap cannot exceed its durable ceiling")
        if self.rejected_attempt_retry_allowance not in (0, 1):
            raise ValueError("Stage 0 permits at most one explicitly approved rejected-attempt retry")
        if self.jev_ceiling > 600 + self.rejected_attempt_retry_allowance or self.jev_max_new > 600:
            raise ValueError("Stage 0 Jev ceiling is hard-capped at 600 requests")
        if self.rejected_attempt_retry_allowance and self.jev_ceiling != 601:
            raise ValueError("the rejected-attempt retry allowance requires the 601-request ceiling")


@dataclass(frozen=True)
class FutureStageCaps:
    """Owner-approved future-stage bounds, committed in evidence but never spent here."""

    stage1_ceiling: int
    stage1_max_new: int
    stage2_ceiling: int
    stage2_max_new: int

    def __post_init__(self) -> None:
        for name in ("stage1_ceiling", "stage1_max_new", "stage2_ceiling", "stage2_max_new"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("future-stage request caps must be non-negative integers")
        if self.stage1_max_new > self.stage1_ceiling or self.stage2_max_new > self.stage2_ceiling:
            raise ValueError("future-stage per-run cap cannot exceed its durable ceiling")


@dataclass(frozen=True)
class Stage0Plan:
    manifest_sha256: str
    labels: tuple[str, ...]
    screen_ids: tuple[str, ...]
    agreement_ids: tuple[str, ...]
    true_labels: Mapping[str, str]
    text_sha256: Mapping[str, str]

    @property
    def cells(self) -> tuple[tuple[str, str], ...]:
        return tuple((condition, item_id) for condition in _CONDITIONS for item_id in self.screen_ids)


@dataclass(frozen=True)
class Stage0Preflight:
    plan: Stage0Plan
    missing_screen_cells: tuple[tuple[str, str], ...]
    missing_agreement_ids: tuple[str, ...]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".stage0-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _safe_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def _manifest_records(manifest: Mapping[str, Any]) -> tuple[str, Sequence[Mapping[str, Any]]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("a validated frozen manifest mapping is required")
    digest = _safe_sha(manifest.get("manifest_sha256"), "manifest_sha256")
    universe = manifest.get("universe")
    if not isinstance(universe, Mapping) or not isinstance(universe.get("records"), Sequence):
        raise ValueError("frozen manifest has no ordered universe records")
    records = universe["records"]
    if len(records) != 1500:
        raise ValueError("Stage 0 requires the frozen first 1500-item universe")
    return digest, records


def build_stage0_plan(manifest: Mapping[str, Any], identity: Mapping[str, str]) -> Stage0Plan:
    """Derive fixed cells only from the frozen manifest, never a mutable cache."""
    from .unified_reviews_manifest import frozen_role_ids
    # The public helper validates the whole frozen schema/hash, not merely fields this runner uses.
    frozen_role_ids(manifest, "heldout")
    digest, records = _manifest_records(manifest)
    if not isinstance(identity, Mapping) or identity.get("split_sha256") != digest:
        raise ValueError("identity split_sha256 must bind the frozen manifest_sha256")
    heldout: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("frozen manifest records are invalid")
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("frozen manifest IDs are invalid")
        seen.add(item_id)
        if record.get("role") == "heldout":
            label = record.get("gold_label")
            if not isinstance(label, str) or not label:
                raise ValueError("frozen heldout item lacks a canonical SME label")
            text_sha = _safe_sha(record.get("normalized_text_sha256"), "frozen review text sha256")
            heldout.append((item_id, label, text_sha))
    split = manifest.get("split")
    labels = tuple(split.get("labels", ())) if isinstance(split, Mapping) else ()
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("frozen manifest must declare canonical split labels")
    if any(label not in labels for _, label, _ in heldout):
        raise ValueError("frozen heldout labels differ from the manifest label set")
    from .unified_sme import agreement_subset
    eligible = [record["id"] for record in records
                if isinstance(record.get("raw_label"), str) and record["raw_label"] in labels]
    agreement = agreement_subset(eligible, 100, seed=1)
    if len(heldout) != 300:
        raise ValueError("frozen manifest must contain exactly 300 heldout items")
    if len(agreement) != 100:
        raise ValueError("frozen manifest must contain the first 100 agreement IDs")
    return Stage0Plan(digest, labels, tuple(item_id for item_id, _, _ in heldout),
                      tuple(agreement), {item_id: label for item_id, label, _ in heldout},
                      {item_id: text_sha for item_id, _, text_sha in heldout})


def _screen_cache(records: Sequence[Mapping[str, Any]], plan: Stage0Plan) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("screen cache records must be a sequence")
    found: dict[tuple[str, str], dict[str, Any]] = {}
    valid = set(plan.cells)
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"item_id", "condition", "predicted_label", "status"}:
            raise ValueError("screen cache records have unsupported fields")
        key = (record["condition"], record["item_id"])
        if key not in valid or key in found:
            raise ValueError("screen cache records must have unique planned cells")
        status, prediction = record["status"], record["predicted_label"]
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("screen cache status is invalid")
        if prediction is not None and (not isinstance(prediction, str) or prediction not in plan.labels):
            raise ValueError("screen cache prediction is invalid")
        if status == "completed" and prediction is None:
            raise ValueError("completed screen cache record requires a prediction")
        found[key] = dict(record)
    return found


def _agreement_cache(records: Sequence[Mapping[str, Any]], plan: Stage0Plan) -> dict[str, dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("agreement cache records must be a sequence")
    found: dict[str, dict[str, Any]] = {}
    expected = set(plan.agreement_ids)
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"item_id", "primary_label", "second_label", "status"}:
            raise ValueError("agreement cache records have unsupported fields")
        item_id = record["item_id"]
        if item_id not in expected or item_id in found:
            raise ValueError("agreement cache records must have unique frozen IDs")
        status = record["status"]
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("agreement cache status is invalid")
        for name in ("primary_label", "second_label"):
            value = record[name]
            if value is not None and (not isinstance(value, str) or value not in plan.labels):
                raise ValueError("agreement cache label is invalid")
        if status == "completed" and (record["primary_label"] is None or record["second_label"] is None):
            raise ValueError("completed agreement cache record requires both labels")
        found[item_id] = dict(record)
    return found


def preflight_stage0(*, manifest: Mapping[str, Any], identity: Mapping[str, str],
                     screen_cache: Sequence[Mapping[str, Any]], agreement_cache: Sequence[Mapping[str, Any]],
                     caps: Stage0Caps) -> Stage0Preflight:
    """Validate static inputs and enumerate misses without creating a ledger or client."""
    if not isinstance(caps, Stage0Caps):
        raise ValueError("validated Stage0Caps are required")
    plan = build_stage0_plan(manifest, identity)
    screen = _screen_cache([{key: row[key] for key in ("item_id", "condition", "predicted_label", "status")}
                            for row in screen_cache], plan)
    agreement = _agreement_cache(agreement_cache, plan)
    missing_screen = tuple(key for key in plan.cells if key not in screen or screen[key]["status"] != "completed")
    missing_agreement = tuple(item_id for item_id in plan.agreement_ids
                              if item_id not in agreement or agreement[item_id]["status"] != "completed")
    return Stage0Preflight(plan, missing_screen, missing_agreement)


def cached_agreement_from_sqlite(manifest: Mapping[str, Any], identity: Mapping[str, str], *,
                                 cache_path: Path, second_model: str | None = None) -> list[dict[str, Any]]:
    """Read only the frozen agreement labels from the existing SQLite cache.

    The connection is explicitly read-only and this returns label/status wires only; reasons,
    quotes, policy text, and review text never leave SQLite through this API.
    """
    plan = build_stage0_plan(manifest, identity)
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError("the existing SME cache is required for cached Stage 0 agreement")
    sme = manifest.get("sme")
    if not isinstance(sme, Mapping) or sme.get("policy_sha256") != identity.get("policy_sha256"):
        raise ValueError("frozen SME policy identity does not match Stage 0 identity")
    primary_model = sme.get("sme_model")
    if not isinstance(primary_model, str) or not primary_model:
        raise ValueError("frozen SME model is invalid")
    second_model = primary_model if second_model is None else second_model
    if not isinstance(second_model, str) or not second_model:
        raise ValueError("second SME model is invalid")
    from .unified_sme import cache_key
    connection = sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True)
    try:
        rows = []
        records = {record["id"]: record for record in manifest["universe"]["records"]}
        for item_id in plan.agreement_ids:
            record = records[item_id]
            primary = record["raw_label"]
            primary_key = cache_key(item_id, identity["policy_sha256"], primary_model, pass_tag="primary")
            primary_stored = connection.execute("SELECT record FROM sme WHERE key = ?", (primary_key,)).fetchone()
            if primary_stored is None:
                raise ValueError("frozen primary SME cache record is missing")
            primary_record = json.loads(primary_stored[0])
            if primary_record.get("label") != primary or primary_record.get("status") not in {"accepted", "label_only"}:
                raise ValueError("frozen primary SME cache record differs from the freeze")
            key = cache_key(item_id, identity["policy_sha256"], second_model, pass_tag="second")
            stored = connection.execute("SELECT record FROM sme WHERE key = ?", (key,)).fetchone()
            second_record = {} if stored is None else json.loads(stored[0])
            second = second_record.get("label")
            if second is not None and (second not in plan.labels or second_record.get("status") not in {"accepted", "label_only"}):
                raise ValueError("second SME cache record is not a canonical completed label")
            rows.append({"item_id": item_id, "primary_label": primary, "second_label": second,
                         "status": "completed" if isinstance(primary, str) and isinstance(second, str) else "missing"})
        return rows
    finally:
        connection.close()


def read_screen_cache(path: Path, plan: Stage0Plan, *, model: str,
                      question_sha256: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Read a text-free, provenance-bound cache.  A missing path is an empty cache."""
    path = Path(path)
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "manifest_sha256", "model", "question_sha256", "cells"}
    if not isinstance(value, Mapping) or set(value) != required or value["schema"] != SCREEN_CACHE_SCHEMA:
        raise ValueError("Stage 0 screen cache schema is invalid")
    if value["manifest_sha256"] != plan.manifest_sha256 or value["model"] != model:
        raise ValueError("Stage 0 screen cache provenance differs from this frozen run")
    expected_questions = ({condition: _safe_sha(question_sha256.get(condition), f"{condition} question sha256")
                          for condition in _CONDITIONS} if question_sha256 is not None else value["question_sha256"])
    if value["question_sha256"] != expected_questions or not isinstance(expected_questions, Mapping):
        raise ValueError("Stage 0 screen cache question fingerprint differs")
    if not isinstance(value["cells"], list):
        raise ValueError("Stage 0 screen cache cells are invalid")
    output = []
    for row in value["cells"]:
        if not isinstance(row, Mapping) or set(row) != {"item_id", "condition", "predicted_label", "status", "text_sha256", "result_model", "usage", "latency_seconds"}:
            raise ValueError("Stage 0 screen cache cell has unsupported fields")
        if row["text_sha256"] != plan.text_sha256.get(row["item_id"]):
            raise ValueError("Stage 0 screen cache review fingerprint differs")
        output.append({key: row[key] for key in ("item_id", "condition", "predicted_label", "status", "result_model", "usage", "latency_seconds")})
    _screen_cache([{key: row[key] for key in ("item_id", "condition", "predicted_label", "status")} for row in output], plan)
    return output


def upsert_screen_cell(path: Path, plan: Stage0Plan, *, model: str, question_sha256: Mapping[str, str],
                       item_id: str, condition: str, predicted_label: str | None, status: str,
                       text_sha256: str, result_model: str | None = None, usage: Mapping[str, Any] | None = None,
                       latency_seconds: float | None = None) -> list[dict[str, Any]]:
    """Atomically replace one planned cell, retaining a safe resume record before the next call."""
    questions = {name: _safe_sha(question_sha256.get(name), f"{name} question sha256") for name in _CONDITIONS}
    _safe_sha(text_sha256, "review text sha256")
    if text_sha256 != plan.text_sha256.get(item_id):
        raise ValueError("Stage 0 cache cell review fingerprint differs from frozen manifest")
    existing = read_screen_cache(path, plan, model=model, question_sha256=questions)
    record = {"item_id": item_id, "condition": condition, "predicted_label": predicted_label, "status": status,
              "result_model": result_model, "usage": dict(usage) if isinstance(usage, Mapping) else None,
              "latency_seconds": latency_seconds}
    combined = [record if (row["condition"], row["item_id"]) == (condition, item_id) else row for row in existing]
    if (condition, item_id) not in {(row["condition"], row["item_id"]) for row in existing}:
        combined.append(record)
    validated = _screen_cache([{key: row[key] for key in ("item_id", "condition", "predicted_label", "status")} for row in combined], plan)
    # Cache text hashes are bound per cell but raw text never enters this file.
    cells = []
    for row in combined:
        cells.append({**row, "text_sha256": plan.text_sha256[row["item_id"]]})
    _atomic_json(Path(path), {"schema": SCREEN_CACHE_SCHEMA, "manifest_sha256": plan.manifest_sha256,
                              "model": model, "question_sha256": questions, "cells": cells})
    return list(validated.values())


def _evidence(*, preflight: Stage0Preflight, screen_cache: Sequence[Mapping[str, Any]],
              agreement_cache: Sequence[Mapping[str, Any]], identity: Mapping[str, str], caps: Stage0Caps,
              future_caps: FutureStageCaps, merge_attempt: int, source_verified: bool) -> dict[str, Any]:
    plan = preflight.plan
    screen = _screen_cache([{key: row[key] for key in ("item_id", "condition", "predicted_label", "status")}
                            for row in screen_cache], plan)
    agreement = _agreement_cache(agreement_cache, plan)
    screen_rows = {condition: [] for condition in _CONDITIONS}
    for condition, item_id in plan.cells:
        record = screen.get((condition, item_id), {"status": "missing", "predicted_label": None})
        screen_rows[condition].append({"item_id": item_id, "true_label": plan.true_labels[item_id],
                                       "predicted_label": record["predicted_label"], "status": record["status"]})
    agreement_rows = []
    for item_id in plan.agreement_ids:
        record = agreement.get(item_id, {"primary_label": None, "second_label": None, "status": "missing"})
        agreement_rows.append({"item_id": item_id, "primary_label": record["primary_label"],
                               "second_label": record["second_label"], "status": record["status"]})
    artifact = evaluate_stage0(identity=identity, labels=plan.labels, screen_expected_ids=plan.screen_ids,
                               agreement_expected_ids=plan.agreement_ids, agreement_rows=agreement_rows,
                               s_rows=screen_rows["S"], f_rows=screen_rows["F"],
                               stage_ceilings={"stage1": {"request_ceiling": future_caps.stage1_ceiling,
                                                           "max_new_requests": future_caps.stage1_max_new},
                                               "stage2": {"request_ceiling": future_caps.stage2_ceiling,
                                                           "max_new_requests": future_caps.stage2_max_new}},
                               synthetic=not source_verified, merge_attempt=merge_attempt)
    usage = {"input_tokens": 0, "output_tokens": 0, "coverage": 0}
    for row in screen_cache:
        value = row.get("usage") if isinstance(row, Mapping) else None
        if isinstance(value, Mapping):
            usage["coverage"] += 1
            for name in ("input_tokens", "output_tokens"):
                if isinstance(value.get(name), (int, float)) and not isinstance(value.get(name), bool):
                    usage[name] += value[name]
    return {"artifact": artifact,
            "counts": {"screen_cells": len(plan.cells), "cached_screen_cells": len(screen),
                       "missing_screen_cells": len(preflight.missing_screen_cells),
                       "agreement_expected": len(plan.agreement_ids), "cached_agreement": len(agreement),
                       "missing_agreement": len(preflight.missing_agreement_ids)}, "usage": usage}


def cached_stage0(*, manifest: Mapping[str, Any], identity: Mapping[str, str],
                  screen_cache: Sequence[Mapping[str, Any]], agreement_cache: Sequence[Mapping[str, Any]],
                  caps: Stage0Caps, future_caps: FutureStageCaps, merge_attempt: int,
                  live_factory: Callable[..., Any] | None = None, source_verified: bool = False) -> dict[str, Any]:
    """Return cache-only evidence.  ``live_factory`` is deliberately never touched."""
    preflight = preflight_stage0(manifest=manifest, identity=identity, screen_cache=screen_cache,
                                 agreement_cache=agreement_cache, caps=caps)
    return _evidence(preflight=preflight, screen_cache=screen_cache, agreement_cache=agreement_cache,
                     identity=identity, caps=caps, future_caps=future_caps, merge_attempt=merge_attempt,
                     source_verified=source_verified)


def native_jev_live_factory(*, inner_client_factory: Callable[[], Any], model: str,
                            task_factory: Callable[[str], Any], item_factory: Callable[[str], Any],
                            max_concurrency: int = 1) -> Callable[[Any], Callable[[str, str], str]]:
    """Build the native ``JevAdapter`` seam without constructing credentials or a client.

    ``task_factory`` and ``item_factory`` may hydrate private text, so they are called only
    when ``run_live_stage0`` has passed every stop gate and selected an actual missing cell.
    The supplied inner client is wrapped by the common durable counter; retries are disabled
    so each provider request is a physical ledger attempt.
    """
    if not isinstance(model, str) or not model or max_concurrency < 1:
        raise ValueError("native Jev factory needs a model and positive concurrency")

    def factory(ledger: Any) -> Callable[[str, str], str]:
        from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration
        from .unified_spend import CountingSyncClient, concurrency_slots

        adapter = JevAdapter(CountingSyncClient(inner_client_factory, ledger, concurrency_slots(max_concurrency)),
                             configuration=JevConfiguration(model=model, max_retries=0))

        def call(condition: str, item_id: str) -> str:
            result = adapter.decide_sync(task_factory(condition), item_factory(item_id))
            return result.label

        return call

    return factory


def run_live_stage0(*, manifest: Mapping[str, Any], identity: Mapping[str, str],
                    screen_cache: Sequence[Mapping[str, Any]], agreement_cache: Sequence[Mapping[str, Any]],
                    caps: Stage0Caps, merge_attempt: int, confirm_live: bool,
                    future_caps: FutureStageCaps,
                    live_factory: Callable[[Any], Callable[[str, str], str]],
                    jev_ledger_path: Path | None = None, screen_cache_path: Path | None = None,
                    model: str | None = None, question_sha256: Mapping[str, str] | None = None,
                    source_verified: bool = False) -> dict[str, Any]:
    """Fill only missing screen cells after all stop gates; never authorizes Stage 1."""
    preflight = preflight_stage0(manifest=manifest, identity=identity, screen_cache=screen_cache,
                                 agreement_cache=agreement_cache, caps=caps)
    if preflight.missing_agreement_ids:
        raise ValueError("complete cached agreement is required before any Stage 0 live screen request")
    if not confirm_live:
        raise ValueError("live Stage 0 misses require explicit --confirm-live")
    if jev_ledger_path is None:
        raise ValueError("live Stage 0 requires a dedicated durable Jev ledger path")
    if screen_cache_path is None or not isinstance(model, str):
        raise ValueError("live Stage 0 requires a durable provenance-bound screen cache")
    if model != identity.get("jev_model"):
        raise ValueError("live Jev model must match the bound Stage 0 identity")
    # Imports and durable reservation are intentionally delayed until every pre-client gate above.
    from .unified_spend import CeilingExhausted, CircuitOpen, SpendLedger
    jev_ledger = SpendLedger(
        Path(jev_ledger_path), caps.jev_ceiling, caps.jev_max_new, run_label="reviews-stage0",
        approved_ceiling_increase=caps.rejected_attempt_retry_allowance,
        ceiling_increase_reason=("replace rejected pre-credit request"
                                 if caps.rejected_attempt_retry_allowance else None),
    )
    live = live_factory(jev_ledger)
    if isinstance(live, tuple) and len(live) == 2:
        call, actual_questions = live
        question_sha256 = actual_questions
    else:
        call = live
    if question_sha256 is None:
        raise ValueError("validated native task fingerprints are required before Stage 0 cache writes")
    # The durable path, not a caller-supplied list, is authoritative after native task validation.
    filled = read_screen_cache(Path(screen_cache_path), preflight.plan, model=model, question_sha256=question_sha256)
    if {(r["condition"], r["item_id"], r["status"], r["predicted_label"]) for r in filled} != {
        (r["condition"], r["item_id"], r["status"], r["predicted_label"]) for r in screen_cache}:
        raise ValueError("supplied Stage 0 screen cache differs from its durable path")
    preflight = preflight_stage0(manifest=manifest, identity=identity, screen_cache=filled,
                                 agreement_cache=agreement_cache, caps=caps)
    if not preflight.missing_screen_cells:
        return _evidence(preflight=preflight, screen_cache=filled, agreement_cache=agreement_cache,
                         identity=identity, caps=caps, future_caps=future_caps, merge_attempt=merge_attempt,
                         source_verified=source_verified)
    for condition, item_id in preflight.missing_screen_cells:
        jev_ledger.set_scope(condition, "0", "screen")
        # A provider exception must not inherit metadata from the previous cell.
        result_model = usage = latency_seconds = None
        try:
            response = call(condition, item_id)
            if hasattr(response, "status") and hasattr(response, "label"):
                status, prediction = response.status, response.label
                result_model = getattr(response, "model", None)
                usage = getattr(response, "usage", None)
                latency_seconds = getattr(response, "latency_seconds", None)
            if isinstance(response, tuple) and len(response) == 2:
                status, prediction = response
                result_model = usage = latency_seconds = None
            elif not hasattr(response, "status"):
                prediction, status = response, "completed"
                result_model = usage = latency_seconds = None
            if status not in _STATUSES or (status == "completed" and prediction is None):
                status, prediction = "malformed", None
        except (CeilingExhausted, CircuitOpen):
            raise
        except Exception:
            # Cache records deliberately retain only a status, never provider exception text.
            prediction, status = None, "failed"
        filled = upsert_screen_cell(Path(screen_cache_path), preflight.plan, model=model,
                                    question_sha256=question_sha256, item_id=item_id, condition=condition,
                                    predicted_label=prediction, status=status,
                                    text_sha256=preflight.plan.text_sha256[item_id], result_model=result_model,
                                    usage=usage, latency_seconds=latency_seconds)
    final = preflight_stage0(manifest=manifest, identity=identity, screen_cache=filled,
                             agreement_cache=agreement_cache, caps=caps)
    return _evidence(preflight=final, screen_cache=filled, agreement_cache=agreement_cache,
                     identity=identity, caps=caps, future_caps=future_caps, merge_attempt=merge_attempt,
                     source_verified=source_verified)
