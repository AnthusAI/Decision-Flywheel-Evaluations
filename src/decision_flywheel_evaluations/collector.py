"""Bounded, injected collection over an already frozen offline preflight plan."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import PerLabelLexicalRetrieval, PrototypeBalanced, RandomBalanced
from decision_flywheel.models import DecisionModel, DecisionResult, Item

from .cache import CacheStore, LogicalCell
from .datasets import DatasetRow
from .manifests import DatasetManifest
from .metrics import Observation
from .preflight import (PreflightResult, RequestCell, _FrozenGlobal,
                        _physical_request_fingerprint, core_plan_fingerprints,
                        preflight, rehydrate_manifest)
from .protocol import FrozenProtocol, Selector


@dataclass(frozen=True)
class CollectionApproval:
    """Human confirmation tied to immutable content, protocol and preflight hashes."""

    preregistration_path: str
    preregistration_sha256: str
    protocol_hash: str
    preflight_checksum: str
    attempt_ceiling: int
    confirmed: bool

    def validate(self) -> None:
        hashes = (self.preregistration_sha256, self.protocol_hash, self.preflight_checksum)
        if (not isinstance(self.preregistration_path, str) or not self.preregistration_path
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in hashes)
            or isinstance(self.attempt_ceiling, bool) or not isinstance(self.attempt_ceiling, int)
            or self.attempt_ceiling < 0 or not isinstance(self.confirmed, bool)):
            raise ValueError("collection approval has invalid frozen provenance")


@dataclass(frozen=True)
class CollectionOptions:
    """Explicit controls for a resumable bounded run; defaults never retry."""

    max_new_attempts: int | None = None
    retry_failed: bool = False
    recover_uncertain: bool = False
    max_retries_per_request: int = 0

    def validate(self) -> None:
        if (self.max_new_attempts is not None and (isinstance(self.max_new_attempts, bool)
            or not isinstance(self.max_new_attempts, int) or self.max_new_attempts < 0)
            or not isinstance(self.retry_failed, bool) or not isinstance(self.recover_uncertain, bool)
            or isinstance(self.max_retries_per_request, bool)
            or not isinstance(self.max_retries_per_request, int) or self.max_retries_per_request < 0):
            raise ValueError("collection options must be explicit non-negative bounds")
        if (self.retry_failed or self.recover_uncertain) and self.max_retries_per_request < 1:
            raise ValueError("retry and recovery require an explicit positive retry bound")


@dataclass(frozen=True)
class CollectionResult:
    observations: tuple[Observation, ...]
    physical_attempts: int
    new_attempts: int
    complete: bool


CommittedChecker = Callable[[str, str], bool]
EngineFactory = Callable[[str], DecisionModel]


async def collect(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    frozen_preflight: PreflightResult,
    cache: CacheStore,
    approval: CollectionApproval,
    engine_factory: EngineFactory,
    *,
    committed_checker: CommittedChecker = None,  # type: ignore[assignment]
    options: CollectionOptions = CollectionOptions(),
) -> CollectionResult:
    """Collect only approved physical requests, then export every logical cell.

    The factory is deliberately invoked inside the reserved-request path, after
    every offline, preregistration and cache-ceiling gate has passed.
    """
    checker = committed_checker or git_preregistration_is_committed
    options.validate()
    execution = _validate_gates(protocol, manifest, rows, frozen_preflight, cache, approval, checker)
    return await _collect_execution(protocol, manifest, execution, cache, engine_factory, options)


async def _collect_execution(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    execution: Sequence[tuple[RequestCell, Item, object]],
    cache: CacheStore,
    engine_factory: EngineFactory,
    options: CollectionOptions,
) -> CollectionResult:
    """Execute an already gate-validated physical plan without widening it."""
    initial_successes = cache.successful_request_ids()
    new_attempts = 0
    models: dict[str, DecisionModel] = {}
    cells_by_physical: dict[str, list[tuple[RequestCell, Item, object]]] = {}
    for cell, target, plan in execution:
        cells_by_physical.setdefault(cell.fingerprint, []).append((cell, target, plan))

    for physical_id, planned in sorted(cells_by_physical.items()):
        state = cache.request_state(physical_id)
        status, attempts = state.status, state.attempts
        if status == "success":
            # Replay every logical cell so the cache records its safe logical mapping.
            for cell, _target, _plan in planned:
                cache.reserve(physical_id, _logical_cell(cell))
            continue
        if status == "reserved":
            if not options.recover_uncertain or attempts >= 1 + options.max_retries_per_request:
                continue
            cache.recover_abandoned(physical_id)
            status = "retryable"
        if status in {"failed", "retryable"} and (
            not options.retry_failed or attempts >= 1 + options.max_retries_per_request
        ):
            continue
        if status not in {None, "failed", "retryable"}:
            continue
        if options.max_new_attempts is not None and new_attempts >= options.max_new_attempts:
            continue
        # Reserve atomically before constructing a client. A completed replay
        # returned here is possible only with a concurrent collector.
        try:
            reserved = cache.reserve(physical_id, _logical_cell(planned[0][0]))
        except ValueError as error:
            if str(error) == "approved attempt ceiling exhausted":
                continue
            raise
        if isinstance(reserved, Mapping):
            for cell, _target, _plan in planned[1:]:
                cache.reserve(physical_id, _logical_cell(cell))
            continue
        new_attempts += 1
        cell, target, plan = planned[0]
        model = models.get(cell.model)
        if model is None:
            try:
                model = engine_factory(cell.model)
            except Exception:
                # Factory/configuration errors are not provider result shape
                # errors; record only a safe operational failure category.
                cache.fail(reserved, "model-failure")
                continue
            models[cell.model] = model
        completed = False
        try:
            result = await model.decide(protocol.task, target, plan.examples)
        except ValueError:
            # DecisionResult validation errors represent an invalid structured
            # provider response; keep only the safe ledger category.
            cache.fail(reserved, "malformed-response")
        except Exception:
            # Provider exception contents must never become result provenance.
            cache.fail(reserved, "model-failure")
        else:
            # Persistence/ledger errors are not model failures and must not try
            # to finalize an already-transitioned reservation a second time.
            completed = cache.complete(reserved, _result_payload(result)).get("status") == "success"
        # A success is replayed into the other logical cells; a failure still
        # exports every cell from physical ledger metadata below.
        if completed:
            for other, _target, _plan in planned[1:]:
                cache.reserve(physical_id, _logical_cell(other))

    snapshot = cache.snapshot()
    observations = tuple(_observation(cell, manifest, snapshot, cell.fingerprint in initial_successes)
                         for cell, _target, _plan in execution)
    return CollectionResult(observations, cache.attempts_used, new_attempts,
                            all(row.status == "completed" for row in observations))


def git_preregistration_is_committed(path: str, content_sha256: str) -> bool:
    """Local-only committed-file checker; it neither fetches nor contacts a remote."""
    candidate = Path(path).resolve()
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=candidate.parent,
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode().strip()
        relative = str(candidate.relative_to(Path(root)))
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=root,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root, check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    return hashlib.sha256(committed).hexdigest() == content_sha256


def _validate_gates(
    protocol: FrozenProtocol, manifest: DatasetManifest, rows: Iterable[DatasetRow],
    frozen_preflight: PreflightResult, cache: CacheStore, approval: CollectionApproval,
    committed_checker: CommittedChecker,
) -> tuple[tuple[RequestCell, Item, object], ...]:
    approval.validate()
    if not approval.confirmed:
        raise ValueError("explicit collection confirmation is required")
    if (not isinstance(frozen_preflight, PreflightResult) or frozen_preflight.stage not in {"optimization", "scoreboard"}
        or frozen_preflight.blockers or (frozen_preflight.stage == "scoreboard" and not frozen_preflight.ready)):
        raise ValueError("only an exact unblocked optimization or ready scoreboard preflight can be collected")
    protocol.validate()
    source_rows = tuple(rows)
    # Recompute the exact core wire plans before anything can construct a model.
    expected_cells = core_plan_fingerprints(protocol, manifest, source_rows, stage=frozen_preflight.stage)
    if frozen_preflight.cells != expected_cells:
        raise ValueError("frozen preflight cells do not match regenerated core plans")
    regenerated = preflight(protocol, manifest=manifest, rows=source_rows, stage=frozen_preflight.stage)
    if frozen_preflight.checksum != regenerated.checksum:
        raise ValueError("frozen preflight checksum does not match regenerated plans")
    if approval.protocol_hash != protocol.identity or approval.preflight_checksum != frozen_preflight.checksum:
        raise ValueError("collection approval does not bind the protocol and preflight")
    preregistration = Path(approval.preregistration_path)
    try:
        actual_preregistration_hash = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("approved preregistration file is unavailable") from error
    content = preregistration.read_bytes()
    if (actual_preregistration_hash != approval.preregistration_sha256
        or protocol.identity.encode() not in content or frozen_preflight.checksum.encode() not in content
        or not committed_checker(str(preregistration), approval.preregistration_sha256)):
        raise ValueError("approved preregistration is not an exact committed file")
    expected_approved = [{"request_id": cell.fingerprint, "task_fingerprint": cell.task_fingerprint,
                          "state_fingerprint": cell.wire_fingerprint, "dataset_revision": cell.dataset_revision,
                          "model_identity": cell.model} for cell in sorted({cell.fingerprint: cell for cell in expected_cells}.values(), key=lambda cell: cell.fingerprint)]
    config = cache.frozen_configuration
    stored_approved = config.get("approved_requests")
    stored_base = ([{key: value for key, value in item.items() if key != "provider_model_identity"} for item in stored_approved]
                   if isinstance(stored_approved, list) and all(isinstance(item, Mapping) for item in stored_approved) else None)
    if (cache.task.fingerprint != protocol.task.fingerprint or config.get("preflight_fingerprint") != frozen_preflight.checksum
        or stored_base != expected_approved or cache.ceiling != approval.attempt_ceiling
        or cache.attempts_used > approval.attempt_ceiling):
        raise ValueError("cache ceiling does not match explicit collection approval")
    return _execution_plans(protocol, manifest, source_rows, frozen_preflight.cells)


def _execution_plans(
    protocol: FrozenProtocol, manifest: DatasetManifest, rows: Sequence[DatasetRow], cells: Sequence[RequestCell],
) -> tuple[tuple[RequestCell, Item, object], ...]:
    phases = {cell.phase for cell in cells}
    if phases - {"optimization", "pilot", "scoreboard"} or len(phases) > 1:
        raise ValueError("execution cells must have one supported frozen phase")
    development_phase = phases <= {"optimization", "pilot"}
    stage = "optimization" if development_phase else "scoreboard"
    candidates, development, scoreboard = rehydrate_manifest(manifest, rows, protocol, stage=stage)
    targets = {row.item.id: row.item for row in (development if development_phase else scoreboard)}
    artifact = protocol.selector_search_artifact
    plans = []
    for cell in cells:
        target = targets.get(cell.target_id)
        if target is None:
            raise ValueError("execution cell target is not in the validated manifest")
        selector = Selector(cell.selector)
        if selector is Selector.ZERO:
            policy, size = RandomBalanced(0), 0
        elif selector is Selector.RANDOM:
            policy, size = RandomBalanced(cell.draw_seed or 0), cell.per_label
        elif selector is Selector.PROTOTYPE:
            policy, size = PrototypeBalanced(), cell.per_label
        elif selector is Selector.RETRIEVAL:
            policy, size = PerLabelLexicalRetrieval(), cell.per_label
        else:
            if artifact is None:
                raise ValueError("selected-global cell has no frozen artifact")
            policy, size = _FrozenGlobal(artifact.ids_by_size[cell.per_label], artifact.checksum), cell.per_label
        plan = build_context_plan(protocol.task, target, candidates, policy, budget=ContextBudget(per_label=size),
                                  display_order=protocol.display_rule, order_seed=0,
                                  presentation_label_order=protocol.task.labels)
        if (plan.token_accounting.serialized_request_fingerprint != cell.wire_fingerprint
            or _physical_request_fingerprint(cell.model, manifest.revision, plan.token_accounting.serialized_request_fingerprint) != cell.fingerprint
            or plan.example_ids != cell.example_ids or cell.task_fingerprint != protocol.task.fingerprint
            or cell.dataset_revision != manifest.revision or cell.display_order != protocol.display_rule):
            raise ValueError("regenerated request plan does not match approved cell provenance")
        plans.append((cell, target, plan))
    return tuple(plans)


def _logical_cell(cell: RequestCell) -> LogicalCell:
    return LogicalCell(cell.id, _condition(cell), cell.draw_seed or 0, cell.display_order, cell.target_id)


def _condition(cell: RequestCell) -> str:
    """Keep phase/model/policy/size dimensions intact for downstream grouping."""
    return f"{cell.phase}:{cell.model}:{cell.selector}:{cell.per_label}"


def _result_payload(result: object) -> Mapping[str, object]:
    if not isinstance(result, DecisionResult):
        return {"answer": {}}
    payload: dict[str, object] = {"choice": result.label}
    if result.probabilities is not None: payload["probabilities"] = result.probabilities
    if result.usage is not None: payload["usage"] = result.usage
    if result.latency_ms is not None: payload["latency_ms"] = result.latency_ms
    if result.model is not None: payload["model"] = result.model
    if result.confidence is not None: payload["confidence"] = result.confidence
    return payload


def _observation(cell: RequestCell, manifest: DatasetManifest, snapshot: object, cache_hit: bool) -> Observation:
    # CacheSnapshot is deliberately structural here to keep export read-only.
    status = snapshot.physical_status.get(cell.fingerprint)  # type: ignore[attr-defined]
    attempts = snapshot.physical_attempts.get(cell.fingerprint, 0)  # type: ignore[attr-defined]
    payload = snapshot.physical_payloads.get(cell.fingerprint, {})  # type: ignore[attr-defined]
    records = manifest.development if cell.phase in {"optimization", "pilot"} else manifest.scoreboard
    record = next(record for record in records if record.id == cell.target_id)
    if status == "success":
        observation_status, prediction = "completed", payload.get("choice")
    elif status == "failed":
        observation_status = "malformed" if payload.get("category") == "malformed-response" else "failed"
        prediction = None
    else:
        observation_status, prediction = "missing", None
    probabilities = payload.get("probabilities") if observation_status == "completed" else None
    usage = payload.get("usage") if observation_status == "completed" else None
    latency = payload.get("latency_ms") if observation_status == "completed" else None
    confidence = payload.get("confidence") if observation_status == "completed" else None
    return Observation(cell.id, cell.target_id, _condition(cell), cell.draw_seed or 0, cell.display_order,
                       record.label, prediction, observation_status, probabilities=probabilities,
                       model_id=cell.model, usage=usage, latency_ms=latency, attempt_count=attempts,
                       cache_hit=cache_hit, physical_request_id=cell.fingerprint,
                       physical_request_provenance={"model": cell.model, "dataset_revision": cell.dataset_revision,
                                                    "task_fingerprint": cell.task_fingerprint,
                                                    "wire_fingerprint": cell.wire_fingerprint},
                       confidence=confidence)
