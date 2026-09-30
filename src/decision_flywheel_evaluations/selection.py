"""Native development selection replayed solely from complete approved cache evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import RandomBalanced
from decision_flywheel.optimizer import (OptimizationResult, TrialSpec,
                                         _checkpoint_key, search_context_policies)

from .cache import CacheStore
from .datasets import DatasetRow
from .manifests import DatasetManifest
from .preflight import (PreflightResult, RequestCell, core_plan_fingerprints,
                        preflight, rehydrate_manifest)
from .protocol import FrozenProtocol, SelectedGlobalArtifact, _optimizer_objective


@dataclass(frozen=True)
class SelectionResult:
    artifact: SelectedGlobalArtifact
    optimization: OptimizationResult
    ledger_attempts: int
    replayed_decisions: int


class _NoCallsModel:
    async def decide(self, *_args):
        raise AssertionError("complete cache evidence must replay; optimizer model call attempted")


async def optimize_from_cache(protocol: FrozenProtocol, manifest: DatasetManifest,
                              rows: Iterable[DatasetRow], frozen_preflight: PreflightResult,
                              cache: CacheStore, model_identity: str,
                              artifact_reference: str) -> SelectionResult:
    """Create a selected-global artifact only from all complete development cells."""
    if not isinstance(artifact_reference, str) or not artifact_reference:
        raise ValueError("artifact reference is required")
    protocol.validate()
    source = tuple(rows)
    if (not isinstance(frozen_preflight, PreflightResult)
            or frozen_preflight.stage != "optimization"
            or frozen_preflight.blockers):
        raise ValueError("exact unblocked optimization preflight is required")
    expected_cells = core_plan_fingerprints(protocol, manifest, source, stage="optimization")
    if frozen_preflight.cells != expected_cells:
        raise ValueError("frozen preflight cells do not match regenerated core plans")
    regenerated = preflight(protocol, manifest=manifest, rows=source, stage="optimization")
    if frozen_preflight.checksum != regenerated.checksum:
        raise ValueError("frozen preflight checksum does not match regenerated plans")
    _validate_cache_configuration(cache, protocol, frozen_preflight, expected_cells)
    models = {item.semantic_identity for item in protocol.models}
    if model_identity not in models:
        raise ValueError("declared model identity is not frozen in protocol")
    candidates, development, scoreboard = rehydrate_manifest(manifest, source, protocol)
    cells = [cell for cell in frozen_preflight.cells if cell.model == model_identity]
    expected_trials = tuple(TrialSpec(RandomBalanced(seed), size)
                            for size in protocol.optimization.per_label_sizes
                            for seed in protocol.optimization.draw_seeds)
    if len(cells) != len(expected_trials) * len(development):
        raise ValueError("optimization matrix does not cover every declared development trial")
    snapshot = cache.snapshot()
    checkpoint = {}
    by_target = {item.item.id: item for item in development}
    for cell in cells:
        payload = snapshot.physical_responses.get(cell.fingerprint)
        if (snapshot.physical_status.get(cell.fingerprint) != "success"
                or not isinstance(payload, Mapping)
                or payload.get("status") != "success"
                or payload.get("model") != cell.model):
            raise ValueError("incomplete or failed approved development evidence cannot select a winner")
        label = payload.get("choice")
        try:
            label = protocol.task.validate_label(label)
        except (TypeError, ValueError) as error:
            raise ValueError("cached development response has invalid label") from error
        target = by_target.get(cell.target_id)
        if target is None:
            raise ValueError("optimization cell is outside development split")
        policy = RandomBalanced(cell.draw_seed or 0)
        plan = build_context_plan(protocol.task, target.item, candidates, policy,
                                  budget=ContextBudget(per_label=cell.per_label),
                                  display_order=protocol.display_rule, order_seed=0,
                                  presentation_label_order=protocol.task.labels)
        if plan.token_accounting.serialized_request_fingerprint != cell.wire_fingerprint:
            raise ValueError("cached request does not match regenerated development plan")
        checkpoint[_checkpoint_key(model_identity, protocol.task, plan)] = {"label": label}
    optimization = await search_context_policies(
        protocol.task, candidates, development, _NoCallsModel(), expected_trials,
        max_model_calls=protocol.optimization.max_model_calls,
        objective=_optimizer_objective(protocol.metric), checkpoint=checkpoint,
        model_fingerprint=model_identity,
        protected_ids=tuple(record.id for record in manifest.scoreboard),
        protected_text_hashes=tuple(record.normalized_text_sha256 for record in manifest.scoreboard),
        display_order=protocol.display_rule, order_seed=0, presentation_label_order=protocol.task.labels,
    )
    if optimization.winner is None or optimization.model_calls_attempted != 0:
        raise ValueError("complete cached evidence did not produce a replay-only winner")
    artifact = SelectedGlobalArtifact.from_optimization(artifact_reference, protocol.task,
                                                         candidates, manifest.revision, optimization)
    artifact.validate_for(protocol, candidates, development, manifest.revision)
    return SelectionResult(
        artifact,
        optimization,
        cache.attempts_used,
        sum(len(trial.decisions) for trial in optimization.trials),
    )


def _validate_cache_configuration(
    cache: CacheStore,
    protocol: FrozenProtocol,
    frozen_preflight: PreflightResult,
    expected_cells: tuple[RequestCell, ...],
) -> None:
    """Require this ledger to be bound to exactly this complete dev matrix."""
    expected_approved = [
        {
            "request_id": cell.fingerprint,
            "task_fingerprint": cell.task_fingerprint,
            "state_fingerprint": cell.wire_fingerprint,
            "dataset_revision": cell.dataset_revision,
            "model_identity": cell.model,
        }
        for cell in sorted(
            {cell.fingerprint: cell for cell in expected_cells}.values(),
            key=lambda cell: cell.fingerprint,
        )
    ]
    config = cache.frozen_configuration
    stored = config.get("approved_requests")
    stored_base = (
        [
            {key: value for key, value in entry.items()
             if key != "provider_model_identity"}
            for entry in stored
        ]
        if isinstance(stored, list) and all(isinstance(entry, Mapping) for entry in stored)
        else None
    )
    if (cache.task.fingerprint != protocol.task.fingerprint
            or config.get("preflight_fingerprint") != frozen_preflight.checksum
            or stored_base != expected_approved
            or cache.attempts_used > cache.ceiling
            or cache.ceiling > protocol.optimization.max_model_calls):
        raise ValueError("cache does not match exact regenerated optimization approval")
