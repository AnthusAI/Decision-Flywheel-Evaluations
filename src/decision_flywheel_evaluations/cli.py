"""Safe native command-line workflow for an already frozen evaluation study.

The commands deliberately do not prepare datasets or optimize a policy.  They
rehydrate local rows, enumerate the core plans, collect only after explicit
approval, and report sanitized observations offline.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from .bootstrap import paired_macro_f1_bootstrap
from .cached_rows import load_cached_manifest_rows
from .cache import ApprovedRequest, CacheStore
from .collector import (CollectionApproval, CollectionOptions, CollectionResult,
                        collect, git_preregistration_is_committed)
from .datasets import DatasetRow
from .manifests import DatasetManifest, read_manifest
from .preflight import PreflightResult, preflight
from .preflight import _physical_request_fingerprint, manifest_sha256
from .protocol import FrozenProtocol, ModelIdentity, transport_config_fingerprint
from .reporting import holm_correction, report_study
from .runtime_identity import validate_jev_runtime
from .selection import optimize_from_cache
from .serialization import (jsonable, read_observations, read_preflight,
                            read_protocol, read_rows_fixture, write_observations,
                            write_preflight, write_protocol, write_selected_global_artifact)


EngineFactory = Callable[[str], object]
CommittedChecker = Callable[[str, str], bool]


async def run_collection(
    protocol: FrozenProtocol, manifest: DatasetManifest, rows: Iterable[DatasetRow], plan: PreflightResult,
    ledger_path: str, preregistration_path: str, attempt_ceiling: int, max_new_attempts: int,
    confirmed: bool, *, provider_model: str, engine_factory: EngineFactory | None = None,
    base_url: str | None = None, timeout_seconds: float | None = None,
    adapter_revision: str | None = None, package_revision: str | None = None,
    retry_failed: bool = False, recover_uncertain: bool = False, max_retries_per_request: int = 0,
    committed_checker: CommittedChecker | None = None,
) -> CollectionResult:
    """Run a bounded JEV-only collection after all non-provider gates pass.

    The injected factory is a test seam.  The live adapter factory is created
    only inside :func:`collect`, after provenance, commit, and budget gates.
    """
    if confirmed is not True:
        raise ValueError("--confirm is required before loading rows or constructing a provider")
    if not isinstance(attempt_ceiling, int) or isinstance(attempt_ceiling, bool) or attempt_ceiling < 0:
        raise ValueError("attempt ceiling must be a non-negative integer")
    if not isinstance(max_new_attempts, int) or isinstance(max_new_attempts, bool) or max_new_attempts < 0:
        raise ValueError("max-new must be a non-negative integer")
    options = CollectionOptions(max_new_attempts=max_new_attempts, retry_failed=retry_failed,
                                recover_uncertain=recover_uncertain,
                                max_retries_per_request=max_retries_per_request)
    options.validate()
    protocol.validate()
    if not isinstance(plan, PreflightResult) or plan.blockers or not plan.cells:
        raise ValueError("run requires an unblocked persisted preflight with actual request cells")
    semantic_identity = _frozen_jev_identity(protocol, provider_model)
    if any(cell.model != semantic_identity for cell in plan.cells):
        raise ValueError("native CLI live collection currently supports one exact JEV model identity only")
    if engine_factory is None:
        _validate_live_transport(protocol, semantic_identity, base_url, timeout_seconds, adapter_revision, package_revision)
    physical = {cell.fingerprint for cell in plan.cells}
    if attempt_ceiling > len(physical) * (1 + max_retries_per_request):
        raise ValueError("attempt ceiling exceeds the frozen request plan and explicit retry bound")
    if plan.stage == "optimization" and attempt_ceiling > protocol.optimization.max_model_calls:
        raise ValueError("attempt ceiling exceeds the declared native optimization budget")
    if max_new_attempts > attempt_ceiling:
        raise ValueError("max-new cannot exceed the approved attempt ceiling")
    preregistration = Path(preregistration_path)
    try:
        preregistration_sha256 = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("approved preregistration file is unavailable") from error
    approved = tuple(ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                                     cell.dataset_revision, cell.model, provider_model)
                     for cell in sorted({cell.fingerprint: cell for cell in plan.cells}.values(), key=lambda item: item.fingerprint))
    cache = CacheStore(ledger_path, task=protocol.task, preflight_fingerprint=plan.checksum,
                       approved_requests=approved, ceiling=attempt_ceiling)
    try:
        if cache.attempts_used > attempt_ceiling or max_new_attempts > attempt_ceiling - cache.attempts_used:
            raise ValueError("max-new exceeds the remaining approved budget")
        approval = CollectionApproval(str(preregistration), preregistration_sha256, protocol.identity,
                                      plan.checksum, attempt_ceiling, True)
        factory = engine_factory or _live_jev_factory(provider_model, semantic_identity, base_url, timeout_seconds)
        return await collect(protocol, manifest, tuple(rows), plan, cache, approval, factory,
                             committed_checker=committed_checker,
                             options=options)
    finally:
        cache.close()


async def select_from_ledger(
    protocol: FrozenProtocol, manifest: DatasetManifest, rows: Iterable[DatasetRow], plan: PreflightResult,
    ledger_path: str, preregistration_path: str, attempt_ceiling: int, *, model_identity: str, provider_model: str,
    artifact_reference: str, committed_checker: CommittedChecker | None = None,
) -> FrozenProtocol:
    """Derive a selected-global artifact from complete cached development evidence only."""
    if plan.stage != "optimization":
        raise ValueError("select requires an optimization preflight")
    if not isinstance(attempt_ceiling, int) or isinstance(attempt_ceiling, bool) or attempt_ceiling < 0:
        raise ValueError("attempt ceiling must be a non-negative integer")
    preregistration = Path(preregistration_path)
    try:
        content = preregistration.read_bytes()
    except OSError as error:
        raise ValueError("optimization preregistration file is unavailable") from error
    digest = hashlib.sha256(content).hexdigest()
    checker = committed_checker or git_preregistration_is_committed
    if (protocol.identity.encode() not in content or plan.checksum.encode() not in content
            or not checker(str(preregistration), digest)):
        raise ValueError("optimization selection requires the exact committed preregistration")
    approved = tuple(ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                                     cell.dataset_revision, cell.model, provider_model)
                     for cell in sorted({cell.fingerprint: cell for cell in plan.cells}.values(), key=lambda item: item.fingerprint))
    cache = CacheStore(ledger_path, task=protocol.task, preflight_fingerprint=plan.checksum,
                       approved_requests=approved, ceiling=attempt_ceiling)
    try:
        result = await optimize_from_cache(protocol, manifest, tuple(rows), plan, cache, model_identity, artifact_reference)
    finally:
        cache.close()
    return replace(protocol, selector_search_artifact=result.artifact)


def report_document(protocol: FrozenProtocol, manifest: DatasetManifest, plan: PreflightResult,
                    observations_path: str | Path) -> dict[str, object]:
    """Regenerate descriptive metrics and paired effects from sanitized observations.

    An incomplete matrix is explicitly marked as incomplete, never presented as
    a study finding or passed to confirmatory paired inference.
    """
    _validate_report_inputs(protocol, manifest, plan)
    rows = read_observations(observations_path)
    _validate_observations_against_plan(rows, manifest, plan)
    expected = tuple(cell.id for cell in plan.cells)
    study = report_study(rows, protocol.task.labels, expected_request_ids=expected)
    complete = bool(expected) and all(row.status == "completed" for row in rows) and len(rows) == len(expected)
    paired = _paired_effects(protocol, manifest, rows, complete, plan.stage)
    inference, holm = _inference_metadata(protocol, paired)
    return {"schema": "decision-flywheel-evaluations/report/v1", "protocol_identity": protocol.identity,
            "preflight_checksum": plan.checksum, "complete": complete,
            "stage": plan.stage,
            "finding_status": ("complete descriptive scoreboard matrix" if complete and plan.stage == "scoreboard"
                               else "incomplete: not a study finding" if not complete
                               else "development optimization observations: not a study finding"),
            "study": jsonable(study), "paired_effects": paired,
            "inference": inference, "holm": holm}


def _inference_metadata(protocol: FrozenProtocol, paired: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """State only the correction frozen in the protocol; intervals remain descriptive."""
    interval_scope = "nominal per-contrast, not multiplicity adjusted"
    if protocol.primary_family_correction == "none; descriptive paired 95% intervals":
        return (
            {"family_correction": protocol.primary_family_correction, "formal_test": "not performed",
             "adjusted_significance_available": False, "interval_scope": interval_scope},
            {"available": False, "adjusted_p_values": {},
             "reason": "familywise correction was not requested for descriptive paired intervals"},
        )
    return (
        {"family_correction": protocol.primary_family_correction, "formal_test": "not available",
         "adjusted_significance_available": False, "interval_scope": interval_scope},
        jsonable(holm_correction({name: None for name in paired})),
    )


def _paired_effects(protocol: FrozenProtocol, manifest: DatasetManifest, rows: tuple[object, ...], complete: bool,
                    stage: str) -> dict[str, object]:
    if not complete or stage != "scoreboard":
        return {"selection": {"available": False, "reason": "missing or failed matrix cells"},
                "size": {"available": False, "reason": "missing or failed matrix cells"}}
    model = protocol.models[0].semantic_identity
    pairs = {
        "selection": (f"scoreboard:{model}:random:16", f"scoreboard:{model}:retrieval:16"),
        "size": (f"scoreboard:{model}:random:1", f"scoreboard:{model}:random:64"),
    }
    output: dict[str, object] = {}
    for name, (baseline, treatment) in pairs.items():
        try:
            interval = paired_macro_f1_bootstrap(rows, protocol.task.labels, baseline=baseline, treatment=treatment,
                                                 metric=protocol.metric,
                                                 expected_target_ids=tuple(record.id for record in manifest.scoreboard))
        except ValueError as error:
            # Bootstrap failure text is internal validation language, not provider output.
            output[name] = {"available": False, "reason": str(error)}
        else:
            output[name] = {"available": True, "effect": interval.effect, "lower": interval.lower, "upper": interval.upper,
                            "confidence_level": 0.95,
                            "interval_scope": "nominal per-contrast, not multiplicity adjusted"}
    return output


def _validate_report_inputs(protocol: FrozenProtocol, manifest: DatasetManifest, plan: PreflightResult) -> None:
    protocol.validate()
    manifest.validate()
    if manifest_sha256(manifest) != protocol.dataset_manifest_sha256:
        raise ValueError("report manifest does not match the frozen protocol")
    if not isinstance(plan, PreflightResult) or plan.stage not in {"optimization", "scoreboard"}:
        raise ValueError("report needs a persisted optimization or scoreboard preflight")
    checksum_payload = {"protocol": protocol.frozen_payload(), "stage": plan.stage,
                        "cells": [asdict(cell) for cell in plan.cells], "excluded": [asdict(item) for item in plan.excluded],
                        "physical": sorted({cell.fingerprint for cell in plan.cells})}
    checksum = hashlib.sha256(json.dumps(checksum_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if checksum != plan.checksum:
        raise ValueError("report preflight checksum does not bind this frozen protocol and cells")


def _validate_observations_against_plan(rows: tuple[object, ...], manifest: DatasetManifest,
                                        plan: PreflightResult) -> None:
    cells = {cell.id: cell for cell in plan.cells}
    if len(cells) != len(plan.cells) or len(rows) != len(cells):
        raise ValueError("report observations do not exactly cover the frozen logical matrix")
    records = {record.id: record for record in (manifest.development if plan.stage == "optimization" else manifest.scoreboard)}
    seen: set[str] = set()
    for row in rows:
        cell = cells.get(row.request_id)
        if cell is None or row.request_id in seen:
            raise ValueError("report observation is outside the frozen logical matrix")
        seen.add(row.request_id)
        if (row.target_id != cell.target_id or row.condition != f"{cell.phase}:{cell.model}:{cell.selector}:{cell.per_label}"
                or row.draw != (cell.draw_seed or 0) or row.order != cell.display_order or row.model_id != cell.model
                or row.physical_id != cell.fingerprint):
            raise ValueError("report observation does not match frozen logical or model provenance")
        record = records.get(row.target_id)
        if record is None or row.true_label != record.label:
            raise ValueError("report observation ground truth does not match the protected manifest")
        expected_provenance = {"model": cell.model, "dataset_revision": cell.dataset_revision,
                               "task_fingerprint": cell.task_fingerprint, "wire_fingerprint": cell.wire_fingerprint}
        if dict(row.physical_request_provenance or {}) != expected_provenance:
            raise ValueError("report observation physical provenance does not match its frozen request")
        if _physical_request_fingerprint(cell.model, cell.dataset_revision, cell.wire_fingerprint) != cell.fingerprint:
            raise ValueError("report preflight physical request fingerprint is invalid")


def _frozen_jev_identity(protocol: FrozenProtocol, provider_model: str) -> str:
    models = [model for model in protocol.models if model.route_identity == f"jev:{provider_model}"]
    if len(models) != 1:
        raise ValueError("provider model must match exactly one frozen JEV route identity")
    return models[0].semantic_identity


def _validate_live_transport(protocol: FrozenProtocol, semantic_identity: str, base_url: str | None,
                             timeout_seconds: float | None, adapter_revision: str | None,
                             package_revision: str | None) -> ModelIdentity:
    model = next(model for model in protocol.models if model.semantic_identity == semantic_identity)
    if not model.transport_fingerprint:
        raise ValueError("live JEV collection requires a non-empty frozen transport fingerprint")
    if base_url is None or timeout_seconds is None or adapter_revision is None or package_revision is None:
        raise ValueError("live JEV collection requires explicit base-url, timeout-seconds, adapter-revision, and package-revision")
    actual = transport_config_fingerprint(base_url=base_url, timeout_seconds=timeout_seconds, retries=0,
                                          adapter_revision=adapter_revision, package_revision=package_revision)
    if actual != model.transport_fingerprint:
        raise ValueError("configured JEV transport does not match the frozen model transport fingerprint")
    validate_jev_runtime(adapter_revision, package_revision)
    return model


def _live_jev_factory(provider_model: str, semantic_identity: str, base_url: str | None,
                      timeout_seconds: float | None) -> EngineFactory:
    def factory(identity: str) -> object:
        # Imports and dotenv access happen only after collect's full gate set.
        from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration
        configuration = JevConfiguration(model=provider_model, base_url=base_url,
                                         timeout_seconds=timeout_seconds, max_retries=0)
        if configuration.model_identity != semantic_identity.split("@transport-", 1)[0] or identity != semantic_identity:
            raise ValueError("configured JEV provider identity does not match the frozen approval")
        return JevAdapter.from_environment(configuration=configuration)
    return factory


def _command_preflight(args: argparse.Namespace) -> int:
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    rows = _load_source_rows(args, manifest, stage=args.stage)
    result = preflight(protocol, manifest=manifest, rows=rows, stage=args.stage)
    write_preflight(args.output, result)
    print(f"wrote {len(result.cells)} logical cells / {result.new_request_count} new physical requests")
    return 0


def _command_run(args: argparse.Namespace) -> int:
    # Check explicit human authorization and numerical caps before reading text.
    if args.confirm is not True:
        raise ValueError("--confirm is required; no rows or provider were loaded")
    if args.max_new < 0 or args.attempt_ceiling < 0 or args.max_new > args.attempt_ceiling:
        raise ValueError("max-new must be non-negative and no greater than attempt-ceiling")
    if (args.base_url is None or args.timeout_seconds is None or args.adapter_revision is None
            or args.package_revision is None):
        raise ValueError("live JEV run requires --base-url, --timeout-seconds, --adapter-revision, and --package-revision before rows are loaded")
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    plan = read_preflight(args.preflight)
    rows = _load_source_rows(args, manifest, stage=plan.stage)
    result = asyncio.run(run_collection(protocol, manifest, rows, plan, args.ledger, args.preregistration,
                                        args.attempt_ceiling, args.max_new, True,
                                        provider_model=args.provider_model, base_url=args.base_url,
                                        timeout_seconds=args.timeout_seconds, adapter_revision=args.adapter_revision,
                                        package_revision=args.package_revision, retry_failed=args.retry_failed,
                                        recover_uncertain=args.recover_uncertain,
                                        max_retries_per_request=args.max_retries_per_request))
    write_observations(args.output, result.observations)
    print(f"wrote {len(result.observations)} observations; {result.new_attempts} new attempts")
    return 0


def _command_select(args: argparse.Namespace) -> int:
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    plan = read_preflight(args.preflight)
    rows = _load_source_rows(args, manifest, stage=plan.stage)
    derived = asyncio.run(select_from_ledger(protocol, manifest, rows, plan, args.ledger, args.preregistration,
                                              args.attempt_ceiling,
                                              model_identity=args.model_identity, provider_model=args.provider_model,
                                              artifact_reference=args.artifact_reference))
    write_protocol(args.protocol_output, derived)
    write_selected_global_artifact(args.artifact_output, derived.selector_search_artifact,
                                   source_protocol_identity=protocol.identity,
                                   source_preflight_checksum=plan.checksum,
                                   derived_protocol_identity=derived.identity)
    Path(args.registration_output).write_text(json.dumps({
        "schema": "decision-flywheel-evaluations/selection-registration/v1",
        "optimization_protocol_identity": protocol.identity,
        "optimization_preflight_checksum": plan.checksum,
        "derived_protocol_identity": derived.identity,
        "selected_global_artifact_checksum": derived.selector_search_artifact.checksum,
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote selected-global protocol {args.protocol_output}; create a separate committed scoreboard preregistration")
    return 0


def _command_report(args: argparse.Namespace) -> int:
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    plan = read_preflight(args.preflight)
    document = report_document(protocol, manifest, plan, args.observations)
    Path(args.output).write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({document['finding_status']})")
    return 0


def _load_source_rows(
    args: argparse.Namespace,
    manifest: DatasetManifest,
    *,
    stage: str,
) -> tuple[DatasetRow, ...]:
    """Load either caller-owned fixtures or the explicit local pinned Arrow cache."""
    if stage not in {"optimization", "scoreboard"}:
        raise ValueError("stage must be optimization or scoreboard")
    if args.rows is not None:
        return read_rows_fixture(args.rows)
    if args.dataset_cache is not None:
        roles = ("candidate", "development") if stage == "optimization" else (
            "candidate", "development", "scoreboard"
        )
        return load_cached_manifest_rows(manifest, cache_root=args.dataset_cache, roles=roles)
    raise ValueError("provide exactly one of --rows or --dataset-cache")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="decision-flywheel-evaluations", description="Native offline-first evaluation workflow")
    commands = result.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="enumerate text-free core request plans from explicit local rows")
    for option in ("protocol", "manifest", "output"):
        preflight_parser.add_argument(f"--{option}", required=True)
    _source_arguments(preflight_parser)
    preflight_parser.add_argument("--stage", choices=("optimization", "scoreboard"), default="scoreboard")
    preflight_parser.set_defaults(handler=_command_preflight)
    run_parser = commands.add_parser("run", help="collect an approved JEV-only frozen preflight")
    for option in ("protocol", "manifest", "preflight", "ledger", "preregistration", "output", "provider-model"):
        run_parser.add_argument(f"--{option}", required=True)
    _source_arguments(run_parser)
    run_parser.add_argument("--attempt-ceiling", type=int, required=True)
    run_parser.add_argument("--max-new", type=int, required=True)
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--timeout-seconds", type=float)
    run_parser.add_argument("--adapter-revision")
    run_parser.add_argument("--package-revision")
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument("--recover-uncertain", action="store_true")
    run_parser.add_argument("--max-retries-per-request", type=int, default=0)
    run_parser.add_argument("--confirm", action="store_true")
    run_parser.set_defaults(handler=_command_run)
    select_parser = commands.add_parser("select", help="derive a selected-global artifact from complete cached development evidence")
    for option in ("protocol", "manifest", "preflight", "ledger", "preregistration", "model-identity", "provider-model",
                   "artifact-reference", "protocol-output", "artifact-output", "registration-output"):
        select_parser.add_argument(f"--{option}", required=True)
    _source_arguments(select_parser)
    select_parser.add_argument("--attempt-ceiling", type=int, required=True)
    select_parser.set_defaults(handler=_command_select)
    report_parser = commands.add_parser("report", help="regenerate offline metrics from sanitized observations")
    for option in ("protocol", "manifest", "preflight", "observations", "output"):
        report_parser.add_argument(f"--{option}", required=True)
    report_parser.set_defaults(handler=_command_report)
    return result


def _source_arguments(command: argparse.ArgumentParser) -> None:
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--rows", help="synthetic or caller-owned .fixture.json input")
    source.add_argument("--dataset-cache", help="explicit local .data/huggingface cache root; never downloaded")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as error:
        parser().error(str(error))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
