"""Ordering preflight, opt-in collection, and reporting after complete initial evidence."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .cached_rows import load_cached_manifest_rows
from .cache import ApprovedRequest, CacheStore
from .cli import _frozen_jev_identity, _live_jev_factory, _validate_live_transport
from .collector import CollectionApproval, CollectionOptions, git_preregistration_is_committed
from .manifests import read_manifest
from .ordering import (_validate_initial_matrix, _validate_initial_observations,
                       plan_ordering)
from .ordering_io import read_ordering, write_ordering
from .ordering_collection import (collect_ordering, validate_ordering_evidence,
                                  validate_ordering_preregistration)
from .ordering_reporting import report_ordering
from .protocol import OrderTreatment, validate_result_reference
from .serialization import (read_observations, read_preflight, read_protocol,
                            read_rows_fixture, write_observations, _write)


async def run_ordering_collection(
    protocol, manifest, rows, initial, initial_observations, plan, ledger_path,
    preregistration_path, attempt_ceiling, confirmed, *, provider_model,
    options: CollectionOptions, base_url=None, timeout_seconds=None,
    adapter_revision=None, package_revision=None, engine_factory=None,
    committed_checker=None,
):
    """An ordering approval and ledger never authorize initial or cross-model calls."""
    if confirmed is not True:
        raise ValueError("--confirm is required before ordering collection")
    options.validate()
    plan.validate()
    if (not isinstance(attempt_ceiling, int) or isinstance(attempt_ceiling, bool)
            or attempt_ceiling < 0 or options.max_new_attempts is None
            or options.max_new_attempts > attempt_ceiling
            or attempt_ceiling > plan.physical_count * (1 + options.max_retries_per_request)):
        raise ValueError("ordering attempt bounds must fit the frozen matrix and retry cap")
    if (plan.initial_protocol_identity != protocol.identity
            or plan.manifest_sha256 != protocol.dataset_manifest_sha256
            or plan.initial_preflight_checksum != initial.checksum):
        raise ValueError("ordering plan does not bind the initial frozen study")
    identity = _frozen_jev_identity(protocol, provider_model)
    if any(cell.request.model != identity for cell in plan.cells):
        raise ValueError("ordering provider model must match every frozen request")
    if engine_factory is None:
        _validate_live_transport(protocol, identity, base_url, timeout_seconds,
                                 adapter_revision, package_revision)
    registration = Path(preregistration_path)
    digest = hashlib.sha256(registration.read_bytes()).hexdigest()
    approval = CollectionApproval(str(registration), digest, protocol.identity,
                                  plan.checksum, attempt_ceiling, True)
    checker = committed_checker or git_preregistration_is_committed
    validate_ordering_preregistration(protocol, initial, plan, approval, checker)
    rows = tuple(rows)
    validate_ordering_evidence(protocol, manifest, rows, initial, initial_observations, plan)
    physical = {cell.request.fingerprint: cell.request for cell in plan.cells}
    approved = tuple(
        ApprovedRequest(request.fingerprint, request.task_fingerprint, request.wire_fingerprint,
                        request.dataset_revision, request.model, provider_model)
        for request in sorted(physical.values(), key=lambda item: item.fingerprint)
    )
    with CacheStore(ledger_path, task=protocol.task, preflight_fingerprint=plan.checksum,
                    approved_requests=approved, ceiling=attempt_ceiling) as cache:
        factory = engine_factory or _live_jev_factory(provider_model, identity, base_url, timeout_seconds)
        return await collect_ordering(protocol, manifest, rows, initial, initial_observations,
                                      plan, cache, approval, factory,
                                      committed_checker=checker, options=options)


def _check_output(args):
    if Path(args.output).exists() and not args.overwrite:
        raise ValueError("ordering output exists; explicit --overwrite is required")


def _preflight(args):
    _check_output(args)
    validate_result_reference(args.initial_result_reference)
    protocol, manifest, initial, observations = _read_initial(args)
    treatments = (OrderTreatment("canonical", 0), OrderTreatment("interleaved", 0),
                  OrderTreatment("reversed", 0),
                  *(OrderTreatment("shuffled", seed) for seed in args.shuffle_seed))
    plan = plan_ordering(protocol, manifest, _load_rows(args, manifest), initial, observations,
                         args.initial_result_reference, treatments)
    write_ordering(args.output, plan)
    print(json.dumps({"status": "unapproved", "ordering_checksum": plan.checksum,
                      "logical_requests": plan.logical_count,
                      "physical_requests": plan.physical_count,
                      "model_calls": 0}, sort_keys=True))
    return 0


def _read_initial(args):
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    initial = read_preflight(args.initial_preflight)
    observations = read_observations(args.initial_observations)
    if not initial.ready or initial.stage != "scoreboard":
        raise ValueError("completed initial scoreboard evidence is required")
    cells = _validate_initial_matrix(protocol, initial)
    _validate_initial_observations(protocol, manifest, cells, observations)
    return protocol, manifest, initial, observations


def _load_rows(args, manifest):
    return (read_rows_fixture(args.rows) if args.rows is not None else
            load_cached_manifest_rows(manifest, cache_root=args.dataset_cache))


def _run(args):
    if args.confirm is not True:
        raise ValueError("--confirm is required; no rows or provider were loaded")
    _check_output(args)
    options = CollectionOptions(max_new_attempts=args.max_new, retry_failed=args.retry_failed,
                                recover_uncertain=args.recover_uncertain,
                                max_retries_per_request=args.max_retries_per_request)
    options.validate()
    protocol, manifest, initial, initial_observations = _read_initial(args)
    result = asyncio.run(run_ordering_collection(
        protocol, manifest, _load_rows(args, manifest), initial, initial_observations,
        read_ordering(args.ordering), args.ledger, args.preregistration, args.attempt_ceiling, True,
        provider_model=args.provider_model, options=options, base_url=args.base_url,
        timeout_seconds=args.timeout_seconds, adapter_revision=args.adapter_revision,
        package_revision=args.package_revision,
    ))
    write_observations(args.output, result.observations)
    print(json.dumps({"new_attempts": result.new_attempts, "cumulative_attempts": result.physical_attempts,
                      "complete": result.complete}, sort_keys=True))
    return 0


def _report(args):
    _check_output(args)
    result = report_ordering(read_protocol(args.protocol), read_manifest(args.manifest),
                             read_ordering(args.ordering), read_observations(args.observations),
                             seed=args.seed, resamples=args.resamples)
    _write(args.output, result)
    print("wrote a descriptive ordering report from local observations")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("preflight", help="enumerate presentation-only follow-up requests; no calls")
    running = commands.add_parser("run", help="collect only a separately preregistered and approved ordering matrix")
    reporting = commands.add_parser("report", help="reproduce descriptive ordering metrics offline")
    for command in (planning, running, reporting):
        for name in ("protocol", "manifest", "output"):
            command.add_argument(f"--{name}", required=True)
        command.add_argument("--overwrite", action="store_true")
    for command in (planning, running):
        for name in ("initial-preflight", "initial-observations"):
            command.add_argument(f"--{name}", required=True)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--rows", help="synthetic or caller-owned local fixture")
        source.add_argument("--dataset-cache")
    planning.add_argument("--initial-result-reference", required=True)
    planning.add_argument("--shuffle-seed", type=int, action="append", required=True,
                          help="repeat for at least two distinct frozen permutation seeds")
    for name in ("ordering", "observations"):
        reporting.add_argument(f"--{name}", required=True)
    for name in ("ordering", "ledger", "preregistration", "provider-model", "base-url",
                 "adapter-revision", "package-revision"):
        running.add_argument(f"--{name}", required=True)
    running.add_argument("--timeout-seconds", type=float, required=True)
    running.add_argument("--attempt-ceiling", type=int, required=True)
    running.add_argument("--max-new", type=int, required=True)
    running.add_argument("--max-retries-per-request", type=int, default=0)
    running.add_argument("--retry-failed", action="store_true")
    running.add_argument("--recover-uncertain", action="store_true")
    running.add_argument("--confirm", action="store_true")
    reporting.add_argument("--seed", type=int, default=0)
    reporting.add_argument("--resamples", type=int, default=1000)
    planning.set_defaults(handler=_preflight)
    running.set_defaults(handler=_run)
    reporting.set_defaults(handler=_report)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError) as error:
        argument_parser.error(f"ordering command failed ({type(error).__name__}); check frozen inputs")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
