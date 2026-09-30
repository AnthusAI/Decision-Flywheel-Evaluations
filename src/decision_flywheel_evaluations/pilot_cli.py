"""Explicit offline planning, opt-in collection, and reporting for a JEV pilot."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .cache import ApprovedRequest, CacheStore
from .cached_rows import load_cached_manifest_rows
from .cli import _frozen_jev_identity, _live_jev_factory, _validate_live_transport
from .collector import CollectionApproval, CollectionOptions, git_preregistration_is_committed
from .manifests import read_manifest
from .pilot import plan_jev_pilot
from .pilot_collection import collect_pilot, validate_pilot_preregistration
from .pilot_io import read_pilot, write_pilot
from .pilot_reporting import summarize_pilot_compatibility
from .serialization import (read_observations, read_preflight, read_protocol,
                            read_rows_fixture, write_observations)


async def run_pilot_collection(
    protocol, manifest, rows, source_preflight, plan, ledger_path, preregistration_path,
    attempt_ceiling, max_new_attempts, confirmed, *, provider_model, base_url=None,
    timeout_seconds=None, adapter_revision=None, package_revision=None,
    engine_factory=None, committed_checker=None,
):
    """A separate tiny ledger cannot authorize or resume the full study ledger."""
    if confirmed is not True:
        raise ValueError("--confirm is required before a pilot can load a provider")
    plan.validate()
    for value in (attempt_ceiling, max_new_attempts):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("pilot attempt bounds must be explicit non-negative integers")
    if attempt_ceiling != plan.physical_count or max_new_attempts > attempt_ceiling:
        raise ValueError("pilot ceiling must equal its frozen unique request count")
    identity = _frozen_jev_identity(protocol, provider_model)
    if any(cell.model != identity for cell in plan.cells):
        raise ValueError("pilot provider model must match every frozen request")
    if engine_factory is None:
        _validate_live_transport(protocol, identity, base_url, timeout_seconds,
                                 adapter_revision, package_revision)
    registration = Path(preregistration_path)
    try:
        digest = hashlib.sha256(registration.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("pilot preregistration is unavailable") from error
    approval = CollectionApproval(str(registration), digest, protocol.identity,
                                  plan.checksum, attempt_ceiling, True)
    validate_pilot_preregistration(
        protocol, source_preflight, plan, approval,
        committed_checker or git_preregistration_is_committed,
    )
    approved = tuple(
        ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                        cell.dataset_revision, cell.model, provider_model)
        for cell in sorted({cell.fingerprint: cell for cell in plan.cells}.values(),
                           key=lambda item: item.fingerprint)
    )
    with CacheStore(ledger_path, task=protocol.task, preflight_fingerprint=plan.checksum,
                    approved_requests=approved, ceiling=attempt_ceiling) as cache:
        factory = engine_factory or _live_jev_factory(provider_model, identity, base_url, timeout_seconds)
        return await collect_pilot(
            protocol, manifest, tuple(rows), source_preflight, plan, cache, approval, factory,
            committed_checker=committed_checker,
            options=CollectionOptions(max_new_attempts=max_new_attempts),
        )


def _load_rows(args, manifest):
    if args.rows is not None:
        return read_rows_fixture(args.rows)
    return load_cached_manifest_rows(manifest, cache_root=args.dataset_cache,
                                    roles=("candidate", "development"))


def _preflight(args):
    output = Path(args.output)
    _check_output(args)
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    source = read_preflight(args.source_preflight)
    plan = plan_jev_pilot(protocol, manifest, _load_rows(args, manifest), source)
    write_pilot(output, plan)
    print(json.dumps({"status": "unapproved", "pilot_checksum": plan.checksum,
                      "logical_cases": len(plan.cells), "physical_requests": plan.physical_count,
                      "maximum_attempt_ceiling": plan.physical_count}, sort_keys=True))
    return 0


def _run(args):
    if args.confirm is not True:
        raise ValueError("--confirm is required; no rows or provider were loaded")
    _check_output(args)
    if args.max_new < 0 or args.attempt_ceiling < 0 or args.max_new > args.attempt_ceiling:
        raise ValueError("pilot maximum-new attempts must fit its explicit ceiling")
    protocol = read_protocol(args.protocol)
    manifest = read_manifest(args.manifest)
    source = read_preflight(args.source_preflight)
    plan = read_pilot(args.pilot)
    result = asyncio.run(run_pilot_collection(
        protocol, manifest, _load_rows(args, manifest), source, plan, args.ledger,
        args.preregistration, args.attempt_ceiling, args.max_new, True,
        provider_model=args.provider_model, base_url=args.base_url, timeout_seconds=args.timeout_seconds,
        adapter_revision=args.adapter_revision, package_revision=args.package_revision,
    ))
    write_observations(args.output, result.observations)
    print(json.dumps({"status": "pilot only; not a study finding", "new_attempts": result.new_attempts,
                      "cumulative_attempts": result.physical_attempts, "complete": result.complete}, sort_keys=True))
    return 0


def _report(args):
    _check_output(args)
    summary = summarize_pilot_compatibility(read_protocol(args.protocol), read_pilot(args.pilot),
                                            read_observations(args.observations))
    Path(args.output).write_text(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
                                 encoding="utf-8")
    print("wrote a development-only compatibility report; not a benchmark result")
    return 0


def _check_output(args):
    if Path(args.output).exists() and not args.overwrite:
        raise ValueError("pilot output exists; explicit --overwrite is required")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("preflight", help="freeze tiny development-only cases without calls")
    running = commands.add_parser("run", help="collect only an explicitly approved capability pilot")
    reporting = commands.add_parser("report", help="summarize local operational evidence without accuracy metrics")
    for command in (planning, running):
        for name in ("protocol", "manifest", "source-preflight", "output"):
            command.add_argument(f"--{name}", required=True)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--dataset-cache")
        source.add_argument("--rows", help="synthetic or caller-owned local fixture only")
    for command in (planning, running, reporting):
        command.add_argument("--overwrite", action="store_true")
    for name in ("pilot", "ledger", "preregistration", "provider-model", "base-url",
                 "adapter-revision", "package-revision"):
        running.add_argument(f"--{name}", required=True)
    running.add_argument("--timeout-seconds", type=float, required=True)
    running.add_argument("--attempt-ceiling", type=int, required=True)
    running.add_argument("--max-new", type=int, required=True)
    running.add_argument("--confirm", action="store_true")
    for name in ("protocol", "pilot", "observations", "output"):
        reporting.add_argument(f"--{name}", required=True)
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
        # Public invalid inputs may contain secrets: never echo exception contents.
        argument_parser.error(f"pilot command failed ({type(error).__name__}); check frozen inputs and approval")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
