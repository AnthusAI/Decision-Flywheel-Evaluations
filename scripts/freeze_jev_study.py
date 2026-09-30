#!/usr/bin/env python3
"""Author an unapproved, text-free initial JEV protocol from a committed manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_flywheel_evaluations.manifests import read_manifest
from decision_flywheel_evaluations.serialization import write_protocol
from decision_flywheel_evaluations.study_setup import PublicJevConfiguration, freeze_initial_jev_protocol


def _is_committed(path: Path, digest: str) -> bool:
    try:
        root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.parent, check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode().strip())
        relative = str(path.resolve().relative_to(root))
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root, check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    return hashlib.sha256(committed).hexdigest() == digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--package-revision", required=True)
    parser.add_argument("--max-per-label", type=int, required=True)
    parser.add_argument("--max-model-calls", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest_bytes = args.manifest.read_bytes()
        if not _is_committed(args.manifest, hashlib.sha256(manifest_bytes).hexdigest()):
            raise ValueError("manifest is not an exact committed file")
        manifest = read_manifest(args.manifest)
        configuration = PublicJevConfiguration(args.model_version, args.base_url, args.timeout_seconds,
                                               args.adapter_revision, args.package_revision, args.max_per_label,
                                               args.max_model_calls)
        protocol, transport = freeze_initial_jev_protocol(manifest, configuration)
        if args.output.exists() and not args.overwrite:
            raise ValueError("output exists; pass --overwrite explicitly")
        write_protocol(args.output, protocol)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "failed", "reason": type(error).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"status": "authored_unapproved", "protocol_identity": protocol.identity,
                      "task_fingerprint": protocol.task.fingerprint,
                      "logical_counts": {**manifest.counts, "development_selection_trials": 20},
                      "declared_max_model_calls": protocol.optimization.max_model_calls}, sort_keys=True))
    print(json.dumps({"status": "public_transport", **transport}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
