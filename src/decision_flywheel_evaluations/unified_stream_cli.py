"""Offline streaming fixtures and text-free reports; this command has no live mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .serialization import _write


def _run_fixture(document, run_dir, **options):
    from .unified_env import install_network_guard

    install_network_guard()
    from .unified_stream import run_fake_fixture

    return run_fake_fixture(document, run_dir, **options)


def _report_payload(document, **options):
    from .unified_stream_reporting import report_stream

    return report_stream(document, **options)


def _render_report(payload):
    from .unified_stream_reporting import render_stream_report

    return render_stream_report(payload)


def _output_guard(args):
    if Path(args.output).exists() and not args.overwrite:
        raise ValueError("output exists; explicit --overwrite is required")


def _run(args):
    _output_guard(args)
    if (Path(args.run_dir) / "stream-results.json").exists() and not args.overwrite:
        raise ValueError("run evidence exists; explicit --overwrite is required")
    if args.seed < 0 or args.max_new_requests < 0 or not 0 <= args.review_probability <= 1:
        raise ValueError("seed, review probability and request cap must be valid")
    document = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("synthetic") is not True:
        raise ValueError("only an explicitly synthetic fixture is supported")
    payload = _run_fixture(document, Path(args.run_dir), seed=args.seed,
                           review_probability=args.review_probability,
                           max_new_requests=args.max_new_requests)
    _write(args.output, payload)
    print("wrote synthetic offline stream outcomes; no live results")
    return 0


def _report(args):
    _output_guard(args)
    document = json.loads(Path(args.input).read_text(encoding="utf-8"))
    payload = _report_payload(document, window_size=args.window_size,
                              seed=args.seed, resamples=args.resamples)
    if args.format == "markdown":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_render_report(payload).rstrip() + "\n", encoding="utf-8")
    else:
        _write(args.output, payload)
    print("wrote a text-free report from local stream outcomes")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="exercise an explicit synthetic fixture without network")
    report = commands.add_parser("report", help="compute curves and checkpoints from local metadata")
    for command in (run, report):
        command.add_argument("--output", required=True)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--seed", type=int, default=0)
    run.add_argument("--fixture", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--review-probability", type=float, default=0.3)
    run.add_argument("--max-new-requests", type=int, required=True)
    report.add_argument("--input", required=True)
    report.add_argument("--window-size", type=int, default=100)
    report.add_argument("--resamples", type=int, default=1000)
    report.add_argument("--format", choices=("json", "markdown"), default="json")
    run.set_defaults(handler=_run)
    report.set_defaults(handler=_report)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        argument_parser.error(f"stream command failed ({type(error).__name__}); check local inputs")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
