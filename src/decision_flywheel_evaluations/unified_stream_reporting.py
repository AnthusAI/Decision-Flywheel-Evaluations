"""Offline, text-free reports for the streaming SME study.

This module consumes ``StreamResult.to_dict()``-style mappings.  It deliberately
does not import the streaming runtime: a saved fake run can be rendered without
credentials, a model client, or the private review/policy material.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

from .unified_stats import ItemResult, accuracy, macro_f1, paired_interval


_SUCCESS = frozenset(("completed", "ok", "success"))
_ARMS = ("B", "L", "E", "X")
_CHECKPOINTS = ("stream-300", "end")
_COUNTERS = ("request_attempts", "request_failures", "refit_rounds", "steering_rounds", "list_rounds")
_USAGE = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")


@dataclass(frozen=True)
class _Outcome:
    item: ItemResult | None
    reviewed: int
    served_count: int
    has_confidence: bool = False


@dataclass(frozen=True)
class _Slice:
    rows: tuple[ItemResult, ...]
    complete: bool


def report_stream(arms: Mapping[str, Mapping[str, Any]], labels: Sequence[str] | None = None, *,
                  references: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
                  window_size: int = 100, resamples: int = 1_000, seed: int = 0) -> dict[str, Any]:
    """Summarize fixed streaming arms; never select an arm or infer a price.

    ``references`` may contain ``baseline`` (Jev+S) and ``ceiling`` (Jev+F),
    each mapping checkpoint names to held-out outcome rows.  Outcome rows carry
    labels only; no source text, SME response, or policy is accepted into output.
    """
    # The public entry point takes the persisted run envelope.  The two-argument
    # form remains convenient for small, pure module specs.
    envelope_synthetic = None
    if labels is None and isinstance(arms, Mapping) and isinstance(arms.get("arms"), Mapping):
        envelope = arms
        labels = envelope.get("labels")
        references = references or envelope.get("references") or {
            name: envelope[name] for name in ("baseline", "ceiling") if name in envelope
        }
        envelope_synthetic = envelope.get("is_synthetic", envelope.get("synthetic"))
        arms = envelope["arms"]
    if not isinstance(arms, Mapping) or not arms:
        raise ValueError("stream report needs one or more arm mappings")
    if (not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))
            or not labels or any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError("stream report needs a non-empty unique label sequence")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size != 100:
        raise ValueError("the streaming report's fixed window_size is 100")
    problems: Counter[tuple[str, str, str]] = Counter()
    normal = {name: value for name, value in arms.items() if name in _ARMS and isinstance(value, Mapping)}
    for name, value in arms.items():
        if name not in _ARMS or not isinstance(value, Mapping):
            problems[("unrecognized", "run", "unrecognized arm result")] += 1
    prequential: dict[str, Any] = {}
    checkpoints: dict[str, dict[str, _Slice]] = {name: {} for name in _CHECKPOINTS}
    spend: dict[str, Any] = {}
    synthetic = []
    if not normal:
        raise ValueError("stream report needs a recognized arm")
    for arm, run in sorted(normal.items()):
        if run.get("schema") != "decision-flywheel-evaluations/stream/v1":
            problems[(arm, "run", "unexpected schema")] += 1
        if run.get("arm") != arm:
            problems[(arm, "run", "arm identity mismatch")] += 1
        synthetic.append(bool(run.get("is_synthetic")))
        served = run.get("served", run.get("stream"))
        rows = _scoreable(served, labels, arm, "stream", problems)
        prequential[arm] = _curve(rows, labels, window_size)
        spend[arm] = _spend(run.get("counters"), run.get("usage"), arm, problems)
        for checkpoint in run.get("checkpoints", ()) if isinstance(run.get("checkpoints", ()), Sequence) else ():
            if not isinstance(checkpoint, Mapping):
                problems[(arm, "checkpoint", "malformed checkpoint")] += 1
                continue
            name = checkpoint.get("name")
            if name not in checkpoints:
                problems[(arm, "checkpoint", "unexpected checkpoint")] += 1
                continue
            checkpoints[name][arm] = _slice(checkpoint.get("heldout"), labels, arm, problems)
        for name in _CHECKPOINTS:
            if name not in checkpoints or arm not in checkpoints[name]:
                problems[(arm, "checkpoint", f"missing {name}")] += 1
    _add_references(checkpoints, references, labels, problems)
    is_synthetic = bool(envelope_synthetic) if envelope_synthetic is not None else bool(synthetic) and all(synthetic)
    return {
        "schema": "decision-flywheel-evaluations/stream-report/v1",
        "scope": "descriptive offline streaming report; paired percentile intervals are not significance tests",
        "provenance": {"synthetic_fixture": is_synthetic,
                       "arm_seeds": _seeds(normal, problems)},
        "prequential": prequential,
        "checkpoints": {name: _checkpoint(rows, labels, resamples, seed)
                        for name, rows in checkpoints.items()},
        "spend": spend,
        "problems": _problems(problems),
        "caveats": _caveats(is_synthetic),
    }


def _scoreable(value: Any, labels: Sequence[str], arm: str, stage: str,
               problems: Counter[tuple[str, str, str]]) -> list[_Outcome]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        problems[(arm, stage, "missing outcomes")] += 1
        return []
    out = []
    for position, row in enumerate(value, 1):
        if not isinstance(row, Mapping):
            problems[(arm, stage, "malformed outcome")] += 1
            if stage == "stream": out.append(_Outcome(None, 0, position))
            continue
        status = row.get("status")
        if status not in _SUCCESS:
            problems[(arm, stage, "missing status" if status is None else "unrecognized outcome status")] += 1
            if stage == "stream": out.append(_Outcome(None, 0, _stream_position(row, position)))
            continue
        required = ("item_id", "true_label", "predicted_label", "probabilities")
        if stage == "stream":
            required += ("stream_index", "reviewed_count_at_prediction")
        missing = next((field for field in required if field not in row), None)
        if missing:
            problems[(arm, stage, f"missing {missing}")] += 1
            if stage == "stream": out.append(_Outcome(None, 0, _stream_position(row, position)))
            continue
        if row["true_label"] not in labels or row["predicted_label"] not in labels:
            problems[(arm, stage, "label outside declared labels")] += 1
            if stage == "stream": out.append(_Outcome(None, 0, _stream_position(row, position)))
            continue
        confidence = _confidence(row)
        if confidence is None:
            problems[(arm, stage, "invalid predicted probability")] += 1
            if stage == "stream": out.append(_Outcome(None, 0, _stream_position(row, position)))
            continue
        value, has_confidence = confidence
        item = ItemResult(str(row["item_id"]), value, int(row["true_label"] == row["predicted_label"]),
                          predicted=str(row["predicted_label"]), gold=str(row["true_label"]))
        reviewed = row.get("reviewed_count_at_prediction", 0)
        if isinstance(reviewed, bool) or not isinstance(reviewed, int) or reviewed < 0:
            problems[(arm, stage, "invalid reviewed count")] += 1
            if stage == "stream": out.append(_Outcome(None, 0, _stream_position(row, position)))
            continue
        served_count = _stream_position(row, position)
        if served_count < 1:
            problems[(arm, stage, "invalid stream index")] += 1
            if stage == "stream": out.append(_Outcome(None, reviewed, position + 1))
            continue
        out.append(_Outcome(item, reviewed, served_count, has_confidence))
    return out


def _stream_position(row: Mapping[str, Any], position: int) -> int:
    """The stream/v1 schema's ``stream_index`` is one-based, like served_count."""
    value = row.get("stream_index", position)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _confidence(row: Mapping[str, Any]) -> tuple[float, bool] | None:
    probabilities = row.get("probabilities")
    predicted = row.get("predicted_label")
    # Accuracy and macro-F1 need labels, not distributions.  A label-only
    # fixture therefore remains scoreable; its placeholder is internal and
    # ``has_confidence`` makes the missing calibration input visible.
    if probabilities is None or probabilities == {}:
        return 0.0, False
    if not isinstance(probabilities, Mapping):
        return None
    value = probabilities.get(predicted)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return float(value), True


def _curve(rows: Sequence[_Outcome], labels: Sequence[str], window_size: int
           ) -> dict[str, list[dict[str, Any]]]:
    window, cumulative = [], []
    for index, event in enumerate(rows, 1):
        if index % window_size and index != len(rows):
            continue
        reviewed, served_count = event.reviewed, event.served_count
        cumulative.append(_point(rows[:index], served_count, reviewed, labels))
        window.append(_point(rows[max(0, index - window_size):index], served_count, reviewed, labels))
    return {"window_100": window, "cumulative": cumulative}


def _point(events: Sequence[_Outcome], served_count: int, reviewed_count: int,
           labels: Sequence[str]) -> dict[str, Any]:
    rows = [event.item for event in events if event.item is not None]
    confidence_count = sum(event.item is not None and event.has_confidence for event in events)
    return {"served_count": served_count, "outcomes_seen": len(events), "scored_count": len(rows),
            "unscoreable_count": len(events) - len(rows), "confidence_count": confidence_count,
            "missing_confidence_count": len(rows) - confidence_count, "reviewed_count": reviewed_count,
            "accuracy": round(accuracy([], [row.correct for row in rows]), 6),
            "macro_f1": round(macro_f1(rows, labels), 6)}


def _slice(value: Any, labels: Sequence[str], arm: str, problems: Counter[tuple[str, str, str]]) -> _Slice:
    is_sequence = isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    raw = value if is_sequence else ()
    outcomes = _scoreable(value, labels, arm, "heldout", problems)
    seen, rows, complete = set(), [], is_sequence and len(outcomes) == len(raw)
    for outcome in outcomes:
        if outcome.item.item_id in seen:
            problems[(arm, "heldout", "duplicate item id")] += 1
            complete = False
            continue
        seen.add(outcome.item.item_id)
        rows.append(outcome.item)
    return _Slice(tuple(rows), complete)


def _add_references(checkpoints: dict[str, dict[str, _Slice]], references: Any,
                    labels: Sequence[str], problems: Counter[tuple[str, str, str]]) -> None:
    if references is None:
        return
    if not isinstance(references, Mapping):
        problems[("references", "run", "malformed references")] += 1
        return
    for reference in ("baseline", "ceiling"):
        supplied = references.get(reference)
        if supplied is None:
            continue
        if not isinstance(supplied, Mapping):
            problems[(reference, "run", "malformed reference")] += 1
            continue
        # A caller may pass a whole StreamResult for a fixed floor/ceiling,
        # rather than extracting its held-out slices first.
        if isinstance(supplied.get("checkpoints"), Sequence):
            supplied = {checkpoint.get("name"): checkpoint.get("heldout")
                        for checkpoint in supplied["checkpoints"] if isinstance(checkpoint, Mapping)}
        for name in _CHECKPOINTS:
            if name in supplied:
                checkpoints[name][reference] = _slice(supplied[name], labels, reference, problems)


def _checkpoint(rows: Mapping[str, _Slice], labels: Sequence[str], resamples: int,
                seed: int) -> dict[str, Any]:
    summaries = {name: _summary(value.rows, labels) for name, value in sorted(rows.items())}
    planned = []
    available = [name for name in _ARMS if name in rows]
    planned.extend((treat, base) for base, treat in combinations(available, 2))
    for reference in ("baseline", "ceiling"):
        if reference in rows:
            planned.extend((arm, reference) for arm in available)
    intervals = {f"{treat}-minus-{base}": _interval(rows[base], rows[treat], labels, resamples, seed)
                 for treat, base in planned}
    return {"arms": summaries, "paired_intervals": intervals}


def _summary(rows: Sequence[ItemResult], labels: Sequence[str]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "no scoreable heldout outcomes"}
    return {"available": True, "n": len(rows), "accuracy": round(accuracy([], [row.correct for row in rows]), 6),
            "macro_f1": round(macro_f1(rows, labels), 6)}


def _interval(base: _Slice, treat: _Slice, labels: Sequence[str],
              resamples: int, seed: int) -> dict[str, Any]:
    if not base.complete or not treat.complete:
        return {"available": False, "reason": "heldout outcomes are incomplete"}
    if not base.rows or not treat.rows:
        return {"available": False, "reason": "no scoreable heldout outcomes"}
    try:
        return {"available": True, "confidence_level": 0.95,
                "interval_scope": "paired percentile bootstrap; descriptive only",
                "metrics": {name: paired_interval(base.rows, treat.rows, name, resamples=resamples, seed=seed, labels=labels)
                            for name in ("accuracy", "macro_f1")}}
    except ValueError:
        return {"available": False, "reason": "heldout outcomes are not pairable"}


def _spend(counters: Any, usage: Any, arm: str, problems: Counter[tuple[str, str, str]]) -> dict[str, Any]:
    counters = counters if isinstance(counters, Mapping) else {}
    if not isinstance(counters, Mapping):
        problems[(arm, "spend", "malformed counters")] += 1
    out = {}
    for name in _COUNTERS:
        value = counters.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[name] = value
        else:
            problems[(arm, "spend", f"missing or invalid {name}")] += 1
            out[name] = None
    safe_usage = _numbers_only(usage)
    if safe_usage:
        out["usage"] = safe_usage
    return out


def _numbers_only(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {name: value[name] for name in _USAGE
            if isinstance(value.get(name), int) and not isinstance(value[name], bool) and value[name] >= 0}


def _seeds(runs: Mapping[str, Mapping[str, Any]], problems: Counter[tuple[str, str, str]]) -> dict[str, int | None]:
    out = {}
    for arm, run in sorted(runs.items()):
        seed = run.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
            out[arm] = seed
        else:
            problems[(arm, "run", "missing or invalid seed")] += 1
            out[arm] = None
    return out


def _problems(problems: Counter[tuple[str, str, str]]) -> list[dict[str, Any]]:
    return [{"arm": arm, "stage": stage, "reason": reason, "count": count}
            for (arm, stage, reason), count in sorted(problems.items())]


def _caveats(synthetic: bool) -> list[str]:
    out = []
    if synthetic:
        out.append("This is a fake synthetic fixture, not a live study result.")
    out.extend((
        "Gold labels come from an LLM SME: this measures learning that SME policy, not human truth.",
        "The stream class mix is enriched; natural mix is reported separately when available.",
        "A one stream order experiment cannot resolve differences below about 5 macro-F1 points.",
        "Real SMEs are vaguer and less consistent than an LLM.",
    ))
    return out


def render_stream_report(report: Mapping[str, Any]) -> str:
    """Render aggregates only, as compact Markdown tables with no chart dependency."""
    lines = ["# Streaming SME report", "",
             "Descriptive offline results; no arm selection or significance claim.", "",
             "## Prequential curves", "",
             "| Arm | Series | Served | Reviewed | Scoreable / seen | Accuracy | Macro-F1 | Confidence coverage |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    curves = report.get("prequential", {})
    wrote_curve = False
    if isinstance(curves, Mapping):
        for arm in _ARMS:
            curve = curves.get(arm)
            if not isinstance(curve, Mapping):
                continue
            for series in ("window_100", "cumulative"):
                points = curve.get(series, ())
                if not isinstance(points, Sequence):
                    continue
                for point in points:
                    if not isinstance(point, Mapping):
                        continue
                    lines.append("| {arm} | {series} | {served} | {reviewed} | {scored} / {seen} | {accuracy} | {f1} | {conf} / {scored} |".format(
                        arm=arm, series="window 100" if series == "window_100" else "cumulative",
                        served=_integer(point.get("served_count")), reviewed=_integer(point.get("reviewed_count")),
                        scored=_integer(point.get("scored_count")), seen=_integer(point.get("outcomes_seen")),
                        accuracy=_metric(point.get("accuracy")), f1=_metric(point.get("macro_f1")),
                        conf=_integer(point.get("confidence_count"))))
                    wrote_curve = True
    if not wrote_curve:
        lines.append("| — | — | — | — | unavailable | unavailable | unavailable | unavailable |")

    lines.extend(("", "## Held-out checkpoints", "",
                  "| Checkpoint | Arm | N | Accuracy | Macro-F1 |",
                  "| --- | --- | ---: | ---: | ---: |"))
    checkpoints = report.get("checkpoints", {})
    wrote_checkpoint = False
    if isinstance(checkpoints, Mapping):
        for checkpoint in _CHECKPOINTS:
            entry = checkpoints.get(checkpoint)
            summaries = entry.get("arms", {}) if isinstance(entry, Mapping) else {}
            if not isinstance(summaries, Mapping):
                continue
            for arm in (*_ARMS, "baseline", "ceiling"):
                summary = summaries.get(arm)
                if not isinstance(summary, Mapping):
                    continue
                if summary.get("available") is True:
                    lines.append(f"| {checkpoint} | {arm} | {_integer(summary.get('n'))} | "
                                 f"{_metric(summary.get('accuracy'))} | {_metric(summary.get('macro_f1'))} |")
                else:
                    reason = _reason(summary.get("reason"))
                    lines.append(f"| {checkpoint} | {arm} | unavailable | unavailable: {reason} | unavailable: {reason} |")
                wrote_checkpoint = True
    if not wrote_checkpoint:
        lines.append("| — | — | unavailable | unavailable | unavailable |")

    lines.extend(("", "## Paired bootstrap intervals", "",
                  "| Checkpoint | Contrast | Accuracy effect [95% CI] | Macro-F1 effect [95% CI] |",
                  "| --- | --- | --- | --- |"))
    wrote_interval = False
    if isinstance(checkpoints, Mapping):
        for checkpoint in _CHECKPOINTS:
            entry = checkpoints.get(checkpoint)
            intervals = entry.get("paired_intervals", {}) if isinstance(entry, Mapping) else {}
            if not isinstance(intervals, Mapping):
                continue
            for contrast, interval in sorted(intervals.items()):
                if not isinstance(interval, Mapping) or not _contrast_name(contrast):
                    continue
                if interval.get("available") is True and isinstance(interval.get("metrics"), Mapping):
                    metrics = interval["metrics"]
                    lines.append(f"| {checkpoint} | {contrast} | {_interval_text(metrics.get('accuracy'))} | "
                                 f"{_interval_text(metrics.get('macro_f1'))} |")
                else:
                    reason = _reason(interval.get("reason"))
                    lines.append(f"| {checkpoint} | {contrast} | unavailable: {reason} | unavailable: {reason} |")
                wrote_interval = True
    if not wrote_interval:
        lines.append("| — | — | unavailable | unavailable |")

    lines.extend(("", "## Recorded physical usage", "",
                  "| Arm | Attempts | Failures | Refit rounds | Steering rounds | List rounds | Input tokens | Output tokens | Total tokens |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
    spend = report.get("spend", {})
    wrote_spend = False
    if isinstance(spend, Mapping):
        for arm in _ARMS:
            entry = spend.get(arm)
            if not isinstance(entry, Mapping):
                continue
            usage = entry.get("usage", {})
            usage = usage if isinstance(usage, Mapping) else {}
            lines.append("| {arm} | {attempts} | {failures} | {refit} | {steering} | {lists} | {input} | {output} | {total} |".format(
                arm=arm, attempts=_integer(entry.get("request_attempts")), failures=_integer(entry.get("request_failures")),
                refit=_integer(entry.get("refit_rounds")), steering=_integer(entry.get("steering_rounds")),
                lists=_integer(entry.get("list_rounds")), input=_integer(usage.get("input_tokens")),
                output=_integer(usage.get("output_tokens")), total=_integer(usage.get("total_tokens"))))
            wrote_spend = True
    if not wrote_spend:
        lines.append("| — | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")

    lines.extend(("", "## Caveats", ""))
    synthetic = isinstance(report.get("provenance"), Mapping) and report["provenance"].get("synthetic_fixture") is True
    lines.extend(f"- {caveat}" for caveat in _caveats(synthetic))
    return "\n".join(lines)


_REASONS = frozenset(("no scoreable heldout outcomes", "heldout outcomes are incomplete",
                      "heldout outcomes are not pairable"))


def _integer(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else "unavailable"


def _metric(value: Any) -> str:
    return (f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) else "unavailable")


def _reason(value: Any) -> str:
    return value if value in _REASONS else "unavailable"


def _contrast_name(value: Any) -> bool:
    return isinstance(value, str) and any(value == f"{treat}-minus-{base}"
                                         for treat in (*_ARMS, "baseline", "ceiling")
                                         for base in (*_ARMS, "baseline", "ceiling") if treat != base)


def _interval_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unavailable"
    effect, lower, upper = value.get("effect"), value.get("lower"), value.get("upper")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item)
           for item in (effect, lower, upper)):
        return "unavailable"
    return f"{effect:.3f} [{lower:.3f}, {upper:.3f}]"
