from .unified_stream_reporting import render_stream_report, report_stream


LABELS = ("approve", "abusive", "promotional")


def _row(number, gold, predicted, reviewed, *, status="completed"):
    return {
        "stream_index": number + 1,
        "batch_index": number // 10,
        "item_id": f"stream-{number}",
        "true_label": gold,
        "predicted_label": predicted,
        "probabilities": {predicted: 0.8},
        "status": status,
        "review_selected": reviewed,
        "review_propensity": 0.3,
        "reviewed_count_at_prediction": reviewed,
    }


def _heldout(prefix, better=False):
    rows = []
    for number in range(12):
        gold = LABELS[number % len(LABELS)]
        predicted = gold if better or number % 2 else LABELS[(number + 1) % len(LABELS)]
        rows.append({"item_id": f"{prefix}-{number}", "true_label": gold,
                     "predicted_label": predicted, "probabilities": {predicted: 0.8},
                     "status": "completed"})
    return rows


def _arm(name, *, better=False, synthetic=True):
    stream = [_row(number, LABELS[number % 3], LABELS[number % 3] if better or number % 2
                   else LABELS[(number + 1) % 3], number // 3)
              for number in range(105)]
    return {"schema": "decision-flywheel-evaluations/stream/v1", "arm": name, "seed": 7,
            "is_synthetic": synthetic, "served": stream, "reviews": [],
            "checkpoints": [{"name": "stream-300", "served_count": 300, "reviewed_count": 90,
                             "heldout": _heldout("a", better)},
                            {"name": "end", "served_count": 600, "reviewed_count": 180,
                             "heldout": _heldout("b", better)}],
            "counters": {"refit_rounds": 2, "steering_rounds": 1, "list_rounds": 3,
                         "request_attempts": 44, "request_failures": 2}}


def test_a_fake_stream_report_has_windowed_and_cumulative_prequential_curves_by_reviewed_count():
    report = report_stream({"B": _arm("B"), "E": _arm("E", better=True)}, LABELS, resamples=20)
    curve = report["prequential"]["E"]
    assert [point["scored_count"] for point in curve["window_100"]] == [100, 100]
    assert curve["window_100"][-1]["reviewed_count"] == 34
    assert set(curve["cumulative"][-1]) == {"served_count", "outcomes_seen", "scored_count", "unscoreable_count",
                                              "confidence_count", "missing_confidence_count", "reviewed_count", "accuracy", "macro_f1"}
    assert report["provenance"]["synthetic_fixture"] is True
    assert "synthetic fixture" in " ".join(report["caveats"])


def test_the_public_run_envelope_is_accepted_without_importing_the_stream_runtime():
    report = report_stream({"schema": "decision-flywheel-evaluations/stream/v1", "is_synthetic": True,
                            "labels": list(LABELS), "arms": {"B": _arm("B")}}, resamples=5)
    assert report["schema"] == "decision-flywheel-evaluations/stream-report/v1"
    assert report["provenance"]["synthetic_fixture"] is True


def test_checkpoints_include_supplied_floor_and_ceiling_and_paired_bootstrap_intervals_without_p_values():
    report = report_stream(
        {"B": _arm("B"), "E": _arm("E", better=True)}, LABELS,
        references={"baseline": {"stream-300": _heldout("a"), "end": _heldout("b")},
                    "ceiling": {"stream-300": _heldout("a", True), "end": _heldout("b", True)}},
        resamples=30,
    )
    end = report["checkpoints"]["end"]
    assert set(end["arms"]) == {"B", "E", "baseline", "ceiling"}
    interval = end["paired_intervals"]["E-minus-B"]
    assert interval["available"] is True
    assert set(interval["metrics"]) == {"accuracy", "macro_f1"}
    assert "p_value" not in repr(report)
    assert "best" not in repr(report).lower()


def test_missing_or_unsuccessful_outcomes_are_visible_and_never_guessed():
    arm = _arm("B")
    arm["served"][0].pop("predicted_label")
    arm["served"][1]["status"] = "failed"
    report = report_stream({"B": arm}, LABELS, resamples=5)
    problems = {(problem["stage"], problem["reason"]): problem["count"] for problem in report["problems"]}
    assert problems[("stream", "missing predicted_label")] == 1
    assert problems[("stream", "unrecognized outcome status")] == 1
    assert report["prequential"]["B"]["cumulative"][-1]["scored_count"] == 103


def test_failed_stream_slots_stay_in_the_100_slot_window_and_show_coverage():
    arm = _arm("B")
    arm["served"][99]["status"] = "anything-private"
    report = report_stream({"B": arm}, LABELS, resamples=5)
    point = report["prequential"]["B"]["window_100"][0]
    assert point["outcomes_seen"] == 100 and point["scored_count"] == 99 and point["unscoreable_count"] == 1
    assert all("anything-private" not in repr(problem) for problem in report["problems"])


def test_incomplete_or_duplicate_heldout_slices_cannot_get_a_paired_interval():
    b, e = _arm("B"), _arm("E", better=True)
    e["checkpoints"][1]["heldout"][0]["status"] = "failed"
    b["checkpoints"][0]["heldout"].append(dict(b["checkpoints"][0]["heldout"][0]))
    report = report_stream({"B": b, "E": e}, LABELS, resamples=5)
    assert report["checkpoints"]["end"]["paired_intervals"]["E-minus-B"] == {
        "available": False, "reason": "heldout outcomes are incomplete"}


def test_unknown_arms_and_usage_keys_are_never_reflected_in_the_report():
    arm = _arm("B")
    arm["usage"] = {"input_tokens": 2, "api_key": 99}
    report = report_stream({"B": arm, "untrusted-name": arm}, LABELS, resamples=5)
    assert set(report["prequential"]) == {"B"}
    assert report["spend"]["B"]["usage"] == {"input_tokens": 2}
    assert "untrusted-name" not in repr(report)


def test_non_finite_predicted_probability_is_a_visible_unscoreable_outcome():
    arm = _arm("B")
    arm["served"][0]["probabilities"] = {arm["served"][0]["predicted_label"]: float("nan")}
    report = report_stream({"B": arm}, LABELS, resamples=5)
    assert any(problem["reason"] == "invalid predicted probability" for problem in report["problems"])
    assert report["prequential"]["B"]["cumulative"][-1]["unscoreable_count"] == 1


def test_label_only_outcomes_score_accuracy_and_macro_f1_but_mark_confidence_coverage_missing():
    arm = _arm("B")
    arm["served"] = [_row(0, "approve", "approve", 0), _row(1, "abusive", "abusive", 0)]
    for row in arm["served"]:
        row["probabilities"] = {}
    report = report_stream({"B": arm}, LABELS, resamples=5)
    point = report["prequential"]["B"]["cumulative"][-1]
    assert point["served_count"] == 2 and point["accuracy"] == point["macro_f1"] == 1.0
    assert point["confidence_count"] == 0 and point["missing_confidence_count"] == 2


def test_spend_reports_recorded_attempts_and_usage_without_fabricated_currency():
    arm = _arm("B", synthetic=False)
    arm["usage"] = {"input_tokens": 123, "output_tokens": 45}
    report = report_stream({"B": arm}, LABELS, resamples=5)
    spend = report["spend"]["B"]
    assert spend == {"request_attempts": 44, "request_failures": 2, "refit_rounds": 2,
                     "steering_rounds": 1, "list_rounds": 3,
                     "usage": {"input_tokens": 123, "output_tokens": 45}}
    assert "$" not in repr(spend) and "cost" not in spend


def test_rendered_report_is_text_free_about_inputs_but_states_required_caveats():
    report = report_stream({"B": _arm("B")}, LABELS, resamples=5)
    rendered = render_stream_report(report)
    assert "LLM SME" in rendered and "one stream order" in rendered
    assert "raw reply" not in rendered.lower() and "policy text" not in rendered.lower()


def test_rendered_report_has_compact_curve_checkpoint_interval_and_physical_usage_tables():
    baseline, enriched = _arm("B"), _arm("E", better=True)
    baseline["usage"] = {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}
    enriched["checkpoints"][1]["heldout"][0]["status"] = "failed"
    report = report_stream({"B": baseline, "E": enriched}, LABELS, resamples=5)
    rendered = render_stream_report(report)
    assert "| B | window 100 |" in rendered and "| E | cumulative |" in rendered
    assert "| stream-300 | B |" in rendered and "| end | E |" in rendered
    assert "heldout outcomes are incomplete" in rendered
    assert "| B | 44 | 2 | 2 | 1 | 3 | 123 | 45 | 168 |" in rendered
    assert "$" not in rendered and "p-value" not in rendered.lower()
