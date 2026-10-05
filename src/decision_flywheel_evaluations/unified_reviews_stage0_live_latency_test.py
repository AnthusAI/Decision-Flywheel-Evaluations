"""Latency and safe metadata contract for the Stage 0 live collector."""
from __future__ import annotations

from .unified_reviews_stage0_live import build_stage0_live_collector
from .unified_reviews_stage0_live_test import _Client, _inputs
from .unified_spend import SpendLedger


def _heldout(manifest):
    return next(entry["id"] for entry in manifest["universe"]["records"] if entry["role"] == "heldout")


def test_successful_results_retain_measured_latency_and_only_token_usage(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=SpendLedger(None, ceiling=2),
        inner_client_factory=lambda: _Client(usage={"prompt_tokens": 3, "completion_tokens": 2,
                                                     "private_numeric_body": 99}),
    )

    result = collector.collect("S", _heldout(frozen))

    assert result.model == "pinned-jev"
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    assert result.latency_seconds is not None and result.latency_seconds >= 0


def test_failed_and_malformed_results_retain_only_measured_latency(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)

    class FailingClient:
        def system_one(self, *, state, questions):
            raise RuntimeError("private provider body")

    failed = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=SpendLedger(None, ceiling=2),
        inner_client_factory=FailingClient,
    ).collect("S", _heldout(frozen))
    malformed = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=SpendLedger(None, ceiling=2),
        inner_client_factory=lambda: _Client(label="not-a-label"),
    ).collect("S", _heldout(frozen))

    for result, status in ((failed, "failed"), (malformed, "malformed")):
        assert result.status == status and result.label is result.model is result.usage is None
        assert result.latency_seconds is not None and result.latency_seconds >= 0
