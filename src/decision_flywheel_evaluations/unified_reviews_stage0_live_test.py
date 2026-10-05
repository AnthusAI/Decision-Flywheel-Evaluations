"""Specs for the counted, lazy native Jev Stage 0 collector."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .unified_reviews_manifest import freeze_reviews_manifest
from .unified_reviews_stage0_live import build_stage0_live_collector
from .unified_sme import FAKE_MODEL, SmeRecord, policy_sha
from .unified_spend import SpendLedger


POLICY = "private F policy for the test\n"


def _inputs(tmp_path):
    labels = ("approve", "abusive", "promotional", "seller_shipping", "price_availability")
    rows = [{"id": f"review-{number:04d}", "text": f"private review {number}",
             "label": labels[number % len(labels)]}
            for number in range(1500)]
    records = {row["id"]: SmeRecord(row["id"], row["label"], "R1", "reason", None, "accepted")
               for row in rows}
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"schema": "test-source"}), encoding="utf-8")
    frozen = freeze_reviews_manifest(rows, records, {"sme_model": FAKE_MODEL, "policy_sha256": policy_sha(POLICY)},
                                     source_manifest_path=source)
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    s_path, f_path = tmp_path / "S.txt", tmp_path / "F.txt"
    s_path.write_text("public S task", encoding="utf-8")
    f_path.write_text(POLICY, encoding="utf-8")
    return frozen, pool, source, s_path, f_path


class _Client:
    def __init__(self, label="approve", usage=None):
        self.label = label
        self.usage = usage or {"input_tokens": 3, "output_tokens": 2}
        self.calls = []

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        labels = list(questions["reviews"]["criteria"])
        probabilities = {label: (1.0 if label == self.label else 0.0) for label in labels}
        return SimpleNamespace(answers={"reviews": {"choice": self.label, "probabilities": probabilities}},
                               model="pinned-jev", usage=self.usage)


def test_construction_reads_no_private_input_or_client_until_a_selected_live_cell(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    created = []
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=SpendLedger(None, ceiling=2),
        inner_client_factory=lambda: created.append(True),
    )

    assert created == []
    assert collector.model == "pinned-jev"


def test_task_fingerprints_are_the_actual_native_tasks_and_need_no_client_or_attempt(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    created, ledger = [], SpendLedger(None, ceiling=2)
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=ledger, inner_client_factory=lambda: created.append(True),
    )
    item_id = next(entry["id"] for entry in frozen["universe"]["records"] if entry["role"] == "heldout")

    fingerprints = collector.task_fingerprints()

    assert fingerprints == {condition: collector._target_and_task(condition, item_id)[0].fingerprint
                            for condition in ("S", "F")}
    assert all(len(value) == 64 for value in fingerprints.values())
    assert created == [] and ledger.used == 0
    s_path.write_text("changed public S task", encoding="utf-8")
    changed = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=SpendLedger(None, ceiling=2), inner_client_factory=lambda: created.append(True),
    )
    assert changed.task_fingerprints()["S"] != fingerprints["S"] and created == []


def test_a_frozen_heldout_cell_uses_one_counted_native_request_with_identical_labels(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    client, ledger = _Client(), SpendLedger(None, ceiling=3)
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=ledger, inner_client_factory=lambda: client,
    )
    item_id = next(entry["id"] for entry in frozen["universe"]["records"] if entry["role"] == "heldout")

    s = collector.collect("S", item_id)
    f = collector.collect("F", item_id)

    assert (s.status, s.label, s.model, s.usage) == ("completed", "approve", "pinned-jev",
                                                       {"input_tokens": 3, "output_tokens": 2})
    assert (f.status, f.label) == ("completed", "approve")
    assert ledger.used == 2 and len(client.calls) == 2
    assert client.calls[0][0]["target"] == client.calls[1][0]["target"]
    assert list(client.calls[0][1]["reviews"]["criteria"]) == list(client.calls[1][1]["reviews"]["criteria"])
    assert client.calls[0][1]["reviews"]["instructions"] == "public S task"
    assert client.calls[1][1]["reviews"]["instructions"].startswith(POLICY.strip())


def test_validation_checks_both_s_and_raw_policy_identity_before_any_client_or_s_request(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    f_path.write_text(POLICY + "changed", encoding="utf-8")
    created, ledger = [], SpendLedger(None, ceiling=2)
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=ledger, inner_client_factory=lambda: created.append(True),
    )

    with pytest.raises(ValueError, match="policy hash"):
        collector.validate_tasks()
    assert created == [] and ledger.used == 0


def test_changed_private_text_refuses_before_a_client_or_request_is_created(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    rows = [json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = "changed private review"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    created, ledger = [], SpendLedger(None, ceiling=2)
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=ledger, inner_client_factory=lambda: created.append(True),
    )
    item_id = next(entry["id"] for entry in frozen["universe"]["records"] if entry["role"] == "heldout")

    with pytest.raises(ValueError, match="text hash"):
        collector.collect("S", item_id)
    assert created == [] and ledger.used == 0


def test_a_malformed_provider_result_is_text_free_but_still_counted(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    client, ledger = _Client(label="not-a-label"), SpendLedger(None, ceiling=2)
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=ledger, inner_client_factory=lambda: client,
    )
    item_id = next(entry["id"] for entry in frozen["universe"]["records"] if entry["role"] == "heldout")

    result = collector.collect("S", item_id)

    assert result.status == "malformed" and result.label is None and result.model is None and result.usage is None
    assert ledger.used == 1


def test_provider_usage_keeps_only_recognized_token_counters(tmp_path):
    frozen, pool, source, s_path, f_path = _inputs(tmp_path)
    client = _Client(usage={"input_tokens": 3, "output_tokens": 2, "private_body": 99})
    collector = build_stage0_live_collector(
        frozen, pool_path=pool, source_manifest_path=source, s_path=s_path, f_path=f_path,
        model="pinned-jev", ledger=SpendLedger(None, ceiling=2), inner_client_factory=lambda: client,
    )
    item_id = next(entry["id"] for entry in frozen["universe"]["records"] if entry["role"] == "heldout")

    assert collector.collect("S", item_id).usage == {"input_tokens": 3, "output_tokens": 2}
