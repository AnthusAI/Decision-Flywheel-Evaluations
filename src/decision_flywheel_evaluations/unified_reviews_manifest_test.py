import json

import pytest

from . import unified_reviews as reviews
from . import unified_reviews_manifest as manifest
from . import unified_sme as sme


POLICY = "Synthetic policy only."
MODEL = sme.FAKE_MODEL


def _pool(n=1500):
    labels = ("approve", "abusive", "promotional", "seller_shipping", "price_availability")
    return [{"id": f"review-{index:04d}", "text": f"synthetic review text {index}",
             "label": labels[index % len(labels)]} for index in range(n)]


def _records(rows):
    return {row["id"]: sme.SmeRecord(row["id"], row["label"], "R1", "synthetic", None, "accepted")
            for row in rows}


def _cache(path, records):
    cache = sme.SmeCache(path)
    for record in records.values():
        cache.put(sme.cache_key(record.item_id, sme.policy_sha(POLICY), MODEL), record, model=MODEL,
                  policy_sha256=sme.policy_sha(POLICY), pass_tag="primary")
    cache.db.close()


def _source(path):
    path.write_text(json.dumps({"schema": "synthetic-source-v1"}) + "\n", encoding="utf-8")
    return path


def test_a_frozen_manifest_keeps_roles_stable_when_the_cache_grows(tmp_path):
    rows = _pool(1525)
    initial = _records(rows[:1500])
    cache_path, source_path = tmp_path / "sme.sqlite", _source(tmp_path / "source.json")
    _cache(cache_path, initial)
    frozen = manifest.freeze_reviews_manifest(rows, initial, {"sme_model": MODEL,
                                                               "policy_sha256": sme.policy_sha(POLICY)},
                                              source_manifest_path=source_path)
    path = tmp_path / "frozen.json"
    manifest.write_reviews_manifest(path, frozen)
    before = manifest.load_frozen_reviews_splits(path, pool_rows=rows, cache_path=cache_path,
                                                 source_manifest_path=source_path)

    _cache(cache_path, _records(rows[1500:]))
    after = manifest.load_frozen_reviews_splits(path, pool_rows=rows, cache_path=cache_path,
                                                source_manifest_path=source_path)

    assert before.pool == after.pool
    assert before.paper600 == after.paper600
    assert before.dev100 == after.dev100


def test_a_frozen_manifest_rejects_a_changed_selected_label_or_text_hash(tmp_path):
    rows = _pool()
    records = _records(rows)
    cache_path, source_path = tmp_path / "sme.sqlite", _source(tmp_path / "source.json")
    _cache(cache_path, records)
    frozen = manifest.freeze_reviews_manifest(rows, records, {"sme_model": MODEL,
                                                               "policy_sha256": sme.policy_sha(POLICY)},
                                              source_manifest_path=source_path)
    path = tmp_path / "frozen.json"
    manifest.write_reviews_manifest(path, frozen)
    changed = list(rows)
    changed[0] = {**changed[0], "text": "different synthetic text"}
    with pytest.raises(ValueError, match="text hash"):
        manifest.load_frozen_reviews_splits(path, pool_rows=changed, cache_path=cache_path,
                                            source_manifest_path=source_path)


def test_a_frozen_manifest_is_text_free_and_refuses_overwrite(tmp_path):
    rows, records = _pool(), _records(_pool())
    source_path = _source(tmp_path / "source.json")
    frozen = manifest.freeze_reviews_manifest(rows, records, {"sme_model": MODEL,
                                                               "policy_sha256": sme.policy_sha(POLICY)},
                                              source_manifest_path=source_path)
    rendered = json.dumps(frozen, sort_keys=True)
    assert "synthetic review text" not in rendered and "synthetic\"" not in rendered
    path = tmp_path / "frozen.json"
    manifest.write_reviews_manifest(path, frozen)
    with pytest.raises(FileExistsError):
        manifest.write_reviews_manifest(path, frozen)


def test_a_frozen_manifest_rejects_wrong_source_identity_and_missing_selected_cache_record(tmp_path):
    rows, records = _pool(), _records(_pool())
    cache_path, source_path = tmp_path / "sme.sqlite", _source(tmp_path / "source.json")
    _cache(cache_path, records)
    frozen = manifest.freeze_reviews_manifest(rows, records, {"sme_model": MODEL,
                                                               "policy_sha256": sme.policy_sha(POLICY)},
                                              source_manifest_path=source_path)
    path = tmp_path / "frozen.json"
    manifest.write_reviews_manifest(path, frozen)
    source_path.write_text('{"schema":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest"):
        manifest.load_frozen_reviews_splits(path, pool_rows=rows, cache_path=cache_path,
                                            source_manifest_path=source_path)

    _source(source_path)
    incomplete = tmp_path / "incomplete.sqlite"
    _cache(incomplete, {item_id: record for item_id, record in records.items() if item_id != rows[0]["id"]})
    with pytest.raises(ValueError, match="SME record is missing"):
        manifest.load_frozen_reviews_splits(path, pool_rows=rows, cache_path=incomplete,
                                            source_manifest_path=source_path)


def test_a_frozen_manifest_rejects_a_changed_selected_cache_label(tmp_path):
    rows, records = _pool(), _records(_pool())
    cache_path, source_path = tmp_path / "sme.sqlite", _source(tmp_path / "source.json")
    _cache(cache_path, records)
    frozen = manifest.freeze_reviews_manifest(rows, records, {"sme_model": MODEL,
                                                               "policy_sha256": sme.policy_sha(POLICY)},
                                              source_manifest_path=source_path)
    path = tmp_path / "frozen.json"
    manifest.write_reviews_manifest(path, frozen)
    changed = dict(records)
    first = rows[0]["id"]
    changed[first] = sme.SmeRecord(first, "promotional", "R1", "synthetic", None, "accepted")
    changed_cache = tmp_path / "changed.sqlite"
    _cache(changed_cache, changed)
    with pytest.raises(ValueError, match="label or status"):
        manifest.load_frozen_reviews_splits(path, pool_rows=rows, cache_path=changed_cache,
                                            source_manifest_path=source_path)


def test_freeze_requires_the_exact_first_limit_pool_records():
    rows = _pool(1501)
    records = _records(rows[:1499])
    with pytest.raises(ValueError, match="missing SME record"):
        manifest.freeze_reviews_manifest(rows, records, {"sme_model": MODEL,
                                                          "policy_sha256": sme.policy_sha(POLICY)},
                                         source_manifest_sha256="a" * 64)


def test_the_freeze_cli_requires_explicit_policy_and_universe_digest_before_writing(tmp_path):
    rows, records = _pool(), _records(_pool())
    pool, cache, source, out = tmp_path / "pool.jsonl", tmp_path / "sme.sqlite", _source(tmp_path / "source.json"), tmp_path / "frozen.json"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _cache(cache, records)
    digest = __import__("hashlib").sha256("\n".join(sorted(row["id"] for row in rows)).encode()).hexdigest()
    assert manifest.freeze_main(["--pool", str(pool), "--cache", str(cache), "--source-manifest", str(source),
                                 "--out", str(out), "--sme-model", MODEL, "--policy-sha256", sme.policy_sha(POLICY),
                                 "--expected-universe-id-sha256", digest]) == 0
    assert manifest.read_reviews_manifest(out)["manifest_sha256"]
    refused = tmp_path / "refused.json"
    with pytest.raises(ValueError, match="cache policy"):
        manifest.freeze_main(["--pool", str(pool), "--cache", str(cache), "--source-manifest", str(source),
                              "--out", str(refused), "--sme-model", MODEL, "--policy-sha256", "0" * 64,
                              "--expected-universe-id-sha256", digest])
    assert not refused.exists()
