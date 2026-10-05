"""Immutable, text-free provenance for the Amazon-reviews stream partitions.

The mutable SME cache is deliberately *not* a split definition.  A freeze records
the first labelled pool universe and its role assignment once; rehydration checks
the private inputs against that record while ignoring later cache additions.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from typing import Mapping, Sequence

from .datasets import normalized_text_hash
from .unified_sme import PROMPT_VERSION, SmeRecord
from .unified_splits import CorpusItem, Splits, assert_partitions_disjoint, fingerprint_ids


SCHEMA = "decision-flywheel-evaluations/reviews-split-manifest/v1"
UNIVERSE_SIZE = 1500
_ROLES = frozenset(("heldout", "stream", "dev", "reserve"))
_IDENTITY_KEYS = frozenset(("sme_model", "policy_sha256"))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_set_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def _source_sha(*, source_manifest_path: Path | None = None, source_manifest_sha256: str | None = None) -> str:
    if (source_manifest_path is None) == (source_manifest_sha256 is None):
        raise ValueError("provide exactly one source manifest path or source manifest sha256")
    return _file_sha(Path(source_manifest_path)) if source_manifest_path is not None else _sha256(source_manifest_sha256, "source manifest sha256")


def _identity(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise ValueError("SME identity must contain only model and policy sha256")
    model = value["sme_model"]
    if not isinstance(model, str) or not model:
        raise ValueError("SME model must be non-empty")
    return {"sme_model": model, "policy_sha256": _sha256(value["policy_sha256"], "policy sha256")}


def _reviews():
    # Kept lazy so ``unified_reviews`` can use this module as its production loader.
    from . import unified_reviews
    return unified_reviews


def _source_pool_hashes(path: Path) -> dict[str, str]:
    """The committed source manifest adds a second, independent text-hash anchor when available."""
    document = json.loads(path.read_text(encoding="utf-8"))
    pool = document.get("pool")
    if pool is None:
        return {}
    if not isinstance(pool, list):
        raise ValueError("source manifest pool is invalid")
    hashes = {str(entry.get("id", "")): entry.get("text_sha256") for entry in pool if isinstance(entry, Mapping)}
    if len(hashes) != len(pool) or any(not item_id or not isinstance(value, str) for item_id, value in hashes.items()):
        raise ValueError("source manifest pool hashes are invalid")
    return hashes


def freeze_reviews_manifest(pool_rows: Sequence[Mapping[str, object]], records: Mapping[str, SmeRecord],
                            identity: Mapping[str, str], *, source_manifest_path: Path | None = None,
                            source_manifest_sha256: str | None = None, limit: int = UNIVERSE_SIZE) -> dict:
    """Create, but do not write, the one-shot first-pool-window role manifest."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit != UNIVERSE_SIZE:
        raise ValueError(f"reviews freeze requires the planned first {UNIVERSE_SIZE} pool rows")
    source_sha = _source_sha(source_manifest_path=source_manifest_path, source_manifest_sha256=source_manifest_sha256)
    identity = _identity(identity)
    rows = list(pool_rows[:limit])
    if len(rows) != limit:
        raise ValueError(f"reviews freeze needs {limit} pool rows")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("frozen pool universe needs unique non-empty IDs")
    source_hashes = _source_pool_hashes(Path(source_manifest_path)) if source_manifest_path is not None else {}
    selected = {}
    for row, item_id in zip(rows, ids):
        record = records.get(item_id)
        if record is None:
            raise ValueError(f"missing SME record for frozen pool item {item_id}")
        if record.item_id != item_id:
            raise ValueError("SME record item ID differs from the frozen pool")
        selected[item_id] = record
    reviews = _reviews()
    splits, report = reviews.build_reviews_splits(rows, selected)
    roles = {item_id: "heldout" for item_id in splits.paper600}
    roles.update({item_id: "stream" for item_id in splits.pool})
    roles.update({item_id: "dev" for item_id in splits.dev100})
    entries = []
    for row, item_id in zip(rows, ids):
        record = selected[item_id]
        raw_label = record.label
        source_text_sha = None
        if source_hashes:
            if item_id not in source_hashes:
                raise ValueError("frozen pool item is absent from the source manifest")
            from .amazon_reviews import text_sha256
            source_text_sha = text_sha256(str(row["text"]))
            if source_text_sha != source_hashes[item_id]:
                raise ValueError("frozen pool text does not match the source manifest")
        entries.append({"id": item_id, "normalized_text_sha256": normalized_text_hash(str(row["text"])),
                        "source_text_sha256": source_text_sha,
                        "status": record.status, "raw_label": raw_label,
                        "gold_label": None if raw_label is None else reviews.gold_label(raw_label, report["merged"]),
                        "role": roles.get(item_id, "reserve")})
    payload = {"schema": SCHEMA, "source_manifest_sha256": source_sha,
               "sme": {**identity, "prompt_version": PROMPT_VERSION, "pass_tag": "primary"},
               "universe": {"selection": "first_pool_rows_in_file_order", "limit": limit, "records": entries,
                            "id_label_status_sha256": _sha([[entry["id"], entry["raw_label"], entry["status"]]
                                                              for entry in entries])},
               "split": {"seed": report["seed"], "minority_floor": report["minority_floor"],
                         "merged": report["merged"], "labels": report["labels"],
                         "counts": {"heldout": len(splits.paper600), "stream": len(splits.pool), "dev": len(splits.dev100)},
                         "fingerprints": report["fingerprints"]}}
    return {**payload, "manifest_sha256": _sha(payload)}


def _validate(manifest: Mapping[str, object]) -> dict:
    required = {"schema", "source_manifest_sha256", "sme", "universe", "split", "manifest_sha256"}
    if not isinstance(manifest, Mapping) or set(manifest) != required or manifest.get("schema") != SCHEMA:
        raise ValueError("reviews split manifest schema is invalid")
    payload = {key: manifest[key] for key in required - {"manifest_sha256"}}
    if manifest["manifest_sha256"] != _sha(payload):
        raise ValueError("reviews split manifest hash does not match")
    _sha256(manifest["source_manifest_sha256"], "source manifest sha256")
    sme = manifest["sme"]
    if not isinstance(sme, Mapping) or set(sme) != {"sme_model", "policy_sha256", "prompt_version", "pass_tag"}:
        raise ValueError("reviews split SME provenance is invalid")
    _identity({"sme_model": sme["sme_model"], "policy_sha256": sme["policy_sha256"]})
    if sme["prompt_version"] != PROMPT_VERSION or sme["pass_tag"] != "primary":
        raise ValueError("reviews split manifest uses an unsupported SME cache identity")
    universe = manifest["universe"]
    if not isinstance(universe, Mapping) or set(universe) != {"selection", "limit", "records", "id_label_status_sha256"}:
        raise ValueError("reviews split universe is invalid")
    if universe["selection"] != "first_pool_rows_in_file_order" or universe["limit"] != UNIVERSE_SIZE:
        raise ValueError("reviews split manifest does not freeze the planned first pool window")
    entries = universe["records"]
    if not isinstance(entries, list) or len(entries) != UNIVERSE_SIZE:
        raise ValueError("reviews split manifest has the wrong universe size")
    expected_entry = {"id", "normalized_text_sha256", "source_text_sha256", "status", "raw_label", "gold_label", "role"}
    if any(not isinstance(entry, Mapping) or set(entry) != expected_entry for entry in entries):
        raise ValueError("reviews split manifest records are invalid")
    if len({entry["id"] for entry in entries}) != len(entries) or any(not isinstance(entry["id"], str) or not entry["id"] for entry in entries):
        raise ValueError("reviews split manifest record IDs are invalid")
    for entry in entries:
        _sha256(entry["normalized_text_sha256"], "review text hash")
        if entry["source_text_sha256"] is not None:
            _sha256(entry["source_text_sha256"], "source review text hash")
        if entry["status"] not in {"accepted", "label_only", "ambiguous", "rejected"} or entry["role"] not in _ROLES:
            raise ValueError("reviews split manifest record status or role is invalid")
    if universe["id_label_status_sha256"] != _sha([[entry["id"], entry["raw_label"], entry["status"]] for entry in entries]):
        raise ValueError("reviews split manifest universe digest does not match")
    split = manifest["split"]
    if not isinstance(split, Mapping) or set(split) != {"seed", "minority_floor", "merged", "labels", "counts", "fingerprints"}:
        raise ValueError("reviews split definition is invalid")
    counts = split["counts"]
    if counts != {"heldout": 300, "stream": 600, "dev": 100}:
        raise ValueError("reviews split counts differ from the approved protocol")
    reviews = _reviews()
    if (not isinstance(split["merged"], bool) or split["seed"] != reviews.SPLIT_SEED
            or split["minority_floor"] != reviews.MINORITY_FLOOR
            or split["labels"] != list(reviews.reviews_labels(split["merged"]))):
        raise ValueError("reviews split manifest configuration is invalid")
    for entry in entries:
        raw = entry["raw_label"]
        if raw is not None and raw not in reviews.REVIEWS_LABELS:
            raise ValueError("reviews split manifest raw label is invalid")
        expects_label = entry["status"] in {"accepted", "label_only"}
        if (raw is None) == expects_label:
            raise ValueError("reviews split manifest status and raw label disagree")
        gold = None if raw is None else reviews.gold_label(raw, split["merged"])
        if entry["gold_label"] != gold:
            raise ValueError("reviews split manifest gold label is invalid")
    by_role = {role: [entry["id"] for entry in entries if entry["role"] == role] for role in _ROLES}
    if {role: len(by_role[role]) for role in ("heldout", "stream", "dev")} != counts:
        raise ValueError("reviews split role counts do not match")
    if split["fingerprints"] != {"heldout": fingerprint_ids(by_role["heldout"]), "stream": fingerprint_ids(by_role["stream"]),
                                 "dev": fingerprint_ids(by_role["dev"])}:
        raise ValueError("reviews split fingerprints do not match")
    return dict(manifest)


def write_reviews_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Write a validated freeze once; accidental re-selection is never an overwrite."""
    _validate(manifest)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen reviews manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(manifest) + b"\n")


def read_reviews_manifest(path: Path) -> dict:
    return _validate(json.loads(Path(path).read_text(encoding="utf-8")))


def frozen_role_ids(manifest: Mapping[str, object], role: str) -> tuple[str, ...]:
    """Text-free IDs for one frozen role; safe for stage preflight and gates."""
    checked = _validate(manifest)
    if role not in _ROLES:
        raise ValueError("unknown frozen reviews role")
    return tuple(sorted(entry["id"] for entry in checked["universe"]["records"] if entry["role"] == role))


def load_frozen_reviews_splits(path: Path, *, pool_rows: Sequence[Mapping[str, object]] | None = None,
                               pool_path: Path | None = None, cache_path: Path,
                               source_manifest_path: Path) -> Splits:
    """Rehydrate only the frozen roles, rejecting selected-input drift and ignoring later cache rows."""
    manifest = read_reviews_manifest(path)
    if _file_sha(Path(source_manifest_path)) != manifest["source_manifest_sha256"]:
        raise ValueError("reviews source manifest hash does not match the frozen split")
    if pool_rows is None:
        if pool_path is None:
            raise ValueError("a private pool path is required to rehydrate frozen reviews")
        pool_rows = [json.loads(line) for line in Path(pool_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = manifest["universe"]["records"]
    rows = list(pool_rows[:UNIVERSE_SIZE])
    by_id = {str(row.get("id", "")): row for row in rows}
    if len(by_id) != UNIVERSE_SIZE or [str(row.get("id", "")) for row in rows] != [entry["id"] for entry in entries]:
        raise ValueError("private pool order or IDs differ from the frozen reviews universe")
    for entry in entries:
        if normalized_text_hash(str(by_id[entry["id"]]["text"])) != entry["normalized_text_sha256"]:
            raise ValueError("private pool review text hash differs from the frozen split")
    source_hashes = _source_pool_hashes(Path(source_manifest_path))
    if source_hashes:
        from .amazon_reviews import text_sha256
        for entry in entries:
            if (source_hashes.get(entry["id"]) != entry["source_text_sha256"]
                    or text_sha256(str(by_id[entry["id"]]["text"])) != entry["source_text_sha256"]):
                raise ValueError("private pool source text hash differs from the frozen split")
    reviews = _reviews()
    records, identity = reviews.read_sme_records(cache_path, [entry["id"] for entry in entries],
                                                 model=manifest["sme"]["sme_model"],
                                                 policy_sha256=manifest["sme"]["policy_sha256"])
    if identity != {"sme_model": manifest["sme"]["sme_model"], "policy_sha256": manifest["sme"]["policy_sha256"]}:
        raise ValueError("SME cache identity differs from the frozen split")
    for entry in entries:
        record = records.get(entry["id"])
        if record is None:
            raise ValueError("a frozen reviews SME record is missing from the cache")
        if (record.status, record.label) != (entry["status"], entry["raw_label"]):
            raise ValueError("a frozen reviews SME label or status differs from the cache")
        expected_gold = None if record.label is None else reviews.gold_label(record.label, manifest["split"]["merged"])
        if expected_gold != entry["gold_label"]:
            raise ValueError("a frozen reviews gold label differs from the manifest")
    items = {}
    roles = {role: [] for role in ("heldout", "stream", "dev")}
    for entry in entries:
        role = entry["role"]
        if role == "reserve":
            continue
        if entry["gold_label"] is None:
            raise ValueError("a frozen scoring role has no gold label")
        item_id = entry["id"]
        roles[role].append(item_id)
        items[item_id] = CorpusItem(item_id, "pool" if role == "stream" else "test", entry["gold_label"],
                                    str(by_id[item_id]["text"]))
    splits = Splits(items, tuple(sorted(roles["stream"])), tuple(sorted((*roles["heldout"], *roles["dev"]))),
                    tuple(sorted(roles["heldout"])), tuple(sorted(roles["dev"])), final_name="heldout-300")
    assert_partitions_disjoint(splits)
    return splits


def freeze_main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin private-data CLI
    parser = argparse.ArgumentParser(description="Freeze the planned first-1500 Amazon reviews split once (no model calls)")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sme-model", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--expected-universe-id-sha256", required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen reviews manifest {args.out}")
    rows = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_policy = _sha256(args.policy_sha256, "policy sha256")
    expected_universe = _sha256(args.expected_universe_id_sha256, "expected universe ID sha256")
    if len(rows) < UNIVERSE_SIZE or _id_set_sha([str(row.get("id", "")) for row in rows[:UNIVERSE_SIZE]]) != expected_universe:
        raise ValueError("the first pool window does not match the explicitly approved universe digest")
    reviews = _reviews()
    records, identity = reviews.read_sme_records(args.cache, [str(row["id"]) for row in rows[:UNIVERSE_SIZE]],
                                                 model=args.sme_model)
    if identity["policy_sha256"] != expected_policy:
        raise ValueError("the SME cache policy does not match --policy-sha256")
    frozen = freeze_reviews_manifest(rows, records, identity, source_manifest_path=args.source_manifest)
    write_reviews_manifest(args.out, frozen)
    print(json.dumps({"manifest_sha256": frozen["manifest_sha256"], "counts": frozen["split"]["counts"]}, sort_keys=True))
    return 0
