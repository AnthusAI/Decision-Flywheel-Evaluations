import importlib.util
import json
import os
from pathlib import Path

import pytest

from decision_flywheel_evaluations import unified_reviews as reviews
from decision_flywheel_evaluations import unified_sme as sme
from decision_flywheel_evaluations.unified_corpus import CORPORA, get_corpus

POLICY = "Toy policy for specs only.\nR1 (abusive). Toy rule.\nR11 (approve). Toy rule.\n"
SHA = sme.policy_sha(POLICY)
MODEL = sme.FAKE_MODEL
REPO = Path(__file__).resolve().parents[2]


def synthetic_pool(counts, extra_status=()):
    """Pool rows and SME records with the given label counts, plus (status, n) records that carry no label."""
    rows, records, n = [], {}, 0
    for label, k in counts.items():
        for _ in range(k):
            item_id = f"amz-{n:05d}"
            rows.append({"id": item_id, "text": f"synthetic review {n} about a {label} matter"})
            records[item_id] = sme.SmeRecord(item_id, label, "R1", "A reason.", None, "accepted")
            n += 1
    for status, k in extra_status:
        for _ in range(k):
            item_id = f"amz-{n:05d}"
            rows.append({"id": item_id, "text": f"synthetic review {n} with no gold"})
            if status != "unlabeled":
                records[item_id] = sme.SmeRecord(item_id, None, None, None, None, status)
            n += 1
    return rows, records


COUNTS = {"approve": 900, "abusive": 40, "promotional": 200, "seller_shipping": 250, "price_availability": 60}
NO_GOLD = (("ambiguous", 30), ("rejected", 5), ("unlabeled", 20))


def test_the_labels_are_in_the_fixed_order_and_the_merge_is_decided_from_the_abusive_count():
    assert reviews.REVIEWS_LABELS == ("approve", "abusive", "promotional", "seller_shipping", "price_availability")
    assert reviews.MERGED_LABELS == ("approve", "remove_other", "seller_shipping", "price_availability")
    assert reviews.should_merge({"abusive": 19}) and not reviews.should_merge({"abusive": 20})
    assert reviews.should_merge({}) and reviews.gold_label("promotional", True) == "remove_other"


def test_held_out_and_stream_are_disjoint_sized_and_give_every_class_its_floor_where_the_pool_allows():
    rows, records = synthetic_pool(COUNTS, NO_GOLD)
    splits, report = reviews.build_reviews_splits(rows, records)
    held, stream, dev = set(splits.paper600), set(splits.pool), set(splits.dev100)
    assert (len(held), len(stream), len(dev)) == (300, 600, 100)
    assert not (held & stream or held & dev or stream & dev)
    combined = {l: report["heldout"]["by_label"].get(l, 0) + report["stream"]["by_label"].get(l, 0)
                for l in reviews.REVIEWS_LABELS}
    assert combined["abusive"] == 40 and combined["price_availability"] == 60   # all there is, below 90
    assert combined["promotional"] >= 90 and combined["seller_shipping"] >= 90 and sum(combined.values()) == 900
    assert report["floor_met"] == {"approve": True, "abusive": False, "promotional": True,
                                   "seller_shipping": True, "price_availability": False}
    for label, total in combined.items():   # each class is cut 1:2 between held-out and stream
        assert abs(report["heldout"]["by_label"].get(label, 0) - total / 3) <= 1
    assert splits.final_name == "heldout-300" and report["merged"] is False


def test_items_without_an_sme_label_never_reach_gold_and_are_counted():
    rows, records = synthetic_pool(COUNTS, NO_GOLD)
    splits, report = reviews.build_reviews_splits(rows, records)
    assert report["excluded_from_gold"] == {"ambiguous": 30, "rejected": 5, "unlabeled": 20}
    no_gold = {r["id"] for r in rows[sum(COUNTS.values()):]}
    assert not no_gold & set(splits.items)
    assert all(item.reference_label in reviews.REVIEWS_LABELS for item in splits.items.values())


def test_the_splits_are_the_same_for_the_same_seed_and_differ_for_another():
    rows, records = synthetic_pool(COUNTS)
    first, _ = reviews.build_reviews_splits(rows, records)
    again, _ = reviews.build_reviews_splits(list(reversed(rows)), records)
    other, _ = reviews.build_reviews_splits(rows, records, seed="another-seed")
    assert first.paper600 == again.paper600 and first.pool == again.pool
    assert first.paper600 != other.paper600


def test_too_few_abusive_cases_merge_abusive_and_promotional_into_remove_other():
    rows, records = synthetic_pool({**COUNTS, "abusive": 12})
    splits, report = reviews.build_reviews_splits(rows, records)
    assert report["merged"] and report["labels"] == list(reviews.MERGED_LABELS)
    assert {item.reference_label for item in splits.items.values()} <= set(reviews.MERGED_LABELS)
    assert report["gold_by_raw_label"]["abusive"] == 12
    forced, forced_report = reviews.build_reviews_splits(rows, records, merge=False)
    assert not forced_report["merged"] and "abusive" in forced_report["gold_by_label"]


def test_a_pool_too_small_for_held_out_and_stream_is_refused():
    rows, records = synthetic_pool({"approve": 500, "abusive": 100})
    with pytest.raises(ValueError, match="only 600 SME-labeled items"):
        reviews.build_reviews_splits(rows, records)


def test_the_natural_mix_is_reported_separately_from_the_enriched_mix():
    rows, records = synthetic_pool(COUNTS)
    natural = {"n1": sme.SmeRecord("n1", "approve", "R11", "x", None, "accepted"),
               "n2": sme.SmeRecord("n2", None, None, None, None, "ambiguous")}
    _, report = reviews.build_reviews_splits(rows, records, natural_records=natural, natural_size=300)
    assert report["natural_mix"] == {"n": 300, "sme_labeled": 1, "by_label": {"approve": 1},
                                      "no_label": 299, "uncached": 298,
                                      "ambiguous": 1, "rejected": 0}
    assert reviews.build_reviews_splits(rows, records)[1]["natural_mix"] is None


def test_an_empty_natural_sme_cache_counts_every_absent_record_as_no_label():
    rows, records = synthetic_pool(COUNTS)
    _, report = reviews.build_reviews_splits(rows, records, natural_records={}, natural_size=300)

    assert report["natural_mix"] == {"n": 300, "sme_labeled": 0, "by_label": {},
                                      "no_label": 300, "uncached": 300,
                                      "ambiguous": 0, "rejected": 0}


def write_cache(path, records, *, model=MODEL, policy=POLICY):
    cache = sme.SmeCache(path)
    for record in records.values():
        cache.put(sme.cache_key(record.item_id, sme.policy_sha(policy), model), record, model=model,
                  policy_sha256=sme.policy_sha(policy), pass_tag="primary")
    cache.db.close()


def test_the_sme_cache_is_read_without_being_changed(tmp_path):
    rows, records = synthetic_pool({"approve": 3, "abusive": 2}, (("ambiguous", 1),))
    path = tmp_path / "sme.sqlite"
    write_cache(path, records)
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    read, identity = reviews.read_sme_records(path, [r["id"] for r in rows] + ["not-labeled"])
    assert identity == {"sme_model": MODEL, "policy_sha256": SHA}
    assert read == records and (path.read_bytes(), os.stat(path).st_mtime_ns) == before
    with pytest.raises(FileNotFoundError):
        reviews.read_sme_records(tmp_path / "missing.sqlite", [])


def test_records_written_by_the_fake_sme_labeling_run_are_read_back(tmp_path):
    items = [sme.SmeItem("a1", "This charger is crap and stopped working after a week of light use."),
             sme.SmeItem("a2", "My kids love this puzzle, the pieces are thick and the colors bright."),
             sme.SmeItem("a3", "Okay.")]
    cache = sme.SmeCache(tmp_path / "sme.sqlite")
    labeled, _ = sme.label_items(items, sme.fake_sme_completion, model=MODEL, policy=POLICY, cache=cache)
    cache.db.close()
    read, _ = reviews.read_sme_records(tmp_path / "sme.sqlite", [i.item_id for i in items])
    assert read == labeled and read["a3"].label is None


def test_two_sme_models_in_one_cache_must_be_named(tmp_path):
    _, records = synthetic_pool({"approve": 2})
    path = tmp_path / "sme.sqlite"
    write_cache(path, records)
    write_cache(path, records, model="gpt-6-luna")
    with pytest.raises(ValueError, match="found 2"):
        reviews.read_sme_records(path, list(records))
    assert reviews.read_sme_records(path, list(records), model="gpt-6-luna")[1]["sme_model"] == "gpt-6-luna"


def test_the_reviews_corpora_are_registered_with_s_the_policy_path_and_a_300_item_final():
    corpus = get_corpus("reviews")
    assert {"reviews", "reviews-merged"} <= set(CORPORA)
    assert corpus.labels == reviews.REVIEWS_LABELS and corpus.multiclass_metrics and corpus.final_size == 300
    assert corpus.instructions == (REPO / "studies/amazon_reviews/S.txt").read_text(encoding="utf-8").strip()
    assert corpus.seed_score()["instructions"] == corpus.instructions
    assert corpus.knn_share_feature == "knn.share.approve" and corpus.fewshot_dropped_label == "price_availability"
    assert tuple(dict(corpus.fake_jev_cues)) == corpus.labels
    merged = get_corpus("reviews-merged")
    assert merged.labels == reviews.MERGED_LABELS and "remove_other (abusive or promotional)" in merged.instructions
    assert tuple(dict(merged.fake_jev_cues)) == merged.labels
    assert reviews.POLICY_PATH == REPO / "var/policy/amazon_reviews_F.txt"


def test_the_private_policy_is_read_only_when_the_ceiling_or_stakeholder_asks_for_it(tmp_path, monkeypatch):
    monkeypatch.setattr(reviews, "POLICY_PATH", tmp_path / "absent.txt")
    corpus = get_corpus("reviews")
    with pytest.raises(FileNotFoundError, match="only the ceiling and stakeholder need it"):
        corpus.ceiling_score()
    (tmp_path / "F.txt").write_text(POLICY, encoding="utf-8")
    monkeypatch.setattr(reviews, "POLICY_PATH", tmp_path / "F.txt")
    assert corpus.ceiling_score()["instructions"] == corpus.stakeholder_guideline() == POLICY.strip()
    assert get_corpus("reviews-merged").stakeholder_guideline().endswith("as remove_other.")
    assert reviews.check_policy_matches(SHA) is True and reviews.check_policy_matches("other") is False


def test_the_corpus_refuses_data_whose_merge_decision_is_the_other_one(monkeypatch):
    rows, records = synthetic_pool({**COUNTS, "abusive": 5})
    monkeypatch.setattr(reviews, "load_reviews_splits", lambda **_: reviews.build_reviews_splits(rows, records))
    with pytest.raises(ValueError, match="use --corpus reviews-merged"):
        get_corpus("reviews").load(Path("."), dev_size=100)
    assert len(get_corpus("reviews-merged").load(Path("."), dev_size=100).paper600) == 300


def load_screen():
    spec = importlib.util.spec_from_file_location("rubric_screen_under_test", REPO / "scripts" / "rubric_screen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_screen_dry_run_counts_the_held_out_reviews_and_their_requests_without_calling_jev(tmp_path, monkeypatch, capsys):
    rows, records = synthetic_pool(COUNTS, NO_GOLD)
    built = reviews.build_reviews_splits(rows, records)
    built[1].update({"sme_model": MODEL, "policy_sha256": SHA})
    (tmp_path / "F.txt").write_text(POLICY, encoding="utf-8")
    monkeypatch.setattr(reviews, "POLICY_PATH", tmp_path / "F.txt")
    screen = load_screen()
    monkeypatch.setattr(screen, "_reviews", lambda: built)
    assert screen.instructions("reviews", "S") == get_corpus("reviews").instructions
    assert screen.instructions("reviews", "F") == POLICY.strip()
    assert [r[0] for r in screen.heldout("reviews")] == list(built[0].paper600)
    assert {r[0] for r in screen.load("reviews")} == set(built[0].pool)
    args = type("Args", (), {"datasets": "reviews", "split": "heldout", "conditions": "S,F", "n": 120})()
    screen.dry_run(args)
    out = capsys.readouterr().out
    assert "reviews screen n=300" in out and "jev_requests_upper_bound=600" in out
    assert "'ambiguous': 30" in out and "synthetic review" not in out
