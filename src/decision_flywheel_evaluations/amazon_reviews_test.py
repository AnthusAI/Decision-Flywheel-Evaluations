import json
from collections import Counter

import pytest

from decision_flywheel_evaluations import amazon_reviews as ar

TS_2014 = 1_400_000_000_000   # 2014-05-13 UTC, in milliseconds like the source
TS_2019 = 1_560_000_000_000

LOW = ["This charger stopped working after two days and now my phone will not charge at all, very disappointed honestly",
       "The blender lid cracked the first time I used it and the motor smells like burning plastic every single time"]
MID = ["The puzzle pieces are fine but a few did not fit very well together, it is an okay product for the price paid"]
HIGH = ["My kids love this board game and we play it every weekend together, the pieces are sturdy and colorful too"]
KW = ["The seller never answered my messages and the package arrived crushed so I asked for a refund right away today",
      "The maker sent this case cheaply in exchange for writing up my thoughts and I think it works well for my phone"]


def synthetic(n=50):
    """About fifty fake raw lines: mixed ratings, years, lengths and keyword hits. No real reviews."""
    rows = []
    for k in range(n):
        if k % 10 == 9:
            text, rating = "Too short to keep.", 1.0
        elif k % 7 == 6:
            text, rating = KW[k % 2] + f" note {k}", float(1 + k % 5)
        elif k % 3 == 0:
            text, rating = LOW[k % 2] + f" note {k}", float(1 + k % 2)
        elif k % 3 == 1:
            text, rating = MID[0] + f" note {k}", 3.0
        else:
            text, rating = HIGH[0] + f" note {k}", float(4 + k % 2)
        rows.append({"rating": rating, "title": "t", "text": text, "asin": f"A{k}", "parent_asin": f"P{k}",
                     "user_id": f"U{k}", "timestamp": TS_2019 if k % 11 == 10 else TS_2014 + k,
                     "helpful_vote": 0, "verified_purchase": True})
    return rows


def as_chunk(rows):
    body = "\n".join(json.dumps(r) for r in rows)
    return ('{"partial": tru\n' + body + '\n{"rating": 5.0, "te').encode()


def reviews(n=50):
    return ar.parse_chunk(as_chunk(synthetic(n)), category="Toys_and_Games", source_file="toys.part0.jsonl")


def test_a_chunk_drops_its_partial_first_and_last_lines():
    parsed = reviews()
    assert len(parsed) == 50
    assert all(r.category == "Toys_and_Games" and r.source_file == "toys.part0.jsonl" for r in parsed)


def test_review_ids_are_stable_unique_and_free_of_text():
    first, second = reviews(), reviews()
    assert [r.review_id for r in first] == [r.review_id for r in second]
    assert len({r.review_id for r in first}) == len(first)
    assert all(r.review_id.startswith("amz-") and "charger" not in r.review_id for r in first)


def test_eligibility_keeps_2013_to_2016_and_15_to_90_words():
    parsed = reviews()
    kept = [r for r in parsed if ar.eligible(r)]
    assert all(2013 <= r.year <= 2016 and 15 <= len(r.text.split()) <= 90 for r in kept)
    assert any(r.year == 2019 for r in parsed) and any(len(r.text.split()) < 15 for r in parsed)
    assert len(kept) < len(parsed)


def test_keyword_cues_cover_the_planned_terms():
    for text in ("the seller was slow", "shipping took ages", "I want a refund", "had to return it",
                 "the package was wet", "cost $20", "a fair price", "see http://x.example", "sent free in exchange for a write-up",
                 "got a discount", "this is crap"):
        assert ar.keyword_hit(text), text
    assert not ar.keyword_hit("my kids love this board game")


def test_rating_bands_split_low_mid_high():
    assert [ar.rating_band(x) for x in (1.0, 2.0, 3.0, 4.0, 5.0)] == ["low", "low", "mid", "high", "high"]


def test_the_pool_follows_the_rating_shares_and_oversamples_keywords():
    kept = [r for r in reviews() if ar.eligible(r)]
    pool = ar.build_pool(kept, size=10, seed=7)
    bands = Counter(p.stratum.split("-")[0] for p in pool)
    assert sum(bands.values()) == 10
    assert bands["low"] == 6 and bands["mid"] in (1, 2) and bands["high"] in (2, 3)
    assert any(p.stratum.endswith("-keyword") for p in pool)


def test_the_pool_is_deterministic_for_a_seed_and_changes_with_it():
    kept = [r for r in reviews() if ar.eligible(r)]
    ids = lambda seed: [p.review.review_id for p in ar.build_pool(kept, size=10, seed=seed)]
    assert ids(7) == ids(7)
    assert ids(7) != ids(8)


def test_a_short_stratum_is_filled_from_the_rest_and_reported():
    kept = [r for r in reviews() if ar.eligible(r)]
    pool = ar.build_pool(kept, size=len(kept), seed=1)
    assert len(pool) == len(kept) and len({p.review.review_id for p in pool}) == len(kept)


def test_the_natural_sample_is_random_and_disjoint_from_the_pool():
    kept = [r for r in reviews() if ar.eligible(r)]
    pool = ar.build_pool(kept, size=10, seed=7)
    natural = ar.natural_sample(kept, exclude={p.review.review_id for p in pool}, size=5, seed=7)
    assert len(natural) == 5
    assert not {r.review_id for r in natural} & {p.review.review_id for p in pool}


def test_duplicate_texts_are_kept_once():
    rows = synthetic(8)
    rows.append(dict(rows[0], user_id="someone-else"))
    parsed = ar.dedupe(ar.parse_chunk(as_chunk(rows), category="c", source_file="f"))
    assert len(parsed) == 8


def test_the_manifest_carries_ids_and_hashes_but_no_text():
    kept = [r for r in reviews() if ar.eligible(r)]
    pool = ar.build_pool(kept, size=10, seed=7)
    natural = ar.natural_sample(kept, exclude={p.review.review_id for p in pool}, size=5, seed=7)
    downloads = [{"file": "toys.part0.jsonl", "bytes": 10, "sha256": "0" * 64}]
    manifest = ar.manifest(pool, natural, downloads=downloads, seed=7, eligible_count=len(kept))
    dumped = json.dumps(manifest)
    for r in kept:
        assert r.text not in dumped and r.text.split()[0] + " " + r.text.split()[1] not in dumped
    item = manifest["pool"][0]
    assert set(item) == {"id", "category", "rating", "stratum", "source_file", "row_sha256", "text_sha256"}
    assert manifest["counts"]["pool"] == 10 and manifest["counts"]["natural"] == 5


def test_pool_files_hold_text_for_var_only():
    kept = [r for r in reviews() if ar.eligible(r)]
    pool = ar.build_pool(kept, size=4, seed=7)
    line = json.loads(ar.pool_lines(pool)[0])
    assert set(line) == {"id", "category", "rating", "stratum", "text"}


def test_chunk_plan_spreads_ranges_through_the_file():
    plan = ar.chunk_plan(1_000_000_000, chunks=4, chunk_bytes=1000)
    assert len(plan) == 4 and plan[0][0] > 0 and plan[-1][1] < 1_000_000_000
    assert all(end - start + 1 == 1000 for start, end in plan)


def test_download_refuses_without_confirm_and_never_touches_the_network(tmp_path):
    with pytest.raises(ar.DownloadRefused):
        ar.download(tmp_path, confirm=False, fetch=lambda *a: pytest.fail("fetched"))


def test_download_records_sizes_and_hashes_with_a_fake_fetch(tmp_path):
    body = as_chunk(synthetic(5))
    record = ar.download(tmp_path, confirm=True, fetch=lambda url, start, end: body,
                         sizes={c: 10_000_000 for c in ar.CATEGORIES}, chunks=2, chunk_bytes=100)
    assert len(record) == 2 * len(ar.CATEGORIES)
    assert all(r["bytes"] == len(body) and len(r["sha256"]) == 64 for r in record)
    assert (tmp_path / record[0]["file"]).read_bytes() == body


def test_the_dumped_manifest_round_trips_as_json():
    kept = [r for r in reviews() if ar.eligible(r)]
    pool = ar.build_pool(kept, size=4, seed=7)
    result = ar.manifest(pool, [], downloads=[], seed=7, eligible_count=len(kept))
    assert json.loads(ar.dump_manifest(result)) == json.loads(json.dumps(result))
