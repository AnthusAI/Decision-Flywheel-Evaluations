import json

import pytest

from decision_flywheel_evaluations import unified_sme as sme
from decision_flywheel_evaluations.unified_spend import CeilingExhausted, SpendLedger

# A synthetic stand-in for the private policy F: same rule ids and labels, invented wording.
POLICY = "\n".join([
    "Toy policy for specs only. Labels in order: abusive, promotional, seller_shipping, price_availability, approve.",
    "R1 (abusive). Swear words of any strength anywhere in the text count against the writer here.",
    "R2 (abusive). Calling people names or threatening them is out of bounds for this toy board.",
    "R3 (promotional). Links, emails and phone numbers are never permitted inside this pretend forum.",
    "R4 (promotional). Free or cheaper goods handed over for writing the text are disclosed sponsorship.",
    "R5 (promotional). Telling readers to shop at some other named brand or store is advertising.",
    "R6 (seller_shipping). Mostly about the merchant, refunds, the order or customer care desks.",
    "R7 (seller_shipping). Mostly about the courier, the box, transit damage or delivery timing.",
    "R8 (price_availability). Store or date specific pricing, coupons or sales events are mentioned.",
    "R9 (price_availability). Stock levels at a particular shop or listing are commented upon.",
    "R10 (approve). Judging worth for money with a figure is fine and stays up on the site.",
    "R11 (approve). Anything that talks about the item itself, however grumpy, stays up.",
    "R12 (approve). A brief aside about the parcel inside an item review also stays up.",
])
ITEMS = [
    sme.SmeItem("r1", "This charger is crap and stopped working after a week of light use at home."),
    sme.SmeItem("r2", "Visit http://example.test for a better deal on cases like this one, trust me on it."),
    sme.SmeItem("r3", "The maker gave me this blender in exchange for writing about it, it blends well."),
    sme.SmeItem("r4", "The seller never replied and my refund took a month, the package was also late."),
    sme.SmeItem("r5", "It was ten dollars cheaper at another store last week, check before you buy here."),
    sme.SmeItem("r6", "My kids love this puzzle, the pieces are thick and the colors are bright and fun."),
    sme.SmeItem("r7", "Great little lamp for the price, bright enough for reading and it arrived fast."),
    sme.SmeItem("r8", "Okay."),
]


def offline_cache(tmp_path):
    return sme.SmeCache(tmp_path / "sme.sqlite")


def test_the_policy_rules_are_read_from_the_policy_text():
    rules = sme.policy_rules(POLICY)
    assert rules["R1"] == "abusive" and rules["R7"] == "seller_shipping" and rules["R12"] == "approve"
    assert len(rules) == 12


def test_the_prompt_shows_the_policy_and_reviews_but_never_a_classifier_verdict():
    messages = sme.sme_messages(ITEMS[:3], POLICY)
    assert POLICY in messages[0]["content"]
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert [p["id"] for p in payload] == ["r1", "r2", "r3"]
    assert "verdict" not in json.dumps(messages).lower() and "predicted" not in json.dumps(messages).lower()


def test_the_fake_sme_follows_its_keyword_rules():
    records, _ = sme.label_items(ITEMS, sme.fake_sme_completion, model=sme.FAKE_MODEL, policy=POLICY,
                                 cache=sme.SmeCache(None))
    got = {i: (r.label, r.rule_id) for i, r in records.items()}
    assert got["r1"] == ("abusive", "R1")
    assert got["r2"] == ("promotional", "R3")
    assert got["r3"] == ("promotional", "R4")
    assert got["r4"][0] == "seller_shipping"
    assert got["r5"] == ("price_availability", "R8")
    assert got["r6"] == ("approve", "R11")
    assert got["r7"][0] == "approve"
    assert records["r8"].status == "ambiguous" and records["r8"].label is None


def test_labeling_is_batched_ten_reviews_per_call():
    calls = []

    def counting(messages):
        calls.append(len(json.loads(messages[1]["content"].split("\n", 1)[1])))
        return sme.fake_sme_completion(messages)

    items = [sme.SmeItem(f"x{k}", ITEMS[k % 7].text) for k in range(23)]
    _, report = sme.label_items(items, counting, model=sme.FAKE_MODEL, policy=POLICY, cache=sme.SmeCache(None))
    assert calls == [10, 10, 3] and report["new_calls"] == 3


def test_a_second_run_is_served_from_the_cache_without_calls(tmp_path):
    sme.label_items(ITEMS, sme.fake_sme_completion, model=sme.FAKE_MODEL, policy=POLICY, cache=offline_cache(tmp_path))
    records, report = sme.label_items(ITEMS, sme.refusing_completion, model=sme.FAKE_MODEL, policy=POLICY,
                                      cache=offline_cache(tmp_path))
    assert report["new_calls"] == 0 and report["cache_hits"] == len(ITEMS)
    assert records["r1"].label == "abusive"


def test_the_cache_key_changes_with_policy_prompt_version_model_and_pass():
    base = sme.cache_key("r1", "p" * 64, "m")
    assert base != sme.cache_key("r1", "q" * 64, "m")
    assert base != sme.cache_key("r1", "p" * 64, "m2")
    assert base != sme.cache_key("r1", "p" * 64, "m", pass_tag="second")
    assert base == sme.cache_key("r1", "p" * 64, "m")


def test_a_replay_miss_refuses_instead_of_calling(tmp_path):
    with pytest.raises(sme.SmeReplayMiss):
        sme.label_items(ITEMS[:1], sme.refusing_completion, model=sme.FAKE_MODEL, policy=POLICY,
                        cache=offline_cache(tmp_path))


def answer(**fields):
    base = {"id": "r6", "label": "approve", "rule_id": "R11", "reason": "Talks only about how sturdy the pieces are."}
    return {**base, **fields}


def test_a_well_formed_decision_is_accepted():
    record = sme.validate(answer(quote="the pieces are thick"), ITEMS[5], POLICY)
    assert record.status == "accepted" and record.quote == "the pieces are thick"


def test_a_label_outside_the_set_is_rejected():
    assert sme.validate(answer(label="spam"), ITEMS[5], POLICY).status == "rejected"


def test_an_unknown_or_mismatched_rule_keeps_only_the_label():
    for rule in ("R99", "R1"):
        record = sme.validate(answer(rule_id=rule), ITEMS[5], POLICY)
        assert record.status == "label_only" and record.label == "approve" and record.reason is None


def test_a_quote_not_found_in_the_review_is_dropped():
    record = sme.validate(answer(quote="pieces that glow in the dark"), ITEMS[5], POLICY)
    assert record.status == "accepted" and record.quote is None and record.problem == "quote-not-verbatim"


def test_a_reason_that_recites_the_policy_keeps_only_the_label():
    recited = "Anything that talks about the item itself, however grumpy, stays up."
    record = sme.validate(answer(reason=recited), ITEMS[5], POLICY)
    assert record.status == "label_only" and record.problem == "reason-recites-policy"


def test_an_overlong_reason_keeps_only_the_label():
    record = sme.validate(answer(reason="word " * 26), ITEMS[5], POLICY)
    assert record.status == "label_only" and record.problem == "reason-too-long"


def test_an_ambiguous_or_unexplainable_answer_carries_no_label():
    for label in ("ambiguous", "UNEXPLAINABLE"):
        record = sme.validate(answer(label=label, rule_id=None), ITEMS[5], POLICY)
        assert record.status == "ambiguous" and record.label is None


def test_items_missing_from_a_reply_are_reported_and_not_cached(tmp_path):
    def partial(messages):
        reply = json.loads(sme.fake_sme_completion(messages))
        reply["decisions"] = reply["decisions"][1:]
        return json.dumps(reply)

    records, report = sme.label_items(ITEMS[:3], partial, model=sme.FAKE_MODEL, policy=POLICY,
                                      cache=offline_cache(tmp_path))
    assert "r1" not in records and report["missing"] == 1
    assert offline_cache(tmp_path).get(sme.cache_key("r1", sme.policy_sha(POLICY), sme.FAKE_MODEL)) is None


def test_a_malformed_reply_marks_the_whole_batch_missing(tmp_path):
    records, report = sme.label_items(ITEMS[:3], lambda m: "not json", model=sme.FAKE_MODEL, policy=POLICY,
                                      cache=offline_cache(tmp_path))
    assert records == {} and report["missing"] == 3 and report["malformed_replies"] == 1


def test_the_ledger_is_reserved_before_each_call_and_its_cap_refuses(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.json", 5, max_new=1)
    items = [sme.SmeItem(f"x{k}", ITEMS[5].text) for k in range(15)]
    with pytest.raises(CeilingExhausted):
        sme.label_items(items, sme.fake_sme_completion, model=sme.FAKE_MODEL, policy=POLICY,
                        cache=offline_cache(tmp_path), ledger=ledger)
    assert ledger.used == 1


def test_a_live_completion_that_switches_model_retries_the_batch_under_the_fallback(tmp_path):
    class Switching:
        model = "primary"

        def __call__(self, messages):
            if self.model == "primary":
                self.model = "fallback"
                raise sme.ModelUnavailable("primary not found")
            return sme.fake_sme_completion(messages)

    ledger = SpendLedger(tmp_path / "ledger.json", 10, max_new=10)
    records, report = sme.label_items(ITEMS[:2], Switching(), model="primary", policy=POLICY,
                                      cache=offline_cache(tmp_path), ledger=ledger)
    assert report["model"] == "fallback" and report["new_calls"] == 1 and ledger.used == 2
    assert offline_cache(tmp_path).get(sme.cache_key("r1", sme.policy_sha(POLICY), "fallback")) is not None


def test_the_report_is_text_free(tmp_path):
    _, report = sme.label_items(ITEMS, sme.fake_sme_completion, model=sme.FAKE_MODEL, policy=POLICY,
                                cache=offline_cache(tmp_path))
    dumped = json.dumps(report)
    assert all(item.text[:20] not in dumped for item in ITEMS)
    assert report["by_label"]["abusive"] == 1 and report["policy_sha256"] == sme.policy_sha(POLICY)


def test_feedback_says_should_be_when_wrong_and_correct_when_right():
    record = sme.SmeRecord("r4", "seller_shipping", "R6", "Mostly a complaint about the refund delay.", None, "accepted")
    assert sme.feedback("approve", record) == "should be seller_shipping because mostly a complaint about the refund delay."
    assert sme.feedback("seller_shipping", record) == "correct: Mostly a complaint about the refund delay."
    bare = sme.SmeRecord("r4", "seller_shipping", None, None, None, "label_only")
    assert sme.feedback("approve", bare) == "should be seller_shipping."
    assert sme.feedback("approve", sme.SmeRecord("r8", None, None, None, None, "ambiguous")) is None


def test_feedback_caps_how_often_one_rule_is_explained():
    records = {f"x{k}": sme.SmeRecord(f"x{k}", "approve", "R11", f"Reason {k} about the toy.", None, "accepted")
               for k in range(5)}
    comments = sme.build_feedback({k: "abusive" for k in records}, records, max_rule_repeats=3)
    explained = [c for c in comments.values() if "because" in c]
    assert len(explained) == 3 and len(comments) == 5


def test_agreement_reports_raw_agreement_and_kappa():
    a = {k: sme.SmeRecord(k, lab, None, None, None, "label_only") for k, lab in
         zip("abcdef", ["approve", "approve", "abusive", "promotional", "approve", "seller_shipping"])}
    same = sme.agreement(a, a)
    assert same["n"] == 6 and same["agreement"] == 1.0 and same["kappa"] == 1.0
    b = dict(a, f=sme.SmeRecord("f", "approve", None, None, None, "label_only"))
    diff = sme.agreement(a, b)
    assert diff["agreement"] == pytest.approx(5 / 6, abs=1e-4) and 0 < diff["kappa"] < 1


def test_the_agreement_subset_is_seeded():
    ids = [f"x{k}" for k in range(50)]
    assert sme.agreement_subset(ids, 10, seed=1) == sme.agreement_subset(ids, 10, seed=1)
    assert len(sme.agreement_subset(ids, 10, seed=1)) == 10


def test_the_cli_refuses_live_without_confirm_cap_and_ledger(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps({"id": i.item_id, "text": i.text}) + "\n" for i in ITEMS))
    policy = tmp_path / "F.txt"
    policy.write_text(POLICY)
    base = ["label", "--pool", str(pool), "--policy", str(policy), "--cache", str(tmp_path / "c.sqlite"),
            "--report", str(tmp_path / "r.json"), "--labels-out", str(tmp_path / "l.jsonl")]
    for extra in (["--live"], ["--live", "--confirm"], ["--live", "--confirm", "--ledger", str(tmp_path / "l.json")],
                  ["--live", "--confirm", "--ledger", str(tmp_path / "l.json"), "--max-calls", "0"],
                  ["--confirm"], ["--max-calls", "3"]):
        with pytest.raises(sme.UsageError):
            sme.main(base + extra)


def test_the_cli_labels_offline_with_the_fake_and_writes_text_free_reports(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps({"id": i.item_id, "text": i.text}) + "\n" for i in ITEMS))
    policy = tmp_path / "F.txt"
    policy.write_text(POLICY)
    assert sme.main(["label", "--pool", str(pool), "--policy", str(policy), "--cache", str(tmp_path / "c.sqlite"),
                     "--report", str(tmp_path / "r.json"), "--labels-out", str(tmp_path / "l.jsonl")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["model"] == sme.FAKE_MODEL and report["items"] == len(ITEMS)
    labels = [json.loads(line) for line in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert {row["id"] for row in labels} == {i.item_id for i in ITEMS}
