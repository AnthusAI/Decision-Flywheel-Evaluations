import json

import pytest

from .unified_labeler import (FAKE_MODEL, LABELER_CALL_CEILING, CommentCache, LabelerItem, LabelerReplayMiss,
                              fake_completion, generate, labeler_prompt, read_comments, refusing_completion,
                              upper_bound_calls, wants_comment, write_comments)
from .unified_spend import CeilingExhausted, SpendLedger

ITEMS = [LabelerItem(f"i{n:03d}", text, label) for n, (text, label) in enumerate([
    ("Our team practiced the new passing drill before the weekend match.", "positive"),
    ("The quarterly report is due to the department manager by Friday.", "negative"),
    ("I loved every minute of the concert last night.", "positive"),
    ("The service was slow and the food arrived cold.", "negative"),
    ("Staff meeting moved to the small conference room this week.", "negative"),
    ("Golf season opens at the club on Saturday morning.", "positive"),
] * 4)]
ITEMS = [LabelerItem(f"{item.item_id}-{n}", item.text + f" ({n})", item.label) for n, item in enumerate(ITEMS)]


class CountingCompletion:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return fake_completion(prompt)


def _generate(tmp_path, complete=None, **kwargs):
    cache = CommentCache(tmp_path / "cache.jsonl")
    return generate(ITEMS, complete or CountingCompletion(), model=kwargs.pop("model", FAKE_MODEL),
                    cache=cache, seed=1, **kwargs)


def test_some_labels_get_no_comment_and_the_rest_a_terse_one(tmp_path):
    comments, report = _generate(tmp_path)
    assert 0 < len(comments) < len(ITEMS)
    assert report["skipped"] == sum(not wants_comment(i.item_id, 1) for i in ITEMS)
    assert all(1 <= len(c.split()) <= 20 for c in comments.values())


def test_the_fake_labeler_follows_the_hidden_convention_without_stating_it(tmp_path):
    comments, _ = _generate(tmp_path)
    by_id = {i.item_id: i for i in ITEMS}
    sporty = [c for i, c in comments.items() if "match" in by_id[i].text or "Golf" in by_id[i].text]
    assert sporty and all("sport" in c.lower() or "game" in c.lower() for c in sporty)
    assert not any("convention" in c.lower() for c in comments.values())


def test_the_prompt_carries_the_text_the_label_and_the_hidden_convention_but_no_prediction():
    prompt = labeler_prompt(ITEMS[1])
    system, user = (m["content"] for m in prompt.messages())
    assert "sports" in system and "workplace" in system and "20 words" in system
    assert ITEMS[1].text in user and ITEMS[1].label in user
    assert "predict" not in (system + user).lower()


def test_two_runs_give_identical_comments(tmp_path):
    first, _ = _generate(tmp_path / "a")
    second, _ = _generate(tmp_path / "b")
    assert first == second


def test_a_second_run_replays_from_the_cache_without_calling_the_model(tmp_path):
    first, report = _generate(tmp_path)
    assert report["new_calls"] == len(first) and report["cache_hits"] == 0
    again, replay = _generate(tmp_path, complete=refusing_completion)
    assert again == first and replay["new_calls"] == 0 and replay["cache_hits"] == len(first)


def test_a_replay_refuses_an_uncached_comment(tmp_path):
    with pytest.raises(LabelerReplayMiss):
        _generate(tmp_path, complete=refusing_completion)


def test_the_cache_is_keyed_by_model_so_another_model_is_asked_again(tmp_path):
    _generate(tmp_path)
    counting = CountingCompletion()
    _, report = _generate(tmp_path, complete=counting, model="another-model")
    assert report["cache_hits"] == 0 and len(counting.prompts) == report["new_calls"] > 0


def test_the_cache_and_the_report_hold_no_dataset_text(tmp_path):
    _, report = _generate(tmp_path)
    blobs = [(tmp_path / "cache.jsonl").read_text(), json.dumps(report)]
    for item in ITEMS:
        for blob in blobs:
            assert item.text not in blob
    assert set(report) >= {"items", "skipped", "commented", "new_calls", "cache_hits", "mentions_topic",
                           "comments_sha256", "prompt_version", "model"}


def test_a_comment_that_quotes_its_text_is_dropped(tmp_path):
    comments, report = _generate(tmp_path, complete=lambda prompt: f"See: {prompt.item.text}")
    assert comments == {} and report["dropped"] == report["new_calls"] > 0


def test_every_call_is_counted_against_the_ledger_and_refused_past_its_cap(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.json", LABELER_CALL_CEILING, max_new=3)
    with pytest.raises(CeilingExhausted):
        _generate(tmp_path, ledger=ledger)
    assert ledger.new_attempts == 3
    assert len(CommentCache(tmp_path / "cache.jsonl").rows()) == 3   # the three paid answers are kept


def test_the_upper_bound_is_the_number_of_items_that_want_a_comment(tmp_path):
    comments, report = _generate(tmp_path)
    assert upper_bound_calls(ITEMS, seed=1) == report["new_calls"] == len(comments) + report["dropped"]


def test_the_comments_file_round_trips_into_the_harness_format(tmp_path):
    comments, _ = _generate(tmp_path)
    write_comments(tmp_path / "comments.jsonl", comments)
    assert read_comments(tmp_path / "comments.jsonl") == comments
    from .unified_cli import _comments
    assert _comments(tmp_path / "comments.jsonl") == comments


# ---- the rubric stakeholder (FOMC style, rubric-dataset step R4) ------------------------------

from .unified_labeler import (STAKEHOLDER_FAKE_MODEL, UNEXPLAINABLE, RubricStakeholder, StakeholderCase,  # noqa: E402
                              explain_errors, fake_stakeholder_completion, noisy_comments, shuffled_comments,
                              stakeholder_prompt, validate_explanation)

LABELS3 = ("dovish", "hawkish", "neutral")
GUIDE = """One-line rubric: hawkish, dovish or neutral.

Label definitions. Dovish sentences were any sentence that indicates future monetary policy easing. Hawkish sentences were any sentence that would indicate a future monetary policy tightening.

Economic Status
 Dovish: when inflation decreases, when unemployment increases, when economic growth is projected as low
 Hawkish: when inflation increases, when unemployment decreases, when economic growth is projected high
Energy/House Prices
 Dovish: when oil/energy prices decrease, when house prices decrease
 Hawkish: when oil/energy prices increase, when house prices increase
 Neutral: N/A"""
CASES = [StakeholderCase(f"fomc-train-{n}", text, verdict, gold) for n, (text, verdict, gold) in enumerate([
    ("Inflation decreases were expected to continue next year.", "neutral", "dovish"),
    ("House prices increase further in most regions.", "neutral", "hawkish"),
    ("Members noted that unemployment increases in several districts.", "hawkish", "dovish"),
    ("Oil prices decrease as supply recovers.", "hawkish", "dovish"),
    ("The weather in the district was unremarkable.", "dovish", "hawkish"),
])]
GOOD = json.dumps({"explanation": "should be dovish because the guideline counts a fall in inflation as a sign of easing",
                   "quote": "Dovish: when inflation decreases, when unemployment increases"})


def _explain(tmp_path, complete=fake_stakeholder_completion, cases=CASES, **kwargs):
    cache = CommentCache(tmp_path / "stakeholder.jsonl")
    return explain_errors(cases, complete, model=kwargs.pop("model", STAKEHOLDER_FAKE_MODEL), cache=cache,
                          guideline=GUIDE, labels=LABELS3, **kwargs)


def test_the_stakeholder_prompt_carries_the_item_the_verdict_the_gold_label_and_the_full_guideline():
    system, user = (m["content"] for m in stakeholder_prompt(CASES[0], GUIDE, LABELS3).messages())
    assert GUIDE in system and "30 words" in system and UNEXPLAINABLE in system
    assert CASES[0].text in user and "neutral" in user and "dovish" in user


def test_a_verbatim_quote_is_accepted_even_with_different_whitespace():
    spaced = json.dumps({"explanation": "should be dovish because the guideline counts a fall in inflation as a sign of easing",
                         "quote": "Dovish:   when inflation\ndecreases, when unemployment increases"})
    assert validate_explanation(GOOD, CASES[0], GUIDE, LABELS3).status == "accepted"
    assert validate_explanation(spaced, CASES[0], GUIDE, LABELS3).status == "accepted"


def test_a_paraphrased_quote_is_rejected():
    raw = json.dumps({"explanation": "should be dovish because the guideline counts a fall in inflation as easing",
                      "quote": "Dovish: when inflation goes down"})
    assert validate_explanation(raw, CASES[0], GUIDE, LABELS3).reason == "quote-not-verbatim"


def test_a_quote_longer_than_30_words_is_rejected():
    long_quote = " ".join(GUIDE.split()[:31])
    raw = json.dumps({"explanation": "should be dovish because the guideline counts a fall in inflation as easing",
                      "quote": long_quote})
    assert validate_explanation(raw, CASES[0], GUIDE, LABELS3).reason == "quote-too-long"


def test_an_explanation_that_only_restates_the_label_is_rejected():
    raw = json.dumps({"explanation": "should be dovish because it is dovish",
                      "quote": "Dovish: when inflation decreases"})
    assert validate_explanation(raw, CASES[0], GUIDE, LABELS3).reason == "restates-label"


def test_an_explanation_whose_rule_is_not_in_the_quoted_guideline_is_rejected_as_invented():
    raw = json.dumps({"explanation": "should be dovish because the speaker sounds worried about the harvest season",
                      "quote": "Dovish: when inflation decreases"})
    assert validate_explanation(raw, CASES[0], GUIDE, LABELS3).reason == "rule-not-in-quote"


def test_an_explanation_for_a_label_other_than_gold_is_rejected():
    assert validate_explanation(GOOD, CASES[1], GUIDE, LABELS3).reason == "not-should-be-gold"


def test_unexplainable_is_logged_as_label_noise_and_never_forwarded(tmp_path):
    comments, report = _explain(tmp_path, complete=lambda prompt: UNEXPLAINABLE)
    assert comments == {} and report["unexplainable"] == len(CASES)


def test_the_offline_fake_produces_valid_quote_backed_explanations(tmp_path):
    comments, report = _explain(tmp_path)
    assert report["accepted"] >= 3 and not report["rejected"]
    for item_id, comment in comments.items():
        gold = next(c.gold for c in CASES if c.item_id == item_id)
        assert comment.startswith(f"should be {gold} because") and "Guideline: \"" in comment


def test_at_most_the_cap_of_errors_is_explained_per_call(tmp_path):
    _, report = _explain(tmp_path, max_explained=2)
    assert report["new_calls"] == 2 and report["errors_seen"] == len(CASES)


def test_one_quoted_rule_is_forwarded_at_most_the_repeat_cap_per_round(tmp_path):
    comments, report = _explain(tmp_path, complete=lambda prompt: GOOD, cases=[c for c in CASES if c.gold == "dovish"] * 1,
                                max_rule_repeats=2)
    assert len(comments) == 2 and report["repeat_capped"] == 1


def test_a_replay_answers_from_the_text_free_cache_without_calling_the_model(tmp_path):
    first, _ = _explain(tmp_path)
    again, report = _explain(tmp_path, complete=refusing_completion)
    assert again == first and report["new_calls"] == 0 and report["cache_hits"] == len(CASES)
    blob = (tmp_path / "stakeholder.jsonl").read_text()
    assert not any(case.text in blob for case in CASES)


def test_the_cache_key_changes_with_the_verdict_so_a_new_mistake_is_asked_again(tmp_path):
    _explain(tmp_path)
    moved = [StakeholderCase(c.item_id, c.text, "hawkish" if c.verdict != "hawkish" else "neutral", c.gold) for c in CASES]
    _, report = _explain(tmp_path, cases=moved)
    assert report["cache_hits"] == 0 and report["new_calls"] == len(CASES)


def test_every_stakeholder_call_is_counted_against_the_ledger(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.json", LABELER_CALL_CEILING, max_new=2)
    with pytest.raises(CeilingExhausted):
        _explain(tmp_path, ledger=ledger)
    assert ledger.new_attempts == 2


def test_shuffled_explanations_never_attach_to_their_source_item_and_are_deterministic():
    comments = {f"i{n}": f"comment {n}" for n in range(7)}
    shuffled = shuffled_comments(comments, "seed1:r1")
    assert set(shuffled) == set(comments) and sorted(shuffled.values()) == sorted(comments.values())
    assert all(shuffled[i] != comments[i] for i in comments)
    assert shuffled == shuffled_comments(comments, "seed1:r1") != shuffled_comments(comments, "seed1:r2")
    assert shuffled_comments({"only": "one"}, "k") == {}


def test_noisy_explanations_replace_about_a_fifth_deterministically_with_vague_or_wrong_rule_ones():
    comments = {f"i{n}": f"should be dovish because reason {n}. Guideline: \"x\"" for n in range(15)}
    golds = {i: "dovish" for i in comments}
    noisy, replaced = noisy_comments(comments, golds, GUIDE, LABELS3, "seed1:r1")
    assert replaced == 3 and sum(noisy[i] != comments[i] for i in comments) == 3
    assert noisy == noisy_comments(comments, golds, GUIDE, LABELS3, "seed1:r1")[0]
    changed = [noisy[i] for i in comments if noisy[i] != comments[i]]
    assert all(c.startswith("should be dovish") for c in changed)


def test_the_stakeholder_applies_the_arm_variant_and_records_text_free_counts(tmp_path):
    stakeholder = RubricStakeholder(GUIDE, LABELS3, fake_stakeholder_completion, model=STAKEHOLDER_FAKE_MODEL,
                                    cache=CommentCache(tmp_path / "s.jsonl"), seed=1)
    plain = stakeholder.comments_for("A-c", 1, CASES)
    shuffled = stakeholder.comments_for("A-c-shuffled", 1, CASES)
    assert set(shuffled) == set(plain) and shuffled != plain
    assert [r["variant"] for r in stakeholder.records] == ["as-given", "shuffled"]
    assert stakeholder.records[1]["new_calls"] == 0   # the same mistakes are answered from the cache
    assert not any(case.text in json.dumps(stakeholder.records) for case in CASES)
