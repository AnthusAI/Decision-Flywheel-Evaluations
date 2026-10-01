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
