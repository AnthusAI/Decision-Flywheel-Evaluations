import asyncio
import json

import pytest

from .unified_fake_jev import FakeJevAsync, FakeJevCore, FakeJevSync, text_key
from .unified_spend import (
    CeilingExhausted, CircuitOpen, CountingAsyncClient, CountingSyncClient, SpendLedger,
    concurrency_slots, final_d_upper_bound, request_upper_bound)

QUESTIONS = {"Sentiment": {"type": "choice", "instructions": "?", "criteria": {"positive": None, "negative": None}}}


def test_every_request_is_counted_per_arm_round_and_kind_without_any_text(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.json", ceiling=10)
    client = CountingAsyncClient(FakeJevAsync, ledger, concurrency_slots(2))
    ledger.set_scope("A", 1, "zero-shot")
    asyncio.run(client.system_one(state={"text": "secret words"}, questions=QUESTIONS))
    ledger.set_scope("B", 1, "few-shot")
    sync = CountingSyncClient(FakeJevSync, ledger, concurrency_slots(1))
    sync.system_one(state={"labeled_examples": [], "target": {"text": "secret words"}}, questions=QUESTIONS)
    assert ledger.by_arm() == {"A": {"zero-shot": 1, "total": 1}, "B": {"few-shot": 1, "total": 1}}
    stored = (tmp_path / "ledger.json").read_text()
    assert json.loads(stored)["used"] == 2 and "secret" not in stored
    assert "secret" not in json.dumps(ledger.summary())


def test_the_ceiling_is_cumulative_across_invocations_and_refuses_before_sending(tmp_path):
    path = tmp_path / "ledger.json"
    core = FakeJevCore()
    first = SpendLedger(path, ceiling=3)
    client = CountingSyncClient(lambda: FakeJevSync(core), first, concurrency_slots(1))
    for _ in range(2):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)
    second = SpendLedger(path, ceiling=3)
    client = CountingSyncClient(lambda: FakeJevSync(core), second, concurrency_slots(1))
    client.system_one(state={"text": "x"}, questions=QUESTIONS)
    with pytest.raises(CeilingExhausted):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)
    assert core.calls == 3 and second.used == 3


def test_a_ledger_cannot_be_reopened_with_a_different_ceiling(tmp_path):
    SpendLedger(tmp_path / "ledger.json", ceiling=9500)
    with pytest.raises(ValueError):
        SpendLedger(tmp_path / "ledger.json", ceiling=20000)


def test_the_per_invocation_cap_stops_a_run_below_the_cumulative_ceiling(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.json", ceiling=100, max_new=1)
    client = CountingSyncClient(FakeJevSync, ledger, concurrency_slots(1))
    client.system_one(state={"text": "x"}, questions=QUESTIONS)
    with pytest.raises(CeilingExhausted):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)


class _Rejected(Exception):
    status = 402


class _Broken:
    def system_one(self, *, state, questions):
        raise _Rejected("payment required")


class _Flaky:
    def system_one(self, *, state, questions):
        raise TimeoutError("slow")


def test_an_account_rejection_trips_the_breaker_and_later_calls_spend_nothing(tmp_path):
    ledger = SpendLedger(None, ceiling=100)
    client = CountingSyncClient(_Broken, ledger, concurrency_slots(1))
    with pytest.raises(_Rejected):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)
    with pytest.raises(CircuitOpen):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)
    assert ledger.used == 1
    with pytest.raises(CircuitOpen):
        ledger.raise_if_tripped()


def test_a_streak_of_consecutive_failures_trips_the_breaker(tmp_path):
    ledger = SpendLedger(None, ceiling=100, max_consecutive_failures=3)
    client = CountingSyncClient(_Flaky, ledger, concurrency_slots(1))
    for _ in range(3):
        with pytest.raises(TimeoutError):
            client.system_one(state={"text": "x"}, questions=QUESTIONS)
    with pytest.raises(CircuitOpen):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)
    assert ledger.used == 3


def test_a_client_that_cannot_be_constructed_spends_no_attempt():
    ledger = SpendLedger(None, ceiling=5)

    def explode():
        raise RuntimeError("no credentials")

    client = CountingSyncClient(explode, ledger, concurrency_slots(1))
    with pytest.raises(RuntimeError):
        client.system_one(state={"text": "x"}, questions=QUESTIONS)
    assert ledger.used == 0


def test_in_flight_requests_never_exceed_the_concurrency_bound():
    ledger = SpendLedger(None, ceiling=100)
    active = {"now": 0, "peak": 0}

    class Slow:
        async def system_one(self, *, state, questions):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return FakeJevCore().answer(state, questions)

    client = CountingAsyncClient(Slow, ledger, concurrency_slots(3))

    async def many():
        await asyncio.gather(*(client.system_one(state={"text": str(i)}, questions=QUESTIONS) for i in range(12)))

    asyncio.run(many())
    assert active["peak"] <= 3 and ledger.used == 12


def test_the_upper_bound_for_the_first_live_step_matches_the_plans_smoke_estimate():
    bound = request_upper_bound(["A", "B"], rounds=1, per_round=100, eval_n=100)
    assert bound == {"A": 200, "B": 200, "total": 400}


def test_free_arms_cost_nothing_and_a_plus_b_reuses_b_few_shot_answers():
    bound = request_upper_bound(["0", "B-local", "B", "A+B"], rounds=3, per_round=100, eval_n=100)
    assert bound["0"] == bound["B-local"] == 0
    assert bound["B"] == 900
    assert bound["A+B"] == 1100  # zero-shot only; includes topping up old elements for new labels
    alone = request_upper_bound(["A+B"], rounds=3, per_round=100, eval_n=100)
    assert alone["A+B"] == 2000


def test_the_fake_is_deterministic_and_few_shot_answers_follow_the_examples():
    core = FakeJevCore()
    state = {"text": "the team won"}
    assert core.answer(state, QUESTIONS) == core.answer(state, QUESTIONS)
    positive = [{"text": "the team won again", "label": "positive"}, {"text": "taxes rose", "label": "negative"}]
    negative = [{"text": "the team won again", "label": "negative"}, {"text": "taxes rose", "label": "positive"}]
    up = core.answer({"labeled_examples": positive, "target": {"text": "the team won"}}, QUESTIONS)
    down = core.answer({"labeled_examples": negative, "target": {"text": "the team won"}}, QUESTIONS)
    assert up.answers["Sentiment"]["probabilities"]["positive"] > down.answers["Sentiment"]["probabilities"]["positive"]


def test_a_planted_signal_makes_element_answers_lean_toward_the_reference_label():
    planted = {text_key("good"): "positive", text_key("bad"): "negative"}
    core = FakeJevCore(planted=planted, strength=0.4)
    question = {"x": {"type": "noul", "instructions": "?"}}
    assert core.answer({"text": "good"}, question).answers["x"]["noul"] > 0.5
    assert core.answer({"text": "bad"}, question).answers["x"]["noul"] < 0.5


def test_the_upper_bound_counts_one_list_request_per_item_per_trial_list():
    bound = request_upper_bound(["F", "F-rand"], rounds=2, per_round=100, eval_n=100)
    # F, round 1: 3 trial lists over 100 labels + the chosen list over 100 labels and 100 evaluation items;
    # round 2: predict 100 new labels, 3 x 200, then 200 + 100.
    assert bound["F"] == (3 * 100 + 200) + (100 + 3 * 200 + 300)
    assert bound["F-rand"] == 200 + 300
    assert request_upper_bound(["A-c"], rounds=3, per_round=100, eval_n=100)["A-c"] == \
        request_upper_bound(["A"], rounds=3, per_round=100, eval_n=100)["A"]


def test_failures_are_tallied_by_error_class_and_status_without_message_text():
    class Rejected(Exception):
        status = 429

    ledger = SpendLedger(None, ceiling=100, max_consecutive_failures=50)
    for error in (TimeoutError("secret detail"), TimeoutError("other"), Rejected("more secret")):
        ledger.reserve()
        ledger.failed(error)
    classes = ledger.summary()["error_classes"]
    assert classes == {"TimeoutError": 2, "Rejected[429]": 1}
    assert "secret" not in repr(ledger.summary())


def test_arm_d_needs_one_request_per_labeled_item_and_per_evaluation_item():
    assert final_d_upper_bound(n_labeled=300, eval_n=600) == {"D": 900, "total": 900}
    assert final_d_upper_bound(n_labeled=80, eval_n=600)["total"] == 680
