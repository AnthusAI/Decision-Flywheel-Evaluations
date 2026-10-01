"""Fixed example lists for the unified flywheel (arms F, F-rand and A-c+F).

The request shape is the simplest one: **one Jev request per item**, carrying the example
list in the state together with every rubric question of the arm's scorecard:

    state = {"labeled_examples": [{"text", "label"}, ...], "target": {"text"}}
    questions = {holistic question, every element question}

Answers are cached per item and question exactly as Jev-Flywheel's ``AnswerCache`` does, with
one addition: the cached question body carries the list's fingerprint (``example_list``). When
the list changes, every item is simply asked again; when the analyst adds a question, only that
question is asked (one request per item, all missing questions together). The extra key is
never sent to Jev.

The list itself is Decision-Flywheel's ``FixedExampleList`` (with its per-label reserve, so a
labeled item that is one of the examples sees the reserve instead of itself), and the
optimizer is its ``improve_example_list``. ``ListModel`` adapts this cache to the optimizer's
``DecisionModel`` interface so the search reads the holistic answer from the same requests the
head is later trained on.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import FixedExampleList, RandomBalanced
from decision_flywheel.example_list import example_list_from_policy
from decision_flywheel.models import DecisionResult, DecisionTask, LabeledItem, ModelCapabilities
from decision_flywheel.models import Item as DFItem
from jev_flywheel.answers import AnswerCache
from jev_flywheel.jev import normalize_answer

from .unified_splits import assert_target_excluded

LIST_KEY = "example_list"


def _as_dict(value: Any) -> dict:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def keyed(question: Mapping[str, Any], fixed: FixedExampleList) -> Dict[str, Any]:
    """The cache key for a question asked with ``fixed`` in the state."""
    return {**question, LIST_KEY: fixed.fingerprint}


def random_list(task: DecisionTask, labeled: Sequence[LabeledItem], *, per_label: int, seed: int) -> FixedExampleList:
    """F-rand's list: a ``RandomBalanced(seed)`` draw, its next item per label as the reserve."""
    return example_list_from_policy(RandomBalanced(seed=seed), task, labeled, per_label=per_label)


class ListAnswers:
    """Jev answers asked with a fixed example list in the request state.

    ``context`` (called ``fixed`` below) is anything with a ``fingerprint`` that ``examples_for``
    can resolve; ``unified_retrieval.RetrievalAnswers`` reuses this class for per-item retrieval
    under its own cache key name.
    """

    key = LIST_KEY

    def __init__(self, path, task: DecisionTask, texts: Mapping[str, str], labels: Mapping[str, str],
                 client_factory: Callable[[], Any], *, concurrency: int = 8):
        self.cache = AnswerCache(path)
        self.task = task
        self.texts = texts
        self.labels = labels
        self.client_factory = client_factory
        self.concurrency = concurrency
        self._client = None
        self._examples: Dict[Tuple[str, str], Tuple[LabeledItem, ...]] = {}

    def _row(self, item_id: str) -> LabeledItem:
        return LabeledItem(DFItem(item_id, {self.task.input_field: self.texts[item_id]}), self.labels[item_id])

    def examples_for(self, fixed: FixedExampleList, item_id: str) -> Tuple[LabeledItem, ...]:
        """The examples ``item_id`` is shown: the list, with the reserve if it is one of them."""
        key = (fixed.fingerprint, item_id)
        if key not in self._examples:
            rows = [self._row(i) for i in fixed.example_ids + fixed.reserve_ids]
            target = DFItem(item_id, {self.task.input_field: self.texts[item_id]})
            plan = build_context_plan(self.task, target, rows, fixed, budget=ContextBudget(per_label=fixed.per_label),
                                      display_order="canonical", order_seed=0,
                                      presentation_label_order=self.task.labels)
            assert_target_excluded(item_id, plan.example_ids)
            self._examples[key] = plan.examples
        return self._examples[key]

    def keyed(self, question: Mapping[str, Any], fixed: Any) -> Dict[str, Any]:
        return {**question, self.key: fixed.fingerprint}

    def answers(self, ids: Sequence[str], questions: Mapping[str, Mapping[str, Any]],
                fixed: FixedExampleList) -> Dict[str, Dict[str, dict]]:
        found = self.cache.bulk_partial_answers(ids, {name: self.keyed(q, fixed) for name, q in questions.items()})
        return {item_id: dict(found[item_id]) for item_id in ids}

    def missing(self, ids: Sequence[str], questions: Mapping[str, Mapping[str, Any]],
                fixed: FixedExampleList) -> Dict[str, Dict[str, dict]]:
        have = self.answers(ids, questions, fixed)
        gaps = {item_id: {name: dict(q) for name, q in questions.items() if name not in have[item_id]}
                for item_id in ids}
        return {item_id: gap for item_id, gap in gaps.items() if gap}

    def fill(self, ids: Sequence[str], questions: Mapping[str, Mapping[str, Any]],
             fixed: FixedExampleList) -> Dict[str, int]:
        """One request per item that lacks anything; returns requests sent and failures."""
        todo = self.missing(list(dict.fromkeys(ids)), questions, fixed)
        if not todo:
            return {"requests": 0, "failures": 0}
        failures = 0

        async def run_all() -> None:
            nonlocal failures
            # A client is bound to the event loop that first used it; every fill runs its own
            # loop, so each gets its own client ("Event loop is closed" otherwise).
            self._client = None
            semaphore = asyncio.Semaphore(self.concurrency)

            async def one(item_id: str, gap: Dict[str, dict]) -> None:
                nonlocal failures
                async with semaphore:
                    try:
                        await self._ask(item_id, gap, fixed)
                    except Exception:  # noqa: BLE001 - counted by the ledger; a gap is reported, not fatal
                        failures += 1

            await asyncio.gather(*(one(i, gap) for i, gap in todo.items()))

        asyncio.run(run_all())
        return {"requests": len(todo), "failures": failures}

    async def _ask(self, item_id: str, gap: Mapping[str, dict], fixed: FixedExampleList) -> None:
        if self._client is None:
            self._client = self.client_factory()
        examples = self.examples_for(fixed, item_id)
        state = {"labeled_examples": [{"text": row.item.values[self.task.input_field], "label": row.label}
                                      for row in examples],
                 "target": {"text": self.texts[item_id]}}
        response = await self._client.system_one(state=state, questions=dict(gap))
        model = getattr(response, "model", None)
        for name, answer in (getattr(response, "answers", None) or {}).items():
            if name in gap:
                self.cache.put(item_id, name, self.keyed(gap[name], fixed), normalize_answer(_as_dict(answer)), model)


class ListModel:
    """``improve_example_list``'s model: the holistic answer from ``ListAnswers``.

    The optimizer hands over the examples already resolved for a target; ``lists`` maps them
    back to the list they came from, so the answer is read from (or written to) that list's
    cache entry. Every request still carries the arm's whole question set.
    """

    name = "jev-fixed-list"
    capabilities = ModelCapabilities(supports_labeled_context=True, supports_probability_distributions=True)

    def __init__(self, answers: ListAnswers, questions: Mapping[str, Mapping[str, Any]], holistic: str,
                 lists: Sequence[FixedExampleList], fingerprint: str):
        self.answers, self.questions, self.holistic = answers, questions, holistic
        self.fingerprint = fingerprint
        self._by_context: Dict[Tuple[str, Tuple[str, ...]], FixedExampleList] = {}
        self._lists = list(lists)
        self.requests = 0   # answers the cache did not have (the harness prefetches, so normally 0)

    def _list_for(self, target_id: str, example_ids: Tuple[str, ...]) -> FixedExampleList:
        key = (target_id, example_ids)
        if key not in self._by_context:
            for fixed in self._lists:
                shown = tuple(row.item.id for row in self.answers.examples_for(fixed, target_id))
                self._by_context.setdefault((target_id, shown), fixed)
        if key not in self._by_context:
            raise ValueError("the optimizer asked about a context that is not one of the declared lists")
        return self._by_context[key]

    async def decide(self, task: DecisionTask, target: DFItem, context: Sequence[LabeledItem]) -> DecisionResult:
        fixed = self._list_for(target.id, tuple(row.item.id for row in context))
        answer = self.answers.answers([target.id], self.questions, fixed)[target.id].get(self.holistic)
        if answer is None:
            gap = self.answers.missing([target.id], self.questions, fixed).get(target.id, {})
            self.requests += 1
            await self.answers._ask(target.id, gap, fixed)
            answer = self.answers.answers([target.id], self.questions, fixed)[target.id][self.holistic]
        return task.validate_result(DecisionResult(answer["choice"], answer.get("probabilities"),
                                                   confidence=answer.get("confidence")))
