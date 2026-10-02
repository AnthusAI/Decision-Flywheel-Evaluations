"""Freeze harness arms as core ``ClassifierBundle`` directories and classify through them.

Import only after ``unified_env.put_clone_first()`` (it reads Jev-Flywheel's ``Score``).

* ``ScoreRubric`` is the bundle's ``scorecard.yaml`` payload: one Jev-Flywheel ``Score`` (the
  arm's rubric questions and its fitted head) as YAML. Core hashes it and never parses it.
* ``CachedJevClient`` sits between a bundle and the engine. The bundle builds its own request
  (state + every question); the client answers from the run's caches -- the shared zero-shot
  cache, or the list cache keyed by the list's fingerprint -- and sends the bundle's own state
  with only the missing questions, counted by the ledger, storing what comes back. So a final
  run is replayable at $0 and a bundle's answers are the ones its in-memory twin was scored on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import yaml
from decision_flywheel.context import FixedExampleList
from decision_flywheel.models import DecisionResult
from jev_flywheel.answers import AnswerCache
from jev_flywheel.jev import normalize_answer
from jev_flywheel.scorecard import Score
from jev_flywheel.scoring import predict

from .unified_lists import ListAnswers, keyed

BUNDLE_ARMS = ("0", "A", "A-c", "F", "F-rand", "A-c+F")


def _as_dict(value: Any) -> dict:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def _class_probabilities(raw: Optional[Mapping[str, float]], top: str, confidence: float, others) -> Dict[str, float]:
    """The head's per-class probabilities with the top class at the (possibly calibrated) confidence.

    Without calibration the top probability already equals ``confidence`` and nothing is rescaled;
    with it, the other classes keep their relative proportions and share ``1 - confidence``. If the
    head reports no probabilities (or none for the others) they share it evenly, as for two classes.
    """
    rest = {c: max(float(raw.get(c, 0.0)), 0.0) for c in others} if raw else {}
    mass = sum(rest.values())
    if mass <= 0.0:
        return {top: confidence, **{c: (1.0 - confidence) / len(others) for c in others}}
    scale = (1.0 - confidence) / mass
    probabilities = {top: confidence, **{c: rest[c] * scale for c in others}}
    return {c: probabilities[c] for c in (top, *others)}


class ScoreRubric:
    def __init__(self, score: Score):
        self.score = score

    @classmethod
    def parse(cls, text: str) -> "ScoreRubric":
        return cls(Score.from_config(yaml.safe_load(text)))

    @staticmethod
    def dump(score: Score) -> str:
        return yaml.safe_dump(score.to_config(), sort_keys=False)  # option order is part of a question

    def questions(self) -> Dict[str, Dict[str, Any]]:
        return self.score.questions()

    def predict(self, answers: Mapping[str, Mapping[str, Any]]) -> DecisionResult:
        normalized = {name: normalize_answer(answer) for name, answer in answers.items()}
        result = predict(self.score, {}, features=self.score.feature_vector(normalized))
        confidence = min(max(float(result.confidence or 0.0), 0.0), 1.0)
        others = [c for c in self.score.decision.classes if c != result.value]
        if len(others) == 1:   # two classes: the exact legacy numbers
            probabilities = {result.value: confidence, **{c: (1.0 - confidence) / len(others) for c in others}}
        else:                  # N > 2: the head's own probabilities, rescaled only if calibration moved the top
            raw = (result.metadata or {}).get("decision", {}).get("probabilities")
            probabilities = _class_probabilities(raw, result.value, confidence, others)
        return DecisionResult(result.value, probabilities, confidence=confidence)


@dataclass
class _Response:
    answers: Dict[str, dict]
    model: Optional[str] = None
    usage: Optional[dict] = None


class CachedJevClient:
    """``async system_one`` for a bundle: cache first, then the counted engine for the gap."""

    def __init__(self, inner: Any, item_ids_by_text: Mapping[str, str], *, zero_shot: AnswerCache,
                 lists: ListAnswers, fixed: Optional[FixedExampleList]):
        self.inner, self.ids, self.zero_shot, self.lists, self.fixed = inner, item_ids_by_text, zero_shot, lists, fixed
        self.sent = 0

    def _have(self, item_id: str, questions: Mapping[str, Any]) -> Dict[str, dict]:
        if self.fixed is None:
            return dict(self.zero_shot.bulk_partial_answers([item_id], questions)[item_id])
        return self.lists.answers([item_id], questions, self.fixed)[item_id]

    async def system_one(self, *, state, questions):
        text = state["target"]["text"] if "target" in state else state["text"]
        item_id = self.ids[text]
        have = self._have(item_id, questions)
        gap = {name: dict(q) for name, q in questions.items() if name not in have}
        if not gap:
            return _Response(have)
        self.sent += 1
        response = await self.inner.system_one(state=state, questions=gap)
        model = getattr(response, "model", None)
        for name, answer in (getattr(response, "answers", None) or {}).items():
            if name not in gap:
                continue
            answer = normalize_answer(_as_dict(answer))
            if self.fixed is None:
                self.zero_shot.put(item_id, name, gap[name], answer, model)
            else:
                self.lists.cache.put(item_id, name, keyed(gap[name], self.fixed), answer, model)
            have[name] = answer
        return _Response(have, model, getattr(response, "usage", None))


def bundle_dir(run_dir, arm: str, round_number: int):
    return run_dir / "bundles" / arm.replace("+", "plus") / f"round-{round_number}"


def text_free(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """What a run summary records about a bundle: its hash and a few identifiers."""
    fewshot = manifest.get("fewshot") or {}
    return {"bundle_hash": manifest["bundle_hash"], "fewshot_fingerprint": fewshot.get("fingerprint"),
            "scorecard_sha256": manifest["rubric"]["scorecard_sha256"],
            "questions": sorted(manifest["rubric"]["questions"]),
            "head_fit_id": manifest["head"].get("fit_id")}
