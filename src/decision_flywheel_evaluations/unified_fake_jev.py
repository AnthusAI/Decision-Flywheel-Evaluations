"""A deterministic fake Jev for running the whole unified flywheel offline.

It implements the two client shapes the harness's real factories return:

* ``FakeJevAsync.system_one`` is awaited, like ``typesafe_sdk.AsyncTypeSafeClient``, which is
  what ``jev_flywheel.jev.JevSession`` calls for zero-shot element answers;
* ``FakeJevSync.system_one`` is called from a worker thread, like ``typesafe_sdk.TypeSafeClient``,
  which is what Decision-Flywheel's ``JevAdapter`` calls for the few-shot answer.

Both accept ``state`` as ``{"text": ...}`` (zero-shot) or ``{"labeled_examples": [...],
"target": {"text": ...}}`` (few-shot) and return an object with ``answers``, ``model`` and
``usage``, the fields both callers read.

Answers are pure functions of (text, question, examples), so two runs are identical. To make
the offline loop exercise promotion as well as rejection, the fake can be handed a *planted
signal*: a map from a text's SHA-256 to its reference label. Element answers then lean toward
that label, and the few-shot answer blends a similarity-weighted vote of the shown examples
with it.

For a corpus with more than two labels, ``cues`` maps each label to a few keywords. The cued label
of a text is the label with most keyword hits (ties go to the earlier label); a ``choice`` answer
leans toward it, and a few-shot answer blends that with a similarity-weighted vote of the shown
examples' labels, so it changes with the examples exactly as the planted fake does. ``noul`` and
``score`` answers carry no lean in this mode. With no ``cues`` (the planted binary corpus) nothing
changes. This is a test double -- its numbers mean nothing about Jev.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from decision_flywheel.context import _tokens

FAKE_MODEL = "fake-jev-0"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


@dataclass(frozen=True)
class FakeResponse:
    answers: Dict[str, Dict[str, Any]]
    model: str = FAKE_MODEL
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class FakeJevCore:
    planted: Mapping[str, str] = field(default_factory=dict)
    positive_label: str = "positive"
    strength: float = 0.3
    calls: int = 0
    cues: Mapping[str, Any] = field(default_factory=dict)   # label -> keywords; non-empty selects the N-label mode

    def cued_label(self, text: str) -> Optional[str]:
        """The label with the most keyword hits in ``text`` (ties: earlier label); None if no keyword hits."""
        tokens = _tokens(text)
        best, best_hits = None, 0
        for label, words in self.cues.items():
            hits = sum(1 for word in words if word in tokens)
            if hits > best_hits:
                best, best_hits = label, hits
        return best

    def _lean(self, text: str) -> float:
        if self.cues:
            return 0.0
        label = self.planted.get(text_key(text))
        if label is None:
            return 0.0
        return self.strength if label == self.positive_label else -self.strength

    def answer(self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]) -> FakeResponse:
        self.calls += 1
        if "target" in state:
            text = state["target"]["text"]
            examples = list(state.get("labeled_examples") or [])
        else:
            text, examples = state["text"], []
        answers = {name: self._one(name, question, text, examples) for name, question in questions.items()}
        words = len(text.split()) + sum(len(e["text"].split()) for e in examples)
        return FakeResponse(answers, usage={"input_tokens": 20 + words, "output_tokens": 12 * len(questions)})

    def _one(self, name: str, question: Mapping[str, Any], text: str, examples) -> Dict[str, Any]:
        kind = question.get("type")
        noise = _unit("noise", name, question, text) - 0.5
        lean = self._lean(text)
        if kind == "noul":
            p = min(max(0.5 + lean + 0.4 * noise, 0.01), 0.99)
            return {"type": "noul", "noul": round(p, 4)}
        options = list(question.get("criteria") or [])
        if kind == "score":
            levels = len(options)
            centre = min(max((0.5 + lean + noise) * (levels - 1), 0), levels - 1)
            weights = [math.exp(-abs(i - centre)) for i in range(levels)]
            total = sum(weights)
            probabilities = {str(i): round(w / total, 4) for i, w in enumerate(weights)}
            level = max(range(levels), key=lambda i: weights[i])
            return {"type": "score", "score": float(level), "confidence": probabilities[str(level)],
                    "probabilities": probabilities, "legend": {str(i): o for i, o in enumerate(options)}}
        if self.cues:
            return self._multi_choice(name, question, text, examples, options)
        # choice: lean toward the first option for the planted positive label
        first = 0.5 + lean + 0.5 * noise
        if examples and self.positive_label in options:
            target = _tokens(text)
            score = {o: 0.0 for o in options}
            for example in examples:
                other = _tokens(example["text"])
                sim = len(target & other) / math.sqrt(max(1, len(target)) * max(1, len(other)))
                if example["label"] in score:
                    score[example["label"]] += sim + 1e-3
            total = sum(score.values()) or 1.0
            vote = score[self.positive_label] / total
            first = 0.5 * vote + 0.5 * (0.5 + lean) + 0.1 * noise
        first = min(max(first, 0.02), 0.98)
        rest = (1.0 - first) / max(len(options) - 1, 1)
        if self.positive_label in options:
            ordered = [self.positive_label] + [o for o in options if o != self.positive_label]
        else:
            ordered = options
        probabilities = {o: round(first if i == 0 else rest, 4) for i, o in enumerate(ordered)}
        choice = max(probabilities, key=probabilities.get)
        return {"type": "choice", "choice": choice, "confidence": probabilities[choice],
                "probabilities": probabilities}

    def _multi_choice(self, name: str, question: Mapping[str, Any], text: str, examples,
                      options) -> Dict[str, Any]:
        cued = self.cued_label(text)
        prior = {o: 1.0 + 0.5 * (_unit("noise", name, question, text, o) - 0.5) + (10.0 * self.strength if o == cued else 0.0)
                 for o in options}
        total = sum(prior.values())
        weights = {o: prior[o] / total for o in options}
        if examples:
            target = _tokens(text)
            votes = {o: 0.0 for o in options}
            for example in examples:
                other = _tokens(example["text"])
                sim = len(target & other) / math.sqrt(max(1, len(target)) * max(1, len(other)))
                if example["label"] in votes:
                    votes[example["label"]] += sim + 1e-3
            vote_total = sum(votes.values()) or 1.0
            weights = {o: 0.5 * votes[o] / vote_total + 0.5 * weights[o] for o in options}
        probabilities = {o: round(weights[o] / sum(weights.values()), 4) for o in options}
        choice = max(options, key=lambda o: (probabilities[o], -options.index(o)))
        return {"type": "choice", "choice": choice, "confidence": probabilities[choice],
                "probabilities": probabilities}


class FakeJevAsync:
    def __init__(self, core: Optional[FakeJevCore] = None):
        self.core = core or FakeJevCore()

    async def system_one(self, *, state, questions):
        return self.core.answer(state, questions)


class FakeJevSync:
    def __init__(self, core: Optional[FakeJevCore] = None):
        self.core = core or FakeJevCore()

    def system_one(self, *, state, questions):
        return self.core.answer(state, questions)
