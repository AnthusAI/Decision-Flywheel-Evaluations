"""Reuse the static study's cached zero-shot answers as the harness's free arm-0 baseline.

The static Emotion study stored Jev's zero-shot answer for each of the 2,000 scoreboard items in a
sqlite scoreboard (``.data/dev/emotion.scoreboard2.sqlite``, condition ending ``:zero:0``). The harness's
answer cache (``jev_flywheel.answers.AnswerCache``) keys an answer by ``(item id, question name, hash of
the question body)``; the model is not part of the key. An imported answer is therefore valid only if the
harness would have sent the SAME request, so the import is guarded three ways:

* the answers' reported model must equal the run's provider model (nothing from another Jev version);
* when the item texts are supplied, the wire fingerprint of the request the harness would send
  (``sha256`` of the compact JSON of state and questions, exactly as core ``build_context_plan`` computes
  ``serialized_request_fingerprint``) must equal the ``state_fingerprint`` the study recorded for that
  cell; an item that does not match is not imported (and is counted);
* every answer's choice and probability labels must be options of the question.

Answers are stored with ``AnswerCache.put_hashed`` under the hash of the question body the harness's seed
score builds, as ``{"type": "choice", "choice", "confidence", "probabilities"}``, the shape a live answer
has. The import is idempotent (a key already present is left alone), text-free (only item ids, labels and
probabilities are read from the file) and sends nothing. Pool items have no zero-shot answers here; they
are paid for live.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from jev_flywheel.answers import AnswerCache, question_hash

ZERO_SHOT_CONDITION_SUFFIX = ":zero:0"


class ZeroShotImportError(ValueError):
    """The cached file cannot be reused as this run's zero-shot answers."""


@dataclass(frozen=True)
class ImportReport:
    """Text-free counts from one import."""
    cells_in_file: int          # zero-shot cells the file holds
    requested: int              # item ids the caller asked for
    imported: int               # rows newly written
    already_present: int        # requested items whose answer was already cached
    not_in_file: int            # requested items the file has no successful answer for
    wire_mismatches: int        # requested items whose recorded wire fingerprint differs from the harness's request
    question_hash: str
    wire_checked: bool

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def wire_fingerprint(state: Mapping[str, Any], name: str, question: Mapping[str, Any]) -> str:
    """sha256 of the compact JSON request, as core ``_serialize_request`` serializes it (option order kept)."""
    serialized = json.dumps({"state": state, "questions": {name: dict(question)}}, ensure_ascii=False,
                            separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def read_zero_shot_cells(path: Path, *, condition_suffix: str = ZERO_SHOT_CONDITION_SUFFIX) -> Dict[str, Dict[str, Any]]:
    """``{target id: {"payload": dict, "state_fingerprint": str}}`` for the file's one zero-shot condition."""
    connection = _connect_read_only(path)
    try:
        conditions = [row[0] for row in connection.execute("SELECT DISTINCT condition FROM cells")
                      if row[0].endswith(condition_suffix)]
        if len(conditions) != 1:
            raise ZeroShotImportError(f"expected exactly one condition ending {condition_suffix!r}, found {len(conditions)}")
        rows = connection.execute(
            "SELECT c.target_id, r.state_fingerprint, r.payload FROM cells c JOIN requests r USING (request_id) "
            "WHERE c.condition = ? AND r.status = 'success' AND r.payload IS NOT NULL", (conditions[0],)).fetchall()
    finally:
        connection.close()
    return {target: {"state_fingerprint": fingerprint, "payload": json.loads(payload)}
            for target, fingerprint, payload in rows}


def _answer(payload: Mapping[str, Any], options: Iterable[str], expected_model: str) -> Dict[str, Any]:
    options = set(options)
    if payload.get("reported_model") != expected_model:
        raise ZeroShotImportError(f"cached answers come from {payload.get('reported_model')!r}, not {expected_model!r}")
    choice, probabilities = payload.get("choice"), dict(payload.get("probabilities") or {})
    if choice not in options or not set(probabilities) <= options:
        raise ZeroShotImportError("a cached answer names a label that is not an option of the question")
    answer: Dict[str, Any] = {"type": "choice", "choice": choice}
    if payload.get("confidence") is not None:
        answer["confidence"] = payload["confidence"]
    if probabilities:
        answer["probabilities"] = probabilities
    return answer


def import_zero_shot(cache: AnswerCache, path: Path, *, name: str, question: Mapping[str, Any],
                     item_ids: Iterable[str], expected_model: str,
                     state_fn: Optional[Callable[[str], Mapping[str, Any]]] = None,
                     texts: Optional[Mapping[str, str]] = None) -> ImportReport:
    """Put the file's zero-shot answers for ``item_ids`` into ``cache`` under ``(id, name, hash(question))``.

    Passing ``state_fn`` and ``texts`` turns on the per-item wire check described in the module docstring.
    """
    cells = read_zero_shot_cells(path)
    qhash = question_hash(question)
    check = state_fn is not None and texts is not None
    options = list(question.get("criteria") or [])
    ids = list(dict.fromkeys(item_ids))
    imported = present = absent = mismatched = 0
    for item_id in ids:
        cell = cells.get(item_id)
        if cell is None:
            absent += 1
            continue
        if check and wire_fingerprint(state_fn(texts[item_id]), name, question) != cell["state_fingerprint"]:
            mismatched += 1
            continue
        answer = _answer(cell["payload"], options, expected_model)
        if cache.get(item_id, name, question) is not None:
            present += 1
            continue
        cache.put_hashed(item_id, name, qhash, answer, cell["payload"].get("reported_model"))
        imported += 1
    return ImportReport(len(cells), len(ids), imported, present, absent, mismatched, qhash, check)
