"""A corpus-neutral copy of Jev-Flywheel's steering procedure, swapped in at runtime for one run.

The pinned ``procedures/steer_scorecard.tac`` tells the steering analyst to "look for what the
positive and negative examples have in common as groups, including things that are not about
sentiment at all". That is the planted sentiment corpus talking; for any other corpus the wording
is wrong (an Emotion item is not "positive" or "negative"). For corpora that set
``reword_steering_prompt`` the harness therefore writes a reworded COPY under ``var/`` and points
``jev_flywheel.steer`` at it while a steering round runs, then restores everything.

The copy is generated from the pinned file by the two exact-text substitutions in
``ORIGINAL_PHRASES`` (each must occur exactly once, so a changed pin fails loudly instead of being
half reworded). The pinned clone is never edited. The planted corpus never reaches this code path's
side effects: it keeps the original file byte for byte and writes nothing.

Why two things are patched, not one: ``steer._run`` renders the procedure with
``render_source(path=PROCEDURE)`` whose default was bound when ``steer`` was imported, so
re-pointing the module variable ``PROCEDURE`` alone changes only the file name Tactus is told about,
not the text it executes. Both ``PROCEDURE`` and ``render_source.__kwdefaults__["path"]`` are
pointed at the copy, and both are restored afterwards, also on error.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

from .unified_corpus import Corpus
from .unified_env import REPO_ROOT

DEFAULT_OUT_DIR = REPO_ROOT / "var" / "unified-flywheel" / "steering-prompts"
PROCEDURE_RELATIVE = Path("procedures") / "steer_scorecard.tac"

# (text in the pinned procedure, replacement template). ``{examples}`` and ``{subject}`` are the
# corpus's ``steer_examples_phrase`` and ``steer_subject_phrase``.
ORIGINAL_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("the positive and\n  negative examples", "{examples}"),
    ("not about sentiment", "not about {subject}"),
)


class SteeringPromptError(RuntimeError):
    """The pinned procedure no longer has the wording the substitution expects."""


def reword_procedure(source: str, *, examples_phrase: str, subject_phrase: str) -> str:
    """The pinned procedure text with the sentiment-specific phrases replaced (pure, deterministic)."""
    for old, template in ORIGINAL_PHRASES:
        if source.count(old) != 1:
            raise SteeringPromptError(
                f"expected exactly one occurrence of {old!r} in the pinned steering procedure, "
                f"found {source.count(old)}")
        source = source.replace(old, template.format(examples=examples_phrase, subject=subject_phrase))
    return source


def write_reworded_procedure(corpus: Corpus, clone: Path, out_dir: Path = DEFAULT_OUT_DIR) -> Tuple[Path, str]:
    """Write the corpus's reworded copy under ``out_dir``; returns ``(path, sha256 of its bytes)``."""
    original = (Path(clone) / PROCEDURE_RELATIVE).read_text(encoding="utf-8")
    text = reword_procedure(original, examples_phrase=corpus.steer_examples_phrase,
                            subject_phrase=corpus.steer_subject_phrase)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"steer_scorecard.{corpus.name}.tac"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def original_procedure_sha256(clone: Path) -> str:
    return hashlib.sha256((Path(clone) / PROCEDURE_RELATIVE).read_bytes()).hexdigest()


@contextmanager
def steering_procedure(corpus: Corpus, clone: Path, out_dir: Path = DEFAULT_OUT_DIR) -> Iterator[Optional[str]]:
    """Run a steering round under the corpus's procedure; yields the override's sha256, or ``None`` (planted)."""
    if not corpus.reword_steering_prompt:
        yield None
        return
    from jev_flywheel import steer

    path, sha = write_reworded_procedure(corpus, clone, out_dir)
    previous_path, previous_default = steer.PROCEDURE, steer.render_source.__kwdefaults__["path"]
    steer.PROCEDURE = path
    steer.render_source.__kwdefaults__["path"] = path
    try:
        yield sha
    finally:
        steer.PROCEDURE = previous_path
        steer.render_source.__kwdefaults__["path"] = previous_default
