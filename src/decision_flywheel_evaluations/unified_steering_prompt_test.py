"""Specs for the steering-prompt override. The text substitution is pure; the PROCEDURE specs need the pinned clone."""
import difflib
import hashlib
import subprocess
from pathlib import Path

import pytest

from . import unified_env
from .unified_corpus import EMOTION_CORPUS, PLANTED
from .unified_steering_prompt import (ORIGINAL_PHRASES, SteeringPromptError, reword_procedure,
                                      steering_procedure, write_reworded_procedure)

try:
    CLONE = Path(unified_env.verify_clone().path)
except unified_env.EnvironmentProblem:   # a checkout without the pinned clone
    CLONE = None
needs_clone = pytest.mark.skipif(CLONE is None, reason="pinned Jev-Flywheel clone unavailable")
needs_jev = pytest.mark.skipif(CLONE is None or not unified_env.jev_dependencies_available(),
                               reason="needs scikit-learn and Tactus (Jev-Flywheel's interpreter) and the clone")

SAMPLE = ("intro\n- look for what the positive and\n  negative examples have in common as groups, including "
          "things that are not about sentiment\n  at all, such as what the text is about.\nrest\n")


def test_the_reworded_text_replaces_exactly_the_sentiment_phrases_and_nothing_else():
    out = reword_procedure(SAMPLE, examples_phrase="the examples of each label", subject_phrase="the emotion")
    assert "positive" not in out and "negative" not in out and "sentiment" not in out
    assert "the examples of each label have in common" in out and "not about the emotion\n  at all" in out
    assert out.startswith("intro\n- look for what ") and out.endswith("such as what the text is about.\nrest\n")


def test_a_pinned_file_that_no_longer_contains_a_phrase_exactly_once_is_refused():
    with pytest.raises(SteeringPromptError, match="expected exactly one"):
        reword_procedure("nothing to see", examples_phrase="x", subject_phrase="y")
    with pytest.raises(SteeringPromptError, match="expected exactly one"):
        reword_procedure(SAMPLE + SAMPLE, examples_phrase="x", subject_phrase="y")


def test_the_reword_is_deterministic():
    kwargs = dict(examples_phrase="the examples of each label", subject_phrase="the emotion")
    assert reword_procedure(SAMPLE, **kwargs) == reword_procedure(SAMPLE, **kwargs)


def test_planted_keeps_the_pinned_wording_and_every_other_registered_corpus_rewords_it():
    from .unified_corpus import CORPORA

    assert PLANTED.reword_steering_prompt is False
    assert all(c.reword_steering_prompt for name, c in CORPORA.items() if name != "planted")
    assert EMOTION_CORPUS.steer_examples_phrase == "the examples of each label"
    assert len(ORIGINAL_PHRASES) == 2   # the substitutions are documented data, not buried code


@needs_clone
def test_the_generated_procedure_differs_from_the_pinned_one_only_in_the_intended_phrases(tmp_path):
    original = (CLONE / "procedures" / "steer_scorecard.tac").read_text(encoding="utf-8")
    path, sha = write_reworded_procedure(EMOTION_CORPUS, CLONE, tmp_path)
    new = path.read_text(encoding="utf-8")
    assert sha == hashlib.sha256(new.encode("utf-8")).hexdigest() and new != original
    changed = [line for line in difflib.unified_diff(original.splitlines(), new.splitlines(), lineterm="", n=0)
               if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    removed = [line[1:] for line in changed if line[0] == "-"]
    added = [line[1:] for line in changed if line[0] == "+"]
    assert len(removed) == 2 and len(added) == 1        # the one sentence that was wrapped over two lines
    assert "positive" in " ".join(removed) and "sentiment" in " ".join(removed)
    joined = " ".join(added)
    assert "positive" not in joined and "negative" not in joined and "sentiment" not in joined
    assert "the examples of each label" in joined and "not about the emotion" in joined
    # undoing the two substitutions by hand gives the pinned file back byte for byte
    undone = new.replace("the examples of each label", "the positive and\n  negative examples").replace(
        "not about the emotion", "not about sentiment")
    assert undone == original


@needs_clone
def test_writing_the_same_override_twice_gives_the_same_file_and_hash(tmp_path):
    assert write_reworded_procedure(EMOTION_CORPUS, CLONE, tmp_path) == write_reworded_procedure(EMOTION_CORPUS, CLONE, tmp_path)


@needs_jev
def test_the_planted_path_never_changes_procedure_or_the_rendering_default(tmp_path):
    unified_env.put_clone_first(CLONE)
    from jev_flywheel import steer

    before = (steer.PROCEDURE, steer.render_source.__kwdefaults__["path"], steer.render_source())
    with steering_procedure(PLANTED, CLONE, tmp_path) as sha:
        assert sha is None
        assert steer.PROCEDURE == before[0] and steer.render_source.__kwdefaults__["path"] == before[1]
        assert steer.render_source() == before[2]
    assert not list(tmp_path.iterdir())          # planted writes nothing at all


@needs_jev
def test_the_override_points_procedure_and_the_rendered_source_at_the_copy_for_one_run_then_restores_both(tmp_path):
    unified_env.put_clone_first(CLONE)
    from jev_flywheel import steer

    original_path, original_default = steer.PROCEDURE, steer.render_source.__kwdefaults__["path"]
    original_source = steer.render_source()
    with steering_procedure(EMOTION_CORPUS, CLONE, tmp_path) as sha:
        assert steer.PROCEDURE != original_path and steer.PROCEDURE.parent == tmp_path
        assert hashlib.sha256(steer.PROCEDURE.read_bytes()).hexdigest() == sha
        rendered = steer.render_source()
        assert "the examples of each label" in rendered and "positive and" not in rendered
    assert steer.PROCEDURE == original_path and steer.render_source.__kwdefaults__["path"] == original_default
    assert steer.render_source() == original_source


@needs_jev
def test_the_override_is_restored_even_when_the_run_raises(tmp_path):
    unified_env.put_clone_first(CLONE)
    from jev_flywheel import steer

    original = steer.PROCEDURE
    with pytest.raises(RuntimeError):
        with steering_procedure(EMOTION_CORPUS, CLONE, tmp_path):
            raise RuntimeError("boom")
    assert steer.PROCEDURE == original and steer.render_source.__kwdefaults__["path"] == original


@needs_clone
def test_the_pinned_clone_stays_clean_and_the_original_procedure_is_not_edited(tmp_path):
    original = (CLONE / "procedures" / "steer_scorecard.tac").read_bytes()
    write_reworded_procedure(EMOTION_CORPUS, CLONE, tmp_path)
    assert (CLONE / "procedures" / "steer_scorecard.tac").read_bytes() == original
    status = subprocess.run(["git", "-C", str(CLONE), "status", "--porcelain", "--untracked-files=all"],
                            check=True, stdout=subprocess.PIPE).stdout.decode()
    assert status == ""
    assert unified_env.verify_clone(CLONE).clean
