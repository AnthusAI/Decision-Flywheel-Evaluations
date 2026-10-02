"""Specs for reusing the static study's cached zero-shot Emotion answers (offline, no network).

They need the pinned Jev-Flywheel clone, so they skip in an interpreter without it; run them with
``make unified-flywheel-test UF_PYTHON=/path/to/python``. A tiny synthetic sqlite file with the static
study's schema stands in for ``.data/dev/emotion.scoreboard2.sqlite``; one spec reads the real file and
skips when it is absent.
"""
import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from . import unified_env

if not unified_env.jev_dependencies_available():
    pytest.skip("needs scikit-learn and Tactus (Jev-Flywheel's interpreter)", allow_module_level=True)
try:
    unified_env.ensure_decision_flywheel()
    CLONE = unified_env.put_clone_first()
except unified_env.EnvironmentProblem as problem:
    pytest.skip(f"pinned Jev-Flywheel clone unavailable: {problem}", allow_module_level=True)

unified_env.install_network_guard()

from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration  # noqa: E402
from decision_flywheel.budget import ContextBudget, build_context_plan  # noqa: E402
from decision_flywheel.context import PerLabelLexicalRetrieval  # noqa: E402
from decision_flywheel.models import DecisionTask, Item  # noqa: E402
from jev_flywheel.answers import AnswerCache, question_hash  # noqa: E402
from jev_flywheel.jev import JevSession  # noqa: E402
from jev_flywheel.scoring import predict  # noqa: E402

from .unified_corpus import EMOTION_CORPUS  # noqa: E402
from .unified_emotion import (EMOTION_INSTRUCTIONS, EMOTION_LABELS, EMOTION_REPO_ROOT, EMOTION_TASK_NAME,  # noqa: E402
                              StudyShapedZeroShotClient, build_emotion_splits, study_zero_shot_state)
from .unified_fake_jev import FakeJevAsync, FakeJevCore  # noqa: E402
from .unified_import import (ImportReport, ZeroShotImportError, import_zero_shot, read_zero_shot_cells,  # noqa: E402
                             wire_fingerprint)
from .unified_loop import Engines, HarnessError, RunConfig, UnifiedFlywheel  # noqa: E402

MODEL = "jev-1.13.0"
ZERO = f"scoreboard:jev:{MODEL}@transport-x:zero:0"


def _seed_question():
    return EMOTION_CORPUS.seed_scorecard().questions()[EMOTION_CORPUS.score_name]


def _payload(choice="joy", model=MODEL):
    probabilities = {label: 0.0 for label in EMOTION_LABELS}
    probabilities[choice] = 0.7
    probabilities["love"] = 0.3 if choice != "love" else 0.7
    return {"choice": choice, "confidence": 0.61, "probabilities": probabilities, "reported_model": model,
            "model": f"jev:{model}@transport-x", "status": "success", "usage": {"input_tokens": 300.0}}


def _static_fingerprint(text):
    """The fingerprint core computes for a zero-shot request: the empty-context plan of the static task."""
    task = DecisionTask(EMOTION_TASK_NAME, EMOTION_LABELS, EMOTION_INSTRUCTIONS)
    plan = build_context_plan(task, Item("x", {"text": text}), [], PerLabelLexicalRetrieval(),
                              budget=ContextBudget(per_label=0))
    return plan.token_accounting.serialized_request_fingerprint


def make_scoreboard(path, texts, *, payloads=None, fingerprint=None, model=MODEL, extra_condition=True):
    """A sqlite file with the static study's three tables, one zero-shot cell per item (and a decoy condition)."""
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE requests (request_id TEXT PRIMARY KEY, task_fingerprint TEXT NOT NULL, state_fingerprint TEXT NOT NULL,
            dataset_revision TEXT NOT NULL, model_identity TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL,
            payload TEXT, token TEXT);
        CREATE TABLE cells (cell_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, condition TEXT NOT NULL, draw INTEGER NOT NULL,
            ord TEXT NOT NULL, target_id TEXT NOT NULL);""")
    for n, (item_id, text) in enumerate(texts.items()):
        wire = (fingerprint or _static_fingerprint)(text)
        payload = (payloads or {}).get(item_id, _payload(model=model))
        for tag, condition in (("z", ZERO), ("d", f"scoreboard:jev:{MODEL}@transport-x:random:4")):
            if tag == "d" and not extra_condition:
                continue
            request_id = f"{tag}{n}"
            decoy = _payload("anger")
            connection.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?)",
                               (request_id, "t", wire if tag == "z" else "other", "r", "m", "success", 1,
                                json.dumps(payload if tag == "z" else decoy), None))
            connection.execute("INSERT INTO cells VALUES (?,?,?,?,?,?)",
                               (f"c-{request_id}", request_id, condition, 0, "0", item_id))
    connection.commit()
    connection.close()


TEXTS = {f"test-{n}": f"a short synthetic text number {n}" for n in range(5)}


@pytest.fixture
def scoreboard(tmp_path):
    path = tmp_path / "scoreboard.sqlite"
    make_scoreboard(path, TEXTS)
    return path


def _import(cache, path, ids=None, **kw):
    return import_zero_shot(cache, path, name=EMOTION_CORPUS.score_name, question=_seed_question(),
                            item_ids=list(ids if ids is not None else TEXTS), expected_model=MODEL,
                            state_fn=study_zero_shot_state, texts=TEXTS, **kw)


# ---- the request the harness sends is the request the static study sent ----------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        raise RuntimeError("recorded; nothing is sent")


def test_the_core_adapter_sends_the_static_zero_shot_request_for_an_empty_context():
    recorder = _Recorder()
    adapter = JevAdapter(recorder, configuration=JevConfiguration(model=MODEL))
    task = DecisionTask(EMOTION_TASK_NAME, EMOTION_LABELS, EMOTION_INSTRUCTIONS)
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.decide(task, Item("x", {"text": "some text"}), []))
    (state, questions), = recorder.calls
    assert state == study_zero_shot_state("some text") == {"labeled_examples": [], "target": {"text": "some text"}}
    assert questions == {EMOTION_TASK_NAME: _seed_question()}
    assert list(questions[EMOTION_TASK_NAME]["criteria"]) == list(EMOTION_LABELS)


def test_the_harness_sends_the_same_state_and_question_bytes_as_the_core_adapter():
    class Recording:
        def __init__(self):
            self.calls = []

        async def system_one(self, *, state, questions):
            self.calls.append((state, questions))
            raise RuntimeError("recorded; nothing is sent")

    inner = Recording()
    session = JevSession(client_factory=lambda: StudyShapedZeroShotClient(inner))
    questions = {EMOTION_CORPUS.score_name: _seed_question()}
    with pytest.raises(RuntimeError):
        asyncio.run(session.ask("some text", questions))
    (harness_state, harness_questions), = inner.calls
    core = _Recorder()
    task = DecisionTask(EMOTION_TASK_NAME, EMOTION_LABELS, EMOTION_INSTRUCTIONS)
    with pytest.raises(RuntimeError):
        asyncio.run(JevAdapter(core, configuration=JevConfiguration(model=MODEL)).decide(task, Item("x", {"text": "some text"}), []))
    (core_state, core_questions), = core.calls
    as_bytes = lambda state, questions: json.dumps({"state": state, "questions": questions}, ensure_ascii=False,
                                                   separators=(",", ":"))
    assert as_bytes(harness_state, harness_questions) == as_bytes(core_state, core_questions)


def test_the_wire_fingerprint_the_importer_checks_equals_the_core_serialized_request_fingerprint():
    text = "i feel quietly happy today"
    assert wire_fingerprint(study_zero_shot_state(text), EMOTION_CORPUS.score_name, _seed_question()) == _static_fingerprint(text)


def test_the_seed_question_hash_equals_the_hash_of_the_static_studys_question_body():
    static = {"type": "choice", "instructions": EMOTION_INSTRUCTIONS, "criteria": {label: None for label in EMOTION_LABELS}}
    assert _seed_question() == static
    assert question_hash(_seed_question()) == question_hash(static)


def test_the_planted_corpus_keeps_the_plain_state_and_the_fixture_seed():
    from .unified_corpus import PLANTED
    assert PLANTED.seed_score is None and PLANTED.zero_shot_client is None and PLANTED.cached_zero_shot is None
    assert PLANTED.seed_scorecard() is None and PLANTED.zero_shot_state is None


def test_a_state_that_already_carries_a_target_is_not_reshaped():
    seen = []

    class Inner:
        async def system_one(self, *, state, questions):
            seen.append(state)

    state = {"labeled_examples": [{"text": "a", "label": "joy"}], "target": {"text": "b"}}
    asyncio.run(StudyShapedZeroShotClient(Inner()).system_one(state=state, questions={}))
    assert seen == [state]


# ---- the seed head ----------------------------------------------------------------------------------

def test_the_seed_head_reproduces_jevs_own_choice_and_distribution():
    score = EMOTION_CORPUS.seed_scorecard().score(EMOTION_CORPUS.score_name)
    assert score.decision.features == [f"self.holistic.clr.{label}" for label in EMOTION_LABELS[:-1]]
    for label in EMOTION_LABELS:
        probabilities = {other: 0.04 for other in EMOTION_LABELS}
        probabilities[label] = 0.8
        answer = {"type": "choice", "choice": label, "confidence": 0.8, "probabilities": probabilities}
        result = predict(score, {EMOTION_CORPUS.score_name: answer})
        assert result.value == label
        assert result.metadata["decision"]["probabilities"][label] == pytest.approx(0.8, abs=1e-6)


# ---- the importer --------------------------------------------------------------------------------

def test_the_importer_stores_each_answer_under_the_item_name_and_question_hash(tmp_path, scoreboard):
    cache = AnswerCache(tmp_path / "answers.jsonl")
    report = _import(cache, scoreboard)
    assert isinstance(report, ImportReport) and report.imported == 5 and report.wire_mismatches == 0 and report.wire_checked
    assert report.cells_in_file == 5 and report.question_hash == question_hash(_seed_question())
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert {(r["item_id"], r["name"], r["qhash"]) for r in rows} == {(i, EMOTION_TASK_NAME, report.question_hash) for i in TEXTS}
    answer = cache.get("test-2", EMOTION_TASK_NAME, _seed_question())
    assert answer == {"type": "choice", "choice": "joy", "confidence": 0.61,
                      "probabilities": _payload()["probabilities"]}
    assert rows[0]["model"] == MODEL


def test_the_decoy_condition_is_never_read(tmp_path, scoreboard):
    cells = read_zero_shot_cells(scoreboard)
    assert set(cells) == set(TEXTS) and all(c["payload"]["choice"] == "joy" for c in cells.values())


def test_importing_again_is_a_no_op_and_does_not_grow_the_file(tmp_path, scoreboard):
    cache = AnswerCache(tmp_path / "answers.jsonl")
    _import(cache, scoreboard)
    before = (tmp_path / "answers.jsonl").read_text()
    again = _import(AnswerCache(tmp_path / "answers.jsonl"), scoreboard)
    assert (again.imported, again.already_present) == (0, 5)
    assert (tmp_path / "answers.jsonl").read_text() == before


def test_the_import_is_text_free_and_only_reads_the_requested_ids(tmp_path, scoreboard):
    cache = AnswerCache(tmp_path / "answers.jsonl")
    report = _import(cache, scoreboard, ids=["test-1", "test-3", "test-999"])
    assert (report.imported, report.not_in_file, report.requested) == (2, 1, 3)
    written = (tmp_path / "answers.jsonl").read_text()
    assert not any(text in written for text in TEXTS.values())
    assert cache.get("test-0", EMOTION_TASK_NAME, _seed_question()) is None


def test_an_item_recorded_under_a_different_request_is_counted_and_not_imported(tmp_path):
    path = tmp_path / "scoreboard.sqlite"
    make_scoreboard(path, TEXTS, fingerprint=lambda text: wire_fingerprint(
        {"text": text}, EMOTION_TASK_NAME, _seed_question()))   # the plain {"text"} state, not the study's
    cache = AnswerCache(tmp_path / "answers.jsonl")
    report = _import(cache, path)
    assert (report.imported, report.wire_mismatches) == (0, 5) and len(cache) == 0


def test_answers_from_another_model_version_are_refused(tmp_path):
    path = tmp_path / "scoreboard.sqlite"
    make_scoreboard(path, TEXTS, model="jev-9.9.9")
    with pytest.raises(ZeroShotImportError, match="not 'jev-1.13.0'"):
        _import(AnswerCache(tmp_path / "answers.jsonl"), path)


def test_a_cached_answer_with_an_unknown_label_is_refused(tmp_path):
    path = tmp_path / "scoreboard.sqlite"
    make_scoreboard(path, TEXTS, payloads={"test-0": {**_payload(), "choice": "disgust"}})
    with pytest.raises(ZeroShotImportError, match="not an option"):
        _import(AnswerCache(tmp_path / "answers.jsonl"), path)


def test_a_file_without_exactly_one_zero_shot_condition_is_refused(tmp_path):
    path = tmp_path / "empty.sqlite"
    make_scoreboard(path, {})
    with pytest.raises(ZeroShotImportError, match="exactly one condition"):
        read_zero_shot_cells(path)


# ---- the harness seeds its cache before arm 0 ---------------------------------------------------------

def _synthetic_splits():
    candidates = [(f"train-{n}", EMOTION_LABELS[n % 6], f"candidate text number {n} about feelings") for n in range(120)]
    scoreboard = [(f"test-{n}", EMOTION_LABELS[(n * 7) % 6], f"scoreboard text number {n} about moods") for n in range(120)]
    return build_emotion_splits(candidates, scoreboard, dev_size=10, final_size=20)[0]


class _Core(FakeJevCore):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.states = []

    def answer(self, state, questions):
        self.states.append((state, dict(questions)))
        return super().answer(state, questions)


def _emotion_flywheel(tmp_path, *, mode="live"):
    splits = _synthetic_splits()
    texts = {i: splits.items[i].text for i in splits.test}
    path = tmp_path / "scoreboard.sqlite"
    make_scoreboard(path, texts)
    corpus = replace(EMOTION_CORPUS, unsupported_reason=None, cached_zero_shot=path,
                     load_splits=lambda fixtures, dev_size: splits)
    core = _Core(cues=dict(corpus.fake_jev_cues))
    engines = Engines(lambda: FakeJevAsync(core), None, "spec-engine", mode)
    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / "run", corpus=corpus, replay=True, provider_model=MODEL,
                    arms=("0",), rounds=1, per_round=12, dev_size=10)
    return UnifiedFlywheel(cfg, engines=engines), splits, core


def test_an_emotion_run_seeds_its_cache_with_the_slice_answers_before_arm_zero_and_sends_nothing(tmp_path):
    flywheel, splits, core = _emotion_flywheel(tmp_path)
    report = flywheel.zero_shot_import
    slice_ids = sorted(set(splits.dev100) | set(splits.paper600))
    assert report["imported"] == len(slice_ids) == 30 and report["wire_mismatches"] == 0 and core.states == []
    holistic = flywheel.v1.questions()[flywheel.corpus.score_name]
    assert all(flywheel.cache.get(i, flywheel.corpus.score_name, holistic) is not None for i in slice_ids)
    assert flywheel.cache.get("train-0", flywheel.corpus.score_name, holistic) is None   # pool items are paid for live
    assert flywheel.cache.plan(flywheel.eval_ids, {flywheel.corpus.score_name: holistic}).is_free


def test_a_live_zero_shot_request_for_a_pool_item_has_the_static_studys_shape(tmp_path):
    flywheel, _, core = _emotion_flywheel(tmp_path)
    flywheel._fill_zero_shot(["train-0"], flywheel.v1.questions())
    (state, questions), = core.states
    assert state == study_zero_shot_state(flywheel.splits.items["train-0"].text) and list(questions) == [EMOTION_TASK_NAME]


def test_the_offline_fake_run_never_imports_real_answers(tmp_path):
    flywheel, _, _ = _emotion_flywheel(tmp_path, mode="offline-fake")
    assert flywheel.zero_shot_import is None and len(flywheel.cache) == 0


def test_a_wire_mismatch_stops_the_run_instead_of_reusing_the_answers(tmp_path, monkeypatch):
    from . import unified_loop
    real = unified_loop.import_zero_shot

    def tampered(cache, path, **kw):
        return real(cache, path, **{**kw, "state_fn": lambda text: {"text": text}})

    monkeypatch.setattr(unified_loop, "import_zero_shot", tampered)
    with pytest.raises(HarnessError, match="different request"):
        _emotion_flywheel(tmp_path)


# ---- the real file (skipped when absent) -------------------------------------------------------------------

_REAL_DB = Path(EMOTION_REPO_ROOT) / ".data" / "dev" / "emotion.scoreboard2.sqlite"
_REAL_ARROW = Path(EMOTION_REPO_ROOT) / ".data" / "huggingface" / "dair-ai___emotion"


@pytest.mark.skipif(not (_REAL_DB.is_file() and _REAL_ARROW.is_dir()), reason="the local Emotion cache or scoreboard is absent")
def test_the_real_zero_shot_answers_cover_all_two_thousand_test_ids_under_the_harnesss_request(tmp_path):
    pytest.importorskip("pyarrow")
    from .unified_emotion import load_emotion_splits

    splits, _ = load_emotion_splits()
    cells = read_zero_shot_cells(_REAL_DB)
    assert len(cells) == 2000 and set(cells) == set(splits.test)
    cache = AnswerCache(tmp_path / "answers.jsonl")
    ids = sorted(splits.test)
    report = import_zero_shot(cache, _REAL_DB, name=EMOTION_CORPUS.score_name, question=_seed_question(), item_ids=ids,
                              expected_model=MODEL, state_fn=study_zero_shot_state,
                              texts={i: splits.items[i].text for i in ids})
    assert (report.imported, report.wire_mismatches, report.not_in_file) == (2000, 0, 0)
    assert all(wire_fingerprint(study_zero_shot_state(splits.items[i].text), EMOTION_TASK_NAME, _seed_question())
               == cells[i]["state_fingerprint"] for i in ids)
    again = import_zero_shot(cache, _REAL_DB, name=EMOTION_CORPUS.score_name, question=_seed_question(), item_ids=ids,
                             expected_model=MODEL)
    assert (again.imported, again.already_present) == (0, 2000)
