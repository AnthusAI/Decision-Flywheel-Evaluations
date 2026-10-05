"""Contract specs for the explicit fake-only actual-learning runtime."""

import pytest

from . import unified_stream_runtime as runtime


def test_an_existing_runtime_workspace_is_not_reused_even_after_a_partial_run(tmp_path, monkeypatch):
    (tmp_path / "stream-ledger.json").write_text("old evidence")
    from . import unified_env
    monkeypatch.setattr(unified_env, "install_network_guard", lambda: pytest.fail("loaded runtime"))
    with pytest.raises(ValueError, match="fresh"):
        runtime.run_synthetic_runtime(tmp_path, seed=0, review_probability=1,
                                      max_new_requests=100, stream_size=32, heldout_size=8)
    assert (tmp_path / "stream-ledger.json").read_text() == "old evidence"


def test_the_synthetic_runtime_rejects_invalid_sizes_and_caps_before_loading_the_pinned_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_synthetic_corpus", lambda *_: pytest.fail("loaded optional runtime"))
    with pytest.raises(ValueError):
        runtime.run_synthetic_runtime(tmp_path, seed=0, review_probability=1.0,
                                      max_new_requests=-1, stream_size=32, heldout_size=8)
    with pytest.raises(ValueError):
        runtime.run_synthetic_runtime(tmp_path, seed=0, review_probability=1.0,
                                      max_new_requests=1, stream_size=True, heldout_size=8)


def test_the_synthetic_corpus_keeps_development_and_scoreboard_items_disjoint():
    corpus = runtime._synthetic_corpus(32, 8)
    splits = corpus.load(None, dev_size=8)

    assert set(splits.pool).isdisjoint(splits.dev100)
    assert set(splits.pool).isdisjoint(splits.paper600)
    assert set(splits.dev100).isdisjoint(splits.paper600)
