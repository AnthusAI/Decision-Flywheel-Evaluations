import pytest

from . import unified_cli, unified_env
from .unified_cli import UsageError, check_live_gates, parse_arms, parser


@pytest.fixture
def nothing_may_load(monkeypatch):
    """Any attempt to reach data, the clone or a client fails the spec."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a live gate let the run reach data or a client")

    monkeypatch.setattr(unified_env, "put_clone_first", forbidden)
    monkeypatch.setattr(unified_env, "ensure_decision_flywheel", forbidden)
    monkeypatch.setattr(unified_env, "install_network_guard", forbidden)


LIVE_OK = ["run", "--live", "--confirm", "--arms", "A,B", "--rounds", "1", "--spend-ledger", "ledger.json",
           "--request-ceiling", "9500", "--max-new-requests", "400", "--provider-model", "jev-1.13.0"]


def _without(argv, *flags):
    out = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in flags:
            skip = token not in ("--confirm", "--live")
            continue
        out.append(token)
    return out


def test_offline_is_the_default_and_the_analyst_is_openai_gpt_6_luna():
    args = parser().parse_args(["run"])
    assert args.live is False and args.confirm is False
    assert (args.analyst_provider, args.analyst_model) == ("openai", "gpt-6-luna")
    assert args.request_ceiling == 9500 and args.final is False


def test_the_first_live_step_passes_every_gate_with_a_400_request_cap():
    args = parser().parse_args(LIVE_OK)
    assert check_live_gates(args, parse_arms(args.arms), 100)["total"] == 400


@pytest.mark.parametrize("missing", ["--confirm", "--spend-ledger", "--provider-model", "--max-new-requests"])
def test_live_mode_refuses_before_loading_anything_when_a_gate_is_missing(missing, nothing_may_load):
    args = parser().parse_args(_without(LIVE_OK, missing))
    with pytest.raises(UsageError):
        unified_cli.run(args)


def test_a_ceiling_above_the_plans_9500_is_refused(nothing_may_load):
    argv = [t if t != "9500" else "9501" for t in LIVE_OK]
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(argv))


def test_a_cap_below_the_runs_upper_bound_is_refused(nothing_may_load):
    argv = [t if t != "400" else "399" for t in LIVE_OK]
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(argv))


def test_live_only_options_without_live_are_refused(nothing_may_load):
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(["run", "--confirm"]))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(["run", "--spend-ledger", "x.json"]))


def test_unknown_arms_are_refused():
    assert parse_arms("A+B,0") == ("0", "A+B")
    with pytest.raises(UsageError):
        parse_arms("A,C")


def test_main_reports_a_usage_error_with_exit_code_two(nothing_may_load, capsys):
    assert unified_cli.main(_without(LIVE_OK, "--confirm")) == 2
    assert "--confirm" in capsys.readouterr().err


ROUND_ONE_F = ["run", "--live", "--confirm", "--arms", "0,F,F-rand", "--rounds", "1",
               "--spend-ledger", "ledger.json", "--request-ceiling", "7500", "--max-new-requests", "1000",
               "--provider-model", "jev-1.13.0"]


def test_round_one_of_f_against_f_rand_passes_every_gate_within_a_1000_request_cap():
    args = parser().parse_args(ROUND_ONE_F)
    bound = check_live_gates(args, parse_arms(args.arms), 100)
    assert bound == {"0": 0, "F": 500, "F-rand": 200, "total": 700}


def test_the_new_arms_parse_and_the_default_stays_the_first_studys_five():
    assert parse_arms("F-rand,A-c+F,F,A-c") == ("F", "F-rand", "A-c", "A-c+F")
    assert parse_arms(parser().parse_args(["run"]).arms) == ("0", "A", "B-local", "B", "A+B")


def test_replay_needs_a_provider_model_and_a_run_dir_and_is_never_live(nothing_may_load):
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(["run", "--replay", "--run-dir", "x"]))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(["run", "--replay", "--provider-model", "jev-1.13.0"]))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(["run", "--provider-model", "jev-1.13.0", "--run-dir", "x"]))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(LIVE_OK + ["--replay"]))


FINAL = ["run", "--live", "--confirm", "--final", "--arms", "0,A-c,F,F-rand,A-c+F", "--run-dir", "x",
         "--spend-ledger", "ledger.final2.json", "--request-ceiling", "3000", "--max-new-requests", "3000",
         "--provider-model", "jev-1.13.0"]


def test_a_final_run_is_bounded_by_one_request_per_paper_600_item_per_bundle():
    args = parser().parse_args(FINAL)
    bound = check_live_gates(args, parse_arms(args.arms), 600)
    assert bound == {"0": 600, "A-c": 600, "F": 600, "F-rand": 600, "A-c+F": 600, "total": 3000}


def test_a_final_run_refuses_a_low_cap_arms_it_cannot_bundle_and_a_missing_run_dir(nothing_may_load):
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args([t if t != "3000" else "2999" for t in FINAL]))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args([t if t != "0,A-c,F,F-rand,A-c+F" else "0,B" for t in FINAL]))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(_without(FINAL, "--run-dir")))


@pytest.fixture
def no_clone(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a labeler gate let the run reach data or a client")

    monkeypatch.setattr(unified_env, "verify_clone", forbidden)
    monkeypatch.setattr(unified_env, "install_network_guard", forbidden)


COMMENTS_LIVE = ["comments", "--out", "c.jsonl", "--live", "--confirm", "--spend-ledger", "labeler-ledger.json",
                 "--max-calls", "300"]


@pytest.mark.parametrize("missing", ["--confirm", "--spend-ledger", "--max-calls"])
def test_the_live_labeler_refuses_before_loading_anything_when_a_gate_is_missing(missing, no_clone):
    with pytest.raises(UsageError):
        unified_cli.comments(parser().parse_args(_without(COMMENTS_LIVE, missing)))


def test_the_live_labeler_is_capped_at_300_calls_and_live_options_need_live(no_clone):
    with pytest.raises(UsageError):
        unified_cli.comments(parser().parse_args([t if t != "300" else "301" for t in COMMENTS_LIVE]))
    with pytest.raises(UsageError):
        unified_cli.comments(parser().parse_args(["comments", "--out", "c.jsonl", "--confirm"]))
    with pytest.raises(UsageError):
        unified_cli.comments(parser().parse_args(["comments", "--out", "c.jsonl", "--model", "gpt-6-luna"]))


FINAL_D = ["run", "--live", "--confirm", "--final", "--arms", "D", "--retriever", "embedding", "--rounds", "3",
           "--run-dir", "x", "--spend-ledger", "ledger.d-emb.json", "--request-ceiling", "900",
           "--max-new-requests", "900", "--provider-model", "jev-1.13.0"]


def test_arm_d_is_bounded_by_300_labeled_plus_600_paper_requests_per_variant():
    for retriever in ("embedding", "bm25"):
        args = parser().parse_args([t if t != "embedding" else retriever for t in FINAL_D])
        assert check_live_gates(args, parse_arms(args.arms), 600) == {"D": 900, "total": 900}


def test_arm_d_refuses_before_loading_anything_when_misused(nothing_may_load):
    bad = [
        [t if t != "900" else "899" for t in FINAL_D],                       # cap below the bound
        [t for t in FINAL_D if t != "--final"],                              # not final
        _without(FINAL_D, "--retriever"),                                    # no retriever
        [t if t != "D" else "D,A-c" for t in FINAL_D],                       # not alone
        _without(FINAL_D, "--confirm"),                                      # live gates still apply
    ]
    for argv in bad:
        with pytest.raises(UsageError):
            unified_cli.run(parser().parse_args(argv))
    with pytest.raises(UsageError):
        unified_cli.run(parser().parse_args(["run", "--final", "--run-dir", "x", "--arms", "A-c", "--retriever", "bm25"]))


def test_the_retriever_must_be_a_known_variant():
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--arms", "D", "--final", "--retriever", "tfidf"])


def test_the_corpus_defaults_to_planted_for_runs_and_comments():
    assert parser().parse_args(["run"]).corpus == "planted"
    assert parser().parse_args(["comments", "--out", "c.jsonl"]).corpus == "planted"


def test_an_unregistered_corpus_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--corpus", "nonesuch"])
    assert "invalid choice: 'nonesuch'" in capsys.readouterr().err



def test_selecting_the_emotion_corpus_on_a_run_or_comments_fails_with_a_clear_error(capsys):
    assert unified_cli.main(["run", "--corpus", "emotion"]) == 2
    assert "corpus 'emotion' cannot run a flywheel run yet" in capsys.readouterr().err
    assert unified_cli.main(["comments", "--corpus", "emotion", "--out", "c.jsonl"]) == 2
    assert "corpus 'emotion' cannot run comment generation yet" in capsys.readouterr().err
