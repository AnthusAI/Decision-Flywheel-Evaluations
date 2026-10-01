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
