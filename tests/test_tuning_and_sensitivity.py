"""Tests for the sweep and sensitivity machinery.

Mostly these check the parts that decide what gets REPORTED rather than what gets computed:
which value a sweep declares best, and whether the sensitivity study correctly announces that a
result is or is not robust. A bug in the summarising layer produces a confident, well formatted,
wrong claim, which is worse than a crash.

The two that guard methodology:

  * test_sweep_defaults_to_training_seeds - the sweep must not default to the evaluation seeds.
    Tuning on the seeds you are scored on is selection on the test set, and nothing about the
    output would look wrong if it happened.
  * test_unstable_ranking_is_announced_as_unstable - if the winner changes across the sweep, the
    study has to say so in the text it emits, not leave it to a reader to diff the rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config_loader import load_config
from experiments import sensitivity, tune


# --------------------------------------------------------------------------- tune


def test_sweep_defaults_to_training_seeds(monkeypatch):
    """No argument means train_seeds, never the evaluation seeds."""
    seen = []

    def fake_run_once(policy_name, scenario, seed, cfg=None, write=True, **kw):
        seen.append(int(seed))
        from analysis.metrics import RunMetrics

        return None, RunMetrics(
            policy=policy_name, scenario=scenario, seed=seed, n_steps=1,
            urllc_rtt_p50=2.0, urllc_rtt_p95=3.0, urllc_rtt_p99=3.0,
            sla_violation_rate=0.0, embb_goodput_mbps=1.0, be_goodput_mbps=1.0,
            total_drops=0, guardrail_intervention_rate=0.0,
            decision_latency_mean_ms=0.1, decision_latency_p99_ms=0.2,
            mean_reward=0.5, link_util_mean=0.9,
        ), None

    monkeypatch.setattr(tune, "run_once", fake_run_once)
    tune.sweep("linucb", scenarios=["burst"], values=[0.1], quiet=True)

    cfg = load_config()
    assert sorted(set(seen)) == sorted(cfg.experiment.train_seeds)
    assert not (set(seen) & set(cfg.experiment.seeds))


def test_sweep_rejects_a_policy_with_no_sweep_defined():
    with pytest.raises(KeyError, match="no sweep defined"):
        tune.sweep("static_equal")


def _sweep_frame(rewards_by_value):
    rows = []
    for value, rewards in rewards_by_value.items():
        for seed, r in enumerate(rewards):
            rows.append(
                {
                    "policy": "linucb", "param": "policy.linucb.alpha", "value": value,
                    "scenario": "burst", "seed": seed, "mean_reward": r,
                    "sla_violation_rate": 0.01, "embb_goodput_mbps": 3.0,
                    "urllc_rtt_p95_ms": 4.0, "guardrail_intervention_rate": 0.02,
                    "decision_latency_mean_ms": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_best_value_is_the_argmax_of_mean_reward():
    runs = _sweep_frame({0.1: [0.1, 0.2], 0.5: [0.4, 0.5], 1.0: [0.3, 0.3]})
    summary = tune.summarise(runs)
    assert tune.best_value(summary) == pytest.approx(0.5)
    assert list(summary["value"]) == [0.1, 0.5, 1.0]     # sorted, so tables read in order


def test_markdown_marks_the_selected_value():
    runs = _sweep_frame({0.1: [0.1, 0.2], 0.5: [0.4, 0.5]})
    text = tune.markdown(tune.summarise(runs), "policy.linucb.alpha")
    assert "**0.5**" in text
    assert "Selected: `policy.linucb.alpha` = 0.5" in text


def test_summarise_reports_an_interval_per_value():
    runs = _sweep_frame({0.1: [0.1, 0.2, 0.3], 0.5: [0.4, 0.5, 0.6]})
    summary = tune.summarise(runs)
    assert (summary["n_runs"] == 3).all()
    assert (summary["mean_reward_ci95"] > 0).all()


# --------------------------------------------------------------------------- sensitivity


def _sensitivity_frame(winner_by_cell):
    """winner_by_cell: {(w_sla, w_drop): winning_policy}. Everyone else scores lower."""
    rows = []
    for (w_sla, w_drop), winner in winner_by_cell.items():
        for policy in ("static_equal", "threshold", "linucb"):
            for seed in range(3):
                rows.append(
                    {
                        "w_sla": w_sla, "w_drop": w_drop, "policy": policy,
                        "scenario": "burst", "seed": seed,
                        "mean_reward": 1.0 if policy == winner else 0.1,
                        "sla_violation_rate": 0.0, "embb_goodput_mbps": 3.0,
                        "urllc_rtt_p95_ms": 4.0, "guardrail_intervention_rate": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_stable_ranking_is_announced_as_stable():
    runs = _sensitivity_frame({(1.0, 0.0): "linucb", (2.0, 0.0): "linucb"})
    text = sensitivity.markdown_rankings(sensitivity.rankings(runs))
    assert "stable across the whole sweep" in text
    assert "`linucb` wins in all 2 cells" in text


def test_unstable_ranking_is_announced_as_unstable():
    """The failure mode this study exists to catch must be stated, not left to be noticed."""
    runs = _sensitivity_frame({(1.0, 0.0): "linucb", (4.0, 0.0): "threshold"})
    text = sensitivity.markdown_rankings(sensitivity.rankings(runs))
    assert "NOT stable" in text
    assert "depends on the reward weights" in text


def test_ranking_orders_policies_by_reward_within_a_cell():
    runs = _sensitivity_frame({(1.0, 0.0): "threshold"})
    rank = sensitivity.rankings(runs)
    assert rank.iloc[0]["winner"] == "threshold"
    assert rank.iloc[0]["ranking"].startswith("threshold >")
    assert rank.iloc[0]["margin_first_to_second"] == pytest.approx(0.9)


def test_physical_table_keeps_cells_separate():
    """Physical metrics are comparable across cells; they must not be collapsed into one row."""
    runs = _sensitivity_frame({(1.0, 0.0): "linucb", (2.0, 0.5): "linucb"})
    phys = sensitivity.physical_table(runs)
    assert len(phys) == 6                                   # 3 policies x 2 cells
    assert set(zip(phys["w_sla"], phys["w_drop"])) == {(1.0, 0.0), (2.0, 0.5)}

# --------------------------------------------------------------------------- resume


def _fake_metrics(policy, scenario, seed, reward=0.5):
    from analysis.metrics import RunMetrics

    return RunMetrics(
        policy=policy, scenario=scenario, seed=seed, n_steps=1,
        urllc_rtt_p50=2.0, urllc_rtt_p95=3.0, urllc_rtt_p99=3.0,
        sla_violation_rate=0.0, embb_goodput_mbps=1.0, be_goodput_mbps=1.0,
        total_drops=0, guardrail_intervention_rate=0.0,
        decision_latency_mean_ms=0.1, decision_latency_p99_ms=0.2,
        mean_reward=reward, link_util_mean=0.9,
    )


def test_study_writes_each_row_as_it_goes(monkeypatch, tmp_path):
    """An interrupted study must keep what it already finished.

    The first real run of this study was killed for memory after an hour and lost everything,
    because it wrote its CSV only at the end. Hence this test.
    """
    calls = []

    def fake_run_once(policy_name, scenario, seed, cfg=None, write=True, **kw):
        calls.append((policy_name, scenario, seed))
        if len(calls) == 3:
            raise KeyboardInterrupt("simulated kill")
        return None, _fake_metrics(policy_name, scenario, seed), None

    monkeypatch.setattr(sensitivity, "run_once", fake_run_once)
    out = tmp_path / "sens.csv"
    with pytest.raises(KeyboardInterrupt):
        sensitivity.study(
            policies=["static_equal", "threshold"], w_sla_values=[1.0], w_drop_values=[0.0],
            scenarios=["burst"], seeds=[100, 101], quiet=True, out_csv=out,
        )
    survived = pd.read_csv(out)
    assert len(survived) == 2, "rows finished before the interruption were lost"


def test_resume_skips_rows_already_present(monkeypatch, tmp_path):
    ran = []

    def fake_run_once(policy_name, scenario, seed, cfg=None, write=True, **kw):
        ran.append((policy_name, scenario, seed))
        return None, _fake_metrics(policy_name, scenario, seed), None

    monkeypatch.setattr(sensitivity, "run_once", fake_run_once)
    out = tmp_path / "sens.csv"
    kwargs = dict(
        policies=["static_equal"], w_sla_values=[1.0], w_drop_values=[0.0],
        scenarios=["burst"], quiet=True, out_csv=out,
    )
    sensitivity.study(seeds=[100], **kwargs)
    assert len(ran) == 1

    ran.clear()
    df = sensitivity.study(seeds=[100, 101], resume=True, **kwargs)
    assert ran == [("static_equal", "burst", 101)], "resume re-ran a completed cell"
    assert len(df) == 2, "resume dropped the rows it was meant to keep"


def test_without_resume_the_file_is_started_fresh(monkeypatch, tmp_path):
    """A non-resumed study must not silently append to a previous study's rows."""

    def fake_run_once(policy_name, scenario, seed, cfg=None, write=True, **kw):
        return None, _fake_metrics(policy_name, scenario, seed), None

    monkeypatch.setattr(sensitivity, "run_once", fake_run_once)
    out = tmp_path / "sens.csv"
    kwargs = dict(
        policies=["static_equal"], w_sla_values=[1.0], w_drop_values=[0.0],
        scenarios=["burst"], seeds=[100], quiet=True, out_csv=out,
    )
    sensitivity.study(**kwargs)
    sensitivity.study(**kwargs)
    assert len(pd.read_csv(out)) == 1
