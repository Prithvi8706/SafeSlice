"""Aggregation tests.

Two of these guard against silent misattribution rather than against a crash:

  * test_parse_run_id_handles_policies_with_underscores - "static_equal_burst_0" must not parse
    as policy "static" scenario "equal". A misparse here would average two policies together and
    every downstream table would still look perfectly well formed.
  * test_paired_deltas_uses_only_shared_seeds - an unbalanced grid must not produce an interval
    computed from a different set of seeds per policy, which would no longer be paired at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.aggregate import (
    aggregate,
    load_runs,
    markdown_table,
    mean_ci,
    paired_deltas,
    parse_run_id,
)
from analysis.metrics import metrics_to_row
from experiments.run_experiment import run_once


# --------------------------------------------------------------------------- run ids


def test_parse_run_id_handles_policies_with_underscores():
    assert parse_run_id("static_equal_burst_0") == ("static_equal", "burst", 0)
    assert parse_run_id("threshold_adversarial_7") == ("threshold", "adversarial", 7)
    assert parse_run_id("epsilon_greedy_sawtooth_12") == ("epsilon_greedy", "sawtooth", 12)


def test_parse_run_id_rejects_an_unknown_scenario():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_run_id("threshold_notascenario_0")


# --------------------------------------------------------------------------- intervals


def test_single_sample_has_no_interval():
    """One seed must not be presented as a converged estimate."""
    out = mean_ci([1.5])
    assert out["n"] == 1
    assert out["mean"] == pytest.approx(1.5)
    assert np.isnan(out["ci95_half"])


def test_interval_covers_the_mean_and_shrinks_with_n():
    small = mean_ci([1.0, 2.0, 3.0, 4.0])
    large = mean_ci([1.0, 2.0, 3.0, 4.0] * 25)
    assert small["ci95_lo"] < small["mean"] < small["ci95_hi"]
    assert large["ci95_half"] < small["ci95_half"]


def test_known_t_interval():
    """Pinned against a hand-checked value so a scipy API change cannot pass silently."""
    out = mean_ci([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert out["mean"] == pytest.approx(5.0)
    assert out["sd"] == pytest.approx(2.13809, rel=1e-4)   # ddof=1
    # sd = sqrt(32/7) = 2.1380899, t(0.975, df=7) = 2.3646243,
    # half width = 2.3646243 * 2.1380899 / sqrt(8) = 1.7874879
    assert out["ci95_half"] == pytest.approx(1.7874879, rel=1e-6)


# --------------------------------------------------------------------------- pairing


def _fake_runs():
    rows = []
    for policy, base in (("static_equal", 0.10), ("threshold", 0.20)):
        # threshold is missing seed 3, so only seeds 0..2 are shared
        seeds = [0, 1, 2, 3] if policy == "static_equal" else [0, 1, 2]
        for seed in seeds:
            rows.append(
                {
                    "policy": policy,
                    "scenario": "burst",
                    "seed": seed,
                    "mean_reward": base + 0.01 * seed,
                    "embb_goodput_mbps": 1.0 + base,
                    "sla_violation_rate": 0.0,
                    "urllc_rtt_p50_ms": 2.0,
                    "urllc_rtt_p95_ms": 3.0,
                    "urllc_rtt_p99_ms": 3.5,
                    "be_goodput_mbps": 3.0,
                    "total_drops": 100,
                    "guardrail_intervention_rate": 0.0,
                    "decision_latency_mean_ms": 0.05,
                    "decision_latency_p99_ms": 0.2,
                    "link_util_mean": 0.9,
                    "n_steps": 270,
                }
            )
    return pd.DataFrame(rows)


def test_paired_deltas_uses_only_shared_seeds():
    d = paired_deltas(_fake_runs(), baseline="static_equal")
    row = d[(d["policy"] == "threshold") & (d["metric"] == "mean_reward")].iloc[0]
    assert row["n_paired"] == 3          # not 4: seed 3 exists only for the baseline
    assert row["mean_delta"] == pytest.approx(0.10)


def test_paired_delta_of_a_constant_offset_is_certain():
    """A perfectly consistent offset has zero variance, so the interval excludes zero."""
    d = paired_deltas(_fake_runs(), baseline="static_equal")
    row = d[(d["policy"] == "threshold") & (d["metric"] == "mean_reward")].iloc[0]
    assert row["ci95_half"] == pytest.approx(0.0, abs=1e-12)
    assert bool(row["separated_from_zero"])


def test_paired_deltas_rejects_an_absent_baseline():
    with pytest.raises(KeyError, match="not in results"):
        paired_deltas(_fake_runs(), baseline="linucb")


# --------------------------------------------------------------------------- round trip


def test_load_runs_recomputes_what_run_once_reported(tmp_path, cfg):
    """The tables must not depend on the cache written at run time."""
    _, metrics, _ = run_once(
        policy_name="threshold", scenario="burst", seed=0, cfg=cfg, out_dir=tmp_path
    )
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    got = runs.iloc[0].to_dict()
    want = metrics_to_row(metrics)
    for key, value in want.items():
        assert got[key] == pytest.approx(value), key


def test_markdown_table_marks_single_seed_runs(tmp_path, cfg):
    run_once(policy_name="static_safe", scenario="burst", seed=0, cfg=cfg, out_dir=tmp_path)
    runs = load_runs(tmp_path)
    table = markdown_table(aggregate(runs), "burst")
    assert "(n=1)" in table
    # With one seed every interval is unknown, and the table must say so rather than print ±0.
    assert "±" not in table
