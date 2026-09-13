"""Tests for experiments/compare_sim_vs_ovs.py.

Written before the comparison was run, so they check the machinery independently of the answer:
that the simulator really is given constant, jitter-free load at the measured base RTT; that it keeps
the same step window as the testbed; that the error arithmetic is right; and that the rule recorded in
docs/PLAN_TESTBED.md section 2.11 is implemented band for band.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.compare_sim_vs_ovs import (
    RULE_METRICS,
    TRACK_THRESHOLD,
    classify_match,
    constant_load_cfg,
    errors_for_mode,
    run_sim_level,
    load_testbed_rows,
)
from traffic.traces import build_trace

LOADS = {"urllc": 3.0, "embb": 10.0, "be": 4.0}


def test_matched_config_has_constant_jitter_free_load_and_the_measured_base_rtt():
    cfg = constant_load_cfg("equal", LOADS, duration_s=120, warmup_s=10, base_rtt_ms=0.093)
    assert str(cfg.sim.excess_sharing) == "equal"
    assert float(cfg.link.base_rtt_ms) == pytest.approx(0.093)
    trace = build_trace(cfg, seed=0)
    offered = trace.step_offered_bps
    assert offered.shape[1] == 120
    for qi, mbps in enumerate((3.0, 10.0, 4.0)):
        assert np.allclose(offered[qi], mbps * 1e6), "load must be constant with zero jitter"


def test_matched_config_is_identical_across_seeds_so_one_run_per_cell_is_enough():
    cfg = constant_load_cfg("demand_proportional", LOADS, 120, 10, 0.093)
    assert np.array_equal(build_trace(cfg, 0).offered_bps, build_trace(cfg, 7).offered_bps)


def test_simulator_window_matches_the_testbed_window():
    """Testbed keeps seconds 10..119, i.e. 109 one-second steps."""
    cfg = constant_load_cfg("demand_proportional", LOADS, 120, 10, 0.093)
    r = run_sim_level(cfg, 0, duration_s=120, warmup_s=10)
    assert r["n_steps"] == 109


def test_simulator_run_is_deterministic():
    cfg = constant_load_cfg("min_rate_proportional", LOADS, 120, 10, 0.093)
    assert run_sim_level(cfg, 3, 120, 10) == run_sim_level(cfg, 3, 120, 10)


# --------------------------------------------------------------------------- error arithmetic


def _rows(values_by_metric):
    n = len(next(iter(values_by_metric.values())))
    rows = []
    for i in range(n):
        r = {"level_index": i}
        for m, vals in values_by_metric.items():
            r[m] = vals[i]
            r[f"{m}_ci95_half"] = 0.1
        rows.append(r)
    return rows


TB = _rows({
    "embb_goodput_l2_mbps": [2.0, 3.0, 4.0, 4.5, 5.0],   # range 3.0
    "be_goodput_l2_mbps": [4.0, 3.5, 3.0, 2.5, 2.0],     # range 2.0
    "urllc_rtt_p50_ms": [0.1, 2.0, 4.0, 5.0, 8.1],       # range 8.0
})


def test_errors_are_mae_normalised_by_the_testbed_range():
    sim = _rows({
        "embb_goodput_l2_mbps": [2.3, 3.3, 4.3, 4.8, 5.3],   # every level off by 0.3
        "be_goodput_l2_mbps": [4.0, 3.5, 3.0, 2.5, 2.0],     # exact
        "urllc_rtt_p50_ms": [0.1, 2.0, 4.0, 5.0, 12.1],      # one level off by 4.0
    })
    e = errors_for_mode(sim, TB)
    assert e["embb_goodput_l2_mbps"]["mae"] == pytest.approx(0.3)
    assert e["embb_goodput_l2_mbps"]["nmae"] == pytest.approx(0.1)
    assert e["be_goodput_l2_mbps"]["nmae"] == pytest.approx(0.0)
    assert e["urllc_rtt_p50_ms"]["mae"] == pytest.approx(0.8)
    assert e["urllc_rtt_p50_ms"]["nmae"] == pytest.approx(0.1)
    assert e["urllc_rtt_p50_ms"]["max_abs_error"] == pytest.approx(4.0)


def test_track_threshold_is_inclusive_at_one_quarter_of_the_effect():
    assert TRACK_THRESHOLD == 0.25
    exact_quarter = _rows({
        "embb_goodput_l2_mbps": [v + 0.75 for v in (2.0, 3.0, 4.0, 4.5, 5.0)],   # 0.75 / 3.0
        "be_goodput_l2_mbps": [v + 0.5 for v in (4.0, 3.5, 3.0, 2.5, 2.0)],      # 0.5 / 2.0
        "urllc_rtt_p50_ms": [v + 2.0 for v in (0.1, 2.0, 4.0, 5.0, 8.1)],        # 2.0 / 8.0
    })
    e = errors_for_mode(exact_quarter, TB)
    assert all(e[m]["tracks"] for m in RULE_METRICS)


def test_misaligned_levels_are_refused():
    sim = [dict(r, level_index=r["level_index"] + 1) for r in TB]
    with pytest.raises(ValueError):
        errors_for_mode(sim, TB)


# --------------------------------------------------------------------------- the rule


def _err(nmae_by_metric):
    return {m: {"nmae": v, "tracks": v <= TRACK_THRESHOLD} for m, v in nmae_by_metric.items()}


def test_tracks_when_the_best_mode_is_within_threshold_on_every_metric():
    errors = {
        "demand_proportional": _err({m: 0.10 for m in RULE_METRICS}),
        "equal": _err({m: 0.60 for m in RULE_METRICS}),
        "min_rate_proportional": _err({m: 0.40 for m in RULE_METRICS}),
    }
    label, best, detail = classify_match(errors)
    assert label == "TRACKS_demand_proportional"
    assert best == "demand_proportional"


def test_tracks_can_name_a_mode_other_than_the_simulators_default():
    errors = {
        "demand_proportional": _err({m: 0.40 for m in RULE_METRICS}),
        "equal": _err({m: 0.60 for m in RULE_METRICS}),
        "min_rate_proportional": _err({m: 0.05 for m in RULE_METRICS}),
    }
    assert classify_match(errors)[0] == "TRACKS_min_rate_proportional"


def test_partial_when_the_best_mode_misses_a_metric():
    errors = {
        "demand_proportional": _err({"embb_goodput_l2_mbps": 0.05, "be_goodput_l2_mbps": 0.05,
                                     "urllc_rtt_p50_ms": 0.40}),
        "equal": _err({m: 0.70 for m in RULE_METRICS}),
        "min_rate_proportional": _err({m: 0.50 for m in RULE_METRICS}),
    }
    label, _, detail = classify_match(errors)
    assert label == "PARTIAL_demand_proportional"
    assert "urllc_rtt_p50_ms" not in detail["metrics_tracked_by_best"]


def test_neither_when_even_the_best_mode_tracks_nothing():
    errors = {k: _err({m: v for m in RULE_METRICS})
              for k, v in (("demand_proportional", 0.5), ("equal", 0.9), ("min_rate_proportional", 0.4))}
    assert classify_match(errors)[0] == "NEITHER"


def test_per_metric_winners_are_reported_even_when_they_differ_from_the_best_mode():
    errors = {
        "demand_proportional": _err({"embb_goodput_l2_mbps": 0.02, "be_goodput_l2_mbps": 0.30,
                                     "urllc_rtt_p50_ms": 0.10}),
        "equal": _err({"embb_goodput_l2_mbps": 0.20, "be_goodput_l2_mbps": 0.01,
                       "urllc_rtt_p50_ms": 0.90}),
        "min_rate_proportional": _err({m: 0.5 for m in RULE_METRICS}),
    }
    _, _, detail = classify_match(errors)
    assert detail["per_metric_winner"]["embb_goodput_l2_mbps"] == "demand_proportional"
    assert detail["per_metric_winner"]["be_goodput_l2_mbps"] == "equal"


# --------------------------------------------------------------------------- reading the sweep


def test_testbed_reader_takes_means_and_half_widths_from_the_aggregate():
    sweep = {"aggregate": [
        {"level_index": 1, "embb_share": 0.35,
         **{m: {"mean": 1.0 + i, "ci95_half": 0.1} for i, m in enumerate(
             RULE_METRICS + ("urllc_rtt_p95_ms", "urllc_goodput_l2_mbps"))}},
        {"level_index": 0, "embb_share": 0.20,
         **{m: {"mean": 0.5, "ci95_half": 0.2} for m in
            RULE_METRICS + ("urllc_rtt_p95_ms", "urllc_goodput_l2_mbps")}},
    ]}
    rows = load_testbed_rows(sweep)
    assert [r["level_index"] for r in rows] == [0, 1], "sorted by level"
    assert rows[1]["embb_goodput_l2_mbps"] == 1.0
    assert rows[0]["urllc_rtt_p50_ms_ci95_half"] == 0.2
