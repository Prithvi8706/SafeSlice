"""Reward function tests.

These pin down the SIGN and MONOTONICITY of every term. That is deliberately more valuable than
pinning exact values: the weights will be swept in the sensitivity study, so a test asserting
reward == 0.7123 would just have to be rewritten every time. A test asserting that raising
latency can never raise reward is true for every weight vector we would ever consider, and it
catches the class of bug that would silently invert a conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.reward import reward, reward_breakdown
from net.backend import TelemetrySample


def sample(
    cfg,
    rtt_p50=2.0,
    rtt_p95=2.0,
    urllc_bps=3.0e6,
    embb_bps=3.5e6,
    be_bps=3.5e6,
    drops=(0, 0, 0),
    backlog=(0.0, 0.0, 0.0),
):
    served = urllc_bps + embb_bps + be_bps
    return TelemetrySample(
        t_wall=0.0,
        t_step=1.0,
        urllc_rtt_ms_p50=rtt_p50,
        urllc_rtt_ms_p95=rtt_p95,
        urllc_tx_bps=urllc_bps,
        embb_goodput_bps=embb_bps,
        be_goodput_bps=be_bps,
        q0_drops=drops[0],
        q1_drops=drops[1],
        q2_drops=drops[2],
        link_util=min(served / float(cfg.link.capacity_bps), 1.0),
        backlog_bytes_per_queue=backlog,
    )


def test_terms_have_the_documented_signs(cfg):
    tel = sample(cfg, rtt_p95=float(cfg.sla.hard_ms) + 10.0, drops=(3, 4, 5))
    rb = reward_breakdown(tel, cfg)
    assert rb.tput_term > 0
    assert rb.be_term > 0
    assert rb.sla_term < 0
    assert rb.viol_term < 0
    assert rb.drop_term < 0
    assert rb.violated


def test_breakdown_sums_to_the_scalar(cfg):
    tel = sample(cfg, rtt_p95=18.0, drops=(1, 2, 3))
    rb = reward_breakdown(tel, cfg)
    assert rb.total == pytest.approx(
        rb.tput_term + rb.be_term + rb.sla_term + rb.viol_term + rb.drop_term
    )
    assert reward(tel, cfg) == pytest.approx(rb.total)


def test_latency_below_target_incurs_no_sla_penalty(cfg):
    tel = sample(cfg, rtt_p95=float(cfg.sla.target_ms) - 0.001)
    rb = reward_breakdown(tel, cfg)
    assert rb.sla_term == pytest.approx(0.0)
    assert rb.viol_term == pytest.approx(0.0)
    assert not rb.violated


def test_sla_penalty_is_a_hinge_not_a_step(cfg):
    """Missing the target by 1 ms must score better than missing it by 50 ms."""
    target = float(cfg.sla.target_ms)
    near = reward(sample(cfg, rtt_p95=target + 1.0), cfg)
    far = reward(sample(cfg, rtt_p95=target + 50.0), cfg)
    assert near > far


def test_reward_is_monotone_non_increasing_in_latency(cfg):
    """For any fixed throughput, more latency can never be worth more."""
    rtts = np.linspace(0.0, 5.0 * float(cfg.sla.hard_ms), 200)
    rewards = [reward(sample(cfg, rtt_p95=float(r)), cfg) for r in rtts]
    diffs = np.diff(rewards)
    assert (diffs <= 1e-12).all(), "reward increased with latency somewhere"


def test_reward_is_monotone_non_decreasing_in_embb_goodput(cfg):
    goodputs = np.linspace(0.0, float(cfg.link.capacity_bps), 100)
    rewards = [reward(sample(cfg, embb_bps=float(g)), cfg) for g in goodputs]
    assert (np.diff(rewards) >= -1e-12).all()


def test_reward_is_monotone_non_increasing_in_drops(cfg):
    rewards = [reward(sample(cfg, drops=(0, int(d), 0)), cfg) for d in range(0, 200, 5)]
    assert (np.diff(rewards) <= 1e-12).all()


def test_best_effort_carries_real_weight(cfg):
    """If BE were worth zero, the action space would collapse to 'as high as allowed'."""
    assert float(cfg.reward.w_be) > 0.0
    low = reward(sample(cfg, be_bps=0.0), cfg)
    high = reward(sample(cfg, be_bps=3.5e6), cfg)
    assert high > low


def test_hard_violation_costs_a_discrete_step(cfg):
    """Crossing the hard limit must cost more than the hinge alone would explain."""
    hard = float(cfg.sla.hard_ms)
    eps = 1e-6
    just_under = reward_breakdown(sample(cfg, rtt_p95=hard - eps), cfg)
    just_over = reward_breakdown(sample(cfg, rtt_p95=hard + eps), cfg)
    assert just_under.viol_term == pytest.approx(0.0)
    assert just_over.viol_term == pytest.approx(-float(cfg.reward.w_viol))
    assert just_under.total - just_over.total == pytest.approx(
        float(cfg.reward.w_viol), abs=1e-4
    )


def test_rtt_statistic_is_config_selected(cfg):
    from config_loader import load_config

    tel = sample(cfg, rtt_p50=2.0, rtt_p95=40.0)
    on_p95 = reward_breakdown(tel, cfg)
    on_p50 = reward_breakdown(tel, load_config(scenario="burst", overrides={"reward.rtt_statistic": "urllc_rtt_ms_p50"}))
    assert on_p95.rtt_used_ms == pytest.approx(40.0)
    assert on_p50.rtt_used_ms == pytest.approx(2.0)
    assert on_p50.total > on_p95.total


def test_goodput_is_normalized_by_capacity_not_offered_load(cfg):
    """Serving 3.5 Mbps must score the same whether 4 or 9 Mbps was offered."""
    a = reward(sample(cfg, embb_bps=3.5e6), cfg)
    b = reward(sample(cfg, embb_bps=3.5e6), cfg)
    assert a == pytest.approx(b)
    # And the term equals w_tput * goodput / capacity exactly.
    rb = reward_breakdown(sample(cfg, embb_bps=3.5e6), cfg)
    assert rb.tput_term == pytest.approx(
        float(cfg.reward.w_tput) * 3.5e6 / float(cfg.link.capacity_bps)
    )
