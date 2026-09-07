"""Simulator tests.

Two groups. The first checks conservation laws that must hold whatever the model does: bytes are
not created, the link is never oversubscribed, backlog stays within the queue limit. The second
checks that the ACTION HAS AN EFFECT, which is the property the whole project depends on. If
raising the eMBB cap does not measurably hurt URLLC in this simulator, there is no control
problem and every downstream result is decoration.

test_excess_sharing_mode_decides_whether_the_problem_exists is the uncomfortable one. It asserts
that under the `equal` sharing mode URLLC is materially better protected than under
`demand_proportional` at the same aggressive action. That is not a bug being pinned, it is the
load-bearing modelling assumption being made visible and testable so Week 1b can check it
against real OVS instead of us quietly relying on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from config_loader import load_config
from net.backend import allocation_from_level
from net.sim_backend import SimBackend
from traffic.traces import build_trace, trace_seed


def run_fixed_level(cfg, seed: int, level_index: int, n_steps=None):
    """Drive the sim at one fixed action for the whole trace. Returns the telemetry list."""
    trace = build_trace(cfg, seed)
    backend = SimBackend(cfg, trace)
    alloc = allocation_from_level(cfg, level_index)
    limit = n_steps if n_steps is not None else trace.n_steps
    out = []
    for _ in range(limit):
        if backend.done:
            break
        backend.apply_allocation(alloc)
        out.append(backend.read_telemetry())
    return out


# --------------------------------------------------------------------------- determinism


def test_same_seed_gives_a_bit_identical_trace(cfg):
    a = build_trace(cfg, 3)
    b = build_trace(cfg, 3)
    assert np.array_equal(a.offered_bps, b.offered_bps)
    assert np.array_equal(a.phase_id, b.phase_id)


def test_different_seeds_give_different_traces(cfg):
    a = build_trace(cfg, 3)
    b = build_trace(cfg, 4)
    assert not np.array_equal(a.offered_bps, b.offered_bps)


def test_trace_seed_is_process_independent():
    """Guards against a regression to hash(), which is salted per process."""
    assert trace_seed("burst", 0) == trace_seed("burst", 0)
    assert trace_seed("burst", 0) != trace_seed("burst", 1)
    assert trace_seed("burst", 0) != trace_seed("ramp", 0)
    # A literal, so a change to the derivation shows up here rather than as silently
    # incomparable results across a re-run of the suite.
    assert trace_seed("burst", 0) == 3460656701478529369


def test_jitter_preserves_the_configured_phase_mean(cfg):
    """Lognormal jitter must not shift the mean, or the YAML no longer means what it says."""
    long_cfg = load_config(
        scenario="burst", overrides={"run.duration_s": 3000, "run.warmup_s": 0}
    )
    trace = build_trace(long_cfg, 7)
    steps = trace.step_offered_bps
    phases = trace.step_phase_id()
    # Phase 0 of burst.yaml holds URLLC at 3.0 Mbps.
    urllc_phase0 = steps[0, phases == 0]
    assert urllc_phase0.mean() == pytest.approx(3.0e6, rel=0.02)


# --------------------------------------------------------------------------- conservation


def test_link_is_never_oversubscribed(cfg):
    for tel in run_fixed_level(cfg, 0, 4):
        assert tel.link_util <= 1.0 + 1e-9


def test_backlog_never_exceeds_the_queue_limit(cfg):
    limit = float(cfg.sim.queue_limit_bytes)
    for tel in run_fixed_level(cfg, 0, 4):
        for b in tel.backlog_bytes_per_queue:
            assert -1e-9 <= b <= limit + 1e-6


def test_served_never_exceeds_offered_over_the_run(cfg):
    """Bytes out cannot exceed bytes in. Catches double-counting in the scheduler."""
    tels = run_fixed_level(cfg, 0, 2)
    served = np.array(
        [[t.urllc_tx_bps, t.embb_goodput_bps, t.be_goodput_bps] for t in tels]
    ).sum(axis=0)
    offered = np.array([list(t.offered_bps_per_queue) for t in tels]).sum(axis=0)
    assert (served <= offered + 1e-3).all()


def test_no_nan_or_negative_telemetry(cfg):
    for tel in run_fixed_level(cfg, 1, 3):
        for v in (
            tel.urllc_rtt_ms_p50,
            tel.urllc_rtt_ms_p95,
            tel.urllc_tx_bps,
            tel.embb_goodput_bps,
            tel.be_goodput_bps,
            tel.link_util,
        ):
            assert np.isfinite(v)
            assert v >= 0.0
        assert tel.urllc_rtt_ms_p95 >= tel.urllc_rtt_ms_p50 - 1e-9
        assert min(tel.q0_drops, tel.q1_drops, tel.q2_drops) >= 0


def test_latency_never_falls_below_the_base_rtt(cfg):
    base = float(cfg.link.base_rtt_ms)
    for tel in run_fixed_level(cfg, 0, 0):
        assert tel.urllc_rtt_ms_p50 >= base - 1e-9


def test_reset_restores_the_initial_state(cfg):
    trace = build_trace(cfg, 0)
    backend = SimBackend(cfg, trace)
    backend.apply_allocation(allocation_from_level(cfg, 4))
    first = [backend.read_telemetry() for _ in range(10)]
    backend.reset()
    backend.apply_allocation(allocation_from_level(cfg, 4))
    second = [backend.read_telemetry() for _ in range(10)]
    assert [t.urllc_rtt_ms_p95 for t in first] == [t.urllc_rtt_ms_p95 for t in second]


def test_trace_exhaustion_raises_rather_than_returning_junk(cfg):
    trace = build_trace(cfg, 0)
    backend = SimBackend(cfg, trace)
    while not backend.done:
        backend.read_telemetry()
    with pytest.raises(RuntimeError):
        backend.read_telemetry()


# --------------------------------------------------------------------------- the action matters


def test_raising_the_embb_cap_raises_embb_goodput(cfg):
    means = []
    for level in range(len(cfg.action.embb_levels)):
        tels = run_fixed_level(cfg, 0, level)
        means.append(np.mean([t.embb_goodput_bps for t in tels]))
    assert (np.diff(means) > 0).all(), f"eMBB goodput not increasing in the level: {means}"


def test_raising_the_embb_cap_hurts_urllc_latency(cfg):
    """The core coupling. Without it there is no control problem and no project."""
    p95s = []
    for level in range(len(cfg.action.embb_levels)):
        tels = run_fixed_level(cfg, 0, level)
        p95s.append(float(np.percentile([t.urllc_rtt_ms_p95 for t in tels], 95)))
    assert p95s[-1] > p95s[0] + 1.0, f"aggressive eMBB did not hurt URLLC at all: {p95s}"
    assert (np.diff(p95s) >= -1e-6).all(), f"URLLC latency not monotone in the level: {p95s}"


def test_the_safe_level_keeps_urllc_inside_the_sla_and_the_aggressive_one_does_not(cfg):
    """The scenario has to actually contain the tradeoff it claims to."""
    hard = float(cfg.sla.hard_ms)
    safe = run_fixed_level(cfg, 0, int(cfg.action.floor_index))
    aggressive = run_fixed_level(cfg, 0, len(cfg.action.embb_levels) - 1)
    safe_viol = np.mean([t.urllc_rtt_ms_p95 > hard for t in safe])
    aggr_viol = np.mean([t.urllc_rtt_ms_p95 > hard for t in aggressive])
    assert safe_viol == 0.0, f"the floor level already violates the SLA ({safe_viol:.3f})"
    assert aggr_viol > 0.0, "the most aggressive level never violates; the scenario is too easy"


def test_excess_sharing_mode_decides_whether_the_problem_exists(cfg):
    """Makes the load-bearing modelling assumption visible instead of implicit.

    Under `equal` (idealised DRR) URLLC's residual demand is largely covered for free, so an
    aggressive eMBB cap hurts it far less. If the real OVS testbed turns out to behave this way,
    the honest Week 1b finding is that this control problem does not exist at this operating
    point. See docs/DESIGN.md.
    """
    aggressive = len(cfg.action.embb_levels) - 1

    def p95_under(mode):
        c = load_config(
            scenario="burst",
            overrides={
                # 90 s, so the run actually reaches the eMBB burst phase. See the note on
                # the cfg fixture in tests/conftest.py.
                "run.duration_s": 90,
                "run.warmup_s": 10,
                "sim.excess_sharing": mode,
            },
        )
        tels = run_fixed_level(c, 0, aggressive)
        return float(np.percentile([t.urllc_rtt_ms_p95 for t in tels], 95))

    pessimistic = p95_under("demand_proportional")
    fair = p95_under("equal")
    assert pessimistic > fair, (
        "demand_proportional is supposed to be the pessimistic case for URLLC; "
        f"got demand={pessimistic:.2f} ms vs equal={fair:.2f} ms"
    )


def test_rejects_an_unknown_excess_sharing_mode():
    c = load_config(scenario="burst", overrides={"sim.excess_sharing": "nonsense"})
    with pytest.raises(ValueError):
        SimBackend(c, build_trace(c, 0))


def test_rejects_min_shares_that_leave_no_headroom():
    c = load_config(
        scenario="burst",
        overrides={
            "slices.urllc.min_share": 0.5,
            "slices.embb.min_share": 0.4,
            "slices.be.min_share": 0.2,
        },
    )
    with pytest.raises(ValueError):
        SimBackend(c, build_trace(c, 0))
