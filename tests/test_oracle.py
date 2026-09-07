"""Oracle tests.

The Oracle is a reference line, so the thing to test is that it is the reference line it claims
to be, and that it is not quietly a better or worse one.

  * test_schedule_is_per_phase_not_one_level_everywhere - if the schedule collapsed to a single
    level for every phase, the "best phase-static" bound would be the best static policy and the
    Oracle column would be redundant without anyone noticing.
  * test_oracle_beats_every_static_level_on_average - a per-phase schedule ought to be at least
    as good as the best single level, since holding one level is a special case of a schedule.
    It is asserted across seeds rather than per seed, because backlog crossing phase boundaries
    means it is not guaranteed on any individual run. That failure mode is exactly why
    compute_oracle_schedule executes the assembled schedule instead of summing the probes.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.policies.base import all_allowed
from agent.policies.oracle import Oracle
from agent.policies.static import _FixedLevel
from experiments.run_experiment import ORACLE_NAME, compute_oracle_schedule, run_once


def make_ctx(step_idx):
    from agent.context import Context

    return Context(vector=np.zeros(9), rtt_ewma_ms=2.0, rtt_p95_ms=2.0, step_idx=int(step_idx))


# --------------------------------------------------------------------------- the schedule


def test_schedule_covers_every_phase_in_the_trace(cfg):
    schedule, step_phase = compute_oracle_schedule(cfg, "burst", 0)
    assert set(schedule) == set(int(p) for p in np.unique(step_phase))
    assert all(0 <= v < len(cfg.action.embb_levels) for v in schedule.values())


def test_schedule_is_per_phase_not_one_level_everywhere():
    """At full run length the phase structure must actually change the chosen level."""
    from config_loader import load_config

    cfg = load_config(scenario="sawtooth")
    schedule, _ = compute_oracle_schedule(cfg, "sawtooth", 0)
    assert len(set(schedule.values())) > 1, f"schedule collapsed to one level: {schedule}"


# --------------------------------------------------------------------------- behaviour


def test_plays_the_scheduled_level_when_unmasked(cfg):
    step_phase = np.array([0, 0, 1, 1, 2])
    oracle = Oracle(cfg, schedule={0: 3, 1: 0, 2: 4}, step_phase=step_phase)
    allowed = all_allowed(oracle.n_actions)
    assert [oracle.select(make_ctx(i), allowed) for i in range(5)] == [3, 3, 0, 0, 4]


def test_falls_back_inside_the_mask(cfg):
    oracle = Oracle(cfg, schedule={0: 4}, step_phase=np.array([0, 0]))
    allowed = np.array([True, True, False, False, False])
    assert oracle.select(make_ctx(0), allowed) == 1     # nearest allowed, downward


def test_steps_past_the_trace_fall_back_to_the_floor(cfg):
    oracle = Oracle(cfg, schedule={0: 4}, step_phase=np.array([0]))
    assert oracle.select(make_ctx(99), all_allowed(oracle.n_actions)) == int(cfg.action.floor_index)


def test_rejects_a_schedule_outside_the_action_set(cfg):
    with pytest.raises(ValueError, match="outside the action set"):
        Oracle(cfg, schedule={0: 99}, step_phase=np.array([0]))


def test_is_flagged_as_an_oracle(cfg):
    """is_oracle keeps it out of tables of deployable methods."""
    oracle = Oracle(cfg, schedule={0: 0}, step_phase=np.array([0]))
    assert oracle.is_oracle
    assert oracle.name == ORACLE_NAME


# --------------------------------------------------------------------------- the bound


def test_oracle_beats_every_static_level_on_average(cfg):
    """Averaged over seeds, the phase schedule must be at least as good as any fixed level."""
    seeds = [0, 1, 2]
    oracle_rewards = []
    for seed in seeds:
        _, m, _ = run_once(ORACLE_NAME, "burst", seed, cfg=cfg, write=False)
        oracle_rewards.append(m.mean_reward)

    for level in range(len(cfg.action.embb_levels)):
        level_rewards = []
        for seed in seeds:
            probe = _FixedLevel(cfg, level, f"probe_{level}")
            _, m, _ = run_once(
                f"probe_{level}", "burst", seed, cfg=cfg, write=False, policy=probe, learn=False
            )
            level_rewards.append(m.mean_reward)
        assert np.mean(oracle_rewards) >= np.mean(level_rewards) - 1e-9, (
            f"level {level} beat the oracle on average"
        )
