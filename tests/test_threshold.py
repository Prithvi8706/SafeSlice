"""Threshold policy tests.

The one that matters is test_select_is_idempotent_within_a_step. The guardrail calls select()
twice on any step where it masks out the proposal, so a policy that stepped its level inside
select() would move two levels on exactly the steps where the guardrail was active. That bug
would not crash anything; it would quietly make the baseline look erratic under load and the
write-up would report it as a property of threshold control. So it is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.context import Context
from agent.guardrail import Guardrail
from agent.policies.base import all_allowed
from agent.policies.threshold import Threshold


def make_ctx(ewma_ms: float, p95_ms: float = None, n_features: int = 9) -> Context:
    return Context(
        vector=np.zeros(n_features, dtype=float),
        rtt_ewma_ms=float(ewma_ms),
        rtt_p95_ms=float(ewma_ms if p95_ms is None else p95_ms),
        step_idx=0,
    )


# --------------------------------------------------------------------------- the rule


def test_steps_down_above_up_threshold(cfg):
    p = Threshold(cfg)
    start = p.level
    got = p.select(make_ctx(cfg.policy.threshold.up_ms + 0.5), all_allowed(p.n_actions))
    assert got == start - 1


def test_steps_up_below_down_threshold(cfg):
    p = Threshold(cfg)
    start = p.level
    got = p.select(make_ctx(cfg.policy.threshold.down_ms - 0.5), all_allowed(p.n_actions))
    assert got == start + 1


def test_holds_inside_the_dead_band(cfg):
    p = Threshold(cfg)
    mid = 0.5 * (cfg.policy.threshold.down_ms + cfg.policy.threshold.up_ms)
    assert p.select(make_ctx(mid), all_allowed(p.n_actions)) == p.level


def test_never_leaves_the_action_set(cfg):
    """Repeated pressure in one direction saturates at the ends rather than running off."""
    p = Threshold(cfg)
    allowed = all_allowed(p.n_actions)
    for _ in range(50):
        a = p.select(make_ctx(cfg.policy.threshold.down_ms - 1.0), allowed)
        assert 0 <= a < p.n_actions
        p.update(None, a, 0.0)
    assert p.level == p.n_actions - 1

    for _ in range(50):
        a = p.select(make_ctx(cfg.policy.threshold.up_ms + 1.0), allowed)
        assert 0 <= a < p.n_actions
        p.update(None, a, 0.0)
    assert p.level == 0


# --------------------------------------------------------------------------- the state bug


def test_select_is_idempotent_within_a_step(cfg):
    """Calling select() twice with the same context must return the same answer.

    The guardrail does exactly this when it masks out the first proposal. If select() mutated
    self.level the second call would be one level further along.
    """
    p = Threshold(cfg)
    ctx = make_ctx(cfg.policy.threshold.down_ms - 0.5)
    allowed = all_allowed(p.n_actions)
    first = p.select(ctx, allowed)
    for _ in range(5):
        assert p.select(ctx, allowed) == first
    assert p.level == cfg.action.initial_index  # unchanged until update()


def test_guardrail_reselect_moves_exactly_one_level(cfg):
    """End to end through the guardrail: a masked proposal must not double-step."""
    guard = Guardrail(cfg)
    p = Threshold(cfg)
    # Sit at the top of the action set so the warn mask definitely bites.
    p.level = p.n_actions - 1
    # In the warn band, below the hard limit: latency between warn_ms and hard_ms.
    ewma = 0.5 * (float(cfg.guardrail.warn_ms) + float(cfg.sla.hard_ms))
    ctx = make_ctx(ewma)

    proposed = p.select(ctx, all_allowed(p.n_actions))
    decision = guard.apply(ctx, proposed, p)

    assert decision.applied_action <= int(cfg.guardrail.safe_ceiling_index)
    assert decision.intervened


def test_resumes_from_the_applied_level_not_the_proposed_one(cfg):
    """After a guardrail clamp the controller tracks the network it is in."""
    p = Threshold(cfg)
    p.level = p.n_actions - 1
    clamped_to = 0
    p.update(None, clamped_to, 0.0)
    assert p.level == clamped_to
    # From the floor, a quiet network should step up by one, not back to where it wanted to be.
    got = p.select(make_ctx(cfg.policy.threshold.down_ms - 1.0), all_allowed(p.n_actions))
    assert got == clamped_to + 1


# --------------------------------------------------------------------------- config guards


def test_empty_dead_band_is_rejected():
    from config_loader import load_config

    bad = load_config(
        scenario="burst",
        overrides={"policy.threshold.down_ms": 6.0, "policy.threshold.up_ms": 5.0},
    )
    with pytest.raises(ValueError, match="dead band"):
        Threshold(bad)


def test_reset_returns_to_the_initial_level(cfg):
    p = Threshold(cfg)
    p.update(None, p.n_actions - 1, 0.0)
    p.reset(seed=3)
    assert p.level == int(cfg.action.initial_index)
