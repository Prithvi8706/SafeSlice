"""Guardrail tests.

The central one is test_applied_action_is_always_inside_the_mask. It is a property test over a
randomized sweep, driven by an ADVERSARIAL policy that always tries to return the most
aggressive arm, including when asked to re-choose from a restricted set. If the invariant only
held because well-behaved policies happen to respect the mask, that test would fail. It is the
test that makes "safe" in the project title mean something checkable.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.context import Context
from agent.guardrail import (
    REASON_HARD_OVERRIDE,
    REASON_HOLDDOWN,
    REASON_OK,
    REASON_WARN_PASS,
    REASON_WARN_RESELECT,
    Guardrail,
)
from agent.policies.base import Policy, nearest_allowed


class AlwaysMaxPolicy(Policy):
    """Adversarial. Ignores the mask entirely and always demands the most aggressive level."""

    name = "always_max"

    def select(self, ctx, allowed):
        return self.n_actions - 1


class WellBehavedPolicy(Policy):
    """Respects the mask, preferring the highest allowed level."""

    name = "well_behaved"

    def select(self, ctx, allowed):
        return int(np.flatnonzero(allowed).max())


def make_ctx(ewma_ms: float, p95_ms: float, n_features: int = 9) -> Context:
    return Context(
        vector=np.zeros(n_features, dtype=float),
        rtt_ewma_ms=float(ewma_ms),
        rtt_p95_ms=float(p95_ms),
        step_idx=0,
    )


# --------------------------------------------------------------------------- the invariant


def test_applied_action_is_always_inside_the_mask(cfg):
    """The load-bearing safety property, checked against a policy that fights the mask."""
    guard = Guardrail(cfg)
    policy = AlwaysMaxPolicy(cfg)
    n_actions = len(cfg.action.embb_levels)
    rng = np.random.default_rng(1234)

    checked = 0
    for _ in range(5000):
        # Sweep well past the hard limit so every branch is exercised, including the ones the
        # simulator may rarely reach on its own.
        ewma = float(rng.uniform(0.0, 4.0 * cfg.sla.hard_ms))
        p95 = float(rng.uniform(0.0, 4.0 * cfg.sla.hard_ms))
        proposed = int(rng.integers(0, n_actions))
        ctx = make_ctx(ewma, p95)

        d = guard.apply(ctx, proposed, policy)

        assert 0 <= d.applied_action < n_actions
        assert d.allowed_mask[d.applied_action], (
            f"applied {d.applied_action} outside mask {d.allowed_mask} "
            f"(reason={d.reason}, ewma={ewma:.2f}, p95={p95:.2f})"
        )
        assert any(d.allowed_mask), "the mask must never be empty"
        assert d.intervened == (d.applied_action != d.proposed_action)
        checked += 1

    assert checked == 5000


def test_every_branch_is_reached_in_the_sweep(cfg):
    """A property test that never enters a branch proves nothing about that branch."""
    guard = Guardrail(cfg)
    policy = AlwaysMaxPolicy(cfg)
    n_actions = len(cfg.action.embb_levels)
    rng = np.random.default_rng(99)

    for _ in range(5000):
        ctx = make_ctx(rng.uniform(0.0, 4.0 * cfg.sla.hard_ms), rng.uniform(0.0, 4.0 * cfg.sla.hard_ms))
        guard.apply(ctx, int(rng.integers(0, n_actions)), policy)

    for reason in (REASON_OK, REASON_WARN_RESELECT, REASON_HARD_OVERRIDE, REASON_HOLDDOWN):
        assert guard.reason_counts[reason] > 0, f"branch {reason} never exercised"


# --------------------------------------------------------------------------- branch behaviour


def test_calm_latency_passes_the_proposal_through(cfg):
    guard = Guardrail(cfg)
    policy = WellBehavedPolicy(cfg)
    ctx = make_ctx(cfg.link.base_rtt_ms, cfg.link.base_rtt_ms)
    d = guard.apply(ctx, 4, policy)
    assert d.applied_action == 4
    assert d.reason == REASON_OK
    assert not d.intervened
    assert all(d.allowed_mask)


def test_warn_band_masks_aggressive_levels(cfg):
    guard = Guardrail(cfg)
    policy = WellBehavedPolicy(cfg)
    ctx = make_ctx(cfg.guardrail.warn_ms + 1.0, cfg.guardrail.warn_ms + 1.0)
    d = guard.apply(ctx, 4, policy)

    ceiling = int(cfg.guardrail.safe_ceiling_index)
    assert d.reason == REASON_WARN_RESELECT
    assert d.intervened
    assert d.applied_action <= ceiling
    assert not any(d.allowed_mask[ceiling + 1 :])


def test_warn_band_does_not_count_as_intervention_when_the_proposal_was_already_safe(cfg):
    """A guardrail that masked nothing away must not inflate the safety headline metric."""
    guard = Guardrail(cfg)
    policy = WellBehavedPolicy(cfg)
    ctx = make_ctx(cfg.guardrail.warn_ms + 1.0, cfg.guardrail.warn_ms + 1.0)
    d = guard.apply(ctx, 0, policy)
    assert d.reason == REASON_WARN_PASS
    assert not d.intervened
    assert guard.interventions == 0


def test_hard_breach_clamps_to_floor_and_arms_the_holddown(cfg):
    guard = Guardrail(cfg)
    policy = AlwaysMaxPolicy(cfg)
    floor = int(cfg.action.floor_index)
    hold = int(cfg.guardrail.holddown_steps)

    breach = make_ctx(cfg.sla.hard_ms + 5.0, cfg.sla.hard_ms + 5.0)
    d = guard.apply(breach, 4, policy)
    assert d.reason == REASON_HARD_OVERRIDE
    assert d.applied_action == floor
    assert d.intervened

    # Latency is healthy again immediately, but the hold-down must still bite.
    calm = make_ctx(cfg.link.base_rtt_ms, cfg.link.base_rtt_ms)
    for i in range(hold):
        d = guard.apply(calm, 4, policy)
        assert d.reason == REASON_HOLDDOWN, f"hold-down released early at step {i}"
        assert d.applied_action == floor

    d = guard.apply(calm, 4, policy)
    assert d.reason == REASON_OK
    assert d.applied_action == 4


def test_hard_override_fires_on_the_instantaneous_tail_even_when_the_ewma_is_calm(cfg):
    """The whole point of watching p95 as well as the EWMA is short sharp spikes."""
    guard = Guardrail(cfg)
    policy = AlwaysMaxPolicy(cfg)
    ctx = make_ctx(ewma_ms=cfg.link.base_rtt_ms, p95_ms=cfg.sla.hard_ms + 20.0)
    d = guard.apply(ctx, 4, policy)
    assert d.reason == REASON_HARD_OVERRIDE
    assert d.escalation_source == "p95"
    assert d.applied_action == int(cfg.action.floor_index)


def test_disabled_guardrail_is_a_pass_through(cfg):
    from config_loader import load_config

    off = load_config(scenario="burst", overrides={"guardrail.enabled": False})
    guard = Guardrail(off)
    policy = AlwaysMaxPolicy(off)
    ctx = make_ctx(1000.0, 1000.0)
    d = guard.apply(ctx, 4, policy)
    assert d.applied_action == 4
    assert not d.intervened
    assert guard.interventions == 0


# --------------------------------------------------------------------------- config validation


@pytest.mark.parametrize(
    "override",
    [
        {"guardrail.warn_ms": 99.0},           # warn above hard, so it can never warn
        {"guardrail.safe_ceiling_index": 99},  # outside the action set
        {"action.floor_index": -1},
        {"guardrail.safe_ceiling_index": 0, "action.floor_index": 2},  # ceiling below floor
    ],
)
def test_incoherent_guardrail_config_is_rejected_at_construction(override):
    from config_loader import load_config

    with pytest.raises(ValueError):
        Guardrail(load_config(scenario="burst", overrides=override))


def test_nearest_allowed_breaks_ties_toward_the_safer_level():
    allowed = np.array([True, False, False, False, True])
    # index 2 is equidistant from 0 and 4; the safe choice is 0
    assert nearest_allowed(2, allowed) == 0
