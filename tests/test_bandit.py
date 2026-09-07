"""Bandit algorithm tests.

These test the algorithms against problems whose answers are known independently of this
project. That is the point: if LinUCB were only ever tested against the simulator, a wrong
implementation and a wrong simulator could agree with each other and the suite would be green
while the headline result was meaningless.

  * test_linucb_learns_a_known_linear_problem  - a synthetic bandit where the best arm per
    context is known in closed form. LinUCB must find it.
  * test_sherman_morrison_matches_an_explicit_inverse - the incremental inverse must not drift.
    Drift here does not crash; it degrades the model slowly and reads as "the bandit did not
    learn", which is a conclusion this project would otherwise report as a finding.
  * test_select_is_idempotent_within_a_step - the guardrail calls select() twice on any step
    where it masks the proposal. See agent/bandit.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.bandit import EpsilonGreedy, LinUCB


# --------------------------------------------------------------------------- LinUCB: learning


def test_linucb_learns_a_known_linear_problem():
    """Two arms, one linear reward each, the better arm depends on the context.

    reward(arm 0) = 1.0 * x1,  reward(arm 1) = 1.0 * x2, both with a bias of 0.
    So arm 0 is optimal when x1 > x2 and arm 1 when x2 > x1. After enough rounds LinUCB must
    pick the right arm on held-out contexts far more often than chance.
    """
    rng = np.random.default_rng(0)
    model = LinUCB(n_actions=2, n_features=3, alpha=0.5)   # [bias, x1, x2]
    true = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    for _ in range(600):
        x = np.array([1.0, rng.random(), rng.random()])
        a = model.select(x)
        r = float(true[a] @ x) + 0.01 * rng.standard_normal()
        model.update(x, a, r)

    correct = 0
    for _ in range(400):
        x = np.array([1.0, rng.random(), rng.random()])
        want = 0 if x[1] > x[2] else 1
        correct += int(model.select(x) == want)
    assert correct / 400 > 0.9

    # And the recovered coefficients should resemble the truth it was never told.
    for a in (0, 1):
        assert np.allclose(model.theta(a), true[a], atol=0.08)


def test_linucb_with_zero_alpha_is_pure_greedy():
    """Freezing must remove exploration, not merely reduce it."""
    model = LinUCB(n_actions=3, n_features=2, alpha=0.0)
    x = np.array([1.0, 0.5])
    model.update(x, 1, 5.0)          # arm 1 is now clearly the best estimate
    assert model.select(x) == 1
    for _ in range(20):
        assert model.select(x) == 1  # never wanders off to an unexplored arm


def test_untrained_linucb_starts_at_the_safest_arm():
    """With no data every arm scores identically, and the tie must break downward."""
    model = LinUCB(n_actions=5, n_features=4, alpha=1.0)
    assert model.select(np.array([1.0, 0.3, 0.2, 0.1])) == 0


# --------------------------------------------------------------------------- LinUCB: numerics


def test_sherman_morrison_matches_an_explicit_inverse():
    rng = np.random.default_rng(7)
    d, k = 9, 5
    model = LinUCB(n_actions=k, n_features=d, alpha=0.5, ridge=1.0)
    explicit = [np.eye(d) for _ in range(k)]

    for _ in range(2000):
        x = rng.random(d)
        a = int(rng.integers(0, k))
        model.update(x, a, float(rng.standard_normal()))
        explicit[a] = explicit[a] + np.outer(x, x)

    for a in range(k):
        assert np.allclose(model.A_inv[a], np.linalg.inv(explicit[a]), atol=1e-8), a


def test_confidence_width_shrinks_with_data():
    model = LinUCB(n_actions=2, n_features=3, alpha=1.0)
    x = np.array([1.0, 0.5, 0.25])
    before = model.scores(x)[0]
    for _ in range(50):
        model.update(x, 0, 0.0)      # zero reward, so only the width can change
    after = model.scores(x)[0]
    assert after < before
    assert after >= 0.0


def test_linucb_respects_the_mask():
    model = LinUCB(n_actions=4, n_features=2, alpha=0.1)
    x = np.array([1.0, 1.0])
    model.update(x, 3, 10.0)         # make the masked arm the obvious favourite
    allowed = np.array([True, True, False, False])
    for _ in range(10):
        assert model.select(x, allowed) in (0, 1)


def test_linucb_rejects_a_wrong_sized_context():
    model = LinUCB(n_actions=2, n_features=3, alpha=0.1)
    with pytest.raises(ValueError, match="features"):
        model.select(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="features"):
        model.update(np.array([1.0, 2.0]), 0, 1.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_actions=0, n_features=3, alpha=0.1), "n_actions"),
        (dict(n_actions=2, n_features=0, alpha=0.1), "n_features"),
        (dict(n_actions=2, n_features=3, alpha=-1.0), "alpha"),
        (dict(n_actions=2, n_features=3, alpha=0.1, ridge=0.0), "ridge"),
    ],
)
def test_linucb_rejects_impossible_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LinUCB(**kwargs)


# --------------------------------------------------------------------------- epsilon-greedy


def test_epsilon_greedy_tries_every_arm_before_exploiting():
    model = EpsilonGreedy(n_actions=4, epsilon=0.0, seed=0)
    seen = []
    for step in range(4):
        a = model.select(step_idx=step)
        seen.append(a)
        model.update(None, a, 0.0)
    assert sorted(seen) == [0, 1, 2, 3]


def test_epsilon_greedy_converges_on_the_best_arm():
    """Stationary three-armed problem; arm 2 is best. Greedy share must be high."""
    rng = np.random.default_rng(3)
    means = [0.1, 0.3, 0.8]
    model = EpsilonGreedy(n_actions=3, epsilon=0.05, seed=11)
    picks = []
    for step in range(3000):
        a = model.select(step_idx=step)
        model.update(None, a, means[a] + 0.05 * rng.standard_normal())
        picks.append(a)
    assert np.mean(np.asarray(picks[-1000:]) == 2) > 0.9
    assert int(np.argmax(model.values)) == 2


def test_epsilon_greedy_estimates_are_running_means():
    model = EpsilonGreedy(n_actions=2, epsilon=0.0, seed=0)
    for r in (1.0, 2.0, 3.0, 6.0):
        model.update(None, 0, r)
    assert model.values[0] == pytest.approx(3.0)
    assert model.counts[0] == 4


def test_select_is_idempotent_within_a_step():
    """The guardrail calls select() again with a tighter mask; the coin must not be re-flipped."""
    model = EpsilonGreedy(n_actions=4, epsilon=1.0, seed=5)  # always explores
    for a in range(4):
        model.update(None, a, 0.0)                           # get past the try-everything phase
    allowed = np.ones(4, dtype=bool)
    first = model.select(allowed=allowed, step_idx=42)
    for _ in range(10):
        assert model.select(allowed=allowed, step_idx=42) == first
    # A new step draws a new coin, so the sequence still advances.
    _ = model.select(allowed=allowed, step_idx=43)
    assert model._coin_step == 43


def test_epsilon_greedy_explores_only_inside_the_mask():
    model = EpsilonGreedy(n_actions=5, epsilon=1.0, seed=2)
    allowed = np.array([True, False, True, False, False])
    for step in range(200):
        a = model.select(allowed=allowed, step_idx=step)
        assert allowed[a]
        model.update(None, a, 0.0)


def test_epsilon_greedy_is_reproducible_from_its_seed():
    def run(seed):
        m = EpsilonGreedy(n_actions=4, epsilon=0.5, seed=seed)
        out = []
        for step in range(200):
            a = m.select(step_idx=step)
            m.update(None, a, float(a))
            out.append(a)
        return out

    assert run(9) == run(9)
    assert run(9) != run(10)


def test_epsilon_greedy_rejects_an_impossible_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedy(n_actions=3, epsilon=1.5)
