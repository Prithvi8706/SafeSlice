"""Policy wrappers around the bandit algorithms.

These are thin on purpose. All the learning lives in `agent/bandit.py`, which knows nothing
about networks; this file is only the adapter between the Policy interface and those algorithms.

TWO THINGS HERE ARE NOT BOILERPLATE.

**Seeding.** The exploration RNG is seeded from the run seed through a hash, not from the run
seed directly. The run seed also generates the traffic (`traffic/traces.py:trace_seed`), and
feeding the same integer to both means that on seed k the exploration sequence and the traffic
sequence are drawn from streams initialised identically. That is a correlation between the
policy's randomness and the environment's randomness, and while it would probably be harmless,
"probably harmless" is not a property you want underneath a headline result. Deriving a separate
stream costs one line.

**Freezing.** `docs/EXPERIMENTS.md` section 1 commits to reporting bandits twice: online, which
is the honest deployment number and includes the cost of exploring, and converged, which is the
model pre-trained on other seeds and then evaluated with exploration off. `freeze()` is what
makes the second one possible, and it is a real change of behaviour, not a flag on a report:
LinUCB's alpha goes to zero and epsilon-greedy's epsilon goes to zero, so the frozen policy is
purely greedy against what it learned. `update()` still runs while frozen unless the caller
stops calling it; `experiments/run_experiment.py` stops calling it, so a converged run really is
evaluated against a fixed model rather than one that keeps drifting during evaluation.
"""

from __future__ import annotations

import hashlib

import numpy as np

from agent.bandit import EpsilonGreedy, LinUCB
from agent.context import N_FEATURES
from agent.policies.base import Policy, nearest_allowed


def policy_seed(run_seed: int, salt: str) -> int:
    """A per-policy random stream derived from the run seed, independent of the traffic stream.

    blake2b rather than hash(): Python's hash() is salted per process, so a run would not be
    reproducible across invocations. traffic/traces.py made the same choice for the same reason
    and pins it with a test.
    """
    h = hashlib.blake2b(f"{salt}:{int(run_seed)}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**32)


class _BanditPolicy(Policy):
    """Shared plumbing: freezing, and the guardrail-safe fallback."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.frozen = False

    def freeze(self) -> None:
        raise NotImplementedError

    def _fallback(self, chosen: int, allowed: np.ndarray) -> int:
        """Belt and braces. The algorithms already respect the mask; this makes it structural."""
        if allowed[chosen]:
            return int(chosen)
        return nearest_allowed(chosen, allowed)


class LinUCBPolicy(_BanditPolicy):
    """Disjoint LinUCB over the context vector from agent/context.py."""

    name = "linucb"

    def __init__(self, cfg):
        super().__init__(cfg)
        lcfg = cfg.policy.linucb
        self.alpha = float(lcfg.alpha)
        self.ridge = float(lcfg.get("ridge", 1.0))
        self.model = LinUCB(
            n_actions=self.n_actions,
            n_features=N_FEATURES,
            alpha=self.alpha,
            ridge=self.ridge,
        )

    def reset(self, seed: int = 0) -> None:
        # LinUCB is deterministic given its data, so the seed does not enter the model. It is
        # accepted so every policy resets the same way.
        self.model.reset()
        self.model.alpha = self.alpha
        self.frozen = False

    def select(self, ctx, allowed: np.ndarray) -> int:
        return self._fallback(self.model.select(ctx.vector, allowed), allowed)

    def update(self, ctx, action: int, reward: float) -> None:
        self.model.update(ctx.vector, action, reward)

    def freeze(self) -> None:
        """Exploration off: pure greedy against the learned coefficients."""
        self.frozen = True
        self.model.alpha = 0.0


class EpsilonGreedyPolicy(_BanditPolicy):
    """Context-free epsilon-greedy. The ablation that tests whether context earns its keep."""

    name = "epsilon_greedy"

    def __init__(self, cfg):
        super().__init__(cfg)
        ecfg = cfg.policy.epsilon_greedy
        self.epsilon = float(ecfg.epsilon)
        self.model = EpsilonGreedy(n_actions=self.n_actions, epsilon=self.epsilon, seed=0)

    def reset(self, seed: int = 0) -> None:
        self.model.reset(seed=policy_seed(seed, "epsilon_greedy"))
        self.model.epsilon = self.epsilon
        self.frozen = False

    def select(self, ctx, allowed: np.ndarray) -> int:
        chosen = self.model.select(ctx.vector, allowed, step_idx=int(ctx.step_idx))
        return self._fallback(chosen, allowed)

    def update(self, ctx, action: int, reward: float) -> None:
        self.model.update(ctx.vector, action, reward)

    def freeze(self) -> None:
        self.frozen = True
        self.model.epsilon = 0.0
