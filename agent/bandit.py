"""The two bandit algorithms, as algorithms.

Nothing in this file knows about network telemetry, the guardrail or the config. It takes a
feature vector and a reward and returns an arm. That separation is deliberate: it means these
can be tested against textbook problems with known answers, rather than only against the
simulator, where a subtly wrong implementation and a subtly wrong simulator can agree with each
other. `tests/test_bandit.py` runs LinUCB against a synthetic linear problem where the right
answer is known in closed form.

WHAT THE TWO OF THEM ARE FOR

`LinUCB` is the method the project proposes. `EpsilonGreedy` is deliberately CONTEXT-FREE: it
estimates one mean reward per arm and ignores the feature vector entirely. That is not a
weakened strawman, it is the ablation that isolates the word "contextual" in "contextual
bandit". If LinUCB does not beat a context-free bandit, then the context vector is not carrying
information and the honest conclusion is that the problem does not need a contextual method. The
project is set up so that result can be reported rather than avoided (docs/PLAN.md section 7).

REWARD SCALE, AND WHY alpha IS TUNED RATHER THAN DERIVED
The textbook LinUCB confidence width alpha = 1 + sqrt(ln(2/delta)/2) is derived for rewards in
[0, 1]. This project's reward (agent/reward.py) is not in [0, 1]: it is roughly bounded above by
0.94 but has no lower bound, because the SLA hinge grows without limit. Quoting the textbook
alpha would therefore be quoting a bound that does not hold here. We tune alpha empirically in
sim instead and say so. The alternative, rescaling the reward into [0, 1] using its observed
range, would require knowing that range before the run, which is lookahead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class LinUCB:
    """Disjoint LinUCB: one independent ridge regression per arm.

    Disjoint, not hybrid, because there is no shared feature structure to exploit here. Every
    arm is the same three queues at a different eMBB cap, and the effect of the context on the
    reward genuinely differs per arm - that is the entire hypothesis. A shared component would
    force those per-arm differences through one set of coefficients.

    Per arm a the model is r ≈ theta_a · x, with

        A_a = ridge * I + sum over rounds where a was played of x x^T
        b_a = sum over those rounds of r x
        theta_a = A_a^-1 b_a

    and the arm is chosen to maximise theta_a · x + alpha * sqrt(x^T A_a^-1 x): the estimate
    plus a confidence width that shrinks as that arm accumulates data in directions resembling
    the current context.

    A_a^-1 is maintained incrementally by the Sherman-Morrison identity rather than inverted
    each step. With d = 9 an explicit inverse would also be fast, so this is not primarily an
    optimisation: the decision latency of this policy is a headline metric of the project and
    is compared against an array lookup, so the implementation should be the one a deployment
    would use. `tests/test_bandit.py::test_sherman_morrison_matches_an_explicit_inverse` checks
    the incremental inverse against an explicit one after many updates, because that identity
    accumulates floating-point error and a slow drift would look like the model failing to
    learn.
    """

    def __init__(self, n_actions: int, n_features: int, alpha: float, ridge: float = 1.0):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative (0 disables exploration)")
        if ridge <= 0:
            raise ValueError("ridge must be positive or A is not invertible at the start")
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.reset()

    def reset(self) -> None:
        d, k = self.n_features, self.n_actions
        self.A_inv = np.stack([np.eye(d) / self.ridge for _ in range(k)])
        self.b = np.zeros((k, d), dtype=float)
        self.counts = np.zeros(k, dtype=int)

    # ------------------------------------------------------------------ inference

    def theta(self, action: int) -> np.ndarray:
        return self.A_inv[int(action)] @ self.b[int(action)]

    def scores(self, x: np.ndarray, allowed: Optional[np.ndarray] = None) -> np.ndarray:
        """Upper confidence score per arm. Disallowed arms score -inf."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_features:
            raise ValueError(f"context has {x.size} features, expected {self.n_features}")

        # Ax = A_inv @ x for every arm at once: (k, d, d) @ (d,) -> (k, d)
        Ax = self.A_inv @ x
        # theta_a · x  ==  (A_inv_a @ b_a) · x  ==  b_a · (A_inv_a @ x), because A_inv is
        # symmetric. The second form reuses Ax and avoids a second matrix product per arm.
        mean = np.einsum("kd,kd->k", self.b, Ax)
        var = np.einsum("d,kd->k", x, Ax)
        # x^T A_inv x is non-negative in exact arithmetic; clip guards the sqrt against a tiny
        # negative produced by accumulated Sherman-Morrison error.
        width = self.alpha * np.sqrt(np.maximum(var, 0.0))

        out = mean + width
        if allowed is not None:
            out = np.where(np.asarray(allowed, dtype=bool), out, -np.inf)
        return out

    def select(self, x: np.ndarray, allowed: Optional[np.ndarray] = None) -> int:
        """Argmax of the UCB scores. Ties break toward the LOWER index.

        np.argmax returns the first maximum, and low indices are low eMBB caps, so an untrained
        model - where every arm scores identically - starts at the most conservative arm and
        works upward as confidence widths shrink. For a system whose claim is bounded SLA risk,
        breaking ties toward the safe end is the right default, and it makes the first steps of
        a run deterministic rather than arbitrary.
        """
        return int(np.argmax(self.scores(x, allowed)))

    # ------------------------------------------------------------------ learning

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_features:
            raise ValueError(f"context has {x.size} features, expected {self.n_features}")
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise IndexError(f"action {a} outside 0..{self.n_actions - 1}")

        # Sherman-Morrison: (A + x x^T)^-1 = A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)
        Ainv = self.A_inv[a]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[a] = Ainv - np.outer(Ax, Ax) / denom
        self.b[a] += float(reward) * x
        self.counts[a] += 1

    # ------------------------------------------------------------------ introspection

    def a_matrix(self, action: int) -> np.ndarray:
        """Reconstruct A_a. Used by tests and by nothing in the hot loop."""
        return np.linalg.inv(self.A_inv[int(action)])


class EpsilonGreedy:
    """Context-free epsilon-greedy over arm means. The ablation, not a strawman.

    It ignores `x` entirely and tracks one running mean per arm. Comparing it against LinUCB on
    identical traffic answers exactly one question: is the context vector worth anything? If the
    two tie, the answer is no, and no amount of tuning alpha changes that.

    Every allowed arm is tried once before any greedy choice is made. Without that, an arm whose
    first sampled reward happened to be low is never revisited unless exploration stumbles onto
    it, and the resulting behaviour depends on the order the guardrail happened to permit arms
    in rather than on their value.

    `select()` is idempotent within a control step. The guardrail calls it a second time when it
    masks out the first proposal (agent/guardrail.py step 3), so a fresh coin flip on the second
    call would both change the answer and consume a different amount of randomness depending on
    how often the guardrail fired, which would make a seeded run irreproducible. The coin for a
    step is drawn once and cached against that step's index.
    """

    def __init__(self, n_actions: int, epsilon: float, seed: int = 0):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.values = np.zeros(self.n_actions, dtype=float)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.rng = np.random.default_rng(int(seed))
        self._coin_step = None
        self._coin = None

    def _coin_for_step(self, step_idx: Optional[int]):
        """One (explore?, uniform) pair per control step, redrawn only when the step advances."""
        if step_idx is None:
            return float(self.rng.random()), float(self.rng.random())
        if self._coin_step != step_idx:
            self._coin_step = step_idx
            self._coin = (float(self.rng.random()), float(self.rng.random()))
        return self._coin

    def select(
        self,
        x: Optional[np.ndarray] = None,
        allowed: Optional[np.ndarray] = None,
        step_idx: Optional[int] = None,
    ) -> int:
        if allowed is None:
            idx = np.arange(self.n_actions)
        else:
            idx = np.flatnonzero(np.asarray(allowed, dtype=bool))
        if idx.size == 0:
            raise ValueError("no allowed arms")

        untried = idx[self.counts[idx] == 0]
        if untried.size:
            return int(untried[0])

        explore_u, pick_u = self._coin_for_step(step_idx)
        if explore_u < self.epsilon:
            return int(idx[min(int(pick_u * idx.size), idx.size - 1)])
        best = self.values[idx]
        return int(idx[int(np.argmax(best))])  # ties break toward the lower (safer) index

    def update(self, x, action: int, reward: float) -> None:
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise IndexError(f"action {a} outside 0..{self.n_actions - 1}")
        self.counts[a] += 1
        # Incremental mean. Equivalent to storing every reward, without the memory, and with no
        # forgetting factor: the environment is non-stationary within a run, but a forgetting
        # factor is another tuned hyperparameter and this baseline exists to be simple.
        self.values[a] += (float(reward) - self.values[a]) / self.counts[a]
