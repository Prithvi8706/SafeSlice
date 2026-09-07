"""The Policy interface.

Every policy, learned or not, sees the same context and returns an action index. The guardrail
wraps all of them, including the static baselines, so the cost of the guardrail is measured
identically everywhere and cannot be confused with a property of the learned policy.

`allowed` is a boolean mask over action indices. A policy MUST return an index where the mask is
True. It is not trusted to: the guardrail re-clamps whatever comes back (see agent/guardrail.py).
The mask is passed in rather than applied afterwards because "choose the best allowed arm" and
"choose the best arm, then clamp" are different policies, and the brief specifies the former.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    name: str = "base"
    #: True for policies that need the whole trace in advance and are upper bounds, not
    #: deployable controllers. Reported separately so they never sit in a table of real methods.
    is_oracle: bool = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_actions = len(cfg.action.embb_levels)

    @abstractmethod
    def select(self, ctx, allowed: np.ndarray) -> int:
        """Return an action index i with allowed[i] True."""

    def update(self, ctx, action: int, reward: float) -> None:
        """Learn from the realized reward. No-op for non-learning policies."""

    def reset(self, seed: int = 0) -> None:
        """Clear per-run state. Called once at the start of every run."""

    def freeze(self) -> None:
        """Stop exploring. No-op for anything that never explored.

        Used by the pre-trained ("converged") evaluation in docs/EXPERIMENTS.md section 1: the
        model is trained on other seeds, then frozen, then evaluated. For a learning policy this
        is a real behavioural change (alpha or epsilon to zero), not a reporting flag.
        """


def all_allowed(n_actions: int) -> np.ndarray:
    return np.ones(n_actions, dtype=bool)


def nearest_allowed(index: int, allowed: np.ndarray) -> int:
    """Closest allowed index to `index`, breaking ties toward the LOWER (safer) index.

    Ties break downward on purpose. When a policy's preferred level is masked out, the
    conservative neighbour is the correct fallback for a system whose whole claim is bounded
    SLA risk. Breaking upward would mean the guardrail occasionally makes things worse.
    """
    idx = np.flatnonzero(allowed)
    if idx.size == 0:
        raise ValueError("allowed mask has no True entries; the guardrail must always leave one")
    distances = np.abs(idx - int(index))
    best = distances.min()
    return int(idx[distances == best].min())
