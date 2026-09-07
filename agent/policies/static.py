"""Static baselines.

StaticEqual is the naive reference: split the link evenly across the three slices and never
move. Its eMBB level is the one nearest 1/3 of capacity, computed from the configured level set
rather than hard-coded, so changing config/default.yaml cannot silently make this baseline mean
something else.

StaticSafe is the over-provisioning reference from the brief: the lowest eMBB level, leaving the
most headroom for URLLC. It is the "never violates the SLA but wastes capacity" corner. It is
the honest thing our method has to beat, because if dynamic slicing cannot buy more eMBB goodput
than StaticSafe at comparable SLA risk, the whole premise is dead.

Both are wrapped by the same guardrail as every learned policy. On most steps the guardrail will
have nothing to do for StaticSafe, which is exactly the point: its guardrail intervention rate is
the floor that the learned policies' rates get compared against.
"""

from __future__ import annotations

import numpy as np

from agent.policies.base import Policy, nearest_allowed


class _FixedLevel(Policy):
    """Always wants one level; falls back to the nearest allowed one when masked."""

    def __init__(self, cfg, target_index: int, name: str):
        super().__init__(cfg)
        self.name = name
        self.target_index = int(target_index)

    def select(self, ctx, allowed: np.ndarray) -> int:
        if allowed[self.target_index]:
            return self.target_index
        return nearest_allowed(self.target_index, allowed)


class StaticEqual(_FixedLevel):
    def __init__(self, cfg):
        levels = np.asarray(cfg.action.embb_levels, dtype=float)
        equal_share = 1.0 / 3.0
        target = int(np.argmin(np.abs(levels - equal_share)))
        super().__init__(cfg, target, "static_equal")


class StaticSafe(_FixedLevel):
    def __init__(self, cfg):
        super().__init__(cfg, int(cfg.action.floor_index), "static_safe")
