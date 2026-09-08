"""The Oracle upper bound, and an honest account of what it is an upper bound on.

WHAT IT IS
For each phase of the scenario, the Oracle plays the single eMBB level that scored best on that
phase when held fixed for the whole run. It knows the phase boundaries and the outcome of every
level in advance. No deployable controller can know either, which is the point: it is a
reference line, not a method, and `Policy.is_oracle` is True so it never sits in a table
alongside policies that could actually be deployed.

WHAT IT IS *NOT* AN UPPER BOUND ON
It is the best PHASE-STATIC allocation, not the best allocation. A dynamic policy is free to
change level within a phase - which is exactly what Threshold and the bandits do - and one that
does so well can beat this reference. **Regret against this Oracle can therefore be negative**,
and a negative value is not a bug and not a paradox. docs/PLAN.md section D3 flagged this in week
1 so it would not be a surprise in week 5. The metric is named "regret vs the best phase-static
allocation" everywhere it appears, and the report must not shorten that to "regret vs optimal".

A true oracle over the joint sequence of actions is exponential in the number of steps
(5 levels ^ 270 steps), so it is not available at any price.

TWO CHOICES THAT KEEP IT HONEST

**The assembled schedule is executed, not summed.** The obvious shortcut is to add up each
phase's best score from the probe runs and call the total the Oracle's result. That number is
unreachable: queue backlog crosses phase boundaries (docs/DESIGN.md section 1), so entering
phase 2 from the level that won phase 1 is not the same as entering it from the level that was
held all run. Summing the probes would inflate the reference and make every real policy look
worse than it is. `experiments/run_experiment.py` runs the assembled schedule end to end and
reports what it actually measured.

**The Oracle is guardrailed like everything else.** It could score higher unguarded, but then
the comparison would confound "knows the future" with "is allowed to violate the SLA", and the
guardrail intervention rate column would mean two different things in the same table. So this is
the best phase-static schedule *subject to the same safety mechanism*.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from agent.policies.base import Policy, nearest_allowed


class Oracle(Policy):
    """Plays a precomputed level per scenario phase.

    `schedule` maps a phase id to a level index, and `step_phase` maps a step index to a phase
    id. Both are computed by `experiments/run_experiment.compute_oracle_schedule`, which needs
    the backend and the trace and therefore cannot live behind the Policy interface.
    """

    name = "oracle"
    is_oracle = True

    def __init__(self, cfg, schedule: Dict[int, int], step_phase: Sequence[int]):
        super().__init__(cfg)
        self.schedule = {int(k): int(v) for k, v in schedule.items()}
        self.step_phase = np.asarray(step_phase, dtype=int)
        self.default_level = int(cfg.action.floor_index)
        for phase, level in self.schedule.items():
            if not 0 <= level < self.n_actions:
                raise ValueError(f"phase {phase} scheduled at level {level}, outside the action set")

    def _level_for(self, step_idx: int) -> int:
        if 0 <= step_idx < self.step_phase.size:
            phase = int(self.step_phase[step_idx])
            return self.schedule.get(phase, self.default_level)
        # Past the end of the trace the run is over; the floor is the safe answer and cannot
        # flatter the Oracle, since no telemetry beyond the trace is ever scored.
        return self.default_level

    def select(self, ctx, allowed: np.ndarray) -> int:
        target = self._level_for(int(ctx.step_idx))
        if allowed[target]:
            return target
        return nearest_allowed(target, allowed)
