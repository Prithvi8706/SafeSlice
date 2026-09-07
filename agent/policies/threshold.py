"""The reactive latency-threshold baseline.

This is the honest non-learning competitor. It is the policy an engineer writes in an afternoon
without any machine learning at all, and if the contextual bandit cannot beat it then the answer
to the research question is that at this scale learning is not warranted. That outcome is
allowed. docs/PLAN.md section 4 says so up front.

The rule, per control step:

    rtt_ewma > up_ms    ->  step DOWN one level  (give capacity back to URLLC)
    rtt_ewma < down_ms  ->  step UP one level    (take capacity for eMBB)
    otherwise           ->  hold

Three things about this design are deliberate and are the reason the comparison is meaningful.

**It watches the smoothed latency only.** `agent/context.py` exposes both a smoothed latency and
an instantaneous tail, and docs/DESIGN.md section 4 identifies the gap between them as the only
mechanism by which a learned policy can respond faster than a threshold. Handing Threshold the
fast signal too would close that gap and make the comparison uninformative. It would also stop
being the baseline it is supposed to represent, which is a controller reacting to a smoothed
telemetry series.

**It is reactive, so it can only respond to damage that has already happened.** Latency rises
because the previous action was too aggressive for the load that arrived. Threshold sees the
consequence; it never sees the cause. `urllc_util` is in the context vector precisely because a
policy that can see URLLC's offered load can move *before* latency rises. Threshold does not
read it. That asymmetry is the hypothesis this project tests, and it has to be visible in the
code rather than asserted in the report.

**`select()` does not mutate state; `update()` does.** The guardrail may call `select()` a second
time in the same control step when its first proposal is masked out (agent/guardrail.py step 3).
A policy that stepped its internal level inside `select()` would therefore move two levels on
any step where the guardrail intervened, and the bug would look like "the threshold baseline
behaves erratically under load", which is exactly when it matters and exactly the kind of thing
that gets written up as a finding. So the current level is committed in `update()`, from the
action that was actually APPLIED. Resuming from the applied level after a guardrail clamp is
also the correct control behaviour: the controller should track the network it is in, not the
one it asked for.

There is no hysteresis state beyond the current level. The dead band between `down_ms` and
`up_ms` is what stops it chattering, and it is stated in config/default.yaml rather than here.
"""

from __future__ import annotations

import numpy as np

from agent.policies.base import Policy, nearest_allowed


class Threshold(Policy):
    name = "threshold"

    def __init__(self, cfg):
        super().__init__(cfg)
        tcfg = cfg.policy.threshold
        self.up_ms = float(tcfg.up_ms)
        self.down_ms = float(tcfg.down_ms)
        self.step = int(tcfg.step)
        self.initial_index = int(cfg.action.initial_index)

        if self.down_ms >= self.up_ms:
            raise ValueError(
                "policy.threshold.down_ms must be below up_ms or the dead band is empty "
                "and the controller chatters every step"
            )
        if self.step < 1:
            raise ValueError("policy.threshold.step must be at least 1")

        self.reset()

    def reset(self, seed: int = 0) -> None:
        self.level = int(self.initial_index)

    def select(self, ctx, allowed: np.ndarray) -> int:
        rtt = float(ctx.rtt_ewma_ms)
        target = self.level
        if rtt > self.up_ms:
            target = self.level - self.step
        elif rtt < self.down_ms:
            target = self.level + self.step
        target = int(np.clip(target, 0, self.n_actions - 1))

        if allowed[target]:
            return target
        return nearest_allowed(target, allowed)

    def update(self, ctx, action: int, reward: float) -> None:
        # Commit the level that was actually applied, not the one we asked for. See the module
        # docstring: this is what keeps select() idempotent within a step and what makes the
        # controller resume from reality after a guardrail clamp.
        self.level = int(action)
