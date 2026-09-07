"""The deterministic SLA guardrail.

It sits BETWEEN the policy and the actuator, not after it. The sequence per control step is:

    1. The policy proposes an action against the unrestricted action set. This is logged as
       `proposed_action` and is the only reason the unrestricted call happens at all: without
       it we could not measure how often the guardrail actually changed the outcome.
    2. The guardrail computes an allowed mask from the context, in physical units.
    3. If the proposal is outside the mask, the policy is asked to choose again from the
       allowed set. It is asked, not overruled, because the brief specifies that the bandit
       "chooses among the remaining ones", which preserves whatever the policy knows about the
       relative value of the safe arms.
    4. Whatever comes back is clamped to the mask regardless. A policy bug must not be able to
       produce an unsafe allocation. The invariant "applied is always inside the mask" is a
       property of this file alone.
    5. Every step is logged with the proposal, the applied action and a reason string.

States, in priority order:

    holddown      a hard override is still in force from a previous step. Clamped to floor.
    hard_override the SLA hard limit is breached now. Clamp to floor and arm the holddown.
    warn_mask     latency is in the warning band. Levels above the safe ceiling are masked out.
    ok            no restriction.

Intervention is defined as `applied_action != proposed_action`. A step where the guardrail
masked arms but the policy had already chosen a safe one is NOT counted as an intervention,
because nothing about the network changed. Counting it would inflate our headline safety metric
by rewarding the guardrail for steps where it did nothing, and that metric is the direct
evidence for the word "safe" in the project title.

WHY THE INSTANTANEOUS TAIL IS CHECKED, NOT ONLY THE EWMA
The EWMA lags by roughly 1/alpha steps. A guardrail that watched only the EWMA would be late by
construction against exactly the short sharp spikes that config/scenarios/adversarial.yaml is
built to produce, and the project would be claiming a safety property it does not have. So the
hard override fires on either the EWMA or the current p95 exceeding the hard limit, and the
warning band likewise. `escalation_source` records which one tripped, so the report can say how
much of the safety came from the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

REASON_OK = "ok"
REASON_DISABLED = "disabled"
REASON_WARN_PASS = "warn_mask_pass"
REASON_WARN_RESELECT = "warn_mask_reselect"
REASON_HARD_OVERRIDE = "hard_override"
REASON_HOLDDOWN = "holddown"

ALL_REASONS = (
    REASON_OK,
    REASON_DISABLED,
    REASON_WARN_PASS,
    REASON_WARN_RESELECT,
    REASON_HARD_OVERRIDE,
    REASON_HOLDDOWN,
)


@dataclass(frozen=True)
class GuardrailDecision:
    proposed_action: int
    applied_action: int
    reason: str
    intervened: bool
    allowed_mask: Tuple[bool, ...]
    escalation_source: str  # "none", "ewma", "p95", or "both"


class Guardrail:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(cfg.guardrail.enabled)
        self.warn_ms = float(cfg.guardrail.warn_ms)
        self.hard_ms = float(cfg.sla.hard_ms)
        self.holddown_steps = int(cfg.guardrail.holddown_steps)
        self.floor_index = int(cfg.action.floor_index)
        self.safe_ceiling_index = int(cfg.guardrail.safe_ceiling_index)
        self.n_actions = len(cfg.action.embb_levels)

        if not 0 <= self.floor_index < self.n_actions:
            raise ValueError("action.floor_index outside the action set")
        if not 0 <= self.safe_ceiling_index < self.n_actions:
            raise ValueError("guardrail.safe_ceiling_index outside the action set")
        if self.safe_ceiling_index < self.floor_index:
            raise ValueError("guardrail.safe_ceiling_index is below action.floor_index")
        if self.warn_ms >= self.hard_ms:
            raise ValueError("guardrail.warn_ms must be below sla.hard_ms or it never warns")

        self.reset()

    def reset(self) -> None:
        self._holddown_remaining = 0
        self.interventions = 0
        self.reason_counts = {r: 0 for r in ALL_REASONS}

    # ------------------------------------------------------------------ masks

    def _floor_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.floor_index] = True
        return mask

    def _warn_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.floor_index : self.safe_ceiling_index + 1] = True
        return mask

    def _escalation_source(self, ewma_ms: float, p95_ms: float, threshold: float) -> str:
        e = ewma_ms > threshold
        p = p95_ms > threshold
        if e and p:
            return "both"
        if e:
            return "ewma"
        if p:
            return "p95"
        return "none"

    # ------------------------------------------------------------------ main entry

    def apply(self, ctx, proposed_action: int, policy) -> GuardrailDecision:
        """Evaluate `proposed_action` against `ctx` and return what will actually be applied."""
        proposed = int(proposed_action)

        if not self.enabled:
            mask = np.ones(self.n_actions, dtype=bool)
            return self._finish(proposed, proposed, REASON_DISABLED, mask, "none")

        ewma = float(ctx.rtt_ewma_ms)
        p95 = float(ctx.rtt_p95_ms)

        # 1. An armed hold-down outranks everything, including a currently healthy latency.
        #    That is the point of a hold-down: it stops the controller oscillating back to an
        #    aggressive level the instant the queue drains, which would re-create the breach.
        if self._holddown_remaining > 0:
            self._holddown_remaining -= 1
            mask = self._floor_mask()
            return self._finish(proposed, self.floor_index, REASON_HOLDDOWN, mask, "none")

        # 2. Hard breach. Clamp to the floor and arm the hold-down.
        hard_src = self._escalation_source(ewma, p95, self.hard_ms)
        if hard_src != "none":
            self._holddown_remaining = self.holddown_steps
            mask = self._floor_mask()
            return self._finish(
                proposed, self.floor_index, REASON_HARD_OVERRIDE, mask, hard_src
            )

        # 3. Warning band. Mask the aggressive levels and let the policy re-choose.
        warn_src = self._escalation_source(ewma, p95, self.warn_ms)
        if warn_src != "none":
            mask = self._warn_mask()
            if mask[proposed]:
                return self._finish(proposed, proposed, REASON_WARN_PASS, mask, warn_src)
            applied = int(policy.select(ctx, mask))
            return self._finish(proposed, applied, REASON_WARN_RESELECT, mask, warn_src)

        # 4. Nothing to do.
        mask = np.ones(self.n_actions, dtype=bool)
        return self._finish(proposed, proposed, REASON_OK, mask, "none")

    def _finish(
        self,
        proposed: int,
        applied: int,
        reason: str,
        mask: np.ndarray,
        escalation_source: str,
    ) -> GuardrailDecision:
        # Final clamp. The invariant that the applied action is inside the mask is enforced
        # here and nowhere else, so a buggy or adversarial policy cannot produce an unsafe
        # allocation. tests/test_guardrail.py proves this with a policy that always returns
        # the most aggressive arm.
        applied = int(applied)
        if not (0 <= applied < self.n_actions) or not mask[applied]:
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                raise RuntimeError("guardrail produced an empty allowed mask")
            distances = np.abs(candidates - applied)
            applied = int(candidates[distances == distances.min()].min())

        intervened = applied != proposed
        if intervened:
            self.interventions += 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

        return GuardrailDecision(
            proposed_action=int(proposed),
            applied_action=applied,
            reason=reason,
            intervened=bool(intervened),
            allowed_mask=tuple(bool(b) for b in mask),
            escalation_source=escalation_source,
        )
