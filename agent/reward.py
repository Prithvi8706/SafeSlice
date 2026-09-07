"""The reward function.

    r = w_tput * norm(embb_goodput)
      + w_be   * norm(be_goodput)
      - w_sla  * max(0, rtt - sla_target) / sla_target
      - w_viol * 1[rtt > sla_hard]
      - w_drop * norm(total_drops)

Choices that need defending, because the reward is the thing that decides what "good" means and
every result in the report is downstream of it:

  1. Which RTT statistic. Config-selected, default p95. A latency SLO stated on the median is
     close to meaningless, since half the traffic can miss it and the metric never notices. p50
     is still logged so the sensitivity of the conclusion to this choice can be checked.

  2. Goodput normalization is by LINK CAPACITY, not by offered load. Normalizing by offered load
     would turn the reward into a fraction-served measure, under which starving eMBB during a
     burst scores the same as serving it fully during a lull. Capacity is the fixed resource we
     are actually allocating, so it is the honest denominator.

  3. Best Effort carries a real positive weight (0.30). If BE were worth zero, the optimal
     policy would be to hand the entire non-URLLC link to eMBB and the action space would
     collapse to "as high as the guardrail permits". A nonzero BE weight is what makes the
     allocation a genuine tradeoff rather than a one-sided maximisation.

  4. The SLA term is a hinge, not a step, so a policy that misses the target by 1 ms is scored
     better than one that misses by 50 ms. The separate w_viol step then makes crossing the hard
     limit discretely expensive. Hinge alone would let a policy trade many small misses for
     throughput; step alone would make everything below the hard limit look identical.

  5. Drops are normalized by the number of packets the link could carry in one control interval.
     That is a physical reference rather than an arbitrary scale, so w_drop is comparable across
     link capacities.

The reward is computed from telemetry ONLY. It never sees the action, the policy identity, or
the guardrail state. If it did, we would be able to reward a policy for its own machinery rather
than for the network outcome it produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from net.backend import TelemetrySample


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-term decomposition. Logged so the report can show which term drove a result."""

    total: float
    tput_term: float
    be_term: float
    sla_term: float
    viol_term: float
    drop_term: float
    rtt_used_ms: float
    violated: bool


def reward_breakdown(tel: TelemetrySample, cfg) -> RewardBreakdown:
    capacity = float(cfg.link.capacity_bps)
    w = cfg.reward
    target = float(cfg.sla.target_ms)
    hard = float(cfg.sla.hard_ms)

    stat = str(w.rtt_statistic)
    rtt = float(getattr(tel, stat))

    drop_ref = max(
        1.0,
        capacity * float(cfg.run.control_interval_s) / 8.0 / float(cfg.sim.packet_bytes),
    )

    tput_term = float(w.w_tput) * (float(tel.embb_goodput_bps) / capacity)
    be_term = float(w.w_be) * (float(tel.be_goodput_bps) / capacity)
    sla_term = -float(w.w_sla) * max(0.0, rtt - target) / target
    violated = rtt > hard
    viol_term = -float(w.w_viol) * (1.0 if violated else 0.0)
    drop_term = -float(w.w_drop) * (float(tel.total_drops) / drop_ref)

    total = tput_term + be_term + sla_term + viol_term + drop_term
    return RewardBreakdown(
        total=float(total),
        tput_term=float(tput_term),
        be_term=float(be_term),
        sla_term=float(sla_term),
        viol_term=float(viol_term),
        drop_term=float(drop_term),
        rtt_used_ms=float(rtt),
        violated=bool(violated),
    )


def reward(tel: TelemetrySample, cfg) -> float:
    """Scalar reward for one control interval."""
    return reward_breakdown(tel, cfg).total
