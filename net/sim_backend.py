"""A queueing simulator for the s1 -> s2 bottleneck.

WHY A TOKEN BUCKET
The real actuator is an OVS HTB queue with other-config:max-rate. That primitive is a token
bucket. Modelling the sim with the same primitive means "cap eMBB at 0.5 of capacity" denotes
the same thing in sim and on the testbed, which is the only property that makes a sim-tuned
policy meaningful on real hardware. A pure fluid model would not reproduce burst absorption;
an M/M/1 model has no rate cap at all, so the action space would not exist.

WHY NOT STRICT PRIORITY
If q0 were served with strict priority, URLLC would never queue, latency would never rise, the
guardrail would never fire, and the bandit would have nothing to learn. We would report zero SLA
violations and call it a result. That is a failure mode that flatters the method, so the sim
uses rate-capped work-conserving sharing, which is what HTB actually does.

THE LOAD-BEARING ASSUMPTION: HOW LEFTOVER CAPACITY IS SPLIT
Once every queue has taken its min-rate guarantee, something has to decide who gets the rest.
The choice is not cosmetic, it decides whether this project has a problem to solve:

  demand_proportional (default)
      A backlogged class with a high ceiling captures most of the leftover. URLLC's small
      residual demand gets crowded out, backlog builds, latency rises. The control problem
      exists. This is the PESSIMISTIC end of the plausible range.

  equal
      Strict per-queue fair share, which is what an idealised DRR scheduler does. Under this,
      URLLC's residual demand is almost always covered for free, no controller is needed, and
      the entire project is vacuous.

  min_rate_proportional
      Leftover split in proportion to the min-rate guarantees. This is what the HTB
      documentation describes for borrowing between classes. Intermediate, and protective of
      URLLC because URLLC has the largest guarantee.

We picked demand_proportional and we are stating plainly that it is an assumption, not a
measurement. Week 1b runs the identical trace through OvsCliBackend and compares all three
variants against it. If the real testbed behaves like `equal`, then the honest finding is that
this control problem does not exist at this operating point, and that is what the report will
say. See docs/DESIGN.md.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from net.backend import (
    N_QUEUES,
    URLLC,
    Allocation,
    NetworkBackend,
    TelemetrySample,
    allocation_from_level,
)
from traffic.traces import Trace

_EXCESS_MODES = ("demand_proportional", "equal", "min_rate_proportional")


class SimBackend(NetworkBackend):
    """Sub-step queue simulation of one shared bottleneck with three shaped queues."""

    name = "sim"

    def __init__(self, cfg, trace: Trace, virtual_time: bool = True):
        self.cfg = cfg
        self.trace = trace
        self.virtual_time = bool(virtual_time)

        self.capacity_bps = float(cfg.link.capacity_bps)
        self.base_rtt_ms = float(cfg.link.base_rtt_ms)
        self.substep_s = float(trace.substep_s)
        self.substeps_per_step = int(trace.substeps_per_step)
        self.interval_s = float(trace.control_interval_s)

        self.queue_limit_bytes = float(cfg.sim.queue_limit_bytes)
        self.packet_bytes = float(cfg.sim.packet_bytes)
        self.burst_multiplier = float(cfg.sim.bucket_burst_multiplier)
        self.burst_min_bytes = float(cfg.sim.bucket_burst_min_bytes)

        self.excess_mode = str(cfg.sim.excess_sharing)
        if self.excess_mode not in _EXCESS_MODES:
            raise ValueError(
                f"sim.excess_sharing must be one of {_EXCESS_MODES}, got {self.excess_mode!r}"
            )

        self.min_shares = np.array(
            [cfg.slices.urllc.min_share, cfg.slices.embb.min_share, cfg.slices.be.min_share],
            dtype=float,
        )
        if self.min_shares.sum() >= 1.0:
            raise ValueError(
                f"min_shares sum to {self.min_shares.sum():.3f}; must be < 1.0 or there is no "
                "leftover capacity for the action to influence"
            )

        # Bytes servable by the whole link in one sub-step, and each queue's guaranteed bytes.
        self._link_bytes_per_sub = self.capacity_bps * self.substep_s / 8.0
        self._min_bytes_per_sub = self.min_shares * self._link_bytes_per_sub

        self.reset()

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        self._sub_idx = 0
        self._step_idx = 0
        self._t0_wall = time.time()
        self.backlog = np.zeros(N_QUEUES, dtype=float)
        self.tokens = np.zeros(N_QUEUES, dtype=float)
        self._drop_bytes_accum = np.zeros(N_QUEUES, dtype=float)
        self._drop_pkts_reported = np.zeros(N_QUEUES, dtype=np.int64)
        self._set_caps(allocation_from_level(self.cfg, int(self.cfg.action.initial_index)))

    @property
    def done(self) -> bool:
        return self._sub_idx >= self.trace.n_substeps

    def apply_allocation(self, alloc: Allocation) -> None:
        self._set_caps(alloc)

    def _set_caps(self, alloc: Allocation) -> None:
        """Program the caps and re-derive each queue's bucket depth from its own cap.

        The bucket must be sized per queue. A global constant larger than a queue's per-substep
        accrual pins the bucket full and the max-rate cap silently stops binding, which is what
        made levels 0.50, 0.65 and 0.80 produce identical telemetry before this was fixed.
        """
        self._alloc = alloc
        self._caps = alloc.caps_array()
        accrual = self._caps * self.substep_s / 8.0
        self._burst_bytes = np.maximum(accrual * self.burst_multiplier, self.burst_min_bytes)
        # A cap change must not leave stale tokens above the new bucket depth.
        if hasattr(self, "tokens"):
            self.tokens = np.minimum(self.tokens, self._burst_bytes)

    # ------------------------------------------------------------------ scheduling

    def _share_excess(self, residual_link: float, residual_demand: np.ndarray) -> np.ndarray:
        """Split `residual_link` bytes among queues, never exceeding residual_demand.

        Water-filling: hand out by the chosen rule, clip at each queue's remaining demand, then
        redistribute whatever that clipping freed among the queues that still want more. With
        three queues this converges in at most three passes.
        """
        granted = np.zeros(N_QUEUES, dtype=float)
        remaining_link = float(residual_link)
        remaining_demand = residual_demand.astype(float).copy()

        if self.excess_mode == "demand_proportional":
            base_weights = residual_demand.astype(float).copy()
        elif self.excess_mode == "equal":
            base_weights = np.ones(N_QUEUES, dtype=float)
        else:  # min_rate_proportional
            base_weights = self.min_shares.copy()

        for _ in range(N_QUEUES):
            if remaining_link <= 1e-12:
                break
            active = remaining_demand > 1e-12
            if not active.any():
                break
            w = np.where(active, base_weights, 0.0)
            if w.sum() <= 1e-12:
                # Every active queue has zero weight (only possible under
                # demand_proportional with vanishing demand). Fall back to equal split.
                w = active.astype(float)
            offer = remaining_link * w / w.sum()
            take = np.minimum(offer, remaining_demand)
            granted += take
            remaining_demand -= take
            remaining_link -= take.sum()

        return granted

    def _serve_one_substep(self, arrivals_bytes: np.ndarray):
        """Run one sub-step. Returns (served_bytes, urllc_rtt_ms)."""
        # 1. Enqueue with tail drop at the queue limit.
        room = np.maximum(self.queue_limit_bytes - self.backlog, 0.0)
        accepted = np.minimum(arrivals_bytes, room)
        self._drop_bytes_accum += arrivals_bytes - accepted
        self.backlog += accepted

        # 2. Accrue tokens up to the bucket depth.
        self.tokens = np.minimum(
            self.tokens + self._caps * self.substep_s / 8.0, self._burst_bytes
        )

        # 3. Each queue may offer min(backlog, tokens) to the link.
        demand = np.minimum(self.backlog, self.tokens)
        total_demand = float(demand.sum())

        if total_demand <= self._link_bytes_per_sub + 1e-12:
            served = demand.copy()
        else:
            # Min-rate guarantees are honoured first, scaled down only in the pathological case
            # where the guarantees themselves oversubscribe the link.
            guaranteed = np.minimum(demand, self._min_bytes_per_sub)
            g_total = float(guaranteed.sum())
            if g_total > self._link_bytes_per_sub:
                served = guaranteed * (self._link_bytes_per_sub / g_total)
            else:
                residual_link = self._link_bytes_per_sub - g_total
                served = guaranteed + self._share_excess(residual_link, demand - guaranteed)

        served = np.minimum(served, demand)
        self.backlog -= served
        self.tokens -= served
        np.maximum(self.backlog, 0.0, out=self.backlog)
        np.maximum(self.tokens, 0.0, out=self.tokens)

        # 4. URLLC one-way queueing delay.
        #
        # Only the s1 -> s2 direction is shaped, so the echo reply crosses an unloaded path and
        # ONE queueing delay is added, not two. An earlier draft of docs/PLAN.md said 2x; that
        # was wrong and would have inflated every latency number in the report.
        #
        # The drain rate is floored at the min-rate guarantee. Without a floor, a sub-step in
        # which URLLC happens to be served zero bytes produces an infinite delay, which would
        # dominate p95 and turn a scheduling artefact into a headline metric.
        drain_bps = max(served[URLLC] * 8.0 / self.substep_s, self.min_shares[URLLC] * self.capacity_bps)
        queue_delay_ms = self.backlog[URLLC] * 8.0 / drain_bps * 1000.0
        rtt_ms = self.base_rtt_ms + queue_delay_ms

        return served, rtt_ms

    # ------------------------------------------------------------------ telemetry

    def read_telemetry(self) -> TelemetrySample:
        if self.done:
            raise RuntimeError("SimBackend trace exhausted; check run.duration_s vs the trace")

        n = min(self.substeps_per_step, self.trace.n_substeps - self._sub_idx)
        served_total = np.zeros(N_QUEUES, dtype=float)
        offered_total = np.zeros(N_QUEUES, dtype=float)
        rtts = np.empty(n, dtype=float)

        for k in range(n):
            offered_bps = self.trace.offered_bps[:, self._sub_idx]
            arrivals = offered_bps * self.substep_s / 8.0
            offered_total += arrivals
            served, rtt = self._serve_one_substep(arrivals)
            served_total += served
            rtts[k] = rtt
            self._sub_idx += 1

        elapsed_s = n * self.substep_s

        # Dropped bytes are accumulated as a float and converted to whole packets only when
        # reported, with the fractional remainder carried forward. Rounding every sub-step would
        # bias the drop count, and the drop count is a headline metric.
        cumulative_pkts = np.floor(self._drop_bytes_accum / self.packet_bytes).astype(np.int64)
        drops_this_step = cumulative_pkts - self._drop_pkts_reported
        self._drop_pkts_reported = cumulative_pkts

        served_bps = served_total * 8.0 / elapsed_s
        offered_bps_avg = offered_total * 8.0 / elapsed_s
        link_util = float(served_total.sum() * 8.0 / (self.capacity_bps * elapsed_s))

        self._step_idx += 1
        t_step = self._sub_idx * self.substep_s
        t_wall = self._t0_wall + t_step if self.virtual_time else time.time()

        return TelemetrySample(
            t_wall=float(t_wall),
            t_step=float(t_step),
            urllc_rtt_ms_p50=float(np.percentile(rtts, 50)),
            urllc_rtt_ms_p95=float(np.percentile(rtts, 95)),
            urllc_tx_bps=float(served_bps[0]),
            embb_goodput_bps=float(served_bps[1]),
            be_goodput_bps=float(served_bps[2]),
            q0_drops=int(drops_this_step[0]),
            q1_drops=int(drops_this_step[1]),
            q2_drops=int(drops_this_step[2]),
            link_util=link_util,
            backlog_bytes_per_queue=tuple(float(b) for b in self.backlog),
            offered_bps_per_queue=tuple(float(b) for b in offered_bps_avg),
        )

    def idle_sample(self) -> TelemetrySample:
        """A synthetic zero-load sample used to prime the context builder at step 0.

        Not a measurement. It exists so the first decision is made against a defined state
        rather than an all-zero vector that implies an impossibly good latency.
        """
        return TelemetrySample(
            t_wall=self._t0_wall,
            t_step=0.0,
            urllc_rtt_ms_p50=self.base_rtt_ms,
            urllc_rtt_ms_p95=self.base_rtt_ms,
            urllc_tx_bps=0.0,
            embb_goodput_bps=0.0,
            be_goodput_bps=0.0,
            q0_drops=0,
            q1_drops=0,
            q2_drops=0,
            link_util=0.0,
            backlog_bytes_per_queue=(0.0, 0.0, 0.0),
            offered_bps_per_queue=(0.0, 0.0, 0.0),
        )


def make_sim_backend(cfg, seed: int, trace: Optional[Trace] = None) -> SimBackend:
    from traffic.traces import build_trace

    if trace is None:
        trace = build_trace(cfg, seed)
    return SimBackend(cfg, trace)
