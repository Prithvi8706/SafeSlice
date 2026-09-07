"""Context feature extraction.

The context is what every policy sees and the only thing a policy is allowed to condition on.
Keeping it in one place with a fixed, exported ordering means LinUCB's per-arm matrices, the
CSV columns and the plots all agree by construction rather than by coincidence.

Design notes worth defending in the report:

  * Features are normalized to roughly [0, 1] so LinUCB's single alpha is meaningful across
    dimensions. Without this, a feature measured in bits per second would dominate the
    confidence term purely because of its units.
  * A constant bias term is included so a disjoint linear model can fit a per-arm intercept.
  * The vector deliberately contains BOTH a smoothed latency (rtt_ewma_norm) and an
    instantaneous tail (rtt_p95_norm). The EWMA is what a reactive threshold policy would use;
    the instantaneous tail is what lets a policy respond faster than the EWMA time constant.
    The gap between them is the only mechanism by which a learned policy can beat a latency
    threshold on the adversarial scenario, so it has to be observable.
  * urllc_util is included because in this system the correct eMBB level is largely determined
    by whether URLLC's offered load sits above or below its min-rate guarantee. A policy that
    can see that can act BEFORE latency rises. A policy that only watches latency cannot.
    If LinUCB fails to exploit this, that is evidence about the method, not about the features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from net.backend import TelemetrySample

FEATURE_NAMES = (
    "bias",
    "rtt_ewma_norm",
    "rtt_p95_norm",
    "urllc_util",
    "embb_util",
    "be_util",
    "link_util",
    "q0_backlog_norm",
    "drops_norm",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class Context:
    """A context vector plus the raw quantities the guardrail needs in physical units.

    The guardrail must not read normalized features. It is a safety component and its thresholds
    are stated in milliseconds in the config, so it gets milliseconds.
    """

    vector: np.ndarray
    rtt_ewma_ms: float
    rtt_p95_ms: float
    step_idx: int


class ContextBuilder:
    """Stateful feature extractor. One instance per run."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha = float(cfg.context.ewma_alpha)
        self.clip_max = float(cfg.context.clip_max)
        self.capacity_bps = float(cfg.link.capacity_bps)
        self.hard_ms = float(cfg.sla.hard_ms)
        self.queue_limit_bytes = float(cfg.sim.queue_limit_bytes)
        # Reference drop count: packets the link could carry in one control interval. Gives
        # drops_norm a physical meaning rather than an arbitrary scale.
        self.drop_ref_pkts = max(
            1.0,
            self.capacity_bps * float(cfg.run.control_interval_s) / 8.0 / float(cfg.sim.packet_bytes),
        )
        self.reset()

    def reset(self) -> None:
        self._rtt_ewma = float(self.cfg.link.base_rtt_ms)
        self._drops_ewma = 0.0
        self._step = 0

    def update(self, tel: TelemetrySample) -> Context:
        """Fold one telemetry sample in and return the resulting context."""
        a = self.alpha
        self._rtt_ewma = a * float(tel.urllc_rtt_ms_p50) + (1.0 - a) * self._rtt_ewma
        self._drops_ewma = a * float(tel.total_drops) + (1.0 - a) * self._drops_ewma

        raw = np.array(
            [
                1.0,
                self._rtt_ewma / self.hard_ms,
                float(tel.urllc_rtt_ms_p95) / self.hard_ms,
                float(tel.urllc_tx_bps) / self.capacity_bps,
                float(tel.embb_goodput_bps) / self.capacity_bps,
                float(tel.be_goodput_bps) / self.capacity_bps,
                float(tel.link_util),
                float(tel.backlog_bytes_per_queue[0]) / self.queue_limit_bytes,
                self._drops_ewma / self.drop_ref_pkts,
            ],
            dtype=float,
        )
        vector = np.clip(raw, 0.0, self.clip_max)
        vector[0] = 1.0  # the bias term is never clipped or scaled

        ctx = Context(
            vector=vector,
            rtt_ewma_ms=float(self._rtt_ewma),
            rtt_p95_ms=float(tel.urllc_rtt_ms_p95),
            step_idx=self._step,
        )
        self._step += 1
        return ctx
