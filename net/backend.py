"""The network backend contract.

Three implementations sit behind this: SimBackend (pure Python, runs anywhere), OvsCliBackend
(real Mininet + OVS driven over the CLI), and RyuBackend (OpenFlow 1.3 via a controller app).
The point of the abstraction is that the experiment runner, the policies, the guardrail and the
metrics are identical across all three, so a result from one is directly comparable to a result
from another.

Timing semantics, which are the part most easily got wrong:

    read_telemetry() returns the aggregate telemetry for ONE elapsed control interval and
    advances the backend past it.

For SimBackend "advances" means running the queue model forward. For the real backends it means
blocking until the next interval boundary and returning the counter deltas across it. Same
contract, so the runner loop does not branch on backend type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# Queue index convention, used everywhere. q0 URLLC, q1 eMBB, q2 Best Effort.
URLLC, EMBB, BE = 0, 1, 2
SLICE_NAMES = ("urllc", "embb", "be")
N_QUEUES = 3


@dataclass(frozen=True)
class TelemetrySample:
    """One control interval's worth of observation.

    Frozen because a telemetry sample is evidence. Nothing downstream should be able to edit it
    after the fact, which also means a policy cannot accidentally leak state through it.

    All rates are bits per second averaged over the interval. Drop counts are per-interval
    deltas in packets, not cumulative counters, because cumulative counters make every
    downstream metric depend on run length.
    """

    t_wall: float                  # unix time at the end of the interval
    t_step: float                  # seconds since run start at the end of the interval
    urllc_rtt_ms_p50: float
    urllc_rtt_ms_p95: float
    urllc_tx_bps: float            # URLLC bits actually served over the bottleneck
    embb_goodput_bps: float
    be_goodput_bps: float
    q0_drops: int                  # packets dropped this interval
    q1_drops: int
    q2_drops: int
    link_util: float               # served bits / (capacity * interval), in [0, 1]
    backlog_bytes_per_queue: Tuple[float, float, float]

    # Offered load is not part of the brief's minimum schema, but without it the Oracle cannot
    # be computed and the sim/OVS comparison has no independent variable to align on. It is
    # observable in the sim and derivable on the testbed from the iperf3 target rates.
    offered_bps_per_queue: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def total_drops(self) -> int:
        return int(self.q0_drops) + int(self.q1_drops) + int(self.q2_drops)

    @property
    def backlog_total_bytes(self) -> float:
        return float(sum(self.backlog_bytes_per_queue))


@dataclass(frozen=True)
class Allocation:
    """A concrete rate-cap assignment for the s1 -> s2 egress port.

    level_index is the action the policy chose; the three caps are the deterministic function
    of it defined by allocation_from_level(). Carrying both means the CSV records what was
    decided and what was actually programmed, which is what you need when the two disagree.
    """

    level_index: int
    embb_share: float
    urllc_cap_bps: float
    embb_cap_bps: float
    be_cap_bps: float

    def caps_array(self) -> np.ndarray:
        return np.array(
            [self.urllc_cap_bps, self.embb_cap_bps, self.be_cap_bps], dtype=float
        )


def allocation_from_level(cfg, level_index: int) -> Allocation:
    """Map an action index onto the three queue rate caps.

    URLLC is never rate-capped. Capping the protected slice would be self-defeating: its
    protection comes from its min-rate guarantee plus whatever headroom the action leaves it.

    Best Effort gets a FIXED cap, deliberately not coupled to the action. An earlier draft set
    be_cap = (1 - level - urllc_reserve) * C, which looks tidy but holds the sum of all three
    demands constant, so changing the action changed nothing about URLLC's service. Keeping the
    BE cap fixed means raising eMBB genuinely takes capacity away from URLLC, which is the
    effect the controller exists to manage.
    """
    capacity = float(cfg.link.capacity_bps)
    levels = list(cfg.action.embb_levels)
    if not 0 <= level_index < len(levels):
        raise IndexError(f"level_index {level_index} outside 0..{len(levels) - 1}")
    share = float(levels[level_index])
    return Allocation(
        level_index=int(level_index),
        embb_share=share,
        urllc_cap_bps=capacity,
        embb_cap_bps=share * capacity,
        be_cap_bps=float(cfg.slices.be_cap_share) * capacity,
    )


class NetworkBackend(ABC):
    """Everything the control loop is allowed to know about the network."""

    name: str = "abstract"

    @abstractmethod
    def read_telemetry(self) -> TelemetrySample:
        """Advance one control interval and return its aggregate telemetry."""

    @abstractmethod
    def apply_allocation(self, alloc: Allocation) -> None:
        """Program the rate caps. Takes effect from the next interval onward."""

    @abstractmethod
    def reset(self) -> None:
        """Return to a clean start-of-run state. Must make runs reproducible."""

    @property
    def done(self) -> bool:
        """True when the backend has no more data. Only SimBackend ever ends on its own."""
        return False

    def close(self) -> None:
        """Release any external resources. No-op by default."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
