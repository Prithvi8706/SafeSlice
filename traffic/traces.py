"""Deterministic offered-load traces.

A trace is the independent variable of every experiment. Same (scenario, seed) must produce the
same offered load bit for bit, otherwise policies are not being compared on identical load and
every between-policy difference is confounded with a difference in the traffic.

Resolution: offered load is drawn once per CONTROL INTERVAL and held constant across the
sub-steps inside it. That matches what the real testbed can actually do, since iperf3 is given a
target rate with -b and holds it until told otherwise; drawing per sub-step would model a
burstiness the traffic generator cannot reproduce, and the sim would then be modelling something
the testbed will never show us.

Jitter is lognormal with the mean preserved exactly, so the phase means in the YAML are the
means of the generated trace and not merely its medians.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np

from net.backend import BE, EMBB, N_QUEUES, URLLC


def trace_seed(scenario_name: str, seed: int) -> int:
    """Process-independent 64-bit seed derived from (scenario, seed).

    Deliberately not built on hash(): str hashing is salted per process, which would make the
    reproducibility guarantee silently false.
    """
    digest = hashlib.blake2b(
        f"{scenario_name}|{int(seed)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class Trace:
    """Offered load per queue, at sub-step resolution."""

    scenario: str
    seed: int
    substep_s: float
    control_interval_s: float
    substeps_per_step: int
    offered_bps: np.ndarray   # shape (N_QUEUES, n_substeps)
    phase_id: np.ndarray      # shape (n_substeps,), which YAML phase each sub-step came from
    step_offered_bps: np.ndarray  # shape (N_QUEUES, n_steps), the per-interval draw

    @property
    def n_substeps(self) -> int:
        return int(self.offered_bps.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.step_offered_bps.shape[1])

    def step_phase_id(self) -> np.ndarray:
        """Phase index per control step, which is what the Oracle brute-forces over."""
        return self.phase_id[:: self.substeps_per_step][: self.n_steps]


def _lognormal_mean_preserving(rng: np.random.Generator, mean: float, cv: float, size: int):
    """Draw `size` samples with expectation exactly `mean` and coefficient of variation `cv`."""
    if mean <= 0.0 or cv <= 0.0:
        return np.full(size, max(mean, 0.0), dtype=float)
    sigma = np.sqrt(np.log(1.0 + cv * cv))
    mu = np.log(mean) - 0.5 * sigma * sigma
    return np.exp(rng.normal(mu, sigma, size=size))


def _expand_phases(phases: List[dict], n_steps: int, interval_s: float, repeat: str):
    """Produce a per-control-step schedule of (phase_index, urllc, embb, be) mean rates in bps.

    A phase may declare `<slice>_mbps_end` to ramp linearly from `<slice>_mbps` across the phase.
    """
    if not phases:
        raise ValueError("scenario has no phases")

    means = np.zeros((N_QUEUES, 0), dtype=float)
    ids = np.zeros(0, dtype=np.int32)
    guard = 0
    while means.shape[1] < n_steps:
        guard += 1
        if guard > 10000:
            raise RuntimeError("phase expansion did not converge; check phase durations")
        for pi, phase in enumerate(phases):
            steps = int(round(float(phase["duration_s"]) / interval_s))
            if steps <= 0:
                raise ValueError(
                    f"phase {pi} of duration {phase['duration_s']}s is shorter than one "
                    f"control interval of {interval_s}s"
                )
            block = np.zeros((N_QUEUES, steps), dtype=float)
            for qi, sl in ((URLLC, "urllc"), (EMBB, "embb"), (BE, "be")):
                start = float(phase[f"{sl}_mbps"]) * 1e6
                end_key = f"{sl}_mbps_end"
                if end_key in phase:
                    end = float(phase[end_key]) * 1e6
                    # endpoint=False so a cyclic repeat does not emit the boundary value twice
                    block[qi, :] = np.linspace(start, end, steps, endpoint=False)
                else:
                    block[qi, :] = start
            means = np.concatenate([means, block], axis=1)
            ids = np.concatenate([ids, np.full(steps, pi, dtype=np.int32)])
            if means.shape[1] >= n_steps:
                break
        if repeat != "cyclic":
            break

    if means.shape[1] < n_steps:
        # repeat: once, and the phases are shorter than the run. Hold the final phase.
        pad = n_steps - means.shape[1]
        means = np.concatenate([means, np.repeat(means[:, -1:], pad, axis=1)], axis=1)
        ids = np.concatenate([ids, np.full(pad, ids[-1], dtype=np.int32)])

    return means[:, :n_steps], ids[:n_steps]


def build_trace(cfg, seed: int) -> Trace:
    """Build the offered-load trace for the scenario in `cfg` under `seed`."""
    scenario = cfg.scenario
    interval_s = float(cfg.run.control_interval_s)
    substep_s = float(cfg.sim.substep_s)
    substeps_per_step = int(round(interval_s / substep_s))
    if substeps_per_step < 1:
        raise ValueError(
            f"sim.substep_s ({substep_s}) must be <= run.control_interval_s ({interval_s})"
        )
    n_steps = int(round(float(cfg.run.duration_s) / interval_s))

    phases = [dict(p) for p in scenario.phases]
    repeat = str(scenario.get("repeat", "cyclic"))
    means, phase_ids_per_step = _expand_phases(phases, n_steps, interval_s, repeat)

    # One generator per (scenario, seed). Streams are drawn per slice in a fixed order so that
    # changing one slice's cv does not shift another slice's draws.
    #
    # blake2b, not hash(). Python salts str.__hash__ per process unless PYTHONHASHSEED is set,
    # so hash((name, seed)) would give a different trace on every invocation and silently break
    # the "same seed means the same traffic bit for bit" guarantee the whole protocol rests on.
    rng = np.random.default_rng(trace_seed(str(scenario.name), int(seed)))
    jitter_cv = scenario.get("jitter_cv", {})
    if hasattr(jitter_cv, "as_dict"):
        jitter_cv = jitter_cv.as_dict()

    step_offered = np.zeros_like(means)
    for qi, sl in ((URLLC, "urllc"), (EMBB, "embb"), (BE, "be")):
        cv = float(jitter_cv.get(sl, 0.0))
        if cv <= 0.0:
            step_offered[qi, :] = means[qi, :]
            continue
        # Draw a unit-mean multiplier so a ramp's per-step mean is preserved along the ramp.
        mult = _lognormal_mean_preserving(rng, 1.0, cv, n_steps)
        step_offered[qi, :] = means[qi, :] * mult

    step_offered = np.maximum(step_offered, 0.0)

    offered = np.repeat(step_offered, substeps_per_step, axis=1)
    phase_id = np.repeat(phase_ids_per_step, substeps_per_step)

    return Trace(
        scenario=str(scenario.name),
        seed=int(seed),
        substep_s=substep_s,
        control_interval_s=interval_s,
        substeps_per_step=substeps_per_step,
        offered_bps=offered,
        phase_id=phase_id,
        step_offered_bps=step_offered,
    )
