"""Policy-wrapper tests: seeding, freezing, and the pre-trained evaluation path.

The two that guard real methodology rather than mechanics:

  * test_pretrained_model_does_not_change_during_evaluation - if the converged run kept learning,
    "converged" would describe a model that was still moving throughout the run being reported,
    and the online/converged distinction docs/EXPERIMENTS.md commits to would be fictional.
  * test_training_seeds_are_disjoint_from_evaluation_seeds - a config edit that let them overlap
    would make every converged number a memorisation result, and nothing would fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.policies.bandit_policy import EpsilonGreedyPolicy, LinUCBPolicy, policy_seed
from agent.policies.base import all_allowed
from config_loader import load_config
from experiments.run_experiment import PRETRAINED, pretrain, run_once
from traffic.traces import trace_seed


# --------------------------------------------------------------------------- seeding


def test_policy_seed_is_process_independent():
    """Same input, same stream, across processes. Python's hash() would not give this."""
    assert policy_seed(3, "epsilon_greedy") == policy_seed(3, "epsilon_greedy")
    assert policy_seed(3, "epsilon_greedy") != policy_seed(4, "epsilon_greedy")
    assert policy_seed(3, "epsilon_greedy") != policy_seed(3, "linucb")


def test_policy_stream_differs_from_the_traffic_stream(cfg):
    """The exploration RNG must not be initialised identically to the traffic RNG."""
    for seed in range(8):
        assert policy_seed(seed, "epsilon_greedy") != trace_seed(cfg, seed)


def test_training_seeds_are_disjoint_from_evaluation_seeds():
    c = load_config()
    assert not (set(c.experiment.train_seeds) & set(c.experiment.seeds))


# --------------------------------------------------------------------------- masks


@pytest.mark.parametrize("cls", [LinUCBPolicy, EpsilonGreedyPolicy])
def test_never_returns_a_masked_arm(cls, cfg):
    from agent.context import Context

    policy = cls(cfg)
    policy.reset(0)
    rng = np.random.default_rng(0)
    n = policy.n_actions
    for step in range(500):
        allowed = rng.random(n) < 0.5
        if not allowed.any():
            allowed[rng.integers(0, n)] = True
        ctx = Context(
            vector=np.concatenate([[1.0], rng.random(8)]),
            rtt_ewma_ms=float(rng.uniform(0, 10)),
            rtt_p95_ms=float(rng.uniform(0, 10)),
            step_idx=step,
        )
        a = policy.select(ctx, allowed)
        assert allowed[a], f"{cls.__name__} returned masked arm {a}"
        policy.update(ctx, a, float(rng.standard_normal()))


# --------------------------------------------------------------------------- freezing


def test_freeze_turns_off_exploration(cfg):
    lin = LinUCBPolicy(cfg)
    lin.reset(0)
    lin.freeze()
    assert lin.frozen and lin.model.alpha == 0.0

    eps = EpsilonGreedyPolicy(cfg)
    eps.reset(0)
    eps.freeze()
    assert eps.frozen and eps.model.epsilon == 0.0


def test_reset_undoes_freeze(cfg):
    """A frozen policy reused for a fresh run must explore again, or run 2 is not run 1."""
    lin = LinUCBPolicy(cfg)
    lin.reset(0)
    lin.freeze()
    lin.reset(1)
    assert not lin.frozen
    assert lin.model.alpha == pytest.approx(float(cfg.policy.linucb.alpha))


# --------------------------------------------------------------------------- pre-training


def test_pretrain_actually_learns_and_then_freezes(cfg):
    policy = LinUCBPolicy(cfg)
    policy.reset(0)
    assert policy.model.counts.sum() == 0
    pretrain(policy, cfg=cfg, scenario="burst", seeds=[100, 101])
    assert policy.model.counts.sum() > 0
    assert policy.frozen


def test_pretrained_model_does_not_change_during_evaluation(cfg):
    """The evaluation run of a converged policy must not update the model."""
    policy = LinUCBPolicy(cfg)
    policy.reset(0)
    pretrain(policy, cfg=cfg, scenario="burst", seeds=[100])
    before_counts = policy.model.counts.copy()
    before_b = policy.model.b.copy()

    run_once(
        policy_name="linucb", scenario="burst", seed=0, cfg=cfg, write=False,
        policy=policy, learn=False,
    )
    assert np.array_equal(policy.model.counts, before_counts)
    assert np.array_equal(policy.model.b, before_b)


def test_pretrained_run_is_registered_and_differs_from_online(cfg, tmp_path):
    """The two variants must be distinguishable in the results, not just in intent."""
    assert set(PRETRAINED) == {"linucb_pretrained", "epsilon_greedy_pretrained"}
    _, online, _ = run_once("linucb", "burst", 0, cfg=cfg, out_dir=tmp_path)
    _, conv, _ = run_once("linucb_pretrained", "burst", 0, cfg=cfg, out_dir=tmp_path)
    assert (tmp_path / "linucb_burst_0.csv").exists()
    assert (tmp_path / "linucb_pretrained_burst_0.csv").exists()
    assert online.mean_reward != conv.mean_reward


def test_pretraining_writes_no_run_logs(cfg, tmp_path):
    """Training logs must never land where the aggregator would average them into results."""
    run_once("linucb_pretrained", "burst", 0, cfg=cfg, out_dir=tmp_path)
    written = sorted(p.stem for p in tmp_path.glob("*.csv"))
    assert written == ["linucb_pretrained_burst_0"]


# --------------------------------------------------------------------------- reproducibility


@pytest.mark.parametrize("policy_name", ["linucb", "epsilon_greedy"])
def test_same_seed_gives_the_same_run(policy_name, cfg, tmp_path):
    _, a, _ = run_once(policy_name, "burst", 0, cfg=cfg, write=False)
    _, b, _ = run_once(policy_name, "burst", 0, cfg=cfg, write=False)
    assert a.mean_reward == pytest.approx(b.mean_reward)
    assert a.sla_violation_rate == pytest.approx(b.sla_violation_rate)
