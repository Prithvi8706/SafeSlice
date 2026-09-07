"""Shared test fixtures and the marker that keeps root-only tests out of the default run.

Anything that needs Mininet, Open vSwitch or root is marked @pytest.mark.requires_mininet and
skipped unless --run-mininet is passed. That means `pytest` is safe to run on a laptop with no
networking stack, and the same suite gains the real tests on the VM without editing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_loader import load_config  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--run-mininet",
        action="store_true",
        default=False,
        help="also run tests that need Mininet, Open vSwitch and root",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_mininet: needs a real Mininet/OVS testbed and root privileges"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-mininet"):
        return
    skip = pytest.mark.skip(reason="needs Mininet/OVS/root; pass --run-mininet to enable")
    for item in items:
        if "requires_mininet" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cfg():
    """Default config with the burst scenario and a shortened run, for fast tests.

    90 s, not less. burst.yaml phase 0 is 40 s long and phase 1 is 35 s, so a run shorter than
    75 s never reaches the eMBB burst at all. An earlier 40 s fixture made four tests fail with
    "the scenario is too easy", which looked like a simulator bug and was really the fixture
    never leaving the idle phase.
    """
    return load_config(
        scenario="burst",
        overrides={"run.duration_s": 90, "run.warmup_s": 10},
    )


@pytest.fixture
def cfg_full():
    """Default config with the burst scenario at the real run length."""
    return load_config(scenario="burst")
