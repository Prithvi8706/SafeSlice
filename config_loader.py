"""Configuration loading.

Deviation from the layout in the brief: the brief lists config/ as a data directory only and
gives no home for the loader. Rather than bury it in net/backend.py (which is specified as
"NetworkBackend ABC, TelemetrySample, Allocation") this lives at the repo root as its own
module. It is the only file outside the specified tree.

A Cfg is a thin attribute-access wrapper over nested dicts. Attribute access is used because
config keys appear constantly in the hot loop and cfg.link.capacity_bps reads better than
cfg["link"]["capacity_bps"]. Missing keys raise AttributeError with the full path, so typos
fail loudly instead of silently returning None.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
SCENARIO_DIR = REPO_ROOT / "config" / "scenarios"


class Cfg:
    """Read-only attribute access over a nested dict."""

    __slots__ = ("_d", "_path")

    def __init__(self, d: Dict[str, Any], path: str = ""):
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            path = object.__getattribute__(self, "_path")
            full = f"{path}.{name}" if path else name
            raise AttributeError(f"config key not found: {full} (available: {sorted(d)})")
        value = d[name]
        if isinstance(value, dict):
            path = object.__getattribute__(self, "_path")
            return Cfg(value, f"{path}.{name}" if path else name)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("Cfg is read-only; edit the YAML or use with_overrides()")

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_d")

    def get(self, name: str, default: Any = None) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            return default
        return getattr(self, name)

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(object.__getattribute__(self, "_d"))

    def keys(self):
        return object.__getattribute__(self, "_d").keys()

    def __repr__(self) -> str:
        return f"Cfg({object.__getattribute__(self, '_d')!r})"


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn {"reward.w_sla": 2.0} into {"reward": {"w_sla": 2.0}}."""
    nested: Dict[str, Any] = {}
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def load_config(
    scenario: Optional[str] = None,
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Cfg:
    """Load default.yaml, merge a scenario file over it, then apply dotted overrides.

    `scenario` is a bare name such as "burst", resolved against config/scenarios/.
    `overrides` uses dotted keys, e.g. {"run.duration_s": 60, "reward.w_sla": 4.0}.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        merged = yaml.safe_load(fh)
    if not isinstance(merged, dict):
        raise ValueError(f"{path} did not parse to a mapping")

    if scenario is not None:
        scenario_path = SCENARIO_DIR / f"{scenario}.yaml"
        if not scenario_path.exists():
            available = sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
            raise FileNotFoundError(
                f"no scenario {scenario!r} at {scenario_path} (available: {available})"
            )
        with open(scenario_path, "r", encoding="utf-8") as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh))

    if overrides:
        merged = _deep_merge(merged, _parse_overrides(overrides))

    return Cfg(merged)


def config_hash(cfg: Cfg) -> str:
    """Stable short hash of the fully merged config, recorded in env.json per run."""
    blob = json.dumps(cfg.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def list_scenarios():
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
