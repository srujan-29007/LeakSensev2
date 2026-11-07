"""Versioned run configs (``configs/*.yaml``): which scenarios, and how features are built."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

SLOT_METHODS = ("tod", "tod_weekday")
ESTIMATORS = ("mean", "median")


@dataclass(frozen=True)
class FeatureConfig:
    slot: str = "tod"  # tod (48 slots) | tod_weekday (336 slots)
    baseline_days: float | None = 14  # most recent N leak-free days nearest the case; None = all
    estimator: str = "mean"  # per-slot mean | median of the baseline rows
    window: int = 48  # steps per window (30 min each)
    stride: int = 24
    max_lag_days: float | None = None  # windows only within D days of the baseline; None = any
    max_windows_per_case: int = 32
    pre_guard_steps: int = 0  # leak-free steps skipped right before the case (pre baselines)
    snr_holdout_days: float = 3.0  # baseline days held out to estimate leak-free noise (SNR)

    def __post_init__(self) -> None:
        if self.slot not in SLOT_METHODS:
            raise ValueError(f"slot must be one of {SLOT_METHODS}, not {self.slot!r}")
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, not {self.estimator!r}")
        if self.window < 1 or self.stride < 1 or self.max_windows_per_case < 1:
            raise ValueError("window, stride and max_windows_per_case must be >= 1")


@dataclass(frozen=True)
class RunConfig:
    name: str = "default"
    seed: int = 0
    n_scenarios: int | None = None  # random subset of case scenarios; None = all
    include_scenarios: tuple[int, ...] = ()  # always in the subset
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Short hash of the feature-relevant settings (names the feature folder)."""
        d = self.to_dict()
        d.pop("name")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def select_scenarios(self, available: list[int]) -> list[int]:
        """Deterministic subset of ``available`` case scenarios."""
        avail = sorted(available)
        if self.n_scenarios is None or self.n_scenarios >= len(avail):
            return avail
        keep = [s for s in self.include_scenarios if s in avail]
        rest = [s for s in avail if s not in keep]
        rng = np.random.default_rng(self.seed)
        extra = rng.choice(rest, size=max(0, self.n_scenarios - len(keep)), replace=False)
        return sorted(keep + [int(s) for s in extra])


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_run_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    feats = _build(FeatureConfig, raw.pop("features", {}) or {})
    if "include_scenarios" in raw:
        raw["include_scenarios"] = tuple(raw["include_scenarios"])
    return _build(RunConfig, {**raw, "features": feats})
