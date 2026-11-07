"""Masked gradient-boosting rung (PLAN.md M8), LightGBM multiclass.

Same input, weights and protocol as the masked MLP (``leaksense.models.mlp``):

- Input per window: ``[T(ΔH·m), m]``. A tree model cannot draw a fresh mask per epoch, so each
  training window is repeated ``n_masks`` times with independent random masks (k ~ U{k_min..31}).
- Sample weight = (1 / windows in its case) × class weight (training fold only).
- Settings given as lists are chosen by weighted multi-log-loss on the inner validation split
  (grouped by scenario); the number of boosting rounds comes from early stopping there, and the
  model is refitted on the whole training set with that many rounds.
- A case's score per pipe is the mean over its windows of the log-probabilities.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, fields, replace

import lightgbm as lgb
import numpy as np
import pandas as pd

from leaksense import transforms
from leaksense.models.mlp import (
    CLASS_WEIGHTS,
    _Data,
    _seed_for,
    _windows,
    class_weights,
    encode,
    inner_split,
    random_k_masks,
)
from leaksense.network import Network


@dataclass(frozen=True)
class GBMConfig:
    transform: str = "signed_log1p"
    n_masks: int = 8  # masked copies of each training window
    k_min: int = 1
    num_leaves: int = 15
    learning_rate: float = 0.05
    max_rounds: int = 1000
    early_stopping: int = 50
    min_child_samples: int = 20
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    class_weight: str = "inverse"
    val_frac: float = 0.15
    val_masks: int = 4
    refit: bool = True
    num_threads: int = 0  # 0 = LightGBM default
    seed: int = 0

    def __post_init__(self) -> None:
        if self.transform not in transforms.TRANSFORMS:
            raise ValueError(f"transform must be one of {transforms.TRANSFORMS}")
        if self.class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}")
        if self.n_masks < 1 or self.k_min < 1 or self.max_rounds < 1:
            raise ValueError("n_masks, k_min and max_rounds must be >= 1")


def parse_config(d: dict) -> tuple[GBMConfig, dict[str, list]]:
    d = {k: v for k, v in d.items() if k != "kind"}
    unknown = set(d) - {f.name for f in fields(GBMConfig)}
    if unknown:
        raise ValueError(f"unknown gbm settings: {sorted(unknown)}")
    grid = {k: list(v) for k, v in d.items() if isinstance(v, list)}
    base = GBMConfig(**{k: (v[0] if isinstance(v, list) else v) for k, v in d.items()})
    for k, vals in grid.items():
        for v in vals:
            replace(base, **{k: v})
    return base, grid


def _masked_rows(
    d: _Data, copies: int, k_min: int, transform: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j = d.x.shape[1]
    x = np.tile(d.x, (copies, 1))
    m = random_k_masks(len(x), j, k_min, rng)
    return encode(x, m, transform).astype(np.float32), np.tile(d.y, copies), np.tile(d.w, copies)


def _params(cfg: GBMConfig, n_classes: int, seed: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1 if cfg.bagging_fraction < 1 else 0,
        "lambda_l2": cfg.lambda_l2,
        "num_threads": cfg.num_threads,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def _train(
    cfg: GBMConfig, tr: _Data, n_classes: int, seed: int, rounds: int, val: _Data | None = None
) -> tuple[lgb.Booster, int, float]:
    rng = np.random.default_rng(seed)
    x, y, w = _masked_rows(tr, cfg.n_masks, cfg.k_min, cfg.transform, rng)
    dtr = lgb.Dataset(x, y, weight=w, free_raw_data=True)
    params = _params(cfg, n_classes, seed)
    if val is None:
        return lgb.train(params, dtr, num_boost_round=rounds), rounds, float("nan")
    vx, vy, vw = _masked_rows(val, cfg.val_masks, 1, cfg.transform, np.random.default_rng(seed + 1))
    dva = lgb.Dataset(vx, vy, weight=vw, reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)],
    )
    best = booster.best_iteration or rounds
    return booster, best, float(booster.best_score["valid_0"]["multi_logloss"])


class MaskedGBM:
    """LightGBM over ``[T(ΔH·m), m]``; see the module docstring."""

    name = "gbm"

    def __init__(self, net: Network, model_cfg: dict | None = None) -> None:
        self.n = len(net.leak_pipe_ids)
        self.base, self.grid = parse_config(model_cfg or {})
        self.cfg = self.base
        self.booster: lgb.Booster | None = None
        self.rounds = 0
        self.history: list[dict] = []

    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None:
        seed = _seed_for(train, self.base.seed) % (2**31 - 1)
        is_val = inner_split(train, self.base.val_frac, seed)
        cand = [
            replace(self.base, **dict(zip(self.grid, vals, strict=True)))
            for vals in itertools.product(*self.grid.values())
        ] or [self.base]

        trials = []
        if is_val.any():
            tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
            tr_cases, va_cases = train.iloc[tr_idx], train.iloc[va_idx]
            for cfg in cand:
                cw = class_weights(tr_cases["y"].to_numpy(), self.n, cfg.class_weight)
                tr = _windows(tr_cases, [windows[i] for i in tr_idx], cw)
                va = _windows(va_cases, [windows[i] for i in va_idx], cw)
                b, rounds, vloss = _train(cfg, tr, self.n, seed, cfg.max_rounds, va)
                trials.append({"cfg": cfg, "booster": b, "rounds": rounds, "val_loss": vloss})
            pick = min(trials, key=lambda t: t["val_loss"])
        else:  # too few cases for an inner split: train the base config for max_rounds
            pick = {"cfg": cand[0], "booster": None, "rounds": cand[0].max_rounds, "val_loss": None}

        cfg = pick["cfg"]
        if cfg.refit or pick["booster"] is None:
            cw = class_weights(train["y"].to_numpy(), self.n, cfg.class_weight)
            full = _windows(train, windows, cw)
            self.booster, _, _ = _train(cfg, full, self.n, seed, max(1, pick["rounds"]))
        else:
            self.booster = pick["booster"]
        self.cfg, self.rounds = cfg, int(pick["rounds"])
        self.history.append(
            {
                "n_train_cases": len(train),
                "n_inner_val_cases": int(is_val.sum()),
                "chosen": {k: getattr(cfg, k) for k in self.grid},
                "epochs": self.rounds,  # boosting rounds
                "val_loss": pick["val_loss"],
                "trials": [
                    {
                        **{k: getattr(t["cfg"], k) for k in self.grid},
                        "rounds": t["rounds"],
                        "val_loss": round(t["val_loss"], 4),
                    }
                    for t in trials
                ],
            }
        )

    def batch_scores(self, xs: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("MaskedGBM.fit was not called")
        lens = np.array([len(x) for x in xs])
        x = np.concatenate(xs).astype(np.float32)
        m = np.broadcast_to(np.asarray(mask, bool), x.shape)
        p = self.booster.predict(encode(x, m, self.cfg.transform), num_iteration=self.rounds)
        lp = np.log(np.clip(p, 1e-12, None))
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        return np.add.reduceat(lp, starts, axis=0) / lens[:, None]

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.batch_scores([x], mask)[0]

    def describe(self) -> dict:
        return {"base": asdict(self.base), "grid": self.grid, "folds": self.history}
