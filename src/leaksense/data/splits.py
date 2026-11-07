"""``leaksense make-splits``: locked test fold plus grouped, pipe-stratified 5-fold CV.

Groups are scenario ids (one case per scenario today; detection in M15 adds several windows per
scenario, so the key stays). The locked test fold is touched only in M13.

Rare pipes (fewer than ``rare_threshold`` cases) cannot be in the test fold and in every CV
fold. Rule: a pipe gets test cases only if at least ``min_train_per_pipe`` of its cases remain
for training; otherwise its test cases are moved back to the pool.

Leak type and case kind are balanced "where possible": among ``n_seed_trials`` candidate seeds
(derived from the base seed), the split with the smallest worst-fold deviation of the small-leak
and solo shares is kept. Everything is deterministic for a given seed.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 20261108
    n_folds: int = 5
    test_fraction: float = 0.15
    rare_threshold: int = 6
    min_train_per_pipe: int = 2
    n_seed_trials: int = 50


class SplitError(AssertionError):
    pass


def _sgkf_assign(df: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """Fold number per row from StratifiedGroupKFold(y=pipe, groups=scenario)."""
    fold = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():  # pipes with fewer members than folds are expected
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, test_idx) in enumerate(sgkf.split(df, df["pipe"], df["scenario_id"])):
            fold[test_idx] = k
    return fold


def _imbalance(df: pd.DataFrame, fold: np.ndarray) -> float:
    small = (df["leak_type"] == "small").to_numpy()
    solo = (df["kind"] == "solo").to_numpy()
    worst = 0.0
    for k in np.unique(fold):
        m = fold == k
        worst = max(worst, abs(small[m].mean() - small.mean()), abs(solo[m].mean() - solo.mean()))
    return worst


def _one_split(df: pd.DataFrame, cfg: SplitConfig, seed: int) -> tuple[np.ndarray, float]:
    """-1 = test, 0..n_folds-1 = CV fold."""
    n_test_folds = max(2, round(1 / cfg.test_fraction))
    assign = _sgkf_assign(df, n_test_folds, seed)
    is_test = assign == 0
    # Rare-pipe rule: keep enough training cases for every pipe.
    for _pipe, idx in df.groupby("pipe").indices.items():
        if (~is_test[idx]).sum() < cfg.min_train_per_pipe:
            is_test[idx] = False
    fold = np.full(len(df), -1)
    rest = np.flatnonzero(~is_test)
    fold[rest] = _sgkf_assign(df.iloc[rest].reset_index(drop=True), cfg.n_folds, seed + 1)
    return fold, _imbalance(df, fold)


def make_splits(cases: pd.DataFrame, cfg: SplitConfig | None = None) -> dict:
    cfg = cfg or SplitConfig()
    df = cases.sort_values("scenario_id").reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    seeds = rng.integers(0, 2**31 - 1, size=cfg.n_seed_trials)
    best = min((_one_split(df, cfg, int(s)) + (int(s),) for s in seeds), key=lambda t: t[1])
    fold, imbalance, chosen = best

    def ids(mask) -> list[int]:
        return sorted(int(x) for x in df.loc[mask, "scenario_id"].unique())

    out = {
        "version": 1,
        "config": asdict(cfg),
        "chosen_seed": chosen,
        "worst_fold_imbalance": round(float(imbalance), 4),
        "test": ids(fold == -1),
        "folds": [ids(fold == k) for k in range(cfg.n_folds)],
    }
    counts = df.groupby("pipe").size()
    out["rare_pipes"] = {
        p: {
            "n_cases": int(n),
            "n_test": int(((df["pipe"] == p) & (fold == -1)).sum()),
            "n_folds_present": int(len(set(fold[(df["pipe"] == p).to_numpy()]) - {-1})),
        }
        for p, n in sorted(counts.items(), key=lambda kv: int(kv[0]))
        if n < cfg.rare_threshold
    }
    out["balance"] = balance_table(df, fold)
    check_splits(out, df)
    return out


def balance_table(df: pd.DataFrame, fold: np.ndarray) -> list[dict]:
    rows = []
    for k in [-1, *range(int(fold.max()) + 1)]:
        m = fold == k
        sub = df[m]
        rows.append(
            {
                "fold": "test" if k == -1 else str(k),
                "n_cases": int(m.sum()),
                "n_pipes": int(sub["pipe"].nunique()),
                "small": int((sub["leak_type"] == "small").sum()),
                "big": int((sub["leak_type"] == "big").sum()),
                "pure": int((sub["kind"] == "pure").sum()),
                "solo": int((sub["kind"] == "solo").sum()),
            }
        )
    return rows


def check_splits(splits: dict, cases: pd.DataFrame) -> None:
    """Raise SplitError if scenarios overlap, are missing, or the rare-pipe rule fails."""
    parts = [set(splits["test"]), *map(set, splits["folds"])]
    seen: set[int] = set()
    for p in parts:
        if seen & p:
            raise SplitError(f"scenario(s) in more than one split: {sorted(seen & p)[:10]}")
        seen |= p
    expected = set(int(s) for s in cases["scenario_id"])
    if seen != expected:
        raise SplitError(
            f"{len(expected - seen)} scenarios unassigned, {len(seen - expected)} unknown"
        )
    test = set(splits["test"])
    min_train = splits["config"]["min_train_per_pipe"]
    for pipe, g in cases.groupby("pipe"):
        n_train = int((~g["scenario_id"].isin(test)).sum())
        if n_train < min(min_train, len(g)):
            raise SplitError(f"pipe {pipe}: only {n_train} training cases")


def fold_of(splits: dict) -> dict[int, str]:
    """scenario id -> 'test' or the CV fold number as a string."""
    out = {s: "test" for s in splits["test"]}
    for k, ids in enumerate(splits["folds"]):
        out.update({s: str(k) for s in ids})
    return out


def non_test_scenarios(splits: dict) -> set[int]:
    return {s for ids in splits["folds"] for s in ids}
