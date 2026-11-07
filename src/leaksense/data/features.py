"""Pressure cache, leak-free baselines, windowed ΔH (PLAN.md M4).

No-oracle rule: the only dataset file read for features is ``Node_pressures.csv``, and only its
junction columns ``2``..``32`` (never ``leak_node*``, never ``Node_demands.csv`` or
``Link_flows.csv``). The scenario ``Labels.csv`` is read only to *assert* that baseline rows
are leak-free.

No-leakage rule for baselines: the baseline profile is computed by :func:`slot_profile` from
``p[B]`` alone, where ``B`` comes from the case's M3 ``baseline_segments``. :func:`baseline_rows`
asserts that ``B`` lies inside those segments, never touches ``[start, end]``, lies strictly
before ``start - pre_guard_steps`` for pre baselines, and is all-zero in ``Labels.csv``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.data.contract import DataContract
from leaksense.runconfig import FeatureConfig

STEPS_PER_DAY = 48
FORBIDDEN_FILES = ("Node_demands.csv", "Link_flows.csv")


class LeakageError(AssertionError):
    """A baseline would use a leak-active, case or post-start row it must not use."""


# ----------------------------------------------------------------------------- pressures


def load_pressures(scenario_dir: Path, contract: DataContract) -> np.ndarray:
    """Junction pressures (n_rows x 31, float32) from ``Node_pressures.csv``; nothing else."""
    path = Path(scenario_dir) / "Node_pressures.csv"
    wanted = set(contract.junctions)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, dtype=np.float32, engine="c")
    cols = list(df.columns)
    if cols != list(contract.junctions) or any(c.startswith("leak") for c in cols):
        raise ValueError(f"{path}: junction columns {cols[:3]}... != contract")
    if len(df) != contract.n_rows:
        raise ValueError(f"{path}: {len(df)} rows, expected {contract.n_rows}")
    return df.to_numpy(dtype=np.float32)


def load_labels(scenario_dir: Path, contract: DataContract) -> np.ndarray:
    df = pd.read_csv(Path(scenario_dir) / "Labels.csv", usecols=["Label"])
    if len(df) != contract.n_rows:
        raise ValueError(f"{scenario_dir}/Labels.csv: {len(df)} rows")
    return df["Label"].to_numpy() != 0


class PressureCache:
    """One ``.npz`` per scenario under ``<artifacts>/pressure/<manifest[:12]>/``."""

    def __init__(
        self,
        data_dir: Path,
        artifacts_dir: Path,
        manifest_sha256: str,
        contract: DataContract | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.contract = contract or DataContract()
        self.root = Path(artifacts_dir) / "pressure" / manifest_sha256[:12]

    def path(self, sid: int) -> Path:
        return self.root / f"Scenario-{sid}.npz"

    def get(self, sid: int) -> np.ndarray:
        p = self.path(sid)
        if p.is_file():
            with np.load(p) as z:
                arr = z["p"]
            if arr.shape == (self.contract.n_rows, len(self.contract.junctions)):
                return arr
        arr = load_pressures(self.data_dir / f"Scenario-{sid}", self.contract)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, p=arr)
        tmp.replace(p)
        return arr

    def labels(self, sid: int) -> np.ndarray:
        return load_labels(self.data_dir / f"Scenario-{sid}", self.contract)


# ----------------------------------------------------------------------------- slots


def slot_index(contract: DataContract, method: str) -> np.ndarray:
    """Slot of every row: time of day (0..47), or weekday * 48 + time of day (0..335)."""
    idx = pd.date_range(
        contract.first_timestamp, periods=contract.n_rows, freq=f"{contract.step_minutes}min"
    )
    per_day = 24 * 60 // contract.step_minutes
    tod = (idx.hour * 60 + idx.minute) // contract.step_minutes
    if method == "tod":
        return np.asarray(tod, dtype=np.int64)
    if method == "tod_weekday":
        return np.asarray(idx.day_of_week * per_day + tod, dtype=np.int64)
    raise ValueError(method)


# ----------------------------------------------------------------------------- baseline


def _rows_from_segments(segments) -> np.ndarray:
    parts = [np.arange(int(a), int(b) + 1) for a, b in segments]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def baseline_rows(case, n_days: float | None, guard: int, labels: np.ndarray) -> np.ndarray:
    """Baseline rows ``B`` for a case: the ``n_days`` leak-free days nearest the case window.

    pre -> the most recent days before start; post -> the earliest days after end; combined ->
    the nearest on either side. Raises :class:`LeakageError` if any invariant fails.
    """
    start, end, kind = int(case["start"]), int(case["end"]), case["baseline_kind"]
    seg_rows = _rows_from_segments(case["baseline_segments"])
    rows = seg_rows
    if kind == "pre" and guard:
        rows = rows[rows < start - guard]
    dist = np.where(rows < start, start - rows, rows - end)
    if n_days is not None:
        n = int(round(n_days * STEPS_PER_DAY))
        rows = np.sort(rows[np.argsort(dist, kind="stable")[:n]])

    tag = case.get("case_id", "?")
    if not np.isin(rows, seg_rows).all():
        raise LeakageError(f"{tag}: baseline rows outside the M3 baseline segments")
    if ((rows >= start) & (rows <= end)).any():
        raise LeakageError(f"{tag}: baseline overlaps the case window")
    if kind == "pre" and len(rows) and rows.max() >= start - guard:
        raise LeakageError(f"{tag}: pre baseline reaches Leak Start")
    if kind == "post" and len(rows) and rows.min() <= end:
        raise LeakageError(f"{tag}: post baseline starts before the case ends")
    if labels[rows].any():
        raise LeakageError(f"{tag}: {int(labels[rows].sum())} baseline rows are leak-active")
    if len(rows) == 0:
        raise LeakageError(f"{tag}: empty baseline")
    return rows


def _agg(
    values: np.ndarray, slots: np.ndarray, n_slots: int, estimator: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot mean/median of ``values`` (rows x J) -> (n_slots x J, counts)."""
    counts = np.bincount(slots, minlength=n_slots)
    out = np.full((n_slots, values.shape[1]), np.nan, dtype=np.float64)
    if estimator == "mean":
        sums = np.zeros((n_slots, values.shape[1]))
        np.add.at(sums, slots, values)
        ok = counts > 0
        out[ok] = sums[ok] / counts[ok, None]
    else:
        order = np.argsort(slots, kind="stable")
        s_sorted, v_sorted = slots[order], values[order]
        bounds = np.searchsorted(s_sorted, np.arange(n_slots + 1))
        for k in np.flatnonzero(counts):
            out[k] = np.median(v_sorted[bounds[k] : bounds[k + 1]], axis=0)
    return out, counts


def slot_profile(
    p_b: np.ndarray,
    rows_b: np.ndarray,
    cfg_slot: str,
    estimator: str,
    slots_tod: np.ndarray,
    slots_wd: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Baseline profile from baseline rows only.

    ``p_b`` is ``p[B]`` (the caller slices it), ``rows_b`` is ``B``. Returns the profile indexed
    by slot and the number of weekday slots that fell back to time of day.
    """
    values = p_b.astype(np.float64)
    tod_prof, _ = _agg(values, slots_tod[rows_b], 48, estimator)
    if cfg_slot == "tod":
        return tod_prof, 0
    wd_prof, counts = _agg(values, slots_wd[rows_b], 336, estimator)
    empty = counts == 0
    wd_prof[empty] = tod_prof[np.arange(336)[empty] % 48]
    return wd_prof, int(empty.sum())


@dataclass
class CaseDeltaH:
    dh: np.ndarray  # (n_steps x 31) float32, rows start..end
    rows_b: np.ndarray
    n_fallback_slots: int


def case_delta_h(
    p: np.ndarray,
    case,
    cfg: FeatureConfig,
    labels: np.ndarray,
    slots_tod: np.ndarray,
    slots_wd: np.ndarray,
) -> CaseDeltaH:
    rows_b = baseline_rows(case, cfg.baseline_days, cfg.pre_guard_steps, labels)
    profile, n_fb = slot_profile(p[rows_b], rows_b, cfg.slot, cfg.estimator, slots_tod, slots_wd)
    start, end = int(case["start"]), int(case["end"])
    slots = (slots_tod if cfg.slot == "tod" else slots_wd)[start : end + 1]
    dh = (p[start : end + 1].astype(np.float64) - profile[slots]).astype(np.float32)
    return CaseDeltaH(dh, rows_b, n_fb)


# ----------------------------------------------------------------------------- windows


@dataclass
class CaseWindows:
    x: np.ndarray  # (n_windows x 31) window-mean ΔH
    starts: np.ndarray  # absolute row of each window's first step
    short: bool  # case shorter than one window


def window_starts(n_steps: int, kind: str, cfg: FeatureConfig) -> np.ndarray:
    """Offsets (within the case) of the windows to use, nearest the baseline first in priority.

    Windows lie fully inside the case. With ``max_lag_days`` = D, only windows within D days of
    the baseline side are eligible (pre: start of the case, post: end, combined: both ends).
    More eligible windows than ``max_windows_per_case`` are thinned evenly.
    """
    W = cfg.window
    if n_steps <= W:
        return np.array([0])
    grid = np.arange(0, n_steps - W + 1, cfg.stride)
    from_start = grid  # first window starts at the case start
    from_end = np.sort((n_steps - W) - grid)  # last window ends at the case end
    if cfg.max_lag_days is None:
        offs = from_end if kind == "post" else from_start
    else:
        d = max(int(round(cfg.max_lag_days * STEPS_PER_DAY)), W)
        near_start = from_start[from_start + W <= d]
        near_end = from_end[from_end >= n_steps - d]
        offs = {"pre": near_start, "post": near_end, "combined": np.union1d(near_start, near_end)}[
            kind
        ]
    if len(offs) > cfg.max_windows_per_case:
        pick = np.linspace(0, len(offs) - 1, cfg.max_windows_per_case).round().astype(int)
        offs = offs[np.unique(pick)]
    return offs


def case_windows(dh: np.ndarray, case, cfg: FeatureConfig) -> CaseWindows:
    n = dh.shape[0]
    offs = window_starts(n, case["baseline_kind"], cfg)
    W = min(cfg.window, n)
    x = np.stack([dh[o : o + W].mean(axis=0) for o in offs]).astype(np.float32)
    return CaseWindows(x, int(case["start"]) + offs, n < cfg.window)


def holdout_noise(
    p: np.ndarray, rows_b: np.ndarray, cfg: FeatureConfig, slots_tod, slots_wd, near_case: str
) -> float:
    """Leak-free window noise (RMS over junctions and windows) from the baseline rows alone.

    The ``snr_holdout_days`` of ``B`` nearest the case are held out; the profile is built from
    the rest and the held-out window means are its residuals.
    """
    n_hold = int(round(cfg.snr_holdout_days * STEPS_PER_DAY))
    rows = np.sort(rows_b)
    if len(rows) < n_hold + 2 * STEPS_PER_DAY:
        return float("nan")
    hold, fit = (
        (rows[-n_hold:], rows[:-n_hold]) if near_case == "end" else (rows[:n_hold], rows[n_hold:])
    )
    prof, _ = slot_profile(p[fit], fit, cfg.slot, cfg.estimator, slots_tod, slots_wd)
    slots = slots_tod if cfg.slot == "tod" else slots_wd
    res = p[hold].astype(np.float64) - prof[slots[hold]]
    W = min(cfg.window, len(hold))
    means = [res[o : o + W].mean(axis=0) for o in range(0, len(hold) - W + 1, max(1, W // 2))]
    return float(np.sqrt(np.mean(np.square(means))))


# ----------------------------------------------------------------------------- build


@dataclass
class FeatureSet:
    x: np.ndarray  # (total windows x 31)
    case_of_window: np.ndarray  # index into ``cases``
    window_start: np.ndarray
    weight: np.ndarray  # 1 / windows-in-case, so every case counts equally
    cases: pd.DataFrame  # case_id, scenario_id, pipe, ..., n_windows, short_window, snr, ...


def build_features(
    cases: pd.DataFrame, cache: PressureCache, cfg: FeatureConfig, progress: bool = False
) -> FeatureSet:
    c = cache.contract
    slots_tod, slots_wd = slot_index(c, "tod"), slot_index(c, "tod_weekday")
    xs, cow, starts, weights, stats = [], [], [], [], []
    cases = cases.reset_index(drop=True).sort_values("scenario_id").reset_index(drop=True)
    for i, case in cases.iterrows():
        if progress and (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(cases)}", flush=True)
        sid = int(case["scenario_id"])
        p, labels = cache.get(sid), cache.labels(sid)
        cd = case_delta_h(p, case, cfg, labels, slots_tod, slots_wd)
        win = case_windows(cd.dh, case, cfg)
        near = "end" if case["baseline_kind"] == "pre" else "start"
        noise = holdout_noise(p, cd.rows_b, cfg, slots_tod, slots_wd, near)
        mean_dh = win.x.mean(axis=0)
        n = len(win.x)
        xs.append(win.x)
        cow.append(np.full(n, i))
        starts.append(win.starts)
        weights.append(np.full(n, 1.0 / n))
        stats.append(
            {
                "n_windows": n,
                "short_window": win.short,
                "n_baseline_rows": len(cd.rows_b),
                "n_fallback_slots": cd.n_fallback_slots,
                "max_abs_mean_dh": float(np.abs(mean_dh).max()),
                "argmax_junction": c.junctions[int(np.abs(mean_dh).argmax())],
                "noise_m": noise,
                "snr": float(np.abs(mean_dh).max() / noise) if noise and noise > 0 else np.nan,
            }
        )
    out = pd.concat([cases, pd.DataFrame(stats)], axis=1)
    return FeatureSet(
        x=np.concatenate(xs).astype(np.float32),
        case_of_window=np.concatenate(cow).astype(np.int64),
        window_start=np.concatenate(starts).astype(np.int64),
        weight=np.concatenate(weights).astype(np.float32),
        cases=out,
    )


def save_features(fs: FeatureSet, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "features.npz",
        x=fs.x,
        case_of_window=fs.case_of_window,
        window_start=fs.window_start,
        weight=fs.weight,
    )
    cols = [c for c in fs.cases.columns if c != "baseline_segments"]
    fs.cases[cols].to_parquet(out_dir / "case_stats.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def features_hash(fs: FeatureSet) -> str:
    return hashlib.sha256(fs.x.tobytes()).hexdigest()[:12]
