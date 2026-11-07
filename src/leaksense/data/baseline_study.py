"""``leaksense baseline-study``: choose slot matching, baseline, window and lag by measurement.

Pseudo-cases are built from leak-free rows only: Scenario-1 (no leak, in no fold) and the
pre-leak stretch of non-test case scenarios. A pseudo-start ``s`` splits a stretch into a
baseline (the N days before ``s``) and an evaluation part (after ``s``, still before the real
leak). The true ΔH there is 0, so whatever remains is baseline error: noise plus drift.

Locked-test scenarios are never loaded (asserted). Label-dependent checks (signal size, the
start-1 check) use non-test cases only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from leaksense.data.features import (
    STEPS_PER_DAY,
    PressureCache,
    baseline_rows,
    case_delta_h,
    case_windows,
    slot_index,
    slot_profile,
)
from leaksense.runconfig import FeatureConfig

WINDOWS = (1, 6, 12, 24, 48, 96, 336)
BASELINE_DAYS = (7, 14, 28, None)
EVAL_DAYS = 7
SHORT_HISTORY_DAYS = 21  # pseudo-cases standing in for the 29 cases with 14-28 days of history
LAG_BINS_DAYS = (0, 7, 14, 30, 60, 90, 180, 300)
NOISE_TARGET_M = 0.2
TIE_TOLERANCE = 0.05


@dataclass(frozen=True)
class Method:
    slot: str
    baseline_days: float | None
    estimator: str

    @property
    def label(self) -> str:
        n = "all" if self.baseline_days is None else f"{self.baseline_days:g}d"
        return f"{self.slot}/{n}/{self.estimator}"

    @property
    def complexity(self) -> tuple:
        # Simpler first: tod before tod_weekday, mean before median.
        return (self.slot != "tod", self.estimator != "mean")


METHODS = [
    Method(s, n, e)
    for s, n, e in product(("tod", "tod_weekday"), BASELINE_DAYS, ("mean", "median"))
]


@dataclass
class Stretch:
    scenario_id: int
    end: int  # exclusive: the stretch is rows [0, end)


def leak_free_stretches(
    cases: pd.DataFrame, scenario_ids: list[int], cache: PressureCache, test_ids: set[int]
) -> list[Stretch]:
    """[0, first leak start) for each case scenario, plus Scenario-1 in full."""
    out = []
    starts = cases.groupby("scenario_id")["start"].min()
    for sid in scenario_ids:
        if sid in test_ids:
            raise AssertionError(f"baseline study must not use locked-test Scenario-{sid}")
        end = cache.contract.n_rows if sid == 1 else int(starts[sid])
        labels = cache.labels(sid)
        if labels[:end].any():
            raise AssertionError(f"Scenario-{sid}: leak-active rows before the first leak")
        out.append(Stretch(sid, end))
    return out


def _residual(p, s, e, method: Method, slots_tod, slots_wd) -> np.ndarray:
    """ΔH on rows [s, e) with a baseline from the N days before s (only rows < s are used)."""
    lo = 0 if method.baseline_days is None else max(0, s - int(method.baseline_days * 48))
    rows_b = np.arange(lo, s)
    assert rows_b.max() < s
    prof, _ = slot_profile(p[rows_b], rows_b, method.slot, method.estimator, slots_tod, slots_wd)
    slots = (slots_tod if method.slot == "tod" else slots_wd)[s:e]
    return p[s:e].astype(np.float64) - prof[slots]


def _window_means(res: np.ndarray, W: int) -> np.ndarray:
    stride = max(1, W // 2)
    offs = np.arange(0, len(res) - W + 1, stride)
    c = np.vstack([np.zeros((1, res.shape[1])), np.cumsum(res, axis=0)])
    return (c[offs + W] - c[offs]) / W


def method_comparison(
    stretches: list[Stretch], cache: PressureCache, progress=False
) -> pd.DataFrame:
    """Per (stratum, method, W): median and p95 over (pseudo-case, junction) of window RMS."""
    c = cache.contract
    slots_tod, slots_wd = slot_index(c, "tod"), slot_index(c, "tod_weekday")
    E = EVAL_DAYS * STEPS_PER_DAY
    records = []
    for n, st in enumerate(stretches, 1):
        if progress and n % 50 == 0:
            print(f"  methods: {n}/{len(stretches)} stretches", flush=True)
        p = cache.get(st.scenario_id)
        pseudo = []
        short = SHORT_HISTORY_DAYS * STEPS_PER_DAY
        if st.end >= short + E:
            pseudo.append(("history 14-28d", short))
        if st.end >= 28 * STEPS_PER_DAY + E:
            if st.scenario_id == 1:  # a full leak-free year: several pseudo-starts
                pseudo += [(">28d", s) for s in range(60 * 48, st.end - E + 1, 30 * 48)]
            else:
                pseudo.append((">28d", st.end - E))  # baseline right before the leak, as in use
        for stratum, s in pseudo:
            for m in METHODS:
                res = _residual(p, s, s + E, m, slots_tod, slots_wd)
                for W in WINDOWS:
                    rms = np.sqrt(np.mean(np.square(_window_means(res, W)), axis=0))
                    records.append((stratum, m.label, W, st.scenario_id, rms))
    rows = []
    df = pd.DataFrame(records, columns=["stratum", "method", "W", "scenario_id", "rms"])
    for (stratum, method, W), g in df.groupby(["stratum", "method", "W"], sort=False):
        v = np.concatenate(g["rms"].to_list())
        rows.append(
            {
                "stratum": stratum,
                "method": method,
                "W": W,
                "n_pseudo": len(g),
                "median_rms_m": float(np.median(v)),
                "p95_rms_m": float(np.percentile(v, 95)),
            }
        )
    return pd.DataFrame(rows)


def lag_study(
    stretches: list[Stretch], cache: PressureCache, methods: list[Method], W: int
) -> pd.DataFrame:
    """Bias and RMS of window-mean ΔH by lag (days after the end of the baseline)."""
    c = cache.contract
    slots_tod, slots_wd = slot_index(c, "tod"), slot_index(c, "tod_weekday")
    start0 = 28 * STEPS_PER_DAY
    recs = []
    for st in stretches:
        if st.end < start0 + 7 * STEPS_PER_DAY:
            continue
        p = cache.get(st.scenario_id)
        starts = range(start0, st.end - 7 * 48, 30 * 48) if st.scenario_id == 1 else [start0]
        for s in starts:
            for m in methods:
                res = _residual(p, s, st.end, m, slots_tod, slots_wd)
                wm = _window_means(res, W)
                lag = (np.arange(len(wm)) * max(1, W // 2) + W / 2) / STEPS_PER_DAY
                b = np.digitize(lag, LAG_BINS_DAYS) - 1
                for k in np.unique(b):
                    if 0 <= k < len(LAG_BINS_DAYS) - 1:
                        sel = wm[b == k]
                        recs.append(
                            (m.label, k, np.abs(sel.mean(axis=0)), np.sqrt(np.mean(sel**2, axis=0)))
                        )
    rows = []
    df = pd.DataFrame(recs, columns=["method", "bin", "abs_bias", "rms"])
    for (method, k), g in df.groupby(["method", "bin"]):
        bias, rms = np.concatenate(g["abs_bias"].to_list()), np.concatenate(g["rms"].to_list())
        rows.append(
            {
                "method": method,
                "lag_days": f"{LAG_BINS_DAYS[k]}-{LAG_BINS_DAYS[k + 1]}",
                "lag_lo": LAG_BINS_DAYS[k],
                "lag_hi": LAG_BINS_DAYS[k + 1],
                "n": len(g),
                "median_abs_bias_m": float(np.median(bias)),
                "median_rms_m": float(np.median(rms)),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "lag_lo"]).reset_index(drop=True)


def choose(comp: pd.DataFrame) -> dict:
    """Decision rules from the module docstring of PLAN M4 (stated in the report)."""
    by_label = {m.label: m for m in METHODS}
    if comp.empty or ">28d" not in set(comp["stratum"]):
        raise ValueError(
            "baseline study: no pseudo-cases with more than 28 days of leak-free history; "
            f"need stretches of at least {28 + EVAL_DAYS} days before the first leak"
        )
    out = {}
    for stratum in comp["stratum"].unique():
        g = comp[(comp["stratum"] == stratum) & (comp["W"] == 48)]
        best = g["median_rms_m"].min()
        ok = g[g["median_rms_m"] <= best * (1 + TIE_TOLERANCE)]
        # Among near-ties: the simpler slot/estimator, then the lower noise.
        rms = dict(zip(ok["method"], ok["median_rms_m"], strict=True))
        pick = min(rms, key=lambda lab: (by_label[lab].complexity, rms[lab]))
        out[stratum] = pick
    long_m = by_label[out[">28d"]]
    g = comp[(comp["stratum"] == ">28d") & (comp["method"] == long_m.label)].sort_values("W")
    under = g[g["median_rms_m"] <= NOISE_TARGET_M]
    W = int(under["W"].iloc[0]) if len(under) else int(g["W"].iloc[-1])
    return {"method_by_stratum": out, "window": W, "stride": max(1, W // 2)}


def choose_lag(lag: pd.DataFrame, method_label: str) -> float | None:
    """Largest lag (days) up to which |bias| stays below the lag 0-7 d window RMS."""
    g = lag[lag["method"] == method_label].sort_values("lag_lo")
    if g.empty:
        return None
    noise = float(g["median_rms_m"].iloc[0])
    d = None
    for _, r in g.iterrows():
        if r["median_abs_bias_m"] < noise:
            d = float(r["lag_hi"])
        else:
            break
    return d


START_CHECK_GUARD = 4  # baseline ends at start-5, so start-4..start-1 are out of sample


def start_check(cases: pd.DataFrame, cache: PressureCache, cfg: FeatureConfig) -> dict:
    """Mean |ΔH| over junctions around Leak Start, for big pure cases with a pre baseline.

    The baseline stops ``START_CHECK_GUARD`` steps before start, so the steps start-4..start-1
    are evaluated out of sample, exactly like the case rows. If the leak signal began before
    Leak Start, start-1 would stand out from start-4..start-2.
    """
    c = cache.contract
    slots_tod, slots_wd = slot_index(c, "tod"), slot_index(c, "tod_weekday")
    sel = cases[
        (cases["kind"] == "pure")
        & (cases["leak_type"] == "big")
        & (cases["baseline_kind"] == "pre")
    ]
    g = START_CHECK_GUARD
    keys = [f"start{k:+d}" if k else "start" for k in range(-g, 2)]
    vals: dict[str, list[float]] = {k: [] for k in keys}
    slots = slots_tod if cfg.slot == "tod" else slots_wd
    for _, case in sel.iterrows():
        sid, s = int(case["scenario_id"]), int(case["start"])
        if int(case["n_steps"]) < 2:
            continue
        p, labels = cache.get(sid), cache.labels(sid)
        rows_b = baseline_rows(case, cfg.baseline_days, g, labels)
        prof, _ = slot_profile(p[rows_b], rows_b, cfg.slot, cfg.estimator, slots_tod, slots_wd)
        for key, k in zip(keys, range(-g, 2), strict=True):
            t = s + k
            vals[key].append(float(np.abs(p[t] - prof[slots[t]]).mean()))
    return {
        "n_cases": len(sel),
        "baseline_ends_at": f"start-{g + 1}",
        **{k: float(np.median(v)) if v else None for k, v in vals.items()},
    }


def signal_check(
    cases: pd.DataFrame, cache: PressureCache, cfg: FeatureConfig, noise_m: float
) -> pd.DataFrame:
    """Largest |case-mean window ΔH| over junctions, by leak type, vs the leak-free noise."""
    c = cache.contract
    slots_tod, slots_wd = slot_index(c, "tod"), slot_index(c, "tod_weekday")
    recs = []
    for _, case in cases.iterrows():
        sid = int(case["scenario_id"])
        p, labels = cache.get(sid), cache.labels(sid)
        cd = case_delta_h(p, case, cfg, labels, slots_tod, slots_wd)
        win = case_windows(cd.dh, case, cfg)
        recs.append((case["leak_type"], float(np.abs(win.x.mean(axis=0)).max())))
    df = pd.DataFrame(recs, columns=["leak_type", "max_abs_dh"])
    rows = []
    for t, g in df.groupby("leak_type"):
        v = g["max_abs_dh"]
        rows.append(
            {
                "leak_type": t,
                "n": len(v),
                "median_m": float(v.median()),
                "p10_m": float(v.quantile(0.1)),
                "share_above_3x_noise": float((v > 3 * noise_m).mean()),
            }
        )
    return pd.DataFrame(rows)


def to_feature_config(
    base: FeatureConfig, decision: dict, method_label: str, max_lag_days: float | None
) -> FeatureConfig:
    m = {x.label: x for x in METHODS}[method_label]
    return replace(
        base,
        slot=m.slot,
        baseline_days=m.baseline_days,
        estimator=m.estimator,
        window=decision["window"],
        stride=decision["stride"],
        max_lag_days=max_lag_days,
    )
