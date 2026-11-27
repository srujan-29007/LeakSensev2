"""M15 stretch goals: leak scan, leak detection, overlapping-leak stress test (PLAN.md M15).

CV scenarios only; the locked test fold is never read here.

- :func:`scan_leaks`: every leak's window, pipe and flow from ``Leaks/`` (reporting and the ramp
  check only; flow is never a model input).
- Detection: a non-learned online statistic. A daily window (``W`` steps) ending at ``t`` is
  compared with the ``R`` days right before it (``tod_weekday`` slot profile, as M4). Each
  junction's window-mean ΔH is scaled by its leak-free spread σ_j (training folds' negatives);
  a leak lowers pressure, so the scores are one-sided: ``max`` = max_j(−ΔH_j/σ_j), ``mean`` =
  mean_j(−ΔH_j/σ_j) over sensed junctions.
- Overlap: ΔH windows from the first days where both leaks are active, against the solo case's
  M3/M4 baseline, scored by a localisation model; hits for either / both leaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leaksense.data.features import STEPS_PER_DAY, case_delta_h, slot_profile

DETECT_STATS = ("max", "mean")


# ----------------------------------------------------------------------------- leak scan


def scan_leaks(data_dir, contract, sids, progress: bool = False) -> pd.DataFrame:
    """One row per leak of ``sids``: window, pipe, type and flow (plus the ramp check)."""
    from leaksense.data.index import _read_leaks
    from leaksense.data.validate import expected_timestamps

    ts_index = {t: i for i, t in enumerate(expected_timestamps(contract))}
    rows = []
    for n, sid in enumerate(sorted(int(s) for s in sids), 1):
        if progress and n % 50 == 0:
            print(f"  scanned {n}/{len(sids)} scenarios", flush=True)
        leaks = _read_leaks(data_dir / f"Scenario-{sid}", sid, ts_index, contract)
        leaks.sort(key=lambda lk: (lk.start, lk.id))
        for order, lk in enumerate(leaks):
            f = np.abs(lk.flow[lk.start : lk.end + 1])
            med = float(np.median(f))
            reach = np.flatnonzero(f >= 0.95 * med)
            rows.append(
                {
                    "scenario_id": sid,
                    "n_leaks": len(leaks),
                    "order": order,  # 0 = first leak to start
                    "leak_id": lk.id,
                    "pipe": lk.pipe,
                    "leak_type": lk.type,
                    "start": lk.start,
                    "end": lk.end,
                    "n_steps": lk.end - lk.start + 1,
                    "flow_median_lps": med,
                    "flow_first_ratio": float(f[0] / med) if med > 0 else np.nan,
                    "steps_to_95pct": int(reach[0]) if len(reach) else -1,
                    "flow_cv": float(f.std() / f.mean()) if f.mean() > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def ramp_summary(leaks: pd.DataFrame) -> dict:
    r = leaks["flow_first_ratio"].to_numpy()
    s = leaks["steps_to_95pct"].to_numpy()
    return {
        "n_leaks": len(leaks),
        "first_step_ratio": {
            "min": float(np.nanmin(r)),
            "p5": float(np.nanpercentile(r, 5)),
            "median": float(np.nanmedian(r)),
        },
        "steps_to_95pct": {
            "median": float(np.median(s)),
            "p95": float(np.percentile(s, 95)),
            "max": int(s.max()),
            "n_zero": int((s == 0).sum()),
        },
        "flow_cv_median": float(np.nanmedian(leaks["flow_cv"])),
        # "full size from the first step": >= 95% of the median at step 0 for (almost) every leak
        "starts_at_full_size": bool((s == 0).mean() >= 0.95),
    }


# ----------------------------------------------------------------------------- detection


@dataclass(frozen=True)
class DetectConfig:
    window: int = 48  # steps per daily window
    ref_days: int = 14  # rolling reference length
    buffer_steps: int = 48  # negatives end at least this long before any leak starts
    neg_per_scenario: int = 20
    ttd_step: int = 2  # time-to-detect grid: a window every hour after the start
    seed: int = 0


def rolling_dh(p, t_end: int, cfg: DetectConfig, slots_tod, slots_wd) -> np.ndarray:
    """Window-mean ΔH of rows [t_end − W, t_end) against the R days right before them."""
    W, R = cfg.window, cfg.ref_days * STEPS_PER_DAY
    ref = np.arange(t_end - W - R, t_end - W)
    win = np.arange(t_end - W, t_end)
    prof, _ = slot_profile(p[ref], ref, "tod_weekday", "mean", slots_tod, slots_wd)
    return (p[win].astype(np.float64) - prof[slots_wd[win]]).mean(axis=0)


def detection_samples(p, labels, leaks: pd.DataFrame, cfg: DetectConfig, slots_tod, slots_wd):
    """Negatives, positives and time-to-detect windows for one scenario.

    Returns (neg_x (n, J), neg_t, pos list of (leak row index, x (J,), ttd (H, J) or None)).
    """
    n = len(labels)
    W, R, B = cfg.window, cfg.ref_days * STEPS_PER_DAY, cfg.buffer_steps
    act = labels.astype(bool)
    csum = np.concatenate([[0], np.cumsum(act)])

    def clean(a: int, b: int) -> bool:  # rows [a, b) leak-free and inside the series
        return a >= 0 and b <= n and csum[b] - csum[a] == 0

    sid = int(leaks["scenario_id"].iloc[0]) if len(leaks) else 0
    cand = [t for t in range(W + R, n + 1, STEPS_PER_DAY) if clean(t - W - R, min(n, t + B))]
    cand = [t for t in cand if t + B <= n]
    rng = np.random.default_rng([cfg.seed, sid])
    pick = np.sort(rng.choice(len(cand), size=min(cfg.neg_per_scenario, len(cand)), replace=False))
    neg_t = np.array([cand[i] for i in pick], dtype=np.int64)
    neg_x = np.array([rolling_dh(p, int(t), cfg, slots_tod, slots_wd) for t in neg_t])

    pos = []
    hours = np.arange(cfg.ttd_step, W + 1, cfg.ttd_step)
    for i, lk in leaks.iterrows():
        s = int(lk["start"])
        if not clean(s - R, s) or s + W > n or not act[s : s + W].all():
            continue
        x = rolling_dh(p, s + W, cfg, slots_tod, slots_wd)
        ttd = None
        if clean(s - R - W, s):
            ttd = np.array([rolling_dh(p, s + int(h), cfg, slots_tod, slots_wd) for h in hours])
        pos.append((i, x, ttd))
    return neg_x.reshape(-1, p.shape[1]), neg_t, pos


def detect_scores(x: np.ndarray, mask: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    """One-sided scores (leaks lower pressure) over the sensed junctions; x is (..., J)."""
    z = -x[..., mask] / sigma[mask]
    return {"max": z.max(axis=-1), "mean": z.mean(axis=-1)}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    rp = ranks[len(neg) :].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------- overlap


def overlap_periods(leaks: pd.DataFrame) -> pd.DataFrame:
    """Both-active period of every two-leak scenario in ``leaks`` (length 0 if none)."""
    rows = []
    for sid, g in leaks[leaks["n_leaks"] == 2].groupby("scenario_id"):
        a, b = g.sort_values("order").iloc[0], g.sort_values("order").iloc[1]
        s, e = max(a["start"], b["start"]), min(a["end"], b["end"])
        rows.append(
            {
                "scenario_id": int(sid),
                "first_pipe": a["pipe"],
                "second_pipe": b["pipe"],
                "same_pipe": a["pipe"] == b["pipe"],
                "first_type": a["leak_type"],
                "second_type": b["leak_type"],
                "first_flow_lps": a["flow_median_lps"],
                "second_flow_lps": b["flow_median_lps"],
                "ov_start": int(s),
                "ov_end": int(e),
                "ov_steps": int(max(0, e - s + 1)),
            }
        )
    return pd.DataFrame(rows)


def overlap_windows(p, labels, solo_case, ov, fcfg, slots_tod, slots_wd, max_days: float = 7):
    """ΔH windows from the first ``max_days`` of the overlap, against the solo case's baseline."""
    case = dict(solo_case)
    case["start"], case["end"] = int(ov["ov_start"]), int(ov["ov_end"])
    case["case_id"] = f"S{int(ov['scenario_id']):03d}-overlap"
    cd = case_delta_h(p, case, fcfg, labels, slots_tod, slots_wd)
    n, W = cd.dh.shape[0], fcfg.window
    lim = min(n, int(round(max_days * STEPS_PER_DAY)))
    offs = np.arange(0, lim - W + 1, fcfg.stride)
    if len(offs) == 0:
        return None, cd.rows_b
    x = np.stack([cd.dh[o : o + W].mean(axis=0) for o in offs]).astype(np.float32)
    return x, cd.rows_b
