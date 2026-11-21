"""Sensor placement (PLAN.md M11).

With the masked model held fixed, choose which k junctions to instrument:

- **greedy** (learned): forward selection. Each step adds the junction that maximises case-level
  top-3 on the fold's inner-validation cases, scored with the net trained on inner-train only
  (the early-stopping net, before the refit), so the greedy never scores cases its net was
  trained on. Ties: mean true-class score (the mean window log-prob), then the lowest junction.
- **coverage** (physics only, no cases): forward selection on the sensitivity matrix S. Each
  S column (pipe x size x level) is scaled to max |.| = 1 over the junctions; each step adds the
  junction that most raises the mean over columns of the largest scaled |S| among the chosen.
- **random**: ``evaluate.random_masks(31, k, n, seed)``, the same sets in every fold.

Every set is scored on the outer-fold cases with the refit net (and with ``sigmatch_all`` for
reference), so placements are compared on cases neither the greedy nor the model saw.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from leaksense.evaluate import case_metrics

COLS = ("top1", "top3", "top5", "hop", "dist_m")


def topk_hits(scores: np.ndarray, ys: np.ndarray, k: int = 3) -> np.ndarray:
    """Tie-aware top-k per case, vectorised; equals ``case_metrics(...)[f"top{k}"]``."""
    s = np.asarray(scores, dtype=float)
    sy = s[np.arange(len(ys)), ys][:, None]
    n_better = (s > sy).sum(axis=1)
    n_tied = (s == sy).sum(axis=1)
    return np.clip((k - n_better) / n_tied, 0.0, 1.0)


def greedy_learned(
    score_fn: Callable[[np.ndarray], np.ndarray], ys: np.ndarray, n_sensors: int, k_max: int
) -> tuple[list[int], list[dict]]:
    """Forward selection maximising mean top-3 of ``score_fn(mask)`` (n_cases, n_classes).

    Returns the junction order (sets are its prefixes) and, per step, the chosen junction's
    inner top-3 and mean true-class score.
    """
    ys = np.asarray(ys, dtype=int)
    rows = np.arange(len(ys))
    chosen: list[int] = []
    trace: list[dict] = []
    for _ in range(min(k_max, n_sensors)):
        best, best_key = None, None
        for j in range(n_sensors):
            if j in chosen:
                continue
            m = np.zeros(n_sensors, dtype=bool)
            m[chosen + [j]] = True
            sc = np.asarray(score_fn(m), dtype=float)
            top3 = float(topk_hits(sc, ys, 3).mean())
            true = float(sc[rows, ys].mean())
            key = (round(top3, 9), round(true, 9), -j)
            if best_key is None or key > best_key:
                best, best_key = j, key
        chosen.append(best)
        trace.append({"junction": best, "top3": best_key[0], "true_score": best_key[1]})
    return chosen, trace


def greedy_coverage(S: np.ndarray, k_max: int) -> list[int]:
    """Forward selection on S (n_sensors x pipes x ...): see the module docstring."""
    A = np.abs(np.asarray(S, dtype=float)).reshape(S.shape[0], -1)
    top = A.max(axis=0)
    A = A[:, top > 0] / top[top > 0]
    n = A.shape[0]
    cover = np.zeros(A.shape[1])
    chosen: list[int] = []
    for _ in range(min(k_max, n)):
        gain = np.round(np.maximum(cover[None, :], A).mean(axis=1), 12)
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))  # first maximum = lowest junction on ties
        chosen.append(j)
        cover = np.maximum(cover, A[j])
    return chosen


def as_mask(junctions: list[int], n_sensors: int) -> np.ndarray:
    m = np.zeros(n_sensors, dtype=bool)
    m[list(junctions)] = True
    return m


def per_case_metrics(
    scores: np.ndarray, ys: np.ndarray, dist: np.ndarray, hops: np.ndarray
) -> np.ndarray:
    """(n_cases, 5) of COLS from per-case scores, via the harness's ``case_metrics``."""
    out = np.zeros((len(ys), len(COLS)))
    for i, y in enumerate(ys):
        cm = case_metrics(scores[i], int(y), dist[i], hops)
        out[i] = [cm[c] for c in COLS]
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def mean_pairwise_jaccard(sets: list) -> float:
    vals = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals)) if vals else float("nan")
