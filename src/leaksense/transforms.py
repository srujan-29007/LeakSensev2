"""Input transforms for ΔH (applied in the models, M8+; the cache stores raw ΔH).

Small leaks move ΔH by about 1 m, big ones by tens of metres (D11), so raw ΔH spans two orders
of magnitude. Both transforms are per sample and need no fitted statistics.
"""

from __future__ import annotations

import numpy as np

TRANSFORMS = ("none", "signed_log1p", "scale_magnitude")


def signed_log1p(x: np.ndarray, scale_m: float = 1.0) -> np.ndarray:
    """sign(x) * log1p(|x| / scale): ~linear below ``scale_m``, logarithmic above."""
    return np.sign(x) * np.log1p(np.abs(x) / scale_m)


def scale_magnitude(
    x: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Divide each sample by its largest |ΔH| over sensed junctions; return the magnitude too.

    ``x`` is (n, J); ``mask`` (n, J) marks sensed junctions (all, if None). Returns the scaled
    pattern (n, J), zero where unsensed, and ``log1p`` of the magnitude (n, 1) as an extra input.
    """
    m = np.ones_like(x, dtype=bool) if mask is None else mask.astype(bool)
    xm = np.where(m, x, 0.0)
    mag = np.abs(xm).max(axis=1, keepdims=True)
    return xm / np.maximum(mag, eps), np.log1p(mag)


def apply(name: str, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Transform (n, J) ΔH into model input; ``scale_magnitude`` appends one column."""
    if name == "none":
        return x if mask is None else np.where(mask, x, 0.0)
    if name == "signed_log1p":
        y = signed_log1p(x)
        return y if mask is None else np.where(mask, y, 0.0)
    if name == "scale_magnitude":
        pattern, mag = scale_magnitude(x, mask)
        return np.concatenate([pattern, mag], axis=1)
    raise ValueError(f"unknown transform {name!r}; one of {TRANSFORMS}")
