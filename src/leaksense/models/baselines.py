"""Non-learned rungs (PLAN.md M6): chance, class prior, max-|ΔH|.

Each returns one score per leak-eligible pipe (``Network.leak_pipe_ids``); higher is better.
Pipe 1 is never a class. The harness zeroes unsensed junctions before calling ``case_scores``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leaksense.network import Network


class Chance:
    """All pipes tie: every metric is the exact expectation (top-k = k / 33)."""

    name = "chance"

    def __init__(self, net: Network) -> None:
        self.n = len(net.leak_pipe_ids)

    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None:
        pass

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.zeros(self.n)


class Prior:
    """Class frequency in the training folds, plus one (Laplace). Ignores the sensors."""

    name = "prior"

    def __init__(self, net: Network) -> None:
        self.n = len(net.leak_pipe_ids)
        self.counts: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None:
        self.counts = np.bincount(train["y"].to_numpy(), minlength=self.n) + 1.0

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.counts is None:
            raise RuntimeError("Prior.fit was not called")
        return np.log(self.counts / self.counts.sum())


class MaxDH:
    """Rank pipes by hop distance from the sensed junction with the largest |mean ΔH|.

    A pipe's key is (d(j1, p), d(j2, p), ...) over the sensed junctions in decreasing |ΔH|,
    where d(j, p) is the node-hop distance from junction j to the nearer end of pipe p (0 if p
    touches j). Pipes are ranked lexicographically; identical keys share a rank.
    """

    name = "maxdh"

    def __init__(self, net: Network) -> None:
        nodes = net.node_index
        sensors = [nodes[j] for j in net.sensor_ids]
        p = net.pipes.set_index("id").loc[net.leak_pipe_ids]
        u, v = p["u"].to_numpy(), p["v"].to_numpy()
        H = net.node_hops
        # (31 sensors x 33 pipes)
        self.d = np.minimum(H[np.ix_(sensors, u)], H[np.ix_(sensors, v)])
        self.n = len(net.leak_pipe_ids)

    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None:
        pass

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sensed = np.flatnonzero(mask)
        if len(sensed) == 0:
            return np.zeros(self.n)
        m = np.abs(np.asarray(x, dtype=float).mean(axis=0))
        order = sensed[np.argsort(-m[sensed], kind="stable")]
        keys = self.d[order]  # (n_sensed x 33), most important row first
        perm = np.lexsort(keys[::-1])  # lexsort uses the last key as primary
        sk = keys[:, perm]
        new = np.concatenate([[True], (sk[:, 1:] != sk[:, :-1]).any(axis=0)])
        rank = np.empty(self.n)
        rank[perm] = np.cumsum(new) - 1
        return -rank


def signatures(net: Network, sensitivity: dict, levels: str = "all") -> np.ndarray:
    """S as (31 junctions x 33 pipes x variants): every (size, level) column is a signature.

    ``levels="mid"`` keeps the median demand level only (2 variants), ``"all"`` keeps all (6).
    Checks that S's junctions and pipes are the network's sensors and leak-eligible pipes.
    """
    if list(sensitivity["pipes"]) != net.leak_pipe_ids:
        raise ValueError("sensitivity pipes do not match the network's leak-eligible pipes")
    if list(sensitivity["junctions"]) != net.sensor_ids:
        raise ValueError("sensitivity junctions do not match the network's sensors")
    S = np.asarray(sensitivity["S"], dtype=float)
    lv = list(sensitivity["levels"])
    if levels == "mid":
        S = S[..., [lv.index("mid")]]
    elif levels != "all":
        raise ValueError(f"levels must be 'mid' or 'all', not {levels!r}")
    return S.reshape(S.shape[0], S.shape[1], -1)


class SigMatch:
    """Cosine between the case-mean ΔH and each pipe's sensitivity column, on sensed junctions.

    ``S`` is (31 junctions x 33 pipes x sizes x levels) from ``leaksense build-sensitivity``.
    The leak size is unknown, so the score is the max over sizes; ``levels="mid"`` uses the
    median demand level, ``levels="all"`` takes the max over all levels too.
    """

    name = "sigmatch"

    def __init__(self, net: Network, sensitivity: dict | None = None, levels: str = "mid") -> None:
        if sensitivity is None:
            raise ValueError("sigmatch needs the sensitivity matrix (leaksense build-sensitivity)")
        S = signatures(net, sensitivity, levels)
        self.S = S
        self.n = S.shape[1]
        self.name = "sigmatch" if levels == "mid" else "sigmatch_all"

    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None:
        pass

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask, dtype=bool)
        v = np.asarray(x, dtype=float).mean(axis=0)[m]
        cols = self.S[m]  # (n_sensed x 33 x variants)
        nv = np.linalg.norm(v)
        nc = np.linalg.norm(cols, axis=0)
        if nv == 0:
            return np.zeros(self.n)
        cos = np.einsum("j,jpv->pv", v, cols) / (nv * np.where(nc > 0, nc, np.inf))
        return np.round(cos.max(axis=1), 12)  # rounding makes numerically equal scores tie


class SigMatchAll(SigMatch):
    def __init__(self, net: Network, sensitivity: dict | None = None) -> None:
        super().__init__(net, sensitivity, levels="all")


MODELS = {
    "chance": Chance,
    "prior": Prior,
    "maxdh": MaxDH,
    "sigmatch": SigMatch,
    "sigmatch_all": SigMatchAll,
}
NEEDS_SENSITIVITY = {"sigmatch", "sigmatch_all"}
