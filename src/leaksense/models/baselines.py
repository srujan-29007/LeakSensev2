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

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.zeros(self.n)


class Prior:
    """Class frequency in the training folds, plus one (Laplace). Ignores the sensors."""

    name = "prior"

    def __init__(self, net: Network) -> None:
        self.n = len(net.leak_pipe_ids)
        self.counts: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> None:
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

    def fit(self, train: pd.DataFrame) -> None:
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


MODELS = {"chance": Chance, "prior": Prior, "maxdh": MaxDH}
