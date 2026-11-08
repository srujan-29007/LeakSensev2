"""Masked MLP rung (PLAN.md M8).

Input per window: ``[T(ΔH·m), m]`` over the 31 junctions, where T is the M4 input transform
(``leaksense.transforms``). Training draws a fresh sensor mask per sample and epoch, with k
uniform in ``k_min..31``, so one model serves every sensor budget.

- Sample weight = (1 / windows in its case) × class weight. Class weights are computed from the
  training cases passed to ``fit`` only (one fold's training set), never from the scored fold.
- Settings given as lists in the config (e.g. ``transform: [none, signed_log1p]``) are chosen by
  weighted loss on an inner validation split of the training scenarios; the model is then
  refitted on the whole training set for the chosen number of epochs.
- A case's score per pipe is the mean over its windows of the log-probabilities.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from leaksense import transforms
from leaksense.network import Network

CLASS_WEIGHTS = ("none", "inverse", "sqrt_inverse")


@dataclass(frozen=True)
class MLPConfig:
    transform: str = "signed_log1p"
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 30  # epochs without inner-validation improvement before stopping
    val_frac: float = 0.15  # share of each pipe's training scenarios held out for early stopping
    val_masks: int = 4  # fixed random masks per validation window
    refit: bool = True  # retrain on the whole training set for the chosen epoch count
    # score the chosen inner-validation net at these budgets, e.g. "2,3,4,6,8" (M10 selection);
    # a string, because list-valued settings are grids
    inner_eval_ks: str = ""
    class_weight: str = "inverse"
    k_min: int = 1  # training mask size k ~ U{k_min..31}
    seed: int = 0

    def __post_init__(self) -> None:
        if self.transform not in transforms.TRANSFORMS:
            raise ValueError(f"transform must be one of {transforms.TRANSFORMS}")
        if self.class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}")
        if self.layers < 1 or self.hidden < 1 or self.k_min < 1:
            raise ValueError("layers, hidden and k_min must be >= 1")


def parse_config(d: dict, cls: type = MLPConfig) -> tuple[Any, dict[str, list]]:
    """Model config dict -> (base config of type ``cls``, grid of list-valued settings)."""
    d = {k: v for k, v in d.items() if k != "kind"}
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    grid = {k: list(v) for k, v in d.items() if isinstance(v, list)}
    base = cls(**{k: (v[0] if isinstance(v, list) else v) for k, v in d.items()})
    for k, vals in grid.items():  # validate every grid value
        for v in vals:
            replace(base, **{k: v})
    return base, grid


# ----------------------------------------------------------------------------- inputs


def encode(x: np.ndarray, mask: np.ndarray, transform: str) -> np.ndarray:
    """(n, J) ΔH and (n, J) mask -> (n, 2J) or (n, 2J+1) model input. Unsensed ΔH is zeroed."""
    m = mask.astype(bool)
    xm = np.where(m, x, 0.0)
    return np.concatenate([transforms.apply(transform, xm, m), m.astype(np.float32)], axis=1)


def split_encoded(x, j: int):
    """Inverse of the ``encode`` layout -> (pattern, magnitude or None, mask), as slices.

    The row is ``[T(ΔH·m) (j), magnitude (1, scale_magnitude only), m (j)]``. Works on numpy
    arrays and torch tensors. Width 2j means no magnitude column, 2j+1 means one.
    """
    w = x.shape[-1]
    if w == 2 * j:
        return x[..., :j], None, x[..., j:]
    if w == 2 * j + 1:
        return x[..., :j], x[..., j : j + 1], x[..., j + 1 :]
    raise ValueError(f"input width {w} does not match {j} junctions")


def random_k_masks(n: int, j: int, k_min: int, rng: np.random.Generator) -> np.ndarray:
    """(n, j) masks, each with k ~ U{k_min..j} random sensed junctions."""
    k = rng.integers(min(k_min, j), j + 1, size=n)
    rank = np.argsort(np.argsort(rng.random((n, j)), axis=1), axis=1)
    return rank < k[:, None]


def class_weights(y: np.ndarray, n_classes: int, kind: str) -> np.ndarray:
    """Per-class weight from training case counts, normalised to mean 1 over the cases."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if kind == "none":
        w = np.ones(n_classes)
    else:
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        if kind == "sqrt_inverse":
            w = np.sqrt(w)
    return w / (w[y].mean() if len(y) else 1.0)


def inner_split(cases: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Boolean validation mask over ``cases``, grouped by scenario, about ``frac`` of each pipe.

    Pipes with fewer than 1 / frac cases keep every case in training.
    """
    rng = np.random.default_rng(seed)
    scen = cases["scenario_id"].to_numpy()
    val_scen: set[int] = set()
    for _, g in cases.groupby("y", sort=True):
        s = np.unique(g["scenario_id"].to_numpy())
        n_val = int(frac * len(s))
        val_scen.update(int(v) for v in rng.choice(s, size=n_val, replace=False))
    return np.isin(scen, list(val_scen))


# ----------------------------------------------------------------------------- network


class _Net(nn.Module):
    def __init__(self, d_in: int, n_out: int, cfg: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(cfg.layers):
            layers += [nn.Linear(d, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden
        layers.append(nn.Linear(d, n_out))
        self.f = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


@dataclass
class _Data:
    x: np.ndarray  # (n_windows, J) raw ΔH
    y: np.ndarray  # (n_windows,)
    w: np.ndarray  # (n_windows,) case weight x class weight


def _windows(cases: pd.DataFrame, windows: list[np.ndarray], cw: np.ndarray) -> _Data:
    ys = cases["y"].to_numpy()
    x = np.concatenate(windows).astype(np.float32)
    y = np.concatenate([np.full(len(w), c) for w, c in zip(windows, ys, strict=True)])
    w = np.concatenate(
        [np.full(len(win), cw[c] / len(win)) for win, c in zip(windows, ys, strict=True)]
    )
    return _Data(x, y.astype(np.int64), (w / w.mean()).astype(np.float32))


def _seed_for(cases: pd.DataFrame, seed: int) -> int:
    """Deterministic per-fold seed: the training scenarios identify the fold."""
    key = ",".join(map(str, sorted(cases["scenario_id"].astype(int))))
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16)


def predict_logits(net: nn.Module, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """``net(x)`` in chunks of rows (bounds memory for graph models); call under no_grad."""
    if len(x) <= chunk:
        return net(x)
    return torch.cat([net(x[i : i + chunk]) for i in range(0, len(x), chunk)])


def _train(
    cfg: Any,
    tr: _Data,
    seed: int,
    epochs: int,
    val: _Data | None = None,
    *,
    make_net: Callable[[Any, int], nn.Module],
) -> tuple[nn.Module, int, float, list[float]]:
    """Train; with ``val``, stop early and return the best epoch's weights.

    ``make_net(cfg, d_in)`` builds the module; it receives ``encode`` output rows of width d_in.
    ``cfg`` needs the training fields of ``MLPConfig`` (transform, lr, batch_size, ...).
    Returns (net, epochs used, best validation loss or nan, validation-loss curve).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    j = tr.x.shape[1]
    d_in = encode(tr.x[:1], np.ones((1, j), bool), cfg.transform).shape[1]
    net = make_net(cfg, d_in)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    yt, wt = torch.from_numpy(tr.y), torch.from_numpy(tr.w)

    if val is not None:
        vr = np.random.default_rng(seed + 1)
        vm = np.concatenate([random_k_masks(len(val.x), j, 1, vr) for _ in range(cfg.val_masks)])
        vx = torch.from_numpy(
            encode(np.tile(val.x, (cfg.val_masks, 1)), vm, cfg.transform).astype(np.float32)
        )
        vy = torch.from_numpy(np.tile(val.y, cfg.val_masks))
        vw = torch.from_numpy(np.tile(val.w, cfg.val_masks))

    best, best_ep, best_state, since, curve = np.inf, 0, None, 0, []
    n = len(tr.x)
    for ep in range(1, epochs + 1):
        net.train()
        xin = torch.from_numpy(
            encode(tr.x, random_k_masks(n, j, cfg.k_min, rng), cfg.transform).astype(np.float32)
        )
        perm = torch.from_numpy(rng.permutation(n))
        for b in range(0, n, cfg.batch_size):
            idx = perm[b : b + cfg.batch_size]
            loss = nn.functional.cross_entropy(net(xin[idx]), yt[idx], reduction="none")
            loss = (loss * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is None:
            continue
        net.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(predict_logits(net, vx), vy, reduction="none")
            vloss = float((vl * vw).sum() / vw.sum())
        curve.append(vloss)
        if vloss < best - 1e-4:
            best, best_ep, since = vloss, ep, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break
    if val is None:
        return net, epochs, float("nan"), curve
    net.load_state_dict(best_state)
    return net, best_ep, best, curve


# ----------------------------------------------------------------------------- model


def inner_scores(
    net: nn.Module,
    transform: str,
    cases: pd.DataFrame,
    windows: list[np.ndarray],
    ks: list[int],
    draws: int = 20,
    seed: int = 0,
) -> dict:
    """Case-level top-3 of ``net`` on inner-validation cases, per k (M10 stage-2 selection).

    Uses ``evaluate.random_masks`` (the same seeded sets as the harness) and its exact tie
    scoring. Returns {k: {"top3": all cases, "small_top3": small leaks (if leak_type known)}}.
    """
    from leaksense.evaluate import case_metrics, random_masks

    net.eval()
    j = windows[0].shape[1]
    lens = np.array([len(w) for w in windows])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    x = np.concatenate(windows).astype(np.float32)
    ys = cases["y"].to_numpy()
    small = (cases["leak_type"] == "small").to_numpy() if "leak_type" in cases else None
    out = {}
    for k in ks:
        top3 = np.zeros(len(cases))
        masks = random_masks(j, k, draws, seed)
        for m in masks:
            mm = np.broadcast_to(m, x.shape)
            with torch.no_grad():
                xin = torch.from_numpy(encode(x, mm, transform).astype(np.float32))
                lp = torch.log_softmax(predict_logits(net, xin), dim=1).numpy()
            sc = np.add.reduceat(lp, starts, axis=0) / lens[:, None]
            n_cls = sc.shape[1]
            d, h = np.zeros(n_cls), np.zeros((n_cls, n_cls))
            top3 += [case_metrics(sc[i], int(ys[i]), d, h)["top3"] for i in range(len(ys))]
        top3 /= len(masks)
        r = {"top3": round(float(top3.mean()), 4), "n": len(ys)}
        if small is not None and small.any():
            r["small_top3"] = round(float(top3[small].mean()), 4)
            r["n_small"] = int(small.sum())
        out[str(k)] = r
    return out


class MaskedMLP:
    """Masked MLP over ``[T(ΔH·m), m]``; see the module docstring."""

    name = "mlp"
    config_cls: type = MLPConfig

    def __init__(self, net: Network, model_cfg: dict | None = None) -> None:
        self.n = len(net.leak_pipe_ids)
        self.base, self.grid = parse_config(model_cfg or {}, self.config_cls)
        self.cfg = self.base
        self.net: nn.Module | None = None
        self.history: list[dict] = []  # one entry per fit (fold)

    def _make_net(self, cfg: Any, d_in: int) -> nn.Module:
        """Build the module for one training run; subclasses swap the architecture."""
        return _Net(d_in, self.n, cfg)

    # -- training
    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None:
        seed = _seed_for(train, self.base.seed)
        is_val = inner_split(train, self.base.val_frac, seed)
        cand = [
            replace(self.base, **dict(zip(self.grid, vals, strict=True)))
            for vals in itertools.product(*self.grid.values())
        ] or [self.base]

        trials = []
        if is_val.any():
            tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
            tr_cases = train.iloc[tr_idx]
            va_cases = train.iloc[va_idx]
            for cfg in cand:
                cw = class_weights(tr_cases["y"].to_numpy(), self.n, cfg.class_weight)
                tr = _windows(tr_cases, [windows[i] for i in tr_idx], cw)
                va = _windows(va_cases, [windows[i] for i in va_idx], cw)
                net, ep, vloss, _ = _train(
                    cfg, tr, seed, cfg.max_epochs, va, make_net=self._make_net
                )
                trials.append({"cfg": cfg, "net": net, "epochs": ep, "val_loss": vloss})
            pick = min(trials, key=lambda t: t["val_loss"])
        else:  # too few cases for an inner split (tiny tests): train the base config as is
            pick = {"cfg": cand[0], "net": None, "epochs": cand[0].max_epochs, "val_loss": None}

        cfg = pick["cfg"]
        inner_eval = {}
        if cfg.inner_eval_ks and pick["net"] is not None:  # before the refit overwrites it
            inner_eval = inner_scores(
                pick["net"],
                cfg.transform,
                train.iloc[va_idx],
                [windows[i] for i in va_idx],
                [int(k) for k in cfg.inner_eval_ks.split(",")],
            )
        if cfg.refit or pick["net"] is None:
            cw = class_weights(train["y"].to_numpy(), self.n, cfg.class_weight)
            full = _windows(train, windows, cw)
            self.net, _, _, _ = _train(
                cfg, full, seed, max(1, pick["epochs"]), make_net=self._make_net
            )
        else:
            self.net = pick["net"]
        self.net.eval()
        self.cfg = cfg
        self.history.append(
            {
                "n_train_cases": len(train),
                "n_inner_val_cases": int(is_val.sum()),
                "chosen": {k: getattr(cfg, k) for k in self.grid},
                "epochs": int(pick["epochs"]),
                "val_loss": pick["val_loss"],
                **({"inner_eval": inner_eval} if inner_eval else {}),
                "trials": [
                    {
                        **{k: getattr(t["cfg"], k) for k in self.grid},
                        "epochs": t["epochs"],
                        "val_loss": round(t["val_loss"], 4),
                    }
                    for t in trials
                ],
            }
        )

    # -- scoring
    def batch_scores(self, xs: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError(f"{type(self).__name__}.fit was not called")
        lens = np.array([len(x) for x in xs])
        x = np.concatenate(xs).astype(np.float32)
        m = np.broadcast_to(np.asarray(mask, bool), x.shape)
        with torch.no_grad():
            xin = torch.from_numpy(encode(x, m, self.cfg.transform).astype(np.float32))
            lp = torch.log_softmax(predict_logits(self.net, xin), dim=1).numpy()
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        return np.add.reduceat(lp, starts, axis=0) / lens[:, None]

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.batch_scores([x], mask)[0]

    def describe(self) -> dict:
        return {"base": asdict(self.base), "grid": self.grid, "folds": self.history}
