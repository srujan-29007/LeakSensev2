"""Masked GAT rung (PLAN.md M9) and its physics-input variants (M10).

Same protocol as the masked MLP (``leaksense.models.mlp``): the input row per window is
``encode(ΔH, mask)`` = ``[T(ΔH·m), (magnitude), m]`` over the 31 junctions, sensor masks with
k ~ U{k_min..31}, class x case sample weights, inner validation for list-valued settings and
early stopping, then a refit. Only the network differs:

- The row is scattered onto the nominal graph: each junction gets ``[T(ΔH·m), m]``; the
  reservoir always gets ``[0, 0]`` (its pressure is always 0, D5, and it is never a sensor).
  An optional learned per-node embedding gives the shared GAT weights node identity.
- 4 GATv2 layers (pre-norm residual) with nominal edge features: log length, diameter,
  roughness and Hazen-Williams resistance, z-scored over the pipes.
- Pipe head: for each leak-eligible pipe (pipe 1 excluded), an MLP on
  ``[h_u + h_v, |h_u - h_v|, edge features, magnitude]`` gives one logit (symmetric in u, v).

Physics inputs (M10, ``physics``) use the sensitivity matrix S from the nominal network (M7).
S enters only inside ``forward``, only through ``masked_cosine``: the cosine between the
window's sensed pattern and each pipe's signature, both restricted to the sensed junctions.
So neither unsensed ΔH nor S at unsensed junctions can reach the output.

- ``node``: one more node channel, the pattern imputed from S, Σ softmax(cos/τ) · Ŝ (Ŝ is each
  S column scaled to max |·| = 1; τ is learned). Unsensed nodes get a value from sensed ones.
- ``edge``: each pipe's cosines are extra edge features (both directions; pipe 1 gets 0) and
  extra pipe-head inputs.
- ``prior``: logit_p += a · max_v cos_pv with a = softplus(θ) learned (starts at
  ``prior_scale``), so training starts near ``sigmatch_all`` and the GAT learns a correction.

Everything here comes from the nominal ``Hanoi.inp``; no scenario parameters reach the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from leaksense.models.mlp import MaskedMLP, MLPConfig, split_encoded
from leaksense.network import EDGE_FEATURES, Network

PHYSICS = ("none", "node", "edge", "prior")
S_LEVELS = ("all", "mid")


@dataclass(frozen=True)
class GATConfig(MLPConfig):
    transform: str = "scale_magnitude"
    hidden: int = 64  # total width; split over ``heads``
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    node_embed: int = 8  # learned per-node embedding size (0 = off)
    max_epochs: int = 150
    patience: int = 20
    num_threads: int = 4
    physics: str = "none"  # none | node | edge | prior (M10)
    s_levels: str = "all"  # S demand levels used: all (6 signatures per pipe) | mid (2)
    prior_scale: float = 10.0  # initial a in the prior logit a · max cos
    tau: float = 0.1  # initial softmax temperature of the node variant

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.heads < 1 or self.hidden % self.heads:
            raise ValueError("hidden must be a positive multiple of heads")
        if self.node_embed < 0 or self.num_threads < 1:
            raise ValueError("node_embed must be >= 0 and num_threads >= 1")
        if self.physics not in PHYSICS:
            raise ValueError(f"physics must be one of {PHYSICS}, not {self.physics!r}")
        if self.s_levels not in S_LEVELS:
            raise ValueError(f"s_levels must be one of {S_LEVELS}, not {self.s_levels!r}")
        if self.physics != "none" and self.transform == "signed_log1p":
            # the cosine is scale-free, so it is the same on raw or scaled ΔH, but not on log ΔH
            raise ValueError("physics inputs need transform scale_magnitude or none")
        if self.prior_scale <= 0 or self.tau <= 0:
            raise ValueError("prior_scale and tau must be > 0")


@dataclass(frozen=True)
class Graph:
    """Fixed nominal graph tensors shared by every window."""

    n_nodes: int
    sensor_pos: torch.Tensor  # (J,) node position of each feature column
    edge_index: torch.Tensor  # (2, 2P) both directions: pipes 0..P-1, then reversed
    edge_attr: torch.Tensor  # (2P, 4) standardised nominal edge features
    leak_pos: torch.Tensor  # (C,) pipe position of each leak-eligible pipe (class order)
    pipe_u: torch.Tensor  # (C,) endpoints of the leak-eligible pipes
    pipe_v: torch.Tensor
    pipe_attr: torch.Tensor  # (C, 4)


def standardised_pipe_features(net: Network) -> np.ndarray:
    """(P, 4) log of ``EDGE_FEATURES`` per pipe, z-scored over pipes (constant columns -> 0)."""
    p = net.pipes
    raw = np.column_stack(
        [p["length_m"], p["diameter_mm"] / 1000.0, p["roughness"], p["resistance"]]
    ).astype(float)
    if (raw <= 0).any():
        raise ValueError("pipe features must be positive to take logs")
    f = np.log(raw)
    sd = f.std(axis=0)
    return np.where(sd > 1e-9, (f - f.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0), 0.0)


def build_graph(net: Network) -> Graph:
    assert len(EDGE_FEATURES) == 4
    idx = net.node_index
    sensor_pos = [idx[j] for j in net.sensor_ids]
    if net.nodes.loc[sensor_pos, "kind"].ne("junction").any():
        raise ValueError("sensors must be junctions")
    feats = standardised_pipe_features(net)
    pipe_index = net.pipe_index
    leak = [pipe_index[p] for p in net.leak_pipe_ids]
    n_pipes = len(net.pipes)
    ei = np.asarray(net.edge_index)
    if ei.shape[1] != 2 * n_pipes or not np.array_equal(ei[:, :n_pipes], ei[::-1, n_pipes:]):
        raise ValueError("edge_index must list every pipe forward, then the same pipes reversed")
    t = torch.as_tensor
    return Graph(
        n_nodes=len(net.nodes),
        sensor_pos=t(sensor_pos, dtype=torch.long),
        edge_index=t(net.edge_index, dtype=torch.long),
        edge_attr=t(np.concatenate([feats, feats]), dtype=torch.float32),
        leak_pos=t(leak, dtype=torch.long),
        pipe_u=t(net.pipes["u"].to_numpy()[leak], dtype=torch.long),
        pipe_v=t(net.pipes["v"].to_numpy()[leak], dtype=torch.long),
        pipe_attr=t(feats[leak], dtype=torch.float32),
    )


def masked_cosine(pattern: torch.Tensor, mask: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """(b, J) pattern, (b, J) 0/1 mask, (J, C, V) S -> (b, C, V) cosine over sensed junctions.

    cos = Σ_j p_j m_j S_jcv / (‖p·m‖ · √Σ_j m_j S²_jcv). Only sensed rows of the pattern and of
    S enter; a window with no signal (or a pipe with no sensed signature) gets 0.
    """
    p = pattern * mask
    num = torch.einsum("bj,jcv->bcv", p, S)
    s_norm = torch.einsum("bj,jcv->bcv", mask, S * S).clamp_min(0).sqrt()
    p_norm = p.norm(dim=1)[:, None, None]
    den = p_norm * s_norm
    return torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.zeros_like(num))


class _GATNet(nn.Module):
    def __init__(self, g: Graph, d_in: int, cfg: GATConfig, S: np.ndarray | None = None) -> None:
        super().__init__()
        j = len(g.sensor_pos)
        if d_in not in (2 * j, 2 * j + 1):
            raise ValueError(f"input width {d_in} does not match {j} junctions")
        self.g, self.j, self.has_mag, self.physics = g, j, d_in == 2 * j + 1, cfg.physics
        if (cfg.physics != "none") != (S is not None):
            raise ValueError("physics inputs need S, and S is only used with physics inputs")
        n_sig = 0
        if S is not None:
            if S.shape[:2] != (j, len(g.pipe_u)):
                raise ValueError(f"S shape {S.shape} does not match {j} junctions x pipes")
            st = torch.as_tensor(S, dtype=torch.float32)
            self.register_buffer("S", st)
            peak = st.abs().amax(dim=0, keepdim=True)
            self.register_buffer("S_hat", st / peak.clamp_min(1e-12))  # each column max |.| = 1
            n_sig = st.shape[2]
        h = cfg.hidden
        self.embed = nn.Embedding(g.n_nodes, cfg.node_embed) if cfg.node_embed else None
        n_node = 2 + int(cfg.physics == "node")
        self.inp = nn.Linear(n_node + cfg.node_embed, h)
        e_dim = g.edge_attr.shape[1] + (n_sig if cfg.physics == "edge" else 0)
        self.convs = nn.ModuleList(
            GATv2Conv(h, h // cfg.heads, heads=cfg.heads, edge_dim=e_dim) for _ in range(cfg.layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(cfg.layers))
        self.drop = nn.Dropout(cfg.dropout)
        d_head = 2 * h + e_dim + int(self.has_mag)
        self.head = nn.Sequential(nn.Linear(d_head, h), nn.GELU(), nn.Linear(h, 1))
        # created after the M9 modules, so physics "none" initialises exactly as M9 did
        if cfg.physics == "node":
            self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.tau)))
        if cfg.physics == "prior":  # softplus(theta) = prior_scale
            self.theta = nn.Parameter(torch.tensor(math.log(math.expm1(cfg.prior_scale))))

    def physics_params(self) -> dict:
        out = {}
        if self.physics == "node":
            out["tau"] = float(self.log_tau.detach().exp())
        if self.physics == "prior":
            out["prior_scale"] = float(nn.functional.softplus(self.theta.detach()))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, j, b, n = self.g, self.j, x.shape[0], self.g.n_nodes
        pattern, mag, mask = split_encoded(x, j)
        cos = masked_cosine(pattern, mask, self.S) if self.physics != "none" else None
        node = x.new_zeros(b, n, 2 + int(self.physics == "node"))  # reservoir stays 0
        node[:, g.sensor_pos, 0] = pattern
        node[:, g.sensor_pos, 1] = mask
        if self.physics == "node":
            w = torch.softmax(cos.reshape(b, -1) / self.log_tau.exp(), dim=1)
            imputed = torch.einsum("bk,jk->bj", w, self.S_hat.reshape(j, -1))
            node[:, g.sensor_pos, 2] = imputed
        if self.embed is not None:
            node = torch.cat([node, self.embed.weight.expand(b, -1, -1)], dim=2)
        h = self.inp(node).reshape(b * n, -1)
        n_e = g.edge_index.shape[1]
        offs = torch.arange(b, device=x.device).repeat_interleave(n_e) * n
        ei = g.edge_index.repeat(1, b) + offs
        pipe_attr = g.pipe_attr.expand(b, -1, -1)
        if self.physics == "edge":
            pc = x.new_zeros(b, n_e // 2, cos.shape[2])  # pipe 1 (and any non-leak pipe) gets 0
            pc[:, g.leak_pos] = cos
            ea = torch.cat([g.edge_attr.expand(b, -1, -1), torch.cat([pc, pc], dim=1)], dim=2)
            ea = ea.reshape(b * n_e, -1)
            pipe_attr = torch.cat([pipe_attr, cos], dim=2)
        else:
            ea = g.edge_attr.repeat(b, 1)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = h + self.drop(nn.functional.gelu(conv(norm(h), ei, ea)))
        h = h.view(b, n, -1)
        hu, hv = h[:, g.pipe_u], h[:, g.pipe_v]
        parts = [hu + hv, (hu - hv).abs(), pipe_attr]
        if mag is not None:
            parts.append(mag.unsqueeze(1).expand(-1, hu.shape[1], -1))
        logits = self.head(torch.cat(parts, dim=2)).squeeze(2)
        if self.physics == "prior":
            logits = logits + nn.functional.softplus(self.theta) * cos.amax(dim=2)
        return logits


class MaskedGAT(MaskedMLP):
    """Masked GATv2 over the nominal graph; see the module docstring."""

    name = "gat"
    config_cls = GATConfig

    def __init__(
        self, net: Network, model_cfg: dict | None = None, sensitivity: dict | None = None
    ) -> None:
        super().__init__(net, model_cfg)
        self.graph = build_graph(net)
        if len(self.graph.pipe_u) != self.n:
            raise ValueError("leak pipe count mismatch")
        self.signatures: dict[str, np.ndarray] = {}
        if self.wants_sensitivity(model_cfg or {}):
            if sensitivity is None:
                raise ValueError("physics inputs need the sensitivity matrix (build-sensitivity)")
            from leaksense.models.baselines import signatures

            self.signatures = {lv: signatures(net, sensitivity, lv) for lv in S_LEVELS}

    @staticmethod
    def wants_sensitivity(model_cfg: dict) -> bool:
        """True if any setting (or grid value) of ``physics`` is not ``none``."""
        p = model_cfg.get("physics", "none")
        return any(v != "none" for v in (p if isinstance(p, list) else [p]))

    def _make_net(self, cfg: GATConfig, d_in: int) -> nn.Module:
        S = None if cfg.physics == "none" else self.signatures[cfg.s_levels]
        return _GATNet(self.graph, d_in, cfg, S)

    def fit(self, train, windows) -> None:
        torch.set_num_threads(self.base.num_threads)
        super().fit(train, windows)
        params = self.net.physics_params()
        if params:
            self.history[-1]["physics_params"] = {k: round(v, 4) for k, v in params.items()}

    def batch_scores(self, xs: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
        torch.set_num_threads(self.cfg.num_threads)
        return super().batch_scores(xs, mask)
