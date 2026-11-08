"""Masked GAT rung (PLAN.md M9).

Same protocol as the masked MLP (``leaksense.models.mlp``): the input row per window is
``encode(ΔH, mask)`` = ``[T(ΔH·m), m]`` over the 31 junctions (plus one magnitude column for
``scale_magnitude``), sensor masks with k ~ U{k_min..31}, class x case sample weights, inner
validation for list-valued settings and early stopping, then a refit. Only the network differs:

- The row is scattered onto the nominal graph: each junction gets ``[T(ΔH·m), m]``; the
  reservoir always gets ``[0, 0]`` (its pressure is always 0, D5, and it is never a sensor).
  An optional learned per-node embedding gives the shared GAT weights node identity.
- 4 GATv2 layers (pre-norm residual) with nominal edge features: log length, diameter,
  roughness and Hazen-Williams resistance, z-scored over the pipes.
- Pipe head: for each leak-eligible pipe (pipe 1 excluded), an MLP on
  ``[h_u + h_v, |h_u - h_v|, edge features, magnitude]`` gives one logit (symmetric in u, v).

Everything here comes from the nominal ``Hanoi.inp``; no scenario parameters reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from leaksense.models.mlp import MaskedMLP, MLPConfig, split_encoded
from leaksense.network import EDGE_FEATURES, Network


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

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.heads < 1 or self.hidden % self.heads:
            raise ValueError("hidden must be a positive multiple of heads")
        if self.node_embed < 0 or self.num_threads < 1:
            raise ValueError("node_embed must be >= 0 and num_threads >= 1")


@dataclass(frozen=True)
class Graph:
    """Fixed nominal graph tensors shared by every window."""

    n_nodes: int
    sensor_pos: torch.Tensor  # (J,) node position of each feature column
    edge_index: torch.Tensor  # (2, 2P) both directions
    edge_attr: torch.Tensor  # (2P, 4) standardised nominal edge features
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
    t = torch.as_tensor
    return Graph(
        n_nodes=len(net.nodes),
        sensor_pos=t(sensor_pos, dtype=torch.long),
        edge_index=t(net.edge_index, dtype=torch.long),
        edge_attr=t(np.concatenate([feats, feats]), dtype=torch.float32),
        pipe_u=t(net.pipes["u"].to_numpy()[leak], dtype=torch.long),
        pipe_v=t(net.pipes["v"].to_numpy()[leak], dtype=torch.long),
        pipe_attr=t(feats[leak], dtype=torch.float32),
    )


class _GATNet(nn.Module):
    def __init__(self, g: Graph, d_in: int, cfg: GATConfig) -> None:
        super().__init__()
        j = len(g.sensor_pos)
        if d_in not in (2 * j, 2 * j + 1):
            raise ValueError(f"input width {d_in} does not match {j} junctions")
        self.g, self.j, self.has_mag = g, j, d_in == 2 * j + 1
        h = cfg.hidden
        self.embed = nn.Embedding(g.n_nodes, cfg.node_embed) if cfg.node_embed else None
        self.inp = nn.Linear(2 + cfg.node_embed, h)
        e_dim = g.edge_attr.shape[1]
        self.convs = nn.ModuleList(
            GATv2Conv(h, h // cfg.heads, heads=cfg.heads, edge_dim=e_dim) for _ in range(cfg.layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(cfg.layers))
        self.drop = nn.Dropout(cfg.dropout)
        d_head = 2 * h + e_dim + int(self.has_mag)
        self.head = nn.Sequential(nn.Linear(d_head, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, j, b, n = self.g, self.j, x.shape[0], self.g.n_nodes
        pattern, mag, mask = split_encoded(x, j)
        node = x.new_zeros(b, n, 2)  # reservoir (and any non-sensor node) stays [0, 0]
        node[:, g.sensor_pos, 0] = pattern
        node[:, g.sensor_pos, 1] = mask
        if self.embed is not None:
            node = torch.cat([node, self.embed.weight.expand(b, -1, -1)], dim=2)
        h = self.inp(node).reshape(b * n, -1)
        offs = torch.arange(b, device=x.device).repeat_interleave(g.edge_index.shape[1]) * n
        ei = g.edge_index.repeat(1, b) + offs
        ea = g.edge_attr.repeat(b, 1)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = h + self.drop(nn.functional.gelu(conv(norm(h), ei, ea)))
        h = h.view(b, n, -1)
        hu, hv = h[:, g.pipe_u], h[:, g.pipe_v]
        parts = [hu + hv, (hu - hv).abs(), g.pipe_attr.expand(b, -1, -1)]
        if mag is not None:
            parts.append(mag.unsqueeze(1).expand(-1, hu.shape[1], -1))
        return self.head(torch.cat(parts, dim=2)).squeeze(2)


class MaskedGAT(MaskedMLP):
    """Masked GATv2 over the nominal graph; see the module docstring."""

    name = "gat"
    config_cls = GATConfig

    def __init__(self, net: Network, model_cfg: dict | None = None) -> None:
        super().__init__(net, model_cfg)
        self.graph = build_graph(net)
        if len(self.graph.pipe_u) != self.n:
            raise ValueError("leak pipe count mismatch")

    def _make_net(self, cfg: GATConfig, d_in: int) -> nn.Module:
        return _GATNet(self.graph, d_in, cfg)

    def fit(self, train, windows) -> None:
        torch.set_num_threads(self.base.num_threads)
        super().fit(train, windows)

    def batch_scores(self, xs: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
        torch.set_num_threads(self.cfg.num_threads)
        return super().batch_scores(xs, mask)
