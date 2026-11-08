import json

import numpy as np
import pandas as pd
import pytest
import torch

from leaksense.models.gat import (
    GATConfig,
    MaskedGAT,
    build_graph,
    masked_cosine,
    standardised_pipe_features,
)
from leaksense.models.mlp import encode, parse_config, random_k_masks, split_encoded
from leaksense.network import load_nominal_network


@pytest.fixture
def net(synthetic):
    """Synthetic graph: reservoir 1, junctions 2-5, pipes 1-5 (pipe 1 never leaks)."""
    data_dir, contract = synthetic
    return load_nominal_network(data_dir, contract)


def _cases(net, ys, n_windows=6, noise=0.05, seed=0):
    """One case per label: each class has its own ΔH pattern over the junctions plus noise."""
    j, c = len(net.sensor_ids), len(net.leak_pipe_ids)
    proto = -np.abs(np.random.default_rng(123).normal(0, 3, size=(c, j)))
    rng = np.random.default_rng(seed)
    cases = pd.DataFrame(
        {"case_id": [f"c{i}" for i in range(len(ys))], "scenario_id": np.arange(len(ys)) + 2}
    )
    cases["y"] = np.asarray(ys)
    windows = [proto[y] + rng.normal(0, noise, size=(n_windows, j)) for y in ys]
    return cases, windows


def toy_sensitivity(net, seed=7):
    """S-like dict for the synthetic net: (junctions x pipes x 2 sizes x 3 levels), all <= 0."""
    j, c = len(net.sensor_ids), len(net.leak_pipe_ids)
    S = -np.abs(np.random.default_rng(seed).normal(1, 0.5, (j, c, 2, 3)))
    return {
        "S": S,
        "pipes": np.array(net.leak_pipe_ids),
        "junctions": np.array(net.sensor_ids),
        "levels": np.array(["low", "mid", "high"]),
    }


def _gat(net, **kw):
    small = {"hidden": 16, "heads": 2, "layers": 2, "node_embed": 4, "val_frac": 0.0}
    cfg = {**small, "num_threads": 1, **kw}
    sens = toy_sensitivity(net) if MaskedGAT.wants_sensitivity(cfg) else None
    return MaskedGAT(net, cfg, sensitivity=sens)


def test_graph_tensors(net):
    g = build_graph(net)
    assert g.n_nodes == 5 and g.edge_index.shape == (2, 10) and g.edge_attr.shape == (10, 4)
    assert len(g.pipe_u) == len(net.leak_pipe_ids) == 4  # pipe 1 excluded
    assert 0 not in g.sensor_pos.tolist()  # reservoir (position 0) is never a sensor
    f = standardised_pipe_features(net)
    assert np.isfinite(f).all() and np.allclose(f[:, 2], 0)  # constant roughness -> 0
    assert np.allclose(f[:, 0].mean(), 0) and np.allclose(f[:, 0].std(), 1)


def test_output_shapes(net):
    cases, windows = _cases(net, [0, 1, 2, 3])
    m = _gat(net, max_epochs=2)
    m.fit(cases, windows)
    s = m.batch_scores(windows, np.ones(4, bool))
    assert s.shape == (4, 4) and np.isfinite(s).all()
    assert (np.logaddexp.reduce(s, axis=1) <= 1e-6).all()  # mean of log-probs
    single = np.stack([m.case_scores(w, np.ones(4, bool)) for w in windows])
    assert np.allclose(s, single, atol=1e-5)


# every M10 variant: (overrides, needs S)
VARIANTS = {
    "gat": {},
    "gat_noembed": {"node_embed": 0},
    "gat_snode": {"physics": "node"},
    "gat_sedge": {"physics": "edge"},
    "gat_sprior": {"physics": "prior"},
    "gat_sprior_mid": {"physics": "prior", "s_levels": "mid"},
}
MASK_CASES = [
    (v, t)
    for v in VARIANTS
    for t in ("scale_magnitude", "none", "signed_log1p")
    if not (t == "signed_log1p" and VARIANTS[v].get("physics"))
]


@pytest.mark.parametrize(("variant", "transform"), MASK_CASES)
def test_masked_junction_cannot_change_output(net, variant, transform):
    cases, windows = _cases(net, [0, 1, 2, 3])
    m = _gat(net, max_epochs=3, transform=transform, **VARIANTS[variant])
    m.fit(cases, windows)
    mask = np.array([True, False, True, False])
    x = windows[0].copy()
    a = m.case_scores(x, mask)
    x[:, ~mask] = 1e3  # garbage in every unsensed junction
    assert np.allclose(a, m.case_scores(x, mask))


def test_reservoir_input_is_zero(net):
    m = _gat(net)
    g = m.graph
    gnet = m._make_net(m.base, 2 * 4 + 1)
    seen = {}
    gnet.inp.register_forward_pre_hook(lambda mod, inp: seen.update(x=inp[0]))
    x = encode(np.full((3, 4), -5.0), np.ones((3, 4), bool), "scale_magnitude")
    gnet(torch.from_numpy(x.astype(np.float32)))
    res = [i for i in range(g.n_nodes) if i not in g.sensor_pos.tolist()]
    assert res == [0] and (seen["x"][:, 0, :2] == 0).all()


@pytest.mark.parametrize("transform", ["scale_magnitude", "none"])
def test_node_and_head_inputs_follow_encode_layout(net, transform):
    """The node mask channel is the mask, and the head gets encode's magnitude column."""
    m = _gat(net, transform=transform)
    g = m.graph
    mask = np.array([[True, False, True, True], [False, True, False, True]])
    x = encode(-np.arange(1.0, 9.0).reshape(2, 4), mask, transform).astype(np.float32)
    gnet = m._make_net(m.base, x.shape[1])
    seen = {}
    gnet.inp.register_forward_pre_hook(lambda mod, inp: seen.update(node=inp[0]))
    gnet.head.register_forward_pre_hook(lambda mod, inp: seen.update(head=inp[0]))
    with torch.no_grad():
        gnet(torch.from_numpy(x))
    node = seen["node"][:, g.sensor_pos].detach()
    assert torch.equal(node[:, :, 1], torch.from_numpy(mask.astype(np.float32)))
    pattern = np.where(mask, encode(-np.arange(1.0, 9.0).reshape(2, 4), mask, transform)[:, :4], 0)
    assert np.allclose(node[:, :, 0].numpy(), pattern)
    if transform == "scale_magnitude":
        mag = torch.from_numpy(x[:, 4:5])  # encode puts the magnitude right after the pattern
        assert torch.equal(seen["head"][:, :, -1], mag.expand(-1, seen["head"].shape[1]))


def test_split_encoded_roundtrip():
    from leaksense.models.mlp import split_encoded

    mask = np.array([[True, False, True]])
    for t, has_mag in (("scale_magnitude", True), ("none", False), ("signed_log1p", False)):
        x = encode(np.array([[-2.0, -9.0, -4.0]]), mask, t)
        p, mag, m = split_encoded(x, 3)
        assert np.array_equal(m, mask.astype(float)) and (mag is not None) == has_mag
        assert p.shape == (1, 3) and p[0, 1] == 0
    with pytest.raises(ValueError):
        split_encoded(np.zeros((1, 5)), 3)


def test_overfits_five_cases(net):
    cases, windows = _cases(net, [0, 1, 2, 3, 1])
    m = _gat(net, k_min=4, dropout=0.0, max_epochs=150, lr=0.005)
    m.fit(cases, windows)
    pred = m.batch_scores(windows, np.ones(4, bool)).argmax(axis=1)
    assert (pred == cases["y"].to_numpy()).all()


def test_fit_is_deterministic(net):
    cases, windows = _cases(net, [0, 1, 2, 3])
    a, b = _gat(net, max_epochs=4), _gat(net, max_epochs=4)
    a.fit(cases, windows)
    b.fit(cases, windows)
    mask = random_k_masks(1, 4, 2, np.random.default_rng(0))[0]
    assert np.array_equal(a.batch_scores(windows, mask), b.batch_scores(windows, mask))


def test_config():
    base, grid = parse_config({"kind": "gat", "hidden": [16, 32], "heads": 4}, GATConfig)
    assert base.hidden == 16 and grid == {"hidden": [16, 32]} and base.layers == 4
    with pytest.raises(ValueError):
        GATConfig(hidden=30, heads=4)
    with pytest.raises(ValueError, match="unknown"):
        parse_config({"hedas": 2}, GATConfig)


def test_cli_train_gat(cli_env):
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        cfg.read_text() + "model:\n  kind: gat\n  max_epochs: 3\n  hidden: 8\n  heads: 2\n"
        "  layers: 2\n  val_frac: 0.0\n  num_threads: 1\n"
    )
    args = common[:4] + common[6:]
    rep = tmp_path / "rep"
    base = ["train", "--k", "2,4", "--draws", "2", "--bootstrap", "20"]
    base += ["--splits", str(tmp_path / "splits.json"), "--reports-dir", str(rep), *args]
    assert main([*base, "--fold", "1"]) == 0
    r = json.loads((rep / "gat_fold1.json").read_text())
    assert r["model"] == "gat" and r["training"]["eval_folds"] == [1]
    assert r["training"]["base"]["heads"] == 2


# ----------------------------------------------------------------------------- physics (M10)


def test_masked_cosine_ignores_unsensed_rows_of_x_and_s():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(3, 5))).float()
    S = torch.from_numpy(-np.abs(rng.normal(size=(5, 4, 6)))).float()
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]).float()
    a = masked_cosine(x, mask, S)
    x2, S2 = x.clone(), S.clone()
    x2[mask == 0] = 1e3
    S2[1] = 50.0  # junction 1 is unsensed in rows 0 and 2
    b = masked_cosine(x2, mask, S2)
    assert torch.allclose(a[[0, 2]], b[[0, 2]])
    assert a.shape == (3, 4, 6) and (a.abs() <= 1 + 1e-6).all()
    assert torch.equal(masked_cosine(torch.zeros(1, 5), mask[:1], S), torch.zeros(1, 4, 6))


@pytest.mark.parametrize("levels", ["all", "mid"])
def test_masked_cosine_equals_sigmatch(net, levels):
    from leaksense.models.baselines import SigMatch, signatures

    sd = toy_sensitivity(net)
    sm = SigMatch(net, sd, levels=levels)
    S = torch.from_numpy(signatures(net, sd, levels)).float()
    rng = np.random.default_rng(1)
    for mask in ([True, False, True, True], [True] * 4, [False, True, False, False]):
        mk = np.array(mask)
        x = -np.abs(rng.normal(2, 1, size=(1, 4))) * mk
        for t in ("scale_magnitude", "none"):
            row = torch.from_numpy(encode(x, mk[None], t).astype(np.float32))
            p, _, m = split_encoded(row, 4)
            cos = masked_cosine(p, m, S)[0].amax(dim=1).numpy()
            assert np.allclose(cos, sm.case_scores(x, mk), atol=1e-5)


def test_prior_with_zero_head_ranks_like_sigmatch_all(net):
    from leaksense.models.baselines import SigMatchAll

    m = _gat(net, physics="prior")
    gnet = m._make_net(m.base, 2 * 4 + 1).eval()
    with torch.no_grad():
        gnet.head[-1].weight.zero_()
        gnet.head[-1].bias.zero_()
    ref = SigMatchAll(net, toy_sensitivity(net))
    rng = np.random.default_rng(2)
    for _ in range(5):
        mk = rng.random(4) < 0.7
        mk[0] = True
        x = -np.abs(rng.normal(2, 1, size=(1, 4))) * mk
        with torch.no_grad():
            lg = gnet(torch.from_numpy(encode(x, mk[None], "scale_magnitude").astype(np.float32)))
        assert np.allclose(lg[0].numpy() / 10.0, ref.case_scores(x, mk), atol=1e-5)


def test_physics_none_ignores_sensitivity(net):
    cases, windows = _cases(net, [0, 1, 2, 3])
    a = MaskedGAT(
        net,
        {"hidden": 16, "heads": 2, "layers": 2, "val_frac": 0.0, "max_epochs": 3, "num_threads": 1},
    )
    b = MaskedGAT(
        net,
        {"hidden": 16, "heads": 2, "layers": 2, "val_frac": 0.0, "max_epochs": 3, "num_threads": 1},
        sensitivity=toy_sensitivity(net),
    )
    a.fit(cases, windows)
    b.fit(cases, windows)
    mask = np.array([True, False, True, True])
    assert np.array_equal(a.batch_scores(windows, mask), b.batch_scores(windows, mask))
    assert "physics_params" not in b.history[-1]


@pytest.mark.parametrize("physics", ["node", "edge", "prior"])
def test_physics_variants_train_and_record_params(net, physics):
    cases, windows = _cases(net, [0, 1, 2, 3])
    m = _gat(net, max_epochs=3, physics=physics)
    m.fit(cases, windows)
    s = m.batch_scores(windows, np.array([True, True, False, True]))
    assert s.shape == (4, 4) and np.isfinite(s).all()
    if physics in ("node", "prior"):
        assert m.history[-1]["physics_params"]


def test_physics_config_errors(net):
    with pytest.raises(ValueError, match="scale_magnitude or none"):
        GATConfig(physics="prior", transform="signed_log1p")
    with pytest.raises(ValueError, match="physics"):
        GATConfig(physics="attention")
    with pytest.raises(ValueError, match="sensitivity"):
        MaskedGAT(net, {"physics": "edge"})
    assert MaskedGAT.wants_sensitivity({"physics": ["none", "prior"]})
    assert not MaskedGAT.wants_sensitivity({"physics": "none"})
