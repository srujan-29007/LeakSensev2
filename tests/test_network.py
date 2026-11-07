import os

import numpy as np
import pytest

from leaksense.config import DATA_DIR_VAR
from leaksense.data.inp import read_network
from leaksense.network import (
    build_network,
    hazen_williams_resistance,
    headloss_m,
    load_nominal_network,
)

from synthetic_data import DEFAULT_SPECS, NOMINAL_PIPES, contract_for

# Hand-computed: 130^1.852 = 8222.8636, 1.016^4.871 = 1.0803868, 0.8^4.871 = 0.3372495.
R_HANOI_PIPE1 = 10.667 * 100 / (8222.8636 * 1.0803868)  # = 0.120072
R_SYNTH_PIPE2 = 10.667 * 1000 / (8222.8636 * 0.3372495)  # = 3.84652


def test_resistance_hand_computed():
    assert hazen_williams_resistance(100, 1016, 130) == pytest.approx(0.120072, rel=1e-5)
    assert hazen_williams_resistance(100, 1016, 130) == pytest.approx(R_HANOI_PIPE1, rel=1e-6)
    # 1000 L/s = 1 m^3/s, so the headloss equals the resistance.
    assert headloss_m(0.12, 1000.0) == pytest.approx(0.12)
    assert headloss_m(0.12, -500.0) == pytest.approx(0.12 * 0.5**1.852)


def test_synthetic_nominal_graph(synthetic):
    data_dir, contract = synthetic
    net = load_nominal_network(data_dir, contract)
    assert net.node_ids == ["1", "2", "3", "4", "5"]
    assert net.pipe_ids == list(NOMINAL_PIPES)
    assert net.sensor_ids == ["2", "3", "4", "5"]
    assert net.leak_pipe_ids == ["2", "3", "4", "5"]
    assert net.is_connected and net.n_loops == 1
    p2 = net.pipes.set_index("id").loc["2"]
    assert p2["resistance"] == pytest.approx(R_SYNTH_PIPE2, rel=1e-5)

    # Edge index: both directions, pipe e at columns e and e + P.
    P = len(net.pipes)
    assert net.edge_index.shape == (2, 2 * P) and net.edge_attr.shape == (2 * P, 4)
    assert (net.edge_index[:, :P] == net.edge_index[::-1, P:]).all()
    assert tuple(net.edge_index[:, 0]) == (0, 1)  # pipe 1: node 1 -> node 2
    assert net.edge_attr[1, 1] == pytest.approx(0.8)  # diameter in m

    # Network distance (metres) on the square 2-3-4-5-2 plus the reservoir stub.
    idx = net.node_index
    assert net.node_dist_m[idx["2"], idx["4"]] == pytest.approx(1800.0)  # 2-3-4, not 2-5-4
    assert net.node_dist_m[idx["1"], idx["5"]] == pytest.approx(1300.0)
    assert np.allclose(net.node_dist_m, net.node_dist_m.T)
    assert net.node_hops[idx["1"], idx["4"]] == 3  # 1-2-3-4 (or 1-2-5-4)
    assert net.node_hops[idx["2"], idx["2"]] == 0

    # Line-graph hops: pipe 1 (1-2) touches pipes 2 and 5; pipe 3 (3-4) is two hops away.
    pi = net.pipe_index
    assert net.pipe_hops[pi["1"], pi["2"]] == 1
    assert net.pipe_hops[pi["1"], pi["3"]] == 2
    assert net.pipe_hops[pi["3"], pi["3"]] == 0


def test_scenario_inp_gives_same_topology(synthetic):
    """A scenario .inp with split pipes merges back to the nominal topology."""
    data_dir, contract = synthetic
    nominal = load_nominal_network(data_dir, contract)
    scen = build_network(read_network(data_dir / "Scenario-4" / "Hanoi_Scenario-4.inp"), contract)
    assert scen.node_ids == nominal.node_ids
    assert scen.pipe_ids == nominal.pipe_ids
    assert (scen.edge_index == nominal.edge_index).all()
    assert (scen.pipe_hops == nominal.pipe_hops).all()


def test_disconnected_graph_detected(synthetic, tmp_path):
    data_dir, contract = synthetic
    text = (data_dir / "Hanoi.inp").read_text()
    # Drop pipes 3 and 4 (node 4 is then isolated).
    lines = [ln for ln in text.splitlines() if not ln.startswith((" 3 3 4", " 4 4 5"))]
    path = tmp_path / "cut.inp"
    path.write_text("\n".join(lines))
    net = build_network(read_network(path), contract_for(DEFAULT_SPECS))
    assert len(net.pipes) == 3
    assert not net.is_connected


@pytest.mark.data
def test_real_hanoi_network():
    net = load_nominal_network(os.environ[DATA_DIR_VAR])
    assert len(net.nodes) == 32 and len(net.pipes) == 34
    assert net.is_connected and net.n_loops == 3
    assert net.sensor_ids == [str(i) for i in range(2, 33)]
    assert net.leak_pipe_ids == [str(i) for i in range(2, 35)]
    p = net.pipes.set_index("id")
    assert (p.loc["1", "start"], p.loc["1", "end"]) == ("1", "2")
    assert p.loc["1", "resistance"] == pytest.approx(R_HANOI_PIPE1, rel=1e-5)
    # Pipe 22 (21 -> 22, 500 m, 304.8 mm): 0.3048^4.871 = 0.00306646.
    assert p.loc["22", "resistance"] == pytest.approx(
        10.667 * 500 / (8222.8636 * 0.00306646), rel=1e-5
    )
    idx = net.node_index
    assert net.node_dist_m[idx["1"], idx["3"]] == pytest.approx(100 + 1350)
