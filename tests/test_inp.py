import pytest

from leaksense.data.inp import (
    InpNetwork,
    Pipe,
    SplitPipeError,
    merge_split_pipes,
    read_inp_sections,
    read_network,
)

from synthetic_data import NOMINAL_PIPES


def test_head_tail_parse_matches_full_parse(synthetic):
    data_dir, _ = synthetic
    path = data_dir / "Scenario-3" / "Hanoi_Scenario-3.inp"
    fast = read_inp_sections(path, tail_bytes=512)  # forces the head/tail path
    full = read_inp_sections(path, tail_bytes=10**9)  # small-file path: full parse
    for section in ("[JUNCTIONS]", "[PIPES]", "[COORDINATES]", "[OPTIONS]", "[TIMES]"):
        assert fast[section] == full[section]
    assert not fast.get("[PATTERNS]")  # pattern bodies are never parsed


def test_nominal_network(synthetic):
    data_dir, _ = synthetic
    net = read_network(data_dir / "Hanoi.inp")
    assert set(net.pipes) == set(NOMINAL_PIPES)
    assert net.options["UNITS"] == "LPS"
    assert net.pipes["2"].length == 1000.0


def test_merge_single_and_double_split(synthetic):
    data_dir, _ = synthetic
    m2 = merge_split_pipes(read_network(data_dir / "Scenario-2" / "Hanoi_Scenario-2.inp", 512))
    assert m2["3"].pieces == ["3", "3l"] and (m2["3"].start, m2["3"].end) == ("3", "4")
    assert m2["3"].leak_fraction("leak_node0") == pytest.approx(0.25)

    m4 = merge_split_pipes(read_network(data_dir / "Scenario-4" / "Hanoi_Scenario-4.inp", 512))
    assert m4["4"].pieces == ["4", "4l", "4ll"]
    assert m4["4"].leak_nodes == ["leak_node0", "leak_node1"]
    assert (m4["4"].start, m4["4"].end) == ("4", "5")


def _net(pipes):
    return InpNetwork({}, {}, [], [], [], {p.id: p for p in pipes}, {}, {}, {})


def test_merge_rejects_broken_chain():
    with pytest.raises(SplitPipeError):
        merge_split_pipes(
            _net([Pipe("7", "a", "leak_node0", 1, 1, 1), Pipe("7l", "x", "b", 1, 1, 1)])
        )
    with pytest.raises(SplitPipeError):
        merge_split_pipes(_net([Pipe("7", "a", "leak_node0", 1, 1, 1)]))
