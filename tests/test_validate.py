import json
import re

import pytest

from leaksense.cli import main
from leaksense.data.validate import validate_dataset

from synthetic_data import DEFAULT_SPECS, LeakSpec, ScenarioSpec, contract_for, make_dataset


def failing(report):
    return {cid for cid, c in report.checks.items() if c.status == "fail"}


@pytest.mark.parametrize("deep", [False, True])
def test_synthetic_dataset_passes(synthetic, deep):
    data_dir, contract = synthetic
    report = validate_dataset(data_dir, contract, deep=deep, inp_tail_bytes=512)
    assert report.ok, {cid: c.errors for cid, c in report.checks.items() if c.errors}
    d8 = report.checks["D8"].details
    assert d8["single_leak_scenarios_per_pipe"] == {"3": 1}
    assert d8["never_leaking_observed"] == ["1", "5"]
    assert d8["double_split_scenarios"] == [4]
    assert report.checks["D10"].status == "info"
    if deep:
        assert set(report.stats["leak_flow_lps_median_per_leak"]) == {"small", "big"}


def test_validation_is_read_only(synthetic):
    data_dir, contract = synthetic
    before = {p: p.stat().st_mtime_ns for p in data_dir.rglob("*")}
    validate_dataset(data_dir, contract, deep=True, inp_tail_bytes=512)
    assert {p: p.stat().st_mtime_ns for p in data_dir.rglob("*")} == before


def _edit(path, old, new, count=1):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, count))


def corrupt_missing_file(d):
    (d / "Scenario-2" / "Link_flows.csv").unlink()


def corrupt_top_label(d):
    _edit(d / "Labels.csv", "\n2,1\n", "\n2,2\n")


def corrupt_extra_column(d):
    _edit(d / "Scenario-2" / "Node_pressures.csv", ",leak_node0,", ",leak_node0,99,")


def corrupt_scenario_label(d):
    _edit(d / "Scenario-2" / "Labels.csv", "2024-01-01 00:00:00,0.0", "2024-01-01 00:00:00,1.0")


def corrupt_truncated_csv(d):
    p = d / "Scenario-2" / "Node_pressures.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")


def corrupt_info_key(d):
    _edit(d / "Scenario-2" / "Leaks" / "Leak_leak_node0_info.csv", "Leak Type", "Type")


def corrupt_pipe_length(d):
    p = d / "Scenario-1" / "Hanoi_Scenario-1.inp"
    text = p.read_text()
    new = re.sub(r"^( 2 2 3 )(\S+)", lambda m: m.group(1) + "5000.0", text, count=1, flags=re.M)
    assert new != text
    p.write_text(new)


def corrupt_coordinates(d):
    p = d / "Scenario-1" / "Hanoi_Scenario-1.inp"
    _edit(p, " 3 100.000000000 300.000000000", " 3 101.000000000 300.000000000")


@pytest.mark.parametrize(
    "corrupt, expected",
    [
        (corrupt_missing_file, "D2"),
        (corrupt_top_label, "D1"),
        (corrupt_extra_column, "D4"),
        (corrupt_scenario_label, "D7"),
        (corrupt_truncated_csv, "D3"),
        (corrupt_info_key, "D6"),
        (corrupt_pipe_length, "D9"),
        (corrupt_coordinates, "D9"),
    ],
)
def test_corruption_is_detected(synthetic, corrupt, expected):
    data_dir, contract = synthetic
    corrupt(data_dir)
    report = validate_dataset(data_dir, contract, inp_tail_bytes=512)
    assert expected in failing(report)


def test_leak_flow_outside_window_needs_deep(synthetic):
    data_dir, contract = synthetic
    p = data_dir / "Scenario-2" / "Leaks" / "Leak_leak_node0_demand.csv"
    _edit(p, "2024-01-01 00:00:00,0.00000", "2024-01-01 00:00:00,5.00000")
    assert "D7" not in failing(validate_dataset(data_dir, contract, inp_tail_bytes=512))
    assert "D7" in failing(validate_dataset(data_dir, contract, deep=True, inp_tail_bytes=512))


def test_nan_pressure_needs_deep(synthetic):
    data_dir, contract = synthetic
    p = data_dir / "Scenario-1" / "Node_pressures.csv"
    lines = p.read_text().splitlines()
    cells = lines[5].split(",")
    cells[2] = ""
    lines[5] = ",".join(cells)
    p.write_text("\n".join(lines) + "\n")
    assert "D5" in failing(validate_dataset(data_dir, contract, deep=True, inp_tail_bytes=512))


def test_leak_on_never_leaking_pipe_fails(tmp_path):
    specs = dict(DEFAULT_SPECS)
    specs[2] = ScenarioSpec([LeakSpec("1", 40, 70)])
    data_dir = make_dataset(tmp_path, specs)
    report = validate_dataset(data_dir, contract_for(specs), inp_tail_bytes=512)
    assert "D8" in failing(report)


def test_cli_writes_report_and_sets_exit_code(synthetic, tmp_path):
    data_dir, contract = synthetic
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract.to_dict()))
    out = tmp_path / "report.json"
    args = [
        "validate-data",
        "--data-dir",
        str(data_dir),
        "--contract",
        str(contract_path),
        "--out",
        str(out),
        "--quiet",
    ]
    assert main(args) == 0
    payload = json.loads(out.read_text())
    assert payload["ok"] is True and len(payload["checks"]) == 10
    assert str(data_dir) not in out.read_text()  # no absolute paths in reports
    assert payload["git"]["commit"] and "seed" in payload  # provenance (CLAUDE.md)
    assert payload["manifest"]["sha256"] and payload["contract"]

    corrupt_missing_file(data_dir)
    assert main(args) == 1


def test_cli_config_error_exit_code(tmp_path, capsys):
    assert main(["validate-data", "--data-dir", str(tmp_path / "missing"), "--quiet"]) == 2
    assert "config error" in capsys.readouterr().err


@pytest.mark.data
def test_real_dataset_matches_contract():
    from leaksense.config import get_data_dir

    report = validate_dataset(get_data_dir())
    assert report.ok, {cid: c.errors[:3] for cid, c in report.checks.items() if c.errors}
