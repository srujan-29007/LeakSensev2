import json

import pytest

ABL = """model:
  kind: gat
  max_epochs: 2
  hidden: 8
  heads: 2
  layers: 2
  node_embed: 2
  val_frac: 0.0
  num_threads: 1
ablation:
  seeds: [0, 1]
  reference: [sigmatch_all]
  variants:
    gat: {}
    gat_noembed: {node_embed: 0}
    gat_sprior: {physics: prior}
    gat_sedge: {physics: edge}
"""


@pytest.fixture
def abl_env(cli_env):
    """cli_env plus features, S and an ablation config; returns (tmp_path, ablate args)."""
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    sens = ["build-sensitivity", *common, "--splits", str(tmp_path / "splits.json")]
    assert main([*sens, "--out", str(tmp_path / "sens"), "--quiet"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(cfg.read_text() + ABL)
    args = ["ablate", "--k", "2,4", "--draws", "2", "--bootstrap", "20"]
    args += ["--splits", str(tmp_path / "splits.json"), "--reports-dir", str(tmp_path / "rep")]
    args += ["--runs-dir", str(tmp_path / "runs"), *(common[:4] + common[6:])]
    return tmp_path, args


def test_ablate_runs_resumes_and_aggregates(abl_env):
    from leaksense.cli import main

    tmp_path, args = abl_env
    assert main([*args, "--variants", "gat,gat_sprior", "--seeds", "0", "--no-aggregate"]) == 0
    runs = tmp_path / "runs"
    first = sorted(p.name for p in runs.glob("*.json"))
    assert first == ["t__gat_s0.json", "t__gat_sprior_s0.json", "t__ref_sigmatch_all.json"]
    stamp = (runs / "t__gat_s0.json").stat().st_mtime_ns

    assert main(args) == 0  # the rest of the variants and seeds, then aggregate
    assert (runs / "t__gat_s0.json").stat().st_mtime_ns == stamp  # done runs are skipped
    assert len(list(runs.glob("*.parquet"))) == 4 * 2 + 1

    rep = json.loads((tmp_path / "rep" / "t.json").read_text())
    assert list(rep["variants"]) == [
        "gat",
        "gat_noembed",
        "gat_sprior",
        "gat_sedge",
        "sigmatch_all",
    ]
    assert rep["variants"]["gat"]["seeds"] == [0, 1]
    assert rep["variants"]["sigmatch_all"]["kind"] == "reference"
    ov = rep["variants"]["gat"]["results"]["2"]["overall"]["top3"]
    assert (
        ov["seed_min"] <= ov["value"] <= ov["seed_max"] or abs(ov["value"] - ov["seed_min"]) < 1e-9
    )
    assert set(rep["comparisons"]) == {"gat", "sigmatch_all"}
    assert "gat" not in rep["comparisons"]["gat"]
    d = rep["comparisons"]["gat"]["gat_sprior"]["4"]["overall"]["top3"]
    assert d["lo"] <= d["value"] <= d["hi"]
    h2 = rep["verdicts"]["H2_physics_vs_plain_top3_k_le_6"]
    assert set(h2) == {"gat_sprior", "gat_sedge"} and set(h2["gat_sprior"]["per_k"]) == {"2", "4"}
    assert rep["fingerprint"]["draws"] == 2 and rep["fingerprint"]["folds"] is None
    md = (tmp_path / "rep" / "t.md").read_text(encoding="utf-8")
    assert "Paired top-3 difference vs `gat`" in md and "H2" in md


def test_ablate_refuses_mixed_runs(abl_env, capsys):
    from leaksense.cli import main

    tmp_path, args = abl_env
    assert main([*args, "--variants", "gat", "--seeds", "0", "--no-aggregate"]) == 0
    mixed = [a if a != "2" else "3" for a in args]  # --draws 3: different sensor sets
    assert main([*mixed, "--aggregate-only"]) == 2  # config error
    assert "different ['draws']" in capsys.readouterr().err


def test_ablation_config_does_not_change_feature_hash(tmp_path):
    from leaksense.runconfig import load_run_config

    a = tmp_path / "a.yaml"
    a.write_text("name: a\nfeatures:\n  slot: tod\n")
    b = tmp_path / "b.yaml"
    b.write_text("name: b\nfeatures:\n  slot: tod\n" + ABL)
    assert load_run_config(a).hash() == load_run_config(b).hash()


def test_spec_errors(tmp_path):
    from leaksense.ablate import parse_spec
    from leaksense.config import ConfigError
    from leaksense.runconfig import RunConfig

    with pytest.raises(ConfigError, match="variants"):
        parse_spec(RunConfig())
    with pytest.raises(ConfigError, match="baseline"):
        parse_spec(RunConfig(ablation={"variants": {"gat_x": {}}}))
    with pytest.raises(ConfigError, match="seed"):
        parse_spec(RunConfig(ablation={"variants": {"gat": {"seed": 3}}}))
    s = parse_spec(RunConfig(ablation={"variants": {"gat": {}}, "folds": [0]}))
    assert s.folds == [0] and s.seeds == [0] and s.reference == ["sigmatch_all"]
