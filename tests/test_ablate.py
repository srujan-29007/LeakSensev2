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
    assert first == [
        "t__gat_s0_f0.json",
        "t__gat_s0_f1.json",
        "t__gat_sprior_s0_f0.json",
        "t__gat_sprior_s0_f1.json",
        "t__ref_sigmatch_all.json",
    ]
    stamp = (runs / "t__gat_s0_f0.json").stat().st_mtime_ns

    assert main(args) == 0  # the rest of the variants and seeds, then aggregate
    assert (runs / "t__gat_s0_f0.json").stat().st_mtime_ns == stamp  # done runs are skipped
    assert len(list(runs.glob("*.parquet"))) == 4 * 2 * 2 + 1  # variants x seeds x folds + ref

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


def test_ablate_resumes_per_fold_and_skips_incomplete(abl_env):
    """A run killed after fold 0 keeps fold 0; aggregation leaves the half-done run out."""
    from leaksense.cli import main

    tmp_path, args = abl_env
    assert main([*args, "--variants", "gat,gat_sprior", "--seeds", "0", "--no-aggregate"]) == 0
    runs = tmp_path / "runs"
    for ext in ("json", "parquet", "md"):
        (runs / f"t__gat_sprior_s0_f1.{ext}").unlink()  # as if killed during fold 1
    stamp = (runs / "t__gat_sprior_s0_f0.json").stat().st_mtime_ns
    assert main([*args, "--aggregate-only"]) == 0
    rep = json.loads((tmp_path / "rep" / "t.json").read_text())
    assert "gat_sprior" not in rep["variants"]
    assert rep["incomplete_runs"] == ["gat_sprior seed 0: folds [0] of [0, 1]"]
    assert main([*args, "--variants", "gat_sprior", "--seeds", "0"]) == 0  # resume: fold 1 only
    assert (runs / "t__gat_sprior_s0_f0.json").stat().st_mtime_ns == stamp
    rep = json.loads((tmp_path / "rep" / "t.json").read_text())
    assert "gat_sprior" in rep["variants"] and rep["incomplete_runs"] == []


def test_ablate_status_and_staged_writes(abl_env, capsys):
    from leaksense.cli import main

    tmp_path, args = abl_env
    runs = tmp_path / "runs"
    assert main([*args, "--variants", "gat", "--seeds", "0", "--no-aggregate"]) == 0
    assert list((runs / ".partial").iterdir()) == []  # everything moved in
    (runs / "t__gat_s0_f1.json").write_text('{"ablation": {"ke')  # cut off mid-write
    capsys.readouterr()
    assert main([*args, "--variants", "gat,gat_sprior", "--seeds", "0", "--status"]) == 0
    out = capsys.readouterr().out
    assert "done  t__ref_sigmatch_all" in out and "done  t__gat_s0_f0" in out
    assert "todo  t__gat_s0_f1" in out and "todo  t__gat_sprior_s0_f0" in out
    assert "2 of 5 runs done" in out
    assert not list(runs.glob("t__gat_sprior*"))  # --status trains nothing


def test_ablate_refuses_to_run_next_to_other_fingerprint(abl_env, capsys):
    from leaksense.cli import main

    tmp_path, args = abl_env
    assert main([*args, "--variants", "gat", "--seeds", "0", "--no-aggregate"]) == 0
    mixed = [a if a != "2" else "3" for a in args]  # --draws 3
    assert main([*mixed, "--variants", "gat_sprior", "--seeds", "0", "--no-aggregate"]) == 2
    assert "different ['draws']" in capsys.readouterr().err
    assert not list((tmp_path / "runs").glob("t__gat_sprior*"))


def _fake_run(variant, seed, top3, small8, physics="prior", s_levels="all", n=10):
    import numpy as np

    from leaksense.ablate import Run

    ie = {
        str(k): {"top3": top3, "n": n, "small_top3": small8, "n_small": 4} for k in (2, 3, 4, 6, 8)
    }
    model = {"physics": physics, "s_levels": s_levels}
    rep = {"provenance": {"config": {"model": model}}, "training": {"folds": [{"inner_eval": ie}]}}
    return Run(variant, seed, "model", [rep, rep], np.zeros((1, 1, 5)))


def test_stage2_uses_inner_validation_and_tie_breaks():
    from leaksense.ablate import AblationSpec, stage2_selection

    runs = {
        "gat_sedge": [_fake_run("gat_sedge", 0, 0.70, 0.60, "edge")],
        "gat_sprior": [_fake_run("gat_sprior", 0, 0.70, 0.60, "prior")],  # ties sedge: simpler
        "gat_snode": [_fake_run("gat_snode", 0, 0.70, 0.65, "node")],  # tie broken by small
        "gat_sprior_mid": [_fake_run("gat_sprior_mid", 0, 0.60, 0.9, "prior", "mid")],
        "gat_snode_noembed": [_fake_run("gat_snode_noembed", 1, 0.99, 0.99, "node")],  # no seed 0
    }
    variants = {v: {"kind": "model", "model": r[0].model} for v, r in runs.items()}
    variants["gat"] = {"kind": "model", "model": {"physics": "none"}}
    variants["gat_noembed"] = {"kind": "model", "model": {"physics": "none"}}
    spec = AblationSpec({}, [0], None, [], "gat")
    s2 = stage2_selection(spec, variants, runs)
    assert s2["ranked"] == ["gat_snode", "gat_sprior", "gat_sedge", "gat_sprior_mid"]
    assert s2["final_variants"] == ["gat", "gat_noembed", "gat_snode", "gat_sprior"]
    assert s2["ineligible_no_inner_eval"] == ["gat_snode_noembed"]


def test_noembed_adoption_rule():
    from leaksense.ablate import noembed_adoption

    def comp(lo8, points):
        per_k = {
            str(k): {"overall": {"top3": {"value": x, "lo": lo8 if k == 8 else -1, "hi": 1}}}
            for k, x in points.items()
        }
        return {"gat": {"gat_noembed": per_k}}

    ks = [2, 8, 31]
    v3 = {"gat": {"seeds": [0, 1, 2]}, "gat_noembed": {"seeds": [0, 1, 2]}}
    good = noembed_adoption(None, v3, comp(0.01, {2: 0.05, 8: 0.04, 31: 0.0}), ks)
    assert good["decision"] == "adopt node_embed: 0"
    assert (
        noembed_adoption(None, v3, comp(-0.01, {2: 0.05, 8: 0.04, 31: 0.0}), ks)["decision"]
        == "keep the embedding"
    )
    assert (
        noembed_adoption(None, v3, comp(0.01, {2: 0.05, 8: 0.04, 31: -0.001}), ks)["decision"]
        == "keep the embedding"
    )
    v1 = {"gat": {"seeds": [0]}, "gat_noembed": {"seeds": [0]}}
    assert noembed_adoption(None, v1, comp(0.01, {8: 0.1}), ks)["decision"] == "pending"
