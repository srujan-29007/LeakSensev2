"""M14: export-serve bundle and the FastAPI endpoints, on the synthetic dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FINAL

WEB = Path(__file__).resolve().parents[1] / "web"


def run_final(tmp_path, args) -> list[str]:
    """M11 + M13 on the synthetic data; returns the common args for final-report/export-serve."""
    from leaksense.cli import main

    assert main(args) == 0
    base = (tmp_path / "cfg.yaml").read_text().split("model:")[0]
    cfg = tmp_path / "final.yaml"
    final = FINAL.replace("RUNS", str(tmp_path / "runs")).replace("REP", str(tmp_path / "rep"))
    cfg.write_text(base + final)
    i = args.index("--config")
    common = args[1:i] + ["--config", str(cfg)] + args[i + 2 :]
    j = common.index("--runs-dir")
    common[j + 1] = str(tmp_path / "final_runs")
    k = common.index("--reports-dir")
    common[k + 1] = str(tmp_path / "rep_final")
    assert main(["final-report", *common, "--confirm-locked-test"]) == 0
    return common


@pytest.fixture
def served(place_env):
    """The exported bundle served by a TestClient; returns (tmp_path, client)."""
    from leaksense.api import create_app
    from leaksense.cli import main

    tmp_path, args = place_env
    common = run_final(tmp_path, args)
    bundle = tmp_path / "bundle.pt"
    assert main(["export-serve", *common, "--out", str(bundle)]) == 0
    return tmp_path, TestClient(create_app(bundle, tmp_path / "rep_final", WEB))


def test_api_endpoints_serve_saved_data(served):
    tmp_path, c = served
    rep = json.loads((tmp_path / "rep_final" / "final.json").read_text())

    net = c.get("/network").json()
    assert len(net["sensors"]) == 4 and len(net["pipes"]) >= len(net["nodes"]) - 1
    assert all("length_m" in p for p in net["pipes"])

    cases = c.get("/cases").json()["cases"]
    assert [x["case_id"] for x in cases] and all(x["scenario_id"] == 4 for x in cases)

    pl = c.get("/placements", params={"k": 2}).json()
    assert pl["greedy"] == rep["placement"]["final_order"][:2]
    want = rep["placement"]["results"]["model"]["2"]["overall"]["greedy"]["top3"]["value"]
    assert pl["scores"]["model"]["overall"]["greedy"] == pytest.approx(want)
    assert c.get("/placements", params={"k": 99}).status_code == 422

    cur = c.get("/curves").json()
    assert cur["headline"] == rep["headline"]
    got = cur["ladder"]["gat_snode"]["overall"]
    assert [r["k"] for r in got] == rep["ks"]
    want3 = rep["results"]["gat_snode"]["1"]["overall"]["top3"]["value"]
    assert got[0]["top3"] == pytest.approx(want3)

    r = c.get("/")
    assert r.status_code == 200 and "LeakSense" in r.text


def test_predict_matches_the_saved_m13_scores(served):
    import pandas as pd

    tmp_path, c = served
    order = c.get("/placements").json()["greedy_order"]
    case = c.get("/cases").json()["cases"][0]
    res = c.post("/predict", json={"case_id": case["case_id"], "sensors": order[:2], "top": 3})
    assert res.status_code == 200
    out = res.json()
    assert out["k"] == 2 and len(out["ranking"]) == 3 and out["true_pipe"] == case["pipe"]
    ranks = [r["rank"] for r in out["ranking"]]
    assert ranks == sorted(ranks) and 0 <= out["top3"] <= 1
    # the same case and sensor set as the M13 headline unit's greedy row
    df = pd.read_parquet(tmp_path / "final_runs" / "ml__gat_snode_s0.parquet")
    row = df[(df["method"] == "greedy") & (df["k"] == 2) & (df["case_id"] == case["case_id"])]
    assert out["top3"] == pytest.approx(float(row["top3"].iloc[0]))

    sm = c.post(
        "/predict",
        json={"case_id": case["case_id"], "sensors": order[:2], "method": "sigmatch_all"},
    )
    assert sm.status_code == 200 and np.isfinite(sm.json()["ranking"][0]["score"])


def test_predict_rejects_bad_input(served):
    _, c = served
    case = c.get("/cases").json()["cases"][0]["case_id"]
    sensors = c.get("/network").json()["sensors"]
    assert c.post("/predict", json={"case_id": "nope", "sensors": sensors[:1]}).status_code == 404
    assert c.post("/predict", json={"case_id": case, "sensors": ["1"]}).status_code == 422
    assert c.post("/predict", json={"case_id": case, "sensors": []}).status_code == 422
    dup = {"case_id": case, "sensors": [sensors[0], sensors[0]]}
    assert c.post("/predict", json=dup).status_code == 422
    bad = {"case_id": case, "sensors": sensors[:1], "method": "oracle"}
    assert c.post("/predict", json=bad).status_code == 422


def test_export_refuses_a_model_that_differs_from_m13(place_env):
    import pandas as pd

    from leaksense.cli import main

    tmp_path, args = place_env
    common = run_final(tmp_path, args)
    p = tmp_path / "final_runs" / "ml__gat_snode_s0.parquet"
    df = pd.read_parquet(p)
    df.loc[df["method"] == "greedy", "top3"] = -1.0  # not what the refit model gives
    df.to_parquet(p, index=False)
    assert main(["export-serve", *common, "--out", str(tmp_path / "b.pt")]) != 0
    assert not (tmp_path / "b.pt").exists()
