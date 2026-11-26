"""FastAPI app for M14 (``leaksense serve``).

Serves only saved data:

- ``/network``: nodes and pipes of the nominal network. Coordinates are for drawing only;
  every distance comes from pipe lengths.
- ``/placements?k=``: the M11 final placement (greedy and coverage orders) and its locked-test
  scores from ``reports/final.json``.
- ``/cases``: the held-out (locked-test) cases in the bundle from ``export-serve``.
- ``/predict``: case + sensor set -> ranked pipes, from the saved model on the cached ΔH windows.
- ``/curves``: ladder, placement and leak-flow curves from ``reports/final.json``.
- ``/``: the page ``web/index.html``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

LADDER_RUNGS = (
    "chance",
    "maxdh",
    "sigmatch_all",
    "mlp",
    "gbm",
    "gat",
    "gat_snode",
    "gat_snode_noembed",
)
CURVE_GROUPS = ("overall", "leak_type=small", "leak_type=big")


def clean(x: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def load_bundle(path: str | Path) -> dict:
    import torch

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no serve bundle at {p}; run `leaksense export-serve` first")
    # our own bundle (pickled models and tables), written by export-serve
    return torch.load(p, weights_only=False)


class PredictRequest(BaseModel):
    case_id: str
    sensors: list[str] = Field(min_length=1)
    method: str | None = None  # default: the headline model
    top: int = Field(default=5, ge=1, le=33)


def _value(d: dict | None) -> float | None:
    return None if not d else d.get("value")


def _ladder(rep: dict) -> dict:
    out = {}
    for rung in LADDER_RUNGS:
        res = rep["results"].get(rung)
        if not res:
            continue
        out[rung] = {
            g: [
                {
                    "k": k,
                    "top3": _value(res[str(k)].get(g, {}).get("top3")),
                    "lo": res[str(k)].get(g, {}).get("top3", {}).get("lo"),
                    "hi": res[str(k)].get(g, {}).get("top3", {}).get("hi"),
                }
                for k in rep["ks"]
            ]
            for g in CURVE_GROUPS
        }
    return out


def _placement_rows(pl: dict, scorer: str, k: int) -> dict | None:
    r = pl["results"].get(scorer, {}).get(str(k))
    if r is None:
        return None
    out = {}
    for g in CURVE_GROUPS:
        if g not in r:
            continue
        x = r[g]
        out[g] = {
            "greedy": _value(x["greedy"]["top3"]),
            "coverage": _value(x["coverage"]["top3"]),
            "random_mean": _value(x["random_mean"]["top3"]),
            "random_p5": x["random_top3_draws"]["p5"],
            "random_p95": x["random_top3_draws"]["p95"],
            "greedy_percentile": x.get("greedy_percentile"),
            "n": x["greedy"].get("n"),
        }
    return out


def create_app(
    bundle_path: str | Path | None = None,
    reports_dir: str | Path = "reports",
    web_dir: str | Path = "web",
) -> FastAPI:
    from leaksense.evaluate import _score, case_metrics
    from leaksense.serve_cmds import default_bundle

    b = load_bundle(bundle_path or default_bundle())
    rep_path = Path(reports_dir) / "final.json"
    if not rep_path.is_file():
        raise FileNotFoundError(f"no {rep_path}; run `leaksense final-report` first")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    page = Path(web_dir) / "index.html"

    net = b["network"]
    nodes, pipes = net.nodes, net.pipes
    sensor_ids = [str(s) for s in net.sensor_ids]
    sensor_pos = {s: i for i, s in enumerate(sensor_ids)}
    classes = [str(c) for c in b["classes"]]
    cases = b["cases"]
    case_pos = {str(c): i for i, c in enumerate(cases["case_id"])}
    head = b["headline"]
    pl = rep["placement"]
    n_order = len(pl["final_order"])

    app = FastAPI(title="LeakSense", version=str(b["provenance"].get("leaksense_version")))

    @app.get("/", include_in_schema=False)
    def index():
        if not page.is_file():
            raise HTTPException(404, f"{page} not found")
        return FileResponse(page, media_type="text/html")

    @app.get("/network")
    def network():
        node_rows = [
            {
                "id": str(r.id),
                "kind": str(r.kind),
                "x": float(r.x),
                "y": float(r.y),
                "sensor": bool(r.sensor),
            }
            for r in nodes.itertuples()
        ]
        ids = list(nodes["id"])
        pipe_rows = [
            {
                "id": str(r.id),
                "from": str(ids[int(r.u)]),
                "to": str(ids[int(r.v)]),
                "length_m": float(r.length_m),
                "diameter_mm": float(r.diameter_mm),
                "leak_eligible": bool(r.leak_eligible),
            }
            for r in pipes.itertuples()
        ]
        return clean(
            {
                "nodes": node_rows,
                "pipes": pipe_rows,
                "sensors": sensor_ids,
                "note": "x, y are drawing coordinates (not to scale); distances use pipe lengths",
            }
        )

    @app.get("/placements")
    def placements(k: int | None = Query(default=None, ge=1)):
        base = {
            "greedy_order": pl["final_order"],
            "coverage_order": pl["coverage_order"],
            "scored_ks": pl["ks"],
            "n_random": pl["n_random"],
            "headline": head,
        }
        if k is None:
            return clean(base)
        if k > n_order:
            raise HTTPException(422, f"k must be 1..{n_order} (the saved placement order)")
        scores = {s: _placement_rows(pl, s, k) for s in pl["results"]}
        return clean(
            {
                **base,
                "k": k,
                "greedy": pl["final_order"][:k],
                "coverage": pl["coverage_order"][:k],
                "scores": {s: v for s, v in scores.items() if v is not None},
            }
        )

    @app.get("/cases")
    def list_cases():
        rows = []
        for i, r in enumerate(cases.to_dict("records")):
            rows.append(
                {
                    "case_id": str(r["case_id"]),
                    "scenario_id": int(r["scenario_id"]),
                    "kind": r.get("kind"),
                    "leak_type": r.get("leak_type"),
                    "pipe": str(r["pipe"]),
                    "leak_flow_lps": r.get("leak_flow_lps"),
                    "baseline_kind": r.get("baseline_kind"),
                    "n_windows": len(b["windows"][i]),
                }
            )
        return clean({"set": "locked test (held out)", "n": len(rows), "cases": rows})

    @app.post("/predict")
    def predict(req: PredictRequest):
        i = case_pos.get(req.case_id)
        if i is None:
            raise HTTPException(404, f"unknown case {req.case_id}")
        method = req.method or head
        model = b["models"].get(method)
        if model is None:
            raise HTTPException(422, f"method must be one of {sorted(b['models'])}")
        bad = [s for s in req.sensors if s not in sensor_pos]
        if bad:
            raise HTTPException(422, f"not sensor junctions: {bad}")
        if len(set(req.sensors)) != len(req.sensors):
            raise HTTPException(422, "sensors must be distinct")
        mask = np.zeros(len(sensor_ids), dtype=bool)
        mask[[sensor_pos[s] for s in req.sensors]] = True
        s = _score(model, [b["windows"][i]], mask)[0]
        y = int(cases["y"].iloc[i])
        dist = b["dist_to_leak"][i]
        order = np.argsort(-s, kind="stable")[: req.top]
        m = case_metrics(s, y, dist, b["class_hops"])
        return clean(
            {
                "case_id": req.case_id,
                "method": method,
                "k": int(mask.sum()),
                "sensors": req.sensors,
                "ranking": [
                    {
                        "pipe": classes[j],
                        "score": float(s[j]),
                        "rank": int((s > s[j]).sum()) + 1,
                        "dist_m": float(dist[j]),
                    }
                    for j in order
                ],
                "true_pipe": classes[y],
                "true_rank": int((s > s[y]).sum()) + 1,
                "n_tied_with_true": int((s == s[y]).sum()),
                "top3": m["top3"],
                "top1_dist_m": m["dist_m"],
                "score_note": "mean log-probability over the case's windows"
                if method == head
                else "signature match score",
            }
        )

    @app.get("/curves")
    def curves():
        prov = rep["provenance"]
        return clean(
            {
                "set": "locked test",
                "headline": head,
                "headline_k": rep["headline_k"],
                "ks": rep["ks"],
                "ladder": _ladder(rep),
                "placement": {
                    s: {str(k): _placement_rows(pl, s, k) for k in pl["ks"]} for s in pl["results"]
                },
                "leak_flow": rep.get("leak_flow_curve"),
                "test_set": rep.get("test_set"),
                "provenance": {
                    "commit": prov["git"]["commit"][:10],
                    "generated_at": prov["generated_at"],
                    "data_manifest": (prov.get("data_manifest_sha256") or "")[:12],
                    "source": "reports/final.json",
                },
            }
        )

    return app
