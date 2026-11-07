# LeakSense: Plan

The work is split into small milestones, done in order. Each one ends with a **Verify** step that runs on CPU.

- The data contract is verified: see SPEC.md §8 and `reports/data_inventory.md`.
- All data access goes through `LEAKSENSE_DATA_DIR`, set in the environment or in a git-ignored `.env` file.

## Proposed layout

```
src/leaksense/
  config.py  network.py  sensitivity.py  train.py  evaluate.py  placement.py  api.py  cli.py
  data/{validate,index,features,splits}.py
  models/{baselines,mlp,gbm,gat}.py
web/index.html                 # single static page (vanilla JS + SVG)
configs/*.yaml                 # cpu_small, mlp_cpu, gat_cpu, ablation_cpu, place_cpu, full
splits/splits_v1.json          # committed
tests/                         # tiny synthetic fixture dataset that mirrors the data contract
reports/                       # summaries committed; reports/raw/ ignored
notebooks/kaggle_run.ipynb     # calls the CLI only, no logic
```

## Rules that apply to every milestone

- Every report records its config, seed, git commit and data-manifest hash.
- The locked test fold is used only in M13.
- A number goes into the README only if a report file produced it.
- `uv run pytest` passes without the real dataset, using the fixture under `tests/fixtures/HanoiOK/`. Tests that need the real data are marked `@pytest.mark.data` and skipped when `LEAKSENSE_DATA_DIR` is unset.
- Features never use `leak_node*` columns, `Node_demands.csv` or `Link_flows.csv`. This is asserted in the feature builder and tested.

---

### M0: Repo scaffold and config
- Run `git init`. Set up `pyproject.toml` for uv with Python 3.11 and the CPU wheel index for torch. Dependencies: torch, torch_geometric, wntr, pandas, pyarrow, scikit-learn, lightgbm, fastapi, uvicorn, python-dotenv, pytest, ruff.
- `config.py` reads `LEAKSENSE_DATA_DIR` (required) and `LEAKSENSE_ARTIFACTS_DIR` (default `./artifacts`), loading `.env` if one exists. It fails with a clear message if the data dir is unset, missing, or lacks `Hanoi.inp`, `Labels.csv` or `Scenario-1/`.
- `.gitignore` and `.env.example` already exist.
- `leaksense --help` stub.
- **Verify:**
  ```
  uv sync
  uv run pytest            # includes config tests: unset var -> clear error; fixture dir -> ok
  uv run python -c "import torch, torch_geometric, wntr; print(torch.__version__)"
  uv run leaksense --help
  git status               # shows no dataset files
  ```

### M1: Data validation (manual inspection done, now automated)
- Manual inspection is complete; the results are in `reports/data_inventory.md`.
- Add `leaksense validate-data`, which asserts D1–D10 from SPEC.md §8 using headers, sizes, info files and `.inp` header sections. The streaming checks only read full files when `--deep` is given.
- It writes `reports/data_validation.json` and exits non-zero if the data has drifted from the contract.
- Also quantify the exact-0.00 pressures: count them per scenario and check whether they line up with big leaks.
- **Verify:**
  - `uv run leaksense validate-data` reports that all checks pass in under 2 minutes.
  - `uv run pytest tests/test_validate.py` checks that the fixture passes and that a deliberately corrupted fixture fails.

### M2: Network graph
- Load `$LEAKSENSE_DATA_DIR/Hanoi.inp` and build:
  - the node table and pipe table
  - the edge index
  - edge features: nominal length, diameter, roughness, and the Hazen-Williams resistance
  - the pipe line-graph, for hop distance
  - a length-weighted node graph, for network distance in metres
- **Verify:**
  - `uv run pytest tests/test_network.py` asserts 32 nodes, 34 pipes, a connected graph, and that pipe 1 connects 1 and 2.
  - `uv run leaksense network-info` prints a summary.

### M3: Case index and pipe labels
- Write `artifacts/cases.parquet` with one row per case:
  - `scenario_id`, `kind` (pure or solo), `pipe` (2–34), `split_fraction`, `leak_type`, `area`
  - `start`, `end` (for solo cases, the end is clipped to the second leak's start)
  - baseline window and baseline kind (pre or post)
  - quality flags
- Read the label from the `Xl` link name and check it against the scenario `.inp`. Handle `Xll` correctly.
- **Verify:**
  - `uv run leaksense build-index` prints 244 pure cases and about 245 solo cases.
  - Every case has one pipe in 2–34.
  - Per-pipe counts match the inventory: pipes 2 and 6 have 2 pure cases each, pipe 30 has 16.
  - Fixture tests pass, including an `Xll` example.

### M4: Features and cache
- Read only `Node_pressures.csv`, only the columns `2`…`32`, as float32.
- Cache one `.npz` per scenario under `artifacts/`. A full cache is about 1.1 GB. `configs/cpu_small.yaml` uses a subset of scenarios and `max_windows_per_case`.
- Compute ΔH against the same scenario's pre-leak baseline, matched by time-of-day and day-of-week. Fall back to a post-leak baseline when there are fewer than 14 days of pre-leak data.
- Take windows (length and stride from config) of averaged ΔH. These suppress the roughly 0.5 m demand noise.
- **Verify:**
  - `uv run leaksense build-features --config configs/cpu_small.yaml` finishes in minutes.
  - Tests:
    - ΔH ≈ 0 on a held-out leak-free window.
    - No leak-active timestep contributes to the baseline.
    - No `leak_node*` column is read.
  - `uv run leaksense show-case <id>` prints the per-junction mean |ΔH|. For a big leak, the endpoints of the leaking pipe come out on top. Reference: Scenario 34, pipe 22, with node 22 at −61 m and node 21 at −33 m.

### M5: Splits
- Grouped 5-fold CV, plus a locked test fold of about 15% of scenarios:
  - grouped by **scenario ID**, so a scenario's case never crosses folds
  - stratified by pipe, and by leak type where possible
  - pipes with fewer than 5 cases (pipe 2 has 4, counting solo cases) are listed in the split report, not forced
- Write `splits/splits_v1.json` and commit it.
- **Verify:** `uv run leaksense make-splits` and `uv run pytest tests/test_splits.py`, which check:
  - no scenario overlap
  - the seed is deterministic
  - pipe and leak-type balance is printed

### M6: Evaluation harness and non-learned rungs
- Case-level aggregation of predictions.
- Metrics: top-1/3/5, hop distance, network distance in metres from the predicted pipe's midpoint to the true leak point.
- Bootstrap CIs, resampled by scenario.
- Breakdown: overall, small vs big, pure vs solo.
- Output as JSON and markdown.
- Rungs: chance, class prior, and the largest-|ΔH| sensor mapped to its adjacent pipes.
- **Verify:**
  - `uv run leaksense eval --model chance` gives top-1 ≈ 1/33.
  - `uv run leaksense eval --model maxdh --k 31` writes `reports/maxdh.json`.
  - `uv run pytest tests/test_metrics.py` tests the metrics on hand-computed examples, including network distance on a toy graph.

### M7: Sensitivity matrix (physics input)
- On the nominal network in WNTR, simulate a leak at the midpoint of each of the 33 leak-eligible pipes, at a representative small size and a representative big size taken from the inventory's leak areas. Use a fixed demand snapshot. No scenario ground truth is used.
- Result: S, a 31 junctions × 33 pipes matrix for each leak size.
- New rung `sigmatch`: cosine similarity between the masked ΔH and the columns of S.
- **Verify:**
  - `uv run leaksense build-sensitivity` finishes in under 5 minutes on CPU.
  - A test confirms that the largest entry in each column of S is at an endpoint of that pipe.
  - `uv run leaksense eval --model sigmatch --k 31` writes a report.

### M8: Learned tabular rungs (MLP, GBM) with masking
- Input `[ΔH·m, m]` over the 31 junctions, with a mask size k drawn uniformly from 1 to 31.
- Output: 33 classes.
- **Verify:**
  - `uv run leaksense train --config configs/mlp_cpu.yaml` finishes in under 10 minutes on CPU.
  - The model overfits a set of 5 cases.
  - Accuracy-vs-k curves are written to `reports/`.

### M9: Masked GAT
- 4 GATv2 layers encode the nodes. The reservoir is a node with mask = 0.
- Pipe head: the two endpoint embeddings plus the edge features give 33 logits. Pipe 1 is excluded.
- **Verify:**
  - `uv run pytest tests/test_gat.py` checks:
    - output shapes
    - changing a masked node's ΔH does not change the output
    - the model overfits a set of 5 cases
  - `uv run leaksense train --config configs/gat_cpu.yaml --fold 0` runs as a smoke test.

### M10: Physics-input GAT and ablation
- Add the S-based prior in three places: as node and edge features, and as a per-pipe prior logit.
- Ablate plain vs physics, with CIs, broken down by leak type.
- **Verify:** `uv run leaksense ablate --config configs/ablation_cpu.yaml` writes `reports/ablation.md`.

### M11: Sensor placement
- Greedy forward selection over the 31 junctions. The masked model stays fixed, and each step maximizes top-3 accuracy on the inner-validation cases.
- Comparisons:
  - 100 random placements per k
  - a greedy that maximizes sensitivity coverage
- Evaluate on the outer held-out cases at k ∈ {2, 3, 4, 6, 8, 12, 16}, overall and for small leaks only.
- **Verify:**
  - `uv run leaksense place --config configs/place_cpu.yaml` writes `reports/placement.json` and a curve PNG.
  - A test checks that greedy placement is deterministic for a fixed seed.

### M12: Full run on Kaggle (optional)
- `notebooks/kaggle_run.ipynb` sets `LEAKSENSE_DATA_DIR` to the Kaggle input path, installs the repo, and runs the CLI with `configs/full.yaml`. The notebook contains no logic of its own.
- **Verify:** download the artifacts, run `uv run leaksense eval --from-artifacts <dir>` locally on the CPU subset, and check the numbers match within tolerance.

### M13: Locked test and final report
- Evaluate the test fold exactly once.
- Write `reports/FINAL.md` covering:
  - every rung, broken down by small/big and pure/solo
  - the placement curves
  - the status of H1–H5
  - limitations
- **Verify:** `uv run leaksense final-report`, then tag the commit.

### M14: API and single page
- FastAPI endpoints:
  - `/network`
  - `/placements?k=`
  - `/cases` (held-out cases only)
  - `/predict` (case ID + sensor set → top-k pipes)
  - `/curves`
- `web/index.html` is served as a static page and shows only data from saved reports and artifacts.
- **Verify:**
  - `uv run pytest tests/test_api.py` passes, using the FastAPI TestClient.
  - Run `uv run leaksense serve`, open http://localhost:8000, choose a budget and a case, and check that predictions and curves appear.

### M15: Stretch goals (detection, overlapping-leak stress test)
- **Detection:** add a no-leak class built from leak-free windows that do not overlap any baseline window. Metrics: false-positive rate and time to detect.
- **Overlap stress test:** take the periods of two-leak scenarios where both leaks are active (evaluation only). Check whether either leak appears in the top-k.
- **Verify:** `uv run leaksense eval --task detect` and `uv run leaksense eval --stress overlap`.
