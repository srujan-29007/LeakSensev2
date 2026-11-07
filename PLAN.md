# LeakSense: Plan

The work is split into small milestones, done in order. Each one ends with a **Verify** step that runs on CPU.

- The data contract is verified and enforced: see SPEC.md §8, `reports/data_inventory.md` and `leaksense validate-data`.
- All data access goes through `LEAKSENSE_DATA_DIR`, set in the environment or in a git-ignored `.env` file.
- Milestones M2–M15 were revised on 2026-11-08 to match the verified data. Each change is marked *(data: …)* with the fact that drove it.

## Layout

```
src/leaksense/
  config.py  network.py  sensitivity.py  train.py  evaluate.py  placement.py  api.py  cli.py
  data/contract.py  data/inp.py  data/validate.py      # done (M1)
  data/{index,features,splits}.py
  models/{baselines,mlp,gbm,gat}.py
web/index.html                 # single static page (vanilla JS + SVG)
configs/*.yaml                 # cpu_small, mlp_cpu, gat_cpu, ablation_cpu, place_cpu, full
splits/splits_v1.json          # committed
tests/synthetic_data.py        # writes a tiny dataset in the exact LeakDB format (pytest fixture `synthetic`)
reports/                       # summaries committed; reports/raw/ ignored
notebooks/kaggle_run.ipynb     # calls the CLI only, no logic
```

## Rules that apply to every milestone

- Every report records its config, seed, git commit and data-manifest hash.
- The locked test fold is used only in M13.
- A number goes into the README only if a report file produced it.
- `uv run pytest` passes without the real dataset. Tests build a tiny dataset in a temp folder using `tests/synthetic_data.py`. Tests that need the real data are marked `@pytest.mark.data` and skipped when `LEAKSENSE_DATA_DIR` is unset.
- **No oracle inputs.**
  - Features never use `leak_node*` columns, `Node_demands.csv` or `Link_flows.csv`. This is asserted in the feature builder and tested.
  - These are ground truth, used only for labels, distance metrics and reporting breakdowns. They are never inputs to a model or to sensor placement:
    - each scenario's own `.inp` parameters
    - leak flow
    - leak type
    - SNR
  - *(data: D9, each scenario's real pipe parameters deviate up to ±20% from nominal, and an operator only knows the nominal network)*

---

### M0: Repo scaffold and config — DONE
- uv project on Python 3.11 with CPU torch, `config.py` (`LEAKSENSE_DATA_DIR` is required and fails loudly if missing or invalid), and the `leaksense` CLI.
- **Verified:** `uv sync`; `uv run pytest`; torch 2.14.1+cpu, torch-geometric 2.8.1 and wntr 1.5.0 all import; `leaksense --help` works; ruff is clean.

### M1: Data validation — DONE
- `leaksense validate-data [--deep]` asserts D1–D9 and reports D10–D11. It writes `reports/data_validation*.json` and exits non-zero on drift.
- **Verified:** fast mode passes in 38 s; `--deep` passes in 265 s; manifest `6281be1c3cf2…`. Tests cover 10 corruption cases and confirm the dataset is only read.

### M2: Network graph — DONE
- Build from the nominal `$LEAKSENSE_DATA_DIR/Hanoi.inp`. Reuse `leaksense.data.inp.read_network` and `merge_split_pipes`; do not add a second parser. WNTR isn't needed until M7.
- One nominal graph serves every case. *(data: D9, topology is identical in all 500 scenarios)*
- Build:
  - the node table and pipe table
  - the edge index
  - edge features: nominal length, diameter, roughness, and Hazen-Williams resistance. Convert units for the resistance: diameter mm → m and flow L/s → m³/s. *(data: D5, `UNITS LPS`, diameter in mm)*
  - the pipe line-graph, for hop distance
  - a node graph weighted by nominal length, for network distance in metres
- **Verify:**
  - `uv run pytest tests/test_network.py`: 32 nodes, 34 pipes, connected, pipe 1 connects 1 and 2, and the resistance on a hand-computed pipe is correct.
  - `uv run leaksense network-info` prints a summary.
- **Verified:** 5/5 tests pass (including the real `Hanoi.inp`: 32 nodes, 34 pipes, connected, 3 loops; pipe 1 is 1→2 with resistance 0.1201). `network-info` prints a summary: 39,420 m in total, at most 12 pipe hops, at most 16,300 m network distance.

### M3: Case index and pipe labels — DONE
- Write `artifacts/cases.parquet` with one row per case:
  - `scenario_id`, `kind` (pure or solo), `pipe` (2–34), `split_fraction`, `leak_type`, `area`
  - `start`, `end` (for solo cases, `end` is clipped to the step before the second leak starts), `n_steps`
  - `leak_flow_lps`: median over the case's active window. For reporting only. *(data: D11, leak type is not a size class, and SPEC H5 needs the leak-flow plot)*
  - baseline window(s), `baseline_kind` (pre / post / combined), quality flags
- Labels and split fractions come from `MergedPipe.leak_fraction`.
  - In the 6 scenarios where the same pipe is split twice (`Xll`), map leak to pipe by leak ID, not by position in the chain.
  - Cross-check every label against the `Xl` column in `Link_flows.csv`.
  - *(data: D8, double-split scenarios 7, 46, 269, 297, 441, 474)*
- **Baseline rule** (SPEC §5): a baseline window needs at least 14 leak-free days. Try these in order and use the first that qualifies:
  1. pre-leak
  2. post-leak (clear of every leak in the scenario)
  3. pre and post combined
  4. otherwise exclude the case

  M3 reports how many cases each option covers and how many are excluded; it does not assume zero.
  - *(data: 8 pure cases have under 14 days of pre-leak history, and 14 leaks end within a day of the year's end, so the old "post-leak fallback" is not always available. A solo case's post-window must also avoid the second leak.)*
- Keep cases shorter than one window and record `n_steps`. *(data: 2 pure cases last under 24 h; the shortest is 18 h)*
- **Verify:**
  - `uv run leaksense build-index` prints exactly 244 pure + 245 solo = 489 cases, minus the excluded ones, which are listed.
  - Every case has one pipe in 2–34.
  - Per-pipe pure counts equal the D8 table in `reports/data_validation.json`.
  - Per-pipe totals (pure + solo) are printed. They range from 4 (pipe 2) to 27 (pipe 30). *(The inventory's hand count said 26: it missed solo leaks on an `Xll` piece, scenarios 46, 269 and 441.)*
  - Synthetic-data tests pass, including an `Xll` scenario and a case with no valid baseline.
- **Verified:** `build-index` takes about 6 s and writes `reports/case_index.json`. It finds 244 pure + 245 solo = 489 cases with 0 excluded. Baselines: 462 pre, 25 post (8 pure, 17 solo), 2 combined. 10 two-leak scenarios have no solo case. Per-pipe pure counts match D8. 9/9 index tests pass.

### M4: Features and cache — DONE
- Read only `Node_pressures.csv`, only the columns `2`…`32`, as float32. Cache one `.npz` per scenario under `artifacts/`; the full cache is about 1.1 GB. `configs/cpu_small.yaml` uses a subset of scenarios.
- **Measure the baseline method instead of assuming it.** Each scenario `.inp` holds a full-year, 30-min demand pattern for every junction, so a weekly cycle was never verified at the junction level. Mean total demand does vary by weekday, by about ±8% (Fri 825, Sun 695 L/s; D12), which supports testing weekday matching. Compare the leak-free ΔH noise (std on held-out leak-free days) for:
  - time-of-day matching
  - time-of-day + weekday matching
  - the most recent N leak-free days with time-of-day matching

  Choose the method by measurement, record it in the config and in `reports/baseline_choice.md`, and prefer the most recent leak-free days over the whole pre-leak period. *(data: leak start times are spread over the year, so baselines from months earlier may drift seasonally)*
- **Measure the noise floor.** The 0.5 m step-to-step figure came from one scenario. Measure the per-junction leak-free ΔH std across scenarios, and set the window length and stride from it.
- **Handle the dynamic range.** Small leaks move ΔH by about 1 m; big leaks move it by tens of metres, with leak flow up to 8,994 L/s.
  - The cache stores raw ΔH.
  - A configurable input transform is applied in the models, e.g. signed `log1p`, or per-sample scaling plus a magnitude channel. *(data: D11)*
- Add per-case `snr` (largest mean |ΔH| over junctions ÷ leak-free window noise, estimated from 3 held-out days of the case's own baseline). It is written to `artifacts/features/<config-hash>/case_stats.parquet`, next to the features it is computed from, not to `cases.parquet`. For reporting only.
- **Windows:**
  - A case shorter than one window gets one shorter window and a flag.
  - Use `max_windows_per_case` and per-case sample weights so that each case counts equally in training, not only for CPU speed. *(data: case durations range from 18 h to 344 days)*
- **Verify:**
  - `uv run leaksense build-features --config configs/cpu_small.yaml` finishes in minutes.
  - Tests:
    - ΔH ≈ 0 on a held-out leak-free window.
    - No leak-active timestep contributes to any baseline.
    - No `leak_node*` column is read.
  - `reports/baseline_choice.md` shows the measured noise for each method, and the chosen method.
  - `uv run leaksense show-case <id>` prints per-junction mean ΔH. Reference: Scenario 34, pipe 22 (21→22), about 665 L/s. Expect node 22 at about −61 m and node 21 at about −33 m, with every other node within about 4 m. *(data: inventory §5)*
- **Verified:**
  - `baseline-study` (122 s, Scenario-1 + 419 non-test scenarios) chose time of day × weekday, 14 days, mean, W = 48, stride 24, and a 7-day maximum lag. Leak-free noise per window is 0.149 m. The Leak Start check is clean.
  - Small leaks: the median largest |ΔH| is 0.90 m, and 72% exceed 3× the window noise. All big leaks do.
  - `build-features` takes 2 s on `cpu_small` and 14 s on the full set (489 cases, 6,244 windows, 13 per case at the median). Only S069 gets a short window. The pressure cache is 1.0 GB.
  - `show-case S034-pure`: node 22 −60.9 m, node 21 −33.4 m, all others ≤ 4.2 m.
  - 13 feature tests pass, including the poison test (rows outside the baseline cannot change ΔH) and the read-only-pressures/labels test.

### M5: Splits — DONE (run before M4's measurements, so they never see the locked test fold)
- Grouped 5-fold CV plus a locked test fold of about 15% of scenarios.
  - Use `StratifiedGroupKFold`, grouped by **scenario ID** and stratified by pipe, with leak type and case kind balanced where possible.
  - Each scenario currently gives exactly one case, so the groups are 1:1 with cases. Keep the scenario key anyway: detection (M15) adds several windows per scenario.
- **Rare pipes:** pipes with fewer than 6 cases cannot appear in the test fold and in every CV fold.
  - Handle them deterministically: such a pipe goes into the test fold only if at least 2 of its cases remain for training.
  - List them in the split report.
  - *(data: totals per pipe range from 4 (pipe 2) to 27 (pipe 30), `reports/case_index.json`)*
- Write `splits/splits_v1.json` and commit it.
- **Verify:** `uv run leaksense make-splits` and `uv run pytest tests/test_splits.py`, which check:
  - no scenario overlap
  - a deterministic seed
  - the rare-pipe rule
  - a printed balance of pipe, leak type and case kind
- **Verified:** 70 test scenarios (14.3%) and folds of 84/84/83/84/84 cases. The worst fold is 3.8 points off on the small-leak and solo shares (best of 50 seeds). Pipe 2 (4 cases) is the only rare pipe: it has 0 test cases and is in 4 of the 5 folds. 5/5 split tests pass.

### M6: Evaluation harness and non-learned rungs — DONE
- Aggregate predictions to case level.
- **Metrics:**
  - top-1/3/5 accuracy
  - hop distance
  - network distance in metres. The true leak point is the split fraction applied to the **nominal** pipe length, and all distances use nominal lengths. *(data: D10, coordinates are not to scale)*
- Bootstrap CIs, resampling by scenario.
- **Breakdowns:**
  - overall
  - small vs big
  - pure vs solo
  - baseline kind: pre vs post/combined *(M4: post baselines use after-the-leak data)*
  - leak-flow terciles
  - SNR terciles (from `artifacts/features/<config-hash>/case_stats.parquet`)

  *(data: D11, leak type does not cleanly separate leak sizes)*
- Output: JSON and markdown.
- **Rungs:**
  - chance
  - class prior
  - `maxdh`: rank all 33 pipes by hop distance from the sensed junction with the largest |ΔH|, breaking ties by the next-largest sensed junction. This works at any k. Pipe 1 is never ranked.
- **Verify:**
  - `uv run leaksense eval --model chance` gives top-1 ≈ 1/33.
  - `uv run leaksense eval --model maxdh --k 31` writes `reports/maxdh.json`, including every breakdown.
  - `uv run pytest tests/test_metrics.py` checks hand-computed examples, including network distance with a leak at a split fraction on a toy graph.
- **Verified** (419 CV cases; the test fold is excluded and asserted absent):
  - **chance:** exactly 3.0 / 9.1 / 15.2% top-1/3/5 at every k.
  - **prior:** 5.5 / 14.6 / 23.4%.
  - **`maxdh`:**

    | k | top-1 | top-3 | small-leak top-3 | median distance |
    |---|---|---|---|---|
    | 8 | 16.0% | 43.1% | 43.1% | 2,314 m |
    | 31 | 33.7% | 88.8% [85.9, 91.9] | 81.7% | 759 m |

    `reports/maxdh.json` has all six breakdown families.
  - **Large leaks:** in the highest leak-flow tercile, `maxdh` top-1 falls to 22.9% while top-3 is 95%. A big leak depresses many junctions, and the largest drop is often at a downstream dead end rather than next to the leak.
  - **Lag check:** `max_lag_days` 14 vs 7 makes no measurable difference, so it stays at 7.
  - 13 metric tests pass, plus a real-data test.

### M7: Sensitivity matrix (physics input)
- On the nominal network, simulate one leak per eligible pipe (33 pipes).
  - The leak is a fixed demand at a junction inserted at the pipe's midpoint, run with `EpanetSimulator`.
  - Do not use `add_leak` with `EpanetSimulator`. EPANET ignores WNTR leak models, which was a bug in the old repo.
  - *(data: within a leak, leak flow is roughly constant)*
- Use two sizes, the median leak flows from D11: about 97 L/s (small) and about 980 L/s (big). Use flows, not leak areas, because the area-to-flow relationship depends on pressure.
- **Units:** WNTR works in SI internally, so convert L/s to m³/s. *(data: D5, `UNITS LPS`)*
- **Demand levels:** scale every nominal base demand by one global multiplier α. The nominal total (1,538.6 L/s) is about 2× typical demand, and headloss grows as Q^1.852, so the raw nominal values would overstate S.
  - Use three levels, from total demand ÷ nominal total: low α ≈ 0.23 (p5, about 355 L/s), **mid α ≈ 0.49 (median, about 760 L/s, the default)** and high α ≈ 0.80 (p95, about 1,228 L/s).
  - Compute α from the training scenarios' `Node_demands.csv` junction columns only. α is a single scalar, not a per-case input.
  - Never use each scenario's own demand patterns, which are ground truth.
  - That makes 3 levels × 2 flows × 33 pipes = 198 snapshot runs, and S is 31 × 33 × 3 × 2.
  - *(data: D12)*
- If any simulated pressure goes negative at the big size, record it as a warning in the report.
- **Result:** S, a 31 junctions × 33 pipes matrix for each size.
- New rung `sigmatch`: cosine similarity between the masked ΔH and each column of S.
- **Verify:**
  - `uv run leaksense build-sensitivity` finishes in under 5 minutes on CPU. It prints the α levels used and, per pipe, the cosine similarity between the low, mid and high columns.
    - If every cosine is above 0.99, `sigmatch` and the M10 prior use mid only.
    - Otherwise M10 may ablate the three levels as extra inputs.
  - Tests:
    - every entry of S is a pressure drop (≤ 0)
    - no two columns are identical
    - `@pytest.mark.data`: of all big-size columns of S, the one for pipe 22 is the most cosine-similar to Scenario 34's observed mean ΔH
  - This replaces the earlier test "the largest entry is at an endpoint of the pipe". That claim was never verified, and drops can be larger at dead ends further downstream.
  - `uv run leaksense eval --model sigmatch --k 31` writes a report.

### M8: Learned tabular rungs (MLP, GBM) with masking
- Input is `[ΔH·m, m]` over the 31 junctions, after the M4 input transform. The mask size k is drawn uniformly from 1 to 31. The output is 33 classes.
- Class-weighted loss, with the weights computed on each training fold. *(data: 4–26 cases per pipe)*
- Case-weighted sampling from M4.
- **Verify:**
  - `uv run leaksense train --config configs/mlp_cpu.yaml` finishes in under 10 minutes on CPU.
  - The model overfits a set of 5 cases.
  - Accuracy-vs-k curves, with all M6 breakdowns, are written to `reports/`.

### M9: Masked GAT
- 4 GATv2 layers encode the nodes.
- The reservoir stays in the graph with features [0, 0] and mask 0. *(data: D5, its pressure is always 0)*
- Pipe head: the two endpoint embeddings plus the nominal edge features give 33 logits. Pipe 1 is excluded.
- Use the same class weights, case weights and input transform as M8.
- **Verify:**
  - `uv run pytest tests/test_gat.py` checks:
    - output shapes
    - changing a masked node's ΔH does not change the output
    - the model overfits a set of 5 cases
  - `uv run leaksense train --config configs/gat_cpu.yaml --fold 0` runs as a smoke test.

### M10: Physics-input GAT and ablation
- Add the S-based prior three ways: as node features, as edge features, and as a per-pipe prior logit.
- Ablate plain vs physics, with CIs, broken down by leak type and by leak-flow tercile.
- **Verify:** `uv run leaksense ablate --config configs/ablation_cpu.yaml` writes `reports/ablation.md`.

### M11: Sensor placement
- Greedy forward selection over the 31 junctions. The masked model stays fixed, and each step maximizes top-3 accuracy on the inner-validation cases.
- **Comparisons:**
  - 100 random placements per k
  - a greedy that maximizes sensitivity coverage
- Evaluate on outer held-out cases at k ∈ {2, 3, 4, 6, 8, 12, 16}:
  - overall
  - small leaks only
  - by leak-flow tercile
- Report how stable the chosen sensor sets are across folds: the overlap of selected junctions. Each outer fold has only about 80 inner-validation cases, so greedy choices may be noisy.
- **Verify:**
  - `uv run leaksense place --config configs/place_cpu.yaml` writes `reports/placement.json` and a curve PNG.
  - A test checks that greedy placement is deterministic for a fixed seed.

### M12: Full run on Kaggle (optional)
- `notebooks/kaggle_run.ipynb` sets `LEAKSENSE_DATA_DIR` to the Kaggle input path, installs the repo, and runs the CLI with `configs/full.yaml`. The notebook contains no logic of its own.
- **First, run `leaksense validate-data` on Kaggle.** Stop if the manifest sha256 is not `6281be1c3cf2…`. *(data: the old notebooks used a third-party Kaggle copy whose contents were never verified)*
- **Verify:**
  - The Kaggle log shows that validate-data passed with the expected manifest.
  - Download the artifacts, run `uv run leaksense eval --from-artifacts <dir>` locally on the CPU subset, and check the numbers match within tolerance.

### M13: Locked test and final report
- Evaluate the test fold exactly once.
- Write `reports/FINAL.md` covering:
  - every rung, with all M6 breakdowns
  - the placement curves
  - the accuracy-vs-leak-flow plot (SPEC H5)
  - the status of H1–H5
  - the number of excluded cases (M3) and the chosen baseline method (M4)
  - limitations
- **Verify:** `uv run leaksense final-report`, then tag the commit.

### M14: API and single page
- FastAPI endpoints:
  - `/network`
  - `/placements?k=`
  - `/cases` (held-out cases only)
  - `/predict` (case ID + sensor set → top-k pipes)
  - `/curves`
- Serve held-out cases from the cached features in `artifacts/`, so the 11 GB raw dataset isn't needed at serve time.
- The map is drawn from the `.inp` coordinates. That is fine for display, but distances shown always come from pipe lengths. *(data: D10)*
- `web/index.html` is served as a static page and shows only data from saved reports and artifacts.
- **Verify:**
  - `uv run pytest tests/test_api.py` passes, using the FastAPI TestClient.
  - Run `uv run leaksense serve`, open http://localhost:8000, choose a budget and a case, and check that predictions and curves appear.

### M15: Stretch goals (detection, overlapping-leak stress test)
- **Detection.**
  - Scenario-1 is the only scenario with no leak at all. So the no-leak class comes from leak-free stretches of the leak scenarios, excluding baseline windows and a buffer before each leak starts.
  - Ground truth is the per-scenario `Labels.csv`, which is the union of active leaks (D7).
  - Before choosing the time-to-detect metric, check across all leaks whether leak flow ramps up. In the sampled leaks it was constant from the first step, in which case time-to-detect mostly measures the window length.
  - Metrics: false-positive rate, and time-to-detect if it turns out to be meaningful.
- **Overlap stress test (evaluation only).**
  - First count the periods where both leaks are active and how long they last. Scenario 273's second leak lasts zero steps.
  - Check whether either leak appears in the top-k.
  - Report the 6 scenarios where both leaks are on the same pipe separately, because "either leak in top-k" is trivial for them.
- **Verify:** `uv run leaksense eval --task detect` and `uv run leaksense eval --stress overlap`.
