# LeakSense: Specification

- **Status:** M0–M6 are done (M5 ran before M4's measurements). The data contract is verified and enforced by `leaksense validate-data` (see `reports/data_inventory.md`).
- **Last updated:** 2026-11-08

## 1. Problem

A single leak has already been detected on the Hanoi water network (LeakDB). Pressure sensors are installed on a limited subset of junctions. The project has two tasks:

- **(a) Localization.** Identify which pipe is leaking.
- **(b) Placement.** Choose where to put *k* pressure sensors so that (a) is as accurate as possible.

## 2. Background

### 2.1 Audit of the old repo (`./reference/`)

**What works.** Commit `24c53c9` has these pieces. HEAD does not.
- A LeakDB pipeline with a 70/15/15 split by scenario, so no scenario leaks across splits.
- Pressure-change (ΔH) features.
- Sensor-mask node features `[x·mask, mask]`.
- A 4×GATv2 classifier that predicts the leak node.

`notebooks/leaksensepignn.ipynb` reports about **40% / 61% / 68%** top-1/3/5 accuracy on 36 held-out scenarios. Treat these numbers as indicative only:
- The test set is small.
- It was looked at many times while the model was being changed.
- The final training runs were interrupted.

**What is broken.**
- **`ml/` cannot be imported at HEAD.** Merge `7c69f22` overwrote `ml/model.py`, `simulation.py` and `optimization.py` with versions that don't match the rest of the package.
- **Backend:**
  - `/simulate` uses undefined names.
  - `/optimize-sensors` is called with the wrong signature, and the input it receives is a coordinate distance matrix, not data.
- **Frontend:**
  - It does not compile.
  - The accuracy curve is hard-coded.
  - Predict sends all-zero inputs.
- **Test scripts:** `test_eval.py` and `test_random.py` call an `/evaluate` endpoint that does not exist.
- **Sensor optimizer:** scores layouts on a fully connected fake graph using training samples, and never compares against random placement.
- **EPANET re-ranking:** broken, and it made results worse.
- **`data/processed/*.npy`:** synthetic, not real LeakDB data.
- **Notebooks:** depend on cells being run out of order, and use `GATv2Conv` without importing it.

**Method flaws, with the data evidence (`reports/data_inventory.md`).**
- **The "physics" loss is a label leak.** Its target is built from simulator flows, and those are non-zero only at the leak.
- **The Scenario-1 baseline is wrong.** Every scenario has its own pipe parameters (each perturbed by up to ±20%), base demands and 30-min demand patterns. ΔH computed against Scenario-1 mixes all of that in with the leak signal.
- **The leak labels are imprecise.** The old label was the nearest node by straight-line distance. The data has an exact label: the pipe that was split to insert the leak, and where along that pipe.
- **Long leaks dominate the metrics.** Averaging over timesteps weights each leak by its duration. The effective sample size is the number of leaks.

**Claimed only in the README.**
- Re-optimizing placement after a sensor fails.
- Uncertainty estimates and "expected error".
- A modular `ml/` layout.
- Any comparison against random placement.

### 2.2 Earlier claims that the data contradicts

| Claim (old README, notebooks or code) | What the data shows |
|---|---|
| Each scenario has a uniform ±10% uncertainty | Per-parameter levels are 5/10/15/20%, with 64 combinations. Deviations reach up to ±20%. |
| Scenario-1 is the leak-free baseline for all scenarios | Demands, patterns and parameters differ in every scenario. |
| Each leak adds 2 split-pipe flow columns | It adds one (`Xl`), or `Xll` when the same pipe is split a second time. |
| Scenario-1 has an empty `Leaks/` folder | The folder does not exist. |
| Coordinate distances are in metres | Coordinates are not to scale. For example, pipe 2 is 1,350 m long, but its endpoints are about 700 units apart. |
| There are 34 or 31 candidate locations | 33 pipes ever leak. Pipe 1 (reservoir to node 2) never does. |
| Every node column is a usable sensor | The `leak_node*` columns sit on the leak itself, so using them would leak the label. |

## 3. Goals

- **G1. Reproducibility.** One CLI command regenerates every reported number from the raw LeakDB data. Each run is pinned by a versioned config, a seed and a data-manifest hash.
- **G2. Model ladder.** Every rung is evaluated with the same harness:
  - **0. Chance and class prior.**
  - **1. Non-learned methods:**
    - the largest-|ΔH| sensor, mapped to its adjacent pipes
    - sensitivity-signature matching, with the sensitivity matrix simulated in WNTR from the nominal `$LEAKSENSE_DATA_DIR/Hanoi.inp`
  - **2. MLP and gradient-boosted trees** on masked features.
  - **3. Masked GAT.**
  - **4. Masked GAT with physics inputs:** the sensitivity prior and hydraulic edge features. **No loss term may use oracle flows.**
- **G3. One model for any budget.** A single sensor-masked model serves every budget from 1 to 31 sensors without retraining.
- **G4. Sensor placement.** A greedy forward-selection optimizer that maximizes the masked model's validation accuracy, compared against random placement on held-out cases.
- **G5. Compute.**
  - Every milestone runs on a laptop CPU, on a data subset, within minutes.
  - Full training can run on Kaggle with the same code and config.
- **G6. Demo.** A FastAPI service plus one static HTML/JS page:
  - the network map
  - a budget picker showing greedy vs random sensor sets
  - a held-out case picker showing the top-k predicted pipes
  - accuracy-vs-budget curves, read from saved evaluation JSON

## 4. Out of scope

- Localizing simultaneous leaks. Overlap periods of two-leak scenarios are only a stretch-goal stress test.
- Leak detection. This is a stretch milestone and does not count toward "done".
- Networks other than Hanoi.
- Flow sensors and AMR. Inputs are pressure only.
- EPANET re-ranking.
- Real-time or streaming inference.
- Re-optimizing placement after a sensor fails.
- Uncertainty quantification beyond softmax confidence.
- Authentication and deployment.
- The old marketing site.
- GPU-specific code.

## 5. Definitions

- **Case.** One labelled single-leak episode. There are two kinds:
  - **pure:** one of the 244 single-leak scenarios
  - **solo:** the stretch of at least 3 days (144 steps) in a two-leak scenario, from the first leak's start until the step before the second leak starts or the first leak ends, whichever comes first. 245 scenarios have one.
    - The **first leak is the one that starts first**, not `leak_node0`: in 126 of the 255 two-leak scenarios `leak_node1` starts earlier (121 of the 245 solo cases use `leak_node1`).
    - The label is mapped by leak ID. In 3 solo cases (scenarios 46, 269, 441) the first leak sits on the `Xll` piece.

  That gives about 489 cases. A two-leak scenario contributes at most one case.
- **Label.** The pipe that was split to insert the leak. The pipe ID comes from the `Xl` link name, or from the scenario `.inp`.
  - **33 leak-eligible pipes:** pipes 2–34. Pipe 1 never leaks.
  - **Leak point:** the leak's position along the pipe, `len(X) / (len(X) + len(Xl))`. It is used only for distance metrics.
- **Sensor candidates.** The 31 original junctions `2`…`32`.
  - The reservoir `1` is excluded because its pressure is always 0.
  - `leak_node*` columns are **never** used.
  - Demand and flow CSVs are never model inputs.
- **ΔH.** Pressure minus a leak-free baseline taken from the same scenario.
  - **Slot matching** was chosen in M4 by measurement on leak-free pseudo-cases from Scenario-1 and the 419 non-test scenarios (`reports/baseline_choice.md`). **Time of day × weekday**, using the per-slot **mean** of the **14 leak-free days nearest the case**, had the lowest window noise: 0.149 m, against 0.202 m for time of day only and 0.263 m for weekday with all history. It was also best for cases with only 14–28 days of history.
  - **Window:** 48 steps (24 h), stride 24, as a mean over the window. The leak-free noise is 0.98 m per step and 0.149 m per 24-h window.
  - **Drift:** baseline error grows with the distance from the baseline. With 24-h windows the median |bias| is 0.09 m in the first week, 0.16 m in the second and about 0.45 m after 3 months. So **samples are taken only within 7 days of the baseline**: the start of the case for pre baselines, the end for post, both ends for combined. A 14-day limit was checked in M6 with `maxdh` and changed nothing measurable: small-leak top-3 at k = 31 was 82.6% vs 81.7%, and at k = 8 42.6% vs 43.1%, with overlapping CIs (`reports/maxdh_lag14.json` vs `reports/maxdh.json`). So 7 days stays.
  - **Leak Start:** out of sample, ΔH at start−4 … start−1 is at noise level (median 0.43–0.60 m mean |ΔH| over big-leak cases) and jumps at start (7.9 m). So no guard step is needed.
  - **Baseline window.** It needs at least 14 leak-free days. Take the first option that qualifies:
    1. the pre-leak period, preferring the most recent days
    2. the post-leak period, clear of every leak in the scenario
    3. pre- and post-leak combined
    4. none of these: the case is excluded and counted in reports

    8 pure cases have fewer than 14 pre-leak days, and 14 leaks end within a day of the year's end.
  - A baseline never overlaps any timestep where a leak is active.
- **Sample.** A window of ΔH at the sensed junctions, all of it inside the case's active period.
  - Small leaks change pressure by about 1 m, which is close to the 0.5 m step-to-step demand noise. So samples must average over a window. Window length and stride are set in M4.
- **Case-level prediction.** The average of the per-sample class probabilities over the case.
  - **Headline metrics are per case.**
  - Per-sample metrics are secondary.
- **Sensor budget k.** The number of junctions that have a pressure sensor, from 1 to 31.

## 6. Success metrics

How every metric is reported:
- Mean ± 95% bootstrap CI over cases, resampled by scenario and pooled across the 5 outer CV folds.
- **Ties are scored exactly.** If the true pipe shares its score with others, top-k is the expected value under random tie-breaking: clip((k − n_better) / n_tied, 0, 1). Distances for a tied top-1 are averaged over the tied set. So the chance rung scores exactly k/33.
- **Budgets k < 31 before placement exists (M6–M10):** each k is scored over 20 seeded random sensor sets, and the same sets are used for every rung. Per-case metrics are averaged over the sets, and the spread across sets (p5 / p50 / p95) is reported. From M11 on, greedy and heuristic placements are scored too.
- **Tercile edges** (leak flow, SNR) are computed once on the evaluated case set and recorded in each report, so every rung uses the same groups.
- Broken down four ways: **overall**, **by leak type** (small / big), **by case kind** (pure / solo), and **by baseline kind** (pre vs post/combined). A post baseline uses data from after the leak, which an operator would not have in real time, so headline numbers are also shown for pre-baseline cases only.
- The locked test fold is evaluated exactly once, in M13.

### Required deliverables

1. **Metrics.** Report these for every ladder rung, at full budget (k = 31) and at k ∈ {2, 3, 4, 6, 8, 12, 16}:
   - top-1, top-3 and top-5 pipe accuracy
   - mean and median **hop distance**: the shortest path on the pipe line-graph between the predicted pipe and the true pipe
   - mean and median **network distance in metres**: the shortest path along pipes, using nominal lengths, from the predicted pipe's midpoint to the true leak point
2. **Sanity checks that must hold:**
   - The chance rung scores top-1 ≈ 1/33.
   - A model trained on 5 cases overfits them, with train top-1 above 95%.
   - No scenario ID appears in more than one split. This is asserted in code.
   - No feature is built from a `leak_node*` column or from the demand or flow CSVs. This is asserted in code.
3. **Placement curves.** Accuracy vs k for:
   - greedy placement
   - random placement: 100 draws per k, shown as the median with a 5th–95th percentile band
   - a sensitivity-coverage heuristic

   All are evaluated on held-out cases that were not used to choose the sensors.
4. **Reproducibility.**
   - `uv sync` followed by `leaksense run --config configs/<x>.yaml` regenerates a report.
   - On CPU, the same config and seed give the same numbers.
   - Kaggle and local runs agree on the same data subset within the stated tolerance.
5. **Demo.** `uv run leaksense serve` runs the API and the page locally. Every number shown comes from saved evaluation JSON.

### Hypotheses to test

These are not promises. The final report says which ones held.

- **H1.** At k ≥ 8, the GAT beats the best non-learned rung on top-3 accuracy, with non-overlapping CIs.
- **H2.** At small budgets (k ≤ 6), physics inputs improve on the plain GAT.
- **H3.** Greedy placement beats the random median at every k from 2 to 16, and lies above the random 95th percentile for k ≤ 8.
- **H4.** The single masked model gets within 3 points of top-3 accuracy of a model trained for one fixed budget, at k = 4 and k = 8.
- **H5.** Big leaks are close to trivial at k ≥ 8, because the typical big leak (median 980 L/s) causes pressure drops of tens of metres. The headline difficulty number is therefore **small-leak top-3 at k = 8**, and claims about model quality rest mainly on it.
  - Leak sizes overlap across the two types (D11), and the smallest leaks (about 4 L/s) are probably below what pressure sensors can detect. So the final report also plots accuracy against leak flow. This plot is for reporting only; leak flow is never a model input.

## 7. Constraints and configuration

- Python 3.11, with a uv-managed environment, `pyproject.toml` and `uv.lock`.
- CPU build of PyTorch, torch_geometric without compiled extensions, and WNTR.
- Seeds are fixed. Every report records its config and seed.
- **Data location:**
  - The data location is set by the environment variable **`LEAKSENSE_DATA_DIR`**, which must point at the `HanoiOK/` folder.
  - It is read from the process environment or from a git-ignored `.env` file; `.env.example` shows the format.
  - There is no default path.
  - The code exits with a clear error if the variable is unset, or if the folder does not contain `Hanoi.inp`, `Labels.csv` and `Scenario-1/`.
- **Artifacts** go to `LEAKSENSE_ARTIFACTS_DIR`, which defaults to `./artifacts`. That folder is git-ignored.
- The dataset is never copied into the repo or committed. Data, caches, checkpoints and large artifacts are git-ignored, and so is `reference/`.
- A number goes into the README only if a generated report produced it.

## 8. Data contract (verified 2026-11-08)

`leaksense validate-data` (PLAN.md M1) asserts D1–D9 and reports D10–D11 as information. On the real dataset:
- **fast mode:** passes D1–D9 in 38 s
- **`--deep` mode:** passes D1–D9 in 265 s

The dataset manifest sha256 is `6281be1c3cf2…`. The evidence is in `reports/data_inventory.md` and `reports/data_validation*.json`.

| # | Fact |
|---|---|
| D1 | `$LEAKSENSE_DATA_DIR` contains `Hanoi.inp` (the nominal network), `Labels.csv` (`Scenario,Label`: 1 / 244 / 255 scenarios with 0 / 1 / 2 leaks) and `Scenario-1`…`Scenario-500`. |
| D2 | Each scenario folder has `Hanoi_Scenario-N.inp`, `Node_pressures.csv`, `Node_demands.csv`, `Link_flows.csv`, `Labels.csv` and `Scenario-N_info.csv`. Scenarios with leaks also have `Leaks/`, which is absent from Scenario-1. |
| D3 | Every CSV has 17,520 rows at 30-min steps, running from 2024-01-01 00:00 to 2024-12-30 23:30. The timestamp is in the first, unnamed column. Timestamps are identical in every scenario. |
| D4 | Node columns are `2`…`32`, then `leak_node0`[, `leak_node1`], then `1`. Link columns are `1`…`34`, plus one `Xl` (or `Xll`) per leak. |
| D5 | Units: pressure in m, flow and demand in L/s, length in m, diameter in mm, Hazen-Williams roughness. The reservoir pressure is always 0. There are no NaNs or negative pressures, and no junction pressure is ever exactly 0 (checked with `--deep`). |
| D6 | Each leak `Leaks/Leak_leak_nodeK_info.csv` has a header row `Description , Value`, then the keys Leak Node, Leak Area, Leak Diameter, Leak Type (small/big), Leak Start, Leak End (ISO format, End inclusive). Keys and values need whitespace stripped. |
| D7 | `Leaks/Leak_leak_nodeK_demand.csv` (columns `,leak_nodeK`) is non-zero exactly over [Start, End]. The per-scenario `Labels.csv` is the union of all leaks' active periods. |
| D8 | A leak splits pipe X into `X: a→leak_nodeK` and `Xl: leak_nodeK→b`. The label is X. Only pipes 2–34 occur. |
| D9 | Topology and node coordinates are identical to `Hanoi.inp` in every scenario, once the split pipes are merged back. Per scenario, pipe length, diameter and roughness, and junction base demands, each differ from nominal by at most ±20%. Uncertainty levels are 5, 10, 15 or 20% (64 combinations). Demand patterns also differ per scenario. |
| D10 | Coordinates are not to scale: pipe length ÷ coordinate distance ranges from 0.13 to 5.99. All distances use pipe lengths. |
| D11 | Leak flow (the median over each leak's active period) is 3.96–997 L/s for small leaks (median 97) and 69–8,994 L/s for big leaks (median 980). Leak type is the generator's label, not a disjoint size class. Leak flow is ground truth and is used only for reporting, never as an input. |
| D12 | Demands are in L/s and demand-driven: each junction's demand = scenario base demand × pattern, with no pressure dependence. The `leak_node*` columns of `Node_demands.csv` are always 0. Reservoir `1` = −(consumer demand + leak flow), so the file is still an oracle. Total consumer demand over all scenarios and steps: median 760 L/s (p5 355, mean 767, p95 1,228, max 2,330). The nominal `Hanoi.inp` base total is 1,538.6 L/s, about 2× typical demand (multipliers average 0.5). See `reports/data_inventory.md` §0b. |
