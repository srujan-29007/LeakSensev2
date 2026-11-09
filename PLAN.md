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

### M7: Sensitivity matrix (physics input) — DONE
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
- **Verified:**
  - **`build-sensitivity`:** 201 EPANET runs in about 20 s; α = 0.231 / 0.494 / 0.798; leak sizes 95.3 / 949.2 L/s; every entry of S ≤ 0. There are 19 negative-pressure warnings, all at the big size on dead-end or thin-branch pipes.
  - **Level stability:** the smallest cosine is 0.978 (< 0.99), so the **levels differ**. `sigmatch` uses mid (the plan's default) and `sigmatch_all` takes the max over every size and level. M10 should start from all levels.
  - **Units:** a hand Hazen-Williams check of node 2's drop agrees with EPANET to within 1%.
  - **Scenario 34:** the observed ΔH is most cosine-similar to pipe 22's column.
  - **Results** at k = 8 (top-3 and small-leak top-3, %):

    | rung | top-3 | small-leak top-3 |
    |---|---|---|
    | `maxdh` | 43.1 | 43.1 |
    | `sigmatch` | 75.1 [72.3, 78.1] | 64.6 |
    | `sigmatch_all` | 78.4 [75.7, 81.2] | 68.5 [63.8, 72.7] |

  - **At k = 31:** top-1 is 33.7% for `maxdh`, 58.0% for `sigmatch` and 64.0% for `sigmatch_all`. Top-3 is about the same for all three (88.5–91.2%).
  - 9 sensitivity tests pass, plus a real-data test.

### M8: Learned tabular rungs (MLP, GBM) with masking — DONE
- Input is `[ΔH·m, m]` over the 31 junctions, after the M4 input transform. The mask size k is drawn uniformly from 1 to 31. The output is 33 classes.
- Class-weighted loss, with the weights computed on each training fold. *(data: 4–26 cases per pipe)*
- Case-weighted sampling from M4.
- **Verify:**
  - `uv run leaksense train --config configs/mlp_cpu.yaml` finishes in under 10 minutes on CPU.
  - The model overfits a set of 5 cases.
  - Accuracy-vs-k curves, with all M6 breakdowns, are written to `reports/`.
- **Harness changes:** the model is fitted once per fold and then scored at every k and mask draw (it was refitted per k). `fit` now gets the training cases and their windows, and a model may define `batch_scores`. The `model:` section of a run config is left out of the feature hash, so model configs share the feature cache.
- **Protocol (both rungs):**
  - Sample weight = (1 / windows in its case) × inverse-frequency class weight from the training fold.
  - Settings given as lists in the config are chosen per fold on an inner validation split (about 15% of each pipe's training scenarios, grouped by scenario; pipes with under 7 training cases stay in training). Early stopping uses the same split; the model is then refitted on the whole training fold for the chosen number of epochs or rounds.
  - A case's score is the mean of its windows' log-probabilities.
  - MLP: 2 × 256, dropout 0.2, a fresh mask per sample and epoch. GBM: LightGBM, each training window repeated with 4 random masks, 4 threads.
- **Verified** (419 CV cases, commit `2acb356`):
  - `train --config configs/mlp_cpu.yaml` took 590 s and `configs/gbm_cpu.yaml` took 242 s, run at the same time on one laptop CPU.
  - Both overfit 5 cases; 14 tests in `tests/test_learned.py` pass.
  - **Input transform:** inner validation chose `scale_magnitude` in all 5 folds for both models. For trees `signed_log1p` is identical to `none` (a monotone per-feature transform does not change splits), so the GBM grid leaves it out.
  - **Results** (%, [95% CI]):

    | rung | k = 8 top-3 | k = 8 small-leak top-3 | k = 31 top-1 | k = 31 top-3 |
    |---|---|---|---|---|
    | `maxdh` | 43.1 [41.0, 45.0] | 43.1 | 33.7 | 88.8 |
    | `sigmatch_all` | **78.4** [75.7, 81.2] | **68.5** | 64.0 [59.4, 68.5] | **91.2** |
    | `mlp` | 63.3 [60.2, 66.3] | 53.2 | 53.7 [48.9, 58.5] | 88.5 |
    | `gbm` | 58.9 [56.1, 61.7] | 51.0 | **69.7** [65.2, 74.2] | 90.0 |

  - **Reading:** with about 335 training cases for 33 classes, neither tabular model gets close to `sigmatch_all` at small k: with few sensors, a physics signature beats what 335 cases can teach. With all 31 sensors the GBM has the best top-1 (69.7 vs 64.0; the CIs overlap). Its largest gains are on the biggest leaks (top leak-flow tercile, k = 31 top-1 83.6 vs 73.6). The MLP is better than the GBM at small k and worse at k = 31. That gap is the case for M10's S-based prior.

### M9: Masked GAT — DONE
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
- **Built:** `models/gat.py` reuses the M8 protocol: `MaskedGAT` subclasses `MaskedMLP` and only swaps the network. The MLP's training loop now takes a network factory; MLP scores stayed bit-identical.
  - Node input `[T(ΔH·m), m]`, with an 8-d learned embedding per node (positional on the fixed nominal graph). The `scale_magnitude` magnitude goes to the pipe head as a global input.
  - Edge features: log of nominal length, diameter, roughness and resistance, z-scored over the pipes.
  - Pipe head: an MLP on `[h_u + h_v, |h_u − h_v|, edge features, magnitude]`, symmetric in u and v.
  - `gat_cpu.yaml` fixes `transform: scale_magnitude` (M8 chose it in every fold for both models) instead of running a grid.
- **Verified** (419 CV cases, re-run at commit `81750f7`, `reports/gat.json`):
  - **Layout bug, fixed.** The first run (`ffb1115`) read the encoded row with the wrong offsets under `scale_magnitude`: junction 2's mask channel held the log magnitude, the other junctions got a neighbour's mask bit, and the pipe head's magnitude was junction 32's mask bit. No unsensed data leaked in (the masking test passed), but it was not the model described. `81750f7` reads the row through one helper (`mlp.split_encoded`) and adds a layout test that fails on the old code. All numbers below are from the re-run.
  - 13 tests in `tests/test_gat.py` pass: shapes, input layout, masked-junction invariance for every transform, reservoir input is always zero, overfits 5 cases, deterministic, CLI.
  - The fold-0 smoke run took 683 s (`reports/gat_fold0.json`). The full 5-fold `train --config configs/gat_cpu.yaml` took 3,437 s on a laptop CPU, about 6× the MLP's 590 s. Early stopping chose 34–61 epochs (the buggy run chose 56–85).
  - **Results** (%, [95% CI]):

    | rung | k = 8 top-3 | k = 8 small-leak top-3 | k = 31 top-1 | k = 31 top-3 | k = 31 small-leak top-3 |
    |---|---|---|---|---|---|
    | `sigmatch_all` | **78.4** [75.7, 81.2] | **68.5** [63.8, 72.7] | 64.0 [59.4, 68.5] | 91.2 [88.5, 93.8] | 83.5 [78.4, 88.1] |
    | `mlp` | 63.3 [60.2, 66.3] | 53.2 [48.4, 57.8] | 53.7 [48.9, 58.5] | 88.5 [85.4, 91.6] | 80.3 [74.8, 84.9] |
    | `gbm` | 58.9 [56.1, 61.7] | 51.0 [47.0, 55.0] | **69.7** [65.2, 74.2] | 90.0 [86.9, 92.8] | 82.1 [77.1, 86.7] |
    | `gat` | 67.9 [65.4, 70.2] | 63.4 [59.4, 67.3] | 69.5 [64.4, 74.0] | **95.9** [93.8, 97.9] | **92.7** [89.0, 95.9] |

  - **The comparisons are like-for-like.** Every rung at a given k is scored on the same 20 random sensor sets (`evaluate.random_masks`, seeded by evaluation seed 0 and k), and `sigmatch_all` at k = 8 uses those 8 sensors only; "all" means all 6 columns of S, not all junctions.
  - **H1** (GAT vs the best non-learned rung, top-3, non-overlapping CIs), CV only:
    - k = 8: **fails**, and the GAT is worse (67.9 vs 78.4, the CIs do not overlap).
    - k = 12 (82.5 vs 83.9) and k = 16 (88.0 vs 86.5): the CIs overlap.
    - k = 31: the CIs touch (GAT lower bound = `sigmatch_all` upper bound = 93.8), so not strictly non-overlapping.
  - **Reading:**
    - **All 31 sensors:** the GAT has the best top-3 and small-leak top-3 (92.7 vs 83.5 for `sigmatch_all`, CIs do not overlap), and the shortest median distance, 362 m (vs 379 for `gbm`, 432 for `sigmatch_all`). Top-1 is level with the GBM (69.5 vs 69.7). In the top leak-flow tercile the GBM has the better top-1 (83.6 vs 77.1).
    - **At k = 8:** the GAT is now ahead of the MLP (67.9 vs 63.3) and clearly ahead on small leaks (63.4 vs 53.2), but still well below `sigmatch_all` (78.4). The S prior in M10 is aimed at this gap.
    - **Training variance is large.** The fixed fold 0 alone barely moved (k = 8 top-3 63.9 → 63.0), while the pooled numbers moved by several points and the chosen epochs changed a lot. Single-seed differences of a few points between learned rungs should not be read as real. M10 runs 3 seeds on its final variants.

### M10: Physics-input GAT and ablation
- **Variants** (SPEC §5 "Physics inputs"), all on the M9 model block. The first 6 were fixed before any M10 run; the last 2 were added after the fold-0 dev run (see below):

  | variant | change |
  |---|---|
  | `gat` | plain (M9) |
  | `gat_noembed` | no learned node embedding |
  | `gat_snode` | S as a node feature (pattern imputed from S) |
  | `gat_sedge` | S as edge features (per-pipe cosines) |
  | `gat_sprior` | S as a per-pipe prior logit (a · max cosine) |
  | `gat_sprior_mid` | `gat_sprior` with the mid demand level only |
  | `gat_snode_noembed` | `gat_snode` without the node embedding (added after fold 0) |
  | `gat_sprior_noembed` | `gat_sprior` without the node embedding (added after fold 0) |

- **Masking.** S enters only through the masked cosine (sensed junctions only, for both the pattern and S). `tests/test_gat.py` runs the M9 masking test on all 6 variants × transforms, checks that the cosine ignores unsensed rows of ΔH and of S, that it equals `sigmatch_all` for one window, and that `gat_sprior` with a zero head ranks exactly like `sigmatch_all`.
- **Same everything.** `leaksense ablate` runs each (variant, model seed) through the `train` harness with the same splits, features, k list, evaluation seed and 20 draws, so every run sees the same sensor sets; the aggregator refuses runs that disagree. Comparisons are paired per case (same cases, same sensor sets), on per-case metrics averaged over model seeds, with a bootstrap over scenarios. The reference rung `sigmatch_all` is scored in the same run.
- **Breakdowns:** overall, leak type, case kind, leak-flow tercile, pre-baseline only. The report gives an H2 verdict (paired top-3 vs `gat`, k ≤ 6) and an H1 re-check (k ≥ 8) for every variant.
- **Fold-0 dev run — DONE** (`reports/ablation_fold0.md`, 84 cases, model seed 0, commit `ba49c3f`). Directional only:
  - The plain `gat` row reproduced `reports/gat_fold0.json` exactly.
  - **Removing the learned node embedding helped more than any physics input:** paired top-3 vs `gat` was +4.5 to +8.9 points at every k from 2 to 16, CIs above 0. `gat` early-stopped at 34 epochs, `gat_noembed` at 119.
  - **Physics at k ≤ 6 (H2):** `gat_snode` (+8 to +14 points), `gat_sprior` and `gat_sprior_mid` (+4.5 to +8.9) were better than `gat` at every k ≤ 6; `gat_sedge` at k = 3–6 only. None helped at k ≥ 8, and none reached `sigmatch_all` at k = 8.
  - `gat_sprior` and `gat_sprior_mid` agreed within 1 point everywhere: the demand levels make no difference here.
  - Two runs took about 11,000–13,000 s instead of about 1,000 s (the laptop was idle or asleep); the times are not model costs.
- **Revised after the fold-0 dev run (2026-11-08), before any full-run result existed:**
  - **Variant list:** + `gat_snode_noembed` and `gat_sprior_noembed` (physics without the node embedding), because every physics variant kept the embedding that fold 0 suggests hurts. 8 variants in `configs/ablation_full.yaml`.
  - **Stage-2 rule replaced.** The old rule (best seed-0 small-leak top-3 at k = 8) used outer-fold results, would have picked two near-identical variants, and could not pick `gat_noembed`.
  - **Runs are saved per (variant, seed, fold)** as each fold finishes, so an interrupted run resumes from the next fold. The aggregator leaves out (and lists) any (variant, seed) missing a fold.
- **Stage 1:** all 8 variants, model seed 0, 5 folds.
- **Stage 1 — DONE** (`reports/ablation.md`, 419 CV cases, 41 of 41 runs, model seed 0, runs at `522e253`/`ef489c7`). **Single seed:** the CIs are bootstraps over scenarios and do not include training-seed variance, which M9 showed can move pooled numbers by several points. Top-3, all cases (%):

  | variant | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 | k=31 |
  |---|---|---|---|---|---|---|---|---|
  | `gat` | 17.4 | 26.9 | 32.6 | 51.1 | 67.9 | 82.5 | 88.0 | 95.9 |
  | `gat_noembed` | 19.5 | 29.4 | 37.3 | 53.8 | 68.4 | 83.5 | 89.2 | 97.6 |
  | `gat_snode` | 26.5 | 39.3 | 46.8 | 60.2 | 71.5 | 83.5 | 88.2 | 95.2 |
  | `gat_sedge` | 23.4 | 34.7 | 42.7 | 57.8 | 70.3 | 83.3 | 88.1 | 94.5 |
  | `gat_sprior` | 21.5 | 32.5 | 38.7 | 55.0 | 68.3 | 82.3 | 87.6 | 95.7 |
  | `gat_sprior_mid` | 21.5 | 33.3 | 39.6 | 54.9 | 69.0 | 82.8 | 87.6 | 95.0 |
  | `gat_snode_noembed` | 29.2 | 43.3 | 49.2 | 64.7 | 76.3 | 87.0 | 92.0 | 97.6 |
  | `gat_sprior_noembed` | 22.6 | 35.1 | 41.7 | 57.7 | 69.5 | 84.0 | 89.8 | 97.9 |
  | `sigmatch_all` | 34.2 | 51.5 | 57.9 | 70.0 | 78.4 | 83.9 | 86.5 | 91.2 |

  Paired top-3 differences below are in points, with 95% CIs; "pooled" is the per-case mean over the listed k. The pooled and per-fold figures were computed from `reports/raw/ablation/*.parquet` with the aggregator's bootstrap (1000 resamples, seed 0), and the per-k values match `reports/ablation.md`.
  - **`gat_noembed` − `gat`:** pooled k = 2–16: +2.1 [+0.9, +3.3]. Better in 3 of 5 folds (f0 +6.4, f1 −0.5, f2 −0.4, f3 +1.9, f4 +3.1). At k = 8: +0.5 [−1.3, +2.4]. Fold 0 carries the effect: its CI is above 0 at every k from 2 to 16. At k = 8, fold 1 is −3.5 [−6.4, −0.4]. Fold 0 is also the dev fold on which the effect was first seen. **Adoption criterion: pending** (it needs seeds 0–2). On seed 0 alone it would fail: the k = 8 lower bound is −1.3, not above 0. Every point estimate is ≥ 0.
  - **H2, physics vs its no-physics twin:**
    - `gat_snode` − `gat`: k = 2–6 +11.2 [+9.4, +13.2]. Better in 5 of 5 folds, and each fold's CI is above 0 at every k ≤ 6. At k = 8, 12, 16, 31: +1.0 [+0.0, +2.1], better in 4 of 5 folds, no fold's CI above 0. On small leaks it is worse at k = 8–16 (k = 8: −3.2 [−5.6, −0.7]).
    - `gat_snode_noembed` − `gat_noembed`: k = 2–6 +11.6 [+9.8, +13.3]. Better in 5 of 5 folds, and each fold's CI is above 0 at every k ≤ 6. At k = 8, 12, 16, 31: +3.5 [+2.7, +4.4], better in 5 of 5 folds. This comes from k = 8–16 (k = 8: +7.9 [+6.4, +9.6]); at k = 31 it is +0.0.
    - **H2 is supported on seed 0, in every fold.** Every physics variant's CI vs `gat` is above 0 at every k ≤ 6.
  - **H1, best learned (`gat_snode_noembed`) vs `sigmatch_all`:**
    - k = 8: 76.3 vs 78.4, paired −2.1 [−4.5, +0.3]. It is below in all 5 folds. **Not met.**
    - k = 12: 87.0 vs 83.9, paired +3.1 [+1.0, +5.2]. The marginal CIs overlap, so **not met** under SPEC's non-overlap rule.
    - k = 16: 92.0 [90.3, 93.4] vs 86.5 [83.9, 89.0]. **Met.** Above in 5 of 5 folds.
    - k = 31: 97.6 vs 91.2. **Met.** Above in 5 of 5 folds.
    - H1 is also met at k = 31 by `gat_noembed` and `gat_sprior_noembed`. No other variant meets it at any k.
    - `gat_snode_noembed` was added after the fold-0 dev run and is the best of 8 variants picked on these same CV cases, so this H1 result is optimistic.
  - **Stage-2 rule** (inner validation; scores 53.3 `gat_snode`, 52.4 `gat_snode_noembed`, 49.3 `gat_sedge`, ...) picked **`gat`, `gat_noembed`, `gat_snode`, `gat_snode_noembed`**.
  - **Timing:** wall-clock times include sleep. Four fold-runs took 8,600–37,200 s instead of about 1,000 s (`gat_noembed` f0, `gat_sedge` f4, `gat_snode` f0, `gat_snode_noembed` f1). Runs on 2026-11-08 used 2 workers (about 13–35 s/epoch). Runs on 2026-11-09 were serial (about 8–10 s/epoch).
- **Stage-2 selection rule (inner validation only; implemented in `ablate.stage2_selection`):**
  - Always `gat` and `gat_noembed`, plus the **2 physics variants** (any `physics` other than `none`, with or without the embedding) with the highest inner-validation score.
  - **Data:** the seed-0 run of each variant. In each of the 5 outer folds, the net chosen on that fold's inner-validation split (the early-stopping net, before the refit on the whole training fold) is scored on that fold's inner-validation cases (about 15% of each pipe's training scenarios). The outer-fold cases are never used. The sensor sets are the harness's 20 seeded random sets per k (`evaluate.random_masks`, seed 0).
  - **Metric:** case-level top-3 (ties scored exactly, as in SPEC §6), per k. For each k, the case-weighted mean over the 5 folds; the score is the mean of that over **k ∈ {2, 3, 4, 6, 8}**.
  - **Tie-break**, in order: (1) the score, compared to 0.1 point; (2) inner-validation small-leak top-3 at k = 8, case-weighted over folds, to 0.1 point; (3) fewer added inputs: `prior` < `node` < `edge`, and mid-only levels before all levels; (4) variant name.
- **Stage 2:** seeds 1 and 2 for the 4 selected variants, 5 folds.
- **Criterion for adopting `node_embed: 0` as the default (M11 onwards):** with seeds 0, 1 and 2 of both `gat` and `gat_noembed` over all 5 folds, take the paired top-3 difference `gat_noembed` − `gat` (all cases, per-case metrics averaged over the 3 seeds, bootstrap over scenarios). **Adopt** if its 95% CI lower bound is above 0 at k = 8 **and** its point estimate is ≥ 0 at every k in {2, 3, 4, 6, 8, 12, 16, 31}. Otherwise keep the embedding. Implemented in `ablate.noembed_adoption` and reported in `reports/ablation.md`. If it is adopted, the noembed form of the chosen physics variant is the M11 model when both forms were run.
- **Compute** (laptop, 14 cores; plain GAT 683 s for fold 0 and 3,437 s for 5 folds). Stage 1 is 40 fold-runs, about 8–9 h serial, about 4.5 h as 2 workers × 4 threads. Stage 2 is 4 × 2 × 5 = 40 fold-runs, about the same. On Kaggle (4 vCPU, measured in M12), each stage is 1–2 sessions; split it with `--variants` and copy `reports/raw/ablation/` back.
- **Verify:** `uv run leaksense ablate --config configs/ablation_full.yaml --variants … --no-aggregate` (stage 1, then stage 2 with `--seeds 1,2`), then `--aggregate-only` writes `reports/ablation.{json,md}`. The fold-0 dev run is `configs/ablation_cpu.yaml` → `reports/ablation_fold0.md`; its raw runs predate the per-fold layout.

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
