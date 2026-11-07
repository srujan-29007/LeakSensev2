# Data inventory: LeakDB Hanoi

- **Inspected:** 2026-11-08, by hand and read-only.
- **Method:** directory listings, file sizes, CSV header lines, `head`/`tail` of sample files, and the header sections of `.inp` files. Large files were not loaded in full, except `Node_pressures.csv` (5 MB) for Scenarios 34 and 229.
- **Location:** the dataset is outside the repo and is referenced only through `LEAKSENSE_DATA_DIR`, which points at the `HanoiOK/` folder.
- **Automation:** PLAN.md M1 turned these checks into `leaksense validate-data`. Section 0 has the results.

## 0. Automated validation (M1, 2026-11-08)

`leaksense validate-data` checks the data contract D1–D10 in SPEC.md §8. It only reads the dataset, and a test confirms that no file timestamps change.

| Mode | Command | Result | Time | Report |
|---|---|---|---|---|
| fast | `uv run leaksense validate-data` | D1–D9 PASS, D10 INFO | 38 s | `reports/data_validation.json` |
| deep | `uv run leaksense validate-data --deep --out reports/data_validation_deep.json` | D1–D9 PASS, D10 INFO | 265 s | `reports/data_validation_deep.json` |

- **Dataset manifest:** 4,510 files, 11,123,746,633 bytes. The sha256 of the sorted (path, size) list is `6281be1c3cf2ad971298fdcd672b17e602da7e5dcca86d23c52b924ecf5dccdf`. Later reports cite this hash.
- **What the fast mode checks:**
  - every header
  - the first and last timestamp of every wide CSV
  - every per-scenario `Labels.csv` in full
  - every info file
  - the header and tail sections of every `.inp`
- **What the deep mode adds:**
  - parses every `Node_pressures.csv` and leak-demand file in full
  - counts the rows of `Node_demands.csv` and `Link_flows.csv`

The deep pass corrected two statements from the manual inspection. Sections 2, 3 and 5 below are updated to match.

1. **No junction ever has a pressure of exactly 0.** There are 0 zero cells across all 500 scenarios and all 31 junctions. The 0.00 seen during manual inspection was the reservoir column, which is always 0.
2. **Leak flow varies far more than the two sampled scenarios suggested.** Each figure below is the median flow over a leak's active period.

   | Type | Leaks | Min (L/s) | Median (L/s) | Max (L/s) |
   |---|---|---|---|---|
   | small | 385 | 3.96 | 96.63 | 996.90 |
   | big | 369 | 69.35 | 979.69 | 8,993.84 |

   The two types overlap: some "small" leaks are larger than some "big" ones. Leak type is a generator setting, not a clean size class.

Other values confirmed by the validator:
- Per-scenario base demands deviate up to ±20% from nominal (max 0.19997).
- Pipe length, diameter and roughness deviate up to ±20% (max 0.19993 / 0.19998 / 0.19996).
- Across pipes, pipe length divided by coordinate distance ranges from 0.128 to 5.99 (median 1.70). Coordinates are therefore not to scale.

Scenarios sampled in detail: 1 (no leak), 2 and 250 (two leaks), 34 and 500 (one leak), 229 (small single leak), 273 (shortest leak). The checks on counts, headers, info files and `.inp` header sections ran over all 500 scenarios.

## 0b. Demand audit (2026-11-08)

A read-only script read all 500 `Node_demands.csv` in 65 s.

- **Units are L/s, and demand is demand-driven.** In Scenarios 1 and 34, every junction value equals that scenario's `.inp` base demand (`UNITS LPS`) × its `P_<node>` multiplier, to within 6e-5. Junction demand never reacts to pressure or to the leak.
- **Mass balance.** In all 500 scenarios, reservoir `1` = −(junctions 2…32 + leak flow from `Leaks/*_demand.csv`), to within 6e-5.
- **Leak columns.** All 754 `leak_node*` columns in `Node_demands.csv` are identically 0. The leak flow is only in `Leaks/` and in the reservoir supply.
- **Nominal base demand.** The base demands in `Hanoi.inp` total **1,538.6 L/s**. Each one is the textbook Hanoi demand in m³/h divided by 12.96 (= 3.6²); for example, node 2: 68.672 × 12.96 = 890 m³/h. LeakDB therefore carries 1/3.6 of the textbook demand. The data is consistent with itself, so it is used as it is.
- **Pattern multipliers** average 0.499, and half of them are exactly 0. That is why actual demand is about half the base total.

Total consumer demand (junctions 2…32), in L/s:

| Scope | min | p5 | median | mean | p95 | max |
|---|---|---|---|---|---|---|
| all 500 scenarios × 17,520 steps | 25.1 | 354.9 | 760.1 | 766.7 | 1,227.7 | 2,330.1 |
| per-scenario mean (500 values) | 720.2 | 746.0 | 766.8 | 766.7 | 787.4 | 812.6 |
| Scenario-1 only | 117.2 | 355.7 | 761.3 | 766.8 | 1,223.0 | 2,100.3 |

Mean over the 500 scenarios by period:
- by hour: about 408 L/s (03:00) to 1,014 (10:00)
- by weekday: Mon 769, Tue 771, Wed 782, Thu 776, Fri 825, Sat 749, Sun 695. This is a weekly cycle of about ±8% in the mean.
- by month: 668 (Dec) to 889 (Jul)

## 1. Folder layout

```
<LEAKSENSE_DATA_DIR> = .../hanoidataset/HanoiOK/
  Hanoi.inp              12 KB   nominal network (no demand patterns, DURATION 0, 1 h steps)
  Labels.csv             3 KB    Scenario,Label   (Label = number of leaks)
  Scenario-1 ... Scenario-500
```

- 500 scenario folders with no gaps in the numbering. The folders total about 11 GB in 4,510 files.
- Everything is already extracted. There are no archives.
- Top-level `Labels.csv` counts: **0 leaks: 1** (Scenario-1), **1 leak: 244**, **2 leaks: 255**.

## 2. Files in each scenario

| File | Size | Shape / format |
|---|---|---|
| `Hanoi_Scenario-N.inp` | ~5.3 MB | Full EPANET network with per-scenario parameters and 30-min demand patterns (`P_<node>`), plus the leak junction(s) and split pipe(s) |
| `Node_pressures.csv` | ~5.3–5.6 MB | 17,520 rows. Columns: unnamed timestamp, `2`…`32`, `leak_node0`[, `leak_node1`], `1` |
| `Node_demands.csv` | ~4.1 MB | Same columns as pressures. The reservoir `1` has negative demand (supply). Junction values are exactly base demand × pattern (L/s). The `leak_node*` columns are **always 0** (all 754). The reservoir column carries the leak: `1` = −(junctions + leak flow) |
| `Link_flows.csv` | ~5.8–6.2 MB | 17,520 rows. Columns: unnamed timestamp, `1`…`34`, plus one `<pipe>l` per leak |
| `Labels.csv` | 438,017 B (same for all 500) | `Timestamp,Label`, with Label 0.0/1.0 = "any leak active" |
| `Scenario-N_info.csv` | ~215 B | `Description , Value` rows: Network_Name, Duration (8760 hours), Time_Step (30.0 min), Uncertainty_Topology (NO), Uncertainty_Length/Diameter/Roughness (%) |
| `Leaks/` | none in Scenario-1 | One `Leak_leak_nodeK_info.csv` and one `Leak_leak_nodeK_demand.csv` per leak |

Time:
- 30-minute steps from `2024-01-01 00:00:00` to `2024-12-30 23:30:00`, 17,520 rows. 2024 is a leap year, so the data ends on Dec 30.
- Timestamps are identical across scenarios. Checked on Labels.csv content for sampled scenarios, and by identical file size for all 500.

Units, taken from `[OPTIONS] UNITS LPS`:
- Flow and demand are in L/s.
- Pressure is in m.
- Pipe length is in m and diameter in mm.
- Hazen-Williams roughness.

Values (deep validation, all 500 scenarios):
- No NaNs or blanks.
- No negative pressures.
- No junction pressure is ever exactly 0. *Correction: the manual inspection wrongly reported zero pressures; it was reading the reservoir column.*
- All junction elevations are 30 m. The reservoir head is 100 m, and its pressure column is always 0.

Header shapes across all 500 scenarios (node columns, link columns):
- (32, 34) for 1 scenario
- (33, 35) for 244 scenarios
- (34, 36) for 255 scenarios

Each leak adds exactly one node column and one link column.

## 3. Leak ground truth

`Leaks/Leak_leak_nodeK_info.csv` is a key/value file whose first row is the header `Description , Value`. All 754 leak files have exactly these keys:

```
Leak Node , leak_node0            <- the inserted junction, NOT an original node
Leak Area , 0.0727764319661792    <- m^2
Leak Diameter , 0.3044041875077182
Leak Type , big                   <- small (385) | big (369)
Leak Start , 2024-11-17 18:00:00
Leak End , 2024-12-09 10:30:00    <- inclusive
```

- **Leak flow.** `Leak_leak_nodeK_demand.csv` has columns `,leak_nodeK` and 17,520 rows. The value is non-zero exactly from Leak Start to Leak End, both inclusive. Checked on scenarios 34, 229 and 273.
  - Within a single leak, the flow is roughly constant. Examples: scenario 229 (small) runs at about 109 L/s, and scenario 34 (big) at about 650–700 L/s.
  - Across leaks the flow varies widely (deep validation, median over each leak's active period):
    - small: 3.96–997 L/s, median 97
    - big: 69–8,994 L/s, median 980
    - The two types overlap. For scale, total consumer demand has a median of 760 L/s and a mean of 767 L/s over all scenarios and timesteps (§0b). *Correction: an earlier "about 640 L/s" was one timestep, the Scenario-1 reservoir supply at 2024-01-01 00:00 (−644 L/s).*
  - Leak area: small 0.0001–0.042 m² (median 0.0035), big 0.0026–0.78 m² (median 0.070).
- **Link to `Labels.csv`.** The per-scenario `Labels.csv` is 1 exactly when at least one leak is active, so it is the union of the leaks' active windows. The top-level `Labels.csv` value equals the number of leak info files in all 500 scenarios.
- **Which pipe leaks.** Each leak is a new junction inserted by splitting a pipe. In the scenario `.inp`, pipe `X: a→b` becomes `X: a→leak_nodeK` and `Xl: leak_nodeK→b`. Both pieces keep X's diameter and roughness.
  - The label is pipe X.
  - The leak position along the pipe is `len(X) / (len(X) + len(Xl))`.
  - When the same pipe is split twice (6 two-leak scenarios: 7, 46, 269, 297, 441, 474), the pieces are `X`, `Xl` and `Xll`.
- **Pipes covered.** Single-leak scenarios cover **33 of the 34 pipes. Pipe 1 (reservoir→2) never leaks.**
  - Scenarios per pipe range from 2 (pipes 2 and 6) to 16 (pipe 30). See the M3 index for the full table.
- **Timing.**
  - Over all leaks, duration runs from 0 h (scenario 273, a two-leak scenario whose second leak lasts one timestep) to 344 days.
  - Single-leak scenarios: minimum 18 h, 2 under 24 h, 11 under 7 days, median 85 days.
  - Leak start times are spread across the whole year, 53–82 per month.
  - Pre-leak history in single-leak scenarios: 1 scenario has under 1 day, 4 under 7 days, 8 under 14 days.
- **Two-leak scenarios.** 245 of the 255 have at least 3 days in which only the first leak is active (a "solo segment"). The first leak is the earliest to start; in 126 of the 255 that is `leak_node1`, not `leak_node0`.

## 4. Are the .inp files the same across scenarios?

**Topology yes, parameters no.**

Identical across all 500 scenarios and the nominal file:
- Topology. After merging each `X`/`Xl`(/`Xll`) chain back into one pipe, the pipe endpoints match `Hanoi.inp` in every scenario.
- The `[COORDINATES]` of the original nodes.
- `[RESERVOIRS]` and `[OPTIONS]`.

Identical across scenarios but different from the nominal file:
- `[TIMES]`: 30-minute steps and 8,760 h, against 1 h steps and duration 0 in the nominal file.

Different in each scenario:
- Pipe length, diameter and roughness, each perturbed by up to ±20% from nominal (observed range −0.200 to +0.200).
  - The uncertainty level per parameter is 5, 10, 15 or 20%, with 64 distinct combinations.
  - The old notebooks claimed a uniform ±10%.
- Junction base demands, plus each scenario's own 30-minute demand pattern for every junction.
- The leak junctions and split pipes.

The nominal `Hanoi.inp` is the right network for the graph and for the sensitivity simulations.

## 5. Inconsistencies, gaps and surprises

1. **Oracle columns.** `Node_pressures.csv` includes `leak_nodeK` columns, which means a "sensor" sits on the leak itself. In `Node_demands.csv` those columns are all 0, but its reservoir column equals −(consumer demand + leak flow), so the file is still an oracle. `Link_flows.csv` includes the split-pipe flows. All of these must be excluded from features.
2. **Big leaks are often extreme.** Median flow is 980 L/s, above the mean total consumer demand of 767 L/s, and the largest reaches 8,994 L/s. In Scenario 34 (pipe 22, about 665 L/s) the end node's mean pressure drops by **61 m**, its neighbour's by 33 m, and every other node's by at most 4 m. These leaks will be easy to locate with dense sensors, so headline metrics must be broken down.
3. **Small leaks are close to the noise.** In Scenario 229 (pipe 32, about 109 L/s, close to the small-leak median) the largest mean pressure drop is **1.25 m**. Pressure moves about 0.5 m between consecutive timesteps because of stochastic demands; many junction demands are exactly 0 at a given step.
   - Single timesteps are therefore noisy, and windowing or averaging is needed.
   - The smallest leaks (around 4 L/s) are likely impossible to locate from pressure alone.
4. **Leak type does not cleanly indicate leak size.** The small and big flow ranges overlap: small leaks reach 997 L/s, while big leaks start at 69 L/s. Breaking results down by type alone can hide the effect of size.
5. **Coordinates are not to scale.** Pipe 2 is 1,350 m long, but its endpoints are about 700 coordinate units apart. Euclidean distance in coordinate units is not a physical metric, so use network distance along pipes in metres.
6. **No common baseline.** Each scenario has its own pipe parameters, base demands and demand patterns. Subtracting Scenario-1's pressures, as the old code did, mixes all of these differences into ΔH.
7. **Scenario-1 has no `Leaks/` folder.** The old code assumed an empty folder; code must cope with it being absent.
8. **The leak info CSV has a header row** (`Description , Value`), and its keys and values carry stray spaces. The old `pd.read_csv` call happened to work because of the header row.
9. **Too little pre-leak history in a few cases.** 8 single-leak scenarios have under 14 days before the leak starts, so they need a post-leak baseline as a fallback.
10. **Pipe 1 is never a label,** so there are 33 classes and chance accuracy is 1/33. Pipes 2 and 6 have only 2 pure single-leak scenarios each, plus 2 and 6 solo segments respectively. *Correction (M3): the hand count said 5 for pipe 6; it missed solo leaks on an `Xll` piece. `reports/case_index.json` has the full table.*
11. **The second leak in Scenario 273 lasts zero length** (start equals end). Harmless here, since only the first leak's solo segment is used.

## 6. Status of the SPEC assumptions A1–A14 (previous draft)

| ID | Old assumption | Status | Finding |
|---|---|---|---|
| A1 | About 500 `Scenario-<n>` folders, possibly under `HanoiOK/` | **PASS** | Exactly 500, under `HanoiOK/` |
| A2 | Top-level `Labels.csv` with `Scenario,Label` = number of leaks | **PASS** | Counts are 1 / 244 / 255 for 0 / 1 / 2 leaks |
| A3 | The expected files are in every scenario | **AMENDED** | All present, but Scenario-1 has no `Leaks/` folder at all |
| A4 | Wide CSVs with node and link column names as listed | **AMENDED** | Node order is `2…32, leak_node*, 1`. There is **one** extra link column per leak (`Xl`, and `Xll` for a second split of the same pipe), not two |
| A5 | 17,520 steps at 30 min, timestamps aligned | **PASS** | Runs 2024-01-01 to 2024-12-30 23:30 |
| A6 | Info CSV with key/value rows and the 6 listed keys | **PASS (detail)** | Has a header row and stray spaces in keys and values. Leak Node is `leak_nodeK`, not an original node. Dates are ISO |
| A7 | Demand CSV holds leak flow; "active" means non-zero; units unknown | **PASS / AMENDED** | Non-zero exactly over [Start, End]. Units are L/s |
| A8 | Leak = split pipe, recoverable as the label | **PASS** | From the `Xl` column name or the scenario `.inp`. The split fraction gives the exact position |
| A9 | 32 nodes, 34 pipes, no pumps or valves | **PASS** | Topology is identical in all 500 scenarios |
| A10 | Each scenario has ±10% uncertainty and its own demands | **AMENDED** | The level is 5/10/15/20% per parameter (64 combinations), with deviations up to ±20%. Base demands and patterns differ per scenario |
| A11 | Scenario 1 has no leak; about 244 single and 255 two-leak | **PASS** | Exactly those counts |
| A12 | Every single-leak scenario has enough pre-leak history | **FAIL (minor)** | 8 of 244 have under 14 days, so a fallback is needed |
| A13 | Pressure in m, coordinates in .inp units | **PASS / AMENDED** | Pressure in m. Coordinates are not to scale; use pipe lengths in m |
| A14 | Old repo's `data/Hanoi.inp` is a scenario `.inp` | **PASS** | The dataset root has the 12 KB nominal `Hanoi.inp`; use it |
