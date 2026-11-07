# Data inventory: LeakDB Hanoi

- **Inspected:** 2026-11-08, by hand and read-only.
- **Method:** directory listings, file sizes, CSV header lines, `head`/`tail` of sample files, and the header sections of `.inp` files. Large files were not loaded in full, except `Node_pressures.csv` (5 MB) for Scenarios 34 and 229.
- **Location:** the dataset is outside the repo and is referenced only through `LEAKSENSE_DATA_DIR`, which points at the `HanoiOK/` folder.
- **Automation:** PLAN.md M1 turns these checks into `leaksense validate-data`.

Scenarios sampled in detail: 1 (no leak), 2 and 250 (two leaks), 34 and 500 (one leak), 229 (small single leak), 273 (shortest leak). The checks on counts, headers, info files and `.inp` header sections ran over all 500 scenarios.

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
| `Node_demands.csv` | ~4.1 MB | Same columns as pressures. The reservoir `1` has negative demand (supply). The `leak_node*` columns hold the leak outflow |
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

Values:
- No NaNs or blanks in the sampled pressure files.
- No negative pressures.
- Some pressures are exactly 0.00. This is probably pressure-deficient behaviour during big leaks; to be quantified in M1.
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
  - Small leaks are roughly constant at about 109 L/s (scenario 229).
  - Big leaks run at about 650–700 L/s (scenario 34), similar to the whole network's demand of about 640 L/s.
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
- **Two-leak scenarios.** 245 of the 255 have at least 3 days in which only the first leak is active (a "solo segment").

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

1. **Oracle columns.** `Node_pressures.csv` and `Node_demands.csv` include `leak_nodeK` columns, which means a "sensor" sits on the leak itself. `Link_flows.csv` includes the split-pipe flows. All of these must be excluded from features.
2. **Big leaks are extreme.** In Scenario 34 (pipe 22, big leak) the end node's mean pressure drops by **61 m**, its neighbour's by 33 m, and every other node's by at most 4 m. Big leaks will be trivial to localise with dense sensors, so the headline metrics must be split by leak type.
3. **Small leaks are close to the noise.** In Scenario 229 (pipe 32, small leak), the largest mean drop is **1.25 m**. Pressure moves about 0.5 m between consecutive timesteps because of stochastic demands; many junction demands are exactly 0 at a given step. Single timesteps are therefore noisy, and windowing or averaging is needed.
4. **Coordinates are not to scale.** Pipe 2 is 1,350 m long, but its endpoints are about 700 coordinate units apart. Euclidean distance in coordinate units is not a physical metric, so use network distance along pipes in metres.
5. **No common baseline.** Each scenario has its own pipe parameters, base demands and demand patterns. Subtracting Scenario-1's pressures, as the old code did, mixes all of these differences into ΔH.
6. **Scenario-1 has no `Leaks/` folder.** The old code assumed an empty folder; code must cope with it being absent.
7. **The leak info CSV has a header row** (`Description , Value`), and its keys and values carry stray spaces. The old `pd.read_csv` call happened to work because of the header row.
8. **Too little pre-leak history in a few cases.** 8 single-leak scenarios have under 14 days before the leak starts, so they need a post-leak baseline as a fallback.
9. **Pipe 1 is never a label,** so there are 33 classes and chance accuracy is 1/33. Pipes 2 and 6 have only 2 pure single-leak scenarios each, plus 2 and 5 solo segments respectively.
10. **The second leak in Scenario 273 lasts zero length** (start equals end). Harmless here, since only the first leak's solo segment is used.

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
