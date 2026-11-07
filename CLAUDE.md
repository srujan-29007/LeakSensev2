# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

LeakSense is a clean rebuild of a hackathon project. It uses a sensor-masked GNN to find which pipe is leaking on the Hanoi water network (LeakDB data) and to choose where to place pressure sensors.

- **No code exists yet.** The source of truth is `SPEC.md` (goals, definitions, metrics, data contract) and `PLAN.md` (milestones M0–M15, each with a Verify command).
- Work milestone by milestone, in order. Run each milestone's **Verify** step before moving on.
- If you change a definition, metric or data fact, update SPEC.md in the same change.
- `reports/data_inventory.md` records the verified dataset structure.

## Commands (planned in PLAN.md M0, with uv and Python 3.11, CPU only)

```
uv sync                                  # install (CPU torch wheels)
uv run pytest                            # all tests; uses tests/fixtures, no real data needed
uv run pytest tests/test_metrics.py::test_name   # single test
uv run pytest -m data                    # tests that need the real dataset
uv run ruff check . && uv run ruff format .
uv run leaksense --help                  # CLI: validate-data, build-index, build-features, make-splits,
                                         # eval, train, ablate, place, final-report, serve
```

## Data access

- The dataset lives **outside the repo**. Never copy it in and never commit it.
- Point `LEAKSENSE_DATA_DIR` at the dataset's `HanoiOK/` folder, either in the environment or in a git-ignored `.env` (see `.env.example`). There is no default path. Code must fail loudly if the variable is unset or the folder is invalid.
- Caches and checkpoints go to `LEAKSENSE_ARTIFACTS_DIR` (default `./artifacts`, git-ignored).
- The full dataset is about 11 GB. Inspect it with headers, `head`/`tail` and file sizes, not full reads. Cache only the `Node_pressures.csv` columns `2`…`32`, as float32.

## Invariants (easy to violate; each is enforced by a test or assertion)

- **No oracle inputs.**
  - Never build features from `leak_node*` columns. Those columns are pressure measured at the leak itself.
  - Never use `Node_demands.csv` or `Link_flows.csv` as features. They contain the leak flow.
  - Sensors may only be the 31 junctions `2`…`32`. The reservoir `1` always has pressure 0.
  - The old "physics loss" built from simulated flows was a label leak. Do not bring it back. Physics enters only through inputs computed from the nominal `Hanoi.inp`, such as the WNTR sensitivity matrix.
- **Labels.** The label is the pipe that was split to insert the leak. Pipe `X` becomes `X` + `Xl`, or `Xll` when the same pipe is split twice. There are 33 classes, pipes 2–34, because pipe 1 never leaks. Chance top-1 is therefore 1/33.
- **Cases.** A case is either a pure single-leak scenario or the solo segment of a two-leak scenario: at least 3 days during which only the first leak is active.
  - Splits are grouped by **scenario ID**.
  - Metrics are aggregated **per case**, not per timestep.
  - Every metric is reported overall, by small vs big leak, and by pure vs solo case.
- **ΔH baseline.** Use the same scenario's pre-leak window, matched on time of day and day of week. If there are fewer than 14 days of pre-leak history, fall back to a post-leak window. The baseline must never overlap a timestep where a leak is active.
  - Never use Scenario-1 as a global baseline: every scenario has its own pipe parameters (±20%) and its own demands.
- **Distances.** `.inp` coordinates are not to scale. Distance metrics must use pipe lengths (network distance in metres) or hop counts, never Euclidean distance between coordinates.
- **Locked test fold.** Touch it only in M13.
- **Reported numbers.** Every number in README or reports comes from a generated report. Each report records its config, seed, git commit and data-manifest hash.
- **CPU first.** No GPU-specific code. Every milestone must run on a laptop CPU, on a subset, within minutes. Kaggle runs only call the same CLI with `configs/full.yaml`.

## Planned architecture (PLAN.md has the details)

The pipeline runs in this order: `data/validate` → `data/index` → `data/features` → `data/splits` → `models/*` → `evaluate` → `placement` → `api` + `web/index.html`.

- **`data/index`** produces `artifacts/cases.parquet`. Each case is one row: scenario, kind, pipe, split fraction, leak type, active window and baseline window.
- **`data/features`** produces windowed, averaged ΔH. The averaging matters: small leaks drop pressure by about 1 m, close to the roughly 0.5 m demand noise between timesteps.
- **Model ladder.** Every rung uses the same evaluation harness:
  1. chance / prior
  2. max-|ΔH|
  3. sensitivity-signature matching
  4. MLP / GBM
  5. masked GAT
  6. masked GAT with physics inputs

  Learned models take masked inputs `[ΔH·m, m]`, with the mask size k drawn uniformly from 1 to 31. This lets one model serve any sensor budget.
- **Placement.** Greedy forward selection, with the masked model held fixed, maximizes inner-validation top-3. It is compared against 100 random placements per k and a sensitivity-coverage greedy, on outer held-out cases.
- **API and page.** FastAPI serves only data from saved reports and artifacts. The page `web/index.html` is a single static page in vanilla JS and SVG, with no Node toolchain.

## `reference/`

`reference/` holds the old hackathon repo. It is git-ignored and kept only for comparison. **Do not copy code from it as-is.**

- HEAD of that repo is broken: `ml/` raises ImportError, and the backend and frontend don't run.
- The coherent version is at commit `24c53c9` (`git -C reference/LeakSense show 24c53c9:ml/simulation.py`).
- Its results (about 40% top-1) used nearest-node labels and the Scenario-1 baseline, so they are not comparable with this project's numbers.

## Git rules
- Commit after every logical unit of work: each passing Verify step, each new module
  with its tests, each bugfix. Aim for small, focused commits.
- Before each commit run `git status` and stage specific files; never commit data/,
  reference/, artifacts/, or large files.
- Message format: short imperative subject (e.g. "Add network graph loader"), optional
  body. No Claude attribution, no Co-Authored-By, no "Generated with" lines.
- Never push; I push myself. Never rewrite history.