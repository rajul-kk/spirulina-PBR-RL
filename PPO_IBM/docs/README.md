# docs/

Start with the three live documents at the top level; everything else is grouped by purpose.

## Live

| file | what it is |
|---|---|
| [`decision_history.md`](decision_history.md) | Primary engineering log. Every non-trivial decision, bug, fix and run result, anchored by `#--slug`. Source comments of the form `(full rationale: docs/decision_history.md#--...)` point here. Entries are historical: paths in them are as they were at the time (see the path map in [`../README.md`](../README.md)). |
| [`known_limitations.md`](known_limitations.md) | Accepted simplifications (`O1`...`O16`) and what each would take to fix. |
| [`USAGE.md`](USAGE.md) | How to run the trainers, launchers and tools. |

## `reports/` — results write-ups

| file | covers |
|---|---|
| [`finalresults.md`](reports/finalresults.md) | PPO era up to v31 (predates TD3+BC and physics v2). |
| [`novelty_report.md`](reports/novelty_report.md) | Novelty and publishability assessment (2026-09-18). |
| [`statistical_validation.md`](reports/statistical_validation.md) | Bootstrap CIs on the PPO/TD-MPC2 held-out sweeps. |
| [`lstm_lru_reset_interval_grid_report.md`](reports/lstm_lru_reset_interval_grid_report.md) | LSTM vs LRU x reset 60/600 (v49-v56). |
| [`entropy_changes_apr_05_2026.md`](reports/entropy_changes_apr_05_2026.md) | PPO entropy-control changes. |
| [`benchmark_results.csv`](reports/benchmark_results.csv) | Early benchmark table. |

## `reference/` — background still in use

| file | covers |
|---|---|
| [`literature.md`](reference/literature.md) | Citations behind each model component. |
| [`real_data_integration.md`](reference/real_data_integration.md) | Calibrating the simulator against real PBR data. |

## `coursework/` — COMP2019 report deliverables

`genetic_env.md` / `light_env.md` and their `.docx` builds (`tools/report/` formats and word-counts
them). Written for the course, not maintained alongside the code.

## `archive/` — superseded, kept for context

| file | why archived |
|---|---|
| [`calibration.md`](archive/calibration.md) | Written for a discontinued Chlorella/BG-11 configuration. |
| [`env_attributes.md`](archive/env_attributes.md) | Describes `light_env.py`, removed 2026-09-23. |
| [`proxy.md`](archive/proxy.md) | Reward terms removed after each caused a mode collapse. |
