# Comparison protocol: LLM-written controllers, CMA-ES and TD3 (LRU vs LSTM) on physics v3

Written 2026-09-30, before any arm below was run on the `final` split. Once an arm's final
controller exists it is frozen; nothing is changed after its `final` result is seen.

## 1. Questions
1. Do Claude-written controllers beat classical tuning of the existing control law (CMA-ES)?
2. Does source access help (white-box vs black-box writer)?
3. With identical training, does an LRU core beat an LSTM core for TD3?
4. How do the learned policies compare with the written controllers and CMA-ES?

## 2. Common ground
- Simulator: physics v3 (branch `worktree-agent-a87c1f2edcbd1556b`; RL runs pinned to tag
  `rl-protocol-v1`). Every controller acts on the same 6 sensor readings (turbidity, pH, pump
  volume, conductivity, temperature, lux) through `harness.py`; the oracle expert is the only
  exception and is a reference, not an arm.
- Task: difficulty D2, 30-day batch, score = harvested biomass (g).

## 3. Arms and budgets

| Arm | Runs | Budget per run | Output evaluated |
|---|---|---|---|
| White-box writer (Claude Opus, source + pilot plant) | wb1-wb5 | 300 pilot batches | `runs/wbN/work/controller.py` |
| Black-box writer (Claude Opus, manual + pilot plant) | bb1-bb5 | 300 pilot batches | `runs/bbN/work/controller.py` |
| CMA-ES on the expert law's 6 knobs | seeds 0-4 | 97 evaluations x 12 search episodes = 1,164 episodes | `best.json` params |
| TD3, LRU core | seeds 1-5 | 2M environment steps | final checkpoint (primary), best-det checkpoint (secondary) |
| TD3, LSTM core | seeds 1-5 | 2M environment steps | as above |
| References: hand-tuned expert, oracle expert | 1 each | none | fixed |

- Writer briefs: `experiments/program_control/blackbox/WRITER_BRIEFS.md`; the prompts are
  identical across repeats except for the run name. Each run has its own pilot seed block.
- TD3 configuration is fixed a priori to the current defaults for both cores (tanh-saturation
  penalty 0.1, tanh harvest reward, hidden-state reset 600, BC scaffold, curriculum gates of
  `ADVANCE_TARGETS`), with `TD3_SEED` set and no early stop at a tier. No hyperparameter tuning
  is done for either core; the planned GPU tuning grid is withdrawn so that tuning cannot favour
  one core. The best-det checkpoint is selected on the training det-eval set only.
  The trainer's built-in D0 capability abort (12 consecutive failed D0 capability checks) stays
  on for both cores, as in every earlier run; an aborted run is scored on its last checkpoint
  like any other and counts as not reaching D1. (Clarified 2026-10-01, before any RL result.)
- CMA-ES gets roughly four times the writers' episode budget, which favours the baseline.

## 4. Evaluation and statistics
- `final` split (harness.py): 160 main + 40 high-inoculum episodes from seed 20,000,000,
  disjoint from every seed used in training, search, pilot trials and the earlier `test` split.
  Each frozen controller is run on it once. The 52-episode `test` split, which has already been
  looked at during development, is kept for development only.
- Primary endpoint: mean paired per-episode harvest difference (g) over yield-scored episodes
  (initial culture > 80 cells). Secondary: median and 25th-percentile harvest, crash rate; for
  TD3 also steps to reach D1/D2 and the number of seeds reaching D2.
- Uncertainty: hierarchical bootstrap (runs within each arm, and episodes jointly), 20,000 draws.
  Seed-level exact permutation test on per-run medians for arms with several runs (with 5 v 5 runs
  the smallest possible p is 0.008; with 3 v 3 it is 0.10, which is why every arm has 5 runs).
- Pre-registered family, Holm-corrected: white-box vs black-box; black-box vs CMA-ES;
  white-box vs CMA-ES; LRU vs LSTM; LRU vs CMA-ES; LSTM vs CMA-ES. Anything else is exploratory.
- Script: `experiments/program_control/results/final/compare.py`; inputs are
  `results/final/<arm>__<run>.json`.

## 5. Also reported (not in the family)
- Small open models evolving programs on Kaggle (llama3.2 3B, qwen2.5-coder 7B, qwen3 8B; three
  repeats each): scored on `final` only if a run beat its seed program on the search split.
- The physics-v2 LRU/LSTM pilot (v62-v65, two seeds per core): reported as a pilot, since two
  seeds per arm cannot support a test.
- Compute per arm: simulated episodes, wall-clock hours, hardware.
