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
- Re-scoring: a `final` run is repeated only when the harness itself invalidated episodes. This
  happened once (2026-10-01, TD3 LRU seed 3): two of 200 episodes hit the harness's 300 s
  per-episode wall-clock guard because the laptop was overloaded. The run was moved to
  `results/final/invalid/` and repeated with the guard lifted (`PC_EPISODE_WALL_S`); episodes
  are deterministic given the seed, so only the two cut-off episodes can differ.
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

## 6. Exploratory arms added 2026-10-01 (after the first RL results; not in the family)
Two more recurrent cores for TD3, same configuration and budget, seeds 1-5 each, pinned to tag
`rl-protocol-v2` (which adds the cores and leaves the LSTM and LRU code paths unchanged):
- GRU (`nn.GRU`), the usual alternative to the LSTM.
- RTU, the Recurrent Trace Unit of Elelimy et al. (NeurIPS 2024): a complex-valued diagonal
  recurrence with the nonlinearity after it (`td3/rtu_core.py`). Our "LRU" core is a real-valued
  diagonal, i.e. the RTU with zero phase, so RTU vs LRU isolates the effect of oscillating
  (complex-eigenvalue) memory. The paper trains RTUs with real-time recurrent learning; here
  every core is trained by backpropagation through the 60-step replay window, so only the
  architecture differs.
They were chosen after seeing that the LSTM stalls at D0, so their comparisons with LRU and LSTM
are reported as exploratory, with the same statistics but outside the Holm family.

## 7. Follow-up: LRU vs LSTM with a 4M-step budget and an LSTM reset of 60 (written 2026-10-06)
Written after the section 4 result (LRU vs LSTM: null on the primary endpoint, LRU ahead on
reaching D1/D2) and before any run below. It is a separate family; it does not change the
section 4 result. Two open questions:
1. Does the result change with more training? (2M steps may be too short for either core.)
2. Is the LSTM's weakness caused by the hidden-state reset of 600? The LSTM's cell state grows
   with the step count (`docs/decision_history.md`, saturation diagnosis). A reset of 60 fixed
   that in v45. Section 3 ran both cores at 600, a setting chosen for the LRU.

Arms (tag `rl-protocol-v3`, which only lets a session set `TD3_STEPS` and
`TD3_HIDDEN_RESET_INTERVAL`; the TD3, LSTM, LRU and simulator code is unchanged from
`rl-protocol-v1`):

| Arm | Seeds | How |
|---|---|---|
| LRU, reset 600, 4M | 1-5 | the section 3 runs, resumed from their 2M state with `TD3_STEPS=4000000` |
| LSTM, reset 600, 4M | 1, 3, 4, 5 resumed; 2 carried over | as above. lstm-s2 hit the D0 capability abort before 2M. Under section 3's rules an aborted run is finished, so its 2M result counts as its 4M result |
| LSTM, reset 60 | 1-5 | fresh runs with `TD3_HIDDEN_RESET_INTERVAL=60`. Each trains to 2M (scored there), then resumes to 4M (scored again) |

Everything else is as in section 3: same seeds, curriculum, gates, D0 capability abort, BC
scaffold and noise schedule. The exploration noise anneal is fixed at 600k steps, so extending a
run does not raise its noise again. A resume reseeds from `TD3_SEED`, as every session boundary
already does. Scoring is the section 4 `final` split. Each actor is evaluated with the reset
it was trained with (600 or 60).

Pre-registered family, Holm-corrected over these three. The endpoint is the mean paired
per-episode harvest of the final checkpoint, with the section 4 bootstrap and the seed-level
exact permutation test:
- F1: LRU-600 vs LSTM-600, both at 4M.
- F2: LRU-600 vs LSTM-60, both at 4M.
- F3: LSTM-60 vs LSTM-600, both at 2M (matched budget, so this isolates the reset).

Secondary, unadjusted: seeds reaching D1 and D2 by 4M; steps to D1 and D2; crash rate; the
best-det checkpoint; the change from 2M to 4M within each seed. GRU and RTU are not extended.

Limits stated in advance. This follow-up was designed after seeing the 2M results, so it can
confirm or overturn them only within its own family. Five seeds per arm put the smallest
seed-level p at 0.008.

Outputs:
- Session logs: `results/rl_4m/<arm>-s<seed>/session<N>`.
- Scores: `results/final/followup/td3_<arm>_<steps>__s<seed>.json`.
- Analysis: `results/final/followup/compare_followup.py`.

## 8. Note added 2026-10-06 (test suite, after all section 4 results)
A simulation in `tests/test_stats.py` found that the hierarchical bootstrap is mildly liberal at
5 runs per arm when runs differ from each other. Nominal 95% CIs covered a known shift in about
84% of 150 replicates; with no run-to-run variance the coverage was fine. The seed-level exact
permutation test does not depend on the bootstrap. It is therefore the conservative headline test
for every comparison here (smallest possible p is 0.008 at 5 v 5), and the bootstrap CIs should
be read as somewhat narrow.
