# Protocol: controller-writing comparison on a public benchmark (PC-Gym CSTR)

Written 2026-10-01, before any arm was run on the `final` split. Purpose: check whether the
photobioreactor result (LLM-written controllers beat classical tuning and RL; source access
helps) holds on a simulator we did not build. Same design as `comparison_protocol.md`.

## 1. Task
PC-Gym's exothermic CSTR model (`cstr_ode`, pcgym 0.1.6, dynamics unmodified), wrapped in
`experiments/pcgym_control/tasks.py`:
- 120 samples over 26 minutes (PC-Gym's standard horizon). Manipulated: jacket temperature
  295-302 K. Measured: Ca (noise sd 0.002 mol/L), T (sd 0.2 K). Setpoint on Ca, two step
  changes per batch within 0.86-0.90 mol/L.
- Unmeasured step disturbances: feed temperature 348.5-351.5 K and feed concentration
  0.98-1.02 mol/L, one or two changes each. Random start-up state.
- Cost per batch = mean over samples of ((Ca - setpoint) / 0.01)^2 on the true state; lower is
  better. A batch whose temperature exceeds 335 K is counted as a runaway (reported separately).

## 2. Arms and budgets
| Arm | Runs | Budget per run | Evaluated |
|---|---|---|---|
| White-box writer (Claude Opus: manual, task and model source, privileged pilot logs) | wb1-wb5 | 300 pilot batches | `runs/wbN/work/controller.py` |
| Black-box writer (Claude Opus: manual and pilot logs only) | bb1-bb5 | 300 pilot batches | `runs/bbN/work/controller.py` |
| CMA-ES over PID gains (`controllers/pid.py`: kp, ki, kd, bias) | seeds 0-4 | 97 evaluations x 12 search batches = 1,164 batches | `best.json` gains |
| SAC (Stable-Baselines3 defaults, 4-sample observation stack) | seeds 1-5 | 100,000 steps = 833 batches | final policy |
| Reference: untuned PID (default gains) | 1 | none | fixed |

Writer briefs: `experiments/pcgym_control/WRITER_BRIEFS.md`; prompts identical across repeats
except for the run name. CMA-ES and SAC each get about 3-4 times the writers' batch budget.
No hyperparameter tuning of SAC or of CMA-ES.

## 3. Evaluation and statistics
- `final` split: 200 batches, seeds 20,000,000+, disjoint from search (3,000,000+), pilot
  (5,000,000+ in a block per run) and SAC training (7,000,000+) seeds. Each frozen controller is
  scored once.
- Primary endpoint: mean paired per-batch cost difference. Secondary: median cost, runaway rate.
- Hierarchical bootstrap over runs and batches (20,000 draws); exact seed-level permutation test
  on per-run mean cost; Holm correction over the pre-registered family: white-box vs black-box;
  black-box vs CMA-ES-PID; white-box vs CMA-ES-PID; black-box vs SAC; white-box vs SAC.
- Script: `experiments/pcgym_control/results/final/compare.py`.

## 4. Limits stated in advance
One PC-Gym model and one scenario family; a single-loop problem that PID suits well, so the
room above the baseline may be smaller than on the photobioreactor. The SAC baseline is
memoryless apart from a 4-sample stack. No NMPC oracle (do-mpc is not installed).

## 5. Caveat recorded 2026-10-01, after two writers had reported and before most were scored
The black-box writers are not blind on this benchmark. bb1 and bb2 both recognised the plant as
the standard textbook exothermic CSTR; bb1 adopted the published constants (which are PC-Gym's)
after finding them indistinguishable from its own fit. A public textbook model is in the
language model's training data, so "manual only" still carries the model. The white-box vs
black-box comparison is therefore not a clean test of source access here (it is on the
unpublished photobioreactor); the writers vs CMA-ES-PID and writers vs SAC comparisons are
unaffected.

## 6. Access audit, 2026-10-05 (after all arms were scored)
Every tool call in the ten writer transcripts was checked. The black-box writers read only
`PLANT_MANUAL.md` and their own run directory: no repository source, no `pcgym` package, no web
search. The prior knowledge came from the language model itself. All five assumed the textbook
model structure. bb1 and bb5 ship the exact published constants (k0 7.2e10, E/R 8750, which are
PC-Gym's), recalled from memory and confirmed on pilot data. bb2, bb3 and bb4 ship constants
fitted to their own pilot batches (E/R 8604-9012). Final mean cost: recalled 0.187 and 0.204,
fitted 0.185, 0.198 and 0.186, so recalled constants gave no visible edge. The control strategy
the writers chose (an EKF that also estimates the feed disturbances, plus NMPC) is generic
control-engineering knowledge, not specific to this benchmark. Writers vs CMA-ES-PID and SAC
therefore compare an agent that knows chemical engineering with two methods that start from
nothing, which is part of what is being measured. The photobioreactor study is the test without
that prior.

## 7. Correction, 2026-10-06
The seed-range comments in `rl_sac.py` and `plant_trial.py` describe SAC training seeds (7M+) as
disjoint from the writers' pilot blocks. They are not: the pilot blocks lie at
5M + 100k x (1..97), up to about 14.7M, and overlap the SAC training range. Both are development
data, so the scores are unaffected. The `final` split (20M+) is disjoint from every other range,
and `tests/test_splits.py` asserts this.
