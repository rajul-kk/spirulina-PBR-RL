# Lab notebook: run bb3 (jacketed CSTR, black-box controller design)

Root: `PPO_IBM/experiments/pcgym_control/`. All commands are run from the root.

## Files read
- `PLANT_MANUAL.md` (root). Key facts: control Ca to a setpoint that changes a couple of times per
  batch; 120 samples of 13 s (26 min); actuator = jacket temperature 295 to 302 K (clipped);
  unmeasured feed temperature and feed concentration shifts; runaway if T > 335 K;
  cost = mean ((Ca_true - sp)/0.01)^2.
- Only my own files under `runs/bb3/` after that (trial CSVs, `results.jsonl`, `budget.json`, my scripts).

## Plan / budget (300 batches)
- ~10 batches: open loop (constant jacket, step tests) to learn gains, time constants, noise, disturbances.
- ~25 batches: first closed-loop design, find where the cost comes from.
- Tuning done mainly on my own surrogate simulator built from the identified model (plant-to-plant
  scenario variance is too large to resolve small tuning differences on the plant itself).
- Keep >= 50 batches for final validation.

---

## E1: constant jacket (b000-b002, 3 batches)
`python plant_trial.py --run bb3 runs/bb3/work/c01_const.py --n 3 --label const` (Tc = 298.5 K)
- Why: see the log format, the operating point, noise and disturbances with no controller.
- Result: cost 10.3, 15.8, 9.8; max T 322-323.5 K. At Tc 298.5: Ca ~0.89-0.92, T ~321-323 K.
- Setpoints 0.86-0.90 mol/L, two changes per batch at varying samples (24-98).
- Logged Ca is the *measured* (noisy) value: cost recomputed from the log differs slightly from the reported cost.
- Noise (from 2nd differences): Ca sd ~0.002 mol/L, T sd ~0.2 K.
- Feed shifts visible: e.g. b000 Ca jumps +0.03 at ~sample 90 with Tc constant.
- Each CSV row k holds obs k and the action returned at k (Tc[k] acts on the interval k -> k+1).

## E2: open-loop step tests (b003-b007, 5 batches)
`python plant_trial.py --run bb3 runs/bb3/work/c02_steps.py --n 5 --label steps`
- Why: identify dynamics; jacket steps over the full range, 5-14 sample holds.
- Result: Tc 295 -> T ~318 K, Ca ~0.92; Tc 302 -> T ~329.5 K, Ca ~0.82-0.83. Gain ~ -0.014 mol/L per K
  of jacket. T responds within ~3 samples, Ca within ~8-10 samples. No dead time beyond one sample.
  Max T 330 K at Tc = 302 for 12 samples, so the full actuator range is safe in this region.
- Grey-box fit (`an02_fit.py`, H-step-ahead least squares on the standard CSTR mass+energy balance
  dCa/dt = a(Caf-Ca) - k(T)Ca, dT/dt = a(Tf-T) + b k(T) Ca + c(Tc-T), k = kr exp(-ER(1/T - 1/323))):
  a = 0.834 /min, kr = 0.105 /min at 323 K, E/R = 9183 K, b = 216 K L/mol, c = 2.03 /min,
  Caf = 0.995, Tf = 357.6 K (nominal values in this parametrisation). One-step rms ~1.5 noise sd, i.e.
  essentially measurement-noise limited; no evidence of unmodelled dynamics.
- EKF replay with Caf, Tf as random-walk states (`an03_ekf.py`): feed shifts are steps of roughly
  Caf +-0.015-0.04 and Tf +-1.5-2.5 K, 0-2 of each per batch, at random times.

## Design 1 (c03_mpc.py): EKF + nonlinear MPC
- Estimator: EKF on [Ca, T, Caf, Tf] (feed conditions as random-walk states give offset-free control),
  optional innovation (NIS) test that reopens disturbance covariance on a detected jump.
- MPC: horizon 20 samples, input = v1 for 2 samples, v2 for 4 samples, then the steady-state jacket
  temperature for the setpoint at the estimated feed conditions; coarse grid over [295, 302]^2 then 3
  local refinements; soft penalty on T > 331 K; small move penalty.
- Surrogate (`sim.py`): the identified model + noise + random setpoints/feed shifts like those observed.
  Surrogate mean ~0.39; move-blocking variants, horizons and move penalties all within +-0.02 (`tune.py`).

## E3: first closed loop (b008-b012, 5 batches)
`python plant_trial.py --run bb3 runs/bb3/work/c03_mpc.py --n 5 --label mpc1`
- Result: cost 0.084, 0.188, 0.493, 0.169, 0.240; max T <= 328 K. (Open loop was ~10.)
- Cost breakdown (`an04_decomp.py`): dominated by setpoint changes (unavoidable transition time).

## E4: baseline closed loop, more batches (b013-b032, 20 batches)
`python plant_trial.py --run bb3 runs/bb3/work/c03_mpc.py --n 20 --label mpc1b`
- Why: baseline statistics, look for failure modes, data for refit.
- Result (25 batches mpc1 + mpc1b): mean 0.194, median ~0.13, range 0.022-0.493, max T 328.5 K, no runaways.
  Breakdown (measured error, noise floor removed): start-up (first 10 samples) 0.093, setpoint-change
  windows 0.173, rest (disturbances + noise) 0.022. High-cost batches are those with large setpoint
  steps (~0.03-0.04) or a large initial offset (0.03-0.04).
- PEM refit through the EKF on 13 batches (`an05_pem.py`): a 0.828, kr 0.115, ER 8852, b 201, c 2.11;
  innovation rms barely changes (1.275 -> 1.267), so the model was already adequate.
- Multi-step (10 sample) open-loop prediction check at setpoint changes (`an07_multistep.py`): Ca bias
  <= 0.004 mol/L, T bias <= 0.4 K; within noise. Model is not the limiting factor.
- Surrogate floor (`sim_oracle.py`, same MPC given the true state and feed): 0.354 vs 0.388 with the
  estimator, so estimation costs only ~0.03; the rest is physics (actuator range limits how fast Ca can move).
- Estimator tuning on the surrogate: lower disturbance random walk + jump detection reduces jacket
  chatter (sd of jacket moves 1.19 -> 0.78 K) at +0.008 cost.

---
(Run interrupted by an API usage limit after b032 and resumed; files and logs intact, budget.json = 33.)

## Analysis after resume (no plant batches)
- PEM refit on all 33 batches (`an05_pem.py c03_mpc '*'`): a 0.737, kr 0.117, ER 8776, b 190.6, c 2.126
  (rel sd 3%, 6%, 5%, 2.6%, 0.9%); innovation rms 1.269 -> 1.257, lag-1 autocorrelation -0.09 (white).
  `a` and the nominal feed temperature are strongly correlated: under the refit the EKF puts the feed
  at Caf ~1.036 (sd 0.014), Tf ~365.4 K (sd 1.2) (`an08_nominal.py`). Feed-shift spread over all
  batches: Caf sd ~0.012-0.014, Tf sd ~1.0-1.2 K.
- Model-mismatch cross check on the surrogate (`sim_cross.py`): plant from either parameter set, controller
  using either: all four combinations within 0.01 of each other. Design is insensitive to which fit is used.
- Stress test (`sim_stress.py`, plant parameters perturbed): kr +25%, b +15%, a +15%, E/R +5% all fine
  (max T <= 329.7 K). kr -25% is infeasible (setpoints need a jacket hotter than 302 K) and is not a
  controller fault. c -15% exposed a real weakness of c03: numeric overflow in the MPC predictions gave
  NaN costs, one surrogate runaway (336.4 K) and a cost of 151. Fixed in controller.py: predictions
  clipped to a physical box, non-finite costs rejected, EKF reset if non-finite, fallback to the
  steady-state jacket temperature, and a hard trip to full cooling if measured T > 332 K. After the fix,
  c -15% gives max T 328.9 K, worst cost 2.8 (from saturation, i.e. infeasible setpoint).
- Estimator choice (`tune.py controller 24`, seed 5): q_caf 0.004/q_tf 0.4 (no jump test): mean 0.310,
  jacket-move sd 1.15 K; q_caf 0.002/q_tf 0.2 + jump test (NIS > 12): mean 0.314, jacket-move sd 0.89 K.
  Chose the calmer setting: 1% cost for 23% less jacket chatter.

## Final controller: runs/bb3/work/controller.py
EKF [Ca, T, Caf, Tf] + NMPC (refit model; horizon 20, blocks 2+4 samples then steady state, 15x15
grid + 3 refinements, soft T limit 331 K, trip at 332 K). Nominal feed Caf 1.032, Tf 365.2 K.

## E5: validation (b033-b052, 20 batches)
`python plant_trial.py --run bb3 runs/bb3/work/controller.py --n 20 --label val`
- Mean 0.233; worst b040 0.940: start-up 0.037 above setpoint plus two ~0.03 setpoint steps. Replay
  (`an06_replay.py`) shows correct saturation and no overshoot, so this cost is unavoidable physics.

## E6: validation (b053-b082, 30 batches)
`python plant_trial.py --run bb3 runs/bb3/work/controller.py --n 30 --label val2`
- All 50 validation batches: mean 0.203 (sem 0.024), median 0.145, p90 0.387, max 0.940; max T 328.4 K;
  0 runaways, 0 errors. Breakdown: start-up 0.121, setpoint-change windows 0.129, rest 0.026.
- Baseline c03 (25 batches): mean 0.194, median 0.169. The two are statistically indistinguishable on
  the plant (the difference between scenarios is much larger than the difference between controllers).

## Budget
83 of 300 batches used: 3 constant jacket, 5 step tests, 25 for the first closed loop and baseline, 50 validation.
Stopped there. The surrogate floor (perfect state knowledge) is only ~0.03 below the estimator-based
controller, and the remaining cost comes from start-up offsets and setpoint steps that the 295-302 K
jacket range cannot remove faster. More batches would not tell controllers apart that are this close.
