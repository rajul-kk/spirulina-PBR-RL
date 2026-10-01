# Lab notebook - run bb1 (jacketed CSTR, Ca control)

All commands run from the root `experiments/pcgym_control/`. Analysis scripts live in `runs/bb1/work/`
and read only `runs/bb1/trials/`.

## Files read
- `PLANT_MANUAL.md` (root) - the operator's manual. Read once at the start.
- `runs/bb1/trials/*.csv`, `results.jsonl`, `budget.json` - my own pilot-batch logs.
- My own files under `runs/bb1/work/`.
- Nothing else: not `plant_trial.py`, no other run folder, no package source.

## What the manual gives
- Control `Ca` (mol/L) to a setpoint that changes a couple of times per batch; actuator = jacket
  temperature 295-302 K; 120 samples per 26-min batch (dt = 0.2167 min = 13 s).
- Measured: `Ca` (noisy), `T` (noisy), `Ca_sp`, `t_min`. Unmeasured feed temperature / feed
  concentration shifts. Runaway if T > 335 K. Cost = mean ((Ca - sp)/0.01)^2 on true Ca.

## Plan (made before the first batch)
1. A couple of open-loop batches to see the log format and natural behaviour.
2. Open-loop excitation batches (random jacket steps) -> identify a first-principles model.
3. Design estimator + controller on the model, tune offline on my own simulator of that model.
4. Short plant trials to confirm the simulator predicts the plant; fix what does not match.
5. A large validation run of the frozen final controller. Keep a reserve of budget.

---

## E1 - open loop, constant jacket (2 batches: b000-b001)
`python plant_trial.py --run bb1 runs/bb1/work/c00_const.py --n 2 --label const`
- Why: see the log format, noise level, natural drift.
- Result: cost 0.59 and 15.9; max T 323.8 / 322.3 K.
- Learned: log columns `step,t_min,Ca,T,Ca_sp,Tc`; logged Ca/T are the noisy measurements. At
  Tc = 298.5 K the reactor sits at T ~ 322 K, Ca ~ 0.885-0.915. Setpoints 0.86-0.89. In b001 Ca jumped
  from 0.887 to 0.915 around sample 28 with the jacket constant and T nearly unchanged -> a feed
  concentration step. Three setpoint segments per batch.

## E2 - open-loop excitation (8 batches: b002-b009)
`python plant_trial.py --run bb1 runs/bb1/work/c01_excite.py --n 8 --label excite`
- Why: identification data. Random-level (295 / 302 / uniform), random-hold (3-25 samples) jacket
  steps, with a guard (full cooling if T > 329 K).
- Result: costs 5.7-15.7 (irrelevant), max T 329.3 K, no runaway.

### Identification (`fit_model.py`, `fit_fixed.py`)
Model structure (standard exothermic CSTR energy/mass balance, time in minutes):

    dCa/dt = a (Caf - Ca) - k(T) Ca
    dT/dt  = a (Tf - T) + beta k(T) Ca + alpha (Tc - T),    k(T) = k0 exp(-(E/R)/T)

Whole-batch simulation fit (RK4), free initial state per batch, feed disturbances Caf / Tf
piecewise-constant in blocks with a total-variation penalty, soft-L1 loss.
(First attempt with a dense finite-difference Jacobian was too slow and was stopped; redone with a
sparse Jacobian.)
- Jacket action timing: the Tc logged in row k acts over k -> k+1 (lag 0 fits to the noise floor:
  rms Ca 0.0020, T 0.20 K; lag 1 is clearly worse: T rms 0.58 K).
- Free fit: a = 1.023 /min, k(323 K) = 0.1246 /min, E/R = 8895 K, beta = 205.3 K L/mol, alpha = 2.082 /min.
- These are within fit uncertainty of the literature values for this textbook reactor
  (a = 1, k0 = 7.2e10 /min, E/R = 8750 K, beta = 209.2, alpha = 2.092; k(323) = 0.1247). With the
  parameters held at the literature values and only disturbances/initial states fitted the cost is
  the same (452.5 vs 452.8), so I adopted the round literature values.
- Noise (residual rms, later corrected upwards, see E5): Ca ~0.0018-0.002 mol/L, T ~0.19-0.2 K, white.
- Feed disturbances: Caf moves in steps within about 0.98-1.02; Tf in steps within about
  348.5-351.5 K; roughly one step of each per batch at no fixed time.
- `summarize.py` on b000-b009: three setpoint segments, values 0.862-0.898, first change at sample
  20-67, second at 49-98. Start-up state Ca 0.85-0.90, T 320-326 K (not at equilibrium).

## Design v1 (`c02_mpc_v1.py`)
- Extended Kalman filter on [Ca, T, Caf, Tf] (Caf, Tf random walks) -> offset-free tracking and
  feed-shift rejection without measuring the feed.
- Nonlinear MPC: minimise sum of squared Ca error over 16 samples, 6 blocked moves, bounds
  295-302 K, projected Gauss-Newton/LM with finite-difference Jacobian, warm start.
- Safety: full cooling if T (measured or estimated) > 331 K.
- Offline simulator `simlab.py` = my model + a scenario generator built from the statistics above.
  Simulated v1: mean 0.23 (first scoring convention), open-loop constant: mean 6.0.

## E3 - first closed-loop trial (10 batches: b010-b019)
`python plant_trial.py --run bb1 runs/bb1/work/c02_mpc_v1.py --n 10 --label mpc1`
- Why: check that the model/simulator predicts the plant before tuning on the simulator.
- Result: mean 0.220, median 0.170, max 0.636, max T 328.8 K, no runaway. Simulator said ~0.15-0.23.
- Learned (`replay.py`, `breakdown.py`):
  - Scoring convention: reported cost matches mean of (Ca[k+1] - sp[k])^2, i.e. the state *after*
    each move against that sample's setpoint (the initial state and the sample at which a setpoint
    change first appears are not charged). Reconstructed cost matches reported to ~0.01 per batch.
  - Cost split: steady 0.018, start-up 0.091, setpoint changes 0.109. Start-up and setpoint
    transients are actuator-limited (the jacket sits on 295 or 302 K for 5-9 samples, bang-bang).
  - The jacket chatters on noise in quiet periods (saturated 19-35 % of the quiet time).

### Simulator studies (no plant batches)
- `decomp.py`: setpoint changes are the largest part; noise ~0.012; disturbance steps ~0.01.
- `tune.py`: horizon 8/16/20/30, more iterations, bigger process noise: all within 0.002. Slight
  gain from halving the disturbance random-walk variance and a small move penalty (w_du = 0.05):
  0.1453 -> 0.1413 on 60 paired seeds. Adopted (v2).

## E4 - v2 trial (30 batches: b020-b049)
`python plant_trial.py --run bb1 runs/bb1/work/c03_mpc_v2.py --n 30 --label mpc2`
- Why: confirm v2 on the plant, collect closed-loop data on disturbances and start-up states.
- Result: mean 0.198 (SE 0.028), median 0.141, max 0.767 (b031: start-up Ca 0.914 vs sp 0.862),
  max T 328.7 K, no runaway. Split: steady 0.024, start-up 0.099, setpoint changes 0.077. Quiet-time
  saturation dropped to 0-5 %.
- `diststats.py` (EKF replay): Caf in 0.978-1.023 from the first minute (mean 1.000), Tf in
  348.4-351.6 K (mean 349.9); 0-2 steps of each per batch. -> prior set to Caf 1.0 +/- 0.012,
  Tf 350 +/- 0.9; simulator scenario generator updated to these ranges.
- Tried innovation-gated jump detection in the EKF (re-open Caf/Tf covariance when the normalised
  innovation is large) with lower baseline random-walk noise: no gain in simulation (0.128-0.131
  for every variant), so it stays off (`jump_thr = 0`).
- `robust.py` (my model): reachable steady Ca at nominal feed is 0.835 (Tc 302) to 0.927 (Tc 295).
  With rich hot feed (Caf 1.02, Tf 351.5) a jacket held at 302 K ignites (no steady state below
  335 K) -> full heating must never be held blindly. With lean cold feed Ca cannot go below ~0.842-
  0.863, so the lowest setpoints are sometimes unreachable.
  Plant/model mismatch test: +10 % k0, +-10 % alpha/beta, +5 % a: little change when the setpoint
  stays reachable; large cost when the perturbation makes setpoints unreachable (physical limit).
- `safety.py` (my model, ignition corner: sp 0.86, Caf 1.02, Tf stepping to 351.5): MPC max T
  329.6 K with or without the guard; a dumb "hold 302" controller with the 331 K guard peaks at
  331.3 K and recovers. Guard kept at 331 K.

## E5 - validation of the frozen controller (100 batches: b050-b149)
`python plant_trial.py --run bb1 runs/bb1/work/controller.py --n 100 --label final`
- Controller = v2 with priors from E4 (Caf 1.0, Tf 350, tighter initial covariance).
- Result: mean 0.188 (SE 0.014), median 0.148, p90 0.41, max 0.656, min 0.015; max T 329.5 K;
  0 runaways, 0 errors. Split: steady 0.031, start-up 0.071, setpoint changes 0.096.
- `noise.py`: noise is Gaussian, no outliers, same level in all batches.
- `chatter.py`: plant innovations in quiet periods are ~12 % larger than my simulator's
  (Ca 0.00245 vs 0.00214, T 0.235 vs 0.218) -> measurement noise is nearer 0.0021 mol/L and 0.21 K
  than the 0.0018 / 0.19 I first took from the fit. Simulator noise raised accordingly.
- `sim_breakdown.py`: simulator steady part 0.013 vs plant ~0.024-0.03: the plant is noisier in
  quiet periods than my simulator; the transients (start-up, setpoint) agree.

### After E5: is the estimator tuned for the plant, not only for my simulator? (no plant batches)
- `innov.py`: on b050-b149 the one-step prediction errors of the fixed model are unbiased in quiet
  periods and at both actuator limits (Ca bias <= 0.0006, T bias <= 0.004 K) -> the dynamic model
  holds in closed loop too. But the innovations are positively autocorrelated (lag-1: Ca 0.25,
  T 0.24; my simulator gives 0.16 / 0.08) -> the filter was too slow for the real feed disturbances.
- `fit_noise.py`: maximum-likelihood fit of the EKF noise model on 35 logged batches, checked on
  the other 50: r_ca 0.0020 mol/L, r_t 0.20 K, q_ca 2.4e-4, q_t 0.0084, q_caf 0.00195, q_tf 0.114
  (per sample). Held-out negative log-likelihood -6.17 -> -6.35, innovations white (-0.03 / 0.05).
  So measurement noise is 0.002 mol/L and 0.2 K, and the feed disturbances are about twice as
  active as the setting I had chosen from the simulator (my halving in v2 was a simulator artefact).
- `tune2.py` (paired seeds, simulator noise now 0.002 / 0.2): ML tuning -0.002 (SE 0.001) vs the E5
  tuning; move penalty 0 vs 0.05 no difference; jump detection -0.004 (SE 0.0016) but it depends on
  my assumed step-shaped disturbances, so not adopted. All variants within 0.005: the design is on
  a plateau, the remaining cost is actuator-limited transients.

## E6 - validation of the final controller (100 batches: b150-b249)
`python plant_trial.py --run bb1 runs/bb1/work/controller.py --n 100 --label final2`
- Controller = E5 controller with the ML noise model (frozen copy `c05_final_v4.py`; the E5 version
  is `c04_final_v3.py`).
- Result: mean 0.176 (SE 0.014), median 0.142, p90 0.35, max 0.743, min 0.016; max T 329.1 K;
  0 runaways, 0 errors. Split: steady 0.026, start-up 0.055, setpoint changes 0.101.
- Against E5 (0.188 +/- 0.014, steady 0.031): better in the expected direction (steady part), but
  the difference in the mean is within one standard error - not a proven improvement. Both
  versions are validated on 100 batches each; `controller.py` is the E6 version.

## Budget
250 of 300 batches used: 2 open loop, 8 identification, 10 + 30 development, 100 + 100 validation.
50 kept in reserve (unused).

## Open points
- About 85 % of the remaining cost is start-up and setpoint-change transients with the jacket on
  a limit (bang-bang); only a wider jacket range or advance notice of setpoint changes would help.
- The ignition corner (rich, hot feed with the jacket held at 302 K) was only explored on my model,
  never on the plant; the highest plant temperature in 240 closed-loop batches was 329.5 K.
- Feed disturbances were inferred, never measured: ranges Caf 0.98-1.02, Tf 348.5-351.5 K are
  estimates from the filter. Larger shifts than seen would make low setpoints unreachable.
- Parameters were taken at literature values because the fit could not tell them apart from the
  free fit (a 1.02 vs 1, E/R 8895 vs 8750, beta 205 vs 209, alpha 2.08 vs 2.09); behaviour far
  outside 317-330 K is untested.
- Move penalty and jump detection could not be separated from noise with this budget.
