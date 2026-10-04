# Lab notebook - run bb5

## Files read
- PLANT_MANUAL.md (root): CSTR, Ca setpoint tracking, Tc in [295,302] K, 120 samples ~13 s (26 min),
  runaway if T>335 K, cost = mean((Ca-sp)/0.01)^2 on true Ca. Unmeasured feed T / CA0 shifts.

## Budget plan (300 batches)
- ~20 open-loop/step probes to learn statics and dynamics
- ~150 for design iterations
- ~60+ reserved for final validation

## Experiments

### E1 - constant Tc = 298.5 K (b000-b001)
`python plant_trial.py --run bb5 runs/bb5/work/probe_const.py --n 2 --label const2985`
Why: baseline, see log format and ranges. Result: cost 8.15, 6.97; T max 325.7.
Learned: Ca ~0.90-0.91 at Tc 298.5; setpoints ~0.86-0.90, 2 changes per batch at random samples.
Files read: runs/bb5/trials/*.csv, results.jsonl (columns step,t_min,Ca,T,Ca_sp,Tc).

### E2 - open-loop staircases (b002-b005)
`probe_stairs_up.py` (296/298/300/302 K, 30 samples each) and `probe_stairs_dn.py` (302/300/298/295), --n 2 each.
Why: static gain, dynamics, temperature margin. Results: costs 6.9-16; max T 330.6 K (Tc=302).
Learned:
- Static map (approx): Tc 295->Ca 0.93-0.95 (T 317.5), 296->0.93-0.94 (T 318.5-319.5), 298->0.91-0.92 (T 321-322),
  300->0.86-0.89 (T 323.5-325.6), 302->0.81-0.85 (T 326.7-330.6). Gain grows with Tc: ~-0.01/K low end, ~-0.02/K high end.
- Ca settles in ~10-15 samples (2-3 min) after a Tc step; T responds within 2-4 samples.
- Batch-to-batch offsets of +-0.02-0.03 in Ca at fixed Tc -> feed disturbances matter.
- Measurement noise (2nd-difference estimate): sigma_Ca ~0.002 mol/L, sigma_T ~0.2 K.
- Initial states Ca 0.86-0.89, T 320-326.

### Analysis A1 - model structure (no batches)
Scripts: simfit.py, timing.py, noise.py, ekf_offline.py (read only runs/bb5/trials).
Tried the textbook first-order exothermic CSTR structure (q/V=1/min, k0=7.2e10/min, E/R=8750 K,
-dH/(rho Cp)=5e4/239, UA/(V rho Cp)=5e4/23900 1/min, Caf=1, Tf=350 K). One-step-ahead residual RMS
0.0037 mol/L / 0.31 K, close to measurement noise -> structure and parameters fit well.
Tc[k] acts over interval k->k+1 (lag-0 fits T better: 0.314 vs 0.346 K).
Offline EKF with random-walk Caf, Tf states: estimates are piecewise constant and do not follow the Tc
staircase (good sign the model is right): Caf in ~0.97-1.03, Tf in ~348.4-351.6 K, steps at random times.

### Design D1 - EKF + nonlinear MPC (controller.py, snapshot mpc_v1.py)
EKF on (Ca, T, Caf, Tf) with the identified model (RK4, 4 substeps per 13 s sample), R = diag(0.002^2, 0.2^2).
MPC: horizon 15 samples (3.25 min), Tc move-blocked into 5 blocks (1,1,2,3,8 samples), cost
sum((Ca-sp)/0.01)^2 + lam*sum(dTc^2) + soft penalty on predicted T > T_lim; box-constrained Gauss-Newton
(finite-difference sensitivities, 3 iterations), warm-started from the previous plan. Setpoint assumed constant
over the horizon (no preview is available). Local bench localsim.py (my own model, noise, random feed steps).
Local bench (30 scenarios): lam 0.03/0.1/0.3/1.0 -> mean 0.343/0.348/0.368/0.415.

### E3 - mpc_v1 on plant (b006-b011), lam=0.3, q_Caf=2e-4, q_Tf=0.05
`python plant_trial.py --run bb5 runs/bb5/work/mpc_v1.py --n 6 --label mpc_v1`
Result: costs 0.213 0.218 0.334 0.162 0.146 0.334 (mean 0.235); max T 327.8. No runaways.
breakdown.py: ~90% of the cost falls in the 15 samples after start-up and after each setpoint change;
steady-state cost 0.01-0.1. Tc saturates during big setpoint moves -> transient is actuator/physics limited.

### Analysis A2 - is the MPC/estimator the bottleneck? (no batches)
Oracle MPC (true state and feed known, no noise) on local bench: 0.321 (lam 0.1) vs 0.348 with EKF;
richer MPC (9 blocks, N=20, 5 GN iterations) under oracle: 0.313 = same as default -> MPC parameterisation
is not limiting; remaining cost is the unavoidable response to unannounced setpoint steps.
innov.py: EKF prediction error on the 12 plant logs vs disturbance random-walk Q. Smaller feed-disturbance Q
predicts better: 3-step Ca RMS 0.00435 (q_Caf 2e-4, q_Tf 0.05) -> 0.00329 (2e-5, 0.02) -> 0.00316 (5e-6, 0.005).
Chose (2e-5, 0.02) as compromise with tracking speed of real feed steps.
paramscan.py: scaling U, J, k0, q/V by 0.9-1.1 does not improve prediction error -> nominal parameters are at
the optimum (surface is flat within +-3%).

### E4 - mpc_v2 on plant (b012-b031), lam=0.1, q=(1e-6,1e-3,2e-5,0.02)
`python plant_trial.py --run bb5 runs/bb5/work/mpc_v2.py --n 20 --label mpc_v2`
Result: mean 0.167, median ~0.13, range 0.036-0.372; max T 328.6; no runaways. Mean |dTc| per sample
0.47 K (v1: 0.57 K).
regret.py (replays each plant batch's setpoints and EKF-estimated feed disturbances in my model with an
oracle MPC): actual is at or below the replay baseline for both versions (v1 regret -0.055 +- 0.024,
v2 -0.030 +- 0.012; baseline is biased high by noisy disturbance estimates). Both are near the causal limit;
v1 vs v2 not distinguishable on the plant. Kept v2 (smoother actuator, better 3-step prediction).

### Analysis A3 - stress / safety (no batches)
stress.py: feed Caf 0.92-1.08, Tf 345-355 K, setpoints 0.84-0.92, start 0.80-0.95/315-330 K, 5-15% model
mismatch. Original v2 produced NaN (model overflow in prediction) and 2 runaways that full cooling (Tc=295)
avoids, plus 1 scenario where even constant Tc=295 runs away (unavoidable).
Fixes: clamp T/Ca inside model integration, NaN fallback to last move, safety override (T > 331 K: cut Tc by
>=1 K per sample; T > 332.5 K: full cooling 295 K), MPC soft limit T_lim 331 K.
After: only the unavoidable scenario runs away. Nominal local bench unchanged (0.350).
Note: the oracle hook was removed from controller.py afterwards; regret.py now uses its own Oracle subclass.

### E5 - final controller smoke test (b032-b035)
`python plant_trial.py --run bb5 runs/bb5/work/controller.py --n 4 --label final_smoke`
Costs 0.185 0.036 0.079 0.154; max T 328.3; ~1.5 s per batch.

### E6 - variant with slower disturbance estimator, q_Caf 5e-6, q_Tf 0.005 (b036-b065)
`python plant_trial.py --run bb5 runs/bb5/work/mpc_slowq.py --n 30 --label slowq`
Why: it gave the best offline prediction error. Result: mean 0.235 (one batch at 0.988 after a +0.031 setpoint
step), but those scenarios were harder: oracle-replay baseline 0.286. actual/baseline ratio: slowq 0.82,
v2 0.85, v1 0.81 -> no resolvable difference. Kept q=(2e-5, 0.02) (estimator reacts faster to real feed steps).

### E7 - VALIDATION of final controller.py (b066-b125)
`python plant_trial.py --run bb5 runs/bb5/work/controller.py --n 60 --label final_val`
Result (60 batches): mean 0.153, median 0.121, std 0.138, p90 0.296, min 0.021, max 0.781.
Runaways 0, errors 0, highest reactor T 329.0 K (limit 335).
regret.py: replay baseline 0.194 (actual/baseline 0.79). breakdown.py: about 90% of the cost is in the
15 samples after start-up and after setpoint changes; steady-state part about 0.02.

Budget used: 126 of 300 (6 open-loop probes, 56 design comparisons [6 v1, 20 v2, 30 slowq], 64 final controller incl. 60 validation).
Remaining 174 batches not used: further tuning differences were below what the batch-to-batch scatter
can resolve.
