# Lab notebook: run wb4 (jacketed CSTR, design model available)

## Files read
- `PLANT_MANUAL.md` (root): task, measurements, actuator 295-302 K, runaway > 335 K, cost definition, budget 300.
- `tasks.py` (root): scenario definition and timing of obs/step/cost.
- `pcgym/model_classes.py` (class `cstr_ode`), `pcgym/integrator.py`, `pcgym/pcgym.py` (package in `.venv`).
- My own files under `runs/wb4/` (trial CSVs, results.jsonl, budget.json).
- Nothing else. (`plant_trial.py --help` was run to see its options; its source was not read.)

## What the source says
Plant (time in minutes, sample time 26/120 = 0.2167 min, integrated by CVODES):

    dCa/dt = (Caf - Ca) - k(T) Ca,                 k(T) = 7.2e10 exp(-8750/T)
    dT/dt  = (Ti - T) + 209.2 k(T) Ca + 2.092 (Tc - T)

- Nominal Ti = 350, Caf = 1. At Ca = 0.88: T = 324.2 K, Tc = 299.9 K. Linearised eigenvalues there
  are -1.07 +/- 0.53i per min (stable, lightly oscillatory open loop).
- Steady-state sensitivity dCa/dT = -0.0088 mol/L per K, so the 0.04 setpoint range is about 4.5 K of
  reactor temperature; the 7 K jacket range is tight.
- Scenario (tasks.py): setpoint has 2 steps, levels U(0.86, 0.90). Ti has 1 or 2 steps, levels
  U(348.5, 351.5); Caf 1 or 2 steps, levels U(0.98, 1.02). Step times are samples 20..99 with every
  segment >= 20 samples. Start state Ca U(0.85, 0.91), T U(320, 326) (not a steady state).
  Sensor noise sd: Ca 0.002, T 0.2 K.
- Timing: at sample k the controller sees noisy x_k and sp[k]; the plant integrates with Tc_k,
  Ti[k], Caf[k]; the cost term is ((Ca_{k+1} - sp[k]) / 0.01)^2. A feed step at sample k is not
  visible in the measurement taken at k; it shows up from k+1.
- The process model is deterministic: all uncertainty is sensor noise + the two feed disturbances.

## Approach
Because the model is exact, I wrote my own copy of it (`work/sim.py`, RK4, scenario generator
re-implemented from the description in tasks.py; it does not import pcgym or tasks) and do the design
and tuning there, at no cost in pilot batches. Pilot batches are used to (1) check that my model
matches the plant, (2) validate the final controller.

Controller (`work/controller.py`):
- Estimator: bank of extended Kalman filters on [Ca, T, Ti, Caf], one per hypothesis about when a
  feed variable last stepped (hazard rate from the scenario rules, new level prior = uniform range),
  keep the M best, use the weighted mean.
- Control: nonlinear MPC over H samples minimising predicted squared Ca error (Gauss-Newton with a
  box-constrained QP per iteration), tiny move penalty, soft ceiling on predicted T (331 K).

## Experiments

### Offline 1 (own simulator, no pilot batches): first controller, 200 scenarios (seeds 1000-1199)
- `python sim.py controller.py --n 200 --oracle` (controller told the true state and disturbances):
  mean 0.136, median 0.093, p90 0.298, max 0.64, no runaways, max T 329.0.
- `python sim.py controller.py --n 200` (with estimator): mean 0.153, median 0.117, p90 0.313,
  max 0.68, no runaways.
- Learned: the cost is dominated by unavoidable transients (start-up state and setpoint steps);
  the estimator costs only about 0.017 on top of perfect knowledge.

### Pilot 1: `python plant_trial.py --run wb4 --privileged runs/wb4/work/ctrl_v1.py --n 5 --label v1check`
- Why: check my model against the real plant before investing in offline tuning; see the CSV layout.
- Result: costs 0.290, 0.035, 0.060, 0.074, 0.186 (mean 0.129); max T 328.4; no runaways. 5/300 used.
- `python work/check_model.py`: replaying the logged true states through my model, the one-step
  prediction error is at most 4e-6 mol/L and 5e-4 K (CVODES tolerance; my RK4 with 4 or 40 substeps
  agree with each other to far better than that). Noise sd measured 0.0018-0.0020 and 0.17-0.22 K.
- CSV layout: row k holds the measurement at sample k, the applied Tc, and the true state AFTER the
  step (plus Ti[k], Caf[k]).
- Learned: my model is the plant. Offline results are trustworthy.

### Offline 2: MPC settings with the oracle estimator (200 scenarios, seeds 1000-1199)
`python tune.py controller.py 200 1000 1 '[...]'`, paired differences vs the default (H 14, rho 1e-3):
- H = 8, 20, 30: identical to 4 decimals (0.1361). The plant is fast (time constants < 1 min = 4-5
  samples); the transients are actuator-limited, so a longer horizon changes nothing.
- rho = 0: +0.0012 +/- 0.0004 (worse, noisier moves); rho = 1e-2: +0.0013 +/- 0.0001. Keep 1e-3.
- n_iter 6, RK4 nsub 8: no change. T_max 331 vs 340: no change; T never exceeded 329.0 K, the soft
  ceiling is never active.
- (The run was interrupted by an API usage limit after this sweep; resumed afterwards with no
  pilot batches lost.)

### Offline 3: where the cost comes from (`python decomp.py controller.py 100 1000 <oracle>`)
Per-batch cost split by phase (12 samples after start / after a setpoint step / after a feed step):

| phase | oracle | estimator |
|---|---|---|
| start-up | 0.067 | 0.070 |
| after setpoint step | 0.068 | 0.070 |
| after feed step | 0.001 | 0.011 |
| rest | 0.000 | 0.002 |
| total | 0.137 | 0.152 |

Learned: ~90% of the cost is the start-up and setpoint-step transients, which even perfect state
knowledge cannot remove (jacket saturates; Ca responds with its own ~1 min time constant). The
estimator's whole penalty is ~0.015, mostly the few samples after an unmeasured feed step.

### Offline 4: estimator settings (200 scenarios, measurements noisy)
- M = 3: +0.0006 +/- 0.0003; M = 16: -0.0006 +/- 0.0003 vs M = 8. Chose M = 12.
- Process-noise std x3 or /3: no change. rho = 3e-3: +0.0008 (worse).
- Added a try/except fallback in act() (steady-state jacket temperature for the setpoint) in case of
  a numerical failure; never triggered in simulation.
- Final settings on 400 fresh scenarios (seeds 5000-5399): mean 0.171, median 0.133, p90 0.350,
  max 0.86, no runaways, max T 329.2 K.

### Pilot 2 (validation): `python plant_trial.py --run wb4 --privileged runs/wb4/work/controller.py --n 150 --label final`
- Why: validate the final controller on the real plant with enough batches for a tight mean.
- Result (`python work/summarize_trials.py final`): n 150, mean 0.161 +/- 0.011 (s.e.), median 0.117,
  p90 0.365, max 0.712, no runaways, max T 329.1 K, no controller errors. 155/300 batches used.
- Consistent with the simulator (0.153 on seeds 1000-1199, 0.171 on 5000-5399).
- `check_model.py` on these 150 batches: one-step model error still <= 5e-6 mol/L and 5e-4 K.
- Decision: stop here. The simulator says ~85-90% of the remaining cost is unavoidable transient
  cost (oracle floor ~0.14), so spending the remaining 145 batches would not buy a measurable gain.

## Final controller
`work/controller.py` (identical to the one validated): multi-hypothesis EKF (M = 12) + nonlinear MPC
(H 14, rho 1e-3, soft T ceiling 331 K, RK4 4 substeps), with a steady-state fallback on exceptions.
