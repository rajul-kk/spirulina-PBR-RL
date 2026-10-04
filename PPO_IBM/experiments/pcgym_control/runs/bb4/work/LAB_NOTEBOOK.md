# Lab notebook, run bb4: jacketed CSTR controller

Rules I work under: plant is a black box; I read only PLANT_MANUAL.md and files under runs/bb4/.
Budget 300 pilot batches.

## Files read
- PLANT_MANUAL.md (root): task, interface, actuator range 295-302 K, runaway limit 335 K, cost definition.
- runs/bb4/trials/*.csv, results.jsonl, budget.json (my own trial logs).
- My own scripts under runs/bb4/work/.
Nothing else in the repository or the Python environment was opened.

## Facts from the manual
- 120 samples per 26 min batch -> dt = 0.21667 min (13 s).
- obs: t_min, Ca (noisy), T (noisy), Ca_sp. Action: jacket temperature Tc, 295..302 K.
- Unmeasured disturbances: feed temperature and feed concentration shifts.
- Cost = mean(((Ca_true - sp)/0.01)^2). Runaway if T > 335 K.

## Plan
1. Open-loop probes (constant Tc, then multi-level steps) to see log format, noise, gain, time constants.
2. Grey-box model: textbook 2-state exothermic CSTR (mass + energy balance, Arrhenius rate), parameters
   fitted from pilot data only.
3. Controller: state/disturbance estimator + model-based control; tune on a simulation of the fitted
   model, confirm on plant in small trials, then keep >= 60 batches for final validation.

## Experiments

### E1 (batches 0-1) constant jacket
`python plant_trial.py --run bb4 runs/bb4/work/c_const.py --n 2 --label const`
Why: see log format, noise level, open-loop level at mid-range jacket (298.5 K).
Result: cost 6.35, 6.31; max T 322.8 K.
Learned: CSV columns step,t_min,Ca,T,Ca_sp,Tc. At Tc=298.5: Ca ~0.89-0.91, T ~321.5-323.
Setpoint changes twice per batch (steps 51/93 and 26/69), values 0.86-0.90. Start-up state is
off steady state (Ca 0.85-0.88, T 320-320.6) and settles in ~10 samples. Slow drifts in Ca and T at
constant Tc (e.g. T 322.8 -> 321.5 around step 75 in b000) = the unmeasured feed shifts.

### E2 (batches 2-5) open-loop multi-level steps
`python plant_trial.py --run bb4 runs/bb4/work/c_step.py --n 4 --label step`
Why: excite the plant over the whole actuator range for model identification.
Result: costs 5.3, 3.8, 2.8, 12.3; max T 331.0 K (at Tc = 302 held 14 samples) - no runaway.
Learned: T responds to Tc within ~5 samples; Ca follows T with a further lag. Tc=302 for 3 min
drives T to ~331 and Ca to ~0.83; Tc=295 gives T ~318.5 and Ca ~0.925. So the full Ca range reachable
is roughly 0.83-0.93 and the setpoint band 0.86-0.90 sits inside it. Negative gain Tc -> Ca.

### A1 grey-box identification (no batches) - fit1.py, fit2.py
Model structure (textbook exothermic CSTR, first-order irreversible reaction):
dCa/dt = a (Caf - Ca) - k(T) Ca,  dT/dt = a (Tf - T) + B k(T) Ca + c (Tc - T),
k(T) = kref exp(-E (1/T - 1/322)). Unknown feed Caf, Tf.
- fit1.py (one-step-ahead prediction from measured states, constant feed per fit): biased by
  measurement noise (a = 0.62) - discarded.
- fit2.py (output-error simulation fit; per-batch initial state and piecewise-constant Caf, Tf with
  one changepoint each, changepoints by coordinate search), batches 0-5:
  a = 0.970 +- 0.011 1/min, kref(322 K) = 0.116 1/min (ln = -2.153 +- 0.017), E = 8604 +- 107 K,
  B = 203.3 +- 1.7 K L/mol, c = 2.081 +- 0.012 1/min.
  Residual rms per batch: Ca 0.0017-0.0022, T 0.18-0.32 K, i.e. at the noise level -> model structure
  adequate. Measurement noise sd ~0.002 mol/L (Ca) and ~0.2 K (T).
  Feed: Caf 0.988-1.025 mol/L, Tf 349.3-352.3 K, shifting within batches.
- model.py steady-state map (nominal feed Caf 1.0, Tf 350.5): Tc 295 -> Ca 0.924/T 317.6;
  298 -> 0.899/321.2; 300 -> 0.875/324.1; 302 -> 0.838/327.9 (hot, rich feed at 302: T 331.9).
  Steady-state gain dCa/dTc ~ -0.006 at low Tc, -0.02 per K near 302 (gain rises with T).
  No runaway steady state inside the actuator range for the feed range seen, but T margin to 335 K
  shrinks to ~3 K at Tc=302 with a hot/rich feed, so a T ceiling (331 K) is put in the controller.

### D1 controller design v1 (controller.py, copied as c_mpc1.py)
- EKF on 4 states (Ca, T, Caf, Tf); disturbances as random walks; RK4 model (2 substeps).
- NMPC: 8 free moves + hold at the model steady-state jacket temperature for the setpoint and the
  estimated feed, 14-step prediction, cost = sum((Ca-sp)/0.01)^2 + 0.002 sum du^2 + soft T > 331 K
  penalty; projected Gauss-Newton with active set, warm start, 3 iterations.
  Rationale: cost penalises only Ca error, actuator is bounded, plant is 2nd order (Tc -> T -> Ca)
  and nonlinear (gain changes x3 over the range) -> constrained model-based control is the
  natural fit; feed disturbance states give offset-free tracking.
- Simulation on fitted model (simtest.py): mean 0.33 over 10 scenarios, ~1 s per batch of compute.

### E3 (batches 6-11) first closed-loop trial of v1
`python plant_trial.py --run bb4 runs/bb4/work/c_mpc1.py --n 6 --label mpc1`
Why: check the design on the plant before investing in tuning.
Result: costs 0.151 0.112 0.145 0.024 0.251 0.317 (mean 0.167, median 0.148); max T 327.8 K; no runaway.
breakdown.py: ~80 % of the cost is the 10 samples after each setpoint change; start-up 0.03;
rest 0.03. b007: a feed concentration drop (~-0.034) at step 73 cost ~0.1. b011: Caf rose by
~+0.033 at step 90, in the middle of a setpoint transient. Steady-state Tc jitter ~+-1.5 K from noise,
but the true-Ca cost of that is small (~0.01-0.02).

### A2 disturbance statistics (dist.py, batches 0-11, no new batches)
Implied feed values from one-step residuals, penalised piecewise-constant segmentation:
- Caf: initial level mean 1.004, sd 0.010, range 0.987-1.023; 0 (6), 1 (5) or 2 (1) detected
  shifts per batch; jumps +-0.013..0.036; times 28-98.
- Tf: initial 350.7 +- 1.0 K, range 349.3-352.4; 0/1/2 shifts in 5/6/1 batches; jumps +-1.5..2.4 K;
  times 30-92.
Looks like levels redrawn within ~[0.985, 1.027] and ~[349.2, 352.5]. simtest.py scenario generator
updated to that (plus observed setpoint timing: first change 20-71, second 69-99; values 0.86-0.90).

### A3 headroom (simulation only: oracle.py, optstep.py)
- Sim, 30 scenarios: v1 mean 0.294; MPC fed true state and feed (oracle) 0.264, of which 0.089 is
  the first sample after each setpoint change (unavoidable: the setpoint is not known in advance).
  Estimation + noise cost only ~0.03.
- Setpoint steps from steady state: the receding-horizon MPC's sum of squared errors is within
  0.1 % of an offline 20-move L-BFGS-B optimum for all 6 test steps (e.g. 0.87->0.90: 25.50 vs 25.47;
  0.90->0.86 hot feed 61.80 vs 61.79). Response is bang-bang for 4-8 samples then lands.
  => On the model, the transient is already optimal; remaining gains can only come from the
  estimator (disturbance jumps) and from model-plant mismatch.
- EKF process-noise tuning (q_caf 0.001..0.004, q_tf 0.1..0.4) changes sim cost by < 0.01.

(Run was interrupted by an API usage limit after A3 and resumed; files and trial logs were intact,
budget.json showed 12 used.)

### A4 simulation checks before more plant time (no batches)
- Innovation-triggered disturbance covariance reset (jump_thr 2.5-3.5, window 2-3): sim mean
  0.269-0.270 vs 0.270 without -> no gain; left in code but disabled (jump_thr=0).
- Move penalty lam 0.0002/0.002/0.02 and EKF noise settings (q x0.5..x2.5, r x1.5): all 0.269-0.275.
- Model-plant mismatch robustness (20 scenarios): plant a +5 %, E +-3 %, B +5 %, c -5 %, kref +5 %,
  and a set of rounded parameters (a=1, E=8750, B=209.2, c=2.092): sim mean 0.294-0.325 vs 0.316
  nominal; max T <= 329.5 K. Controller not sensitive to the model errors I can plausibly have.
Conclusion: design frozen; spend plant batches on confirmation rather than tuning.

### E4 (batches 12-51) v1 at scale
`python plant_trial.py --run bb4 runs/bb4/work/c_v1b.py --n 40 --label v1b` (c_v1b.py = controller.py)
Why: 6 batches too few to judge; look for plant-specific failure modes.
Result: mean 0.178, median 0.158, p90 0.364, max 0.484, max T 328.7 K, no runaway, no errors.
Breakdown (mean per batch): start-up 0.113, setpoint changes 0.128, rest 0.018 (measured-Ca basis,
noise floor removed). Worst batches are start-ups 0.02-0.05 mol/L away from the setpoint with the
jacket already saturated at 302 K from sample 0 for 7-12 samples (b051, b030, b029) - actuator-limited,
not a controller deficiency. No sign of a plant feature the model misses.

### E5 (batches 52-151) final validation
`python plant_trial.py --run bb4 runs/bb4/work/controller.py --n 100 --label final`
(controller.py identical in content to c_v1b.py; only line endings differ.)
Result over 100 batches: mean 0.184 (bootstrap 95 % CI 0.156-0.214), median 0.132, p90 0.419,
max 0.775; no runaways, no controller errors; highest reactor temperature 329.5 K (limit 335).
Worst: b111 0.775 (start-up error +0.046, saturated), b091 0.624 (two 0.032 setpoint steps),
b069 0.595 (start-up +0.041).

## Budget
152 / 300 batches used: 6 identification, 46 development closed loop, 100 validation.
The remaining 148 were not needed: simulation showed the transients are within 0.1 % of the
model optimum and the rest of the cost is set by the actuator range and unannounced setpoint changes.
