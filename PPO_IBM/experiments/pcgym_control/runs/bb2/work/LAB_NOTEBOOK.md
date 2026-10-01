# Lab notebook: run bb2, jacketed CSTR concentration control

All commands are run from the experiment root. Plant batches are only ever run through
`python plant_trial.py --run bb2 ...`. Budget: 300 batches.

## Files read
- `PLANT_MANUAL.md` (operator's manual): the only file read outside `runs/bb2/`.
- `runs/bb2/trials/*.csv`, `runs/bb2/trials/results.jsonl`: my own trial logs.
- Nothing else in the repository or in the Python environment was opened, listed or imported
  (my scripts import numpy, scipy.optimize, and my own files in `runs/bb2/work/`).

## Files written (all in `runs/bb2/work/`)
| file | purpose |
|---|---|
| `c01_const.py` | open-loop constant jacket 298.5 K |
| `c02_steps.py` | open-loop random jacket steps for identification |
| `c03_mpc_v1.py` | first EKF + NMPC controller, hand-guessed model numbers |
| `c04_mpc_v2.py` | same controller with the model fitted to b000-b012 |
| `c05_mpc_v3.py` | refit model (b002-b032) plus model-predicted temperature ceiling |
| `controller.py` | the deliverable (= v3 plus a guard for non-finite measurements) |
| `show.py`, `spstats.py`, `phases.py` | log viewers (read only `runs/bb2/trials/`) |
| `fit.py`, `fit2.py` | model fit by EKF innovation likelihood (read only `runs/bb2/trials/`) |
| `sim.py`, `sweep.py`, `decomp.py`, `steptest.py`, `robust.py` | simulator of my own fitted model, for tuning without spending batches |
| `innov.py`, `stats.py` | held-out model check; cost statistics per label (read only `runs/bb2/trials/`) |
| `fit_p.npy`, `fit2_p.npy`, `fit3_p.npy`, `fit3_out.txt` | fit results |

## Prior knowledge used
The model structure (two balances with an Arrhenius rate) and the starting guesses for the fit
are standard textbook CSTR knowledge that I brought with me, not something read from the
repository. Every number in the controller was then fitted to, and checked against, my own
pilot batches.

## What the manual says (starting knowledge)
- Exothermic CSTR, constant feed flow; control Ca by the jacket temperature, 295-302 K.
- 120 samples per 26 min batch (dt = 0.2167 min = 13 s). Measurements: Ca (noisy), T (noisy), Ca_sp.
- Unmeasured feed temperature and feed concentration shifts. Runaway if T > 335 K.
- Cost = mean of ((Ca - sp)/0.01)^2 on the true Ca.

---

## E1: open loop at mid-range (batches b000-b001, 2 used)
`python plant_trial.py --run bb2 runs/bb2/work/c01_const.py --n 2 --label const2985`

Why: see the log format, the operating point, noise levels and natural time scale before
moving the actuator.

Result: cost 8.90 and 3.03, max T 323.8 K, no runaway.
- Log columns: step, t_min, Ca, T, Ca_sp, Tc. The logged Ca is the noisy measurement (cost
  recomputed from the log differs slightly from the reported cost).
- At Tc = 298.5 K the reactor sits at Ca about 0.90-0.91 mol/L, T about 321.4 K: open-loop stable.
- Start-up is off equilibrium (Ca0 about 0.855, T0 321-324 K) and settles in about 10 samples, so
  the dominant time constant is roughly 1 min.
- Setpoints 0.86-0.90 mol/L, two changes per batch at irregular times.
- Feed shifts are visible as steps: b000 Ca drops about 0.02 around sample 80; b001 T rises
  about 1.7 K around sample 42.
- Noise by eye: Ca about 0.003 mol/L, T about 0.3 K.

Engineering hypothesis: textbook single-reaction CSTR,
`dCa/dt = a(Caf - Ca) - k(T) Ca`, `dT/dt = a(Tf - T) + b k(T) Ca + c(Tc - T)`, Arrhenius k(T).
Hand check of the b000 steady state with textbook-style numbers (a = 1/min, k about 0.11/min at
321 K, b about 209 K L/mol, c about 2.09/min, Tf about 350 K, Caf about 1) closes the heat and
mass balances to about 1 K/min, so this structure is plausible. To be confirmed by a fit.

Runaway estimate from that model before stepping the jacket: even at Tc = 302 K the steady
state is about 329 K and the ignition point is above 340 K, so full-range steps should be safe.
A T > 331 K full-cooling override was added to the test signal anyway.

## E2: open-loop identification steps (b002-b007, 6 used, total 8)
`python plant_trial.py --run bb2 runs/bb2/work/c02_steps.py --n 6 --label idsteps`

Why: excite the plant over the whole actuator range (random levels 295-302 K, random holds
of 4-21 samples) to fit the model.

Result: costs 2.3-28 (irrelevant), max T 330.7 K, no runaway, override never needed.
Setpoint and start-up statistics over b000-b007: Ca0 0.855-0.906, T0 320.2-325.4 K; setpoints
0.860-0.899; first change at sample 20-60, second at 57-99.

## E3: first closed-loop controller, EKF + nonlinear MPC (b008-b012, 5 used, total 13)
`python plant_trial.py --run bb2 runs/bb2/work/c03_mpc_v1.py --n 5 --label mpc_v1`

Why: the first fit script was slow on the shared machine, so in the meantime I tried the
intended structure with the hand-checked model numbers (a=1, kref=0.11 at 322 K, E/R=8750,
b=209, c=2.09).

Controller: EKF with states (Ca, T, Caf, Tf), feed disturbances as random walks; MPC with
horizon 12 samples, 4 free moves then the steady-state jacket for the setpoint, squared Ca error
only, Gauss-Newton with clipping.

Result: costs 0.052, 0.124, 0.503, 0.071, 0.134 (mean 0.18), max T 327.5 K. No runaway.
Roughly 50x better than open loop already.
- b010 (0.50): the first setpoint 0.864 needed the jacket pinned at 302 K for the whole phase.
  The actuator range, not the controller, limited that batch.
- The jacket is very busy in steady state (jumps between the limits on noise). Costs nothing
  in the score but noted.

## A1: model fit (no batches)
`python runs/bb2/work/fit.py "b00[2-9]*" "b01[0-2]*"`, then `python runs/bb2/work/fit2.py "b00*" "b01[0-2]*"`

Method: maximise the EKF innovation likelihood over (a, kref, E/R, b, c, noise levels,
disturbance random-walk rates), starting from the hand-checked numbers. (The first version of
`fit.py` used a numerical Jacobian and did not finish in 10 min; I stopped my own job and
rewrote it with an analytic Jacobian.)

Results:
- Input timing: likelihood with the logged Tc acting over the following sample is -7176; with one
  extra sample of delay -3996. So there is no dead time.
- The first fit also floated the initial disturbance guess and returned a meaningless
  Caf0 = -5 (not identifiable; the first 10 samples are not scored). `fit2.py` fixes
  Caf0 = 1, Tf0 = 350.
- Fit on b000-b012: **a = 1.0445 /min, kref = 0.11245 /min at 322 K, E/R = 8948 K,
  b = 207.5 K L/mol, c = 2.076 /min**; noise sCa = 0.0020 mol/L, sT = 0.20 K. Nll -9143 against
  -8491 for the hand-checked numbers (most of the gain is from the noise levels).
- Innovation std about 0.0026 mol/L and 0.25 K.
- Filtered feed disturbances over 13 batches: Caf between 0.973 and 1.022 with steps of up to
  about 0.03; Tf (in model units) between 347.3 and 350.6 K with steps of about 2 K;
  0-2 shifts of each per batch. Mean Tf about 349 K.

## A2: own-model simulator and tuning (no batches)
`sim.py` simulates the fitted model with the scenario statistics above.
- Calibration: v1 controller in the simulator: mean 0.256, median 0.18 (plant: mean 0.18 on 5
  batches). Same order; the simulator is a little pessimistic.
- `decomp.py` (40 scenarios): normal 0.255, noise-free measurements 0.230, oracle knowledge of
  state and disturbances 0.2245. So estimation costs only about 12 %; the rest is transients.
- `sweep.py`: horizon (N 8-20, M 3-6), move penalty rho (0-0.1), filter noise settings: all
  within 0.265-0.285 on 60 paired scenarios. Slower disturbance random walk (qCaf 0.001-0.002,
  qTf 0.1-0.2) is about 0.02 better than my first guess (0.005, 0.5). A move penalty does not help.
- `steptest.py`: noise-free setpoint steps against a brute-force search over bang-bang
  sequences. MPC matches the brute-force optimum to 0.1 % (e.g. 0.87->0.90: 15.12 vs 15.13;
  0.90->0.862: 43.42 vs 43.46 summed squared error). A longer horizon or more iterations
  changes nothing. **The setpoint response is at the actuator limit**: a 0.03 mol/L step
  costs 0.13 (upward) to 0.19 (downward) cost units per batch all by itself. Downward steps in
  Ca are slower because heating is limited by the 302 K ceiling.

## E4: controller v2 with fitted model (b013-b032, 20 used, total 33)
`python plant_trial.py --run bb2 runs/bb2/work/c04_mpc_v2.py --n 20 --label mpc_v2`

Why: check the fitted-model controller on the plant with a sample big enough to compare with
the simulator.

Result: mean 0.165, median 0.159, min 0.028, max 0.358, max T 328.6 K, no runaway.
`python runs/bb2/work/phases.py "b*mpc_v2.csv"` (measured error by phase, cost units):
start-up 0.078, 10 samples after setpoint changes 0.153, everything else 0.050.
- Measured cost exceeds true cost by about 0.11 per batch: about 0.05 is analyser noise, the rest
  is the sample at which the setpoint changes. So the true cost evidently compares the
  setpoint shown at sample i with the concentration reached at sample i+1 (the result of the
  action), which is what the MPC already optimises.
- The steady-state part is almost entirely analyser noise (0.04-0.05): true regulation error
  is about 0.01-0.02.
- Batch cost tracks setpoint step sizes (b023: -0.035 and +0.024 -> 0.36; b028: -0.001 and
  +0.007 -> 0.03).
Exact v2 statistics (`python runs/bb2/work/stats.py`): n=20, mean 0.1646, median 0.1590,
sd 0.101, min 0.028, max 0.358, no runaways, max T 328.6 K.

Check of the cost-timing reading: b028 (tiny setpoint steps) measured minus true = 0.050, which
is pure analyser noise and gives sCa = 0.0022 mol/L; b023 (steps -0.035, +0.024) measured
minus true = 0.204 = 0.05 noise + 0.15 from the two change samples. Consistent.

## A3: refit on 31 batches, jump-detection trial, safety ceiling (no batches)
- `python runs/bb2/work/fit2.py "b00[2-9]*" "b0[123]*"` (b002-b032): **a = 0.997 /min,
  kref = 0.1070 /min at 322 K, E/R = 9012 K, b = 215.5 K L/mol, c = 2.108 /min**, sCa = 0.0020,
  sT = 0.197, random-walk rates qCaf = 0.0018, qTf = 0.17 per sample. Output kept in
  `fit3_out.txt`, parameters in `fit3_p.npy` (`fit2_p.npy` holds the b000-b012 fit).
  The parameters moved about 5 % from the 13-batch fit: a, kref and the feed levels trade off
  against each other (only combinations are identifiable). In this model's units the feed sits
  at Caf 0.994 +/- 0.010 (range 0.973-1.022) and Tf 351.1 +/- 1.0 K (range 348.6-353.1).
- Jump detection in the filter (inflate the disturbance covariance when the normalised
  innovation is large): 60 paired simulated scenarios, 0.2648 without, 0.2626-0.2654 with.
  No measurable benefit; left in the code but switched off (`jump=0`).
- `robust.py`: v3 controller against simulated plants that differ from its model. Plant = first
  fit: 0.245 (no loss). Rate constant +10 %, heat of reaction +10 %, cooling -10 %: 0.22-0.24.
  Doubled noise: 0.295. Cooling +10 % or feed 1.5-2 K colder: 0.42-2.0, but these cases make low
  setpoints unreachable with the 302 K jacket ceiling (a feasibility limit, not a tuning one).
- Safety: the first version used a hard override (full cooling above 331.5 K), which chattered
  in a harsh simulated case (Caf 1.06, Tf 356 K, setpoint 0.80, start at 331 K; peak 333.4 K).
  Replaced by a model-predicted ceiling: the jacket is lowered just enough that the predicted
  next-sample T stays below 332 K; full cooling if T is already above 333 K. Same harsh case
  now settles at 332.1 K (peak 333.4 K in the first samples while the feed is being learned).
  In normal operation the ceiling is never active (plant max T so far 329.5 K).

## E5: controller v3 = refit model + safety ceiling (b033-b072, 40 used, total 73)
`python plant_trial.py --run bb2 runs/bb2/work/c05_mpc_v3.py --n 40 --label mpc_v3`

Result: mean 0.1684, median 0.1147, sd 0.145, min 0.013, max 0.657, no runaways, max T 328.9 K.
Statistically the same as v2 (0.165 +/- 0.023), as the simulator predicted.
- Worst batches are the ones with the largest setpoint steps: b054 (+0.032, -0.036) 0.657;
  b050 (-0.034) 0.445; b044 (-0.027, +0.033) 0.344. The model's time-optimal cost for a
  +0.03 / -0.036 pair is already about 0.47, plus start-up.
- `python runs/bb2/work/innov.py "b*mpc_v3.csv"` (these 40 batches were not used in any fit):
  innovation mean -0.00003 mol/L and +0.006 K, std 0.00254 and 0.247, lag-1 autocorrelation
  +0.02 and -0.05; 5-step-ahead prediction error std 0.0036 mol/L and 0.32 K. Both parameter
  sets predict equally well (nll -27788 vs -27800). The model is adequate; nothing left to fit.

## E6: validation of the final controller (b073-b152, 80 used, total 153)
`controller.py` = v3 plus a guard for non-finite measurements (verified identical outputs to v3
on 10 simulated scenarios).
`python plant_trial.py --run bb2 runs/bb2/work/controller.py --n 80 --label final`

Result: **mean 0.1824, median 0.1425**, sd 0.146 (standard error of the mean 0.016), min 0.026,
max 0.739, **0 runaways**, max T 329.5 K, no controller errors.
Pooled with the 40 v3 batches (same control law): n = 120, mean 0.178.
Phase split of the measured error: start-up 0.118, setpoint changes 0.144 (of which about 0.06
is the uncounted change sample), rest 0.053 (of which about 0.04 is analyser noise).
Worst batch b074 (0.739): setpoint steps +0.031 and -0.036 plus a large start-up offset.

## Summary of batches
| batches | label | n | purpose | mean cost |
|---|---|---|---|---|
| b000-b001 | const2985 | 2 | open loop, constant jacket | 5.97 |
| b002-b007 | idsteps | 6 | open-loop identification steps | 11.0 |
| b008-b012 | mpc_v1 | 5 | EKF+NMPC, hand-checked model | 0.177 |
| b013-b032 | mpc_v2 | 20 | fitted model (13 batches) | 0.165 |
| b033-b072 | mpc_v3 | 40 | refit model (31 batches) + T ceiling | 0.168 |
| b073-b152 | final | 80 | validation of `controller.py` | 0.182 |

153 of 300 batches used; 147 not spent (no experiment left that the plant could resolve:
batch-to-batch sd is 0.15, so a 0.01 difference between tunings would need about 900 batches
per arm).

## Open points
- a, kref and the nominal feed levels are only identified in combination (two fits 5 % apart
  predict equally well). Does not matter for control here.
- Low setpoints with a cold/lean feed pin the jacket at 302 K; high-side authority is the
  binding limit. Feed shifts or setpoints outside the ranges seen (Caf 0.97-1.02, Tf span of
  about 4.5 K, setpoints 0.86-0.90) are untested on the plant.
- The temperature ceiling has only been exercised in simulation.
- The jacket command is very active (often limit to limit on noise). Free in the cost; a real
  utility system might object. `rho` = 0.02 calms it at a cost of about 0.003 in simulation (0.2703 vs 0.2674 on 60 scenarios).
