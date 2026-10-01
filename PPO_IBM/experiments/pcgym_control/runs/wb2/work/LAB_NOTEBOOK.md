# Lab notebook: run wb2, jacketed CSTR

## Files read
- `PLANT_MANUAL.md` (root): interface, measurements, actuator range, cost, budget.
- `tasks.py` (root): scenario generator, noise levels, cost/timing definition.
- `pcgym/model_classes.py` (`cstr_ode` only), `pcgym/integrator.py`, `pcgym/pcgym.py`: plant equations,
  integrator (CVODES over one sample), order of disturbance/setpoint handling in `step`.
- Own files under `runs/wb2/` (trial CSVs, `results.jsonl`, `budget.json`).
- Seen by name only in a directory listing, not opened: `WRITER_BRIEFS.md`, `plant_trial.py`,
  `harness.py`, `cmaes_tune.py`, `rl_sac.py`, `run_baselines.sh`, `controllers/`, `results/`.

## What the source says
Plant (time in minutes, q/V = 1 /min):

    dCa/dt = (Caf - Ca) - k Ca                      k = 7.2e10 exp(-8750 / T)
    dT/dt  = (Ti - T) + 209.2 k Ca + 2.092 (Tc - T)

- Sample time 26/120 = 0.2167 min. Tc clipped to 295..302 K.
- Scenario per batch: Ca setpoint has 3 levels (2 changes) uniform in 0.86..0.90; Ti has 2 or 3 levels
  in 348.5..351.5 K; Caf 2 or 3 levels in 0.98..1.02; segments at least 20 samples long.
  Start state Ca 0.85..0.91, T 320..326 K (not at steady state). Noise sd: Ca 0.002, T 0.2 K.
- Timing: the action at sample k is integrated with Ti[k], Caf[k]; the resulting Ca is scored
  against sp[k], which is the setpoint the controller is shown at sample k. So a setpoint change
  is seen in the same sample in which it starts to count; there is no preview.
- Runaway flag: true T above 335 K at any sample.
- Steady state for Ca = 0.88, nominal feed: T = 324.2 K, Tc = 299.9 K. Local gain dCa/dT is about
  -0.009 mol/L per K, so the 0.04 setpoint span needs about 4.5 K of reactor temperature. The
  required steady Tc over all setpoint/feed corners runs from 295.6 K to 302.04 K: the actuator
  range only just covers the task, and the corner (sp 0.86, Caf 1.02, Ti 348.5) is marginally
  unreachable. Operating points in the range are open-loop stable (eigenvalues about -1.1 +/- 0.5i /min).

## Design approach
Because the model is known exactly, design work is done on my own copy of the equations
(`sim.py`, RK4, 20 substeps per sample; scenario generator re-written from the description in
tasks.py; no pcgym/tasks import). Pilot batches are used to (1) verify that copy against the real
plant, (2) validate candidates and the final controller.

Controller structure (`controller.py`):
- EKF on [Ca, T, Ti, Caf], feed disturbances modelled as random walks, estimates clipped to the
  known feed ranges.
- Nonlinear MPC: free jacket moves for M samples, then the steady-state Tc for the estimated feed;
  bounded Gauss-Newton (finite-difference Jacobian from vectorised RK4 rollouts, box QP by
  projected coordinate descent, short line search); soft ceiling on predicted T at 331 K; hard
  override Tc = 295 K if estimated T > 333 K.

## Experiments

### E1 (plant, 5 batches, b000-b004) first controller, model check
`python plant_trial.py --run wb2 --privileged runs/wb2/work/c01_mpc_ekf.py --n 5 --label c01`
Why: get privileged logs to verify my model before trusting any offline design.
Result: costs 0.197, 0.528, 0.773, 0.368, 0.103 (mean 0.394), max T 328.8 K, no runaway.
`verify_model.py`: one-step prediction of the true state from the logged previous true state,
Tc, Ti, Caf: max error 4.7e-6 mol/L and 5.5e-4 K (integrator tolerance). Noise sd 0.00204 and
0.2035 K, zero mean. Cost recomputed from the logs matches results.jsonl.
Learned: my model is the plant. Row k of the CSV holds the measurement taken before the move and
the true state after it. Nearly all the cost in these batches is the start-up transient with Tc
pinned at 302 K (start Ca well above the first setpoint with a cold reactor), e.g. b002: start
error 0.047 mol/L, 9 samples saturated, 0.75 of the 0.77 cost.

### S1 (own model, no batches) baseline and floor
`python sim.py controller.py --n 100 --jobs 4` (seeds 100000..100099)
- EKF + MPC (H16/M8): mean 0.165, median 0.139, max 0.53, no runaway.
- Oracle (true state and true current feed given to the same MPC): mean 0.137, median 0.120.
- Oracle with H30/M12, H10/M5, 8 iterations, rho 1e-5, 2 RK4 substeps: all 0.1368-0.1369.
Learned: the MPC is converged; horizon 10 / 5 moves is enough. 0.137 is the floor without foresight
of setpoint and feed changes, so at most about 0.03 (17%) can be gained from better estimation.
Phase split (`phases.py`, batch-cost units): oracle start 0.052, setpoint changes 0.084, feed
changes 0.002, quiet 0.000; EKF version 0.055 / 0.089 / 0.014 / 0.007.
Note: the 5 plant batches (mean 0.39) are well above the simulated mean; 4 of the 5 happened to
start with Ca 0.885-0.909 against a first setpoint of 0.860-0.870. To be watched as more plant
batches arrive (is the pilot plant's scenario mix harder than my re-written generator?).

### S2 (own model) estimator and move-penalty sweep, 100 seeds, H10/M5
`python phases.py controller.py --n 100 --params ...`
Random-walk sd (q_ti, q_caf) from (0.05, 0.0006) to (0.3, 0.004): mean cost 0.183, 0.167, 0.165
(0.14/0.0018), 0.165, 0.166, 0.170. Slow filter loses after feed steps (feed phase 0.030), fast
filter loses in quiet periods (0.012). Move penalty rho 1e-5 / 1e-3 / 1e-2 / 1e-1: 0.165 / 0.165 /
0.170 / 0.190. A first jump detector (sum of normalised innovation squared over 1-3 samples, then
re-open the feed covariance) gave 0.164-0.168: no gain.
Learned: the optimum is flat; a plain random-walk EKF cannot do much better than 0.165 here.

### E2 (plant, 20 batches, b005-b024) same design with H10/M5
`python plant_trial.py --run wb2 --privileged runs/wb2/work/c02_h10.py --n 20 --label c02`
Why: more plant scenarios, to see whether the plant's batches are really harder than my simulated ones.
Result: mean 0.231, median 0.149, max 0.682, max T 328.6 K, no runaway. All 25 so far: mean 0.263,
median 0.195.

### S3 (own model) replay of the 25 logged plant batches
`python replay.py c02_h10.py -v` rebuilds each logged scenario (setpoints, feed, back-solved start
state, the actual noise sequence) and runs the controller on my model.
Result: replayed cost equals the logged cost to 2e-4 in every batch (mean 0.2633 vs 0.2634).
So my simulation loop is the plant, and the 25 logged batches are a paired test bench that costs
no budget. Oracle MPC on the same 25: mean 0.2335, median 0.165, i.e. the 25 plant scenarios are
simply harder than average (start phase 0.129 vs 0.068 in 400 simulated seeds; mean squared start
error 5.8 vs 4.3 units). Estimation gap on the plant scenarios: 0.030 +/- 0.006, same as in simulation.
400 simulated seeds (0..399): EKF+MPC mean 0.179, median 0.126; oracle mean 0.151, median 0.104.
The distribution is heavy-tailed: the expensive batches start with high Ca, a cold reactor and a
low first setpoint, where the jacket is pinned at 302 K (heating authority about +4 K/min against
-10 K/min for cooling) for up to 10 samples.

### S4 (own model) CUSUM jump filter, 200 seeds (0..199)
Change: two-sided CUSUM on the normalised innovations of Ca and T (drift 0.5 sd, threshold h). On
an alarm the filter goes back L samples, adds variance to Ti and Caf there and re-filters the stored
measurements (`c03_cusum.py`, `python phases.py c03_cusum.py --n 200 --seed0 0 --params ...`).
The first sweep was stopped by the 30-minute background limit after 5 of 7 settings (machine busy);
the remaining settings were re-run in a second sweep (`sweep3.txt`).

| setting | mean | median | feed | quiet |
|---|---|---|---|---|
| plain EKF, q 0.14 / 0.0018 | 0.1791 | 0.1242 | 0.0146 | 0.0068 |
| CUSUM h4, q 0.05 / 0.0006 | 0.1781 | 0.1225 | 0.0164 | 0.0037 |
| CUSUM h3, q 0.05 / 0.0006 | 0.1766 | 0.1241 | 0.0151 | 0.0051 |
| CUSUM h5, q 0.05 / 0.0006 | 0.1796 | 0.1212 | 0.0175 | 0.0035 |
| CUSUM h4 L6, q 0.05 / 0.0006 | 0.1789 | 0.1212 | 0.0179 | 0.0035 |
| CUSUM h3, q 0.10 / 0.0012 | 0.1775 | 0.1218 | 0.0141 | 0.0059 |
| CUSUM h3, q 0.14 / 0.0018 | 0.1784 | 0.1270 | 0.0133 | 0.0072 |
| CUSUM h2.5 L3, q 0.07 / 0.0009 | 0.1788 | 0.1297 | 0.0138 | 0.0074 |
| CUSUM h3 scale 0.3, q 0.10 / 0.0012 | 0.1772 | 0.1212 | 0.0142 | 0.0055 |

Learned: everything sits between 0.177 and 0.180; the jump filter is worth about 1% at most and the
differences between its settings are not distinguishable. Reason: a typical feed-concentration
step (0.016 mol/L) makes Ca drift 0.0035 mol/L per sample against 0.002 of analyser noise, so any
filter needs 2-4 samples to size the step, while the oracle is told at once. The remaining gap to
the oracle (about 0.026) is set by the measurement noise, not by tuning. Estimator work stops here.

### S5 (own model) final settings, paired replay and stress test
Final settings in `controller.py`: H10/M5, 3 Gauss-Newton iterations, rho 1e-3, CUSUM h3, L4,
scale 0.3, q 0.10 K / 0.0012 mol/L. The oracle hook used for the floor studies was removed from
`controller.py` (it remains in the development copy `c03_cusum.py`); added a second trip of the
cold-jacket override on the raw T measurement (> 334 K) and a full estimator reset in the fallback.
- `python replay.py controller.py`: the 25 logged plant scenarios with their real noise: mean 0.2600,
  median 0.1925 against 0.2634 / 0.1950 logged for the plain-EKF versions; paired difference
  -0.0034 +/- 0.0012.
- `python stress.py controller.py`: 108 corner scenarios (setpoint 0.86/0.90 swings, Ti and Caf at
  their limits and swinging, start Ca 0.85/0.91, start T 320/326): highest T 330.35 K (constant
  hottest/richest feed, setpoint 0.86), no runaway; slowest act() 54 ms on the loaded machine; a NaN
  measurement is survived through the fallback.
(Several shell calls in this period were refused by a transient failure of the tool's permission
check and simply repeated later; no batches were involved.)

### E3 (plant, 100 batches, b025-b124) validation of the final controller
`python plant_trial.py --run wb2 --privileged runs/wb2/work/controller.py --n 100 --label final`
Why: validate the final `controller.py` unchanged on fresh plant scenarios.
Result (`python summary.py final`, before E4): mean 0.1855 (se 0.014), median 0.149, p90 0.375,
max 0.685, no runaway, highest T 329.0 K, no controller errors. Model re-check on these logs:
one-step error 5e-6 mol/L / 7e-4 K. Replay of the 100 scenarios reproduces the mean to 1e-4.
Oracle floor on the same 100: 0.1605 (gap 0.025 +/- 0.002).

### E4 (plant, 100 batches, b125-b224) second validation set, same file
`python plant_trial.py --run wb2 --privileged runs/wb2/work/controller.py --n 100 --label final2`
Why: halve the uncertainty of the validation mean; the controller was not changed between E3 and E4.
Result: mean 0.1582 (se 0.0125), median 0.126, p90 0.291, max 0.652, no runaway, highest T 328.8 K.

### Validation summary (E3 + E4, 200 batches, controller.py as delivered)
- mean cost 0.1718 (se 0.0094), median 0.1358, p90 0.330, max 0.685, min 0.012
- runaways 0 of 200; highest reactor temperature 329.0 K (limit 335 K); controller errors 0
- cost by phase (batch-cost units): start-up 0.059, after setpoint changes 0.093, after feed
  changes 0.014, quiet 0.006
- oracle MPC (true state and feed, no foresight) on the same 200 scenarios, on my model: 0.1460,
  median 0.113. Gap 0.026 +/- 0.002.
- The two validation sets differ (0.186 vs 0.158) by 1.4 standard errors: scenario luck.
- The first 25 batches (earlier plain-EKF versions, mean 0.263) were an unlucky draw of scenarios:
  the delivered controller scores 0.260 on those same 25 in replay.

### Files in work/
`controller.py` (deliverable), `c01_mpc_ekf.py`, `c02_h10.py` (versions run in E1, E2),
`c03_cusum.py` (development copy with the oracle hook for my simulator), `sim.py` (own plant model
and scenario generator), `phases.py`, `replay.py`, `stress.py`, `verify_model.py`, `summary.py`
(analysis), `sweep3.txt`, `final_a.txt`, `final_b.txt` (raw outputs).

### Budget
225 of 300 batches used: 5 (model check) + 20 (scenario check) + 200 (validation). 75 unused.

### Background jobs that were stopped
- First CUSUM sweep (own model): killed by the 30-minute background limit after 5 of 7 settings;
  the missing settings were covered by the second sweep.
- A shell loop that only waited for that sweep's output: killed by the same limit; nothing lost.

### Open points
- About 85% of the cost is the unavoidable response to start-up and setpoint steps with the jacket
  at a limit (heating authority is small: 302 K is between 0 and 6 K above the steady jacket
  temperature, typically 1-2 K). No controller without foresight can remove it.
- The 0.026 gap to the oracle is the price of the analyser noise when sizing feed steps; I did not
  try a full Bayesian jump filter or a smoother, which might recover a small part of it.
- The controller relies on the model being exact (verified to integrator tolerance on 225 batches)
  and on the feed ranges in tasks.py (estimates are clipped to them). A plant that drifts from the
  design model would need wider process noise.
- Runaway margin: no batch above 329.0 K on the plant, 330.4 K in corner simulations; the 333/334 K
  cold-jacket override was never triggered, so it is untested on the plant.
- act() takes 10-50 ms on this loaded machine.
