# Lab notebook: run wb1 (jacketed CSTR, Ca setpoint control)

## Files read
- `PLANT_MANUAL.md` (root): job, measurements, actuator 295-302 K, runaway at 335 K, cost definition.
- `tasks.py` (root): scenario generator and scoring.
- `pcgym/model_classes.py` (class `cstr_ode`): the plant equations and parameters.
- `pcgym/integrator.py`: the plant is integrated with CVODES over one sample, input held constant.
- `pcgym/pcgym.py` (grep of `step`/`reset`/disturbance lines only): the disturbance value of
  sample k is applied during sample k; dt = tsim/N.
- Files under `runs/wb1/` (my own trials and work).
- Nothing else. In particular not plant_trial.py, harness.py, controllers/, results/, docs/, other runs.

## What the source says
Plant (time in minutes, q/V = 1 /min):

    k(T)    = 7.2e10 * exp(-8750 / T)
    dCa/dt  = (Caf - Ca) - k(T) Ca
    dT/dt   = (Ti - T) + 209.205 k(T) Ca + 2.09205 (Tc - T)

- Sample time 26/120 = 0.2167 min; the jacket temperature is held over the sample.
- No process noise. Sensor noise: Ca sd 0.002 mol/L (0.2 cost units), T sd 0.2 K.
- Scenario: 3 setpoint levels uniform in 0.86-0.90 (2 changes); Ti uniform in 348.5-351.5 K and
  Caf uniform in 0.98-1.02, each with 1 or 2 step changes; all changes fall in samples 20-99 and
  are at least 20 samples apart within a signal. Start: Ca0 in 0.85-0.91, T0 in 320-326 K.
- Scoring: the state at the END of sample k is compared with the setpoint shown at sample k. So a
  setpoint change costs at least one sample of full error, whatever the controller does.
- Steady state: Ca = Caf/(1+k). For Ca 0.86 / 0.88 / 0.90 at nominal feed: T = 326.3 / 324.2 /
  321.7 K and Tc = 301.0 / 299.9 / 298.2 K. Feed shifts move the required Tc by about -0.48 K per K
  of Ti; low setpoints with rich, cold feed need Tc slightly above 302 K (infeasible corner:
  sp 0.86, Caf 1.02, Ti 348.5 needs 302.1 K).
- Open loop is stable at the operating points (eigenvalues about -1.07 +- 0.53i /min at Ca 0.88),
  gain sign is constant (Tc up -> T up -> Ca down).
- Runaway mechanism: with Tc = 302 K held, hot rich feed (Ti 351.5, Caf 1.02) has no low-temperature
  steady state: heat balance stays positive through 336 K. A controller that parks at 302 K
  (wind-up, wrong sign) will ignite; one that follows Ca will not, because Ca falls well below
  the setpoint long before (T 328 K already gives Ca 0.859).

## Plan
The plant is deterministic and fully known, so design is done on my own copy of the model
(`plantsim.py`, written from the equations above; it does not import pcgym or tasks.py). The pilot
budget is used to (1) check that my model reproduces the plant, (2) validate the final controller.

Controller concept: extended Kalman filter on [Ca, T, Ti, Caf] + nonlinear MPC (minimise predicted
squared Ca error over the horizon, jacket limits as hard bounds).

## Offline work (my own model, no budget)
Tools: `plantsim.py` (model, scenario generator with the same distributions, runner), `tune.py`
(variants on common scenarios), `breakdown.py` (cost by event type), `dbg*.py` (traces).
"oracle" = MPC fed the true state and true current disturbances; an optimistic bound.

1. First MPC (clip-all-violations bounded least squares): mean 0.22 (30 scen.), oracle 0.20.
   Traces showed the optimiser stalling at the steady-state jacket value during transients: the
   unconstrained Gauss-Newton step is bang-bang (+45, -55, ...), clipping all of it at once gave
   no descent and the plan never moved.
2. Second attempt (gradient-sign active set): mean 1.5, max 12. Bug: ratio test returned step 0
   when a free variable sat on its bound; plan stuck at 302 K, Ca undershot to 0.805, T to 332 K.
   Lesson: a wrong optimiser parks the jacket at 302 K, which is exactly the runaway path.
3. Proper primal active-set box QP: EKF+MPC 0.172, oracle 0.140 (60 scen.). Horizon 15, 25 or 40
   samples and 1, 3 or 6 Gauss-Newton iterations give the same cost: converged. Chose horizon 15
   (3.25 min), blocks (1,1,1,1,2,2) then steady-state jacket value, 2 iterations.
4. Estimator tuning (100 scen., random-walk disturbance model): q_Ti 0.02 K^2, q_Caf 3e-6 per
   sample (second moments of the scenario's step changes) are at the optimum; x4 or /4 is worse
   by 0.001-0.007. Move penalty rho=0.01: worse by 0.007. Clipping the estimates to the feed
   range: 0.0005 better.
5. Cost breakdown (100 scen.), EKF vs oracle: start-up 0.051 / 0.049, setpoint steps 0.095 /
   0.091, Ti steps 0.007 / 0.004, Caf steps 0.020 / 0.007, quiet periods 0.006 / 0.000. So 80% of
   the cost is start-up and setpoint transitions at the actuator limits; estimation costs 0.03.
6. Hedging against the next setpoint change (sitting slightly off-setpoint): estimated gain
   below 0.001; not pursued.
7. Jump-hypothesis filter (GPB1: no change / Ti stepped / Caf stepped, merged each sample):
   100 scen.: random-walk 0.1788; p_jump 0.02: 0.1773; p_jump 0.005: 0.1744 (-0.0044 +- 0.0013).

## Pilot trials
| batches | command | why | result | learned |
|---|---|---|---|---|
| b000-b003 | `python plant_trial.py --run wb1 --privileged runs/wb1/work/controller.py --n 4 --label v1` | check my model against the plant (controller = first MPC, item 1 above) | costs 0.167, 0.057, 0.178, 0.127; max T 327.9 K | `validate_model.py`: one-step prediction error of my model vs the true states is max 4.7e-6 mol/L and 4.8e-4 K over 476 steps; measured noise sd 0.0019-0.0021 and 0.18-0.22 K. The model is the plant. Timing convention confirmed (logged truth at row k is the state after sample k). |
| b004-b043 | `python plant_trial.py --run wb1 --privileged runs/wb1/work/controller_v3.py --n 40 --label v3` | first plant check of the frozen candidate (GPB1 filter + MPC; `controller_v3.py` is byte-identical to `controller.py`) | mean 0.174 (sem 0.028), median 0.130, max 0.887, max T 328.6 K, no runaway, no controller errors; 2.2 s per batch | Agrees with the offline prediction (0.170). Worst batch b008: two large setpoint steps (0.895 -> 0.865 at start-up, 0.894 -> 0.862) ridden at Tc = 302 K for 8-9 samples each; nothing left to gain there. Model check again: 4.8e-6 mol/L, 6.2e-4 K. |
| b044-b153 | `python plant_trial.py --run wb1 --privileged runs/wb1/work/controller.py --n 110 --label final` | validation of the frozen controller | mean 0.195 (sem 0.017), median 0.146, p90 0.395, max 1.135, max T 329.6 K, no runaway, no errors | Together with b004-b043 (same controller): n = 150, mean 0.189 (sem 0.015), median 0.140, p90 0.395, max 1.135, 0 runaways, hottest batch 329.6 K. `worst.py`: 93.6% of all cost is incurred in samples where the jacket is at 295 or 302 K. |

Budget: 154 of 300 batches used; 146 left unused (nothing further to learn from the plant: my model
reproduces it to 5e-6 mol/L per step).

## Offline work, continued
8. Fresh 200 scenarios: random-walk EKF 0.1769; GPB1 p_jump 0.002 / 0.005 / 0.01: 0.1706 / 0.1708 /
   0.1710 (flat); with the change probability lowered outside samples 20-99: 0.1697-0.1699;
   oracle 0.1475. Slow-drift floor q x3 or /10: no difference.
   Final: p_jump 0.01 inside samples 20-99, 0.0005 outside (soft, not zero), q_Ti 1e-3, q_Caf 1e-7:
   0.1698. Hard window: 0.1697. No window, p 0.005: 0.1708.
9. Keeping the disturbance estimates inside the feed range: no clip +0.0002, margin 0.005 mol/L
   (0.375 K) +0.0001. Final uses the margin. Soft temperature limit (333 K) never active: same cost
   with it off; kept as a guard.
10. Numerical guards added: log-domain hypothesis weights; `act` falls back to steady-state jacket
    value + proportional term if anything throws (counter `n_fallback`; never triggered in 1000+
    offline batches, and the pilot logs show error = null for every batch).
11. Stress test (`stress.py`, 30 scenarios, plant differs from the controller's model): nominal
    0.171; UA -10% 0.287; UA +10% 1.18; k0 +10% 0.179; k0 -10% 0.843; sensor noise x3 0.370. No
    runaway in any case (max T 331.0 K). The controller leans on the model: mismatch that the two
    disturbance estimates cannot absorb inside their allowed range leaves an offset.

## Final design (controller.py)
- Estimator: EKF on [Ca, T, Ti, Caf] with the plant equations; three hypotheses per sample (no
  change, Ti stepped with variance 1.5 K^2, Caf stepped with variance 2.67e-4), weighted by the
  measurement likelihood and merged. R = (0.002^2, 0.2^2).
- Control: nonlinear MPC, horizon 15 samples, 6 move blocks (1,1,1,1,2,2) then the steady-state
  jacket value for the estimated feed; cost = sum of squared predicted Ca error (+ soft limit on
  T > 333 K); Gauss-Newton with finite-difference Jacobian and an exact box-constrained QP
  (active set), 2 iterations, warm start. No move penalty.
- Only the measurements in the manual are used. `obs["_truth"]` is read only if present, which
  happens only in my offline oracle studies, never on the plant.
