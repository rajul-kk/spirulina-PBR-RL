# Lab notebook - run wb3 (jacketed CSTR, Ca setpoint tracking)

Budget: 300 plant batches. All plant runs: `python plant_trial.py --run wb3 --privileged runs/wb3/work/<file>.py --n N --label L` from the root.

## Files read
- `PLANT_MANUAL.md` (root): obs keys, actuator 295-302 K, runaway at T > 335 K, cost = mean ((Ca - sp)/0.01)^2 on true Ca.
- `tasks.py` (root): scenario generator and episode loop.
  - 120 samples, 26 min, dt = 0.21667 min. Sensor noise sd: Ca 0.002, T 0.2 K (Gaussian, added to the observation only).
  - Setpoint: piecewise constant, exactly 2 changes, levels U(0.86, 0.90), change indices in 20..99, segments >= 20 samples.
  - Feed disturbances Ti U(348.5, 351.5), Caf U(0.98, 1.02): each 1 or 2 step changes (equally likely), same timing rules.
  - Start state Ca U(0.85, 0.91), T U(320, 326).
  - Timing: obs at sample k = noisy state at k and sp[k]; action applied over [k, k+1) with Ti[k], Caf[k]; cost term k = (Ca_{k+1} - sp[k])^2. So a setpoint change is seen with no preview and the first sample after it is charged almost in full.
  - A controller exception gives Tc = 298.5 for that sample.
- `pcgym/model_classes.py` (cstr_ode): dCa/dt = q/V (Caf - Ca) - k0 exp(-E/RT) Ca; dT/dt = q/V (Ti - T) + (-dH)/(rho C) r + UA/(rho C V) (Tc - T). q/V = 1 /min, k0 = 7.2e10, E/R = 8750 K, (-dH)/(rho C) = 209.2 K L/mol, UA/(rho C V) = 2.092 /min.
- `pcgym/integrator.py`: casadi cvodes over one sample (accurate integration, zero-order hold on Tc and disturbances).
- `pcgym/pcgym.py` (reset/step): confirmed disturbance index used for interval k is k; no extra noise (noise flag is False).

## Model analysis (desk work, no batches)
- Steady state for Ca = 0.88, nominal feed: k = 0.1364 /min, T = 324.2 K, Tc = 299.9 K. For sp 0.86..0.90 and the feed ranges the required Tc runs from about 295.6 K to about 302.0 K: the actuator range only just covers the steady states, so moves are saturation limited.
- Linearised at the operating point: stable (eigenvalues about -1.07 +/- 0.53i per min), Ca time constant about 0.9 min (4 samples). Tc acts on Ca through T (relative degree 2).
- Own simulator `mysim.py` (RK4, 8 substeps) and own copy of the scenario distributions; `evalsim.py`, `decomp.py` run controllers in it offline (no plant budget).

## Experiments on the plant
### E1 - b000-b002, `c0_probe.py`, n=3, label probe
Why: check the tool, the log format and whether my model reproduces the plant.
Result: costs 1.27, 0.62, 0.93 (crude P + dither). Log columns: step,t_min,Ca,T,Ca_sp,Tc,Ca_true,T_true,Ti,Caf; Ca_true/T_true in row k are the state after the step (k+1).
`check_replay.py`: my model replays the logged batches with max one-step error 4.5e-6 mol/L, 5.7e-4 K; open-loop over a whole batch 9e-6 mol/L, 9e-4 K. Noise sd measured 0.0020-0.0022, 0.19-0.21 K.
Learned: the model is exact for design purposes; all tuning can be done offline, the plant budget is for validation.

### Offline design (my simulator, seeds are my own)
- `c1_nmpc.py`: EKF on [Ca, T, Ti, Caf] (random-walk disturbances) + NMPC (Gauss-Newton on a 24-sample horizon with box constraints). 20 seeds: 0.1405; with perfect information ("oracle": true state and current disturbances, future unknown) 0.128. Slow (4.4 s/batch).
- `c2_nmpc.py`: same in scalar math + active-set box QP, 5x faster. 100 seeds (2000..2099): EKF 0.1605, oracle 0.1391. Cost by cause (oracle): start-up 0.053, setpoint changes 0.086, disturbances 0.001, quiet 0.000. The MPC settings do not matter (H 12/24/40, 1..6 GN iterations, 2 or 4 RK4 substeps all give 0.1391): the floor is the unavoidable transition cost. EKF adds: quiet +0.005, disturbance +0.009, sp +0.005.
- `c3_mht.py`: EKF replaced by a multi-hypothesis filter (hypotheses = when the feed shifts happened; hazard from the generator rules; on a shift the disturbance is reset to its prior; 16 best hypotheses kept). 100 seeds: 0.1567 (quiet 0.0019, dist 0.0086). 40 hypotheses 0.1556, 6 hypotheses 0.1572.
- Setpoint hedging: target pulled towards 0.88 (mean of a new setpoint) by h c/(1 + h c), h = chance of a setpoint change at that sample (exactly two changes, >= 20 apart, indices 20..99). 100 seeds: c=1 0.1512, c=2 0.1496, c=4 0.1507, c=8 0.1621 (c=0 0.1567).
- 300 fresh seeds (5000..5299): EKF (c2) 0.1788; MHT 0.1728; MHT + hedge c=1.5 0.1661, c=2 0.1655, c=3 0.1659; oracle without hedge 0.1510. No runaway offline, max T 329.5 K.

### E2 - b003-b022, `c3_mht.py` (hedge_c = 2), n=20, label mht_h2
Why: first plant check of the designed controller; confirm the offline numbers and the scenario statistics.
Result: mean 0.1787, median 0.143, max 0.462, no runaway, max T 328.6 K. 36 s for 20 batches.
`analyze_trials.py`: sp changes exactly 2 per batch at indices 21..99; Ti and Caf shifts 1 or 2 per batch (13/7 each) at indices 20..98; noise sd 0.00195 / 0.204 K, zero mean; replaying the controller on the logged measurements reproduces the logged Tc exactly; model replay error still < 1e-5 mol/L. Filter one-step prediction rms Ca 0.0011, T 0.10 K; disturbance estimate rms Ti 0.28 K, Caf 0.0035.
Learned: plant agrees with the offline simulator (0.179 +/- 0.03 vs 0.166 offline); batch-to-batch sd is about 0.13, so the plant cannot resolve differences of 0.01 between designs in a few dozen batches - comparisons are done offline on common seeds.

### Offline sweep (sweep1.log, 300 seeds 5000..5299, base c3 = 0.1655)
n_hyp 32: 0.1651; q_T/q_ca x0.3 or x3: 0.1655; jump hazard x2: 0.1660, x0.5: 0.1659; 2 GN iterations: 0.1655.
Learned: the design is on a flat optimum; nothing left to tune at the 0.001 level. Kept c3 settings.

(Session interrupted by a usage limit here and resumed; files and logs intact, budget 23 used at resume.)

### Final controller `controller.py`
= c3_mht.py with the offline-only "oracle" option removed, plus a safety net: any exception or non-finite
value -> estimator restarts at the next sample and this sample uses a steady-state feedforward + P law
(alone it scores 0.36 offline). Fault injection offline (NaN Ca at one sample: 0.252 vs 0.246 clean; NaN for
30 samples: 0.695, no runaway). Offline 300 seeds: 0.1655 (identical to c3).

### E3 - b023-b122, `controller.py`, n=100, label final (validation)
Why: validation of the final controller on fresh plant batches.
Result: mean 0.1664 (SEM 0.016), median 0.117, 90th percentile 0.335, max 0.949 (b106: start-up offset Ca0 0.909 vs
sp 0.869 is 31% of its cost, plus two large setpoint steps). No runaway, no controller error, max T 328.9 K.
Replay of the logged measurements reproduces every logged Tc exactly. Filter one-step rms Ca 0.0012, T 0.107 K;
disturbance estimate rms Ti 0.30 K, Caf 0.0037. Scenario statistics match tasks.py (Ti 50/50 one/two shifts).
Learned: plant result equals the offline prediction (0.1655). Budget used 123/300; the rest is left unused because
the offline simulator reproduces the plant to 1e-5 mol/L and further plant batches could not separate designs
whose offline differences are under 0.001.
