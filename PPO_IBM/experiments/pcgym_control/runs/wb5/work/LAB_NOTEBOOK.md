# Lab notebook - run wb5 (CSTR concentration control)

## Files read
- PLANT_MANUAL.md (root)
- tasks.py (root): task definition. N=120 samples over 26 min (dt=0.2167 min); Tc in [295,302];
  setpoint piecewise constant in [0.86,0.90] with 2 changes >=20 samples apart; Ti steps in
  [348.5,351.5] and Caf steps in [0.98,1.02], 1-2 changes each; x0 Ca~U(0.85,0.91), T~U(320,326);
  noise sd Ca 0.002, T 0.2 K. Cost at step k uses state AFTER the action at k against sp[k]
  (the setpoint shown in obs at step k), so the setpoint is known when acting.
  Controller exception -> Tc held at 298.5.
- E:\SEGP\.venv\Lib\site-packages\pcgym\model_classes.py (class cstr_ode): standard
  Seborg/Bequette CSTR. dCa/dt = q/V(Caf-Ca) - k0 exp(-E/RT) Ca; dT/dt = q/V(Ti-T)
  + (-dH)/(rho C) rA + UA/(rho C V)(Tc-T). q/V=1 /min, k0=7.2e10, E/R=8750,
  -dH/(rho C)=209.2, UA/(rho C V)=2.092 /min.
- pcgym/pcgym.py (reset/step): disturbance value at index t is used during step t; casadi path.
- pcgym/integrator.py: casadi cvodes integration over dt (accurate ODE solution).

## Hand analysis from the source
- Nominal steady state Ca=0.88, Caf=1, Ti=350: rA=0.12, T=324.2 K, Tc=299.9 K.
- Worst corner Ca=0.86, Caf=1.02, Ti=348.5 needs Tc~302.1 K: slightly beyond the actuator limit,
  so some scenarios may have unavoidable offset.
- Ca responds slowly to Tc (Tc -> T -> k(T) -> Ca, relative degree 2, d ln k/dT = 0.083/K);
  setpoint steps of up to 0.04 mol/L will saturate the actuator for several samples.
  Transitions + initial mismatch should dominate the cost; Caf shifts of 0.02 move Ca by ~0.018,
  so offset-free (disturbance-estimating) control is required.

## Experiments
### E1 - first EKF+NMPC controller, model check (batches b000-b004)
- Offline (my own re-implementation mysim.py, 20 of my own scenarios): mean 0.264, median 0.162, max 1.53, 0 runaways.
- Command: `python plant_trial.py --run wb5 --privileged runs/wb5/work/c_v1.py --n 5 --label v1`
- Why: confirm the interface, timing, and that my model reproduces the plant.
- Result: costs 0.116, 0.168, 0.114, 0.284, 0.069 (mean 0.150); max T 328.3 K.
- Model check (modelcheck.py): one-step prediction from true state with logged Tc, Ti, Caf
  matches true next state to |dCa|<5e-6, |dT|<5e-4 K on all 595 transitions. The plant equals
  the source model; CSV row k holds obs at k, applied Tc, and true state AFTER step k.
  => Offline design on my own simulator is trustworthy; spend pilot batches on validation.

### Offline design work (no plant batches)
- Worst offline seed (cost 1.53) traced: Tc stuck at 300.44 for 20 samples while Ca was 0.026 above
  setpoint. Cause: projected Gauss-Newton with clipping failed its line search (ill-conditioned
  J'J, all clipped steps worse). Fix: solve each GN subproblem as a box-constrained QP by
  coordinate descent. Offline 20 seeds: mean 0.264 -> 0.196, max 1.53 -> 0.47.
- Oracle test (oracle.py: same MPC fed the TRUE state and disturbances), 10 seeds: oracle 0.148 vs
  EKF 0.174. Longer horizon (H=20) or more GN iterations change nothing (0.148), so the MPC is
  converged; remaining cost is intrinsic (actuator-limited setpoint transitions and start-up,
  unknown setpoint-change times). Estimation costs only ~0.026.
- EKF process-noise sweep (20 seeds each): defaults 0.196; q_Ti 0.005/q_Caf 1e-6 0.195;
  q_Ti 0.08/q_Caf 1.6e-5 0.207; q_Ti 0.02/q_Caf 1e-6 0.193; q_Ti 0.005/q_Caf 4e-6 0.196;
  q_T 1e-4/q_ca 1e-8 0.196. All within noise -> keep defaults.
- Corner stress test (stress.py, 16 extreme cases: Ti 348.5/351.5, Caf 0.98/1.02, setpoints
  0.86<->0.90, hot/cold start): 0 runaways, Tmax 330.3 K (5 K margin), costs 0.37-2.36
  (worst: Caf 1.02, Ti 348.5, sp 0.86 needs Tc>302 -> unavoidable offset).

### E2 - validation of final controller.py (batches b005-b064)
- Command: `python plant_trial.py --run wb5 --privileged runs/wb5/work/controller.py --n 60 --label val`
- Why: independent validation on unseen plant scenarios (no tuning was done on plant batches).
- Result: 60 batches, mean 0.232 (se 0.028), median 0.157, p90 0.498, max 0.942; 0 runaways,
  max T 328.8 K; no controller errors.
- Worst batch b011_val (0.942), analysed with worst.py: 68% of its cost is the first 10 samples
  (start-up Ca 0.888 vs setpoint 0.861, Tc already saturated at 302 K - intrinsic); the rest
  are transients after unmeasured Caf steps (1.016->0.981, 0.981->1.019), which the EKF
  detects within a few samples and then removes offset.

### E3 - second validation (batches b065-b104)
- Command: `python plant_trial.py --run wb5 --privileged runs/wb5/work/controller.py --n 40 --label val2`
- Why: tighten the estimate of mean cost.
- Result: 40 batches, mean 0.219 (se 0.033), median 0.152, max 0.942, 0 runaways, max T 329.3 K.
- Combined 100 validation batches: mean 0.227 (se 0.021), median 0.153, p90 0.494,
  max 0.942, 0 runaways, max T 329.3 K.

## Budget
105 of 300 batches used (5 model check, 100 validation). Remaining batches not needed: the
plant matches the source model to integrator tolerance, so design was done on my own simulator.

## Open uncertainties
- Offline the MPC with perfect state knowledge scores ~0.148 vs 0.174 with the EKF; a
  step-detecting disturbance estimator might recover part of that ~0.03 gap.
- No anticipation of setpoint changes is possible (their timing/value is unknown).
- Scenarios needing Tc > 302 K (low setpoint, rich cold feed) have unavoidable offset.
- Compute: ~25-50 ms per sample on this machine; fine for a 13 s sample.
