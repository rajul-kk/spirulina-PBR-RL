# wb5 lab notebook (white-box writer, privileged logs)

## Files read
- PLANT_MANUAL.md (root)
- plant_trial.py (root): batch = 7200 steps, harvest events at t=600..6600 (11 events, 12..132 h), assay just before each event; inoculum sampler: 10% 30-80, 20% 600-5000 (log-uniform), 70% 100-400.
- PPO_IBM/environments/genetic_env.py (whole file; imports only os/sys/gymnasium/numpy, nothing from training/).

## What the source says (design model)
- Strain: mu_max ~ N(0.040, 0.006) (floor 0.02), Ks_light ~100, Kii ~2500, T_opt ~ N(36,1), tau_acclim 1-4 h.
- Each agent ~12.4 mg dry wt -> inoculum N cells gives OD ~ N*0.00207 (200 cells = OD 0.41; 30 = 0.06; 5000 = OD ~9).
- Growth = mu_max * f_I * f_Q * f_P * f_carbon * tox * f_T * shock * f_O2 * f_pH * osmo * repair_tax * fatigue.
  - Light: 6.7 cm path, extinction 0.2/0.25/0.06 per mg/L (R/B/G); Haldane on red light with total-light inhibition; response partly integrated over path (w=0.5 at 50 rpm -> 1.0 at 200 rpm). Dense culture + high surface light is most productive.
  - repair_tax = 1-0.35*sigmoid(0.12*(rpm-100)): ~1% at 60 rpm, 8% at 90, 17% at 100, 32% at 120, 35% at 200.
  - Membrane fatigue above 80 rpm, max 15%, slow.
  - Photo-shock only for light ABOVE the acclimated (path-mean) level: exp(-3e-6 diff^2).
  - Temperature: thermostat holds 35 C until light heat exceeds chiller (0.6 C/h): T_eq ~ 25+10*(0.001*I-0.6) above ~1600 umol -> 1700 gives ~36, 2000 gives ~39.
  - Flocculation: clump mass grows at od*0.05*max(0.1,1-rpm/250) per h; shear breakup only above 80 rpm ((rpm-80)/120)^2; clumps self-shade (mass^-1/3) and lower the turbidity reading by the same factor.
  - Respiration tiny (1% of mu_max; 2x in dark); lysis 0.05%/h baseline. Extinction = <10 agents or <1 mg. Only over-harvesting of a tiny culture can realistically kill it.
  - Harvest = mean of requested frac over the 12 h interval (clipped 0-0.5); removes random agents; refills fresh Zarrouk.
  - Turbidity = 250*od*(0.7+0.3*pigment)*clump^-1/3/(1+0.05 od)*drift(0.95-1.05)*noise, clipped 1000. Temp sensor has +-5% multiplicative drift and lag.
  - Nutrients auto-dosed and never limiting in normal runs; pH held ~10 by CO2 pH-stat.
- Objective only counts harvested mg; standing biomass at the end (after the 132 h event) is wasted.

## Experiments
### E1 (b000-b003) probe.py stir 90, inoc 200, n=4
`python plant_trial.py --run wb5 --privileged runs/wb5/work/probe.py --n 4 --inoculum 200 --label p90`
Why: calibrate growth, temperature, turbidity bias. Probe = light ramped with OD estimate (600 -> 1700 umol, +100/h), no harvest until 108 h, then 0.5 (so harvests at 120, 132 h).
Result: 13.4, 17.9, 17.4, 17.7 g. Thin-culture specific growth 0.022-0.029/h; above OD ~1.8 growth is ~linear at ~0.033 OD/h (10 mg/L/h). Temp at 1700 umol: 36.1 C (as predicted). Turbidity/true-OD ratio falls from ~1.0 to ~0.6 by 108 h (clumping) - turbidity-only OD estimate reads ~40% low late.
### E2 (b004-b007) stir 120; E3 (b008-b011) stir 60; inoc 200, n=4 each
Result: stir 120: 10.5, 9.0, 14.2, 10.2 g (growth ~30% slower: repair tax; turbidity ratio stays ~0.9-1.0, few clumps). Stir 60: 17.9, 11.4, 17.9, 16.0 g ~ same as 90.
Learned: stir 60-90 fine, >=100 costly. Mean-field clump ODE from source (clumpfit.py) reproduces the turbidity ratio decline within ~5% at 60/90 rpm.
Built mfsim.py (mean-field model from source: light, clumps, temp, O2, repair, fatigue); with mu_eff 0.031-0.034 it reproduces the probe OD trajectories within ~10%.
### E4 (b012-b019) controller.py v1, inoc 200, n=8
v1 = OD estimator (turbidity EMA + saturation inversion + internal clump model), light follows OD (600->1800), stir 80, per-event OD caps (2.4 hold, then 1.7/0.65/0/0 drawdown over 96-132 h), interval-mean steering of harvest request.
Result: 21.6, 10.6, 21.6, 19.1, 16.7, 15.1, 15.7, 17.0 g (median 16.9). Offline replay (replay.py) of the estimator on probe logs: OD estimate within ~10-20% of true (mostly low; drift and clump model residual).
### E5 (b020-b035) v1, natural inoculum mix, n=16; E6 (b036-b039) inoc 30; E7 (b040-b043) inoc 5000
No culture lost. Mix: 8.5-60.8 g. Inoc 30: 8.2, 15.4, 8.5, 9.7 g. Inoc 5000: 68-72 g (first harvest 28 g). Estimator tracks tiny cultures within 5%; at OD 9 turbidity saturates -> estimator clamps to 6, harvest 0.5 correctly.
### Digital twin (twin.py)
Closed-loop twin: runs controller file against mfsim physics with synthetic turbidity. Sensitivity (tune.py, weighted inoculum/strain scenarios, base 22.5 g): stir 60 -2.3%, 90 -3%, 70 ~0; light cap 1700 -1%, 1900 ~0, 2000 -2%; caps variants within +-0.7% (flat optimum); stir bursts to 200 rpm for 2 h every 24 h +2.1% (break clumps).
### E8 (b044-b051) cb.py = v1 + burst (24 h, 2 h, 200 rpm), inoc 200, n=8
Result: 20.2, 21.4, 17.7, 16.2, 13.6, 10.8, 22.6, 23.4 g.
Analysis (cov.py): harvest regressed on strain early growth mu0 (log-OD slope 6-24 h, before any burst) removes most strain variance: resid sd 0.73 g (raw sd ~4 g). Burst effect +0.68 +- 0.37 g (+4%), consistent with twin. Adopted.
### E9 (b052-b059) cb_late.py (drawdown only from 108 h: caps ...,2.4,2.4,1.0,0,0) and E10 (b060-b067) cb_early.py (drawdown from 84 h: ...,1.5,1.0,0.5,0,0); inoc 200, n=8 each
Result late: 17.0, 19.6, 14.7, 20.4, 18.5, 17.0, 16.0, 20.7 g. Early: 14.5, 17.6, 17.4, 16.4, 14.0, 19.2, 17.7, 17.7 g.
Covariate-adjusted vs cb: late -0.86 +- 0.36 g, early -0.87 +- 0.36 g. The cb drawdown schedule (start at the 96 h event) is at the optimum; the plant's optimum is sharper than the twin's.
Decision: final controller = cb.py (v1 + burst) -> controller.py. Large-inoculum tuning not pursued: median and p25 are set by 30-400 cell batches.
### E11 validation (b068-b163) final controller.py
Commands: `--n 80 --label val` (natural inoculum mix), `--n 6 --inoculum 30 --label val30`, `--n 6 --inoculum 80 --label val80`, `--n 4 --inoculum 5000 --label val5000`. Summary by valsum.py.
| set | n | median g | p25 g | min g | lost |
|---|---|---|---|---|---|
| natural mix, all | 80 | 20.7 | 15.4 | 7.1 | 0 |
| mix, inoc 30-99 | 7 | 8.4 | 7.9 | 7.1 | 0 |
| mix, inoc 100-400 | 55 | 19.8 | 15.2 | 7.7 | 0 |
| mix, inoc 401-5000 | 18 | 34.6 | 33.0 | 19.6 | 0 |
| inoc 30 | 6 | 6.7 | 5.6 | 3.0 | 0 |
| inoc 80 | 6 | 14.2 | 12.4 | 9.4 | 0 |
| inoc 5000 | 4 | 73.4 | 71.4 | 68.6 | 0 |
(v1 without burst, mix n=16: median 18.0, p25 14.7.) No controller errors in any batch. Lowest result (3.0 g, inoc 30) is a slow strain (specific growth ~0.02/h throughout); the culture just never got dense.
Budget: 164/300 used; the remaining 136 were left unused.
## Open uncertainties
- The strain's T_opt is unknown; the light cap (1800 umol -> ~37 C) is a compromise that is fine for T_opt 35-38.
- The OD estimate relies on the clump model and a nominal growth rate; turbidity drift (+-5%) and model residuals leave ~10-20% error (mostly reading low). Harvest caps were tuned with that bias included.
- Large-inoculum (>600) harvest schedule was not tuned in the plant (it does not set the median or p25).
- The burst benefit (+0.7 +- 0.4 g) is a ~2 sigma result, backed by the twin.
