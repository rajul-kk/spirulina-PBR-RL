# wb4 lab notebook

Run name: wb4. Budget: 300 pilot batches. All trials run with `--privileged`.

## Files read
- blackbox/PLANT_MANUAL.md (operator's manual).
- blackbox/plant_trial.py (the trial tool; imports harness, which I did not read).
- PPO_IBM/environments/genetic_env.py (the full simulator source, lines 1-1403). It imports only
  os/sys/gymnasium/numpy/typing, so there are no PPO_IBM/training modules to read.
- runs/wb4/ (empty when I started).

## What the source says (design-model notes)
- Plant difficulty is 2: full thermal physics, O2 inhibition, sensor drift and lag, +-5% actuator noise, and strain micro-drift.
- Harvest events happen at step_count 600, 1200, ... 6600: **11 events** (hours 12-132). Nothing is
  harvested at 144 h, so all standing biomass at the end is lost. The applied fraction is the
  mean of harvest_frac over the steps since the previous event (the first event averages 601 samples).
- Harvest removes a random Bernoulli fraction of cells (agents), and the refill is fresh medium.
- Inoculum: each agent is about 12.5 mg dry weight (1.25e8 mass units x 1e-7 mg; 1.1e8 at 5000 cells).
  So X0 = 0.625 mg/L per cell: 30 cells gives 19 mg/L (OD 0.06), 200 gives 125 mg/L, and 5000 gives
  2750 mg/L (OD 9, beyond turbidity saturation).
- Extinction happens when there are fewer than 10 agents or less than 1 mg total. Healthy lysis is
  5e-4/h, rising to 2.5e-3/h when mu is below respiration (darkness). Cells lighter than 1e7 die.
- Growth is mu_max (N(0.040, 0.006)) x f_I x f_Q x f_P x f_C x tox x T x shock x O2 x pH x osm x repair x fatigue.
  - Light: Haldane on red light (Ks about 100, Ki about 2500), over an 8-point profile across the
    6.7 cm path. Extinction is 0.2/0.25/0.06 m^-1 per mg/L for red/blue/green, plus 0.004*rpm from scatter.
    Flashing-light integration weight is 0.5 at 50 rpm and 1.0 at 200 rpm.
  - Shock: exp(-3e-6*diff^2), where diff is the path-mean light above the acclimated level
    (acclimation tau is 1-4 h). Increases in light must be ramped.
  - Repair tax: 1-0.35*sigmoid(0.12*(rpm-100)), which is about 3% at 80 rpm and 17.5% at 100 rpm.
    The fatigue tax applies above 80 rpm (at most 15%).
  - Temperature: CTMI with T_opt N(36,1) and Tmax 44.5. The thermostat holds 35 C (the chiller can
    remove at most 0.6 C/h). Light heats at I*0.001 C/h. The tank stays at 35 C up to about
    1500 umol and reaches about 39 C at 2000 umol.
  - Clumping: sticking at od*0.05/h*(1-rpm/250), with shear breakup only above 80 rpm. Clumps shade
    (mass^-1/3) and make turbidity read low by the same factor. Daughter cells start unclumped.
  - O2 inhibition: 1/(1+(DO/35)^4). kLa rises with stir.
- Turbidity is 250*od*pigment*clump^-1/3/(1+0.05*od)*(1 +- 3%*rpm/200 noise), with 2% jitter and a
  per-batch drift of 0.95-1.05, clipped at 1000.
- Nutrients (N/P dosing) and pH (CO2 pH-stat) are automatic and not limiting for productivity at 10+ mg/L/h.

## Design studies with my own reduced model (model.py, steady.py, design.py)
I wrote a mean-field ODE model by hand from the source; it does not import the simulator.
Steady-state productivity P = mu*X (mg/L/h) keeps rising with density up to X of about 600-1300 mg/L
if light is high (1600-1800) and clumps are controlled by stirring. Examples: X=225 gives 6.5,
X=400 gives 9.0, X=600 gives 9.9 (rpm 80), and X=900-1300 gives 10.7-11.6 (at rpm 150, where clumps
break up). The optimal light is about 600 at low density and 1700-1800 at OD >= 0.75.
So the plant team's OD 0.75 target looks too low.

## Experiments
### E1: baseline v1, 6 batches, natural inoculum mix (b000-b005)
`python plant_trial.py --run wb4 --privileged runs/wb4/work/ctrl_v1.py --n 6 --label v1`
- Why: shake-down run. Checks the sensors against the truth and gives a first look at growth rates.
- v1 design: stir 70/80/140 rpm by estimated density; light 500+5*X, ramped at 120 umol/h up
  to 1700. Harvests down to X=600 mg/L, with 0.5 at the last 2 events.
- Result (inoculum: mg): 4107: 64564, 48: 10369, 145: 12700, 249: 17337, 128: 12504, 295: 22091. No losses.
- Learned:
  - Turbidity/(250*true_od) falls from 1.0 to 0.5-0.6 over days. This is filament clumping
    (c of about 3-6 at 80 rpm) plus the 1/(1+0.05od) saturation. v1 underestimated density, so it
    harvested late (first harvests came at 60-110 h).
  - Low-density growth rate is about 0.031/h. At X of about 400-550 the culture makes about
    7.9 mg/L/h. True temperature holds 35-36 C at 1700 umol, as the source predicts.
  - Offline replay (clumpfit.py) of the source's clump equations on the logged stir and true OD,
    with births resetting clumps, gives predicted/observed turbidity of 0.95-1.03 (the
    per-batch drift). So a physics-based clump observer works. A turbidity-only replay
    (obsfit.py) tracks true X to within about 0.85-1.03.
- Reduced-model check: the model reproduces the b006 no-harvest trajectory within 5-10%
  (model 176/247/337/441/557 mg/L at 12/24/36/48/60 h; truth 173/237/312/401/496).

### DP study (dp.py): harvest schedule in the reduced model
DP over the 11 event fractions, with productivity P(X) from the model (calibrated by 0.93) at
80 rpm. The optimum holds pre-harvest X at about 600 (post-harvest about 560) for events 1-7,
then about 470 at event 8 and 250-300 at event 9, with 0.5 at events 10-11. Model totals:
X0=19: 8.2 g; 62: 15.0 g; 125: 19.3 g; 250: 23.8 g; 1000: 39.3 g; 2750: 72.2 g.

### E2: v2 (DP-shaped schedule, Xt=560, late multipliers 0.85/0.45, n_end=2), 8 batches at inoculum 200 (b006-b013)
`python plant_trial.py --run wb4 --privileged runs/wb4/work/ctrl_v2.py --n 8 --inoculum 200 --label v2`
- Result (mg): 18238, 23195, 20040, 18076, 21682, 9347, 20910, 18032. Median 19.1 g, no losses.
- b011 was a slow strain (mu of about 0.015/h). It never reached Xt, so it was harvested only at events 9-11 and gave 9.3 g.
- The model's DP estimate at X0=125 was 19.3 g, which matches.

### E3: v2 at the extremes, inoculum 30 (n=4, b014-b017) and 5000 (n=4, b018-b021)
`plant_trial.py ... ctrl_v2.py --n 4 --inoculum 30 --label v2_i30` (and the same with 5000)
- Why: safety check at both ends of the inoculum range before tuning.
- Inoculum 30: 5250, 4725, 4008, 8379 mg. Harvested only at the endgame; no losses.
- Inoculum 5000: 74862, 71442, 71878, 67475 mg. The saturated nephelometer gives F=0.5 at events 1-2,
  then the culture settles to the target. No losses.

### Reduced-model sensitivity sweep (design2.py; 12 cases: X0 19/125/250/1500 x mu_max 0.028/0.04/0.05)
Mean model harvest (g) by arm:
- base 26.48; lo55 26.49; Xt700 26.51; ramp250 26.60; late1 26.45; late0 26.42; nend3 26.48; Islope8 26.38; I2000 26.23.
- Xt850 26.29; Xt420 25.97; mid90 25.96; I1500 25.61; hi100 25.57; hi140 25.04; hi120 24.93.
- Conclusion: a plateau. Stirring above 80 rpm in the working range costs 3-6% (repair tax outweighs the
  de-clumping). Light 1700 is at the optimum. Harvest details matter less than 2%.

### E4: v2 at inoculum 100, n=6 (b022-b027)
- 7815, 6670, 6693, 10362, 13227, 15854 mg. Strain spread dominates; three slow strains never reached Xt before the endgame.

### Observer check (implied.py)
The controller's density estimate, inferred from interior harvest decisions, compared with the lab
assay: median estimate/true is 0.95 (n=31). It drifts to 0.8-0.9 by events 7-8, so the
controller under-reads dense cultures by 5-15%. The offline replay with the same kinetics
(obsfit2.py) is unbiased (0.97-1.00), so the online growth-rate term is the likely cause. It is
harmless here because it just shifts the effective target up.

### E5: target-density A/B/C at inoculum 250, n=10 each (b028-b057)
`ctrl_v2_xt400.py` / `ctrl_v2.py` (560) / `ctrl_v2_xt800.py`, each with `--n 10 --inoculum 250`
- Xt400: mean 18.5 g, median 18.7, min 15.0.
- Xt560: mean 19.7 g, median 19.3, min 16.1.
- Xt800: mean 20.2 g, median 19.9, min 15.8.
- A higher standing density helps a little, as the model predicted. No losses.

### Empirical productivity (prod.py): dX/dt from the privileged true OD, excluding harvest windows, all batches
| X (mg/L) | 0-50 | 50-100 | 100-150 | 150-225 | 225-300 | 300-400 | 400-500 | 500-600 | 600-700 | 700-800 | 800-1000 | 1000-1300 | 1300-2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P (mg/L/h) | 0.80 | 1.81 | 3.05 | 4.52 | 5.88 | 7.01 | 7.86 | 8.31 | 8.78 | 9.25 | 9.33 | 9.14 | 7.51 |
P plateaus at X of about 700-1100 (OD 2.3-3.7), well above the plant team's OD 0.75 (where P is only about 6).
The low-density specific growth rate is 0.024-0.032/h.

### DP on the empirical curve (dp_emp.py)
The optimum keeps pre-harvest X below 1000 (post-harvest about 850) for events 1-7, then draws down at
events 8-9, with 0.5 at events 10-11. Rule vs DP (g), by X0:
- X0=156: DP 19.6; Xt400 18.5; Xt560 19.3; Xt800 19.6.
- X0=250: DP 23.3; Xt800 23.2.
- X0=19: all rules give 6.9.
The Xt800 rule is within 0.1 g of the DP everywhere. The empirical model at X0=156 (inoculum 250)
predicts 19.6 g, and the plant gave a mean of 20.2 (Xt800). So the empirical model is calibrated.
A light-ramp study (ramp_study.py) found the ramp rate and the low-density light law flat within 1%.

### E6: final controller.py (v3: Xt=800, high stir only above 1100 mg/L), inoculum 250, n=10 (b058-b067)
`python plant_trial.py --run wb4 --privileged runs/wb4/work/controller.py --n 10 --inoculum 250 --label v3`
- 16615, 17653, 26151, 24302, 20618, 23809, 19992, 19541, 22476, 18392 mg. Mean 20.95, median 20.3, no losses.

### E7: validation of v3 on 60 natural-mix batches (b068-b127)
`python plant_trial.py --run wb4 --privileged runs/wb4/work/controller.py --n 60 --label val`
- Median 18.9 g, 25th percentile 15.8 g, mean 20.6 g, minimum 7.6 g. **0 lost.**

### E8: fixed-inoculum validation (b128-b153)
Same controller, `--inoculum 30 / 80 / 1000 / 5000`, with n = 8 / 6 / 6 / 6.
| inoculum | median (g) | p25 (g) | min (g) | lost |
|---|---|---|---|---|
| 30 | 9.0 | 5.7 | 2.2 | 0 |
| 80 | 13.7 | 11.4 | 8.1 | 0 |
| 1000 | 31.5 | 27.7 | 24.9 | 0 |
| 5000 | 71.9 | 69.7 | 69.2 | 0 |
- The worst batch was b135 (inoculum 30), a very slow strain (mu of about 0.015/h). It ended with 107 agents,
  still 10x above the extinction floor.
- Check: the thermal back-off (measured temperature above 38 C) fired on noise and sensor drift. 227 of 12384 hourly
  rows read above 38 C, while the true temperature never exceeded 37.96 C (that maximum is the initial ambient).

### Change to the final controller (thermal guard only)
The temperature used for the back-off is now a 1 h EMA with a 39.5 C threshold. The back-off can no
longer push the light below 400 umol, because darkness doubles respiration and raises lysis; a
simulated stuck reading of 41 C now leaves the light at 400. v3 is kept as ctrl_v3_frozen.py.

### E9: final controller.py, 20 natural-mix batches (b154-b173)
`python plant_trial.py --run wb4 --privileged runs/wb4/work/controller.py --n 20 --label val2`
- Median 22.0 g, p25 14.9 g, mean 27.0 g, minimum 10.3 g. 0 lost.

## Validation summary (controller.py: v3 plus the final thermal-guard version, 116 batches)
| group | n | median (g) | p25 (g) | mean (g) | min (g) | lost |
|---|---|---|---|---|---|---|
| natural mix (val + val2) | 80 | 19.1 | 15.5 | 22.2 | 7.6 | 0 |
| inoculum 30-80 | 21 | 10.0 | 8.5 | 10.6 | 2.2 | 0 |
| inoculum 100-400 | 69 | 19.1 | 15.9 | 19.0 | 7.6 | 0 |
| inoculum 600-5000 | 26 | 38.6 | 31.8 | 46.8 | 24.9 | 0 |

For comparison, the v1 shake-down controller (target 600, observer under-reads) gave 12.5-17.3 g for inocula 128-249.

Batches used: 174 of 300. I stopped there because the design sits on a plateau (every model
and pilot lever moves the result by less than about 3%). The remaining budget is left unspent.

## Open uncertainties
- The strain growth rate dominates the spread (for example, 6.7-15.9 g at a fixed inoculum of 100). The controller does not adapt
  the harvest plan to a slow strain beyond its density feedback. The DP says an earlier endgame helps slow strains, but by under 1%.
- The density observer under-reads dense cultures by 5-15% late in the batch. This only raises the effective target (the
  optimum is broad), but the endgame fractions carry that error.
- The 1000-2000 mg/L band has few samples. High stir (140 rpm) above 1100 mg/L relies on the reduced model.
- Light above 1700 and T_opt-seeking heating were not tested in the plant; the model predicts at most about 1%.
