# Lab notebook - run bb2

## Files read
- PLANT_MANUAL.md (root): 144 h batch, 7200 steps of 72 s; harvest every 12 h of mean requested fraction (0-0.5); stir 50-200, light 0-2000; sensors turbidity (~250 NTU/OD fresh, reads low up to 30% with clumping, saturates 1000), pH, pump_L, conductivity, temp_c, lux, t. Thermostat 35 C with limited heat removal. Target OD ~0.75 (=225 mg/L). Inoculum 30-5000 (mostly 100-400). Objective: total harvested biomass; never lose a culture.

## Experiments
Analysis scripts (all read only runs/bb2/trials/): analyze.py (per-batch DW/harvest summary), an_cycle.py / an_reg.py (block slopes of log turbidity vs actuator level; regression with batch fixed effects + time trend), an_turb.py / fit_turb.py (turbidity vs lab DW calibration).
Files read under runs/bb2/: trials/results.jsonl, trials/b*.csv (hourly logs), budget.json (listed only).

### E01 - open loop, stir 100, light 400, no harvest (4 batches, 4 used)
`python plant_trial.py --run bb2 runs/bb2/work/probe_const.py --n 4 --label e01_const_s100_l400_h0`
Why: baseline growth curve, sensor ranges, output format.
Result: no losses. DW at 12 h ~0.75 x inoculum (mg/L per cell-equivalent). Growth slow: mu(12-60h) 0.012-0.021 /h. Inoc 326 -> 624 mg/L at 132 h; inoc 33 -> 208.
Learned: log format (hourly csv; results.jsonl has 11 lab assays at 12..132 h -- there is NO harvest at 144 h, so biomass left at the end is wasted). Temp 34-36.9 C, pH ~9.9, conductivity ~28-30k, all fairly uninformative.

### E02/E03 - light 800 (3) and 1400 (3), stir 100, no harvest (10 used)
Why: light response. Result: 1400 clearly faster: mu(12-60h) 0.020-0.024 vs 0.013-0.021 at 400. Inoc 4222 at 800: DW flat ~2350 all batch (carrying capacity/light limit). Inoc 893 at 800: only 582->1034 (light limited at high density).

### E04/E05 - light 2000 (3); light 1400 + stir 180 (3) (16 used)
Result: light 2000 at low density mu 0.013-0.018 (worse than 1400: mild photo-inhibition, ~+1 C heating). Stir 180: mu 0.016-0.017, no benefit.

### E06 - within-batch light cycling 400/1600/800/2000/1200 in 4 h blocks, n=4 (20 used)
Why: remove strain-to-strain variance from light comparison. Result (regression, batch FE): vs 400: 800 +0.71, 1200 +0.62, 1600 +0.74, 2000 +0.50 %/h (se 0.39). Plateau from ~800; 2000 slightly worse at low density. Light 2000 raises broth temp ~+1 C vs 400.

### E07 - light 1200, constant harvest 0.25 (3) (23 used)
Why: learn harvest mechanics / volume. Result: harvested_mg / (frac x DW) ~ 20 -> working volume ~20 L (pump_L +5 L per harvest). Per-harvest weighing noise ~+-15%. With 0.25 every 12 h cultures slowly declined (needs mu > 0.024/h); totals 4.9k, 7.3k (inoc 162,187), 21k (inoc 942).
Learned: steady harvest at 25% is not sustainable for these strains; biomass value ~ 20 L x frac x DW.

### E08/E09 - inoculum 1500 at light 1200 (2) vs 2000 (2) (27 used)
Result: productivity at DW 1000-1500: 1200 light ~4.7 mg/L/h; 2000 light 6.1-7.7 mg/L/h. At high density more light pays (self-shading); low density prefers ~1000-1400.

### Turbidity calibration (fit_turb.py on all pre-harvest points)
NTU ~ 700*(1-exp(-DW/780))*(1-0.14*hour/144), rms log error 0.11. Turbidity is strongly sub-linear: ~0.78 NTU/(mg/L) below 200 mg/L, ~0.5 at 1000, ~0.3 above 1500 (it effectively saturates ~500-550 NTU, far below the nominal 1000 limit). Above ~1200 mg/L turbidity is nearly uninformative.

### E10 - within-batch stir cycling 100/50/150/200, light 1600, inoc 500, n=3 (30 used)
Result: turbidity itself responds to stir: +3.3 (150) and +4.4 %/h (200) vs 50 within a block, i.e. stir de-clumps filaments and changes the reading with a multi-hour time constant. DW grew smoothly (369->1296). Learned: stir is a SENSOR disturbance; keep stir constant so turbidity calibration holds.

### E11 - light 1400, inoc 300, stir 60/100/160, n=3 each (39 used)
Result: DW at 132 h: s60 1025/1124/818, s100 963/1112/982, s160 1095/814/1044. No detectable growth effect of stir. Decision: stir fixed at 100.

### E12 - inoculum 30 at light 2000, no harvest (2) (41 used)
Why: is a tiny culture at full light at risk (photo-inhibition/heat)? Result: survived; mu 0.018 / 0.0135 (low vs ~0.023-0.025 for tiny cultures at 800-1000). Tank ran 40-41 C at 2000 in these batches.

### E13 - inoculum 100, harvest 0.5 every 12 h, light 1200 (2) (43 used)
Why: find the loss mechanism/threshold. Result: both LOST at 84 h, when post-harvest DW fell to ~4-5 mg/L (DW before harvest 9 and 8). Harvested mg still counted (2.3k). Learned: extinction = washout below ~5 mg/L. Rule: never harvest below 40 mg/L estimated.

### Model work (an_prod.py, an_temp.py, an_mu.py, dp_plan.py, pol_eval.py)
- Productivity (mg/L/h) peaks at DW ~800-1200 at 1400-2000 light (~8-9), vs ~3-5 at 400-800 light. Carrying capacity ~2400 at 800 light.
- Temperature: IMPORTANT - at 2000 light many batches heat to 38-41 C (thermostat saturates; ambient varies batch to batch, start temps 32.9-38.7 C). At 1400 temps stay ~33.5-36.5.
- Offline DP on a fitted growth model: optimal policy = grow with no harvest, hold at ~850 mg/L if reached, at 96 h harvest down to ~500, then 50% at 108/120/132 h. Late dumping (only 120/132) loses 5-10%; dumping from 96 h loses up to 15% on small cultures. Hold level 700-1000 makes <2% difference.

### E14 - controller v1 (ctl_v1.py), random inocula (6) (49 used)
v1: stir 100; X from turbidity calibration; light 1000->2000 ramp over X 150-600; hold X_HOLD 850; k=8 (96 h) -> 500; 108/120/132 h: 50%; floor 40 mg/L.
Result: inoc 112: 9.1k, 167: 11.8k, 606: 21.6k, 325: 23.4k, 32: 5.1k, 2400: 42.9k. No losses. Estimator vs lab DW within ~10% below 600 mg/L, but under-reads by 15-25% at high density late in the batch.

### E15 - light A/B at held density (X_HOLD 700, light 2000 vs 1400 alternating 24 h), inoc 1000 (4) (53 used)
Result: production per 12 h interval (DW_{k+1}-DW_k(1-f)): 2000 -> 111 mg/L, 1400 -> 80 mg/L (se ~15 each) at X ~750, even though 2000 ran 36.5-38.7 C vs 33-36.6. Full light pays at high density.

### Calibration refit (fit_turb2.py, fit_turb3.py; stir-100 batches only)
ln NTU = ln(996(1-exp(-X/1230))) - 0.056 t/100 - 0.182 (t/100)(X/1000), rms 0.080 (vs 0.11-0.14 for the fixed clumping model). Clumping grows with age x density. Model peaks in X, so inversion searches only up to the peak.

### E16 - controller v2 (ctl_v2.py: new calibration + temperature governor >40 C), random inocula (8) (61 used)
Result: 164: 10.0k, 211: 13.6k, 228: 21.5k, 2109: 46.5k, 109: 8.9k, 112: 18.7k, 182: 14.6k, 1590: 38.1k. No losses. Big strain-to-strain spread (inoc 109 vs 112: 8.9k vs 18.7k).

### E17 - low-density light test: v2 harvest logic, light cycled 700/1300/2000 in 24 h blocks, 3 phases x 2, inoc 150 (6) (67 used)
Result (interval mu from DW, batch FE, lnX covariate, hours<96): 1300 vs 700 +0.0036/h (se 0.0023), 2000 vs 700 +0.0024 (se 0.0023). 2000 ran ~2.3 C hotter. Learned: at low density 1300 ~ 2000 > 700.

### Growth regression on all intervals (an_dil.py, n=650 intervals, 60+ batches, batch FE)
mu = FE - 0.0517x + 0.0057x^2 - 0.0013L + 0.0128 L x - 0.0072 f + ... (x g/L, L in 1000 umol). No growth boost from fresh medium after harvest. Strong density dependence at low x.
DP with this model (dp2.py, pol_eval2.py): optimum hold rises with strain speed (500 slow, 650 typical, 1000 fast); fixed 650-850 is within ~4% everywhere. End game unchanged.

### Controller v3 (ctl_v3.py)
Changes from v2: light vs X piecewise 1000 (<60) -> 1300 (150) -> 2000 (>=500); X_HOLD 750.

### E18 - controller v3, random inocula (8) (75 used)
Result: inoc 103: 9.8k, 239: 16.1k, 323: 13.8k, 338: 20.2k, 341: 16.8k, 172: 16.3k, 356: 17.2k, 145: 10.6k. No losses. Median ~16k.

### E19 - v3 at inoculum 5000 (3) and 30 (3) (81 used)
Result: 5000: 72.6k, 66.6k, 69.8k (first two harvests take 50% from ~2800 mg/L). 30: 4.5k, 8.9k, 6.8k. No losses. v3 estimator under-reads by 20-30% at high density after a heavy-density history (5000 batches held ~1000 true instead of 750).

### Estimator v4 (est_cmp.py, an_est2.py)
Clumping modelled from the running integral of the controller's own X estimate: NTU = 973(1-exp(-X/1188)) exp(-0.082 t/100 - 0.196 I/1e5). Closed-form inverse, monotone. Replay vs lab DW (rms log error): e14 0.122->0.090, e15 0.124->0.111, e16 0.122->0.083, e17 0.069->0.061, e18 0.091->0.079, e19 0.106. -> ctl_v4.py (v3 + this estimator).

### E20 - end-game timing, inoc 200, n=8 per arm (24) (105 used)
Arms: mid = v4 (96 h -> 500, then 50% x3); late = END_K 9 (108 h -> 500, then 50% at 120/132); early = END_K 7 (84 h -> 500, then 50% x4).
Analysis (ancova.py): ln(total) ~ arm + ln(DW at 72 h) (arms identical until 72 h, so DW72 removes strain variance; resid sd 7%).
Result: raw means mid 14.6k, late 16.6k, early 13.9k; adjusted vs mid: late +2.8% (se 3.8), early +1.0% (se 3.6). Flat optimum.

### E21 - hold level, inoc 1000, n=6 per arm (18) (123 used)
Arms: X_HOLD 750 (v4), 550, 1000. Medians 29.3k / 27.9k / 31.2k; adj (DW12 covariate, weak) 550 -8.1% (se 8.1), 1000 +1.7% (se 8.5). Low hold looks worse; 750-1000 equal within noise. Decision: X_HOLD 850.

### E22 - end-game replicate, inoc 120, mid vs late n=8 each (16) (139 used)
Adjusted late vs mid: -2.7% (se 3.0). Combined with E20: no difference (~ -0.5 +- 2.4%). Keep mid (DP model favours it for dense cultures).

### Final controller: controller.py = ctl_v4.py with X_HOLD 850.

### VALIDATION - final controller.py (74 batches; 213 used)
Commands (from root): `python plant_trial.py --run bb2 runs/bb2/work/controller.py --n N --inoculum I --label val_iI` for I:N = 30:6 60:6 100:8 150:8 250:8 400:8 1000:6 2500:4 5000:4, then `--n 16 --label val_rand` (random inocula). Summary via val_summary.py:
```
inoculum       n   median      p25      min lost
30-49          7     5576     3823     1159    0
50-89          9     9570     7156     3149    0
90-129         9    12363    11421     9793    0
130-199       12    13249    11955    10736    0
200-329       14    18630    15988     9150    0
330-599        8    22286    20433    17643    0
600-1499       6    29844    27981    25328    0
1500-3499      5    48260    44694    41380    0
>=3500         4    68589    66800    65345    0
ALL val        n=74 median 15945 p25 10756 lost 0 errors 0
random draws   n=16 median 13400 p25 10622 lost 0 errors 0
inoc 100-400   n=43 median 16013 p25 12633 lost 0 errors 0
```
Worst batch b139 (inoc 30, very slow strain, mu ~0.012/h): 1.2k; the 40 mg/L floor limited the last harvests, culture kept alive. No errors, no losses in any of the 213 batches except the deliberate washout test E13.
Remaining budget (87 batches) deliberately left unused: the model and the A/B tests show the policy sits on a flat optimum; further tuning would be below the batch-to-batch noise.
