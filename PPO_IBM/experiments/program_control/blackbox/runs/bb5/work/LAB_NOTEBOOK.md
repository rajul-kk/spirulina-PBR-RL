# bb5 lab notebook

## Files read
- PLANT_MANUAL.md (root): 144 h batch, 7200 steps of 0.02 h, harvest every 12 h of mean requested frac (0-0.5).
  Actuators stir 50-200, light 0-2000, harvest 0-0.5. Sensors turbidity (~250 NTU/OD, clumps read low up to 30%, sat 1000),
  ph, pump_L, conductivity, temp_c, lux, t. 1 OD = 300 mg/L. Target OD~0.75. Inoculum 30-5000 cells (mostly 100-400).
  Budget 300 batches.

## Experiments
All commands run from blackbox root as `python plant_trial.py --run bb5 runs/bb5/work/<c>.py --n N --inoculum I --label L`.
Analysis scripts (read only runs/bb5/trials/): show.py (hourly log + lab assays), labs.py (per-batch lab assay table),
mu.py (specific growth rate fits), calib.py (turbidity/lab ratio vs hour).

### E1 open-loop constant light, no harvest (b000-b012), 13 batches, inoculum 200
Why: identify light/stir response and growth curve without harvest confounds.
- c_l600_h0 (stir100, light600) x3; c_l1200_h0 x2; c_l300_h0 x2; c_l2000 x2; c_l1200_s200 x2; c_l1200_s50 x2.
- Result (lab mg/L at 132 h): L300 ~380, L600 ~560-670, L1200 ~800-870, L2000 ~960-1070. Stir 200 ~840, stir 50 744/1101.
- Inoculum 200 cells -> ~130-150 mg/L at t0 (~0.6-0.7 mg/L per cell); turbidity 98 NTU at t0.
- Specific growth rate falls with density: mu(12-60h) ~0.02/h at L2000, 0.015 at L600, 0.011 at L300; mu(60-132h) ~half.
  => light-limited by self-shading; higher light always better at X>=150 mg/L so far.
- Temperature: batch-specific offset (33.5-36.4 C at L600 same settings); L2000 adds ~+1.5-2 C (37-38).
- Only 11 harvests (12..132 h); nothing at 144 h -> standing biomass at the end is wasted.

### E2 harvest schedules (b013-b016), L2000 stir100, inoc 200
- c_h05from72: frac 0.5 requested from 72 h on: 10.6 / 11.8 g. c_h02from12 (0.2 from 12 h): 5.9 / 6.5 g, X stuck ~150 mg/L, T ~39.6 C.
- Learned: tank volume ~20 L (0.5 harvest pumps ~10 L); harvested_mg ~= frac*20 L*X. Early harvesting at low X is bad.
  Low-density culture under high light runs hotter (39-40 C).

### E3 extremes at L2000 (b017-b020)
- inoc 30 -> X0 ~20 mg/L; very slow first 24 h (lab 21,22,32 / 25,32,39) then mu~0.025; 156 / 289 at 132 h. T 38-39 C.
- inoc 5000 -> X0 ~2800-2900 mg/L, no net growth all batch (carrying capacity ~3000 mg/L); T 38-40.5, turbidity saturated ~1000 then reading ~770 as it clumps.

### E4 (b021-b026)
- inoc 30: L600 -> 455 / 269; L1200 -> 319 / 223 (vs L2000 156/289). Low density prefers moderate light (photoinhibition/heat).
  mu at X~25 under L600 ~0.03/h.
- inoc 200 L2000 stir 50: 896 / 1164 (similar or better than stir 100).
- Turbidity/lab ratio: ~0.77 at 12 h falling to ~0.59 by 120-132 h (spread 0.37-0.89 across batches).

### E5 high inocula growth (b027-b030), c_l2000_s50, no harvest
- inoc 1500 -> X0 ~1000: 1003 -> ~1790 at 132 h (P ~11 mg/L/h at 1000-1400, ~5 at 1400-1800).
- inoc 3000 -> X0 ~1850: 1864 -> 2181 / 2569. Productivity collapses above ~1800; plateau ~2900 (i5000).
- pfit.py: P(X) at L2000 ~ 3.6 (100-200), 5.3 (200-300), 6.4 (300-450), 7.5-8 (450-1000), falls >1500. mu at X<50: L600 0.044/h vs L2000 0.020/h (photoinhibition at low density).

### DP harvest planning (dp.py, dp2.py, dp3.py; model only, no batches)
- Harvest mg = frac*20 L*X. With flat P(X) over ~400-1300 mg/L, DP says: never harvest while growing below ~1000 mg/L;
  hold ~1000-1100 if reached; end-game 0.5 at 120 and 132 always, at 108 if X>~360, at 96 if X>~720, partial at 84 if X>~800.
  Simple rule (0.5 from 96/108 by threshold) within 1-3% of DP optimum for typical inocula; big inocula need holding.

### E6 controller v1 (b031-b040): density-scheduled light (600 at X<40 -> 2000 at X>200), end-game thresholds
- i200: 14.2, 13.2, 10.1, 15.1 g. i30: 10.8, 8.2, 5.8 g. Variant L_LOW=300 at i30: 5.0, 5.3, 6.3 g (worse -> keep 600+).
- Temperature at light 2000 climbs to 39-41 C once culture is dense; at <=1400 it stays at thermostat (~34-36 reading).

### E7 light cap (b041-b052), v1 with L_MAX 1100/1400/1700, i200, n=4 each
- 1100: 15.9,16.1,15.9,14.7; 1400: 19.0,15.3,17.3,14.3; 1700: 11.9,21.2,18.6,12.6; (2000: 13.2 mean).
- X at 108 h similar 815-850 for caps 1100-1700 vs 645 for 2000 -> light saturates ~1100-1400; 2000 only heats.

### E8 large inocula with v2 (b053-b056)
- Found turbidity is strongly nonlinear/clumped at high density: ~410 NTU at 1000-1280 mg/L on day 3-4 (ratio 0.35),
  ~480-510 at 1750-1950; saturates ~1000 NTU only for fresh dense inocula. Linear turbidity/ratio estimator useless >600 mg/L.
- calib2.py/calib3.py: fitted NTU(X) table (NTU at 48 h: 50->38, 150->112, 300->195, 550->310, 850->~400, 1200->~445,
  1700->508, 2300->603, 3000->675) times (1-0.0023*(h-48)); inversion error p10-p90 ~0.85-1.25 below 700 mg/L, wider above.
- Initial turbidity ~0.5 NTU per inoculum cell up to ~1500 cells; >=3000 cells saturates (~1000 NTU).

### E9 v3/v4: model+turbidity fusion estimator (b057-b074)
- v3: P(X) dead-reckoning corrected to linear turbidity; bug: harvest decision taken at the step the harvest happens
  (pre-harvest X) -> double 0.5 harvest for i3000; fixed by deciding 5 steps into each interval.
- v4: fusion with nonlinear NTU table (replay.py offline check: rms log error 0.18 vs lab; good <500 mg/L).
- v4 i200 (n=10): mean 15.4 g, p25 13.6; i1000: 28.1/22.4; i3000: 46.0/50.8.

### E10 variants at i200 n=6 (b075-b098) and low-density light (b099-b108)
- L_MAX 1000: mean 14.0 (X84 545); 1700: 15.7 (605); 1300 (v4): 15.4 (615); STIR 120: 12.9 (502, worse; shear/heat).
- i50: L_LOW 600 -> 8.5 g mean; L_LOW 900 -> 9.4 g.

### v5 (b109-b122): L_LOW 800, L_MAX 1500, smooth DP-derived harvest ramps (84 h: from 750; 96 h: 400->720; 108 h: 200->360), hold 1050
- i200 n=6: 17.5,17.0,21.2,19.8,19.8,19.2 (mean 19.1, X84 759); i50 n=3: 9.1,10.2,11.8; i400 n=3: 23.7,20.6,20.0; i1000: 30.3,30.2.

### E11 v6 (b123-b131): temp guard (light cut above 38.5 C filtered), NaN guards, post-harvest NTU recalibration
- replay.py over v3-v5 logs chose recalibration exponent 0.3 (rms log err 0.155 vs 0.170 none, 0.286 full).
- i5000: 68.2 / 63.6 g but NO hold harvests 48-96 h although X was 1100-1540 -> diagnosed: pump_L is noisy
  (+-0.3-0.5 L) and v3-v6 detected harvests from step-to-step pump changes (>0.3 L), so after the first harvest
  noise kept "diluting" the estimate. Replay on hourly logs hid this.
- i30: 6.7, 4.3, 9.8; i200: 15.4, 19.0, 19.6, 16.2.

### E12 v7 (b132-b139): harvest booked from our own mean requested fraction at each 12 h boundary (pump_L ignored),
  turbidity filter reset/recalibration 10 steps after the boundary.
- i5000: 68.2, 66.2 g (hold harvests now happen); i1000: 29.3, 24.7; i200: 19.9, 21.6, 15.4, 14.0.

### E13 high light at high density (b140-b155), i400 n=8 each
- v7 (L 1500): mean 21.9, median 21.7, p25 20.4, X84 892.
- v7 + 1900 umol when X>600: mean 22.3, median 22.6, p25 21.8, X84 909; tank +2-3 C (up to 39 C). Not significant; kept 1500.

### Final controller = controller.py (v7 + never harvest if predicted X < 40 mg/L; cleanup). Frozen before validation.

### VALIDATION (b156-b222), frozen controller.py, 67 batches, script valsum.py
Command: for (I,N) in (30,6)(60,6)(100,8)(150,8)(200,8)(300,8)(400,8)(700,5)(1500,4)(3000,3)(5000,3):
  python plant_trial.py --run bb5 runs/bb5/work/controller.py --n N --inoculum I --label val_iI
| inoc | n | median g | p25 g | min g | lost |
|---|---|---|---|---|---|
| 30 | 6 | 7.0 | 5.3 | 4.9 | 0 |
| 60 | 6 | 10.2 | 9.6 | 7.9 | 0 |
| 100 | 8 | 12.8 | 12.1 | 11.2 | 0 |
| 150 | 8 | 13.2 | 12.2 | 10.6 | 0 |
| 200 | 8 | 17.9 | 16.2 | 14.3 | 0 |
| 300 | 8 | 21.7 | 18.6 | 15.0 | 0 |
| 400 | 8 | 20.6 | 20.5 | 19.6 | 0 |
| 700 | 5 | 23.5 | 22.1 | 20.5 | 0 |
| 1500 | 4 | 32.5 | 30.1 | 29.2 | 0 |
| 3000 | 3 | 49.2 | 48.1 | 47.0 | 0 |
| 5000 | 3 | 69.7 | 68.3 | 66.9 | 0 |
All 67: median 18.2 g, p25 12.4 g, 0 lost, 0 errors. Inocula 100-400 (n=40): median 17.5 g, p25 13.6 g.
Observation: i3000 hold settled at ~900-1000 mg/L (estimator slightly high at that density) - slightly under the DP hold.
Budget: 223/300 used; 77 left unused as reserve.
Summary of cultures lost over the whole programme: 0 of 223.
