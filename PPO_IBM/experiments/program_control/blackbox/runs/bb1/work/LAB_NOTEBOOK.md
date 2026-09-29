# Lab notebook: run bb1 (Spirulina PBR supervisory controller)

## Files read
- PLANT_MANUAL.md (root of blackbox/)
- runs/bb1/trials/results.jsonl and runs/bb1/trials/*.csv (my own trial logs), read via my scripts
  an_assays.py and an_intervals.py in this folder, and with awk/head.
- Nothing else in the repository was opened.

## Conventions
All commands run from blackbox/ root: `python plant_trial.py --run bb1 runs/bb1/work/<ctl>.py ...`.
"DW" = lab dry-weight assay (mg/L) taken just before each 12 h harvest (hours 12..132, 11 harvests;
there is no harvest at 144 h, so biomass left in the tank at the end is lost).
Tank volume inferred from pump log: 0.2 frac -> 4.18 L, so V ~ 20.9 L.

---
## Exp 1 (batches b000-b003, 4 used, total 4)
`plant_trial.py --run bb1 runs/bb1/work/c01_const.py --n 4 --label c01_const`
Why: baseline and to see output format. Stir 120, light 400, harvest 0.2 whenever NTU > 190 (~OD 0.75).
Result: 2774, 2547, 6437 (inoc 129-275) and 18064 mg (inoc 1103). No losses.
Learned:
- Inoculum N cells -> DW ~0.65*N mg/L at start (167 -> ~100 mg/L; 1103 -> ~765 at 12 h).
- Growth is slow: ~0.010-0.012 /h at light 400 (doubling ~60 h). Harvesting at OD 0.75 keeps the
  tank at low density where volumetric productivity is low; the inoc-1103 batch was drained from
  765 to 305 mg/L by 20 %/12 h, i.e. growth < 0.0186 /h.
- Turbidity ~0.83 NTU per mg/L at low density, reads increasingly low at high density/late in batch.

## Exp 2 (b004-b015, 12 used, total 16): light sweep
`c02_L{150,300,600,1000,1500,2000}.py --n 2 --inoculum 200`; stir 120, no harvest before 96 h, then 0.5.
Why: dose-response of light; test "grow then drain" harvest.
Result (mg): L150 4363/4993, L300 8742/7843, L600 11148/10690, L1000 10903/11710,
L1500 12144/7912, L2000 11575/13992. No losses. Temperature: +3-5 C at L2000 (38-40 C).
Learned: late harvest ~doubles yield vs Exp 1. Growth rises with light up to 2000, no sign of harm
at inoc 200. Strain-to-strain spread is large (L1500: 7.9k vs 12.1k).

## Exp 3 (b016-b021, 6 used, total 22): stir 50 vs 200 at L1500, inoc 200
Result: S50 19645/14703/18030; S200 15604/15964/5612.
Learned: stir 50 clearly best (vs S120 L1500 12144/7912). Less shear/heat. Adopt stir 50.

## Exp 4 (b022-b026, 5 used, total 27): inoculum 30, stir 120
L2000: 1763/1726/1338 (T 39-40 C); L300: 3052/1335. No losses.
Learned: small cultures grow only ~4x in 96 h. Hint that high light is not better at very low
density (photoinhibition or heat), but n is tiny.

## Exp 5 (b027-b035, 9 used, total 36): light sweep at stir 50, inoc 200
L600 12526/11720/10894; L1000 11104/13173/12903; L2000 15576/17364/15709.
Pooled interval analysis (an_intervals.py): productivity dX/dt keeps rising with density at high
light (S50 L1500-2000: ~5.5 mg/L/h at 200-300 mg/L, ~7 at 300-600, ~7-10 at 600-900).
Specific rate mu drops from ~0.021 /h (200-450 mg/L) to ~0.011 /h (>600).
=> keep the culture dense, harvest late. Stir 50 + light 1500-2000 is the best so far.

## Exp 6 (b036-b038, 3 used, total 39): large inocula, S50 L2000, late drain
`c05_S50_L2000.py --n 1 --inoculum {1000,2500,5000}` -> 27236 / 38927 / 53008 mg.
Assays: inoc 1000 grows 743 -> 1596 mg/L by 108 h; inoc 2500 stalls at ~2000; inoc 5000 flat at
~3000 mg/L (zero net growth). => density ceiling ~2000-3000 mg/L; productivity collapses above
~1500. Large cultures should be harvested down early. Growth after a 0.5 harvest is faster
(self-shading relieved).

## Exp 7 (b039-b044, 6 used, total 45): within-batch light blocks, inoc 200, S50
`c07_lightblocks.py --n 6 --inoculum 200`: light random from {800,1400,2000} per 6 h block.
Analysis an_blocks.py (ln NTU slope per block, batch fixed effects, temp and ln NTU covariates).
Yields 14.6-19.1k. Early (first 48 h, <~400 mg/L): 800 ~ 1400 ~ 2000 (light saturated; 2000
slightly worse, hotter). Later (dense): 1400 best (+0.009 /h vs 800), 2000 +0.006.
Temperature coefficient ~ -0.0015 to -0.0023 /h per C (not significant).
=> light ~800-900 at low density, ~1400-1500 when dense; do not use 2000 (heats to 38-40 C).

## Exp 8 (b045-b050, 6 used, total 51): v1 controller c08_v1.py, inoc 200
Light 800->1500 scheduled on filtered NTU, temperature servo cap 37 C, drain 0.5 from 96 h.
14651 14801 11905 14482 18360 19202 (mean 15.6k). Same as fixed-light arms within noise.

## Exp 9 (b051-b054, 4 used, total 55): v1 at inoc 30
8868 8923 5678 5394. Low density growth ~0.03 /h at S50/L800 (vs ~0.015 at S120/L2000 in Exp 4):
the Exp 4 small-culture results were depressed by stir/heat, not inherent.

## Exp 10 (b055-b058, 4 used, total 59): within-batch light blocks at low density, inoc 40
`c10_lowblocks.py`: {300,600,1000,1500} per 6 h block. 300 is worse (-0.007 /h); 600-1500 equal.
No sign of photoinhibition up to 1500 at 30-250 mg/L. Yields 9071 7297 6168 4097.

## Exp 11 (b059-b062, 4 used, total 63): within-batch stir blocks {50,100,200}, L1400, inoc 200
Yields 14692 12707 16161 18298. NTU slope analysis unusable: stirring de-clumps filaments so
turbidity jumps with stir (reads ~higher at 200 rpm) -> sensor artefact. Batch-level evidence
(Exp 3, Exp 5 vs Exp 2) stands: stir 50 best. Also: turbidity calibration depends on stir, so the
final controller keeps stir fixed at 50.

## Model fit and DP (an_fitP.py, dp_harvest.py; no batches)
Fit dX/dt = a X/(1+X/K) - m X to all stir-50 assay intervals (after correcting for harvest via
pump volume). Latest fit (650 intervals): a=0.040 /h, K=522 mg/L, m=0.0075 /h, resid sd 0.08 (ln).
Productivity P(X): 2.7 (100 mg/L), 4.4 (200), 6.2 (400), 6.8 (800, max), 5.7 (1200), 1.8 (2000),
<0 (3000) mg/L/h. DP over harvest fractions (0-0.5 at 12..132 h): hold pre-harvest density ~650
mg/L; at 96 h cut to ~420; at 108 harvest 0.5 if >~150 mg/L; 120 and 132: 0.5. Drain start at
96 vs 108 is worth only ~1-3 % for 100-400 inocula (model).
Turbidity calibration (an_calib.py): NTU/DW 0.76 at <50 mg/L early, 0.5 at 500-800, 0.2 at >1500;
also falls with time (clumping). Fitted ln NTU = quadratic(ln DW) + t terms, resid sd 0.09.

## Exp 12 (b063-b072, 10 used, total 73): v2 controller c12_v2.py
Density estimate from turbidity calibration; light 900->1500 (X 150->400 mg/L), T cap 37;
DP harvest policy (X_hold 750, X_96 420, X_108 150).
inoc 200: 17386 16791 15675 15057 17275 18026 (mean 16.7k, min 15.1k: tighter than v1).
inoc 2500: 44737 41374 (vs 38927 with late drain). inoc 30: 12198 7619.

## Exp 13 (b073-b092, 20 used, total 93): v3 (T cap 37.5, light floor 800) natural inocula
`c13_v3.py --n 20` (no --inoculum). Inocula 75-4671. Yields: median ~15.9k, 25th pct ~14.0k,
no losses. Big inocula (1275, 1638) held at ~900-1050 mg/L with little growth -> lower hold target.

## Exp 14 (b093-b100, 8 used, total 101): final controller (X_hold 650 from refit DP), inoc 30
8583 4496 12123 6826 9237 3752 8159 8119. No losses; slow strains give ~4k.
Then added a safety bound to controller.py: biomass estimate used for harvesting is capped at
1.3 x first reading x exp(0.035 h) (reduced by each harvest), so a stuck-high turbidity signal
cannot drain a small culture. Offline replay on logged batches: decisions unchanged; with a
simulated stuck-at-1000 NTU sensor on a 20 mg/L culture, harvesting still waits until 96 h.

## Batch-level temperature check (no batches)
an_batchmu.py over 88 stir-50 batches: early growth rate vs mean broth temperature 32-37 C shows
no trend (corr 0.007; slope -0.0002 /h per C). The 37.5 C light ceiling is a safeguard only.

## Validation of final controller.py (b101-b172, 72 used, total 173)
Commands (from blackbox/ root):
- `python plant_trial.py --run bb1 runs/bb1/work/controller.py --n 30 --label val_nat_a`
- `python plant_trial.py --run bb1 runs/bb1/work/controller.py --n 30 --label val_nat_b`
- `python plant_trial.py --run bb1 runs/bb1/work/controller.py --n 6 --inoculum 50 --label val_i50`
- `python plant_trial.py --run bb1 runs/bb1/work/controller.py --n 6 --inoculum 100 --label val_i100`
(plus Exp 14, inoc 30, n 8; the only change since then is the sensor-fault bound, which does not
alter decisions on normal batches, per the offline replay).
Summary (an_validation.py):
| set | n | median mg | p25 mg | min | lost |
|---|---|---|---|---|---|
| natural inocula (a+b) | 60 | 16914 | 13715 | 10001 | 0 |
|  natural, inoc <100 | 6 | 14177 | 11256 | 10394 | 0 |
|  natural, inoc 100-200 | 27 | 13973 | 12096 | 10001 | 0 |
|  natural, inoc 200-400 | 13 | 20533 | 18876 | 16944 | 0 |
|  natural, inoc 400-1000 | 3 | 25174 | 24440 | 23706 | 0 |
|  natural, inoc >=1000 | 11 | 53561 | 39314 | 28184 | 0 |
| fixed inoc 30 | 8 | 8139 | 6244 | 3752 | 0 |
| fixed inoc 50 | 6 | 9774 | 9381 | 9183 | 0 |
| fixed inoc 100 | 6 | 15424 | 14885 | 12795 | 0 |
For reference, the Exp 1 baseline (harvest at the OD 0.75 target, stir 120, light 400) gave
2.5-6.4k at inoc 129-275, and the first "grow then drain" constant-light arm ~11k at inoc 200.

Batches used: 173 of 300. Remaining 127 deliberately unspent: the remaining candidate tweaks
(light 1500 vs 1800 when dense, drain timing) are worth ~1-3 % by the within-batch data and the
model, below what 20-30 batch A/B tests can resolve against the ~20 % strain-to-strain spread.
