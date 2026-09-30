# Lab notebook - run bb4 (Spirulina PBR supervisory controller)

All commands run from the blackbox root. Batches numbered b000.. in runs/bb4/trials/.

## Files read
- PLANT_MANUAL.md (root) - the operator's brief.
- runs/bb4/trials/results.jsonl, runs/bb4/trials/b*.csv, runs/bb4/trials/budget.json (listing only) - my own pilot outputs.
- My own scripts in runs/bb4/work/. Nothing else in the repository was opened.

## Analysis tools (runs/bb4/work/)
- show.py <pattern> <every_h> - hourly log + lab dry weight (DW) per harvest.
- ratio.py - turbidity / (0.833*DW) over time.
- fitntu*.py - nephelometer model fits (NTU vs lab DW).
- prod.py - per-12h-interval productivity vs density and light.
- fitgrowth.py - fits a light-limited growth model to all stir-50 intervals; saves th.npy.
- simpol*.py - policy simulation on the fitted model.
- altana.py, lowana.py - paired within-batch light tests.

## Budget plan
~60 batches for plant identification, ~10 for candidate check, ~90+ for validation across
inoculum sizes, remainder reserve.

---

## E1 open-loop baseline (b000-b003, 4 batches)
`python plant_trial.py --run bb4 runs/bb4/work/c_open.py --n 4 --label open400_noharv`
stir 120, light 400, no harvest, random inocula (30, 215, 2830, 292).
Why: see the unperturbed plant, sensor behaviour, the assay/harvest record format.
Result: nothing harvested (as intended). DW(12h) ~ 0.6-0.7 x inoculum (inoc 215 -> 154 mg/L;
2830 -> 1735). Inoc 30: 24 -> 173 mg/L in 120 h (mu ~0.0165/h). Inoc 215/292: roughly linear
~3 mg/L/h. Inoc 2830 sat at ~1700 mg/L with zero net growth at light 400 (light-limited ceiling).
Temp 33-37 C; pH 9.6-10.2; conductivity ~28-30 mS - uninformative.
Learned: slow plant (doubling >40 h at these settings); dense cultures are light-limited.

## E2 light level, stir 120 (b004-b009, 6 batches)
c_open1000.py / c_open1600.py, --n 3 --inoculum 200 each.
Why: productivity vs light at moderate density.
Result: DW at 132 h: L1000 598/684/789, L1600 904/793/936 (L400 earlier ~485-566).
Temperature essentially unaffected by light up to 1600 (thermostat copes).
Learned: at 150-900 mg/L growth is light-limited and increases with light.

## E3 low inoculum at high light, and stir (b010-b017, 8 batches)
- c_open1600 --inoculum 30 (b010-11): 217 / 123 mg/L at 132 h.
- c_open2000 --inoculum 30 (b012-13): 97 / 126 mg/L; broth heats to 40-41.5 C at 2000.
- c_s200l1600 --inoculum 200 (b014-15): 628 / 701 at 132 h, temp ~37-38.
- c_s50l1600 --inoculum 200 (b016-17): 1148 / 1030 at 132 h.
Learned: STIR IS THE BIG LEVER - 50 rpm grows ~45 % more biomass than 120 and ~60 % more than
200 (shear). High light does not help thin cultures (photo-inhibition); 2000 heats the tank.
Turbidity badly under-reads dense, low-stir cultures (NTU/DW down to 0.38 of fresh calibration).

## E4 harvest mechanics + stir-50 light (b018-b023, 6 batches)
- c_fix.py (stir 50, L1600, harvest 0.15 constant), --inoculum 200 (b018-19): 8.3 / 11.6 g.
- c_s50l2000 --inoculum 200 (b020-21): DW 841 / 1147 at 132 h, temp 38-41.5.
- c_s50l800 --inoculum 30 (b022-23): 515 / 497 mg/L at 132 h (vs ~120-220 at stir 120).
Learned: pump volume 3.1 L per 0.15 -> working volume ~20 L; harvested_mg = frac x 20 L x DW
(checked to <1 %). Exponential rate at low density with stir 50 / L800: mu ~0.026-0.030 /h.

## E5 low-density light at stir 50 (b024-b027, 4 batches)
c_s50l400 / c_s50l1600, --inoculum 30, n=2 each.
Result (DW at 48 h / 132 h): L400 55,57 / 255,250; L800 77,82 / 515,497; L1600 52,50 / 449,368.
Learned: thin cultures prefer ~800; at 1600 they are inhibited early but catch up once dense.

## Analysis A1 - nephelometer model
NTU/DW ratio falls with density (saturation) and with time (clumping); faster at low stir.
Best fit on stir-50 data (352 assay points by E-v3): NTU = S(1-exp(-g c X/S)), S~535, g~0.85,
c = 1 - ~0.002/h after ~10 h. rms 8.9 % (log). => X estimate +-~12 % in the operating range.

## Analysis A2 - growth model (fitgrowth.py)
Per-interval (12 h) fit, stir 50, 240+ intervals: mu = mumax * <I/(Ks+I+I^2/Ki)>_depth - m,
Beer-Lambert over depth with attenuation kx*X. mumax 0.084/h, Ks 698, Ki 722, kx 0.00906 L/mg,
m 0.0008/h; rms 7.6 % per interval (close to assay noise).
Productivity (mg/L/h) at L2000: X=200 5.0, 400 7.6, 700 8.0, 1000 7.8, 1500 7.5.
Max specific rate ~0.027/h at X<100 with L 800-1200.
Batch (strain) residuals +-0.04 per 12 h; no dependence on broth temperature 33-41 C (r=0.004).

## E6 v1 closed loop (b028-b033, 6 batches)
c_v1.py: stir 50, light 700+3X (<=1800), target 450 mg/L, final two harvests 0.5.
Result: 21.3, 16.7, 27.8 (inoc 1222), 20.9, 9.1 (inoc 61), 12.6 g. Density ran 500-630 (estimator
under-read, no clumping term). Noise-driven over-harvest (3.4 g then 0.5 g) from 1-h NTU filter.

## E7 v2 target 700 (b034-b039, 6 batches)
c_v2.py: target 700, clumping term, 2-h filter, harvest-dilution applied to the filter.
Result: 15.7, 31.0 (1482), 18.6, 5.6 (inoc 33), 31.7 (948), 18.8 g. Not better than v1.

## Analysis A3 - policy simulation (simpol*.py)
On the fitted model: target 400-600 all within 1-2 %; final three harvests at 0.5 (108/120/132 h)
beat two (+2-5 %) and four; light schedule 700+5X capped 2000 best (+3 % vs 1600 cap);
constant 800 light costs ~25 %. Same ranking with a 25 %-slower strain.

## E8 v3 model-predictive harvest (b040-b047, 8 batches)
c_v3.py: growth-model prediction of X at the pump event; target 500; L 700+5X<=2000; end (9,10,11).
Result: 20.6, 32.1, 14.1, 29.2, 16.8, 14.4, 37.0, 14.0 g. b045 (inoc 280) 14.4 g: slow strain
(residual +0.058/12 h, ~25 % slower).

## E9 paired light test at operating density (b048-b053, 6 batches)
c_altA/c_altB.py --inoculum 700 --n 3: light alternates 1300/2000 each 12-h interval (both orders).
Result: productivity 2000: 8.2 mg/L/h, 1300: 7.2 (se ~1.3) - no significant difference.
Batch totals 14.7-23.5 g under identical treatment: strain spread dominates.

## E10 paired light test at low density (b054-b059, 6 batches)
c_lowA/c_lowB.py --inoculum 60 --n 3: alternate 700/1400 for first 96 h.
Result: mu 0.0241 +- 0.0022 (700) vs 0.0242 +- 0.0017 (1400) per h. No difference.
Some strains slow (b057 reached only 231 mg/L by 96 h).
Learned: light is not a strong lever anywhere in 700-2000 at stir 50; stir and density are.

## Final controller (controller.py) - "v4"
stir 50; X from nephelometer model (S 535, g 0.85, clumping 0.002/h after 10 h), 2-h filter with
+-25 % innovation clip, filter scaled by the harvested fraction at each pump event, capped by an
optimistic open-loop growth bound (x1.3 growth) so a faulty high reading cannot cause heavy
over-harvest; light 800+5X (<=2000), slew +20/step, backed off above 40.5 C; harvest planned to
leave ~550 mg/L using the growth model; 0.5 at 108/120/132 h (if X>150).

## E11 v4 pilot (b060-b069, 10 batches)
`python plant_trial.py --run bb4 runs/bb4/work/controller.py --n 10 --label v4`
Result: 20.1 (277), 15.6 (130), 30.1 (763), 33.3 (1230), 10.6 (128, slow strain), 14.8 (168),
27.1 (764), 24.3 (817), 17.4 (156), 13.3 (122) g. No losses. Pre-harvest DW held 600-750 mg/L.

## E12 validation of controller.py, fixed inocula + random (b070-b159, 90 batches)
`python plant_trial.py --run bb4 runs/bb4/work/controller.py --n N --inoculum I --label val_iI` for
I/N = 30/8, 60/8, 100/10, 200/10, 300/10, 400/8, 1000/6, 2500/4, 5000/6; then `--n 20 --label val_rand`.
Summary script: valsum.py. No culture lost, no controller errors. (Table in E15.)
Early-phase growth rate vs broth temperature across 35 batches: r = 0.13, no usable effect.

## E13 paired stir test at operating density (b160-b165, b182-b187, 12 batches)
c_stirA/B.py (controller + stir alternating 50/85 per 12-h interval after 12 h, both orders,
end harvests off), --inoculum 700 --n 3, twice (labels stirA/B, stirA2/B2).
Result: intervals at 85 grew 3.5 +- 0.6 mg/L/h more than intervals at 50 (order A +4.8, B +2.2).
BUT whole-batch productivity of the alternating batches was 7.5-8.5 mg/L/h, same as constant 50.

## E14 constant plateau stir (b166-b181, 16 batches)
c_s2_85/120/160.py and controller.py (stir 50 for 0-12 h, then constant), --inoculum 700 --n 4.
Productivity (mg/L/h, 12-132 h): 50: 7.0 (and 7.8 in val_i1000), 85: 7.5, 120: 5.5, 160: 4.4.
Learned: 50-85 rpm equivalent on the plateau; >=120 costs 25-40 %. The E13 interval effect is a
transient/lag artefact (zero net gain), not a steady-state benefit.

## E14b stir burst before harvest (b188-b193, 6 batches)
c_burstA/B.py: stir 120 in the last 1 h before alternate harvests, --inoculum 700 --n 3 each.
Hypothesis: better mixing at the pump raises the harvested concentration. Result: burst-ended
intervals +0.75 (A) and -3.8 (B) mg/L/h - no consistent effect; total productivity 8.0 as usual.
Checked for a 24-h periodicity in growth (even vs odd intervals) across all validation data: none
(0.0175 vs 0.0178 /h).
Pooled productivity vs density (L>1500, 1400+ intervals): 200-300 5.5, 300-400 6.6, 400-500 7.1,
500-600 8.4, 600-700 7.6, 700-850 7.9 mg/L/h -> operating band (550 post-harvest, ~650-700
pre-harvest) sits on the plateau. Controller left unchanged.

## E15 extra random validation (b194-b213, 20 batches)
`python plant_trial.py --run bb4 runs/bb4/work/controller.py --n 20 --label val2_rand`
Result: median 16.9 g, p25 14.4 g, min 7.9 g, 0 lost.

### Validation summary, controller.py (110 val_* batches; the 10-batch v4 pilot E11 not included)
| inoculum | n | median g | p25 g | min g | lost |
|---|---|---|---|---|---|
| 30-45 | 8 | 4.0 | 2.9 | 1.6 | 0 |
| 45-80 | 13 | 10.5 | 8.0 | 4.7 | 0 |
| 80-150 | 16 | 13.4 | 11.6 | 7.9 | 0 |
| 150-250 | 19 | 18.2 | 13.9 | 7.4 | 0 |
| 250-350 | 17 | 18.0 | 16.9 | 14.6 | 0 |
| 350-600 | 13 | 21.6 | 18.8 | 13.4 | 0 |
| 600-1500 | 11 | 29.7 | 28.1 | 23.9 | 0 |
| 1500-3500 | 5 | 42.5 | 40.3 | 34.7 | 0 |
| 3500-5000 | 8 | 70.1 | 65.5 | 56.5 | 0 |
| all | 110 | 17.4 | 12.8 | 1.6 | 0 |
| random inocula only | 40 | 17.1 | 12.9 | 4.7 | 0 |

Budget used: 214 / 300 (86 unused, held in reserve).
