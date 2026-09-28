# Lab notebook: Spirulina PBR supervisory controller (black-box)

## Files read
- 2026-09-29: PLANT_MANUAL.md (the only plant documentation).
- Directory listing of blackbox/ (names only): PLANT_MANUAL.md, plant_trial.py, work/ (contains an empty-looking __pycache__ I did not create; not opened).
- Not opened: plant_trial.py or anything outside blackbox/work and blackbox/trials.

## Experiments

### E1 (b000-b001): baseline constant stir 120, light 400, h 0.2, n=2, random inoculum
`python plant_trial.py work/c_base.py --n 2 --label base_s120_l400_h02`
- Totals 4241 / 4279 mg. No loss. 11 harvests (12..132 h); the one at 144 h does not exist, so tank inventory after 132 h is wasted.
- DW stays ~85-110 mg/L all batch (steady state: net growth = dilution 0.2/12h, mu ~ 0.0186/h).
- pump_L steps ~4.1 L per 0.2 harvest -> working volume ~20-21 L. Sum(DW)*0.2*V = total gives V ~ 20 L.
- Per-harvest weighed mg is noisy (+-30% vs 0.2*V*DW), totals are consistent.
- Inoculum N cells -> ~0.6-0.8 mg/L dry weight at 12 h.

### E2 (b002-b009): no-harvest light sweep, stir 120, h 0, L = 200/800/1400/2000, n=2 each
`python work/mk.py c_const_s120_l{L}_h0 120 {L} 0; python plant_trial.py work/c_const_s120_l{L}_h0.py --n 2 --label L{L}_h0`
- Lab DW is still assayed at every 12 h even when harvest frac is 0 (useful: free calibration data).
- Growth is slow: L200 98->200 mg/L in 120 h (mu~0.006/h); L2000 59->286 (mu~0.013) but temperature rises to 37-38 C (thermostat can't cope at 2000).
- L1400 (inoc 283): 215->610 mg/L, roughly linear ~3.3 mg/L/h (light-limited). L800 (inoc 619): 413->779, ~3 mg/L/h.
- Large inocula (2421, 3499 cells -> 1500-2000 mg/L) barely grow (~1.7 mg/L/h), plateau.
- NEPHELOMETER IS NON-MONOTONIC: at DW 1500-2200 mg/L it reads only 600-750 NTU (reads 900-1000 at t=0 then falls as clumping sets in). NTU/DW ~0.70 at 100 mg/L, ~0.70 at 600 mg/L early, drops with batch age.
- No cultures lost.
- Puzzle: base (L400, h0.2) sustained mu~0.0186/h at ~95 mg/L, faster than any h0 batch at similar density. Harvest/fresh medium may stimulate growth, or strains differ. -> E3.

### E3 (b010-b017): inoculum fixed 200; (L400,h0) (L1000,h0) (L1000,h0.2) (L400,h0.4), stir 120, n=2
- L400 h0: 140->300/410 mg/L; L1000 h0: 160->424/554 (2.2-3.3 mg/L/h, roughly LINEAR in time from ~150 mg/L => light-limited growth, constant volumetric productivity).
- L1000 h0.2: 6429 / 5695 mg total; density slowly declined 166->145 / 168->105 (dilution 0.0186/h slightly exceeds net growth at ~150 mg/L).
- L400 h0.4: BOTH LOST at 120 h (washout). DW fell 145->7-11 mg/L. First 3 harvests gave 1.1-1.2 g, 0.8 g ... total 3.8-4.0 g. Net mu at 7-100 mg/L ~ 0.012-0.02 /h (doubling ~35-60 h). Loss declared when DW falls to ~7 mg/L or below.
- Specific growth rate is low everywhere: max mu ~0.02/h. Overharvesting is the obvious killer.

### Turbidity calibration (work/calib.py on all batches so far)
- NTU/DW ratio ~0.78 at 50-150 mg/L, ~0.70 at 300, ~0.65 at 600-800, ~0.45 at 1500, ~0.31-0.48 at 2000 (and falling over the batch).
- A saturating fit NTU = 1000*(1-exp(-DW/1230)) captures it (inverse DW = -1230*ln(1-NTU/1000)). Above ~1200 mg/L the reading is ambiguous (drifts down with clumping), so never operate there.

### E4 (b018-b020): inoculum 500 (~350 mg/L), h0, stir 120, light stepped in 24 h blocks (500/1000/1500/2000 in 3 balanced orders)
`python work/mksched.py c_E4A 24 "..."` etc.; `python plant_trial.py work/c_E4A.py --n 1 --inoculum 500 --label E4A` (A,B,C)
- Temperature: steady ~33.5-34.5 C at 500-1500 umol; at 2000 umol it climbs over ~8-12 h to 38-38.8 C (thermostat saturated). Time constant ~5-8 h. Recovery after lowering light takes ~3 h.
- Per-12h growth from lab DW (work/blocks.py) and from NTU (work/ntublocks.py) is too noisy (sd 2-4 mg/L/h) to resolve light response in single 12 h blocks. Means: 500 -> 0.6-2.1, 1000 -> 2.1-2.8, 1500 -> 2.9-4.5, 2000 -> 2.6-3.9 mg/L/h; first 12 h at 2000 was good (DW-based 6.1) and second 12 h (T>36.5) poor (-0.8) => heat above ~36.5 C seems to hurt.
- Conclusion: more light helps up to where the tank overheats; ~1500 is safe thermally at stir 120.

### E5 (b021-b029): stir sweep 50/125/200, L1200, h0, inoculum 400, n=3
- Whole-batch DW slope: s50 4.51/4.79/4.43, s125 3.35/3.67/2.55, s200 2.48/3.23/3.30 mg/L/h. LOW STIR GROWS ~50% FASTER (shear).
- But at s50 the nephelometer reads progressively low: clumping factor (NTU / saturating-calibration) falls 1.0 -> 0.60 by 132 h. At s>=125 no clumping (factor ~1.05-1.1).

### E6 (b030-b035): stir 80 const vs stir 50 with 150-rpm pulse 1.5 h before each harvest; L1200 h0 inoc 400, n=3
- s80: slopes 4.78/4.78/6.42 mg/L/h; pulse: 4.05/5.09/5.04. s80 is at least as good as s50. Pulsing does not undo clumping.
- Clumping at s80 similar to s50 (factor ->0.66 by 132 h at DW 300->1000).

### E7 (b036-b039): inoculum 30, h0, s80, L1200 vs L400, n=2
- No losses. Exponential growth mu ~0.025-0.03 /h from 20 up to ~250 mg/L at BOTH light levels (low-density culture is light-saturated at 400). mu falls above ~250-300 mg/L (self-shading) -> volumetric productivity plateau ~4-6 mg/L/h at s80.
- Clumping much weaker at low density (factor 0.85-0.9 by 132 h) -> clumping rate scales with biomass, modelled as dc/dt = -kc*X.

### Grey-box model (work/mysim.py, my own model from the data above)
- dX/dt = Pm*tanh(mu*X/Pm), mu 0.018-0.025, Pm 3-5. Optimising a "grow, hold at X_post, dump 0.5 on the last n harvests" policy gives 8-12 g for inoculum 140 mg/L vs 4.2 g baseline; best X_post 300-600, n_dump 3-4. (Model has no high-density penalty, so it over-favours high X_post.)

### E8 (b040-b045): first real controller work/ctrl_v1.py (s80, light integral-controlled to keep T<=36 C within 600-1800, NTU->X with saturating inverse + clumping correction kc=4.7e-6, hold X_post, dump 0.5 on harvests 9-11, floor 60 mg/L), inoculum 200, n=3 each
`python work/variant.py work/ctrl_v1.py work/c_v1_X300.py Xpost=300` (and X600); `python plant_trial.py work/c_v1_X300.py --n 3 --inoculum 200 --label E8_X300`
- X_post 300: 13503 / 15450 / 12007 mg. X_post 600: 12256 / 11034 / 11581 mg (never reached hold before dump). Both ~3x the constant baseline (4.2 g). No losses.
- Light controller settles ~1500-1800 umol with T ~35.5-36.5.
- Replay of estimator on logs (work/replay.py): X_hat/DW drifts to 0.7-0.8 late in batch => clumping under-corrected. Grid fit (work/fitkc.py) over s<=80 batches: K=1230, kc ~8e-6 gives unbiased mean log-error. Updated ctrl_v1 kc -> 8e-6.

### E9 (b046-b053): ctrl_v1 (kc 8e-6) X_post 250 vs 400, inoculum 200, n=4
- X250: 16600 / 8590 / 16067 / 11280; X400: 10879 / 12404 / 13675 / 12092 mg. Strain-to-strain spread (8.6-16.6 g) dominates.
- KEY FINDING: in several batches (b047, b050, b055, b058, b045) the temperature reading sits at 35.3-36.5 C even at 600 umol, so the T-capped light loop drove light to its 600 floor all batch; those batches were among the worst (8.6-10.9 g). At equal light, batch-to-batch temperature differs by +-1-1.5 C (seen in all constant-light runs too) -- either a per-batch sensor offset (manual says sensors drift) or ambient. A hard temperature setpoint on the raw reading is not robust.

### E10 (b054-b063): Tcap 35 vs 37 (Lmax 2000), X_post 300, inoculum 200, n=5
- T35: 15023/10714/9076/12434/10431 (mean 11.5 g; mean light ~1065); T37: 13184/14595/9967/14763/10434 (mean 12.6 g; mean light ~1780). Higher light helped slightly.
- Pooled regression on early mu (strain covariate) (work/pool.py): total ~ 6.7 + 283*mu; residuals do not separate the treatments clearly.

### E11 (b064-b079): 2^4 full factorial, n=1 per cell, inoculum 200: light 1400 vs 2000 (only a 38.5 C runaway guard), stir 60 vs 80, X_post 250 vs 350, n_dump 3 vs 4
`python work/variant.py work/ctrl_v1.py work/c_F_L{L}_s{s}_X{X}_d{d}.py Lmax=.. stir=.. Xpost=.. ndump=.. Tcap=38.5`; analysis work/fact.py
- Main effects (g, high minus low): L2000 -1.41 (-0.70 adj. for strain mu), stir80 -1.22 (-0.43), X350 -0.55 (+0.17), ndump4 +1.04 (+0.28). Resid sd ~2.1-2.4 g.
- L2000 runs sat at 37.1-38.5 C (guard); L1400 runs at 33.4-36.9 C. So 2000 umol overheats and is worse; 1400-1800 is the sweet spot. Stir 60 >= 80. X_post and n_dump are flat in this range.

### v3 controller (work/ctrl_v3.py): stir 60, fixed light Lset with slow guard (only lowers light if T rises >2.5 C above the batch's own hour 3-9 baseline, floor 1000), X_post 300, n_dump 4
### E12 (b080-b085): v3 at inoculum 30, 1000, 5000 (n=2 each)
- i30: 2773 / 3842 mg; still in exponential growth (DW 140-190 mg/L) when the 0.5 dumps started at 96 h -> dumping a small culture early wastes growth.
- i1000: 21011 / 24890 mg. i5000: 63603 / 65618 mg (mostly the inoculum itself, harvested 0.5 per event from 2800-3000 mg/L down to ~300). No losses.

### DP on my grey-box model (work/dp.py): dX/dt = Pm tanh(mu X/Pm) - m X, mu 0.022, Pm 4.5, m 0.001
- Optimal policy: hold ~430-580 mg/L until harvest 7, then draw down with rising thresholds: harvest only if pre-harvest X > ~350 (h8), ~160 (h9), ~70 (h10), ~45 (h11). Heuristic (hold 300, dump 4) is within ~3% of DP for typical inocula, 5-10% worse for tiny ones.
- -> ctrl_v4.py = v3 + endgame post-harvest minima Xend = (300, 120, 60, 35) for harvests 8..11.

### E13 (b086-b101): v4, 2x2 Lset 1400/1750 x X_post 300/450, inoculum 200, n=4
- Means (g): L1400/X300 13.60, L1400/X450 13.82, L1750/X300 14.21, L1750/X450 13.03. All within noise; ranges 11.5-15.8. Flat optimum.
- T: L1400 ~35.0 C, L1750 ~36.3-37.1 C.
- Estimator replay: X_hat/DW ~0.91 (partly an artefact of replaying hourly logs across harvest steps).

### E14 (b102-b109): v4 with stir 50 vs 100 (Lset 1600, X_post 350), inoculum 200, n=4
- s50: 15974 / 12601 / 15044 / 14914 (mean 14.6 g); s100: 8611 / 16467 / 9513 / 11596 (mean 11.5 g). Lowest stir is best; 50 rpm adopted.

### v5 = v4 with stir 50, Lset 1600, X_post 350 (work/ctrl_v5.py, copy work/c_v5.py)
### E15 (b110-b121): v5 across inocula
- i30: 4088 / 3755 / 7359 (v3 gave 2773/3842: the endgame change helps small cultures); i60: 9039 / 9799; i100: 7544 / 10314 / 14626; i400: 16113 / 16100 / 16842; i2000: 37386. No losses.

### Pooled look (E8-E14 inoc 200): total vs mean T: slope ~ -0.45 g/C (weak); pH mean correlates +0.42 with total (a growth indicator, not a lever). At fixed 1400 umol, batch temperature readings differ from 33.4 to 36.9 C.

### E16 (b122-b137): v5 vs v6 (Lset 1900, absolute guard 37.0 C, floor 1400), inoculum 200, n=8 each
- v5: 10597 15664 14727 14502 14072 13566 16190 15734 (mean 14.38, median 14.6)
- v6: 14109 12262 10773 13814 14224 14370 12353 15843 (mean 13.47, median 13.95)
- More light is not better; keep v5. work/controller.py = v5 + defensive obs parsing (NaN/missing keys).

### VALIDATION (b138-b197): final work/controller.py, 60 batches at the pilot plant's own inoculum distribution
`python plant_trial.py work/controller.py --n 10 --label VAL` x6 ; summary `python work/val.py VAL`
- ALL n=60: median 13.50 g, 25th pct 11.26 g, mean 15.85 g, min 4.22 g, cultures lost 0.
- inoculum <100 (n=5): median 7.04, p25 5.57, min 4.22; 100-200 (n=24): median 11.94, p25 11.03; 200-400 (n=22): median 16.32, p25 14.17; 400-1000 (n=3): median 22.37; >=1000 (n=6): median 32.9 (max 59.2 at 4454 cells).
- For reference the constant baseline (stir 120, light 400, h 0.2) gave ~4.2 g at inoculum ~120.

Budget used: 198 / 300 batches.
