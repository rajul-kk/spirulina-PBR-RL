# LAB NOTEBOOK - run bb3

## Files read
- PLANT_MANUAL.md (root) - operator's manual. Key facts: 144 h batch, 7200 steps of 0.02 h; harvest every 12 h removes
  MEAN of requested fractions over the interval; objective = total harvested biomass, never lose culture; inoculum 30-5000
  (mostly 100-400); strains vary; target OD ~0.75; turbidity ~250 NTU/OD, reads low up to 30% with clumping, saturates 1000.
  Thermostat 35 C setpoint, limited heat removal. Light saturates / photoinhibits / heats.

## Experiments
All commands run from the blackbox root as `python plant_trial.py --run bb3 runs/bb3/work/<ctl>.py ...`.
Files read during analysis: runs/bb3/trials/results.jsonl and runs/bb3/trials/b*.csv (own trial logs only).
Analysis scripts (all in work/): show.py (summaries), mu.py / mubin.py (12 h growth rates from assays),
calib.py / calib2.py / calib3.py (NTU vs lab dry weight), dp.py / chk.py (logistic model + DP for harvest schedule).
mkconst.sh generates constant-action controllers.

### E1 (b000-b003, 4 batches) const_L400_H10.py: stir 120, light 400, harvest 0.1, random inoculum
Why: first look at plant, sensor formats, scale of numbers.
Result: inoc 113 -> 2747, 2982 mg; inoc 339 -> 5294; inoc 2797 -> 24294 (culture shrinking 1700->714 mg/L).
Learned: DW(mg/L) ~ 0.65-0.75 x inoculum at 12 h. Broth volume ~20 L (0.1 frac pumps ~2.0 L; later fit
V=19.9 L, IQR 18.4-21.3, n=313). Harvest mg = frac x DW x V. Growth slow: mu ~0.014/h at DW 100 falling to ~0 at
DW 1700 (L400). Harvested total is dominated by standing biomass.

### E2 (b004-b009, 6) constant light 1000 / 1600, stir 120, harvest 0.1, inoculum 200 (3 each)
Why: light response. Result: L1000 5389/6346/4253; L1600 6674/5242/6585. mu ~0.014-0.018/h, weak light effect
at DW 150-450. Big strain-to-strain variation (+-20%).

### E3 (b010-b015, 6) no-harvest growth curve at L1000 (n=2); stir 50 vs 200 at L1000 H0.1 (n=2 each), inoc 200
Result: H0: DW 152->624 and 159->812 at 132 h (mu falls from ~0.02 to ~0.008 as DW rises past 500).
S50: 8859, 6265; S200: 5649, 4680 (S120 same conditions: 5389/6346/4253). Stir 200 keeps NTU/DW ratio ~1 (no
clumping); stir 50 lets NTU read up to ~45% low by day 5.

### E4 (b016-b021, 6) dense start (inoc 1500, DW~1000), harvest 0.1, light 400/1000/1800 (n=2 each)
Result: 15366/15554, 18094/19015, 20424/20860. At high DW light matters: L1800 holds DW ~950 with 0.1/12 h
(mu~0.009), L400 lets DW decay to ~500. L1800 can heat broth to 38-39 C.

### E5 (b022-b033, 12) no-harvest growth curves, inoc 30 at L250/500/800/1800 and inoc 200 at L500/1800 (n=2 each)
Result (DW ratio 132h/12h): inoc30: L250 6.0/6.9x, L500 8.0/12x, L800 9.0/8.9x, L1800 5.2/5.2x (photo-inhibition
of thin cultures). inoc200: L500 3.3/3.1x, L1000 4.1/5.1x, L1800 4.2/5.7x.
Learned: optimal light rises with density: ~500-800 when DW<80, >=1400 when DW>250.

### E6 (b034-b039, 6) dense start (inoc 1500), L1800 harvest 0.2 / 0.05, L2000 harvest 0.1 (n=2 each)
Result: H0.2 28136/32646 (DW settles 450-600, ~2000-2400 mg per 12 h); H0.05 12723/13713 (DW ~1300);
L2000 H0.1 20886/20978 (= L1800; broth 37.5-38.7 C). Steady productivity peaks around DW 500-900 at
~8-10 mg/L/h (~2000 mg per harvest); flat optimum.
Growth table (mubin.py, mu per h by start-of-interval DW x mean light):
DW 0-80: 0.021 at L450-900 vs 0.013-0.015 at L>=1400; DW 150-250: 0.017 at L>=1400; 400-600: 0.0156 (L>=1400);
600-850: 0.0095; 850-1100: 0.006. No detectable temperature effect in 33-39 C.
Turbidity calibration (stir 120): DW = 0.84*NTU^1.106 (rms 11%); NTU saturates ~1000 (DW>~1400).
DP on logistic model (r 0.02, K 1700, which reproduced E4/E6 totals to ~10%): grow without harvest to the
hold level, hold, then take 0.5 at the last three harvests (108, 120, 132 h).

### E7 (b040-b045, 6) ctl_v1.py random inoculum: stir 120, light 400+1.6*DW (400-1800), hold 750, endgame 0.5x3
Result: inoc 116 6905, 1772 34924, 393 12623, 67 5856, 51 4500, 300 11578 (baseline ~2x). No losses.
Learned: light ramp too shallow (DW 300 only got ~900 umol); growth at DW 300-600 slow (0.007-0.01).

### E8 (b046-b055, 10) ctl_v2 (light 450+6*DW clip 500-1800, hold 500, 96h hold 450): stir 120 vs stir 50, inoc 200
Result: s120: 10194 15276 15582 13668 12801 (median 13668). s50: 14314 18456 18876 17447 15834 (median 17447).
Assay-based early growth (12-48 h): s50 ~0.023/h vs s120 ~0.018/h. Stir 50 is genuinely better (less shear),
and it also runs hotter (38-39.5 C) without harm. Clumping at s50 made the controller under-read DW.

### E9 (b056-b065, 10) ctl_v3 (stir 50, s50 calibration DW=exp(-0.647+1.181 ln NTU+0.00307 h), rms 9%), hold 500 vs 800, inoc 250
Result: h500: 22561 15783 17908 25189 17780 (median 17908); h800: 15392 22036 19219 19832 14962 (median 19219).
No resolvable difference: strain variation dominates; hold level optimum is flat between 500 and 800.

### E10 (b066-b075, 10) ctl_v4.py (stir 50, light from depth-averaged irradiance target I_OPT=253 with fitted
extinction k=0.0398 L/mg, cap 1800, temperature back-off above 38.5 C; hold 550 / 450 at 96 h; 0.5 from 108 h
unless DW<150 at 108 h). Inoculum 30 (n=3), 5000 (n=3), 120 (n=4).
Why: growth model fit (sim.py, fit_p2.npy: Haldane on depth-averaged light + density loss, rms 0.0062/h)
suggested light proportional to biomass; check extremes for losses.
Result: i30: 10142 5334 6056; i5000: 67958 74044 72656; i120: 13557 16119 10173 15463. No losses.
Learned: at stir 50 turbidity of dense-start cultures reads much lower than the growth-batch calibration; the
controller held i5000 at ~900 mg/L instead of 550.
Temperature: within stir level, residual growth falls ~0.0004/h per C (33-39 C) -> weak; keep light cap 1800.

### E11 (b076-b079, 4) dense start (inoc 1500) L1800 H0.1 at stir 50 vs 200 (compare b020/b021 at stir 120)
Result: S50 22690/20645, S120 20424/20860, S200 23565/25698 -> at high density more stir may help.

### E12 (b080-b087, 8) within-batch alternating stir 50/200 per 12 h interval (paired, removes strain variance),
L1800; A: inoc 300 no harvest (both phase orders, n=2 each); B: inoc 1000 harvest 0.15 (n=2 each)
Result (assay mu per h, 20 intervals per arm): A growing culture: s50 0.0157+-0.0016 vs s200 0.0109+-0.0015
(s50 better at every density band incl. DW>700). B dense start held at DW 600-750: s50 0.0113+-0.0014 vs
s200 0.0132+-0.0012 (not significant). B totals 22480 23626 24828 22405.
Decision: stir 50 always, except batches that start with saturated turbidity (dense inoculum): 200 rpm.

### E13 (b088-b097, 10) ctl_v5.py: quadratic log-calibration fitted on 295 stir-50 assays
(ln DW = 2.194 - 0.0187 lnN + 0.1264 lnN^2 + 0.00298 h, rms 14%, linear beyond 600 NTU) + dense flag
(stir 200 and x1.3 on DW estimate). Random inocula.
Result: 311 22621; 3140 46765; 57 11465; 113 8627; 1632 32179; 141 17479; 36 15076; 77 13566; 228 11697;
2234 38376. Dense-start batches were over-harvested (true DW held ~350 instead of 550): at stir 200 the
filaments do not clump, so the x1.3 gain and the stir-50 time drift were wrong for them.

### E14 (b098-b101, 4) ctl_v6.py: dense flag uses stir-120 calibration DW=0.84*NTU^1.106, no gain
Result: i2500: 48097 46309; i5000: 78105 74358 (v4 gave 68-74k).

### E15 (b102-b119, 18) irradiance target I_OPT 120/170/253/380 (inoc 60)
Result (assay mu 12-60 h; totals): I120 0.0240 (n=4; 10092 9510 5461 9550); I170 0.0268 (n=6; 14633 7584
14952 11739 6787 7451); I253 0.0269 (n=4; 8810 13456 13095 8276); I380 0.0214 (n=4; 4893 5688 7682 8019).
Optimum 170-250 -> I_OPT = 210 in the final controller.py.

## Validation of final controller.py (= ctl_v6 with I_OPT 210)
### V1 (b120-b144, 25) random inoculum (plant's natural distribution), label valR
Result: all 25 survived. Totals (mg): 11636 23676 19002 17750 13908 12355 16758 18507 27727 22250 11355 17324
13540 12995 15632 34279 22829 20050 29477 12741 12190 14108 14752 21485 37302.
Dense-ish starts (1085-2007) are harvested at 0.5 until turbidity unsaturates, then held at DW ~500-650 as designed
(b138, inoc 1085, initial NTU 512 < 700, so it ran at stir 50 without the dense flag; fine).
(Session interrupted here once; resumed from files in runs/bb3/.)

### V2 (b145-b186, 42) fixed inoculum 30/100/200/400/1000/3000/5000, n=6 each, labels valI<inoc>
Why: check the controller across the whole stated range of starting cultures, including the extremes.
### V3 (b187-b211, 25) second random-inoculum block, label valR2
Stats (valstats.py), total harvested mg:
| set | n | median | p25 | min | lost |
|---|---|---|---|---|---|
| random inoculum (V1+V3) | 50 | 17041 | 14165 | 7564 | 0 |
| random, inoc 100-200 | 21 | 15588 | 14069 | 11636 | 0 |
| random, inoc 200-400 | 20 | 18564 | 15966 | 11355 | 0 |
| fixed 30 | 6 | 8965 | 6005 | 5244 | 0 |
| fixed 100 | 6 | 15224 | 12789 | 12054 | 0 |
| fixed 200 | 6 | 17255 | 17245 | 16531 | 0 |
| fixed 400 | 6 | 22096 | 20891 | 20771 | 0 |
| fixed 1000 | 6 | 29171 | 28366 | 21734 | 0 |
| fixed 3000 | 6 | 51068 | 49201 | 47742 | 0 |
| fixed 5000 | 6 | 69603 | 68229 | 65427 | 0 |
| all validation | 92 | 18036 | 14424 | 5244 | 0 |
No culture lost in any of the 212 batches of the run. Reference (E1/E2, constant stir 120 / light 400-1600 /
harvest 0.1): inoc 113 ~2.9k, inoc 200 ~4-6.6k, inoc 339 ~5.3k -> the controller gives ~3-4x.
Weakest case seen: b174 (inoc 1000, 21.7k): slow strain held at DW ~650 grew only ~0.004/h; the controller
correctly harvested little, but a lower hold level might have suited it.
Budget: 212/300 used; 88 left unused as reserve.
