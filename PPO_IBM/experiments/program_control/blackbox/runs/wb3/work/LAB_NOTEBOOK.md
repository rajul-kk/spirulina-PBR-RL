# wb3 lab notebook

## Files read
- PLANT_MANUAL.md (operator brief)
- plant_trial.py (trial tool; imports harness._ctrl_obs/to_env_action, which I am not allowed to read)
- PPO_IBM/environments/genetic_env.py (full simulator, 1403 lines; it imports nothing from PPO_IBM/training)

## Source notes (genetic_env.py, difficulty 2)
- 1 agent ~ 1e8 mass units = ~10 mg DW (divides at 14 mg, 7-14 mg range); start mass 12.5 mg/agent (less for big inocula).
  OD = mg/20 L/300 -> OD ~ cells/600. Inoculum 30 -> OD ~0.06; 300 -> ~0.6; 5000 -> ~9.
- Extinction: <10 agents or <1 mg total. Lysis base 5e-4/h, up to 2.5e-3/h when mu < respiration.
- mu = mu_max(N(0.04,0.006)) * f_I * f_Q * f_P * f_C * f_tox * f_T * shock * f_O2 * f_pH * f_osm * repair_tax * fatigue.
  - repair tax: 35% * sigmoid(0.12*(rpm-100)) -> ~0 at 50 rpm, 3% at 80, 17.5% at 100.
  - fatigue: builds for rpm>80 (max 15%).
  - light: Haldane (Ks~100, Ki~2500) on path light over 6.7 cm; red drives, total inhibits; optimum path-mean total ~500.
    k_red = 0.5+0.2*X(mg/L), k_blue = 0.2+0.25X, k_green=0.05+0.06X (per m). Flashing-light integration weight 0.5 at 50 rpm -> 1 at 200 rpm.
  - photo-shock exp(-3e-6*diff^2) only when path light exceeds the acclimated EMA (tau 1-4 h): ramp light up slowly.
  - O2 inhibition 1/(1+(DO/35)^4); kLa 1.4/h at 50 rpm .. 5.6/h at 200 rpm, divided by 1+(OD/10)^2.
  - temperature: light heats 0.001 C/h per umol; ambient loss 0.1*(T-25); thermostat 35 C, chiller max 0.6 C/h
    -> light <= ~1600 holds 35 C, 2000 -> ~39 C. CTMI T_opt N(36,1), Tmax 44.5.
- Harvest: at steps 600..6600 (11 events), removes the MEAN requested fraction over the preceding 600 steps (Bernoulli per agent); biomass left at 144 h is not counted.
- Turbidity = 250*OD*(0.7+0.3*pigment)*clump^-1/3/(1+0.05 OD)*drift(0.95-1.05)*jitter(+-2%)*flow noise; clips 1000.
  Pigment bleaches (0.01/h) when path-mean light > 1000 or N < 75.
- Clumping at low rpm: stick 0.05*OD/h*(1-rpm/250); shear breakup only above 80 rpm.

## Experiments

### E1 (b000-b001) probe0: `python plant_trial.py --run wb3 --privileged runs/wb3/work/probe0.py --n 2 --label probe0`
Why: see the CSV format, timing (~7 s/batch) and a baseline. Stir 80, light ramped with turbidity to 1500, harvest to OD 0.75 from turbidity.
Result: inoc 102/102 -> 7.3 g / 11.5 g, none lost. Growth ~0.030/h at OD 0.2-0.7. Turbidity per OD fell 250 -> ~170 by 138 h (clumping), so the controller held true OD ~1.3 while believing 0.9.

### Modelling (no batches)
- mf.py: my own mean-field version of the light/temperature/O2 equations. Steady productivity mu(X)*X keeps rising with density up to OD ~5 (light-limited, respiration only 1% of mu_max) -> harvesting early is wasteful.
- dp.py: DP over harvest fractions on mu(X): optimum = grow without harvest, then drain 0.5 at the last 3 events (108/120/132 h); big inocula held around OD 4-5.
- mfsim.py: time-stepping mean-field plant (thermostat PI, acclimation shock, clumping, pigment, O2, interval-mean harvest, sensor model) that runs a Controller on synthetic obs.
- ctrl_v1.py: observer (turbidity EMA corrected by internally integrated clump and pigment), light = intensity giving path-mean PAR 500 (Haldane optimum sqrt(Ks*Ki)), capped at I_MAX, ramp-up 40 umol/h; harvest: cap OD_CAP, drain last N_DRAIN intervals at 0.5, never harvest below OD 0.25.

### E2 (b002-b007) ctrl_v1 defaults (I_MAX 1600, OD_CAP 3, N_DRAIN 3, stir 80): `... --privileged runs/wb3/work/ctrl_v1.py --n 6 --label v1`
Why: first test of grow-then-drain, privileged logs to calibrate the model at high OD.
Result: inoc 206/133/712/234/257/38 -> 18.5/9.6/26.4/24.6/22.6/5.4 g, none lost.
Learned:
- Turbidity/OD falls to ~110 at OD 3.3 (clump ~7). clumpfit.py: integrating the source clump ODE with the TRUE cell-count growth reproduces the clump implied by turbidity within ~10% -> clump model right; clumps are diluted only by division.
- fitmu.py: one per-batch scale mu_eff reproduces the whole true-OD trajectory (OD 0.06-3.4, incl. harvests) to 0.5-2% rms log error. mu_eff = 0.022-0.041, mean 0.0323 -> lumped constant factor 0.81 (f_Q*f_P*f_C*f_pH). Model structure validated; mfsim LUMP = 0.81.

### Model sweeps (sweep1.txt, olopt1.txt; 40 model batches per setting, plant-weighted inocula)
- base v1: weighted mean 21.5 g. I_MAX 1800 +2.8% (2000: +1.2%, 1400: -4%). RAMP_UP 2 (100 umol/h) +2%; 0.4 -4%. OD_CAP 2 +2% (only big inocula); 4 -2%. N_DRAIN 3 best (4: +1% but hurts small inocula; 5 worse). F_DRAIN 0.5 best. Stir 65-72 ~= 80; 88 -1.6%, 100 much worse. TM_TARGET 400-600 flat.
- Open-loop schedule search: small inocula want [0..0, 0.4, 0.5, 0.5]; mid (150-400) want 0.2-0.3 at events 6-8 then 0.5x3; large (1500) harvest from the start.
- sweep2/3 (model): with I_MAX 1800, RAMP_UP 2-4, OD_CAP 2, stir 75: weighted mean 23.1 g (+7% over v1). I_MAX 1700-1900 flat, stir 70-80 flat, STIR_HI 90-110 above OD 2 worse (repair tax > clump breakup gain). OD_CAP 1.5/2.5 slightly worse.

### E3 (b008-b019) ctrl_v2 = v1 with STIR 75, I_MAX 1800, RAMP_UP 4 (200 umol/h), OD_CAP 2: `... --privileged runs/wb3/work/ctrl_v2.py --n 12 --label v2`
Why: pilot the model-tuned settings on random inocula.
Result (inoc -> g): 380->24.9, 182->12.4, 49->9.9, 168->16.8, 255->12.7, 271->19.1, 57->7.3, 63->15.6, 331->26.7, 48->5.2, 142->17.6, 265->15.0. None lost. Median 15.3 g.
Learned: strain variance dominates (inoc 63 got 15.6 g, inoc 255 got 12.7 g). fitmu (single scale, T_opt 36, no shock) gives mu_eff 0.020-0.038 but residuals in days 1-2 are more negative than in v1 (-0.015..-0.046 vs ~-0.01), hinting the faster ramp photo-shocks, or 37 C hurts low-T_opt strains. -> fit2.py fits mu_eff, T_opt and tau per batch.
- fit2.py (per-batch fit of mu_eff, T_opt in {34..38}, tau in {1,2.5,4}, with photo-shock) on b000-b019: rms 0.4-2.9% log. With v2 (light 1800 -> tank 36-37 C) most batches fit T_opt 34 -> the warm tank costs growth for some strains (or the model is optimistic above 35 C). T_opt/tau are not identifiable when T stays 35.
- cf.py (counterfactual replay of the fitted strains in mfsim): model reproduces v2 actual harvests within ~2-3%. Under fitted strains I_MAX 1600/1700/1800 -> 0.989/1.000/1.000 (relative); RAMP_UP 1.0 -> 0.991. The v2 settings would have given the two probe0 batches 14.6/17.8 g instead of 7.3/11.5 g.
- Decision: I_MAX 1700 (tank ~36 C: hedge against low-T_opt strains at no modelled cost). ctrl_v3 = v2 + I_MAX 1700 + optional F_PRE (partial harvest just before the drain). sweep4: F_PRE 0.15/0.3 and N_DRAIN 4 give no gain -> F_PRE 0.

### E4 (b020-b029) ctrl_v3 at extreme inocula: `--inoculum 30 --n 4`, `--inoculum 1500 --n 3`, `--inoculum 5000 --n 3` (label v3_i30/v3_i1500/v3_i5000)
Why: extremes of the plant range were untested (max so far 712): extinction risk at 30, nephelometer saturation and dense-culture behaviour at 1500/5000.
Result: 30 -> 13.4/8.5/3.0/6.9 g; 1500 -> 31.0/27.9/33.1 g; 5000 -> 67.0/67.5/72.3 g. None lost. 5000 is accepted (MAX_CELLS >= 5000).
Learned:
- BUG in b022 (slow strain, inoc 30): after two drain harvests the estimated OD fell below OD_MIN_HARVEST 0.25, so the last drain request was ~0 (37 mg harvested, ~4 g left in the tank). Fix (ctrl_v4): during the drain phase only refuse to harvest below OD 0.08 (~40+ agents, so a 0.5 cut leaves ~20 > 10 extinction limit).
- 1500: growth at OD ~2 under the cap was slow (b025: 1.9 -> 2.2 OD in 48 h with small harvests); turbidity/OD fell to ~130 (clump ~5-7).
- fit2 on b020-b029: dense batches fit lower mu_eff (1500: 0.020-0.026; 5000: 0.026-0.034) than the population mean ~0.031; clump model verified again on b025 (implied vs modelled clump within ~10%). pH rises only to ~10.0 at OD 2-3 (f_pH ~0.96), not enough to explain it; cause unresolved (possibly chance, 6 batches).
- cf.py on fitted strains with ctrl_v4: OD_CAP 1.2/1.5/3.0 -> 0.977/0.994/0.988; STIR_HI 95/110 above OD 2 -> 0.991/0.972. Keep OD_CAP 2, stir 75. The b022 fix recovers 3.0 -> 3.7 g in replay.
- sweep5 (model): I_MIN 150/500 and OD_MIN_DRAIN 0.05/0.15 flat.
- FINAL controller.py = ctrl_v4 (stir 75, light for path-mean PAR 500 capped at 1700 with 200 umol/h ramp, OD cap 2, drain 0.5 at 108/120/132 h). act() ~0.11 ms.

### E5 VALIDATION (b030-b149) final controller.py, plant inoculum distribution: `python plant_trial.py --run wb3 --privileged runs/wb3/work/controller.py --n 120 --label val`
Result (valsum.py val): all 120: median 18.9 g, p25 14.9 g, min 6.6 g, mean 21.8 g, 0 lost, 0 errors.
  inoc 30-99 (n14): median 11.0, p25 9.3 | 100-400 (n85): median 18.1, p25 15.1 | 401-999 (n2): 30.9 | >=1000 (n19): median 41.1, p25 35.5.
  Model (mfsim) forecast for the same distribution: p25 14.6 g -> plant agrees with the model.

### E6 VALIDATION extremes: `--n 16 --inoculum 30 --label val30`, `--n 6 --inoculum 5000 --label val5000`
Result: inoc 30 (n16): median 7.6 g, p25 6.0 g, min 3.4 g, 0 lost. inoc 5000 (n6): median 70.9 g, p25 67.1 g, 0 lost.
Safety margin: lowest agent count after 100 h in any validation batch = 55 (extinction at <10). Max true temperature 37.9 C (initial ambient draw), steady ~36 C at 1700 umol.

Budget: 172/300 used (8 calibration probes, 12 v2 pilot, 10 extreme-inoculum, 142 validation). 128 left unspent: further A/B tests of settings the model rates as flat would need hundreds of batches given the strain spread (mu_eff 0.020-0.041).
