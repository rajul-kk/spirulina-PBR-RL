# Lab notebook: run wb1 (white-box controller writer)

## Files read
- blackbox/PLANT_MANUAL.md (operator's manual)
- blackbox/plant_trial.py (trial tool: 11 harvest events at t=600..6600 steps, i.e. 12..132 h;
  culture_lost = episode ended before 7200 steps; D2 difficulty; lab assay = true OD*300 +-5%)
- PPO_IBM/environments/genetic_env.py (full; imports only numpy/gym, nothing from training/)

## Design-model notes from genetic_env.py
- Each agent ("cell") ~1.25e8 mass units = 12.5 mg DW (1e-7 mg/unit). OD = mass_mg/20 L/300.
  Inoculum N cells -> OD ~ N*12.5/6000: 30 cells OD 0.06, 300 cells OD 0.6, 5000 cells OD ~9
  (starting mass shrinks slightly with inoculum: 1.25e8 - 0.45e8*N/15000).
- Extinction: num_active < 10 or mass < 1 mg. Only mechanisms: harvest (per-cell Bernoulli with
  prob frac), lysis (baseline 5e-4/h, up to 2.5e-3/h when mu < respiration), starvation (cell mass
  < 1e7, i.e. long darkness). Respiration is only 1% of mu_max (2x in the dark). So a small culture
  is only at risk if I harvest it while it is still small.
- Harvest: at steps 600,1200,...,6600 (11 events, last at 132 h; the final 12 h of growth is
  never harvested). Fraction applied = MEAN of requested fractions over the preceding 600 steps.
  Standing biomass at the end counts for nothing.
- Growth mu = mu_max(N(0.040,0.006)) * f_I * f_Q * f_P * f_C * f_CO2tox * f_T * shock * f_O2 * f_pH
  * f_osm * repair_tax * fatigue_tax.
  - Light: spectral Beer-Lambert over 6.7 cm path, k_red = 0.5 + 0.20*X(mg/L) + 0.004*rpm (1/m).
    Haldane on red (40% of I) with total-light inhibition, Ks_light~100, Kii~2500, normalised so
    the max is 1 at total I = sqrt(Ks*Ki) = 500. Path-mean (integration) weight rises linearly
    from 0.5 at 50 rpm to 1.0 at 200 rpm (flashing-light effect).
  - Clump self-shading factor clump^-1/3 on each cell's light; clumping rate 0.05/h*OD*(1-rpm/250),
    shear breakup only above 80 rpm ((rpm-80)/120)^2.
  - repair_tax = 1-0.35*sigmoid(0.12*(rpm-100)): 35% growth penalty at high rpm, ~0 below 70 rpm.
  - fatigue_tax up to 15% for sustained rpm > 80.
  - shock = exp(-3e-6*max(pathmean_I - acclimated_I,0)^2), acclimation EMA tau 1-4 h: only light
    INCREASES (in path-mean terms) hurt; a harvest (dilution) raises the path-mean light too.
  - Temperature: CTMI, T_opt ~N(36,1), Tmax 44.5. Thermostat holds 35 C unless light heat exceeds
    chiller: steady T ~ 25 + (0.001*I - 0.6)/0.1 above I~1600 (I=2000 -> ~39 C, f_T ~0.92).
- Turbidity = 250*OD*(0.7+0.3*pigment)*clump^-1/3/(1+0.05*OD)*(1+-3%*rpm/200 noise)*drift(0.95-1.05)
  *jitter(+-2%). Pigment bleaches if path-mean cell light >1000 or N<75.

## Offline design calculations (no batches)
- growth_calc.py: static growth-rate vs OD/light/rpm from the equations. Findings: optimal
  surface light rises with OD (~550 at OD 0.05, ~1150 at OD 0.5, ~1700-1850 above OD 1);
  volumetric productivity mu*X keeps rising with OD up to ~OD 5 (ignoring clumps and O2), so the
  manual's "OD 0.75" target is far below the productivity optimum. rpm 50-80 best; above 90 the
  shear repair tax (up to -35%) dominates.
- mfmodel.py: my own reduced mean-field ODE of the plant (X, mean clump, acclimation, membrane,
  temperature/thermostat, 2-layer DO). Carbonate/pH/quota replaced by a constant factor 0.786.

## Exp 1 (batches b000-b003, label v0): baseline probe
- cmd: python plant_trial.py --run wb1 --privileged runs/wb1/work/ctrl_v0.py --n 4 --label v0
- why: first look at real dynamics; calibrate my model. ctrl_v0: rpm 70, light 450+1400*OD_est
  (cap 1800, ramp 200/h), harvest toward OD 2 using naive turbidity/250, 0.5 bleed from event 9.
- result: inoc 269/372/116/1179 -> 22764/16377/13348/33567 mg; no losses.
- learned:
  * Turbidity badly under-reads at 70 rpm because of clumping: turb/ideal fell from ~0.95 to ~0.5
    as OD rose (b003: turbidity flat at ~450-490 NTU while true OD went 2.4 -> 4.6). So the naive
    estimate never saw OD>2 and the target-OD harvest never fired; all yield came from the bleed.
  * Temperature 37 C at 1800 umol as predicted.
  * Cells divide in synchronized waves (269 -> 535 -> 1046 -> 2041), max cells seen 3556 (no
    slot cap reached).
  * calib.py replays logged actions through mfmodel: with a single fitted mu_max per batch
    (0.028-0.046, k=0.786) the model tracks true OD with 1-3% log-RMSE, and its clump factor
    c^-1/3 matches turbidity/ideal within ~5% (the sensor drift). Model validated.

## Offline policy search on mfmodel (no batches)
- opt2.py (oracle OD, coordinate search over rpm, light schedule, ramp, target OD, bleed start;
  inocula 40..3000, mu_max 0.032/0.042): extremely flat optimum around v0's settings: rpm 70,
  I = min(1800, 450+1400*OD), target OD ~2, bleed (0.5 per event) from event 9 (108 h).
  Changes of any single knob moved the log-sum score by <1%. Ramp-limiting light gave no gain.
- controller_v1.py: v0 + clump observer (mean clump integrated from stick/breakup and dilution by
  growth, growth rate from the design light model) so OD_est = turbidity corrected by c^-1/3;
  deadbeat harvest (the requested fraction is set so that the interval MEAN hits the target
  computed from the projected pre-harvest OD); post-harvest OD floor 0.15 (extinction guard).
  simtest.py (my model with synthetic sensors): estimate within ~2-5% of true OD, clump tracked.
- tune.py on v1: odt 1.5-3.0 x bleed 8-10 all within 0.5% (odt 2, bleed 9 best).

## Exp 2 (b004-b011, label v1): controller_v1, random inocula, n=8
- cmd: python plant_trial.py --run wb1 --privileged runs/wb1/work/controller_v1.py --n 8 --label v1
- result: inoc 135/58/166/340/221/184/196/159 -> 14596/7368/17673/22596/17390/17541/20831/14840 mg,
  no losses. The observer works: post-harvest true OD held at 1.85-2.1 vs target 2.0.

## Exp 3 (b012-b019): v1 at the extremes, inoculum 30 (n=4) and 5000 (n=4)
- result: 30 cells -> 10611/7615/6973/6031 mg (no losses; growth-limited, first harvest at 108 h).
  5000 cells -> 66142/73610/70447/75458 mg. After dumping (0.5 at events 1-2) the dense start had
  built large clumps (turb/ideal ~0.5, c~7) and the hold at OD ~2 grew only ~0.008/h.

## Offline: high stir for dense cultures
- tune.py (controller_v2 = v1 + rpm_hi above od_rpm): rpm 200 above OD 3 gave +3.5% at inoculum
  5000 in the model, nothing elsewhere; clump-triggered switching was worse.

## Exp 4 (b020-b023, v2, inoculum 5000, n=4)
- result: 73570/76640/72708/73908 mg (mean 74.2k vs 71.4k for v1; +4%, as predicted).

## Offline: v3 = odt 3.0 + 200 rpm above OD 2.5 (hysteresis 0.3)
- model: +4.6% at inoc 1000, +1% at 5000, ~+1% at 250, 0 at 40.

## Exp 5 (b024-b027, v3, inoculum 1000, n=4)
- result: 28422/28600/31461/39912 mg. Looked low, so I checked with a better replay (calib.py now
  forces the model's harvests to the logged harvested mg): fitted mu_max 0.032/0.032/0.036/0.052,
  i.e. three slow strains. Model log-RMSE 0.5-1.6% in the 200 rpm regime too (clumps fully
  broken, turb/ideal ~0.95). Refit of all regimes (b000,b003,b007,b016,b020,b021,b024-27):
  RMSE 0.5-1.9% with mu_max as the only free parameter (range 0.028-0.052, mean ~0.039).
  Conclusion: the model is trustworthy, so policy tuning is done in-model; pilot batches are for
  validation.

## Offline: coordinate search around v3 (coord.py; inocula 40/250/1000/5000, mu 0.03/0.04/0.05)
- every knob at its local optimum: imax 1800, ib 1400, ia 450, odt 3.0, od_rpm 2.5, rpm_hi 200,
  rpm 70, bleed 9. bleed 9 also best for inocula 30/60/120.
- controller.py = controller_v3 with documentation. act() ~0.01 ms/step.

## Exp 6 validation of the final controller.py (b028-b147; all runs with the same file)
- cmds (from blackbox/): plant_trial.py --run wb1 --privileged runs/wb1/work/controller.py
  --n 40 --label val ; --n 16 --inoculum 30 --label val30 ; --n 8 --inoculum 5000 --label val5000 ;
  --n 8 --inoculum 100 --label val100 ; --n 8 --inoculum 400 --label val400 ; --n 40 --label val2
- stats.py results (harvest mg; no culture lost, no controller errors in any of the 148 batches):
  | set | n | median | p25 | min |
  |---|---|---|---|---|
  | random inocula (val+val2) | 80 | 17652 | 14329 | 6430 |
  |   of which 30-79 cells | 10 | 10957 | 8519 | 6430 |
  |   of which 100-399 cells | 63 | 17835 | 15170 | 7068 |
  |   of which 600-1999 | 3 | 38798 | 33851 | 28904 |
  |   of which 2000-5000 | 4 | 50447 | 48187 | 41937 |
  | fixed 30 cells | 16 | 7911 | 4299 | 2780 |
  | fixed 100 cells | 8 | 13895 | 13238 | 11889 |
  | fixed 400 cells | 8 | 25558 | 22792 | 20811 |
  | fixed 5000 cells | 8 | 73511 | 71456 | 67309 |
- Extinction margin: at inoculum 30 the minimum live cell count after 100 h was 83 (loss is <10),
  thanks to the OD 0.15 post-harvest floor. Lowering the floor to 0.10 was worth only +0.5% in the
  model for the smallest cultures, so I kept 0.15.
- Worst mid-size batch b111_val2 (126 cells, 7069 mg): a very slow strain (true growth ~0.015/h
  even at optimal light), not a control fault.
- The spread within an inoculum class is dominated by strain mu_max (fitted 0.026-0.052).

## Budget: 148 of 300 batches used (4 probe, 8+8 v1, 4 v2, 4 v3, 120 validation). Rest unused:
the model is validated and the optimum is flat, so further A/B tests (effects ~1%) would be
below the strain-to-strain noise.
