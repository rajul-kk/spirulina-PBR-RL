# Lab notebook: run wb2 (white-box / privileged)

## Files read
- PLANT_MANUAL.md (root)
- plant_trial.py (root): 11 harvest events actually occur (t%600==0 for t=600..6600; step 7200 never runs,
  so the 132 h harvest is the LAST one and biomass grown 132-144 h is never harvested). Inoculum
  distribution: 10% 30-80 cells, 20% 600-5000 (log-uniform), 70% 100-400 (log-uniform). DIFFICULTY=2.
  Imports harness (_ctrl_obs, to_env_action, MAX_CELLS); harness.py itself is off-limits, not read.
- PPO_IBM/environments/genetic_env.py (full). Imports only numpy/gym, nothing from PPO_IBM/training.

## What the source says (my reading of genetic_env.py), key numbers
- Biomass is ~agents x mass; 1 agent ~12.5 mg at small inoculum (starting mass 1.25e8 units x 1e-7 mg),
  so OD0 = n0*12.5/20/300: 30 cells -> OD 0.06, 200 -> 0.42, 5000 -> ~9.2. Divide at 1.4e8.
- Extinction only if <10 agents or <1 mg. Lysis baseline 5e-4/h; respiration 0.0004/h (x2 in darkness).
  => with any light, cultures essentially cannot die except by over-harvesting a tiny population.
- Growth mu = mu_max(N(0.040,0.006)) x f_I x f_Q x f_P x f_C x tox x f_T x shock x f_O2 x f_pH x osm x repair x fatigue.
  * f_I: Haldane on RED light (Ks~100, Ki~2500 on TOTAL light) over an 8-point 6.7 cm light path, extinction
    0.20/0.25/0.06 m2/g for red/blue/green. Normalised so max=1 at path-mean total ~500 umol.
    Weighted between integrated (path-mean) and per-point response by w = 0.5..1 (50..200 rpm).
  * Clump shading: each agent's light x clump^-1/3. Clumps grow with prob od*0.05/h*(1-rpm/250); break
    only by shear above 80 rpm (0.5*((rpm-80)/120)^2*sqrt(c) per h) or weak Brownian. Children reset to 1.
  * repair tax: 1-0.35*sigmoid(0.12(rpm-100)) -> 0.99 at 70 rpm, 0.81 at 100, 0.60 at 200.
    fatigue: integrity loses 0.05/h*(rpm-80)/100, recovers 0.1/h; <=15% tax.
  * shock: exp(-3e-6*(I_pathmean - acclimated)^2), acclimation EMA tau 1-4 h, one-sided (only increases).
  * Temperature: light heats 0.001 C/h per umol, ambient loss 0.1/h*(T-25), chiller max 0.6 C/h.
    -> thermostat holds 35 C only while L <~ 1500; steady T = 25+(L/1000+stir_heat-0.6)/0.1 above that
    (1800 -> 37 C, 2000 -> 39 C). CTMI with Tmax 44.5 is steep above T_opt (N(36,1)): 39 C -> 0.92, 40 C -> 0.84.
  * pH-stat CO2 feed holds pH~10 (f_pH ~0.96); N/P auto-dosed; carbon, osmotic not limiting in practice.
- Harvest: Bernoulli removal of agents with p = mean requested fraction over the 12 h interval; fresh medium refill.
- Sensors (D2): +-2% jitter and a per-batch +-5% multiplicative drift on EVERY channel incl. pump_L and temp;
  turbidity = 250*od*(0.7+0.3*pigment)*clump^-1/3/(1+0.05 od) x (1+3%*rpm/200 noise); temp/pH lagged EMAs.
  Pigment only bleaches if path-mean light >1000 or N<75 (avoidable); it affects only the sensor.

## Own model
- model.py / sim.py: mean-field re-implementation (X, mean clump, acclimation, T with PI thermostat, membrane, DO).
  Replaying the probe logs (calib.py), the model tracks true growth with a constant ratio 0.75-1.0 (strain mu_max),
  with ~8%/OD-unit extra decline at high density that the model misses (calib2.py) -> dens_pen=0.08 correction.
- Model conclusions: light should follow the density-optimal intensity (path-mean ~ optimum), capped ~1800
  (thermal: 1500 -1.5%, 1950 -2%); stir 60-80 rpm (100 rpm -15%); productivity rises with density
  (light fully absorbed; respiration negligible) so moderate-high post-harvest OD (1.5-2) is best;
  last 3 harvests at max (kf=3) beat kf=2 by ~2-3%; periodic 2 h 200-rpm de-clumping bursts every 24 h +2-3%
  at OD>0.8.

## Experiments
| # | command (from root, all with --run wb2 --privileged) | why | result | learned |
|---|---|---|---|---|
| 1 | c01_probe.py --n 2 --label probe | calibrate model/sensors, time cost | b000 (47 cells) 2785 mg, b001 (34) 0 mg; ~10 s/batch | naive harvest rule missed final events -> always harvest max at the end. Model growth tracked truth (ratio 0.8-0.9). lux ~28/umol |
| 2 | c02_v1.py --n 6 --label v1 | first model-based controller (sp 1.2, kf 2, rpm 75) | 17.6, 19.5, 56.8 (3415 cells), 16.1, 7.5 (55), 6.9 (37) g | erratic harvests: BUG, pump_L is jittered +-2%/step, my step-differencing treated noise as harvests and shrank the OD estimate |
| 3 | c03_v2.py --n 6 --label v2 | pump bug fixed (use requested fraction at t%600==1) | 12.4 (128), 17.1 (193), 23.2 (757), 30.7 (1016), 11.4 (56), 10.4 (108) g | post-harvest true OD 1.0-1.3 vs setpoint 1.2: estimator OK. Clumping pulls turbidity/OD to ~0.6 at OD 1.5 |
| 4 | c04_v3.py --n 6 --label v3 | sp 1.7, kf 3, 2 h 200-rpm burst every 24 h (model: +2-3%) | 12.1 (149), 20.4 (327), 22.3 (357), 16.4 (118), 48.3 (2422), 7.1 (50) g | a burst lifts turbidity/OD 0.70->0.82 (clumps break), rebuilds over ~24 h; post-harvest OD 1.5-2.0 as intended |
| 5 | c04_v3.py --n 2 --inoculum 30; --n 2 --inoculum 5000 | extremes | 30: 8.0, 4.4 g; 5000: 69.3, 73.2 g; none lost | dense start: clumps reach ~5 in 24 h (model agrees), growth ~0 until diluted; turbidity clips at 1000 but harvest=0.5 anyway |
| 6 | c05_v4.py --n 8 --label v4 | clump-threshold bursts (C>2 -> 200 rpm until C<1.2, model +0-2%); temp trim moved 38->41 C (temp sensor drift is +-5% = +-1.9 C, a 38 C trim could cut light on a high-reading probe) | 11.3 (56), 54.1 (2967), 28.0 (372), 13.2 (116), 18.7 (107), 17.3 (128), 18.6 (225), 18.9 (276) g | fine; per-version comparisons on random inocula are too noisy (sd ~17%) |
| 7 | A/B at --inoculum 150, n=8 each: c05b_v4 (=v4), c06_noburst, c07_sp12, c08_L1600 | test model's key choices | raw means 17.2 / 15.1 / 16.8 / 16.5 g | ANCOVA (ancova.py) on the strain's early growth rate (true OD 3-30 h, same policy for all arms there) cuts residual sd to 4%: noburst -0.3+-2.1%, sp1.2 -5.6+-2.0%, L1600 -2.8+-2.0% vs v4 |
| 8 | A/B inoculum 150: v4 +4, c09_sp23 n6, c10_sp30 n6, c11_L2000 n6 | bracket setpoint & light cap | adj. vs v4: sp2.3 -2.9+-2.2%, sp3.0 -1.5+-2.2%, L2000 -6.0+-2.2% (noburst -1.5, sp1.2 -6.9, L1600 -4.0 after pooling) | density setpoint plateau 1.7-3.0 (1.2 clearly worse); light cap 1800 (T~37 C) is the peak: 1600 and 2000 both lose 4-6% |
| 9 | A/B inoculum 150: c12_kf2 n6, c13_kf4 n6 | end-game drawdown | kf2 -8.7+-2.2%, kf4 -0.9+-2.2% | 3 final max harvests is right (matches value-of-stock argument: keeping 1 mg after the 108 h harvest returns ~1.0 mg) |

Decision after exp 9: keep v4 settings (rpm 75, Lmax 1800, sp 1.7, kf 3, clump bursts 2.0/1.2).
Added an extinction guard (never harvest below ~60 estimated filaments; est. filaments = 1.9 x mg/L) and an OD
estimate floor. Smallest final population seen so far: 107 filaments (30-cell inoculum, slow strain).
| 10 | controller.py (= c14_final.py) --n 100 --label val | held-out-style validation, natural inoculum mix | all: median 20.7 g, p25 16.5 g, min 9.0 g, 0 lost, 0 errors. 30-80: med 11.3/p25 9.7 (n8); 100-199: 17.8/14.1 (n33); 200-400: 21.7/20.2 (n35); 401-999: 26.2/23.9 (n3); 1000-2499: 36.8/34.3 (n12); 2500+: 60.1/55.6 (n9) | spread within an inoculum class is mostly strain growth rate (mu_max sd 15%) |
| 11 | controller.py --n 10 --inoculum 30 --label valx30; --n 6 --inoculum 5000 --label valx5000 | worst-case extremes | 30: median 7.5 g, p25 6.5, min 4.3, 0 lost; 5000: median 73.9, p25 72.6, 0 lost | fewest filaments during the harvest phase in any validation batch: 104 (30-cell start) -> ample margin over the extinction limit of 10; max true temp 37.9 C |

Budget: 214/300 used; 86 left unused (interrupted by an API rate limit after exp 10, resumed; stopped since validation
was complete and remaining A/B deltas were within noise).
