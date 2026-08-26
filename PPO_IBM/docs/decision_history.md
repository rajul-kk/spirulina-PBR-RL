# Decision history

Relocated comment blocks (originally >2 lines) from source files, verbatim. Each source location keeps a 1-line summary plus a pointer here; nothing was deleted, only moved.

## ./bc/bc_pretrain.py:45 {#--bc-bc_pretrain-py-45}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./bc/bc_pretrain.py:85 {#--bc-bc_pretrain-py-85}

```
# HARVEST IS A FEEDBACK LAW, NOT A CONSTANT.
#
# A constant harvest fraction was the original plan, based on a sweep that scored
# frac=0.18 at 139.9mg / time_avg_od 0.0131 / reward 1116.8. That sweep ran on the BARE
# env at a FIXED initial_cells=300. Measured through the real training stack — where
# CurriculumStartWrapper draws initial_cells log-uniformly (100-400 "low", 600-1500 "mid",
# 2000-5000 "high") — the identical constant action produced:
#
#     30.7, 41.3, 46.3, 57.3, 63.4, 93.3, 95.8, 186.0, 248.3, 347.8, 368.2  mg
#     (median ~93mg, a 12x spread, time_avg_od 0.0025-0.0326)
#
# i.e. the episode outcome is dominated by the cold-start draw, not by the action. A
# constant fraction strips a small starting culture before it can establish (46mg at
# od 0.0054) while under-harvesting a large one (368mg at od 0.0326). Cloning a
# state-independent constant would therefore teach the policy to IGNORE its observation
# on roughly half of all episodes — actively wrong, and a poor foundation for fine-tuning.
#
# Instead the expert harvests the SURPLUS above an OD setpoint, proportionally:
#     frac = clip(GAIN * (od / OD_SETPOINT - 1), 0, FRAC_CAP)
# Below setpoint it harvests nothing and lets the culture build; above it, it removes
# roughly the excess. This drives time_avg_od toward OD_SETPOINT regardless of where the
# episode started, which is exactly what the curriculum's time_avg_od criterion rewards.
#
# OD_SETPOINT sits above the D2 gate's time_avg_od>=0.011 with margin, and above
# genetic_env's OD_TARGET=0.012 (the peak of reward_od), so the controller holds the
# culture in the band the reward function itself pays most for.
```

## ./bc/bc_pretrain.py:118 {#--bc-bc_pretrain-py-118}

```
# Down-weights the value objective relative to the action objective during BC. Returns are
# O(100s) and action targets O(1), so an unweighted sum lets the critic dominate the shared
# trunk and degrade the actor clone.
```

## ./bc/bc_pretrain.py:178 {#--bc-bc_pretrain-py-178}

```
            # Feedback law: read the CURRENT od off the raw env each step, so the demo
            # adapts to how the culture is actually developing rather than replaying a
            # fixed number. This is the whole point of the redesign.
```

## ./bc/bc_pretrain.py:188 {#--bc-bc_pretrain-py-188}

```
            # Fix #14: record the NORMALIZED reward stream so discounted returns can be
            # computed for value-function pretraining. vec_env has norm_reward=True, so
            # reward[0] is already on the same scale the critic will be trained against
            # during PPO — using raw env rewards here would produce a critic calibrated to
            # the wrong magnitude, which is worse than no pretraining at all.
```

## ./bc/bc_pretrain.py:208 {#--bc-bc_pretrain-py-208}

```
        # Read the episode outcome from the terminal INFO dict, not off the raw env:
        # DummyVecEnv auto-resets on done, which zeroes cumulative_harvested_mg before
        # any post-loop attribute read can see it. Same convention held_out_sweep.py and
        # deterministic_eval.py already use.
```

## ./bc/bc_pretrain.py:308 {#--bc-bc_pretrain-py-308}

```
            # NB: predict_values returns a BARE tensor (sb3_contrib policies.py:280-285), not
            # the (value, states) tuple that get_distribution returns — unpacking it would
            # silently split along the batch dimension instead.
```

## ./bc/bc_pretrain.py:314 {#--bc-bc_pretrain-py-314}

```
            # VALUE_LOSS_COEF keeps the critic from dominating the shared trunk: returns are
            # O(100s) while action targets are O(1), so an unweighted sum would let the value
            # objective swamp the actor gradients and degrade the clone we actually need.
```

## ./bc/bc_pretrain.py:330 {#--bc-bc_pretrain-py-330}

```
    # ── Critic-only refinement ────────────────────────────────────────────────────────
    # The joint phase above deliberately throttles the value gradient by VALUE_LOSS_COEF
    # (0.001) to stop the O(100s) return targets from dragging the shared trunk and degrading
    # the actor clone. The side effect is an under-fit critic: the joint phase ends with value
    # MSE still falling steeply. Since an uncalibrated critic is exactly what Fix #14 exists to
    # prevent, refine it here at FULL weight with the actor's own parameters frozen — the value
    # head cannot damage what it can no longer move.
```

## ./bc/bc_pretrain.py:397 {#--bc-bc_pretrain-py-397}

```
    # Expected shape for the FEEDBACK expert: episode-mean frac ~0.04-0.15 (it varies with
    # how big the culture gets), first600 near ZERO while the culture establishes, rising
    # thereafter. A high-then-decaying profile would be v16b's failure mode returning.
```

## ./bc/bc_pretrain.py:482 {#--bc-bc_pretrain-py-482}

```
    # Refuse to clone an expert that cannot clear the tier the curriculum starts working
    # toward. Cloning a sub-D1 expert guarantees a sub-D1 policy, and the 8M-step run that
    # would follow could only confirm that at great cost.
```

## ./diagnostics/check_training.py:2 {#--diagnostics-check_training-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/curriculum_gate_sweep.py:15 {#--diagnostics-curriculum_gate_sweep-py-15}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/dynamic_profile_sweep.py:15 {#--diagnostics-dynamic_profile_sweep-py-15}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/dynamic_profile_sweep_od.py:10 {#--diagnostics-dynamic_profile_sweep_od-py-10}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/dynamic_profile_sweep_od.py:34 {#--diagnostics-dynamic_profile_sweep_od-py-34}

```
# Two operating points: the reference one used for the original sweep, and the one the v15
# trained policy actually converged to (measured via test_actions.py on both v15 archives:
# stir 55-63rpm — near the 50rpm floor — and light 875-930umol).
```

## ./diagnostics/evaluate_agent.py:2 {#--diagnostics-evaluate_agent-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/fouling_feasibility.py:24 {#--diagnostics-fouling_feasibility-py-24}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/held_out_sweep.py:16 {#--diagnostics-held_out_sweep-py-16}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/held_out_sweep.py:68 {#--diagnostics-held_out_sweep-py-68}

```
        # Fix #23 (v25): `stochastic` samples actions instead of taking the distribution
        # mean. Needed so a run gated on stochastic rollouts can be VALIDATED on the same
        # policy it was gated on. Gating one way and validating the other is what produced
        # the v14 and v17 false positives; the fix is consistency, not loosening.
```

## ./diagnostics/noise_sensitivity.py:24 {#--diagnostics-noise_sensitivity-py-24}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/reward_ab.py:21 {#--diagnostics-reward_ab-py-21}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/reward_breakdown.py:12 {#--diagnostics-reward_breakdown-py-12}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/tdmpc2_cost_probe.py:38 {#--diagnostics-tdmpc2_cost_probe-py-38}

```
    # Values chosen to span this domain's actual measured magnitude: per-block rewards and
    # bootstrapped Q values are order single-to-double-digits (see the vmin/vmax comment in
    # TD_MPC2.py), not the thousands the wider Dreamer-style range would target.
```

## ./diagnostics/tdmpc2_cost_probe.py:45 {#--diagnostics-tdmpc2_cost_probe-py-45}

```
    # Decode the ENCODING directly via _expected_value (skipping decode()'s softmax, which is
    # only correct for raw network LOGITS — applying it to an already-normalised distribution
    # distorts it and is not a fair round-trip check; this was a real bug in an earlier version
    # of this test that masked the vmin/vmax miscalibration this file's git history fixed).
```

## ./diagnostics/test_actions.py:15 {#--diagnostics-test_actions-py-15}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/test_actions.py:42 {#--diagnostics-test_actions-py-42}

```
# ── Action decoding constants ─────────────────────────────────────────────────
# Harvest fraction range (0, F_MAX) matches genetic_env.py's F_MAX (0.5); only actually
# applied every HARVEST_INTERVAL_STEPS=600 steps, ignored on other steps. Kept as a
# literal here since this script builds the env fresh each run.
```

## ./diagnostics/test_heavy.py:2 {#--diagnostics-test_heavy-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/test_why_die.py:2 {#--diagnostics-test_why_die-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/zombie_diagnosis.py:15 {#--diagnostics-zombie_diagnosis-py-15}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./diagnostics/_verify_envs.py:2 {#--diagnostics-_verify_envs-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./environments/alpha_env.py:43 {#--environments-alpha_env-py-43}

```
        # Observation Space (6 Dims)
        # 1. Turbidity  2. pH  3. Nutrients  4. Temp  5. Conductivity  6. RGB
        # Note: dissolved_co2 is internal state only — agent reads it indirectly via pH
        # DO2 remains internal state in this observation design.
```

## ./environments/alpha_env.py:99 {#--environments-alpha_env-py-99}

```
        # --- DAY/NIGHT CYCLE ---
        # lights_off_hour: hour of day (0-24) when lights turn off. None = always on (default).
        # lights_on_hour:  hour of day (0-24) when lights come back on.
        # Example: lights_off_hour=20, lights_on_hour=6  ->  14h light / 10h dark.
```

## ./environments/alpha_env.py:143 {#--environments-alpha_env-py-143}

```
        # Super-Agent Scaling: 1 Agent = 2,500,000 Cells (~500pg each)
        # Density-dependent starting mass:
        # At 300 cells, mass is ~1.25e8. At 15,000 cells (Log Ladder limit),
        # mass drops to ~0.8e8 (starving) due to immediate shelf-shading/nutrient competition.
```

## ./environments/alpha_env.py:269 {#--environments-alpha_env-py-269}

```
        # Dynamic RPM-coupled sensor lag for pH and temperature (D1+ only).
        # Better mixing (high RPM) -> faster response (~2-step effective lag).
        # Poor mixing (low RPM) -> slower response (up to ~8-step effective lag).
```

## ./environments/alpha_env.py:379 {#--environments-alpha_env-py-379}

```
        # --- Biofouling Accumulation ---
        # Cells adhere to surfaces at low mixing and high biomass density.
        # exp(-0.5) ≈ 60% light transmission at full fouling (cap 0.5).
```

## ./environments/alpha_env.py:387 {#--environments-alpha_env-py-387}

```
        # --- Physics (Chaotic Turbulence) ---
        # Apply only to active cells
        # We replace simple Brownian motion with structured "Swirls"
        # Flow V(z, t) = Sum( A * sin(k*z - w*t) )
```

## ./environments/alpha_env.py:400 {#--environments-alpha_env-py-400}

```
            # 1. Aggregation (Sticking) - Orthokinetic + Perikinetic
            # Orthokinetic: Stirring INCREASES collision frequency (Smoluchowski)
            # Sticking = Base (Brownian) + Shear-Induced (RPM)
            # Fix scaling bug: Use true physical OD, not an assumption based on cell count!
```

## ./environments/alpha_env.py:406 {#--environments-alpha_env-py-406}

```
            # --- New Physics: 
            # - Base chance: 1e-3 (Stronger - 5x boost)
            # - RPM Boost: Increases linearly with Mixing (more collisions)
            # --- Flocculation: Stirring BREAKS clumps (shear dispersal dominates at moderate RPM)
            # rpm_factor now REDUCES sticking — higher RPM = more shear = less aggregation
            # At 0 RPM: rpm_factor=1.0 (max sticking). At 200 RPM: rpm_factor=0.2 (80% less sticking)
```

## ./environments/alpha_env.py:425 {#--environments-alpha_env-py-425}

```
            # Brownian/diffusive breakup (always active, weak)
            # Prevents runaway aggregation at low RPM
            # Small clumps (1-5) barely affected, large clumps (50+) slowly erode
```

## ./environments/alpha_env.py:441 {#--environments-alpha_env-py-441}

```
            # --- 2D Kinematic Turbulence (Airlift / Convection Loop) ---
            # Center (x=0.5): Upward Flow (-z)
            # Walls (x=0,1): Downward Flow (+z)
            # Top/Bottom: Turnaround (Horizontal Flow)
```

## ./environments/alpha_env.py:449 {#--environments-alpha_env-py-449}

```
            # 1. Vertical Velocity (Vz)
            # Cosine profile: Max Up at 0.5, Max Down at 0, 1.
            # Scale: 0.01 m/s * intensity
```

## ./environments/alpha_env.py:517 {#--environments-alpha_env-py-517}

```
        # 1. Shear Stress (RPM > 400)
        # Random death probability for cells if mixing is too violent
        # Note: We already have this logic downstream at line 550, but let's keep the flow clean.
        # Actually, let's just fall through to the Biology block.
```

## ./environments/alpha_env.py:528 {#--environments-alpha_env-py-528}

```
            # 1. Spectral Light Field (RGB Physics)
            # Action 'light' sets Total Surface Intensity (PAR)
            # I_surface is already calculated at top of step()
```

## ./environments/alpha_env.py:541 {#--environments-alpha_env-py-541}

```
            # Attenuation Coefficients (k)
            # Red: Absorbed STRONGLY by Chlorophyll (Growth)
            # k_red boosted to 3.5 (was 10.0) to allow deep biological growth past 12k cells
```

## ./environments/alpha_env.py:554 {#--environments-alpha_env-py-554}

```
            # ── Turbulent Flash-Light Effect (Biologically Accurate) ──────────
            # In real Spirulina PBRs, turbulent mixing causes cells to cycle
            # between the photic zone (surface) and dark zone (deep) rapidly.
            # This "flash-light effect" dramatically increases photosynthetic
            # efficiency (Kok effect): brief intense surface flashes > sustained dim light.
            #
            # At 0 RPM  : cells see only their actual static depth (fully stratified).
            # At 500 RPM: cells see a near-random depth distribution each step (fully mixed).
```

## ./environments/alpha_env.py:588 {#--environments-alpha_env-py-588}

```
            # --- Photo-Acclimation (Hysteresis) ---
            # Cells adapt to the TOTAL light they see
            # k_accum = 0.1 (Fast integration)
```

## ./environments/alpha_env.py:597 {#--environments-alpha_env-py-597}

```
            # Photo-Inhibition / Shock
            # Cells experience stress when light changes suddenly
            # Scalar reduced from 0.0001 to 0.000001 to prevent startup death
            # At diff=300: Old penalty=99.99%, New penalty=9% (survivable!)
```

## ./environments/alpha_env.py:612 {#--environments-alpha_env-py-612}

```
            # 3. Growth Rate (Haldane)
            # Growth is driven by RED light availability
            # Inhibition is driven by TOTAL light intensity
            # f_I = I_growth / (Ks + I_growth + I_total^2/Ki)
```

## ./environments/alpha_env.py:640 {#--environments-alpha_env-py-640}

```
            # pH Inhibition (Asymmetric Gaussian — Spirulina alkaliphile)
            # Peak at 9.5 (true optimum per Richmond 1988; Vonshak 1997; Habib FAO 2008)
            # Acid side: steep drop (σ=1.2) — Spirulina intolerant of low pH
            # Alkaline side: gentle drop (σ=2.0) — obligate alkaliphile tolerates high pH well
```

## ./environments/alpha_env.py:673 {#--environments-alpha_env-py-673}

```
            # Calculate Rate
            # --- Shear Repair Tax (sigmoid, centered at 175 RPM) ---
            # Steep onset above 175 RPM matches real shear fragmentation threshold.
            # Max 35% penalty at sustained >200 RPM.
```

## ./environments/alpha_env.py:680 {#--environments-alpha_env-py-680}

```
            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Hydrodynamic shear stress on Spirulina trichomes accumulates over time.
            # Onset at ~80 RPM (Kolmogorov eddy scale for 30L tank).
            # ~5h time constant: equilibrium integrity ≈ 0.5 at sustained 150 RPM.
            # 15% max growth penalty at full degradation (0.0 integrity).
```

## ./environments/alpha_env.py:696 {#--environments-alpha_env-py-696}

```
            # Carbon-Limited Growth (Monod saturation on dissolved CO2)
            # At dissolved_co2 = 0.5 mg/L: f_carbon = 0.50
            # At dissolved_co2 = 2.0 mg/L: f_carbon = 0.80
```

## ./environments/alpha_env.py:725 {#--environments-alpha_env-py-725}

```
            # --- PROBABILISTIC LYSIS DEATH (replaces dead-code hard starvation check) ---
            # Background lysis: ~0.5%/day (realistic Spirulina batch culture baseline).
            # Stress lysis: scales up to ~5%/day when mean current_mu < m_respiration.
            # Never a hard cliff — always a smooth gradient signal for the RL agent.
```

## ./environments/alpha_env.py:781 {#--environments-alpha_env-py-781}

```
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e8 pg (12% above init mass of 1.25e8)
            # Use >= to avoid edge case where cells hover exactly at boundary.
```

## ./environments/alpha_env.py:838 {#--environments-alpha_env-py-838}

```
        # 2. Gas Exchange (O2 & CO2)
        # Closed-tank model: gas composition is set by baseline air + injected pure CO2.
        # kLa scales with agitation, gas throughput, and broth resistance at high biomass.
        # Bug fix: Use cached self.od instead of calling _get_obs() which
        # would corrupt the pH lag buffer by appending mid-step.
```

## ./environments/alpha_env.py:863 {#--environments-alpha_env-py-863}

```
        # Dissolved Oxygen Dynamics
        # Production: Proportional to Growth (approx 1.5g O2 per g Biomass)
        # Respiration: Proportional to maintenance (approx 1.0g O2 per g Biomass lost)
        # Calculate net biomass change from biology step (approx)
```

## ./environments/alpha_env.py:897 {#--environments-alpha_env-py-897}

```
        # 3. DIC-Driven pH (alkaline carbonate model)
        #
        # O1: In alkaline Zarrouk medium (pH 9.5-11), virtually all injected CO2 converts
        # to bicarbonate/carbonate immediately.  The equilibrium DIC scales with CO2 partial
        # pressure but with square-root dampening from the high-alkalinity buffer capacity.
        # At atmospheric CO2: co2_sat ≈ 2 mg/L DIC.  At max injection (~29% CO2): ≈ 52 mg/L.
```

## ./environments/alpha_env.py:915 {#--environments-alpha_env-py-915}

```
        # O2: pH driven by Henderson-Hasselbalch — replaces the ad hoc blend.
        # Higher DIC → more bicarbonate → pH drops from alkaline baseline.
        # No CO2 + active photosynthesis → DIC depletes → pH rises naturally.
```

## ./environments/alpha_env.py:949 {#--environments-alpha_env-py-949}

```
        # OD ~ Mass^0.8 (Self-Shading effect)
        # 1e11 cells ~ 1g/L ~ OD 1.0
        # density_gL = (total_mass_mg * 1e-9) / self.volume_L # BUG: 1e-9 is wrong units (pg->mg happened already)
        # turbidity = 1.0 * (density_gL ** 0.8)
```

## ./environments/alpha_env.py:960 {#--environments-alpha_env-py-960}

```
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # At D2, simulate slow natural evolution/mutation of the strain over weeks of deployment.
        # Every ~5 hours (250 steps at dt=0.02h), strain parameters wander by ±1%.
        # This forces the internal LMU to constantly adapt its latent state tracking.
```

## ./environments/alpha_env.py:980 {#--environments-alpha_env-py-980}

```
        # --- Reward (Sim2Real Proxy Tuning) ---
        # In a real physical PBR, we cannot measure true mass on every step.
        # The RL agent must learn to optimize growth using *only* high-frequency sensor proxies.
```

## ./environments/alpha_env.py:984 {#--environments-alpha_env-py-984}

```
        # 1. DO2 Production Proxy (Photosynthesis)
        # We calculated 'o2_production' on line 602 as the raw biological O2 exhaust.
        # In reality, the agent reads `self.do2` and subtracts expected `k_La` off-gassing.
        # Here we directly use the simulated O2 production to train the proxy behavior.
```

## ./environments/alpha_env.py:995 {#--environments-alpha_env-py-995}

```
        # 2. pH Drift Proxy (Carbon Uptake)
        # If CO2 is OFF, and pH rises, cells are actively growing.
        # Base pH drift without biology is 0. Biological growth adds delta_ph_bio (calculated line 629)
        # Keep legacy magnitude while making the scale explicitly volume-normalised.
```

## ./environments/alpha_env.py:1003 {#--environments-alpha_env-py-1003}

```
        # 2b. Carbon transfer progress proxy (fractions-aware)
        # Gives short-horizon credit when dissolved CO2 moves toward a soft target band,
        # helping credit assignment through gas-fraction and kLa transfer delays.
```

## ./environments/alpha_env.py:1061 {#--environments-alpha_env-py-1061}

```
        # Proportional stagnation: severity scales with how much mass is lost, not binary.
        # Floor of 0.1 ensures any decline is noticed; ceiling of 1.0 caps at -0.15 for large crashes.
        # Threshold 0.5 mg/step ≈ 1.3% of a healthy 300-agent culture at full mass.
```

## ./environments/alpha_env.py:1083 {#--environments-alpha_env-py-1083}

```
        # 5. Potential-Based Reward Shaping (PBRS)
        # Φ(s) = future growth capacity — maintained nutrient/DIC buffers + O2 headroom + quota
        # F(s,s') = γΦ(s') - Φ(s)  — provably policy-invariant, reshapes landscape without bias
```

## ./environments/alpha_env.py:1102 {#--environments-alpha_env-py-1102}

```
        # ── Phase 0 Soft Reward Shaping (Difficulty 0 only) ──────────────────
        # At Difficulty 0, add small Gaussian bonuses proportional to sensor
        # proximity to known Spirulina optima. This scaffolds early learning by
        # giving the agent dense micro-signal without a single correct answer.
        # These bonuses are DISABLED at Difficulty 1+ so the agent generalises.
```

## ./environments/alpha_env.py:1125 {#--environments-alpha_env-py-1125}

```
            # Weaker pH shaping persists at D1/D2 so agent never fully loses
            # the pH→growth gradient signal across curriculum phases.
            # DO2 and temp shaping are omitted: O2 toxicity (f_O2) and f_pH
            # in the growth model already provide implicit signals for those.
```

## ./environments/genetic_env.py:9 {#--environments-genetic_env-py-9}

```
# Per-step physics trace, OFF by default. See the gate at the `[EnvDebug]` print for why:
# it accounted for 43% of every training log in this project. Turn on when debugging physics:
#     ENV_DEBUG=1 python diagnostics/dynamic_profile_sweep_od.py
```

## ./environments/genetic_env.py:37 {#--environments-genetic_env-py-37}

```
        # --- Gas-Phase / Carbonate Configuration (closed 20L PBR) ---
        # No CO2 injection: validated empirically that Spirulina's Zarrouk bicarbonate
        # reservoir (~200 mM) self-buffers pH near 9.5 without any active control — across
        # 6000 random-action steps CO2 injection never fired and pH stayed in [8.54, 9.44].
        # Only the baseline ambient-air sparge remains (420ppm atmospheric CO2).
```

## ./environments/genetic_env.py:57 {#--environments-genetic_env-py-57}

```
        # --- Semi-continuous cycle: periodic, agent-controlled harvest events ---
        # First attempt at reintroducing the earlier semi-continuous design used a
        # CONTINUOUS per-step dilution rate D (agent chooses D every single 0.02h step).
        # That is the textbook chemostat-chaos mechanism, but a live 5M-step training run
        # showed it's too punishing for this organism: mu_max is deliberately slow
        # (~0.055h^-1, ~12.6h doubling, from the literature-calibration work), so a
        # continuous action exposes the policy to a lethal washout region on literally
        # every step of every episode — the run spent its entire budget stuck unable to
        # survive past ~step 1700 on average, never advancing past D0.
        #
        # Real small-scale indoor Spirulina cultivation is also practically never run as
        # true continuous chemostat (contamination risk, low economic payoff for a
        # slow-growing photoautotroph at this scale) — periodic partial-harvest-and-
        # replenish (harvest a fraction every N hours) is the actual standard practice,
        # and it also avoids the repeated-lag-phase cost of full batch restarts. This
        # keeps the genuine "avoid over-harvesting -> washout, don't under-harvest ->
        # yield left on the table" control problem, at a far coarser and more survivable
        # decision cadence.
```

## ./environments/genetic_env.py:78 {#--environments-genetic_env-py-78}

```
        # Action: [Stirring, Light, Harvest fraction] — CO2 and Nutrient dosing remain
        # automated/implicit. Harvest fraction is only applied on periodic harvest-event
        # steps (see step() dilution/harvest block); ignored on other steps.
```

## ./environments/genetic_env.py:83 {#--environments-genetic_env-py-83}

```
        # Observation Space (8 Dims)
        # Channels 0-5 are real hardware sensors; 6-7 are derived from those plus the
        # controller's own clock (no additional hardware assumed — see Fix #18 below):
        # 0: Turbidity (SEN0189, 0-1000 NTU)    1: pH (SEN0161)
        # 2: Harvest integral (pump counter, L) 3: Conductivity (DFR0300)
        # 4: Temperature (DS18B20)              5: Light (BH1750, 0-65535 lux)
        # 6: Turbidity EMA (software filter over channel 0)
        # 7: Phase (episode fraction, or harvest-cycle fraction — see USE_EPISODE_PHASE)
        # Dropped: n_pool (no sensor), RGB (unreliable)
        # Fix #18 (v21): two ADDITIONAL channels, 6 -> 8.
        #   6: turbidity_ema  — long-window EMA of the turbidity sensor
        #   7: episode_phase  — step_count / max_steps, in [0, 1]
        #
        # Motivation, from the failure pattern across v11-v20: `time_avg_od` was the SOLE
        # failing criterion in literally every run, while the agent's only biomass signal is
        # `turbidity_obs` — which is corrupted in ways that are unusually hostile to control:
        #   * multiplied by pigment_contrast (0.7-1.0) and clump_scatter (avg_clump^-1/3)
        #   * multiplied by saturation_factor = 1/(1+0.05*od), i.e. NONLINEAR in the very
        #     quantity being estimated
        #   * flow_noise = 1 + 0.03*(rpm/200)*N(0,1) — multiplicative noise whose magnitude
        #     scales with the stir RPM the agent itself sets
        #   * a difficulty-scaled noise floor, +-1-2% jitter, and at D1+ an EMA lag whose
        #     length ALSO depends on stir RPM
        # So the policy is graded on time-averaged OD while observing OD through a nonlinear,
        # multiplicatively-noised, self-coupled, lagged channel. The scripted expert reads true
        # `od` and beats every learned policy; that gap is at least partly an OBSERVABILITY
        # gap, not a learning-algorithm gap.
        #
        # Neither new channel assumes new hardware — both are computable by any real
        # controller from what it already has (a filter over the existing turbidity sensor,
        # and its own clock), so sim-to-real transferability is preserved. This deliberately
        # does NOT expose true `od`: that would be privileged information a real reactor's
        # nephelometer cannot provide, and would make results non-transferable.
        # REVERTED TO 6D BY DEFAULT (v26). Fix #18's two extra channels are retained behind
        # OBS_EXTENDED (class attribute) rather than deleted, because they were the single
        # best-measured change of the PPO series (od +74%, first from-scratch policy to clear
        # all four D1 criteria). Reasons for making 6D the default again:
        #   * The 8D change silently orphaned every earlier checkpoint, including
        #     model_data/BEST_bc_clone_D2_validated — the project's recommended deliverable and
        #     the only artefact that passes held-out D2. It could not even be LOADED against
        #     the 8D env (scripts/validate.py now reports that mismatch explicitly).
        #   * legacy/TD_MPC2.py is written against OBS_DIM=6, so 6D removes one of the three
        #     interface breaks that stopped it running at all.
        #   * 6D is the real-hardware-sensor set, so it is the honest sim-to-real baseline;
        #     channels 6-7 are derived quantities and belong behind a flag.
        # Set OBS_EXTENDED=True to restore the 8D observation.
```

## ./environments/genetic_env.py:200 {#--environments-genetic_env-py-200}

```
        # --- DAY/NIGHT CYCLE ---
        # lights_off_hour: hour of day (0-24) when lights turn off. None = always on (default).
        # lights_on_hour:  hour of day (0-24) when lights come back on.
        # Example: lights_off_hour=20, lights_on_hour=6  ->  14h light / 10h dark.
```

## ./environments/genetic_env.py:218 {#--environments-genetic_env-py-218}

```
            # Was N(0.080, 0.015) — ~8.7h doubling. This is faster than the project's OWN
            # cited source (literature.md: Torzillo et al. 1993, mu_max 0.04-0.07 h^-1;
            # the old mean of 0.08 sat above the ENTIRE cited range). Independently
            # confirmed via fresh literature search: real Zarrouk-medium Arthrospira batch
            # studies report generation times of ~2.7-3.2 days (not ~9h), and PBR-optimal
            # cases report specific growth rates as low as ~0.12/day (~0.005/h). Recentered
            # on the cited range's midpoint (~0.055/h, ~12.6h doubling) — still on the
            # faster/optimistic end of real reported values, but no longer contradicting
            # the project's own citation.
```

## ./environments/genetic_env.py:254 {#--environments-genetic_env-py-254}

```
        # Super-Agent Scaling: 1 Agent = 2,500,000 Cells (~500pg each)
        # Density-dependent starting mass:
        # At 300 cells, mass is ~1.25e8. At 15,000 cells (Log Ladder limit),
        # mass drops to ~0.8e8 (starving) due to immediate shelf-shading/nutrient competition.
```

## ./environments/genetic_env.py:303 {#--environments-genetic_env-py-303}

```
        # --- Sim-to-Real Sensor Drift & Lag (D1+) ---
        # 8 channels: [Turbidity, pH, Dosing_integral, Conductivity, Temperature, Light(BH1750),
        #              turbidity_EMA (Fix #18), episode_phase (Fix #18)]
```

## ./environments/genetic_env.py:309 {#--environments-genetic_env-py-309}

```
            # Fix #18: the EMA is DERIVED from channel 0, so it must inherit that channel's
            # drift rather than draw an independent one — an independent draw would let the
            # policy average the two to cancel a drift a real controller cannot cancel.
```

## ./environments/genetic_env.py:380 {#--environments-genetic_env-py-380}

```
            # Fix #19 (v22): window biofilm scatters extra light into the detector — the
            # reading drifts HIGH while true biomass is unchanged. Monotonic within an
            # episode, so an EMA cannot filter it out (see the accumulation comment in step()).
```

## ./environments/genetic_env.py:406 {#--environments-genetic_env-py-406}

```
        # Fix #18 (v21): long-window EMA of turbidity. The raw channel carries multiplicative
        # noise that scales with the agent's own stir setting, so a single reading is a poor
        # OD estimate; averaging it over a long window recovers most of the underlying signal.
        # ALPHA corresponds to a ~600-step (one harvest-interval) effective window, matching
        # the timescale the harvest decision actually operates on.
```

## ./environments/genetic_env.py:425 {#--environments-genetic_env-py-425}

```
            # Fix #21 (v22): HARVEST-CYCLE phase, replacing the episode phase used in v21.
            # Episode phase (step_count/max_steps) had two defects. (1) Sim-to-real: a real
            # semi-continuous reactor runs indefinitely and has no "episode", so the signal has
            # no hardware counterpart. (2) Worse, `time_avg_od` is scored over step_count>=3600
            # — exactly the back half — so episode phase told the policy precisely when the
            # scoring window opened, inviting "coast, then maximise OD in the graded half".
            # `held_out_sweep.py` could NOT have caught that, since it uses the same episode
            # length and the same back-half metric — the same blind spot that produced the v14
            # and v17 false positives.
            # An action trace of v21's best checkpoint showed NO gaming (harvest/stir/light are
            # smooth across the step-3600 boundary), so v21's result stands. This is removing a
            # latent hazard, not correcting a corrupted one.
            # Harvest-cycle phase is strictly better on all three counts: any real controller
            # knows time since its last harvest exactly, it is the timing signal actually
            # relevant to the harvest decision, and being PERIODIC it cannot locate the
            # back-half scoring window.
```

## ./environments/genetic_env.py:447 {#--environments-genetic_env-py-447}

```
        # Truncate to the configured width (6 by default, 8 with OBS_EXTENDED). Channels 6-7
        # are still COMPUTED above — the turbidity EMA has to keep updating regardless so it is
        # warm if the flag is switched on, and the cost is two float ops per step.
```

## ./environments/genetic_env.py:496 {#--environments-genetic_env-py-496}

```
    # Per-event harvest target for reward_harvest below (mg per harvest event, fired every
    # HARVEST_INTERVAL_STEPS=600 steps / 12h, 12 events per 144h episode). Derived from a
    # (stir=80rpm, light=1000umol, harvest_frac-sweep) grid sweep at 20L/D2 physics
    # (dynamic_profile_sweep.py): frac in {0, 0.05, ..., 0.40} all ran full episodes with
    # 0% crash; frac=0.50 (=F_MAX) crashed 100% of episodes — the washout cliff sits
    # between 0.40-0.50, i.e. only the top ~20% of the action range is dangerous (a much
    # safer margin than the earlier continuous-D design's cliff at ~33% of its range).
    # Best sustainable fraction was 0.15: 147.9mg total/144h, 12.32mg/event, 0% crash — the
    # legitimate per-event ceiling. Follow-up 4-seed run at that setpoint: harvested_mg
    # 118-175 (median ~150), time_avg_od 0.014-0.022 (median ~0.019), 0% crash.
```

## ./environments/genetic_env.py:508 {#--environments-genetic_env-py-508}

```
    # Light-path biofouling coefficient. NOTE: 0.0002 is calibrated for conventional OD units
    # (~1-10, lab OD600), but this sim's `od` is volume-normalised as mass_mg/volume_L/300 and
    # sits at ~0.015-0.02 — roughly 250x smaller. At this value the term accumulates only
    # ~0.0003 over a full 144h episode against its own 0.5 cap, i.e. it has been INERT for
    # this project's entire history: `enable_fouling=True` has had no measurable effect.
    # Left at the historical value so runs v11-v22 remain comparable; exposed as a class
    # attribute so feasibility probes can raise it without editing physics.
    # A realistic value for this OD scale is ~0.075 (see Fix #19's turbidity-fouling note).
```

## ./environments/genetic_env.py:518 {#--environments-genetic_env-py-518}

```
    # ── Switchable realism / observation options (v23) ────────────────────────────────────
    # v21 -> v22 changed THREE things at once plus the seed, and v22 regressed
    # (best-det 101.9mg/od 0.0094 -> 82.5mg/od 0.0056). That confounds attribution, so these
    # are now flags rather than hard-coded behaviour, and v23 replicates v21's configuration
    # to establish its variance before drawing conclusions.
    #
    # TURB_FOULING_COEF: nephelometer window fouling (Fix #19). Set to 0.0 to disable.
    #   Kept available because sensor corruption is genuinely realistic, but note it works
    #   DIRECTLY AGAINST Fix #18: #18 raised time_avg_od 74% by improving the biomass signal,
    #   and this degrades that same signal by up to +10% bias. Enabling both in consecutive
    #   runs pulled in opposite directions.
    # HARVEST_PUMP_ERROR: +-fraction delivery error on the harvest pump (Fix #20). 0.0 disables.
    #   Realistic, and it forces closed-loop harvest control, but it is an added difficulty.
    # USE_EPISODE_PHASE: True gives obs channel 7 = step_count/max_steps (v21's setting);
    #   False gives the periodic harvest-cycle phase (v22's setting).
    #   TRADE-OFF, deliberately documented: episode phase is NOT sim-to-real transferable (a
    #   continuous reactor has no episode) and it reveals when the time_avg_od scoring window
    #   opens at step 3600 — a latent gaming hazard that held_out_sweep.py could not detect.
    #   BUT an action trace of v21's best checkpoint showed NO gaming (harvest/stir/light all
    #   smooth across step 3600), so on the evidence that channel was providing legitimate
    #   long-horizon timing value. Set True while the objective is "can PPO reach D2 at all";
    #   set False for anything intended for deployment.
```

## ./environments/genetic_env.py:544 {#--environments-genetic_env-py-544}

```
    # OBS_EXTENDED: False -> 6 channels (real hardware sensors only; the default and the
    # sim-to-real baseline). True -> 8 channels, adding Fix #18's turbidity EMA and phase.
    # Fix #18 measurably helped PPO (od +74% in v21) but its dimension change orphaned every
    # prior checkpoint, so it is opt-in. Changing this invalidates saved models in BOTH
    # directions — scripts/validate.py detects and reports the mismatch rather than crashing.
```

## ./environments/genetic_env.py:564 {#--environments-genetic_env-py-564}

```
        # Curriculum metric: time-averaged OD over the back half of the episode (steps
        # 3600-7200) — a "sustained healthy steady-state" proxy that can't be gamed by a
        # brief early spike.
```

## ./environments/genetic_env.py:571 {#--environments-genetic_env-py-571}

```
        # 1. Standing OD — dense, rewards building/maintaining a productive culture.
        # Weight reduced 0.15->0.05 alongside reward_harvest's increase below — a fixed-
        # action physics sweep (dynamic_profile_sweep.py, D2) found this term's raw
        # per-episode ceiling (0.15*7200=1080) so far exceeded reward_harvest's
        # (0.5*12=6, harvest only fires on 12 event-steps/episode) that mean total reward
        # was HIGHEST at harvest_frac=0.0 and fell monotonically as harvest increased
        # (+301.7 at frac=0 vs +161.7 at the physically-best frac=0.15) — the standing-OD
        # term alone made never-harvesting the reward-maximizing choice, independent of
        # this session's earlier od_delta fix (verified not the cause: od_delta stayed
        # flat ~69 across all fracs, exactly as its population-invariant design intends).
        #
        # REVERTED to 0.15 (v11). The 0.05/0.06 reduction (v8/v9/v10, paired with
        # reward_harvest=4.0 below) was the session's clearest mistake: three consecutive
        # runs under it failed to cleanly clear even D0 (v8 froze 8 chunks, v9 over-
        # harvested and stalled, v10 collapsed late), whereas the best result of the whole
        # session — v7, which reached a genuinely-validated D1 and hovered at the D2 edge —
        # used the ORIGINAL 0.15. Critically, the fixed-action sweep that motivated the
        # reduction turned out to be an ANTI-predictor of trained behavior at the time: v7's
        # static landscape favored never-harvest yet v7 harvested fine; v8/v9/v10's favored
        # restraint yet v9 over-harvested. So the static-sweep-driven REWEIGHTING was
        # methodologically unsound back then.
        #
        # Fix #10 (v15): reshaped, not just reweighted (post-v14). Unlike the v8-era
        # reweighting above, this isn't a single static-sweep-driven guess — it's the
        # explanation for a pattern independently confirmed by THREE separate trained
        # policies (the original pre-session run, v4, and v14) that all converged to the
        # identical degenerate "never harvest, let biomass grow forever" deterministic
        # policy despite different seeds, curriculum states, and (for v14) a working
        # deterministic-eval gate along the way. `tanh(od/0.20)` is monotonically
        # increasing in od with no ceiling below saturation (~od=0.6+) — reward_od's raw
        # per-episode ceiling (0.15*7200=1080, dense every step) so dwarfs
        # reward_harvest's (0.5*12=6, only 12 discrete event-steps/episode) that growing
        # OD forever is always reward-dominant over harvesting it back down, regardless of
        # training hyperparameters. Directly reconfirmed via a fresh dynamic_profile_sweep.py
        # run immediately before this fix: frac=0.0 scored 300.6, monotonically DECREASING
        # as harvest fraction rose, bottoming at the physically-best frac=0.15 (161.4) — and
        # v14's own held-out sweep + test_actions.py trace showed the exact predicted
        # behavior (harvest pinned at 0.00 frac all episode, OD climbing unboundedly to
        # 0.03-0.07+, reward 149 purely from od/biomass compounding on unconstrained growth).
        #
        # Replaced tanh's always-increasing shape with a peaked target-band reward
        # (x*e^(1-x), x=od/OD_TARGET): rises from 0 toward a peak of the full weight at
        # od=OD_TARGET, then DECAYS for od beyond it (value ~74% of peak at 2x target, ~9%
        # at 5x target) — so standing OD above the healthy operating range actively stops
        # paying, rather than merely plateauing as tanh did. OD_TARGET=0.012 chosen from
        # directly measuring a healthy frac=0.15/stir=80/light=1000 trajectory's own
        # operating band (median od=0.0095, p10-p90 0.007-0.012) — the target sits at the
        # top of that band, not a value picked to force a particular harvest fraction.
        # Everything else (weight 0.15, reward_biomass, reward_od_delta, reward_harvest)
        # left untouched — one structural change at a time, same discipline as every prior
        # reward fix this session.
```

## ./environments/genetic_env.py:626 {#--environments-genetic_env-py-626}

```
        # (Fix #28 attempt, reverted): a rolling-window OD-average reward term was tried here
        # to close the instantaneous-vs-time-averaged gap between reward_od and the gate's
        # time_avg_od metric. Live-verified via the same D0/D1/D2 harvest-fraction sweep used
        # to design it, and found to REINTRODUCE the "never harvest, grow forever" exploit
        # Fix #10 closed: a never-harvesting trajectory doesn't oscillate, so its rolling
        # average trivially equals its own (monotonically rising) instantaneous value —
        # rewarding "sustained average OD" rewards the UNHARVESTED baseline more than a real
        # harvesting policy, whose average is necessarily dragged down by its own periodic
        # troughs. Confirmed via direct comparison against the pre-change reward at D2,
        # init=300: OLD reward had frac~0.10 beating frac=0.00 (1115 vs 1047, harvesting
        # correctly favored); adding this term flipped that ordering (1318 vs 1420, never-
        # harvest winning). Not a weight-tuning issue — the mechanism itself rewards the
        # wrong baseline. Left out rather than patched further; the time_avg_od gate-alignment
        # problem is real (see the D2 sweep in finalresults.md) but this specific shape isn't
        # the fix.
```

## ./environments/genetic_env.py:642 {#--environments-genetic_env-py-642}

```
        # 2. Per-cell biological growth — incentivises steady healthy growth, and (folded
        # in, previously a separate "stagnation" term) penalises decline/flatlining.
        # Tried shifting the tanh's zero-crossing to 0.01 as a single smooth curve first,
        # but that made the flatline penalty ~25x weaker at the boundary than the original
        # flat -0.01 (tanh is very shallow near its center, e.g. at growth=0 the shifted
        # curve alone gives only -0.0004) — verified by direct calculation before training,
        # not discovered live. Kept as the original tanh curve plus the same flat penalty,
        # just accumulated into one bucket instead of two: numerically identical behavior
        # to the previous two-term version, simplified only in bookkeeping (one fewer
        # tracked term), not in the actual incentive it applies. Simplified after three
        # training attempts made clear extra reward-shaping terms are a liability — see the
        # washout term below, which needed two separate live-tested bug fixes before it
        # could even run once.
```

## ./environments/genetic_env.py:660 {#--environments-genetic_env-py-660}

```
        # 3. OD movement — dense guidance on the *direction* of change, not the absolute
        # level. reward_od (term 1) uses tanh(od/0.20), which is nearly flat near od=0 —
        # a culture sinking toward extinction and one that's climbing back out of it look
        # almost identical to that term, since both produce a tiny reward near zero. This
        # showed up directly as a diagnosed failure mode: ~1/3 of cold-start episodes on a
        # trained checkpoint entered an extended (100-1500 step) near-zero-OD stretch that
        # never hard-crashed but scored heavily negative — and the policy's actions during
        # that stretch were statistically indistinguishable from its actions in healthy
        # episodes, i.e. there was no gradient teaching it to react. Deliberately no
        # absolute threshold/floor here (replaces an earlier flat "-0.05 if od<0.001"
        # term) — just the sign and size of the OD delta each step, so the signal is
        # present at every OD level, not only near a hand-picked cutoff. Skipped on
        # harvest-event steps: the OD drop there is the intended, rewarded effect of
        # harvesting (reward_harvest), not decline, and penalizing it would fight that
        # incentive directly.
        # Uses RELATIVE OD change (like reward_biomass already uses per-cell growth
        # rather than raw delta_mass) rather than absolute delta — an early version used
        # absolute delta_od and, live-tested, caused harvest to collapse toward 0 over
        # several chunks at D2: absolute OD deltas scale with population size, so a
        # larger standing culture generates larger absolute deltas at the same per-cell
        # growth rate, making "never harvest, just keep growing the population" directly
        # reward-maximizing under that version — confirmed via direct calculation (a
        # healthy episode's total was 552.79 od_delta at a large init vs. 43.90 at a small
        # init, off the same underlying growth *rate*) before switching to the relative
        # form, which is verified population-size-invariant (44-49 at both a 300-cell and
        # 3,000-cell init under an identical policy) and doesn't penalize a real harvesting
        # policy. Scale (2e-4) and weight (0.01) calibrated the same way as before —
        # against actual relative OD-change magnitudes measured under several scripted
        # policies — so this term's per-episode total lands alongside reward_od's rather
        # than dominating it.
        # Fix #11 (v15): floor the denominator at OD_RATE_FLOOR (1e-4) rather than the
        # near-infinitesimal +1e-6 used previously. In the near-zero-OD "zombie" regime
        # (od~1e-6-1e-5, the exact regime this term exists to guide recovery from), the old
        # +1e-6 denominator meant physically-insignificant noise in delta_od produced
        # relative changes of order 1+, saturating tanh(rel_delta_od/2e-4) to a near-random
        # +-1 sign every step — pure noise exactly where the signal should be most
        # informative for teaching recovery. Flooring the denominator at a fixed, physically
        # meaningful scale keeps the term identical for the vast majority of normal
        # operation (od >> 1e-4) while replacing that noise-driven saturation with a
        # graduated, non-random signal near zero.
```

## ./environments/genetic_env.py:706 {#--environments-genetic_env-py-706}

```
        # 4. Periodic harvest yield — fires only on harvest-event steps (0.0 otherwise),
        # rewarding the size of that event's yield against TARGET_MG_PER_EVENT.
        #
        # Weight history: originally 0.25, contributed only ~4% of realized episode reward
        # (reward_breakdown.py, 10ep D2: harvest=+1.10 vs od=+25.76, biomass=+10.68) since it
        # only fires 12/7200 steps vs the dense terms firing every step. First correction
        # raised it to 2.0 (8x) — live-tested for a full 4M-step run and found to be an
        # OVERCORRECTION: reward_od scales with actual OD (0.15*tanh(od/0.20)*7200 steps),
        # so in the exact low-OD regime a struggling D0 policy sits in (od~0.0005-0.0015,
        # observed throughout that run), reward_od's per-episode total (2.7-8.1) fell BELOW
        # reward_harvest's near-constant per-episode total (~9.24 at weight=2.0, since
        # harvest yield is roughly independent of overall culture health) — i.e. harvest
        # reward structurally dominated od reward exactly when sustaining OD mattered most.
        # D0 never advanced across the full 4M-step budget; time_avg_od regressed to 0.0010
        # by the end (confirmed via direct calculation, not just correlation, before this
        # fix). Reduced to 0.5: harvest's per-episode total (~2.31) stays below reward_od's
        # even at the lowest OD observed in that run (od=0.0005 -> reward_od=2.70), while
        # still being 2x the original under-weighted 0.25. Re-validate via chunk summaries
        # after (another) training run — this is not a proof the imbalance is now perfectly
        # resolved, only that the specific dominance case observed is closed off.
        # REVERTED to 0.5 (v11), undoing the 4.0 overcorrection that accompanied the
        # reward_od reduction. At 4.0, action traces showed the trained policy converging
        # to an over-aggressive harvest fraction (0.30-0.36 vs the physically-sustainable
        # ~0.10-0.15), stripping the culture so hard that OD/time_avg_od collapsed even
        # though harvested_mg looked fine — the mirror image of the original under-harvest
        # problem. 0.5 is v7's proven value: v7 (relative od_delta + od=0.15 + harvest=0.5)
        # reached a genuinely-validated D1 and got to the D2 edge, harvesting healthily
        # (66-83mg det) the whole way. Not chasing the D2 harvest ceiling with weight
        # surgery again — three attempts (v8/v9/v10) failed. If D2 needs pushing, the
        # untried, lower-risk levers are training-side (longer PPO rollout horizon for the
        # delayed-OD credit-assignment gap; best-seen-chunk checkpoint selection), not more
        # reward reweighting.
```

## ./environments/genetic_env.py:740 {#--environments-genetic_env-py-740}

```
        # Fix #28: harvest-event OD-collapse penalty. reward_harvest above saturates almost
        # immediately past the physically-optimal harvest fraction — directly measured: at D0,
        # harvested_mg stayed roughly flat (~50-60mg) across harvest_frac 0.12-0.38, while
        # time_avg_od collapsed 8x (0.0069 -> 0.0006) over that same range, i.e. reward_harvest
        # gives no signal at all distinguishing "harvested efficiently" from "over-harvested
        # for no extra yield." reward_od already penalizes the resulting low od, but only on
        # THIS one step — the damage persists for many subsequent steps as the culture
        # recovers, a long-horizon credit-assignment gap in a 7200-step episode with only 12
        # harvest decisions. This term makes the cost immediate and locally attributable to
        # the decision that caused it: on a harvest-event step only, penalize the post-harvest
        # od (self.od is already updated to its post-dilution value above) falling below a
        # floor fraction of OD_TARGET. Floor and weight chosen so the measured "healthy" ~0.12
        # fraction (post-harvest od comfortably above the floor) draws no penalty while
        # fractions of 0.20+ (post-harvest od well under it) do.
```

## ./environments/genetic_env.py:806 {#--environments-genetic_env-py-806}

```
        # Fix #16 (v19): INTERVAL-AVERAGED HARVEST — the credit-assignment fix.
        #
        # Previously the harvest dimension was read ONLY on event steps (every
        # HARVEST_INTERVAL_STEPS=600). On the other 7188 of 7200 steps the policy emitted a
        # harvest value that this env discarded outright — yet PPO still assigned those
        # timesteps an advantage and updated the harvest dimension toward whatever was sampled
        # there. 599 of every 600 gradient samples on that dimension were therefore pure noise.
        #
        # That is the one structural difference between the dimensions that work and the one
        # that does not. Across v4/v14/v15/v16b/v17/v18 the agent learned stir and light
        # correctly EVERY time (light reliably settles near the 1000umol sweep optimum) — both
        # act on every step and receive honest per-step credit. Harvest, the only dimension with
        # 1-in-600 credit, failed EVERY time, and in a different direction each run
        # (never-harvest / drift-up / coast-on-low-light / decay-to-zero / over-harvest-early),
        # which is the signature of a dimension driven by noise rather than by a consistent
        # gradient. Every hyperparameter-level explanation was measured and eliminated first:
        # reward exploit (reward_ab.py: reward prefers the expert by +313), exploration noise
        # (noise_sensitivity.py: expert dominates at every sigma), and credit horizon
        # (Fix #13's gamma 0.9995, which did not stop the drift).
        #
        # Now the applied fraction is the MEAN of the harvest action over the interval. Two
        # consequences, both wanted:
        #   1. Every step's harvest output causally affects the outcome, so PPO's per-step
        #      credit on that dimension becomes honest rather than spurious.
        #   2. The applied value averages ~600 samples, cutting the sampling noise that
        #      actually reaches the culture by ~sqrt(600) ~ 24x at the observed train/std~0.5 —
        #      without touching the entropy schedule, which noise_sensitivity.py showed is fine.
        # The physics is unchanged: still one discrete dilution event every 12h, same F_MAX,
        # same washout cliff. Only WHICH number that event uses changes.
```

## ./environments/genetic_env.py:838 {#--environments-genetic_env-py-838}

```
        # 1b. Automated PID Controller (Nutrient N/P threshold control only)
        # No CO2 control: Spirulina's Zarrouk bicarbonate reservoir self-buffers pH near
        # 9.5 without any active carbon dosing (validated empirically — see genetic_env
        # gas-phase config comment). Only ambient air sparge feeds the carbonate system.
        # Gate on EITHER N or P running low — dosing replenishes both (87% N, 8% P per
        # BG-11 ratio), but N typically depletes slower than P relative to its dose threshold.
        # Gating on N alone left P to starve silently while N sat in the hold band.
```

## ./environments/genetic_env.py:913 {#--environments-genetic_env-py-913}

```
        # --- Biofouling Accumulation ---
        # Cells adhere to surfaces at low mixing and high biomass density.
        # exp(-0.5) ≈ 60% light transmission at full fouling (cap 0.5).
```

## ./environments/genetic_env.py:917 {#--environments-genetic_env-py-917}

```
            # LIGHT_FOULING_COEF is a class attribute so it can be overridden for feasibility
            # probes without editing physics. Default 0.0002 is the historical value — see the
            # inertness note below; it is left unchanged so v11-v22 stay comparable.
```

## ./environments/genetic_env.py:924 {#--environments-genetic_env-py-924}

```
            # Fix #19 (v22): NEPHELOMETER WINDOW FOULING (D1+).
            # The block above fouls only the LIGHT PATH. The turbidity sensor's own optical
            # window fouls too, and on real hardware that is a well-known failure: biofilm on
            # the window scatters light into the detector, so the reading drifts HIGH while
            # true biomass is unchanged. It matters more here than it would elsewhere because
            # Fix #18 added an EMA of that same sensor as a biomass estimate — an EMA is robust
            # to zero-mean noise but faithfully tracks a MONOTONIC drift, so without this the
            # smoothed channel would look far cleaner in sim than it ever could on hardware,
            # and a policy could learn to trust it more than is warranted.
            # Same driver as light fouling (adhesion at low mixing and high density) so high
            # stir remains the mitigation, and it is a genuine control trade-off rather than
            # unavoidable noise. Capped at +25% so it degrades the estimate without destroying
            # it. D1+ only, matching the existing sensor-imperfection gating.
            # COEFFICIENT NOTE: 0.075, NOT the 0.0002 used by the light-fouling term above.
            # That 0.0002 is calibrated for conventional OD units (~1-10, lab OD600). This
            # sim's `od` is volume-normalised as mass_mg/volume_L/300 and sits at ~0.015-0.02
            # — roughly 250x smaller — so with 0.0002 the term accumulates only ~0.0004 over a
            # full 144h episode against its 0.5 cap, i.e. ~1300x below its own ceiling. It is
            # effectively inert (see the report on the existing light-fouling term, which has
            # the same defect and has therefore never been active in this project).
            # 0.075 is set so that at a typical stir=60rpm and od=0.016 the factor reaches
            # ~0.12 (+12% reading bias) over 144h: material enough to punish a policy that
            # blindly trusts the smoothed turbidity channel, small enough to remain correctable
            # by raising stir.
```

## ./environments/genetic_env.py:953 {#--environments-genetic_env-py-953}

```
        # --- Physics (Chaotic Turbulence) ---
        # Apply only to active cells
        # We replace simple Brownian motion with structured "Swirls"
        # Flow V(z, t) = Sum( A * sin(k*z - w*t) )
```

## ./environments/genetic_env.py:966 {#--environments-genetic_env-py-966}

```
            # 1. Aggregation (Sticking) - Orthokinetic + Perikinetic
            # Orthokinetic: Stirring INCREASES collision frequency (Smoluchowski)
            # Sticking = Base (Brownian) + Shear-Induced (RPM)
            # Fix scaling bug: Use true physical OD, not an assumption based on cell count!
```

## ./environments/genetic_env.py:972 {#--environments-genetic_env-py-972}

```
            # --- New Physics: 
            # - Base chance: 1e-3 (Stronger - 5x boost)
            # - RPM Boost: Increases linearly with Mixing (more collisions)
            # --- Flocculation: Stirring BREAKS clumps (shear dispersal dominates at moderate RPM)
            # rpm_factor now REDUCES sticking — higher RPM = more shear = less aggregation
            # At 0 RPM: rpm_factor=1.0 (max sticking). At 200 RPM: rpm_factor=0.2 (80% less sticking)
```

## ./environments/genetic_env.py:991 {#--environments-genetic_env-py-991}

```
            # Brownian/diffusive breakup (always active, weak)
            # Prevents runaway aggregation at low RPM
            # Small clumps (1-5) barely affected, large clumps (50+) slowly erode
```

## ./environments/genetic_env.py:1007 {#--environments-genetic_env-py-1007}

```
            # --- 2D Kinematic Turbulence (Airlift / Convection Loop) ---
            # Center (x=0.5): Upward Flow (-z)
            # Walls (x=0,1): Downward Flow (+z)
            # Top/Bottom: Turnaround (Horizontal Flow)
```

## ./environments/genetic_env.py:1015 {#--environments-genetic_env-py-1015}

```
            # 1. Vertical Velocity (Vz)
            # Cosine profile: Max Up at 0.5, Max Down at 0, 1.
            # Scale: 0.01 m/s * intensity
```

## ./environments/genetic_env.py:1083 {#--environments-genetic_env-py-1083}

```
        # 1. Shear Stress (RPM > 400)
        # Random death probability for cells if mixing is too violent
        # Note: We already have this logic downstream at line 550, but let's keep the flow clean.
        # Actually, let's just fall through to the Biology block.
```

## ./environments/genetic_env.py:1093 {#--environments-genetic_env-py-1093}

```
            # 1. Spectral Light Field (RGB Physics)
            # Action 'light' sets Total Surface Intensity (PAR)
            # I_surface is already calculated at top of step()
```

## ./environments/genetic_env.py:1106 {#--environments-genetic_env-py-1106}

```
            # Attenuation Coefficients (k)
            # Red: Absorbed STRONGLY by Chlorophyll (Growth)
            # k_red boosted to 3.5 (was 10.0) to allow deep biological growth past 12k cells
```

## ./environments/genetic_env.py:1119 {#--environments-genetic_env-py-1119}

```
            # ── Turbulent Flash-Light Effect (Biologically Accurate) ──────────
            # In real Spirulina PBRs, turbulent mixing causes cells to cycle
            # between the photic zone (surface) and dark zone (deep) rapidly.
            # This "flash-light effect" dramatically increases photosynthetic
            # efficiency (Kok effect): brief intense surface flashes > sustained dim light.
            #
            # At 0 RPM  : cells see only their actual static depth (fully stratified).
            # At 500 RPM: cells see a near-random depth distribution each step (fully mixed).
```

## ./environments/genetic_env.py:1168 {#--environments-genetic_env-py-1168}

```
            # Photo-Inhibition / Shock
            # Cells experience stress when light changes suddenly
            # Scalar reduced from 0.0001 to 0.000001 to prevent startup death
            # At diff=300: Old penalty=99.99%, New penalty=9% (survivable!)
```

## ./environments/genetic_env.py:1182 {#--environments-genetic_env-py-1182}

```
            # 3. Growth Rate (Haldane)
            # Growth is driven by RED light availability
            # Inhibition is driven by TOTAL light intensity
            # f_I = I_growth / (Ks + I_growth + I_total^2/Ki)
```

## ./environments/genetic_env.py:1209 {#--environments-genetic_env-py-1209}

```
            # pH Inhibition (Asymmetric Gaussian — Arthrospira/Spirulina platensis)
            # Peak at 9.3 (Zarrouk operating range 8.5-11; native soda-lake alkaliphile)
            # Acid side: σ=0.7 — steep falloff below pH 8, intolerant of neutral pH
            # Alkaline side: σ=1.0 — tolerates up to pH 11 with moderate inhibition
```

## ./environments/genetic_env.py:1218 {#--environments-genetic_env-py-1218}

```
            # Osmotic Stress — conductivity as ionic strength proxy (all ions: N, P, HCO3-, salts)
            # Spirulina is a soda-lake alkaliphile adapted to high ionic strength; Zarrouk
            # medium baseline is ~19,000 µS/cm (vs BG-11's ~3200). Onset raised accordingly.
            # Uses previous step's conductivity (one-step lag, 72s — negligible).
```

## ./environments/genetic_env.py:1245 {#--environments-genetic_env-py-1245}

```
            # Calculate Rate
            # --- Shear Repair Tax (sigmoid, centered at 100 RPM) ---
            # Spirulina (Arthrospira) is a filamentous cyanobacterium — helical trichomes
            # fragment under shear far more readily than Chlorella's rigid unicells.
            # Max 35% penalty at sustained 200 RPM.
```

## ./environments/genetic_env.py:1253 {#--environments-genetic_env-py-1253}

```
            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Filament breakage accumulates faster than unicell wall fatigue.
            # Onset at ~80 RPM; max 15% growth penalty.
```

## ./environments/genetic_env.py:1267 {#--environments-genetic_env-py-1267}

```
            # Carbon-Limited Growth — Arthrospira/Spirulina has an efficient bicarbonate CCM
            # (active HCO3- transport + carbonic anhydrase), the adaptation that lets it
            # dominate alkaline soda lakes where free CO2 is scarce. HCO3- is the primary
            # DIC source at Zarrouk concentrations (~200 mM); dissolved CO2 contributes little.
```

## ./environments/genetic_env.py:1301 {#--environments-genetic_env-py-1301}

```
            # Droop quota dilution: as cells grow, intracellular quota (N/biomass) is diluted.
            # dQ/dt = V(N) - µ*Q; this applies the -µ*Q term per cell.
            # Only positive net_mu dilutes (shrinking cells retain their quota concentration).
```

## ./environments/genetic_env.py:1308 {#--environments-genetic_env-py-1308}

```
            # --- PROBABILISTIC LYSIS DEATH (replaces dead-code hard starvation check) ---
            # Background lysis: ~0.5%/day (realistic Spirulina batch culture baseline).
            # Stress lysis: scales up to ~5%/day when mean current_mu < m_respiration.
            # Never a hard cliff — always a smooth gradient signal for the RL agent.
```

## ./environments/genetic_env.py:1336 {#--environments-genetic_env-py-1336}

```
            # O4: cells below the death threshold face certain lysis on this cycle
            # Threshold set to 8% of starting mass (1.25e8 pg) — the prior 5e5 floor was
            # unreachable before stochastic lysis killed the cell first (dead code).
```

## ./environments/genetic_env.py:1360 {#--environments-genetic_env-py-1360}

```
            # P uptake: Monod saturation with strain-specific Ks_P.
            # Factor 0.0014 = 0.01 / 7.2 (Redfield N:P ratio by mass — a broadly cross-species
            # phytoplankton constant, applies to cyanobacteria as well as green algae)
```

## ./environments/genetic_env.py:1365 {#--environments-genetic_env-py-1365}

```
            # nut_flow dosing composition: 79% N, 16% P, 5% inorganic salts — matches Zarrouk
            # stock ratio (NaNO3 2.5 g/L : K2HPO4 0.5 g/L ~ 5:1 N:P by mass, far richer in P
            # than BG-11's ~28:1)
```

## ./environments/genetic_env.py:1369 {#--environments-genetic_env-py-1369}

```
            # N waste penalty removed: it caused mode collapse where agent overdosed early,
            # earned heavy penalties, then locked to zero dosing for the entire episode.
            # phi_cur N Gaussian (peak at 200 mg/L) + starvation penalty below provide the equilibrium signal.
```

## ./environments/genetic_env.py:1378 {#--environments-genetic_env-py-1378}

```
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e8 pg (12% above init mass of 1.25e8)
            # Use >= to avoid edge case where cells hover exactly at boundary.
```

## ./environments/genetic_env.py:1416 {#--environments-genetic_env-py-1416}

```
                # B9 removed: when slots are full, let cells continue growing up to the
                # hard 5e8 cap (line ~775). Capping at the division threshold caused OD
                # to plateau at max_cells, killing all growth reward past population ceiling.
```

## ./environments/genetic_env.py:1424 {#--environments-genetic_env-py-1424}

```
            # shock_factor is otherwise only assigned in the num_active>0 branch above;
            # defining it here guarantees it always exists by the time the reward/debug
            # section reads it (previously relied on lazy ternary short-circuit evaluation
            # at each read site — fragile if that code ever gets refactored/extracted).
```

## ./environments/genetic_env.py:1435 {#--environments-genetic_env-py-1435}

```
        # 2. Gas Exchange (O2 & CO2)
        # Closed-tank model: gas composition is set by baseline air + injected pure CO2.
        # kLa scales with agitation, gas throughput, and broth resistance at high biomass.
        # Bug fix: Use cached self.od instead of calling _get_obs() which
        # would corrupt the pH lag buffer by appending mid-step.
```

## ./environments/genetic_env.py:1453 {#--environments-genetic_env-py-1453}

```
        # kLa correlation is stir/gas-flow driven, not volume-parametrized (no volume_L
        # term in the formula below), so it carries over unchanged from the 30L->20L
        # resize. Originally tuned/validated against a 30L sparged tank; re-validate at
        # 20L via the dilution/grid sweep rather than assuming. (units: 1/hour)
        # Agitation drives eddy renewal; gas throughput adds bubble interfacial area.
```

## ./environments/genetic_env.py:1464 {#--environments-genetic_env-py-1464}

```
        # Dissolved Oxygen Dynamics
        # Production: Proportional to Growth (approx 1.5g O2 per g Biomass)
        # Respiration: Proportional to maintenance (approx 1.0g O2 per g Biomass lost)
        # Calculate net biomass change from biology step (approx)
```

## ./environments/genetic_env.py:1484 {#--environments-genetic_env-py-1484}

```
        # --- Periodic Harvest / Dilution (Semi-Continuous Operation) ---
        # Removes a representative random fraction of the just-grown standing culture and
        # replaces it with fresh medium (well-mixed CSTR assumption), but ONLY on periodic
        # harvest-event steps (every HARVEST_INTERVAL_STEPS) — not every step. Computed
        # AFTER delta_mass_mg (biological growth signal, used for gas stoichiometry/reward)
        # so harvest removal is never mistaken for biological decline/decomposition.
        # Harvest fraction too high, too often -> washout (existing crash-termination
        # path); too low/infrequent -> throughput left on the table. Gives the culture a
        # full HARVEST_INTERVAL_STEPS (12h) to recover between decisions, unlike the
        # earlier continuous-D design which exposed the policy to washout risk every step.
```

## ./environments/genetic_env.py:1496 {#--environments-genetic_env-py-1496}

```
        # Fix #16 (v19): apply the INTERVAL MEAN, not the instantaneous sample. See the comment
        # at the harvest decode above for why. The accumulator is reset after each event so the
        # next interval averages only its own steps.
```

## ./environments/genetic_env.py:1504 {#--environments-genetic_env-py-1504}

```
            # Fix #20 (v22): HARVEST PUMP DELIVERY ERROR (D1+).
            # The existing actuator-noise block applies +-5% to stir_rpm and nut_flow but
            # conspicuously skips the harvest pump — the one actuator whose command the agent
            # has never learned to set correctly across v11-v21. Real peristaltic/diaphragm
            # pumps have delivery error from tubing compliance, wear and head pressure, so a
            # commanded fraction is not the delivered fraction. Modelling it matters here
            # beyond realism: it means the agent CANNOT rely on exact open-loop harvest
            # control and must regulate against the measured state, which is precisely the
            # closed-loop behaviour the scripted expert uses and every learned policy so far
            # has failed to acquire. Same +-5% magnitude and D1+ gating as the other actuators.
```

## ./environments/genetic_env.py:1522 {#--environments-genetic_env-py-1522}

```
            # Per-cell Bernoulli removal (not round(frac*n)) — at realistic D and small
            # early-episode populations, frac_diluted*n is often <0.5 and would round to
            # 0 every step, silently disabling dilution. Independent per-cell removal
            # probability = frac_diluted correctly gives the right EXPECTED removal rate
            # over time regardless of population size, matching a well-mixed CSTR.
```

## ./environments/genetic_env.py:1560 {#--environments-genetic_env-py-1560}

```
        # 3. 2-Layer Gas Exchange (surface z<10cm = 10L, bulk z>=10cm = 20L)
        # Surface cells photosynthesize more (better light) → O2 accumulates at surface,
        # CO2 depletes there. Mixing inter-layer exchange dissipates gradients at high RPM.
```

## ./environments/genetic_env.py:1600 {#--environments-genetic_env-py-1600}

```
        # Photosynthetic stoichiometry: 6CO2 → C6H12O6; 6×44/(6×12) = 3.67 mg CO2/mg C fixed.
        # Biomass is ~50% C by dry weight, so per mg DW the CO2 demand is 3.67×0.5 = 1.835 mg CO2/mg DW.
        # Both uptake and release use same ratio: decomposition re-releases the same CO2 per mass.
```

## ./environments/genetic_env.py:1615 {#--environments-genetic_env-py-1615}

```
        # Bicarbonate balance: depleted by photosynthesis (85% of DIC uptake via HCO3-),
        # replenished by CO2 sparging — fraction that equilibrates to HCO3- depends on pH.
        # 85% fraction matches f_carbon's 0.85 * f_hco3 term (Spirulina CCM, same as growth model).
```

## ./environments/genetic_env.py:1623 {#--environments-genetic_env-py-1623}

```
        # NOTE: this ceiling (5.0) is 40x below the Zarrouk medium baseline bicarbonate is
        # reset to (200.0 mM, see reset()) — confirmed via direct testing that it's NOT
        # merely a cosmetic mismatch: raising the ceiling to let bicarbonate sit near its
        # true 200 mM value pushes the Henderson-Hasselbalch pH equilibrium up to ~10.5
        # and holds it there (vs. the intended 9.3-9.5 Spirulina optimum), cutting terminal
        # batch yield roughly in half in direct testing (892mg -> 393mg, same seed/policy).
        # The pH-equilibrium constants (pKa1, co2_aq scaling) were evidently never
        # validated against the documented 200 mM baseline — this clip has been doing load-
        # bearing (if accidental) work keeping pH in the growth-viable range. Left at 5.0
        # deliberately: fixing this properly requires re-deriving the carbonate-system
        # constants against the correct bicarbonate scale, not just widening this clip.
        # Flagged as a real follow-up, not resolved here.
```

## ./environments/genetic_env.py:1667 {#--environments-genetic_env-py-1667}

```
        # OD ~ Mass^0.8 (Self-Shading effect)
        # 1e11 cells ~ 1g/L ~ OD 1.0
        # density_gL = (total_mass_mg * 1e-9) / self.volume_L # BUG: 1e-9 is wrong units (pg->mg happened already)
        # turbidity = 1.0 * (density_gL ** 0.8)
```

## ./environments/genetic_env.py:1678 {#--environments-genetic_env-py-1678}

```
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # At D2, simulate slow natural evolution/mutation of the strain over weeks of deployment.
        # Every ~5 hours (250 steps at dt=0.02h), strain parameters wander by ±1%.
        # This forces the internal LMU to constantly adapt its latent state tracking.
```

## ./environments/genetic_env.py:1687 {#--environments-genetic_env-py-1687}

```
        # 3. Conductivity — Kohlrausch molar conductance formula (µS/cm)
        # σ (mS/cm) = Σ λᵢ (S·cm²/mol) × cᵢ (mol/L), then ×1000 → µS/cm
        # λ values at 25°C (literature): NO₃⁻=71.4, Na⁺=50.1, HPO₄²⁻=57.0,
        # K⁺=73.5, SO₄²⁻=160.0, Na⁺=50.1, Cl⁻=76.4, OH⁻=198.0, H⁺=349.8
```

## ./environments/genetic_env.py:1723 {#--environments-genetic_env-py-1723}

```
        # Extinction check: population OR total biomass. Cells can hover just above the
        # per-cell starvation threshold (1e7 pg) without individually triggering death,
        # leaving a "zombie" culture of a few surviving cells with near-zero total mass —
        # this stalls the episode at flat negative reward (washout+stagnation) for
        # thousands of steps with no learning signal instead of ending the rollout.
```

## ./environments/genetic_env.py:1729 {#--environments-genetic_env-py-1729}

```
            # Reduced from -1000: that scale was 300-1000x larger than typical achievable
            # per-episode reward (~10-20), so occasional exploration-driven crashes were
            # corrupting the LSTM's learned weights for millions of steps to recover from
            # (observed repeatedly as regression-recovery cycles during Spirulina training).
            # -100 still clearly signals "bad" without being catastrophically destabilizing.
```

## ./environments/genetic_env.py:1741 {#--environments-genetic_env-py-1741}

```
        # Per-step debug trace. Gated behind ENV_DEBUG (default OFF) because it dominated every
        # log this project produced: 4,519 of 10,586 lines (43%) in a single chunk, multi-MB per
        # run, and every single grep across 25 runs needed `grep -v EnvDebug` to be readable.
        # Enable per-invocation with `ENV_DEBUG=1 python ...` when actually debugging physics.
```

## ./environments/heavy_env.py:28 {#--environments-heavy_env-py-28}

```
        # Observation Space (6 Dims)
        # 1. OD  2. pH  3. Nutrients  4. Temp  5. Conductivity  6. RGB
        # Note: dissolved_co2 and DO2 remain internal state only.
```

## ./environments/heavy_env.py:285 {#--environments-heavy_env-py-285}

```
        # --- Physics (Chaotic Turbulence) ---
        # Apply only to active cells
        # We replace simple Brownian motion with structured "Swirls"
        # Flow V(z, t) = Sum( A * sin(k*z - w*t) )
```

## ./environments/heavy_env.py:302 {#--environments-heavy_env-py-302}

```
            # Turbulent fluctuations (Small Eddies)
            # We use randomized phases based on cell index to simulate spatial decorrelation without full spatial grid
            # This is a "Lagrangian Particle" trick: each particle has a unique phase offset
```

## ./environments/heavy_env.py:322 {#--environments-heavy_env-py-322}

```
        # Boundary Conditions
        # Note: Vectorized logical ops on masked arrays can be tricky, 
        # so we modify the whole array but valid data is only at active_mask.
```

## ./environments/heavy_env.py:351 {#--environments-heavy_env-py-351}

```
            # Spirulina Extinction: Gentler attenuation matching genetic_env
            # Allows deep biological growth past early population walls
            # (Matches k_red from genetic_env)
```

## ./environments/heavy_env.py:364 {#--environments-heavy_env-py-364}

```
            # --- Photo-Acclimation (Hysteresis) ---
            # Cells adapt to current light intensity over time
            # dA/dt = (I - A) / tau
```

## ./environments/heavy_env.py:370 {#--environments-heavy_env-py-370}

```
            # Update Acclimation State
            # Simple Euler integration
            # cells_I is (MaxCells,), current_acclim is (NumActive,)
```

## ./environments/heavy_env.py:378 {#--environments-heavy_env-py-378}

```
            # Photo-Inhibition / Shock
            # Growth is penalized if current light is very different from acclimated state
            # "Light Shock" factor
            # Normalized difference: delta = (I - A) / (A + 10)
```

## ./environments/heavy_env.py:407 {#--environments-heavy_env-py-407}

```
            # pH Inhibition (Asymmetric Gaussian — Spirulina alkaliphile)
            # Peak at 9.5 (true optimum per Richmond 1988; Vonshak 1997; Habib FAO 2008)
            # Acid side: steep drop (σ=1.2) — Spirulina intolerant of low pH
            # Alkaline side: gentle drop (σ=2.0) — obligate alkaliphile tolerates high pH well
```

## ./environments/heavy_env.py:432 {#--environments-heavy_env-py-432}

```
            # Clamp mu to prevent explosion
            # --- Shear Repair Tax (sigmoid, centered at 195 RPM for Medium) ---
            # Max 25% penalty at sustained >200 RPM.
```

## ./environments/heavy_env.py:508 {#--environments-heavy_env-py-508}

```
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e7 pg (12% above init mass of 1.25e7)
            # Use >= to avoid edge case where cells hover exactly at boundary.
```

## ./environments/heavy_env.py:549 {#--environments-heavy_env-py-549}

```
        # 2. Gas Exchange (O2 & CO2)
        # k_La determines how fast gases exchange with air.
        # Stirring increases surface area and turbulence -> Higher k_La
        # Base transfer + Mixing enhancement
        # Mass Transfer Efficiency (Viscosity): High biomass = thick soup = poor mixing
        # Bug fix: Use cached self.od instead of calling _get_obs() which
        # would fire the sensor pipeline (and pH lag buffer) mid-step.
```

## ./environments/heavy_env.py:564 {#--environments-heavy_env-py-564}

```
        # Dissolved Oxygen Dynamics
        # Production: Proportional to Growth (approx 1.5g O2 per g Biomass)
        # Respiration: Proportional to maintenance (approx 1.0g O2 per g Biomass lost)
        # Calculate net biomass change from biology step
```

## ./environments/heavy_env.py:599 {#--environments-heavy_env-py-599}

```
        # pH driven by DIC concentration (log-linear carbonate chemistry)
        # Add a small carbonate-buffer reserve so low-CO2 behavior is smooth,
        # avoiding the hard 10.9 pH pin from a clamped log-floor.
```

## ./environments/heavy_env.py:627 {#--environments-heavy_env-py-627}

```
        # 5. Salinity Accumulation
        # Salt added by Nutrients (1:1) and Decay (0.5:1)
        # We assume nutrient inflow is proportional to consumption (implicitly replenished) 
        # OR explicitly added? For now, let's say "Makeup Water" adds salt
```

## ./environments/heavy_env.py:639 {#--environments-heavy_env-py-639}

```
        # OD ~ Mass^0.8 (Self-Shading effect)
        # 1e11 cells ~ 1g/L ~ OD 1.0
        # density_gL = (total_mass_mg * 1e-9) / self.volume_L # BUG: 1e-9 is wrong units (pg->mg happened already)
        # turbidity = 1.0 * (density_gL ** 0.8)
```

## ./environments/heavy_env.py:672 {#--environments-heavy_env-py-672}

```
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # Simulate slow natural evolution/mutation of the strain over weeks of deployment.
        # Every ~5 hours (500 steps), strain parameters wander by ±1%.
```

## ./environments/heavy_env.py:679 {#--environments-heavy_env-py-679}

```
        # 1. Productivity: Scale by 1e-8 (1e8 pg = 0.1 mg growth -> Reward 1.0)
        # 2. Spawns: Scale by 0.1 (10 spawns -> Reward 1.0)
        # 3. Stress Shaping: Photoinhibition penalty
```

## ./environments/light_env.py:174 {#--environments-light_env-py-174}

```
        # RPM-coupled EMA sensor lag for pH (idx 1) and DO2 (idx 3).
        # High RPM -> fast mixing -> short lag (2 steps).
        # Low RPM  -> slow mixing -> long lag  (6 steps).
```

## ./environments/light_env.py:325 {#--environments-light_env-py-325}

```
            # pH Inhibition (Asymmetric Gaussian — Spirulina alkaliphile)
            # Peak at 9.5 (true optimum per Richmond 1988; Vonshak 1997; Habib FAO 2008)
            # Acid side: steep drop (σ=1.2) — Spirulina intolerant of low pH
            # Alkaline side: gentle drop (σ=2.0) — obligate alkaliphile tolerates high pH well
```

## ./environments/light_env.py:334 {#--environments-light_env-py-334}

```
            # --- Nutrient Inhibition (Osmotic Stress) ---
            # Safe zone: 0 - 2000 mg/L
            # Penalty starts above 2000. Simplified, no salinity tracking.
```

## ./environments/light_env.py:345 {#--environments-light_env-py-345}

```
            # Clamp mu to prevent explosion
            # --- Shear Repair Tax (150 to 200 RPM) ---
            # 0.0 at 150 RPM, 1.0 at 200 RPM
            # Models the metabolic cost of cellular repair under mechanical stress
```

## ./environments/light_env.py:396 {#--environments-light_env-py-396}

```
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e7 pg (12% above init mass of 1.25e7)
            # Rationale: Doubling (2.5e7) was too slow; deaths outpaced births.
            # 12% growth allows cells to rapidly divide before respiration starvation.
```

## ./environments/prod_env.py:31 {#--environments-prod_env-py-31}

```
        # --- Gas-Phase / Carbonate Configuration (closed 30L PBR) ---
        # No CO2 injection: validated empirically that Spirulina's Zarrouk bicarbonate
        # reservoir (~200 mM) self-buffers pH near 9.5 without any active control — across
        # 6000 random-action steps CO2 injection never fired and pH stayed in [8.54, 9.44].
        # Only the baseline ambient-air sparge remains (420ppm atmospheric CO2).
```

## ./environments/prod_env.py:51 {#--environments-prod_env-py-51}

```
        # --- Harvest action range (pre-calibrated safe band) ---
        # Re-derived at difficulty=2 (O2 toxicity, intra-episode strain drift, actuator
        # noise all engaged — the original 0.15-0.45 band was calibrated at a lower/no-
        # stressor difficulty and its productivity numbers didn't hold once O2 toxicity was
        # active). Re-swept 0.10-1.30 L/h at D2 with 12-15 seeds/rate under a stir/light
        # combo tuned for D2 (low stir=100rpm/high light=1000umol reduces shear+heat
        # penalties, which mattered more than kLa-driven O2 venting). Results: harvested
        # mass is broad and flat from 0.15-1.00 L/h (149-205mg, peak at 0.32 L/h) with ZERO
        # crashes anywhere in that range; the failure cliff only appears at 1.30 L/h (67%
        # crash rate). The old 0.45 ceiling was cutting off a range (0.6-1.0 L/h) that's
        # just as productive and still fully safe. Raised ceiling to 0.70 L/h: keeps a
        # comparable safety margin below the observed 1.30 L/h crash onset to what the old
        # 0.45 kept below the old 0.50 test ceiling, while giving the agent real headroom.
```

## ./environments/prod_env.py:70 {#--environments-prod_env-py-70}

```
        # Observation Space (6 Dims)
        # 6D obs — real hardware sensors only:
        # 0: Turbidity (SEN0189, 0-1000 NTU)   1: pH (SEN0161)
        # 2: Harvest integral (pump counter, L) 3: Conductivity (DFR0300)
        # 4: Temperature (DS18B20)               5: Light (BH1750, 0-65535 lux)
        # Dropped: n_pool (no sensor), RGB (unreliable)
```

## ./environments/prod_env.py:142 {#--environments-prod_env-py-142}

```
        # --- DAY/NIGHT CYCLE ---
        # lights_off_hour: hour of day (0-24) when lights turn off. None = always on (default).
        # lights_on_hour:  hour of day (0-24) when lights come back on.
        # Example: lights_off_hour=20, lights_on_hour=6  ->  14h light / 10h dark.
```

## ./environments/prod_env.py:187 {#--environments-prod_env-py-187}

```
        # Super-Agent Scaling: 1 Agent = 2,500,000 Cells (~500pg each)
        # Density-dependent starting mass:
        # At 300 cells, mass is ~1.25e8. At 15,000 cells (Log Ladder limit),
        # mass drops to ~0.8e8 (starving) due to immediate shelf-shading/nutrient competition.
```

## ./environments/prod_env.py:389 {#--environments-prod_env-py-389}

```
        # 1b. Automated PID Controller (Nutrient N/P threshold control only)
        # No CO2 control: Spirulina's Zarrouk bicarbonate reservoir self-buffers pH near
        # 9.5 without any active carbon dosing (validated empirically — see genetic_env
        # gas-phase config comment). Only ambient air sparge feeds the carbonate system.
        # Gate on EITHER N or P running low — dosing replenishes both (87% N, 8% P per
        # BG-11 ratio), but N typically depletes slower than P relative to its dose threshold.
        # Gating on N alone left P to starve silently while N sat in the hold band.
```

## ./environments/prod_env.py:471 {#--environments-prod_env-py-471}

```
        # --- Biofouling Accumulation ---
        # Cells adhere to surfaces at low mixing and high biomass density.
        # exp(-0.5) ≈ 60% light transmission at full fouling (cap 0.5).
```

## ./environments/prod_env.py:479 {#--environments-prod_env-py-479}

```
        # --- Physics (Chaotic Turbulence) ---
        # Apply only to active cells
        # We replace simple Brownian motion with structured "Swirls"
        # Flow V(z, t) = Sum( A * sin(k*z - w*t) )
```

## ./environments/prod_env.py:492 {#--environments-prod_env-py-492}

```
            # 1. Aggregation (Sticking) - Orthokinetic + Perikinetic
            # Orthokinetic: Stirring INCREASES collision frequency (Smoluchowski)
            # Sticking = Base (Brownian) + Shear-Induced (RPM)
            # Fix scaling bug: Use true physical OD, not an assumption based on cell count!
```

## ./environments/prod_env.py:498 {#--environments-prod_env-py-498}

```
            # --- New Physics: 
            # - Base chance: 1e-3 (Stronger - 5x boost)
            # - RPM Boost: Increases linearly with Mixing (more collisions)
            # --- Flocculation: Stirring BREAKS clumps (shear dispersal dominates at moderate RPM)
            # rpm_factor now REDUCES sticking — higher RPM = more shear = less aggregation
            # At 0 RPM: rpm_factor=1.0 (max sticking). At 200 RPM: rpm_factor=0.2 (80% less sticking)
```

## ./environments/prod_env.py:517 {#--environments-prod_env-py-517}

```
            # Brownian/diffusive breakup (always active, weak)
            # Prevents runaway aggregation at low RPM
            # Small clumps (1-5) barely affected, large clumps (50+) slowly erode
```

## ./environments/prod_env.py:533 {#--environments-prod_env-py-533}

```
            # --- 2D Kinematic Turbulence (Airlift / Convection Loop) ---
            # Center (x=0.5): Upward Flow (-z)
            # Walls (x=0,1): Downward Flow (+z)
            # Top/Bottom: Turnaround (Horizontal Flow)
```

## ./environments/prod_env.py:541 {#--environments-prod_env-py-541}

```
            # 1. Vertical Velocity (Vz)
            # Cosine profile: Max Up at 0.5, Max Down at 0, 1.
            # Scale: 0.01 m/s * intensity
```

## ./environments/prod_env.py:609 {#--environments-prod_env-py-609}

```
        # 1. Shear Stress (RPM > 400)
        # Random death probability for cells if mixing is too violent
        # Note: We already have this logic downstream at line 550, but let's keep the flow clean.
        # Actually, let's just fall through to the Biology block.
```

## ./environments/prod_env.py:619 {#--environments-prod_env-py-619}

```
            # 1. Spectral Light Field (RGB Physics)
            # Action 'light' sets Total Surface Intensity (PAR)
            # I_surface is already calculated at top of step()
```

## ./environments/prod_env.py:632 {#--environments-prod_env-py-632}

```
            # Attenuation Coefficients (k)
            # Red: Absorbed STRONGLY by Chlorophyll (Growth)
            # k_red boosted to 3.5 (was 10.0) to allow deep biological growth past 12k cells
```

## ./environments/prod_env.py:645 {#--environments-prod_env-py-645}

```
            # ── Turbulent Flash-Light Effect (Biologically Accurate) ──────────
            # In real Spirulina PBRs, turbulent mixing causes cells to cycle
            # between the photic zone (surface) and dark zone (deep) rapidly.
            # This "flash-light effect" dramatically increases photosynthetic
            # efficiency (Kok effect): brief intense surface flashes > sustained dim light.
            #
            # At 0 RPM  : cells see only their actual static depth (fully stratified).
            # At 500 RPM: cells see a near-random depth distribution each step (fully mixed).
```

## ./environments/prod_env.py:694 {#--environments-prod_env-py-694}

```
            # Photo-Inhibition / Shock
            # Cells experience stress when light changes suddenly
            # Scalar reduced from 0.0001 to 0.000001 to prevent startup death
            # At diff=300: Old penalty=99.99%, New penalty=9% (survivable!)
```

## ./environments/prod_env.py:708 {#--environments-prod_env-py-708}

```
            # 3. Growth Rate (Haldane)
            # Growth is driven by RED light availability
            # Inhibition is driven by TOTAL light intensity
            # f_I = I_growth / (Ks + I_growth + I_total^2/Ki)
```

## ./environments/prod_env.py:735 {#--environments-prod_env-py-735}

```
            # pH Inhibition (Asymmetric Gaussian — Arthrospira/Spirulina platensis)
            # Peak at 9.3 (Zarrouk operating range 8.5-11; native soda-lake alkaliphile)
            # Acid side: σ=0.7 — steep falloff below pH 8, intolerant of neutral pH
            # Alkaline side: σ=1.0 — tolerates up to pH 11 with moderate inhibition
```

## ./environments/prod_env.py:744 {#--environments-prod_env-py-744}

```
            # Osmotic Stress — conductivity as ionic strength proxy (all ions: N, P, HCO3-, salts)
            # Spirulina is a soda-lake alkaliphile adapted to high ionic strength; Zarrouk
            # medium baseline is ~19,000 µS/cm (vs BG-11's ~3200). Onset raised accordingly.
            # Uses previous step's conductivity (one-step lag, 72s — negligible).
```

## ./environments/prod_env.py:771 {#--environments-prod_env-py-771}

```
            # Calculate Rate
            # --- Shear Repair Tax (sigmoid, centered at 100 RPM) ---
            # Spirulina (Arthrospira) is a filamentous cyanobacterium — helical trichomes
            # fragment under shear far more readily than Chlorella's rigid unicells.
            # Max 35% penalty at sustained 200 RPM.
```

## ./environments/prod_env.py:779 {#--environments-prod_env-py-779}

```
            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Filament breakage accumulates faster than unicell wall fatigue.
            # Onset at ~80 RPM; max 15% growth penalty.
```

## ./environments/prod_env.py:793 {#--environments-prod_env-py-793}

```
            # Carbon-Limited Growth — Arthrospira/Spirulina has an efficient bicarbonate CCM
            # (active HCO3- transport + carbonic anhydrase), the adaptation that lets it
            # dominate alkaline soda lakes where free CO2 is scarce. HCO3- is the primary
            # DIC source at Zarrouk concentrations (~200 mM); dissolved CO2 contributes little.
```

## ./environments/prod_env.py:827 {#--environments-prod_env-py-827}

```
            # Droop quota dilution: as cells grow, intracellular quota (N/biomass) is diluted.
            # dQ/dt = V(N) - µ*Q; this applies the -µ*Q term per cell.
            # Only positive net_mu dilutes (shrinking cells retain their quota concentration).
```

## ./environments/prod_env.py:834 {#--environments-prod_env-py-834}

```
            # --- PROBABILISTIC LYSIS DEATH (replaces dead-code hard starvation check) ---
            # Background lysis: ~0.5%/day (realistic Spirulina batch culture baseline).
            # Stress lysis: scales up to ~5%/day when mean current_mu < m_respiration.
            # Never a hard cliff — always a smooth gradient signal for the RL agent.
```

## ./environments/prod_env.py:862 {#--environments-prod_env-py-862}

```
            # O4: cells below the death threshold face certain lysis on this cycle
            # Threshold set to 8% of starting mass (1.25e8 pg) — the prior 5e5 floor was
            # unreachable before stochastic lysis killed the cell first (dead code).
```

## ./environments/prod_env.py:886 {#--environments-prod_env-py-886}

```
            # P uptake: Monod saturation with strain-specific Ks_P.
            # Factor 0.0014 = 0.01 / 7.2 (Redfield N:P ratio by mass — a broadly cross-species
            # phytoplankton constant, applies to cyanobacteria as well as green algae)
```

## ./environments/prod_env.py:891 {#--environments-prod_env-py-891}

```
            # nut_flow dosing composition: 79% N, 16% P, 5% inorganic salts — matches Zarrouk
            # stock ratio (NaNO3 2.5 g/L : K2HPO4 0.5 g/L ~ 5:1 N:P by mass, far richer in P
            # than BG-11's ~28:1)
```

## ./environments/prod_env.py:895 {#--environments-prod_env-py-895}

```
            # N waste penalty removed: it caused mode collapse where agent overdosed early,
            # earned heavy penalties, then locked to zero dosing for the entire episode.
            # phi_cur N Gaussian (peak at 200 mg/L) + starvation penalty below provide the equilibrium signal.
```

## ./environments/prod_env.py:904 {#--environments-prod_env-py-904}

```
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e8 pg (12% above init mass of 1.25e8)
            # Use >= to avoid edge case where cells hover exactly at boundary.
```

## ./environments/prod_env.py:942 {#--environments-prod_env-py-942}

```
                # B9 removed: when slots are full, let cells continue growing up to the
                # hard 5e8 cap (line ~775). Capping at the division threshold caused OD
                # to plateau at max_cells, killing all growth reward past population ceiling.
```

## ./environments/prod_env.py:962 {#--environments-prod_env-py-962}

```
        # 2. Gas Exchange (O2 & CO2)
        # Closed-tank model: gas composition is set by baseline air + injected pure CO2.
        # kLa scales with agitation, gas throughput, and broth resistance at high biomass.
        # Bug fix: Use cached self.od instead of calling _get_obs() which
        # would corrupt the pH lag buffer by appending mid-step.
```

## ./environments/prod_env.py:988 {#--environments-prod_env-py-988}

```
        # Dissolved Oxygen Dynamics
        # Production: Proportional to Growth (approx 1.5g O2 per g Biomass)
        # Respiration: Proportional to maintenance (approx 1.0g O2 per g Biomass lost)
        # Calculate net biomass change from biology step (approx)
```

## ./environments/prod_env.py:1008 {#--environments-prod_env-py-1008}

```
        # --- Harvest Dilution (Semi-Continuous Operation) ---
        # Removes a fraction of culture volume and replaces it with fresh Zarrouk medium.
        # Applied after biology (delta_mass_mg is biological growth only — harvest dilution
        # does not count as "stagnation") but before gas exchange (so DO2/CO2 pools are
        # also diluted, matching a real fresh-medium exchange).
```

## ./environments/prod_env.py:1046 {#--environments-prod_env-py-1046}

```
        # 3. 2-Layer Gas Exchange (surface z<10cm = 10L, bulk z>=10cm = 20L)
        # Surface cells photosynthesize more (better light) → O2 accumulates at surface,
        # CO2 depletes there. Mixing inter-layer exchange dissipates gradients at high RPM.
```

## ./environments/prod_env.py:1086 {#--environments-prod_env-py-1086}

```
        # Photosynthetic stoichiometry: 6CO2 → C6H12O6; 6×44/(6×12) = 3.67 mg CO2/mg C fixed.
        # Biomass is ~50% C by dry weight, so per mg DW the CO2 demand is 3.67×0.5 = 1.835 mg CO2/mg DW.
        # Both uptake and release use same ratio: decomposition re-releases the same CO2 per mass.
```

## ./environments/prod_env.py:1101 {#--environments-prod_env-py-1101}

```
        # Bicarbonate balance: depleted by photosynthesis (85% of DIC uptake via HCO3-),
        # replenished by CO2 sparging — fraction that equilibrates to HCO3- depends on pH.
        # 85% fraction matches f_carbon's 0.85 * f_hco3 term (Spirulina CCM, same as growth model).
```

## ./environments/prod_env.py:1141 {#--environments-prod_env-py-1141}

```
        # OD ~ Mass^0.8 (Self-Shading effect)
        # 1e11 cells ~ 1g/L ~ OD 1.0
        # density_gL = (total_mass_mg * 1e-9) / self.volume_L # BUG: 1e-9 is wrong units (pg->mg happened already)
        # turbidity = 1.0 * (density_gL ** 0.8)
```

## ./environments/prod_env.py:1152 {#--environments-prod_env-py-1152}

```
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # At D2, simulate slow natural evolution/mutation of the strain over weeks of deployment.
        # Every ~5 hours (250 steps at dt=0.02h), strain parameters wander by ±1%.
        # This forces the internal LMU to constantly adapt its latent state tracking.
```

## ./environments/prod_env.py:1161 {#--environments-prod_env-py-1161}

```
        # 3. Conductivity — Kohlrausch molar conductance formula (µS/cm)
        # σ (mS/cm) = Σ λᵢ (S·cm²/mol) × cᵢ (mol/L), then ×1000 → µS/cm
        # λ values at 25°C (literature): NO₃⁻=71.4, Na⁺=50.1, HPO₄²⁻=57.0,
        # K⁺=73.5, SO₄²⁻=160.0, Na⁺=50.1, Cl⁻=76.4, OH⁻=198.0, H⁺=349.8
```

## ./environments/prod_env.py:1192 {#--environments-prod_env-py-1192}

```
        # --- Reward ---
        # Semi-continuous operation, agent-controlled harvest within a narrow pre-
        # calibrated band (see HARVEST_MIN/MAX_LPH): 5 components.
```

## ./environments/prod_env.py:1196 {#--environments-prod_env-py-1196}

```
        # 1. Harvested biomass — dense, fires whenever the agent is actively diluting.
        # Divisor rescaled 10.0 -> 0.2: per-step harvest_yield_mg is inherently tiny
        # (dilution_fraction ~ harvest_lph*dt/volume_L ~ 2e-4, times standing mass ~50-400mg
        # gives ~0.015-0.05mg/step). At /10.0 this term never left tanh's near-zero linear
        # region (measured contribution: ~4% of a saturated term, ~18% of total episode
        # reward) even at the empirically-optimal harvest rate — meaning reward_od (a
        # smaller-weighted but better-scaled term) was silently dominating at ~72% of total
        # reward, the opposite of the intended weighting (harvest primary, OD secondary/
        # safety-net). 0.2 was chosen over more aggressive rescalings (0.03-0.06) because
        # those over-saturate tanh and invert the reward ranking relative to actual
        # cumulative harvested mass across harvest rates; 0.2 is the largest-signal divisor
        # that still preserves monotonic tracking of true harvested mass, while restoring
        # harvest as the dominant term (~5-25x reward_od, matching intended weighting).
```

## ./environments/prod_env.py:1219 {#--environments-prod_env-py-1219}

```
        # 2. Standing OD — dense, rewards building/maintaining a productive culture
        # regardless of whether harvest happens to be active this exact step. Without
        # this term the agent has no incentive to build density beyond the trivial
        # washout floor — empirical calibration found peak productivity at OD~0.18-0.24,
        # so that's the reference scale here too.
```

## ./environments/prod_env.py:1231 {#--environments-prod_env-py-1231}

```
        # 4. Stagnation — penalises decline AND flatlining (harvest dilution is excluded
        # since delta_mass_mg is captured before the harvest block runs).
        # Threshold 0.01 sits well above the near-zero drift of a "parked" culture
        # (~0.0001 observed) and well below healthy active growth (~0.02-0.1+), so it
        # closes the flatline loophole without touching normal growth-noise behavior.
```

## ./environments/prod_env.py:1262 {#--environments-prod_env-py-1262}

```
        # Extinction check: population OR total biomass. Cells can hover just above the
        # per-cell starvation threshold (1e7 pg) without individually triggering death,
        # leaving a "zombie" culture of a few surviving cells with near-zero total mass —
        # this stalls the episode at flat negative reward (washout+stagnation) for
        # thousands of steps with no learning signal instead of ending the rollout.
```

## ./environments/prod_env.py:1268 {#--environments-prod_env-py-1268}

```
            # Reduced from -1000: that scale was 300-1000x larger than typical achievable
            # per-episode reward (~10-20), so occasional exploration-driven crashes were
            # corrupting the LSTM's learned weights for millions of steps to recover from
            # (observed repeatedly as regression-recovery cycles during Spirulina training).
            # -100 still clearly signals "bad" without being catastrophically destabilizing.
```

## ./environments/total_env.py:31 {#--environments-total_env-py-31}

```
        # --- Gas-Phase / Carbonate Configuration (closed 30L PBR) ---
        # No CO2 injection: validated empirically that Spirulina's Zarrouk bicarbonate
        # reservoir (~200 mM) self-buffers pH near 9.5 without any active control — across
        # 6000 random-action steps CO2 injection never fired and pH stayed in [8.54, 9.44].
        # Only the baseline ambient-air sparge remains (420ppm atmospheric CO2).
```

## ./environments/total_env.py:51 {#--environments-total_env-py-51}

```
        # --- Batch cycle: single terminal harvest, no agent-controlled dilution ---
        # The real process is a ~144h (6-day) batch cycle: grow undisturbed, harvest once
        # at the end, restart. This replaces the earlier semi-continuous/agent-harvest
        # design (narrow harvest-rate band, per-step dilution physics) which assumed a
        # continuously-diluted turbidostat-style operation that doesn't match the actual
        # process (see docs/real_data_integration.md's example harvest schema: one
        # harvest-measurement row per ~144h run, not a continuous rate).
```

## ./environments/total_env.py:59 {#--environments-total_env-py-59}

```
        # Action: [Stirring, Light] — CO2, Nutrient, and Harvest are all automated/implicit.
        # Harvest is a single event applied automatically at episode end (see step()),
        # not an agent-controlled action.
```

## ./environments/total_env.py:64 {#--environments-total_env-py-64}

```
        # Observation Space (6 Dims)
        # 6D obs — real hardware sensors only:
        # 0: Turbidity (SEN0189, 0-1000 NTU)   1: pH (SEN0161)
        # 2: Harvest integral (pump counter, L) 3: Conductivity (DFR0300)
        # 4: Temperature (DS18B20)               5: Light (BH1750, 0-65535 lux)
        # Dropped: n_pool (no sensor), RGB (unreliable)
```

## ./environments/total_env.py:135 {#--environments-total_env-py-135}

```
        # --- DAY/NIGHT CYCLE ---
        # lights_off_hour: hour of day (0-24) when lights turn off. None = always on (default).
        # lights_on_hour:  hour of day (0-24) when lights come back on.
        # Example: lights_off_hour=20, lights_on_hour=6  ->  14h light / 10h dark.
```

## ./environments/total_env.py:153 {#--environments-total_env-py-153}

```
            # Was N(0.080, 0.015) — ~8.7h doubling. This is faster than the project's OWN
            # cited source (literature.md: Torzillo et al. 1993, mu_max 0.04-0.07 h^-1;
            # the old mean of 0.08 sat above the ENTIRE cited range). Independently
            # confirmed via fresh literature search: real Zarrouk-medium Arthrospira batch
            # studies report generation times of ~2.7-3.2 days (not ~9h), and PBR-optimal
            # cases report specific growth rates as low as ~0.12/day (~0.005/h). Recentered
            # on the cited range's midpoint (~0.055/h, ~12.6h doubling) — still on the
            # faster/optimistic end of real reported values, but no longer contradicting
            # the project's own citation.
```

## ./environments/total_env.py:189 {#--environments-total_env-py-189}

```
        # Super-Agent Scaling: 1 Agent = 2,500,000 Cells (~500pg each)
        # Density-dependent starting mass:
        # At 300 cells, mass is ~1.25e8. At 15,000 cells (Log Ladder limit),
        # mass drops to ~0.8e8 (starving) due to immediate shelf-shading/nutrient competition.
```

## ./environments/total_env.py:382 {#--environments-total_env-py-382}

```
        # Curriculum metric: time-averaged OD over the back half of the episode (steps
        # 3600-7200) — still meaningful in batch mode as a "is the culture on track for a
        # good harvest" proxy that can't be gamed by a brief early spike.
```

## ./environments/total_env.py:397 {#--environments-total_env-py-397}

```
        # 3. Stagnation — penalises decline AND flatlining.
        # Threshold 0.01 sits well above the near-zero drift of a "parked" culture
        # (~0.0001 observed) and well below healthy active growth (~0.02-0.1+), so it
        # closes the flatline loophole without touching normal growth-noise behavior.
```

## ./environments/total_env.py:408 {#--environments-total_env-py-408}

```
        # 5. Terminal harvest bonus — fires once, only on a natural episode end (not a
        # crash), scaled to final standing biomass (the whole-tank yield at harvest).
        # Divisor re-derived (500 -> 650) after the mu_max realism fix (0.08 -> 0.055 h^-1,
        # see _randomize_strain): a fresh stir/light grid sweep (25 combos x 4 seeds, D2)
        # under the corrected growth kinetics found a realistic ceiling of ~717mg mean
        # (best combo: stir=80rpm/light=1000umol; individual episodes 342-1336mg) — about
        # 3x lower than the ~2400mg peak the old (too-fast) kinetics produced, since mu_max
        # compounds over ~12 doublings across 144h. At /500 the new ceiling was pushing
        # into tanh's flatter region (717mg -> arg 1.43, already 89% saturated); /650 keeps
        # more gradient at the realistic top end (717mg -> arg 1.10, 80% saturated) while
        # still giving a meaningful capstone bonus, not dominant over dense reward (see
        # note below on why dense terms are intentionally kept primary).
```

## ./environments/total_env.py:460 {#--environments-total_env-py-460}

```
        # 1b. Automated PID Controller (Nutrient N/P threshold control only)
        # No CO2 control: Spirulina's Zarrouk bicarbonate reservoir self-buffers pH near
        # 9.5 without any active carbon dosing (validated empirically — see genetic_env
        # gas-phase config comment). Only ambient air sparge feeds the carbonate system.
        # Gate on EITHER N or P running low — dosing replenishes both (87% N, 8% P per
        # BG-11 ratio), but N typically depletes slower than P relative to its dose threshold.
        # Gating on N alone left P to starve silently while N sat in the hold band.
```

## ./environments/total_env.py:535 {#--environments-total_env-py-535}

```
        # --- Biofouling Accumulation ---
        # Cells adhere to surfaces at low mixing and high biomass density.
        # exp(-0.5) ≈ 60% light transmission at full fouling (cap 0.5).
```

## ./environments/total_env.py:543 {#--environments-total_env-py-543}

```
        # --- Physics (Chaotic Turbulence) ---
        # Apply only to active cells
        # We replace simple Brownian motion with structured "Swirls"
        # Flow V(z, t) = Sum( A * sin(k*z - w*t) )
```

## ./environments/total_env.py:556 {#--environments-total_env-py-556}

```
            # 1. Aggregation (Sticking) - Orthokinetic + Perikinetic
            # Orthokinetic: Stirring INCREASES collision frequency (Smoluchowski)
            # Sticking = Base (Brownian) + Shear-Induced (RPM)
            # Fix scaling bug: Use true physical OD, not an assumption based on cell count!
```

## ./environments/total_env.py:562 {#--environments-total_env-py-562}

```
            # --- New Physics: 
            # - Base chance: 1e-3 (Stronger - 5x boost)
            # - RPM Boost: Increases linearly with Mixing (more collisions)
            # --- Flocculation: Stirring BREAKS clumps (shear dispersal dominates at moderate RPM)
            # rpm_factor now REDUCES sticking — higher RPM = more shear = less aggregation
            # At 0 RPM: rpm_factor=1.0 (max sticking). At 200 RPM: rpm_factor=0.2 (80% less sticking)
```

## ./environments/total_env.py:581 {#--environments-total_env-py-581}

```
            # Brownian/diffusive breakup (always active, weak)
            # Prevents runaway aggregation at low RPM
            # Small clumps (1-5) barely affected, large clumps (50+) slowly erode
```

## ./environments/total_env.py:597 {#--environments-total_env-py-597}

```
            # --- 2D Kinematic Turbulence (Airlift / Convection Loop) ---
            # Center (x=0.5): Upward Flow (-z)
            # Walls (x=0,1): Downward Flow (+z)
            # Top/Bottom: Turnaround (Horizontal Flow)
```

## ./environments/total_env.py:605 {#--environments-total_env-py-605}

```
            # 1. Vertical Velocity (Vz)
            # Cosine profile: Max Up at 0.5, Max Down at 0, 1.
            # Scale: 0.01 m/s * intensity
```

## ./environments/total_env.py:673 {#--environments-total_env-py-673}

```
        # 1. Shear Stress (RPM > 400)
        # Random death probability for cells if mixing is too violent
        # Note: We already have this logic downstream at line 550, but let's keep the flow clean.
        # Actually, let's just fall through to the Biology block.
```

## ./environments/total_env.py:683 {#--environments-total_env-py-683}

```
            # 1. Spectral Light Field (RGB Physics)
            # Action 'light' sets Total Surface Intensity (PAR)
            # I_surface is already calculated at top of step()
```

## ./environments/total_env.py:696 {#--environments-total_env-py-696}

```
            # Attenuation Coefficients (k)
            # Red: Absorbed STRONGLY by Chlorophyll (Growth)
            # k_red boosted to 3.5 (was 10.0) to allow deep biological growth past 12k cells
```

## ./environments/total_env.py:709 {#--environments-total_env-py-709}

```
            # ── Turbulent Flash-Light Effect (Biologically Accurate) ──────────
            # In real Spirulina PBRs, turbulent mixing causes cells to cycle
            # between the photic zone (surface) and dark zone (deep) rapidly.
            # This "flash-light effect" dramatically increases photosynthetic
            # efficiency (Kok effect): brief intense surface flashes > sustained dim light.
            #
            # At 0 RPM  : cells see only their actual static depth (fully stratified).
            # At 500 RPM: cells see a near-random depth distribution each step (fully mixed).
```

## ./environments/total_env.py:758 {#--environments-total_env-py-758}

```
            # Photo-Inhibition / Shock
            # Cells experience stress when light changes suddenly
            # Scalar reduced from 0.0001 to 0.000001 to prevent startup death
            # At diff=300: Old penalty=99.99%, New penalty=9% (survivable!)
```

## ./environments/total_env.py:772 {#--environments-total_env-py-772}

```
            # 3. Growth Rate (Haldane)
            # Growth is driven by RED light availability
            # Inhibition is driven by TOTAL light intensity
            # f_I = I_growth / (Ks + I_growth + I_total^2/Ki)
```

## ./environments/total_env.py:799 {#--environments-total_env-py-799}

```
            # pH Inhibition (Asymmetric Gaussian — Arthrospira/Spirulina platensis)
            # Peak at 9.3 (Zarrouk operating range 8.5-11; native soda-lake alkaliphile)
            # Acid side: σ=0.7 — steep falloff below pH 8, intolerant of neutral pH
            # Alkaline side: σ=1.0 — tolerates up to pH 11 with moderate inhibition
```

## ./environments/total_env.py:808 {#--environments-total_env-py-808}

```
            # Osmotic Stress — conductivity as ionic strength proxy (all ions: N, P, HCO3-, salts)
            # Spirulina is a soda-lake alkaliphile adapted to high ionic strength; Zarrouk
            # medium baseline is ~19,000 µS/cm (vs BG-11's ~3200). Onset raised accordingly.
            # Uses previous step's conductivity (one-step lag, 72s — negligible).
```

## ./environments/total_env.py:835 {#--environments-total_env-py-835}

```
            # Calculate Rate
            # --- Shear Repair Tax (sigmoid, centered at 100 RPM) ---
            # Spirulina (Arthrospira) is a filamentous cyanobacterium — helical trichomes
            # fragment under shear far more readily than Chlorella's rigid unicells.
            # Max 35% penalty at sustained 200 RPM.
```

## ./environments/total_env.py:843 {#--environments-total_env-py-843}

```
            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Filament breakage accumulates faster than unicell wall fatigue.
            # Onset at ~80 RPM; max 15% growth penalty.
```

## ./environments/total_env.py:857 {#--environments-total_env-py-857}

```
            # Carbon-Limited Growth — Arthrospira/Spirulina has an efficient bicarbonate CCM
            # (active HCO3- transport + carbonic anhydrase), the adaptation that lets it
            # dominate alkaline soda lakes where free CO2 is scarce. HCO3- is the primary
            # DIC source at Zarrouk concentrations (~200 mM); dissolved CO2 contributes little.
```

## ./environments/total_env.py:891 {#--environments-total_env-py-891}

```
            # Droop quota dilution: as cells grow, intracellular quota (N/biomass) is diluted.
            # dQ/dt = V(N) - µ*Q; this applies the -µ*Q term per cell.
            # Only positive net_mu dilutes (shrinking cells retain their quota concentration).
```

## ./environments/total_env.py:898 {#--environments-total_env-py-898}

```
            # --- PROBABILISTIC LYSIS DEATH (replaces dead-code hard starvation check) ---
            # Background lysis: ~0.5%/day (realistic Spirulina batch culture baseline).
            # Stress lysis: scales up to ~5%/day when mean current_mu < m_respiration.
            # Never a hard cliff — always a smooth gradient signal for the RL agent.
```

## ./environments/total_env.py:926 {#--environments-total_env-py-926}

```
            # O4: cells below the death threshold face certain lysis on this cycle
            # Threshold set to 8% of starting mass (1.25e8 pg) — the prior 5e5 floor was
            # unreachable before stochastic lysis killed the cell first (dead code).
```

## ./environments/total_env.py:950 {#--environments-total_env-py-950}

```
            # P uptake: Monod saturation with strain-specific Ks_P.
            # Factor 0.0014 = 0.01 / 7.2 (Redfield N:P ratio by mass — a broadly cross-species
            # phytoplankton constant, applies to cyanobacteria as well as green algae)
```

## ./environments/total_env.py:955 {#--environments-total_env-py-955}

```
            # nut_flow dosing composition: 79% N, 16% P, 5% inorganic salts — matches Zarrouk
            # stock ratio (NaNO3 2.5 g/L : K2HPO4 0.5 g/L ~ 5:1 N:P by mass, far richer in P
            # than BG-11's ~28:1)
```

## ./environments/total_env.py:959 {#--environments-total_env-py-959}

```
            # N waste penalty removed: it caused mode collapse where agent overdosed early,
            # earned heavy penalties, then locked to zero dosing for the entire episode.
            # phi_cur N Gaussian (peak at 200 mg/L) + starvation penalty below provide the equilibrium signal.
```

## ./environments/total_env.py:968 {#--environments-total_env-py-968}

```
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e8 pg (12% above init mass of 1.25e8)
            # Use >= to avoid edge case where cells hover exactly at boundary.
```

## ./environments/total_env.py:1006 {#--environments-total_env-py-1006}

```
                # B9 removed: when slots are full, let cells continue growing up to the
                # hard 5e8 cap (line ~775). Capping at the division threshold caused OD
                # to plateau at max_cells, killing all growth reward past population ceiling.
```

## ./environments/total_env.py:1014 {#--environments-total_env-py-1014}

```
            # shock_factor is otherwise only assigned in the num_active>0 branch above;
            # defining it here guarantees it always exists by the time the reward/debug
            # section reads it (previously relied on lazy ternary short-circuit evaluation
            # at each read site — fragile if that code ever gets refactored/extracted).
```

## ./environments/total_env.py:1025 {#--environments-total_env-py-1025}

```
        # 2. Gas Exchange (O2 & CO2)
        # Closed-tank model: gas composition is set by baseline air + injected pure CO2.
        # kLa scales with agitation, gas throughput, and broth resistance at high biomass.
        # Bug fix: Use cached self.od instead of calling _get_obs() which
        # would corrupt the pH lag buffer by appending mid-step.
```

## ./environments/total_env.py:1051 {#--environments-total_env-py-1051}

```
        # Dissolved Oxygen Dynamics
        # Production: Proportional to Growth (approx 1.5g O2 per g Biomass)
        # Respiration: Proportional to maintenance (approx 1.0g O2 per g Biomass lost)
        # Calculate net biomass change from biology step (approx)
```

## ./environments/total_env.py:1071 {#--environments-total_env-py-1071}

```
        # --- Batch mode: no mid-episode dilution/harvest physics ---
        # The culture grows undisturbed for the full 144h cycle; harvest is a single
        # terminal event applied in the reward block below, not a per-step process.
```

## ./environments/total_env.py:1079 {#--environments-total_env-py-1079}

```
        # 3. 2-Layer Gas Exchange (surface z<10cm = 10L, bulk z>=10cm = 20L)
        # Surface cells photosynthesize more (better light) → O2 accumulates at surface,
        # CO2 depletes there. Mixing inter-layer exchange dissipates gradients at high RPM.
```

## ./environments/total_env.py:1119 {#--environments-total_env-py-1119}

```
        # Photosynthetic stoichiometry: 6CO2 → C6H12O6; 6×44/(6×12) = 3.67 mg CO2/mg C fixed.
        # Biomass is ~50% C by dry weight, so per mg DW the CO2 demand is 3.67×0.5 = 1.835 mg CO2/mg DW.
        # Both uptake and release use same ratio: decomposition re-releases the same CO2 per mass.
```

## ./environments/total_env.py:1134 {#--environments-total_env-py-1134}

```
        # Bicarbonate balance: depleted by photosynthesis (85% of DIC uptake via HCO3-),
        # replenished by CO2 sparging — fraction that equilibrates to HCO3- depends on pH.
        # 85% fraction matches f_carbon's 0.85 * f_hco3 term (Spirulina CCM, same as growth model).
```

## ./environments/total_env.py:1142 {#--environments-total_env-py-1142}

```
        # NOTE: this ceiling (5.0) is 40x below the Zarrouk medium baseline bicarbonate is
        # reset to (200.0 mM, see reset()) — confirmed via direct testing that it's NOT
        # merely a cosmetic mismatch: raising the ceiling to let bicarbonate sit near its
        # true 200 mM value pushes the Henderson-Hasselbalch pH equilibrium up to ~10.5
        # and holds it there (vs. the intended 9.3-9.5 Spirulina optimum), cutting terminal
        # batch yield roughly in half in direct testing (892mg -> 393mg, same seed/policy).
        # The pH-equilibrium constants (pKa1, co2_aq scaling) were evidently never
        # validated against the documented 200 mM baseline — this clip has been doing load-
        # bearing (if accidental) work keeping pH in the growth-viable range. Left at 5.0
        # deliberately: fixing this properly requires re-deriving the carbonate-system
        # constants against the correct bicarbonate scale, not just widening this clip.
        # Flagged as a real follow-up, not resolved here.
```

## ./environments/total_env.py:1186 {#--environments-total_env-py-1186}

```
        # OD ~ Mass^0.8 (Self-Shading effect)
        # 1e11 cells ~ 1g/L ~ OD 1.0
        # density_gL = (total_mass_mg * 1e-9) / self.volume_L # BUG: 1e-9 is wrong units (pg->mg happened already)
        # turbidity = 1.0 * (density_gL ** 0.8)
```

## ./environments/total_env.py:1197 {#--environments-total_env-py-1197}

```
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # At D2, simulate slow natural evolution/mutation of the strain over weeks of deployment.
        # Every ~5 hours (250 steps at dt=0.02h), strain parameters wander by ±1%.
        # This forces the internal LMU to constantly adapt its latent state tracking.
```

## ./environments/total_env.py:1206 {#--environments-total_env-py-1206}

```
        # 3. Conductivity — Kohlrausch molar conductance formula (µS/cm)
        # σ (mS/cm) = Σ λᵢ (S·cm²/mol) × cᵢ (mol/L), then ×1000 → µS/cm
        # λ values at 25°C (literature): NO₃⁻=71.4, Na⁺=50.1, HPO₄²⁻=57.0,
        # K⁺=73.5, SO₄²⁻=160.0, Na⁺=50.1, Cl⁻=76.4, OH⁻=198.0, H⁺=349.8
```

## ./environments/total_env.py:1242 {#--environments-total_env-py-1242}

```
        # Extinction check: population OR total biomass. Cells can hover just above the
        # per-cell starvation threshold (1e7 pg) without individually triggering death,
        # leaving a "zombie" culture of a few surviving cells with near-zero total mass —
        # this stalls the episode at flat negative reward (washout+stagnation) for
        # thousands of steps with no learning signal instead of ending the rollout.
```

## ./environments/total_env.py:1248 {#--environments-total_env-py-1248}

```
            # Reduced from -1000: that scale was 300-1000x larger than typical achievable
            # per-episode reward (~10-20), so occasional exploration-driven crashes were
            # corrupting the LSTM's learned weights for millions of steps to recover from
            # (observed repeatedly as regression-recovery cycles during Spirulina training).
            # -100 still clearly signals "bad" without being catastrophically destabilizing.
```

## ./experiments/env_diagnosis/diagnose.py:71 {#--experiments-env_diagnosis-diagnose-py-71}

```
# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP 1 — REWARD MAGNITUDE / OUTLIER AUDIT
# ═══════════════════════════════════════════════════════════════════════════
```

## ./experiments/env_diagnosis/diagnose.py:112 {#--experiments-env_diagnosis-diagnose-py-112}

```
# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP 2 — ENVIRONMENT: CRASH RATE BY INIT-POPULATION BUCKET x DIFFICULTY
# ═══════════════════════════════════════════════════════════════════════════
```

## ./experiments/env_diagnosis/diagnose.py:142 {#--experiments-env_diagnosis-diagnose-py-142}

```
# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP 3 — ACTIONS: HARVEST-FRACTION CRASH BOUNDARY
# ═══════════════════════════════════════════════════════════════════════════
```

## ./experiments/harvest_ablation/deterministic_eval_harvest_fixed.py:40 {#--experiments-harvest_ablation-deterministic_eval_harvest_fixed-py-40}

```
    # .unwrapped (not .env) — Monitor wraps HarvestFixedWrapper wraps the raw env here,
    # one layer deeper than the original deterministic_eval.py's Monitor(raw env), so a
    # single .env unwrap lands on HarvestFixedWrapper, not GeneticPhotobioreactorEnv.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:2 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:25 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-25}

```
# Re-exported here (not just used internally) so that:
#  1) old checkpoints pickled while this file was a monolith and recorded
#     `__main__.ActionSmoothnessWrapper` etc. can still resolve those names
#     when this script is run directly as `python recurrent_ppo.py`.
#  2) other scripts (e.g. evaluate_agent.py) that import these names from
#     `recurrent_ppo` keep working unchanged.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:62 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-62}

```
# Linear LR decay, driven by OUR OWN steps_done/TOTAL_TRAINING_STEPS tracking rather than
# SB3's built-in progress_remaining. SB3 computes progress_remaining from the
# total_timesteps argument passed to THIS model.learn() call, not the grand training
# budget — and this codebase calls learn() once per 100k-step chunk with
# reset_num_timesteps=False. Verified against SB3's source before wiring this in: each
# chunk call re-derives its own "total_timesteps" as num_timesteps-so-far + this chunk's
# size, so progress_remaining restarts near 1.0 at the start of every chunk and always
# hits 0 by the end of that same chunk — a 40-chunk sawtooth, not a smooth 4M-step decay.
# A naive SB3 schedule would have silently cut the LR toward its floor inside nearly every
# chunk rather than only near the true end of training. Sidestepped the same way this file
# already handles ent_coef (see model.ent_coef assignment in the chunk loop): an external,
# manually-updated value the schedule function reads from, refreshed once per chunk from
# the real steps_done/TOTAL_TRAINING_STEPS ratio.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:79 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-79}

```
# Fix #17 (v20): the behaviour-cloned controller's held-out D2-passing scores, printed beside
# every deterministic eval as a fixed reference line. That policy (model_data/
# BEST_bc_clone_D2_validated/) is the best this project has produced — median harvest 109.4mg,
# p25 63.8, time_avg_od 0.0191, 0% crash over 40 held-out seeds — and it required no RL at all.
# Showing it inline makes "is PPO anywhere near the thing we already have?" answerable at a
# glance instead of by cross-referencing a separate document mid-run.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:87 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-87}

```
# Fix #23 (v25): which policy the curriculum gate advances on. "dual" = stochastic AND
# deterministic (default; see the conjunction site below for the full rationale).
# "stochastic" = stochastic only, for a self-consistent stochastic-deployment experiment.
# Override per run with the GATE_MODE environment variable so no source edit is needed:
#     GATE_MODE=stochastic python training/recurrent_ppo.py
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:96 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-96}

```
# Fix #24 (v26): explicit, RECORDED seed. Until now nothing seeded numpy/torch/the env, so two
# runs of the same configuration differed by an unknown mixture of config effect and RNG draw.
# That directly weakens a conclusion already reported: "v21's time_avg_od 0.0094 was not
# reproducible" rested on v23 (same config) returning 0.0066 — but with no seed control, that
# spread cannot be attributed to the configuration rather than the seed. With the seed pinned
# and logged, a replication isolates config effects, and a deliberate seed sweep measures RNG
# variance separately. Set RUN_SEED to compare configurations; vary it to measure variance.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:120 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-120}

```
    # Fix #24 (v26): seed numpy, torch and Python's RNG before anything samples. SB3's
    # set_random_seed covers all three plus CUDA; the env and action space are seeded separately
    # below since they draw from their own generators.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:152 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-152}

```
    # Fix #17 (v20): preserve the run's best DEPLOYABLE (deterministic) policy, so a run that
    # degrades still yields its peak rather than its final weights. See the scoring comment
    # at the [BEST-DET] block below.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:158 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-158}

```
    # Fix #29: early-stop signal for a sustained deterministic-gate failure streak AT D0.
    # capability_fail_streak below is gated on current_difficulty>0 for the DEMOTION branch
    # (there is no tier below D0 to demote to), but that guard also meant D0 had NO active
    # response to an in-place policy collapse at all. Confirmed live in v29: PPO's det crash
    # rate climbed 0%->80% over 5 chunks while capability_fail_streak sat structurally stuck
    # at 0 (the guard prevented it from ever incrementing) and the already-active plateau-kick
    # mechanism (entropy bumps, unrelated to this failure mode) kept firing on its own schedule
    # without arresting the decline. Rather than attempt a live mid-training weight reload
    # (risky: SB3 optimizer/rollout-buffer state can desync from a hot-swapped policy), this
    # stops the run cleanly with a clear diagnostic once the same failure signal that would
    # demote at D1/D2 sustains at D0 — burning the rest of an 8M-step budget on a policy known
    # to be failing its own gate is worse than stopping and deciding explicitly what to do next.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:212 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-212}

```
            # Belt-and-suspenders: custom_objects above should already install this, but
            # explicitly re-assign in case a saved checkpoint's pickled schedule survives
            # deserialization instead of being overridden — the sawtooth bug this schedule
            # exists to avoid would otherwise silently return on any future resume.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:273 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-273}

```
            # Fix #13 (v18): gamma 0.995 -> 0.9995. This is a CREDIT-ASSIGNMENT fix, arrived at
            # only after two competing hypotheses were measured and REFUTED:
            #
            #   (a) reward-structure exploit — refuted by reward_ab.py: on 8 identical episodes
            #       the reward function ranks the scripted expert +313 ABOVE v17 (1079 vs 766),
            #       entirely via reward_od. reward_biomass contributed 11.8, not the ~1440 its
            #       theoretical ceiling suggested (tanh(per_cell_growth/5) is tiny at realistic
            #       growth rates, and the flat -0.010 penalty offsets most of the rest), and
            #       differed between the two policies by 0.4. The reward is NOT exploitable.
            #   (b) exploration noise making the expert's strategy unachievable — refuted by
            #       noise_sensitivity.py: the expert keeps 94.8% of its noise-free reward at
            #       sigma=0.50 (exactly the train/std v15/v16b/v17 all sat at) and dominates
            #       v17 at EVERY sigma from 0.0 to 0.70. No crossover. Entropy left untouched.
            #
            # What the evidence does point at: v17 learned stir and light CORRECTLY (light
            # settled at ~1000umol, the sweep optimum) and only harvest incorrectly. The
            # distinguishing feature is credit frequency. Stir and light act on all 7200 steps;
            # the harvest action is applied only on the 12 event steps
            # (HARVEST_INTERVAL_STEPS=600), so on 7188 of 7200 steps the policy emits a harvest
            # value the env ignores while PPO still assigns it advantage — 599 of every 600
            # gradient samples on that dimension are spurious credit.
            #
            # gamma compounds this. At 0.995 the effective horizon is 1/(1-gamma) = 200 steps,
            # while a harvest decision's consequence unfolds over the following 600+ steps and
            # compounds for thousands. 0.995^600 = 0.049, so the immediate harvest reward is
            # undiscounted while its OD cost is ~95% invisible — the agent cannot see past the
            # current harvest cycle, which is precisely the trade-off the expert exploits
            # (forgo harvest now, hold OD, harvest more across the remaining ~100h).
            # At 0.9995: horizon 2000 steps (~3.3 harvest cycles) and 0.9995^600 = 0.741, a 15x
            # improvement in the visibility of the next cycle. Not pushed to 0.9999 (horizon
            # 10000 steps, beyond the 7200-step episode) to avoid the value-variance blowup that
            # very-near-1 discounting causes; 0.9995 matches the task's actual causal timescale.
            # Consistent with every failure this session having been in the harvest dimension
            # specifically, in whichever direction the local gradient happened to favour.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:309 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-309}

```
            # target_kl=0.02 tested in v12 and DISABLED after a clear regression: det crash
            # rate climbed to 73.3% by chunk 7 (vs. v11's clean 0% crash at a comparable
            # point under the identical reward config, no other change), ep_rew_mean sat
            # deeply negative and flat (~-50) instead of the healthy early climb v11 showed,
            # and an entropy plateau-kick made it worse, not better. Hypothesis: at 0.02,
            # target_kl's per-minibatch early-stopping fired often enough (observed
            # "Early stopping ... max kl: 0.03-0.09" on most iterations) to cut PPO's 4
            # nominal epochs short most of the time, starving the policy of the gradient
            # steps needed to correct crash-prone behavior during early, fast-changing
            # training — exactly when full updates matter most. Not re-tried at a looser
            # value yet; disabled (None) so the concurrent LR-decay change could be tested
            # in isolation as v13. Re-enable only as a deliberate, isolated test, not
            # bundled with other changes — same lesson as the reward-weight guessing
            # earlier this session: change one variable at a time.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:352 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-352}

```
    # Instantiated once and reused across the whole chunk loop (not recreated per chunk)
    # so its per-difficulty rolling history (maxlen=MASTERY_WINDOW) survives chunk
    # boundaries — see EpisodeMetricsCallback docstring in callbacks.py.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:357 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-357}

```
    # Persistent, per-difficulty rolling history of DETERMINISTIC evaluation episodes —
    # see deterministic_eval.py. Separate from metrics_cb's stochastic history; advancement
    # requires both gates to pass, closing the exploration-noise loophole found this session.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:365 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-365}

```
        # Difficulty is now sampled per-episode in CurriculumStartWrapper.reset().
        # train_diff here equals mastery level and is used only for the streak
        # accounting check (criteria_passed and train_diff == current_difficulty).
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:389 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-389}

```
        # Linear LR decay across the TRUE 4M-step budget (steps_done, not SB3's per-call
        # progress_remaining — see _lr_schedule_fn comment for why). Computed from
        # steps_done at the START of this chunk, same timing convention as ent_coef above.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:396 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-396}

```
        # Fix #22 (v24): anneal a hard cap on actor std over the back half of training, so the
        # MEAN policy converges toward the sampled policy. The deterministic gate and any real
        # deployment use the mean; PPO optimises the samples. `train/std` sat at ~0.50-0.54 in
        # every run to date and the reactive std-band controller never brought it down, so this
        # is applied as an explicit schedule. See entropy_schedule.annealed_std_cap.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:484 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-484}

```
        # Deterministic evaluation pass (see deterministic_eval.py): a handful of genuinely
        # deterministic episodes per chunk, gated the same way as the stochastic rollout
        # above. Closes the exploration-noise loophole confirmed this session — a policy
        # whose deterministic (mean) action never harvests can still look fine under the
        # stochastic gate purely from action-sampling noise around that mean.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:505 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-505}

```
        # ── Fix #17 (v20): best-deterministic checkpoint tracking ─────────────────────────
        # v17, v18 and v19 ALL ended at or near their worst deterministic policy of the run,
        # because training simply stops at the budget with whatever weights it currently has.
        # v19 is the clearest case: its deterministic policy was 149.1mg / od 0.0203 / 0% crash
        # at chunk 1 and 28.3mg / od 0.0002 / 80-93% crash at chunk 80 — the run PRODUCED a good
        # policy and then threw it away. Nothing in the loop preserved it.
        #
        # Score = median harvested_mg scaled by how well time_avg_od meets the CURRENT tier's
        # threshold, hard-zeroed on any crash. Crash-zeroing is deliberate: a policy that
        # crashes is unusable regardless of yield, and crash rate is the metric that exposed
        # v19's collapse while the stochastic gate stayed clean. Tracked across the whole run
        # (not per-tier) so the artifact is simply "the best deployable policy this run found".
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:559 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-559}

```
        # Fix #23 (v25): GATE_MODE selects which policy the curriculum advances on.
        #
        # "dual" (default, v5-v24): the stochastic-rollout gate AND the deterministic gate must
        #   both pass. Added because v4 declared D2 mastery on stochastic metrics inflated by
        #   exploration noise, then scored median 0.4mg against a 90mg gate on held-out data.
        #
        # "stochastic" (v25): advance on the stochastic gate alone. The point is NOT that the
        #   deterministic check was wrong — v24 proved the deterministic policy really is far
        #   worse, because the harvest action is clipped at 0 and E[clip(x)] != clip(E[x]) when
        #   the mean sits near that floor, so the sampled policy gets a systematic upward
        #   harvest bias that survives interval-averaging. The point is CONSISTENCY: the real
        #   error in v14/v17 was gating on one policy while validating with another
        #   (held_out_sweep.py is deterministic). If a stochastic controller is acceptable to
        #   deploy, then gating stochastically is legitimate — provided validation is ALSO
        #   stochastic. `held_out_sweep.py --stochastic` exists for exactly that.
        #   Two caveats that remain true in this mode and must be stated with any result:
        #     * the criterion's difficulty drifts, because stochastic metrics depend on
        #       train/std, which the entropy schedule moves during the run;
        #     * these are the rollouts being trained on, so the metric is optimistically
        #       biased in the same way training accuracy is.
        #   The deterministic eval still RUNS and is still logged, so the gap stays visible; it
        #   just no longer blocks advancement.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:597 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-597}

```
                    # Terminal tier — nowhere further to advance, but this is the
                    # "sustained mastery at full difficulty" signal used for early
                    # stopping (see the while-loop condition below).
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:605 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-605}

```
            # Demotion: sustained high crash rate at D1/D2 drops back one level.
            #
            # Fix #15 (v18): ALSO demote on sustained CAPABILITY failure, not just crashes.
            # v17 exposed the gap concretely: it advanced to D2 with a genuinely good policy
            # (det harvest 113mg, time_avg_od 0.0215), then degraded across the following 48
            # D2 chunks to harvest 72-80mg / time_avg_od 0.0022 — failing the SAME criterion
            # (time_avg_od) on all 48 of them — while crash rate stayed at exactly 0.00%.
            # Because demotion keyed only on crash_rate, nothing ever walked it back down, and
            # it burned ~4.8M steps sitting at a tier it could no longer do. Held-out validation
            # then failed at BOTH D1 and D2.
            #
            # A tier the policy cannot satisfy is not a useful training distribution: dropping
            # back one level restores a solvable task and lets it re-earn the advance. Keyed on
            # the DETERMINISTIC gate (det_criteria_passed) rather than the stochastic one, since
            # deterministic behaviour is what the held-out validation and any real deployment
            # actually use. Threshold is deliberately long (CAPABILITY_DEMOTION_CHUNKS) so
            # ordinary chunk-to-chunk noise or a normal pre-advance plateau cannot trigger it —
            # only a sustained inability to perform at the current tier.
            # Fix #29: no longer gated on current_difficulty>0 — see d0_capability_abort's
            # definition above for why. The counter now tracks sustained det-gate failure at
            # ANY tier, including D0; only the RESPONSE differs below (demote vs. abort),
            # since D0 has no tier to demote to.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:652 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-652}

```
                    # Fix #29 correction (v30 retry): capability_failing alone means "hasn't
                    # cleared the det gate yet", not "is collapsing" — live-verified this fires
                    # on a run with 0.00% crash rate and steadily growing harvest (38.8/30.0,
                    # p25 29.8/15.0, both passing; only time_avg_od lagging, 0.0007/0.0040) just
                    # as readily as on genuine collapse (the original v29 trigger: crash rate
                    # climbing 0%->80% with harvest/od both declining). The D1/D2 demotion
                    # branch above deliberately does NOT require a crash floor (Fix #15 exists
                    # specifically to catch v17-style quality regression at 0% crash), but that
                    # rationale doesn't transfer to D0: D1/D2 demotion had a proven prior-good
                    # baseline to fall back to, while a D0 run stuck below the OD bar from the
                    # start has no such baseline to compare against — "never yet passed" and
                    # "regressed from passing" are not the same signal at the floor tier. Only
                    # abort D0 when crash rate is ALSO elevated, the same threshold and
                    # rationale used for D1/D2's crash-based demotion_streak.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:674 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-674}

```
                # else: sustained det-gate failure at D0 but crash rate is healthy — this is
                # "slow but not broken" (the v30 case above), not the collapse pattern this
                # exists to catch. Keep training; capability_fail_streak stays pinned at/above
                # threshold and is silently re-checked each subsequent chunk at no real cost.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:696 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-696}

```
                # Cap the plateau multiplier so ent_coef stays below 50% of ENTROPY_MAX.
                # Absolute cap = 0.5 * ENTROPY_MAX / decayed_base. Prevents runaway regardless
                # of how far the base has decayed. High entropy ≠ useful exploration when stuck.
                #
                # Fix #12 (v16): the boost is also budgeted — MAX_PLATEAU_KICKS_PER_DIFFICULTY
                # kicks per difficulty, then plateau chunks stop touching entropy at all so the
                # policy is allowed to converge. See curriculum_schedule.py for the rationale.
```

## ./experiments/harvest_ablation/recurrent_ppo_harvest_fixed.py:848 {#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-848}

```
    # Lower LR for fine-tuning: avoids catastrophic forgetting of curriculum knowledge.
    # Setting model.learning_rate here is a no-op — SB3 reads model.lr_schedule during
    # training, not the raw attribute, and lr_schedule is already fixed at load time to
    # _lr_schedule_fn (re-pickled from the checkpoint), which reads the module-level
    # _lr_state dict instead. Update that directly so the value actually takes effect.
```

## ./legacy/recurrent_sac.py:70 {#--legacy-recurrent_sac-py-70}

```
# ═════════════════════════════════════════════════════════════════════════════
#  NETWORKS
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/recurrent_sac.py:216 {#--legacy-recurrent_sac-py-216}

```
# ═════════════════════════════════════════════════════════════════════════════
#  SEQUENCE REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/recurrent_sac.py:292 {#--legacy-recurrent_sac-py-292}

```
# ═════════════════════════════════════════════════════════════════════════════
#  SAC UPDATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/recurrent_sac.py:348 {#--legacy-recurrent_sac-py-348}

```
# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/recurrent_sac.py:398 {#--legacy-recurrent_sac-py-398}

```
    # ══════════════════════════════════════════════════════════════════════════
    #  CURRICULUM LOOP
    # ══════════════════════════════════════════════════════════════════════════
```

## ./legacy/recurrent_sac.py:587 {#--legacy-recurrent_sac-py-587}

```
# ═════════════════════════════════════════════════════════════════════════════
#  FINE-TUNE MODE
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/recurrent_sac.py:655 {#--legacy-recurrent_sac-py-655}

```
# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/ssm_core.py:69 {#--legacy-ssm_core-py-69}

```
        # 3. Parallel Vectorized Scan (replaces the slow Python for-loop)
        # We want to compute: h_t = dA_t * h_{t-1} + dB_t * x_t
        # This is a first-order linear recurrence. We can solve it in parallel 
        # using the general formula:
        # h_t = \sum_{i=0}^t ( \prod_{j=i+1}^t dA_j ) * (dB_i * x_i)
```

## ./legacy/ssm_core.py:78 {#--legacy-ssm_core-py-78}

```
        # We need to compute the cumulative product of dA. 
        # Since dA is strictly positive (it's exp(something)), we can use cumsum in log-space 
        # for numerical stability and massive speedup.
```

## ./legacy/ssm_core.py:89 {#--legacy-ssm_core-py-89}

```
        # To avoid exp() exploding, we factor it out:
        # h_t = \sum_{i=0}^t exp( log_dA_cumsum[t] - log_dA_cumsum[i] + log(inputs_i) ) -> wait, inputs can be negative.
        # Better: h_t = exp(log_dA_cumsum[t]) * \cumsum( exp(-log_dA_cumsum_i) * inputs_i )
```

## ./legacy/TD3.py:69 {#--legacy-TD3-py-69}

```
# TD3+BC actor regularization (Fujimoto & Gu 2021):
#   actor_loss = -lambda * Q1(s, pi(s)) + BC_COEF * MSE(pi(s_demo), a_demo)
#   lambda = TD3BC_ALPHA / mean(|Q1(s, pi(s))|).detach()  (normalizes Q-term to MSE's scale)
# BC term is evaluated on a fresh demo-only batch, not the mixed batch, so it never clones
# this run's own online actions.
```

## ./legacy/TD3.py:92 {#--legacy-TD3-py-92}

```
# Default budget is smaller than PPO/TD-MPC2's 8M-step convention: this file's per-step
# recurrent actor+twin-critic update measured well under 10 it/s on this CPU-only machine
# (>1 week for 8M steps). Override via TD3_STEPS. The dual-gate apparatus is chunk-based,
# so a smaller budget still produces a valid, honestly-reported outcome.
```

## ./legacy/TD3.py:107 {#--legacy-TD3-py-107}

```
# ═════════════════════════════════════════════════════════════════════════════
#  NETWORKS
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD3.py:181 {#--legacy-TD3-py-181}

```
# ═════════════════════════════════════════════════════════════════════════════
#  SEQUENCE REPLAY BUFFER (episode-based, truncated-BPTT sampling)
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD3.py:186 {#--legacy-TD3-py-186}

```
    # Harvest fires every HARVEST_INTERVAL_STEPS; uniform window sampling gives a
    # SEQ_LEN=25 window only ~4% odds of containing one at all (v33's collapse). Bias
    # sampling toward windows that include a harvest step.
```

## ./legacy/TD3.py:272 {#--legacy-TD3-py-272}

```
# ═════════════════════════════════════════════════════════════════════════════
#  SCRIPTED-EXPERT DEMONSTRATION COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD3.py:335 {#--legacy-TD3-py-335}

```
# ═════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC EVAL EPISODE (dual-gate — noise-free rollout)
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD3.py:368 {#--legacy-TD3-py-368}

```
# ═════════════════════════════════════════════════════════════════════════════
#  TD3 UPDATE
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD3.py:394 {#--legacy-TD3-py-394}

```
    # Huber, not MSE: genetic_env.py's crash penalty (-100) is a ~700x outlier against
    # typical per-step reward (experiments/env_diagnosis/), and GAMMA's long bootstrap
    # horizon spreads it across many Q-targets. Huber caps that outlier's gradient
    # contribution to linear instead of quadratic; identical to MSE for normal TD-errors.
```

## ./legacy/TD3.py:432 {#--legacy-TD3-py-432}

```
# ═════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD3.py:487 {#--legacy-TD3-py-487}

```
# ═════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
```

## ./legacy/TD_MPC2.py:21 {#--legacy-TD_MPC2-py-21}

```
# Project curriculum gate — this file used to keep its own local ADVANCE_TARGETS keyed on
# median_od only. Rewired to the same gate PPO uses (harvest_mg / p25 / time_avg_od / crash)
# so a TD-MPC2 result is directly comparable to every PPO run in finalresults.md.
```

## ./legacy/TD_MPC2.py:30 {#--legacy-TD_MPC2-py-30}

```
# Fix (v27): action space was 4D [Stir, Light, Nutrient, CO2] — written against a pre-redesign
# env with manual CO2/nutrient dosing. The live env (genetic_env.py) has automated PID N/P
# dosing, no CO2 injection, and a 3D action space [stir, light, harvest]. This file could not
# construct the env at all with ACTION_DIM=4.
```

## ./legacy/TD_MPC2.py:37 {#--legacy-TD_MPC2-py-37}

```
# Fix (v27): world-model MACRO-TIMESTEP. Each MPPI horizon step previously corresponded to one
# RAW env step (dt=0.02h), so horizon=24 saw only 0.48h ahead — the harvest event fires every
# HARVEST_INTERVAL_STEPS=600 raw steps (12h), so the planner was structurally blind to the one
# decision that has failed in every PPO run in this project (v4 through v24). Extending raw
# horizon to 600 was measured and rejected: cost scales ~linearly-to-superlinear with horizon,
# so h=600 vs h=24 projects to roughly 25x the already-measured 13h planning cost alone.
# Instead the dynamics/reward model is trained on MACRO-transitions spanning MACRO_STEPS raw
# steps (action held constant across the block, reward = discounted sum over the block). A
# planner horizon of 12 macro-steps then sees 12*MACRO_STEPS raw steps ahead. At MACRO_STEPS=50,
# horizon=12 -> 600 raw steps = exactly one harvest interval, at unchanged per-call planning
# cost. This also cuts the (measured, dominant) update() cost: replay stores ~1.5M/50=30,000
# macro-transitions instead of 1.5M raw ones, since update() is called once per macro-transition
# now rather than once per raw step.
```

## ./legacy/TD_MPC2.py:52 {#--legacy-TD_MPC2-py-52}

```
# Module-level (not local to train_td_mpc2) because TDMPC2Agent.update() also needs it for the
# block-length-adjusted Bellman bootstrap (GAMMA ** MACRO_STEPS) — a class method can't see a
# training-function-local variable.
```

## ./legacy/TD_MPC2.py:57 {#--legacy-TD_MPC2-py-57}

```
# Fix (v27): was 300_000 — measured directly (env.step() timing, not assumed) to cost 13.7ms/
# call vs 4.6ms/call at 7_500, a 3x per-step physics overhead, while ACTUAL active population
# in both cases never exceeded ~2,990 cells. max_cells is an array-allocation/masking cap, not
# something that changes physics outcomes below the cap, so 300_000 was buying nothing while
# tripling the dominant cost component (env.step() turned out to be far more expensive than
# plan()+update()+compressor combined — a cost this project's first TD-MPC2 measurement missed
# entirely by never timing env.step() in isolation). 7_500 matches max_cells everywhere else in
# this project (PPO's env_factory.py, all diagnostics), for direct comparability.
```

## ./legacy/TD_MPC2.py:201 {#--legacy-TD_MPC2-py-201}

```
        # ── Learnable Readout Network ──
        # Takes the raw observation (Dim) + the flattened LMU memory (Dim * Order)
        # and projects it to the 64D feature map the rest of the model expects.
```

## ./legacy/TD_MPC2.py:222 {#--legacy-TD_MPC2-py-222}

```
        # A_curr shape: (obs_dim, order, order)
        # B_curr shape: (obs_dim, order)
        # self.A_base is (16, 16). self.B_base is (16, 1)
```

## ./legacy/TD_MPC2.py:228 {#--legacy-TD_MPC2-py-228}

```
        # (Note: Using simple Forward Euler discretisation here for dynamic stability)
        # A_discrete = I + A_curr * dt (where dt=1 in simulation steps)
        # B_discrete = B_curr * dt
```

## ./legacy/TD_MPC2.py:242 {#--legacy-TD_MPC2-py-242}

```
            # m_t_minus_1 shape: (BATCH, OBS_DIM, ORDER)
            # A shape: (ORDER, ORDER)
            # Deal with potential unbatched inputs (e.g. from single-step env interaction without unsqueeze)
```

## ./legacy/TD_MPC2.py:279 {#--legacy-TD_MPC2-py-279}

```
        # ── Simplicial Normalization (SimNorm) ──
        # Projects the unbounded latent vector onto a positive simplex.
        # This provides a bounded state space for the dynamics model, vastly 
        # improving sample efficiency and preventing exploding latents.
```

## ./legacy/TD_MPC2.py:372 {#--legacy-TD_MPC2-py-372}

```
        # vmin/vmax are in SYMLOG units, not raw units — symlog(x)=sign(x)*log(|x|+1), so
        # +-20 symlog-units corresponds to raw values up to ~+-4.85e8. That range is standard
        # in the TD-MPC2/Dreamer literature because THEIR reward/value magnitudes reach into
        # the thousands; this project's per-block rewards and bootstrapped values are order
        # single-to-double-digits (measured: full-episode PPO rewards up to ~1120 over ~144
        # macro-blocks/episode -> ~5-8 per block, geometric Q-sum a low multiple of that). A
        # +-20 range would waste nearly all 101 bins on magnitudes never seen and give
        # terrible resolution exactly where this domain operates (verified directly: caught
        # by diagnostics/tdmpc2_cost_probe.py's round-trip test, which showed >3.0 raw-unit
        # decode error near symlog=3, i.e. raw~19, before this fix). +-6 symlog-units (raw
        # ~+-400) keeps generous headroom while giving ~4x finer resolution in-range.
```

## ./legacy/TD_MPC2.py:390 {#--legacy-TD_MPC2-py-390}

```
        # Fix (v27): Q-ENSEMBLE, replacing the twin-Q pair. This is the second of the two
        # changes that make this genuinely "TD-MPC2" rather than "MPC with a learned model
        # and 2 critics" — the paper's ensemble (5 critics, random-subset-of-2 for the Bellman
        # target each update) reduces overestimation bias further than a fixed pair, since the
        # SAME two critics never get to collude with each other update after update.
```

## ./legacy/TD_MPC2.py:477 {#--legacy-TD_MPC2-py-477}

```
            # 2. Policy Prior warm-start: bias the distribution mean
            # Without a prior: mean = zeros (blind search)
            # With a prior: mean = pi(h) (informed search around best guess)
```

## ./legacy/TD_MPC2.py:539 {#--legacy-TD_MPC2-py-539}

```
                # ── The Latent CBF (Guillotine) ──
                # Cumulative sustainability check (Trajectory-wide)
                # If sum < 0, culture is net dying across the horizon
```

## ./legacy/TD_MPC2.py:612 {#--legacy-TD_MPC2-py-612}

```
        # Fix (v27): 'qs'/'target_qs' is the current ModuleList format. Old checkpoints saved
        # under the twin-Q ('q1'/'q2') format cannot be loaded here — the network shapes
        # differ (num_bins-logit heads vs 1-scalar heads, 5 critics vs 2) — so this is a
        # deliberate hard break, not silently-wrong weights.
```

## ./legacy/TD_MPC2.py:663 {#--legacy-TD_MPC2-py-663}

```
        # Fix (v27): rewards are now per-MACRO-BLOCK discounted sums (see the training loop's
        # block_reward accumulation), not raw per-step rewards — MUCH wider dynamic range than
        # before, which is exactly the regime two-hot/symlog regression is meant for.
```

## ./legacy/TD_MPC2.py:671 {#--legacy-TD_MPC2-py-671}

```
        # 1. Target Encoding (No Gradients). Fix (v27): random-subset-of-2 ensemble minimum
        # from the 5 TARGET critics, decoded from two-hot logits, then re-encoded as the
        # two-hot classification target for ALL 5 online critics (standard ensemble Bellman
        # backup with random subsampling — TD-MPC2's overestimation-reduction mechanism).
```

## ./legacy/TD_MPC2.py:679 {#--legacy-TD_MPC2-py-679}

```
            # GAMMA here is the per-MACRO-BLOCK discount — the block reward already folds in
            # GAMMA**t for t within the block, so bootstrapping the NEXT block needs GAMMA
            # raised to the block length once more, i.e. GAMMA_BLOCK = GAMMA ** MACRO_STEPS.
```

## ./legacy/TD_MPC2.py:717 {#--legacy-TD_MPC2-py-717}

```
        # 4. Policy Prior Loss (Behavioral Cloning on actual env actions)
        # We supervise the Prior to predict the action the agent actually took.
        # Over time, 'actions' will increasingly be MPPI-elite actions,
        # so the Prior learns to warm-start the planner from real experience.
```

## ./legacy/TD_MPC2.py:824 {#--legacy-TD_MPC2-py-824}

```
    # Fix (v27 diagnostic): was hardcoded to 3000, giving this side of the gate a large,
    # policy-independent time_avg_od advantage over the stochastic side's curriculum-sampled
    # starts (100-1400 typical at D0) — a no-op policy alone clears D0's OD threshold at
    # init_cells=3000 (0.217 vs the 0.004 target). Sample the same way training does so both
    # sides of the dual gate are evaluated on comparable initial conditions.
```

## ./legacy/TD_MPC2.py:872 {#--legacy-TD_MPC2-py-872}

```
    # Fix (v27): ACTION_REPEAT == MACRO_STEPS by construction — the plan() replan cadence and
    # the world-model macro-transition size are the SAME thing. Replanning more often than the
    # model's own timestep resolution would be replanning on stale/unlearned dynamics; less
    # often would waste the model's resolution. horizon=12 * MACRO_STEPS=50 = 600 raw steps =
    # exactly one HARVEST_INTERVAL_STEPS, so the planner can now see across a harvest event.
```

## ./legacy/TD_MPC2.py:881 {#--legacy-TD_MPC2-py-881}

```
    # Fix (v27): TOTAL_TRAINING_STEPS was 1_500_000 (~1.9 days measured pre-macro-transition
    # cost, dominated by update() at 1 call/raw-step). With macro-transitions update() now
    # fires once per MACRO_STEPS raw steps, so cost drops accordingly — re-measured before this
    # number is trusted (see diagnostics/tdmpc2_cost_probe.py). Overridable via total_steps= /
    # TDMPC2_STEPS env var so a run's budget doesn't require a source edit.
    # 8,000,000 matches every PPO run's budget in finalresults.md, for direct comparability.
    # Measured cost at this budget (diagnostics/tdmpc2_cost_probe.py): ~13.0h, under PPO's
    # own ~17h — affordable now that macro-transitions cut the update()-call count ~30x.
```

## ./legacy/TD_MPC2.py:921 {#--legacy-TD_MPC2-py-921}

```
    # Capacity in MACRO-transitions now, not raw steps — each entry already spans MACRO_STEPS
    # raw steps, so 25,000 macro-transitions covers 1.25M raw steps of experience, comparable
    # coverage to the old raw-step buffer at a fraction of the memory.
```

## ./legacy/TD_MPC2.py:983 {#--legacy-TD_MPC2-py-983}

```
        # Fix (v27): prefill also produces MACRO-transitions (random action held for
        # MACRO_STEPS raw steps each), not raw single-step transitions. Mixing 1-raw-step and
        # 50-raw-step transitions in the same buffer would teach the dynamics model two
        # different, contradictory timestep resolutions.
```

## ./legacy/TD_MPC2.py:1019 {#--legacy-TD_MPC2-py-1019}

```
    # Persistent per-difficulty rolling episode history — mirrors recurrent_ppo.py's
    # EpisodeMetricsCallback (metrics_cb.history_by_diff), NOT the original file's
    # chunk-local `chunk_metrics.clear()` — a chunk-local window under-samples badly here:
    # CHUNK_STEPS=100,000 raw steps / MACRO_STEPS=50 ~= 2,000 macro-decisions/chunk, and a
    # ~7200-step episode is ~144 macro-transitions, so a chunk holds only ~14 episodes —
    # below MASTERY_MIN_EPISODES=20 on its own. A persistent window (matching PPO's
    # MASTERY_WINDOW=40) lets the gate accumulate across chunk boundaries the same way PPO's
    # does, rather than resetting evidence every chunk.
```

## ./legacy/TD_MPC2.py:1029 {#--legacy-TD_MPC2-py-1029}

```
    # Ported from recurrent_ppo.py's Fix #15 + Fix #29 (crash-floor corrected, v31-validated).
    # Without this, sustained det-gate failure at 0% crash rate has NO corrective response here
    # — only stats["crash_rate"] >= DEMOTION_CRASH_RATE ever changes next_difficulty. v27 already
    # showed the precursor: D1 held for the rest of its 8M-step budget with time_avg_od
    # oscillating 0.0051-0.0071 against a 0.008 target, never sustaining a crossing, and nothing
    # would have caught it had it instead REGRESSED the way PPO's v17 did (48 straight chunks
    # failing the same criterion at exactly 0.00% crash, never demoted, until Fix #15 existed to
    # catch it). See recurrent_ppo.py's capability_fail_streak/d0_capability_abort for the full
    # rationale, including why D0's abort branch requires the crash-rate floor (v30's false
    # positive) while D1/D2's demotion deliberately does not (that branch exists specifically to
    # catch v17-style 0%-crash quality regression from a proven prior-good baseline).
```

## ./legacy/TD_MPC2.py:1080 {#--legacy-TD_MPC2-py-1080}

```
        # Forces a fresh plan()+block at the NEXT loop iteration regardless of the raw
        # step % MACRO_STEPS alignment. Needed after an episode reset: `step` is the
        # CHUNK-level counter and does not reset per episode, so without this an episode
        # could start mid-way through what the accumulator thinks is an old block, applying
        # a stale action (planned for the previous episode's last state) to a brand-new one.
```

## ./legacy/TD_MPC2.py:1132 {#--legacy-TD_MPC2-py-1132}

```
                # Fix (v27): read the PROJECT's own harvest_mg / time_avg_od metrics from the
                # step info dict (genetic_env.py always populates these) rather than the
                # original file's ad-hoc "peak_od" / "population < 10" proxies. This is what
                # makes stats directly comparable against ADVANCE_TARGETS and every PPO run.
```

## ./legacy/TD_MPC2.py:1183 {#--legacy-TD_MPC2-py-1183}

```
                # See the force_new_block comment above the loop: `step` does not reset per
                # episode, so this is what makes the NEXT iteration replan on the fresh state
                # instead of continuing a block that belonged to the episode that just ended.
```

## ./legacy/TD_MPC2.py:1196 {#--legacy-TD_MPC2-py-1196}

```
            # Fix (v27 diagnostic): was every 2,000 raw steps (4,000 saves over an 8M-step
            # budget), each one pickling the FULL 25,000-transition replay buffer on top of
            # network weights — ~7MB/save, ~15GB and rising over the v27 run, and a plausible
            # contributor to the chunk-time variance observed all session under CPU contention.
            # PPO's CheckpointCallback saves every 10,000 steps and weights only (no persistent
            # buffer to dump). Widened 25x; still ~320 saves over the full budget.
```

## ./legacy/TD_MPC2.py:1225 {#--legacy-TD_MPC2-py-1225}

```
        # Deterministic side: a handful of noise-free planning episodes per chunk, same
        # project rationale as recurrent_ppo.py's det_eval_history — a policy that only
        # "looks like" it works under exploration noise should not be able to advance alone.
```

## ./legacy/TD_MPC2.py:1299 {#--legacy-TD_MPC2-py-1299}

```
                    # D0 has no tier to demote to. Only abort when crash rate is ALSO elevated
                    # (v30's false-positive lesson on the PPO side) — "never yet passed" and
                    # "regressed from passing" are different signals at the floor tier.
```

## ./legacy/TD_MPC2.py:1362 {#--legacy-TD_MPC2-py-1362}

```
    # Fix (v27): NOT YET UPDATED for the macro-timestep/ensemble/two-hot rewrite below this
    # function still uses the pre-Fix#27 4D action, raw-step transitions, and the removed
    # q1/q2 attributes — it would fail with a confusing AttributeError deep in agent.load()/
    # update() rather than a clear one here. Failing loudly at the entry point instead of
    # leaving it silently inconsistent with train_td_mpc2().
```

## ./legacy/Var_MPC.py:113 {#--legacy-Var_MPC-py-113}

```
        # A_curr shape: (obs_dim, order, order)
        # B_curr shape: (obs_dim, order)
        # self.A_base is (16, 16). self.B_base is (16, 1)
```

## ./legacy/Var_MPC.py:119 {#--legacy-Var_MPC-py-119}

```
        # (Note: Using simple Forward Euler discretisation here for dynamic stability)
        # A_discrete = I + A_curr * dt (where dt=1 in simulation steps)
        # B_discrete = B_curr * dt
```

## ./legacy/Var_MPC.py:166 {#--legacy-Var_MPC-py-166}

```
        # ── Simplicial Normalization (SimNorm) ──
        # Projects the unbounded latent vector onto a positive simplex.
        # This provides a bounded state space for the dynamics model, vastly 
        # improving sample efficiency and preventing exploding latents.
```

## ./legacy/Var_MPC.py:389 {#--legacy-Var_MPC-py-389}

```
                    # ── Variance-Maximizing MPPI ──
                    # Maximize: Mean(Q) + beta * Variance(Q)
                    # We now have 2 critics to estimate epistemic uncertainty
```

## ./legacy/Var_MPC.py:402 {#--legacy-Var_MPC-py-402}

```
                # ── The Latent CBF (Guillotine) ──
                # Cumulative sustainability check (Trajectory-wide)
                # If sum < 0, culture is net dying across the horizon
```

## ./legacy/Var_MPC.py:572 {#--legacy-Var_MPC-py-572}

```
        # 4. Policy Prior Loss (Behavioral Cloning on actual env actions)
        # We supervise the Prior to predict the action the agent actually took.
        # Over time, 'actions' will increasingly be MPPI-elite actions,
        # so the Prior learns to warm-start the planner from real experience.
```

## ./legacy/visualize_env.py:54 {#--legacy-visualize_env-py-54}

```
# ══════════════════════════════════════════════════════════════════════
#  CO2 Bubble particle system
# ══════════════════════════════════════════════════════════════════════
```

## ./legacy/visualize_env.py:97 {#--legacy-visualize_env-py-97}

```
# ══════════════════════════════════════════════════════════════════════
#  Stirring vortex flow particle
# ══════════════════════════════════════════════════════════════════════
```

## ./legacy/visualize_env.py:143 {#--legacy-visualize_env-py-143}

```
# ══════════════════════════════════════════════════════════════════════
#  Inline sparkline graph widget
# ══════════════════════════════════════════════════════════════════════
```

## ./legacy/visualize_env.py:194 {#--legacy-visualize_env-py-194}

```
# ══════════════════════════════════════════════════════════════════════
#  Slider bar helper
# ══════════════════════════════════════════════════════════════════════
```

## ./legacy/visualize_env.py:215 {#--legacy-visualize_env-py-215}

```
# ══════════════════════════════════════════════════════════════════════
#  Main visualiser
# ══════════════════════════════════════════════════════════════════════
```

## ./scripts/finish_run.py:26 {#--scripts-finish_run-py-26}

```
# The behaviour-cloned controller: the only artefact in this project that passes held-out D2,
# and it uses no RL. Printed alongside every result so "did this beat what we already have?"
# is answered without cross-referencing.
```

## ./scripts/run_training.py:87 {#--scripts-run_training-py-87}

```
        # Fix #24: gate mode and seed are part of a run's identity. Without the seed recorded,
        # a replication cannot distinguish a configuration effect from an RNG draw — which is
        # exactly the ambiguity that weakened the v21-vs-v23 comparison.
```

## ./scripts/run_training.py:166 {#--scripts-run_training-py-166}

```
    # Reset the checkpoint dir by MOVING it aside, never deleting: overlapping step numbers
    # across runs already made 'highest step wins' unsafe, and the old files are the only
    # record of the previous run's trajectory.
```

## ./scripts/validate.py:42 {#--scripts-validate-py-42}

```
    # Surface the specific failure that WILL recur: a checkpoint saved under a different
    # observation dimension than the current env. The observation space has changed once
    # already (6 -> 8 channels, Fix #18), which silently orphaned every earlier checkpoint —
    # including model_data/BEST_bc_clone_D2_validated, the project's best artefact. Reporting
    # "parse failed" for this would bury a hard incompatibility as a formatting problem.
```

## ./training/callbacks.py:3 {#--training-callbacks-py-3}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/callbacks.py:169 {#--training-callbacks-py-169}

```
            # episode_train_diff is injected by CurriculumStartWrapper.step() on done,
            # ensuring we record the difficulty this episode actually ran at (not the
            # next episode's difficulty, which the env has already reset to).
```

## ./training/curriculum_schedule.py:5 {#--training-curriculum_schedule-py-5}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/curriculum_schedule.py:24 {#--training-curriculum_schedule-py-24}

```
# MASTERY_WINDOW: size of the persistent, per-difficulty rolling episode-history buffer
# used for advancement/demotion decisions (see EpisodeMetricsCallback in callbacks.py).
# Raised 10->40 and actually wired in (previously imported but never used — decisions were
# made on whatever the current 100k-step chunk contained, ~14 episodes, reset every chunk).
# A held-out sweep found harvested_mg swings ~18x and time_avg_od ~55x across seeds based
# almost entirely on initial population size, so a 14-episode chunk could pass a gate on a
# lucky draw of larger cold starts — which is exactly what let a since-invalidated "D2
# mastery" fire on too narrow a sample. 40 episodes, persisting across chunk boundaries,
# closes that gap.
```

## ./training/curriculum_schedule.py:37 {#--training-curriculum_schedule-py-37}

```
# Deterministic-evaluation gate (see deterministic_eval.py): a policy that only "looks like"
# it harvests under exploration noise (stochastic rollouts collected during model.learn())
# can pass the stochastic gate above while its actual deterministic/deployed behavior is
# degenerate — confirmed directly this session (a "D2 mastery" checkpoint whose deterministic
# policy never harvested, letting biomass grow unbounded, passed the stochastic gate on
# incidental harvest events from action-sampling noise, then failed held-out validation
# badly). A handful of genuinely deterministic episodes per chunk, gated the same way,
# closes this: advancement now requires BOTH gates to pass.
```

## ./training/curriculum_schedule.py:54 {#--training-curriculum_schedule-py-54}

```
# Fix #12 (v16), second half: bound the NUMBER of plateau kicks per difficulty, not just their
# ceiling. ENTROPY_PLATEAU_CAP was lowered 3.6x -> 2.0x -> 1.3x across prior runs, each time
# because kicks were producing near-random actions — but lowering the ceiling doesn't help when
# the kick re-fires every PLATEAU_CHUNKS chunks indefinitely. v15 sat in D1 for ~55 chunks
# without advancing, so the kick fired roughly 9 times, re-boosting entropy to the cap each time
# the multiplier had relaxed back down: a self-reinforcing loop (no advance -> more noise -> can't
# hold a steady setpoint -> no advance). Early kicks are genuinely useful (they help D0 discover
# that harvesting pays at all); repeated late kicks are counterproductive, because by then the
# policy has already found the right region — the D1 sweep shows the gate-passing frac window
# (0.05-0.20) is wide and forgiving, so the agent doesn't need more exploration to find it, it
# needs less noise to hold it. Counter resets on every difficulty change.
```

## ./training/curriculum_schedule.py:67 {#--training-curriculum_schedule-py-67}

```
# Fix #15 (v18): consecutive chunks failing the DETERMINISTIC gate at D1/D2 before demoting a
# tier. Existing demotion keys only on crash_rate, which v17 showed is insufficient: it held
# D2 for 48 chunks failing the same criterion (time_avg_od) every single time at a 0.00% crash
# rate, so nothing walked it back down and ~4.8M steps went into a tier it could not perform.
# Set well above MASTERY_WINDOW's turnover and above the 6-chunk PLATEAU_CHUNKS cadence so a
# normal pre-advance plateau cannot trigger it — only sustained inability at the current tier.
```

## ./training/curriculum_schedule.py:75 {#--training-curriculum_schedule-py-75}

```
# Advancement criteria applied to cold-start (non-stitched) episodes only.
# Thresholds are lower than previous warm-start-mixed values because cold starts are harder.
#
# RE-CALIBRATED for the periodic semi-continuous harvest redesign (genetic_env.py
# action[2]: harvest fraction, applied only every HARVEST_INTERVAL_STEPS=600 steps/12h,
# replacing a first attempt that used a CONTINUOUS per-step dilution rate D). The
# continuous-D version was live-training-tested for a full 5M-step budget and never
# advanced past D0 — `time_avg_od` stayed at exactly 0.0000 for all 50 chunks because
# nearly every episode crashed before step 3600 (the point the back-half OD tracker even
# starts), confirmed via raw per-episode termination traces (~step 1000-2600, population
# declining to the extinction floor). Root cause: this strain's mu_max is deliberately
# slow (~0.055h^-1, ~12.6h doubling), leaving very little margin against a dilution
# action exposed to a lethal washout region on literally every single step.
#
# The periodic redesign gives the culture a full 12h to recover between harvest
# decisions instead. A (stir=80rpm, light=1000umol, harvest_frac-sweep) grid sweep at
# 20L/D2 physics (dynamic_profile_sweep.py) found frac in {0, 0.05, ..., 0.40} all ran
# full 144h episodes with 0% crash; only frac=0.50 (=F_MAX) crashed 100% — the washout
# cliff sits between 0.40-0.50, i.e. only the top ~20% of the action range is dangerous
# (vs ~33% of the range for the earlier continuous-D design). Best sustainable fraction
# was 0.15 (147.9mg/144h, 12.32mg/event). A follow-up 4-seed run at that setpoint found:
# harvested_mg 118-175 (median ~150), time_avg_od 0.014-0.022 (median ~0.019), 0% crash —
# critically, episodes now actually reach step 3600, so time_avg_od is a real signal.
#
# D0/D1/D2 thresholds below are a first-pass scaling off this single-setpoint D2 data
# point (not yet a full grid sweep across D0/D1 physics or multiple stir/light/frac
# combos) — same relative structure as prior calibrations (D1 ~40% of D2 ceiling, D2
# tighter crash cap than D1, D0 conservative/safely-clearable given weaker phys_scale).
# Flagged for refinement once live training data comes in.
```

## ./training/curriculum_schedule.py:190 {#--training-curriculum_schedule-py-190}

```
        # ── Per-Episode Difficulty Sampling ──────────────────────────────────
        # Sample a fresh training difficulty for every episode, not per-chunk.
        # This gives true stochastic mixing (e.g. at D1: 80% D1, 20% D0 per episode)
        # rather than locking all episodes in a chunk to the same difficulty.
```

## ./training/curriculum_starts.py:2 {#--training-curriculum_starts-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/deterministic_eval.py:23 {#--training-deterministic_eval-py-23}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/entropy_schedule.py:3 {#--training-entropy_schedule-py-3}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/entropy_schedule.py:22 {#--training-entropy_schedule-py-22}

```
# Fix #12 (v16): lowered 0.20 -> 0.08. This constant is the *floor* below which the std-band
# controller pushes ent_coef UP (recurrent_ppo.py: `latest_std < STD_BAND_LOW` -> multiplier +=
# ENTROPY_ADJUST_UP), so it sets a hard lower bound on how tightly the policy is ever allowed to
# converge. At 0.20 that bound was actively preventing convergence, and it directly contradicted
# this file's own note below describing "healthy 0.03-0.08" std — the controller was defending a
# std 2.5-6x above what the same comment calls healthy.
#
# Measured evidence (v16 pre-work, D1): a fixed-action physics sweep at BOTH the reference
# operating point (stir=80/light=1000) and the v15 policy's own measured one (stir=60/light=900)
# put the reward optimum at harvest frac 0.18-0.20 (mean reward ~1117), and that optimum also
# clears the D1 curriculum gate with margin on every criterion. The v15 trained policy instead
# sat at frac 0.30-0.44 across all seeds and both archives, earning only ~842 (frac 0.30) to
# ~646 (frac 0.35) — i.e. it left 250-470 reward per episode unclaimed, so it was NOT at its own
# reward optimum and this is not a reward-shape problem (Fix #10's peaked reward_od is working as
# designed: its optimum sits squarely inside the gate-passing window).
# The mechanism: the 6.5M archive's decoded harvest std was 0.05-0.09, and the raw->frac map has
# slope 0.25 (raw [-1,1] -> frac [0,0.5]), so decoded 0.05 == raw std 0.20 — pinned *exactly* at
# STD_BAND_LOW. The controller was holding the policy at its own floor, making it unable to hold
# any steady harvest fraction, and the sweep shows time_avg_od is monotonically decreasing in
# frac, so drifting frac directly explains the decaying deterministic time_avg_od that blocked
# every D1->D2 attempt.
# 0.08 is the top of this file's own stated healthy range: genuine std collapse is still caught,
# but convergence into 0.08-0.65 is now permitted instead of fought.
```

## ./training/entropy_schedule.py:55 {#--training-entropy_schedule-py-55}

```
# Lowered from 3.6x -> 2.0x: repeated plateau kicks compounding up to 3.0x caused the
# policy's action distribution to become near-random (std 0.5-0.9 vs healthy 0.03-0.08),
# which then trained the network on garbage rollouts and produced a real, non-recovering
# reward regression (-75 peak -> -260) even after the multiplier decayed back down.
# Lowered again 2.0x -> 1.3x: a full training run at 2.0x reproduced the same failure
# signature (a live "Std hard-cap applied: 0.714 -> 0.484 (cap=0.65)" correction logged
# right after plateau kicks pushed the multiplier to 2.0x) — the plateau mechanism's
# aggressive upward kick was directly fighting the std-band feedback loop below (which
# already has its own escalation path via ENTROPY_ADJUST_UP), and there's up to a full
# 100k-step chunk of lag between a kick and the next std correction. The policy
# demonstrably could already clear its curriculum gate under high entropy (twice, in
# that run) but couldn't sustain it — consistent with exploration noise, not insufficient
# exploration, being the limiting factor once training has progressed this far.
```

## ./training/entropy_schedule.py:70 {#--training-entropy_schedule-py-70}

```
# ── Fix #22 (v24): late-training policy-std annealing ────────────────────────────────────
# THE PROBLEM. Every run's DETERMINISTIC performance is far worse than its stochastic
# performance — v16b det 20-48mg vs stoch 85-212mg; v22 det 39.7mg/od 0.0086 vs stoch
# 211mg/od 0.0209. The curriculum gate and any real deployment use the deterministic (mean)
# policy; PPO optimises the stochastic one. So the agent is graded on a criterion it is
# never trained on, and the gate reads as an obstacle rather than a filter.
#
# WHY THE MEAN LOSES TO ITS OWN SAMPLES. The harvest action is clipped at 0. When the
# policy's mean sits near that floor, samples can only deviate UPWARD, so the sampled
# policy harvests substantially while the mean harvests almost nothing — E[f(x)] != f(E[x]),
# asymmetrically, because of the boundary. This predicts the gap shrinks when the mean sits
# in the interior, and that is observed: v21's mean harvest fraction was 0.16-0.18 (off the
# floor) and it had both the best deterministic numbers and the smallest gap of any run.
#
# WHY IT NEVER RESOLVES ITSELF. `train/std` sat at ~0.50-0.54 in every single run
# (v23: 0.543 at chunk 74). The entropy schedule actively sustains it: with weak advantage
# gradients the entropy bonus dominates and std equilibrates high. Fix #12a's STD_BAND_LOW
# reduction was inert precisely because std never descended far enough to touch it.
#
# THE FIX. Rather than hoping entropy tuning lowers std, explicitly anneal a HARD CAP on it
# over the back half of training, so mean and samples converge and the two objectives stop
# diverging. Deliberately schedule-based, not reactive: the std-band controller is reactive
# and has demonstrably failed to bring std down for 13 runs.
# STD_ANNEAL_FINAL is kept comfortably ABOVE STD_BAND_LOW (0.08) so the two controllers
# cannot fight — a cap below the band floor would have the annealer clamp std down while the
# band controller pushes entropy up to raise it, the same conflict Fix #12b had to bound.
```

## ./training/env_factory.py:3 {#--training-env_factory-py-3}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/env_utils.py:3 {#--training-env_utils-py-3}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/recurrent_ppo.py:2 {#--training-recurrent_ppo-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/recurrent_ppo.py:25 {#--training-recurrent_ppo-py-25}

```
# Re-exported here (not just used internally) so that:
#  1) old checkpoints pickled while this file was a monolith and recorded
#     `__main__.ActionSmoothnessWrapper` etc. can still resolve those names
#     when this script is run directly as `python recurrent_ppo.py`.
#  2) other scripts (e.g. evaluate_agent.py) that import these names from
#     `recurrent_ppo` keep working unchanged.
```

## ./training/recurrent_ppo.py:57 {#--training-recurrent_ppo-py-57}

```
# Linear LR decay, driven by OUR OWN steps_done/TOTAL_TRAINING_STEPS tracking rather than
# SB3's built-in progress_remaining. SB3 computes progress_remaining from the
# total_timesteps argument passed to THIS model.learn() call, not the grand training
# budget — and this codebase calls learn() once per 100k-step chunk with
# reset_num_timesteps=False. Verified against SB3's source before wiring this in: each
# chunk call re-derives its own "total_timesteps" as num_timesteps-so-far + this chunk's
# size, so progress_remaining restarts near 1.0 at the start of every chunk and always
# hits 0 by the end of that same chunk — a 40-chunk sawtooth, not a smooth 4M-step decay.
# A naive SB3 schedule would have silently cut the LR toward its floor inside nearly every
# chunk rather than only near the true end of training. Sidestepped the same way this file
# already handles ent_coef (see model.ent_coef assignment in the chunk loop): an external,
# manually-updated value the schedule function reads from, refreshed once per chunk from
# the real steps_done/TOTAL_TRAINING_STEPS ratio.
```

## ./training/recurrent_ppo.py:74 {#--training-recurrent_ppo-py-74}

```
# Fix #17 (v20): the behaviour-cloned controller's held-out D2-passing scores, printed beside
# every deterministic eval as a fixed reference line. That policy (model_data/
# BEST_bc_clone_D2_validated/) is the best this project has produced — median harvest 109.4mg,
# p25 63.8, time_avg_od 0.0191, 0% crash over 40 held-out seeds — and it required no RL at all.
# Showing it inline makes "is PPO anywhere near the thing we already have?" answerable at a
# glance instead of by cross-referencing a separate document mid-run.
```

## ./training/recurrent_ppo.py:82 {#--training-recurrent_ppo-py-82}

```
# Fix #23 (v25): which policy the curriculum gate advances on. "dual" = stochastic AND
# deterministic (default; see the conjunction site below for the full rationale).
# "stochastic" = stochastic only, for a self-consistent stochastic-deployment experiment.
# Override per run with the GATE_MODE environment variable so no source edit is needed:
#     GATE_MODE=stochastic python training/recurrent_ppo.py
```

## ./training/recurrent_ppo.py:91 {#--training-recurrent_ppo-py-91}

```
# Fix #24 (v26): explicit, RECORDED seed. Until now nothing seeded numpy/torch/the env, so two
# runs of the same configuration differed by an unknown mixture of config effect and RNG draw.
# That directly weakens a conclusion already reported: "v21's time_avg_od 0.0094 was not
# reproducible" rested on v23 (same config) returning 0.0066 — but with no seed control, that
# spread cannot be attributed to the configuration rather than the seed. With the seed pinned
# and logged, a replication isolates config effects, and a deliberate seed sweep measures RNG
# variance separately. Set RUN_SEED to compare configurations; vary it to measure variance.
```

## ./training/recurrent_ppo.py:115 {#--training-recurrent_ppo-py-115}

```
    # Fix #24 (v26): seed numpy, torch and Python's RNG before anything samples. SB3's
    # set_random_seed covers all three plus CUDA; the env and action space are seeded separately
    # below since they draw from their own generators.
```

## ./training/recurrent_ppo.py:147 {#--training-recurrent_ppo-py-147}

```
    # Fix #17 (v20): preserve the run's best DEPLOYABLE (deterministic) policy, so a run that
    # degrades still yields its peak rather than its final weights. See the scoring comment
    # at the [BEST-DET] block below.
```

## ./training/recurrent_ppo.py:153 {#--training-recurrent_ppo-py-153}

```
    # Fix #29: early-stop signal for a sustained deterministic-gate failure streak AT D0.
    # capability_fail_streak below is gated on current_difficulty>0 for the DEMOTION branch
    # (there is no tier below D0 to demote to), but that guard also meant D0 had NO active
    # response to an in-place policy collapse at all. Confirmed live in v29: PPO's det crash
    # rate climbed 0%->80% over 5 chunks while capability_fail_streak sat structurally stuck
    # at 0 (the guard prevented it from ever incrementing) and the already-active plateau-kick
    # mechanism (entropy bumps, unrelated to this failure mode) kept firing on its own schedule
    # without arresting the decline. Rather than attempt a live mid-training weight reload
    # (risky: SB3 optimizer/rollout-buffer state can desync from a hot-swapped policy), this
    # stops the run cleanly with a clear diagnostic once the same failure signal that would
    # demote at D1/D2 sustains at D0 — burning the rest of an 8M-step budget on a policy known
    # to be failing its own gate is worse than stopping and deciding explicitly what to do next.
```

## ./training/recurrent_ppo.py:207 {#--training-recurrent_ppo-py-207}

```
            # Belt-and-suspenders: custom_objects above should already install this, but
            # explicitly re-assign in case a saved checkpoint's pickled schedule survives
            # deserialization instead of being overridden — the sawtooth bug this schedule
            # exists to avoid would otherwise silently return on any future resume.
```

## ./training/recurrent_ppo.py:268 {#--training-recurrent_ppo-py-268}

```
            # Fix #13 (v18): gamma 0.995 -> 0.9995. This is a CREDIT-ASSIGNMENT fix, arrived at
            # only after two competing hypotheses were measured and REFUTED:
            #
            #   (a) reward-structure exploit — refuted by reward_ab.py: on 8 identical episodes
            #       the reward function ranks the scripted expert +313 ABOVE v17 (1079 vs 766),
            #       entirely via reward_od. reward_biomass contributed 11.8, not the ~1440 its
            #       theoretical ceiling suggested (tanh(per_cell_growth/5) is tiny at realistic
            #       growth rates, and the flat -0.010 penalty offsets most of the rest), and
            #       differed between the two policies by 0.4. The reward is NOT exploitable.
            #   (b) exploration noise making the expert's strategy unachievable — refuted by
            #       noise_sensitivity.py: the expert keeps 94.8% of its noise-free reward at
            #       sigma=0.50 (exactly the train/std v15/v16b/v17 all sat at) and dominates
            #       v17 at EVERY sigma from 0.0 to 0.70. No crossover. Entropy left untouched.
            #
            # What the evidence does point at: v17 learned stir and light CORRECTLY (light
            # settled at ~1000umol, the sweep optimum) and only harvest incorrectly. The
            # distinguishing feature is credit frequency. Stir and light act on all 7200 steps;
            # the harvest action is applied only on the 12 event steps
            # (HARVEST_INTERVAL_STEPS=600), so on 7188 of 7200 steps the policy emits a harvest
            # value the env ignores while PPO still assigns it advantage — 599 of every 600
            # gradient samples on that dimension are spurious credit.
            #
            # gamma compounds this. At 0.995 the effective horizon is 1/(1-gamma) = 200 steps,
            # while a harvest decision's consequence unfolds over the following 600+ steps and
            # compounds for thousands. 0.995^600 = 0.049, so the immediate harvest reward is
            # undiscounted while its OD cost is ~95% invisible — the agent cannot see past the
            # current harvest cycle, which is precisely the trade-off the expert exploits
            # (forgo harvest now, hold OD, harvest more across the remaining ~100h).
            # At 0.9995: horizon 2000 steps (~3.3 harvest cycles) and 0.9995^600 = 0.741, a 15x
            # improvement in the visibility of the next cycle. Not pushed to 0.9999 (horizon
            # 10000 steps, beyond the 7200-step episode) to avoid the value-variance blowup that
            # very-near-1 discounting causes; 0.9995 matches the task's actual causal timescale.
            # Consistent with every failure this session having been in the harvest dimension
            # specifically, in whichever direction the local gradient happened to favour.
```

## ./training/recurrent_ppo.py:304 {#--training-recurrent_ppo-py-304}

```
            # target_kl=0.02 tested in v12 and DISABLED after a clear regression: det crash
            # rate climbed to 73.3% by chunk 7 (vs. v11's clean 0% crash at a comparable
            # point under the identical reward config, no other change), ep_rew_mean sat
            # deeply negative and flat (~-50) instead of the healthy early climb v11 showed,
            # and an entropy plateau-kick made it worse, not better. Hypothesis: at 0.02,
            # target_kl's per-minibatch early-stopping fired often enough (observed
            # "Early stopping ... max kl: 0.03-0.09" on most iterations) to cut PPO's 4
            # nominal epochs short most of the time, starving the policy of the gradient
            # steps needed to correct crash-prone behavior during early, fast-changing
            # training — exactly when full updates matter most. Not re-tried at a looser
            # value yet; disabled (None) so the concurrent LR-decay change could be tested
            # in isolation as v13. Re-enable only as a deliberate, isolated test, not
            # bundled with other changes — same lesson as the reward-weight guessing
            # earlier this session: change one variable at a time.
```

## ./training/recurrent_ppo.py:347 {#--training-recurrent_ppo-py-347}

```
    # Instantiated once and reused across the whole chunk loop (not recreated per chunk)
    # so its per-difficulty rolling history (maxlen=MASTERY_WINDOW) survives chunk
    # boundaries — see EpisodeMetricsCallback docstring in callbacks.py.
```

## ./training/recurrent_ppo.py:352 {#--training-recurrent_ppo-py-352}

```
    # Persistent, per-difficulty rolling history of DETERMINISTIC evaluation episodes —
    # see deterministic_eval.py. Separate from metrics_cb's stochastic history; advancement
    # requires both gates to pass, closing the exploration-noise loophole found this session.
```

## ./training/recurrent_ppo.py:360 {#--training-recurrent_ppo-py-360}

```
        # Difficulty is now sampled per-episode in CurriculumStartWrapper.reset().
        # train_diff here equals mastery level and is used only for the streak
        # accounting check (criteria_passed and train_diff == current_difficulty).
```

## ./training/recurrent_ppo.py:384 {#--training-recurrent_ppo-py-384}

```
        # Linear LR decay across the TRUE 4M-step budget (steps_done, not SB3's per-call
        # progress_remaining — see _lr_schedule_fn comment for why). Computed from
        # steps_done at the START of this chunk, same timing convention as ent_coef above.
```

## ./training/recurrent_ppo.py:391 {#--training-recurrent_ppo-py-391}

```
        # Fix #22 (v24): anneal a hard cap on actor std over the back half of training, so the
        # MEAN policy converges toward the sampled policy. The deterministic gate and any real
        # deployment use the mean; PPO optimises the samples. `train/std` sat at ~0.50-0.54 in
        # every run to date and the reactive std-band controller never brought it down, so this
        # is applied as an explicit schedule. See entropy_schedule.annealed_std_cap.
```

## ./training/recurrent_ppo.py:479 {#--training-recurrent_ppo-py-479}

```
        # Deterministic evaluation pass (see deterministic_eval.py): a handful of genuinely
        # deterministic episodes per chunk, gated the same way as the stochastic rollout
        # above. Closes the exploration-noise loophole confirmed this session — a policy
        # whose deterministic (mean) action never harvests can still look fine under the
        # stochastic gate purely from action-sampling noise around that mean.
```

## ./training/recurrent_ppo.py:500 {#--training-recurrent_ppo-py-500}

```
        # ── Fix #17 (v20): best-deterministic checkpoint tracking ─────────────────────────
        # v17, v18 and v19 ALL ended at or near their worst deterministic policy of the run,
        # because training simply stops at the budget with whatever weights it currently has.
        # v19 is the clearest case: its deterministic policy was 149.1mg / od 0.0203 / 0% crash
        # at chunk 1 and 28.3mg / od 0.0002 / 80-93% crash at chunk 80 — the run PRODUCED a good
        # policy and then threw it away. Nothing in the loop preserved it.
        #
        # Score = median harvested_mg scaled by how well time_avg_od meets the CURRENT tier's
        # threshold, hard-zeroed on any crash. Crash-zeroing is deliberate: a policy that
        # crashes is unusable regardless of yield, and crash rate is the metric that exposed
        # v19's collapse while the stochastic gate stayed clean. Tracked across the whole run
        # (not per-tier) so the artifact is simply "the best deployable policy this run found".
```

## ./training/recurrent_ppo.py:554 {#--training-recurrent_ppo-py-554}

```
        # Fix #23 (v25): GATE_MODE selects which policy the curriculum advances on.
        #
        # "dual" (default, v5-v24): the stochastic-rollout gate AND the deterministic gate must
        #   both pass. Added because v4 declared D2 mastery on stochastic metrics inflated by
        #   exploration noise, then scored median 0.4mg against a 90mg gate on held-out data.
        #
        # "stochastic" (v25): advance on the stochastic gate alone. The point is NOT that the
        #   deterministic check was wrong — v24 proved the deterministic policy really is far
        #   worse, because the harvest action is clipped at 0 and E[clip(x)] != clip(E[x]) when
        #   the mean sits near that floor, so the sampled policy gets a systematic upward
        #   harvest bias that survives interval-averaging. The point is CONSISTENCY: the real
        #   error in v14/v17 was gating on one policy while validating with another
        #   (held_out_sweep.py is deterministic). If a stochastic controller is acceptable to
        #   deploy, then gating stochastically is legitimate — provided validation is ALSO
        #   stochastic. `held_out_sweep.py --stochastic` exists for exactly that.
        #   Two caveats that remain true in this mode and must be stated with any result:
        #     * the criterion's difficulty drifts, because stochastic metrics depend on
        #       train/std, which the entropy schedule moves during the run;
        #     * these are the rollouts being trained on, so the metric is optimistically
        #       biased in the same way training accuracy is.
        #   The deterministic eval still RUNS and is still logged, so the gap stays visible; it
        #   just no longer blocks advancement.
```

## ./training/recurrent_ppo.py:592 {#--training-recurrent_ppo-py-592}

```
                    # Terminal tier — nowhere further to advance, but this is the
                    # "sustained mastery at full difficulty" signal used for early
                    # stopping (see the while-loop condition below).
```

## ./training/recurrent_ppo.py:600 {#--training-recurrent_ppo-py-600}

```
            # Demotion: sustained high crash rate at D1/D2 drops back one level.
            #
            # Fix #15 (v18): ALSO demote on sustained CAPABILITY failure, not just crashes.
            # v17 exposed the gap concretely: it advanced to D2 with a genuinely good policy
            # (det harvest 113mg, time_avg_od 0.0215), then degraded across the following 48
            # D2 chunks to harvest 72-80mg / time_avg_od 0.0022 — failing the SAME criterion
            # (time_avg_od) on all 48 of them — while crash rate stayed at exactly 0.00%.
            # Because demotion keyed only on crash_rate, nothing ever walked it back down, and
            # it burned ~4.8M steps sitting at a tier it could no longer do. Held-out validation
            # then failed at BOTH D1 and D2.
            #
            # A tier the policy cannot satisfy is not a useful training distribution: dropping
            # back one level restores a solvable task and lets it re-earn the advance. Keyed on
            # the DETERMINISTIC gate (det_criteria_passed) rather than the stochastic one, since
            # deterministic behaviour is what the held-out validation and any real deployment
            # actually use. Threshold is deliberately long (CAPABILITY_DEMOTION_CHUNKS) so
            # ordinary chunk-to-chunk noise or a normal pre-advance plateau cannot trigger it —
            # only a sustained inability to perform at the current tier.
            # Fix #29: no longer gated on current_difficulty>0 — see d0_capability_abort's
            # definition above for why. The counter now tracks sustained det-gate failure at
            # ANY tier, including D0; only the RESPONSE differs below (demote vs. abort),
            # since D0 has no tier to demote to.
```

## ./training/recurrent_ppo.py:647 {#--training-recurrent_ppo-py-647}

```
                    # Fix #29 correction (v30 retry): capability_failing alone means "hasn't
                    # cleared the det gate yet", not "is collapsing" — live-verified this fires
                    # on a run with 0.00% crash rate and steadily growing harvest (38.8/30.0,
                    # p25 29.8/15.0, both passing; only time_avg_od lagging, 0.0007/0.0040) just
                    # as readily as on genuine collapse (the original v29 trigger: crash rate
                    # climbing 0%->80% with harvest/od both declining). The D1/D2 demotion
                    # branch above deliberately does NOT require a crash floor (Fix #15 exists
                    # specifically to catch v17-style quality regression at 0% crash), but that
                    # rationale doesn't transfer to D0: D1/D2 demotion had a proven prior-good
                    # baseline to fall back to, while a D0 run stuck below the OD bar from the
                    # start has no such baseline to compare against — "never yet passed" and
                    # "regressed from passing" are not the same signal at the floor tier. Only
                    # abort D0 when crash rate is ALSO elevated, the same threshold and
                    # rationale used for D1/D2's crash-based demotion_streak.
```

## ./training/recurrent_ppo.py:669 {#--training-recurrent_ppo-py-669}

```
                # else: sustained det-gate failure at D0 but crash rate is healthy — this is
                # "slow but not broken" (the v30 case above), not the collapse pattern this
                # exists to catch. Keep training; capability_fail_streak stays pinned at/above
                # threshold and is silently re-checked each subsequent chunk at no real cost.
```

## ./training/recurrent_ppo.py:691 {#--training-recurrent_ppo-py-691}

```
                # Cap the plateau multiplier so ent_coef stays below 50% of ENTROPY_MAX.
                # Absolute cap = 0.5 * ENTROPY_MAX / decayed_base. Prevents runaway regardless
                # of how far the base has decayed. High entropy ≠ useful exploration when stuck.
                #
                # Fix #12 (v16): the boost is also budgeted — MAX_PLATEAU_KICKS_PER_DIFFICULTY
                # kicks per difficulty, then plateau chunks stop touching entropy at all so the
                # policy is allowed to converge. See curriculum_schedule.py for the rationale.
```

## ./training/recurrent_ppo.py:843 {#--training-recurrent_ppo-py-843}

```
    # Lower LR for fine-tuning: avoids catastrophic forgetting of curriculum knowledge.
    # Setting model.learning_rate here is a no-op — SB3 reads model.lr_schedule during
    # training, not the raw attribute, and lr_schedule is already fixed at load time to
    # _lr_schedule_fn (re-pickled from the checkpoint), which reads the module-level
    # _lr_state dict instead. Update that directly so the value actually takes effect.
```

## ./training/training_state.py:2 {#--training-training_state-py-2}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

## ./training/wrappers.py:3 {#--training-wrappers-py-3}

```
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
```

