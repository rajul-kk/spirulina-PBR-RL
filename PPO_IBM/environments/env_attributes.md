# `light_env.py` — Physics Attributes Reference

## Reactor Configuration

| Attribute | Value | Description |
|---|---|---|
| `dt` | 0.01 h | Simulation time step per `step()` call |
| `reactor_depth` | 0.30 m | 1D depth axis (z), cells simulated vertically |
| `volume_L` | 30.0 L | Total bioreactor volume |
| `max_steps` | 14400 | Episode length (= 144 simulated hours / 6 days) |

---

## Cell Population Arrays (vectorized, size = `max_cells`)

| Array | Units | Description |
|---|---|---|
| `cells_z` | m | Per-cell z-depth position in reactor [0, reactor_depth] |
| `cells_mass` | pg | Per-cell biomass. Initial: N(1.25e7, 1e6) pg |
| `cells_quota` | (dimensionless) | Droop internal nutrient quota per cell [Q_min, Q_max] |
| `cells_acclimation` | µmol/m²/s | Per-cell light acclimation state (photo-adaptation EMA target) |
| `clump_mass` | (# cells in aggregate) | Per-cell flocculation state; 1.0 = single cell |
| `active_mask` | bool | True = alive cell slot |

---

## Bulk Chemistry State

| Variable | Units | Initial | Description |
|---|---|---|---|
| `ext_nutrients` | mg/L | 300.0 | External nitrogen/carbon dissolved nutrient pool |
| `ph` | — | 8.5 | Bulk pH (scale 6–11.5) |
| `do2` | mg/L | 7.0 | Dissolved oxygen concentration |
| `temp` | °C | 25.0 | Bulk temperature |
| `dissolved_co2` | mg/L | 2.0 | Dissolved inorganic carbon (CO2) |
| `salt` | mg/L | 1000.0 | Accumulated dissolved salts (from nutrient impurities + lysis) |
| `od` | (abs units) | computed | Optical density: `(total_mass_mg / volume_L) / 300.0` |

---

## Strain Biology Parameters (randomized per episode)

| Parameter | Distribution | Clipped | Description |
|---|---|---|---|
| `mu_max` | N(0.10, 0.05) h⁻¹ | ≥ 0.05 | Max specific growth rate |
| `Ks` | N(20.0, 5.0) mg/L | — | Nutrient half-saturation (Droop uptake) |
| `Ki` | N(120.0, 30.0) µmol/m²/s | ≥ 10 | Photo-inhibition threshold (Haldane) |
| `Kii` | N(2000.0, 500.0) µmol/m²/s | — | Haldane photoinhibition denominator |
| `T_opt` | N(27.0, 2.0) °C | [20, 35] | Optimal temperature for growth |
| `Q_min` | 1.5 (fixed) | — | Droop minimum quota (below = zero growth) |
| `Q_max` | 5.0 (fixed) | — | Upper Droop quota saturation limit |
| `tau_acclim` | U(1.0, 4.0) h | — | Photo-acclimation time constant (unused directly; EMA fixed at 0.1) |

**Genetic Micro-Drift:** Every 500 steps (≈ 5 simulated hours), `mu_max` and `Ks` each multiply by U(0.99, 1.01). Always active (no difficulty gate).

---

## Growth Rate Factors

All factors are multiplied together into `current_mu`:

```
current_mu = mu_max × f_I × f_Q × f_CO2 × temp_factor × shock_factor × f_O2 × f_pH × f_Osmosis × repair_tax
```

| Factor | Formula | Notes |
|---|---|---|
| `f_I` | `cells_I / (Ki + cells_I + cells_I²/2500)` | Haldane photo-inhibition (single-channel light) |
| `f_Q` | `max(0, 1 - Q_min / quota)` | Droop nutrient quota limitation |
| `f_CO2` | `dissolved_co2 / (0.5 + dissolved_co2)` | Monod CO2 limitation (Kc = 0.5 mg/L) |
| `temp_factor` | `exp(-0.5 × ((temp - T_opt)/5)²)` | Gaussian temperature response, σ=5°C |
| `shock_factor` | `exp(-3e-6 × (cells_I - acclimation)²)` | Photo-acclimation shock penalty |
| `f_O2` | `max(0, 1 - (do2/26)⁴)` | O2 toxicity onset at 26 mg/L |
| `f_pH` | Asymmetric Gaussian, peak 9.5 (σ=1.2 acid / σ=2.0 alkaline) | Spirulina alkaliphile model (Richmond 1988) |
| `f_Osmosis` | `exp(-0.5 × ((nutrients-2000)/500)²)` if nutrients > 2000 else 1.0 | Osmotic stress from nutrient overdose |
| `repair_tax` | `1 - 0.25 × clip((RPM-150)/50, 0, 1)` | Metabolic cost of membrane repair at RPM > 150 |

**Maintenance respiration:** `m_respiration = 0.010 × mu_max`  
**Net growth:** `net_mu = current_mu - m_respiration`  
**Mass update:** `cells_mass *= clip(exp(net_mu × dt), 0.5, 2.0)`

---

## Photo-Acclimation

| Attribute | Value | Description |
|---|---|---|
| EMA coefficient | 0.1 | `acclimation += 0.1 × (cells_I - acclimation)` per step |
| Shock scalar | 3e-6 | `shock_factor = exp(-3e-6 × (cells_I - acclimation)²)` |
| Light attenuation | Beer-Lambert with clump shading | `k_ext = 0.2 + 4.5×OD + k_scatter` |
| Clump self-shading | `clump_mass^(-1/3)` | Geometric surface/volume scaling |

---

## Clumping / Flocculation

| Parameter | Formula | Description |
|---|---|---|
| Stick probability | `(OD × 1e-3) × (1 - RPM/250)` | OD-driven, RPM-suppressed aggregation |
| Shear breakup | `0.5 × ((RPM-80)/120)² × clump^0.5 × dt` | Onset at 80 RPM, aggressive above 120 RPM |
| Clump floor | 1.0 | Cells always remain at least a single-cell aggregate |

---

## Cell Mixing / Vertical Transport

| Mode | Formula | Description |
|---|---|---|
| Active mixing | `v_macro + v_micro + Brownian diffusion` | RPM-coupled sinusoidal turbulence + diffusion noise |
| Sinking (no mix) | `-0.001 × dt` m/step | Gravity sedimentation at zero RPM |
| Boundary | Reflective (fold-back) | Cells bounce off z=0 and z=reactor_depth |
| Bubble scattering | `k_scatter = RPM × 0.025` | Contributes to light attenuation via `k_ext` |

---

## Cell Division & Lysis

| Parameter | Value | Description |
|---|---|---|
| Division threshold | `cells_mass > 1.4e7 pg` | ~12% above starting mass (2.5e7 for true doubling) |
| Division mechanics | Symmetric binary fission | Parent halved; child inherits mass, z, quota, acclimation |
| Background lysis rate | `1e-4 h⁻¹` | Always-on minimum death |
| Stress lysis | `2e-3 × stress_factor² h⁻¹` | Scales with `(m_respiration - mu) / m_respiration` |
| Death trigger | Probabilistic per step | `death_prob = lysis_rate × dt` |
| Population crash | `num_active < 10` → episode terminates | Hard constraint |

---

## Gas Exchange (k_La Model)

| Parameter | Formula | Description |
|---|---|---|
| Flow resistance | `1 + (OD/30)² × clump^0.5` | Viscosity/aggregation resistance, gentle at low OD |
| Surface-to-volume ratio | `(1/volume)^(1/3) ≈ 0.322` | 30L reactor constant |
| Base k_La | `sv_ratio × (0.5 + 4.0 × RPM/200)` | RPM-driven mass transfer coefficient |
| Effective k_La | `base_kLa / flow_resistance` | |
| O2 saturation | 8.0 mg/L | Air equilibrium at 25°C |
| CO2 saturation | 0.6 mg/L | Atmospheric equilibrium |
| CO2 dissolution | Bidirectional: `k_La × (co2_sat - dissolved_co2)` | Note: slightly more permissive than genetic_env (which uses degassing-only) |
| CO2 injection | `co2_flow × 0.44 / volume_L × dt` | CO2 sparging rate |
| CO2 Calvin consumption | `delta_mass_mg × 0.015` | Fixation proportional to growth |

---

## pH Model

| Step | Formula | Description |
|---|---|---|
| Biological rise | `(O2_production / volume) × 0.1` | Photosynthesis raises pH |
| Respiration drop | `(total_mass_mg / volume) × 0.0001` | Biomass respiration lowers pH |
| Buffer restore | `(9.5 - pH) × 0.10 × dt` | Zarrouk carbonate buffer spring (matches genetic_env) |
| `ph_biotic` | `pH + ph_rise - ph_drop_resp + ph_restore` | Biological pH trajectory |
| `ph_from_co2` | `8.5 - 1.2 × log10((dissolved_co2 + 0.03) / 0.6)` | DIC Henry's Law proxy |
| Blend | `0.95×pH + 0.05×(0.7×ph_from_co2 + 0.3×ph_biotic)` | 95% inertia EMA, 70/30 CO2/biotic blend |
| Clamp | [6.0, 11.5] | Hard physical limits |

---

## Temperature

| Parameter | Formula | Description |
|---|---|---|
| Ambient | 25.0°C | Passive cooling target |
| Heating | `(I_surface × 0.001) × dt` | Light-to-heat conversion |
| Cooling | `0.1 × (temp - 25) × dt` | Newton's law of cooling, τ=10h |
| Growth penalty | Gaussian `exp(-0.5 × ((T-T_opt)/5)²)` | σ=5°C Gaussian around T_opt |

---

## Pigment & Salinity

| Variable | Description |
|---|---|
| `pigment` | Health of photosynthetic pigment [0.2, 1.0]. Degrades under I > 1000 µmol/m²/s or N < 100 mg/L. Recovers at 0.01/h. |
| `rgb_absorbance` | `OD × pigment`. Proxy for chlorophyll, returned as last observation dimension. |
| `salt` | Accumulates from nutrient impurities (`uptake × 0.1`) and lysis (`mass × 5e-11`). Contributes to conductivity. |
| `conductivity` | `(salt + nutrients + 100 × |7-pH|) × (1 + 0.02 × (temp - 25))`. Returned as observation. |

---

## Biofouling

| Parameter | Formula | Description |
|---|---|---|
| `fouling_factor` | `+= (OD × 1e-5) × dt` every 100 steps | Slow accumulation proportional to biomass density |
| Effect | `I_surface × exp(-fouling_factor)` | Attenuates incident light reaching cells |

---

## Actuator Hardware (EMA Smoothing)

| Channel | EMA α | Action → Physical Range | Response time |
|---|---|---|---|
| Stirring (RPM) | 0.05 | [-1, 1] → [50, 200] RPM | ~20 steps (2 simulated hours) |
| Light (I_surface) | Instant | [-1, 1] → [0, 2000] µmol/m²/s | Immediate |
| Nutrients (flow) | 0.06 | [-1, 1] → [0, 100] mL/step | ~10 min horizon |
| CO2 (flow) | 0.15 | [-1, 1] → [0, 5] mL/min | ~4 min horizon |

---

## Sensor Noise & Lag

| Feature | Description |
|---|---|
| **RPM-coupled EMA lag** | pH and DO2 observations lag by 2–6 steps depending on RPM (high RPM = fast mixing = short lag) |
| **Stochastic jitter** | ±2% uniform noise applied to all observations each step |
| **Static drift** | Per-episode multiplicative drift `U(0.98, 1.02)` on each sensor channel |

---

## Reward Components

| Term | Formula | Signal type |
|---|---|---|
| `reward_do2` | `O2_production × 0.05` | Proxy for photosynthesis activity |
| `reward_biomass` | `0.10 × tanh(Δmass_mg / 0.01)` | Symmetric biomass delta; `1.5×` below OD 0.05 |
| `reward_lysis` | `-0.01 × pop_loss` | Per-cell death penalty |
| `reward_growth_rate` | `max(0, (OD - prev_OD)/dt) × 20.0` | OD growth rate bonus (pull-based, no plateau penalty) |
| `penalty_do2` | `-0.002 × max(0, do2-18)²` | Quadratic O2 toxicity penalty |
| `penalty_clump` | `-(clump_mean - 1.0) × 0.01` | Aggregation penalty |
| `action_smooth_penalty` | `0.001 × Σ(Δaction²)` | L2 smoothness constraint |
| **Population crash** | `-1000` + episode end | `num_active < 10` |
