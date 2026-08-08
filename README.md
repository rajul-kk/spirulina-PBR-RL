# PPO-IBM Photobioreactor Reinforcement Learning

An Individual-Based Model (IBM) photobioreactor simulation trained with Recurrent Proximal Policy Optimisation (RecurrentPPO). The agent controls four actuators in real-time to maximise algal biomass growth across a 6-day simulated batch culture.

---

## Repository Structure

```
ppo_ibm/
├── recurrent_ppo.py          # Training script — RecurrentPPO + SSM + Curriculum
├── evaluate_agent.py         # Benchmarking script — PPO vs TD-MPC2 vs SAC vs Random
├── visualize_env.py          # Pygame real-time environment visualiser
└── environments/
    ├── genetic_env.py        # Core IBM environment (Genetic Domain Randomisation)
    └── light_env.py          # Simplified single-spectrum environment (evaluation / ablation baseline)
```

---

## Files

### `recurrent_ppo.py`

The main training entry point. Implements a full adaptive curriculum with the following components:

| Component | Description |
|---|---|
| **Algorithm** | `RecurrentPPO` (sb3-contrib) with the default LSTM policy backbone |
| **Curriculum** | 3-tier difficulty (D0→D2) advancing on biological mastery criteria (`median_growth_factor`, `crash_rate`) |
| **EntropyTuning** | Automatically adjusts `ent_coef` to maintain policy `std` in the [0.35, 0.70] exploration band |
| **Population Stitching** | Saves high-population episode states to seed future "warm start" episodes |
| **VecNormalize** | Observation normalisation active; reward normalisation active only for fresh training |

**Key Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_steps` | 7,200 | 3 simulated days per update — captures full biological growth cycle |
| `batch_size` | 256 | Stable minibatch gradient diversity |
| `n_epochs` | 4 | Prevents Critic overfitting on individual rollouts |
| `ent_coef` | 0.05 | Maintains exploration entropy; prevents premature policy collapse |
| `learning_rate` | 3e-4 | Standard RecurrentPPO starting point |
| `gamma` | 0.99 | Long-horizon discounting for 6-day episodes |

**CLI Usage:**
```bash
# Fresh training
python ppo_ibm/recurrent_ppo.py

# Resume from latest checkpoint
python ppo_ibm/recurrent_ppo.py --continue

# Resume from specific checkpoint
python ppo_ibm/recurrent_ppo.py --continue path/to/model.zip

# Reset all training state
python ppo_ibm/recurrent_ppo.py --reset-training
```

---

### `genetic_env.py`

The primary training environment. Simulates a 30L flat-panel photobioreactor at individual-cell resolution using vectorised NumPy operations.

**Core Design:**
- Each "agent" in the simulation represents **250,000 physical cells** (super-agent scaling)
- Up to **300,000 agents** are tracked simultaneously with a fixed-size array and boolean `active_mask`
- **Genetic Domain Randomisation** randomises strain parameters (`mu_max`, `T_opt`, `Ki`, `Ks`) on every episode reset, forcing the policy to generalise across biological variance

**Action Space** `(4,)` continuous `[-1, 1]`:
| Index | Actuator | Physical Range |
|---|---|---|
| 0 | Stirring | 0–500 RPM |
| 1 | Light intensity | 0–2000 µmol/m²/s |
| 2 | Nutrient flow | 0–2000 mg/hr |
| 3 | CO₂ sparging | 0–440 mL/min |

**Observation Space** `(6,)`:
| Index | Sensor | Units |
|---|---|---|
| 0 | Turbidity | NTU |
| 1 | pH | — |
| 2 | External nutrients | mg/L |
| 3 | Temperature | °C |
| 4 | Conductivity | µS/cm |
| 5 | RGB ratio (Red/Blue) | — |

*Note: Dissolved Oxygen (DO₂) was removed from the observation space (but physical toxicity penalties remain active internally).*

**Curriculum Difficulty Tiers:**

| Tier | Temperature Physics | O₂ Toxicity | Strain Drift |
|---|---|---|---|
| D0 | 25% | Off | Off |
| D1 | 50% | On (halved) | Off |
| D2 | 100% | On (full) | Every 500 steps |

---

### `light_env.py`

A simplified, single-spectrum photobioreactor environment used as an **ablation baseline** and for rapid evaluation. It reduces physical complexity relative to `genetic_env` to isolate the effect of spectral modelling on agent performance.

**Key Differences from `genetic_env`:**

| Feature | `genetic_env` | `light_env` |
|---|---|---|
| Light spectrum | 3-channel (Red/Green/Blue) | Single broadband channel |
| Stirring range | 0–500 RPM | 0–200 RPM (constrained) |
| CO₂ range | 0–440 mL/min | 0–5 mL/min (lab-scale) |
| Nutrient range | 0–2000 mg/hr | 0–100 mg/hr |
| 2D cell positions | ✅ (X + Z axes) | ❌ (Z-axis only) |
| RGB ratio sensor | ✅ (phycocyanin proxy) | ✅ (pigment-weighted) |
| Intra-episode genetic drift | D2 only | Always on |
| Sensor calibration drift | Difficulty-scaled | ±2% per episode (always) |
| Shear repair tax | ❌ | ✅ (>150 RPM onset) |
| Stagnation penalty | Mass-based (`delta_mass < -0.0001`) | OD-based (`steps_since_od_high > 800`) |

**Action Space** `(4,)` continuous `[-1, 1]` — same indices as `genetic_env` but tighter physical ranges:
| Index | Actuator | Physical Range |
|---|---|---|
| 0 | Stirring | 50–200 RPM |
| 1 | Light intensity | 0–2000 µmol/m²/s |
| 2 | Nutrient flow | 0–100 mg/hr |
| 3 | CO₂ sparging | 0–5 mL/min |

**Observation Space** `(6,)` — identical indexing to `genetic_env`:
| Index | Sensor | Notes |
|---|---|---|
| 0 | Turbidity (NTU) | Mie scattering model with bubble/RPM noise |
| 1 | pH | RPM-coupled EMA lag (2–6 steps) |
| 2 | External nutrients (mg/L) | Direct reading |
| 3 | Temperature (°C) | Direct reading |
| 4 | Conductivity (µS/cm) | Salt + nutrient + pH proxy |
| 5 | RGB absorbance | pigment × OD |

**Notable Physics in `light_env`:**

**Mie Scattering Turbidity Model** — turbidity noise is explicitly RPM-dependent:
```python
flow_noise = 1.0 + 0.03 * (rpm / 200.0) * np.random.normal(0, 1)
```
This models bubble-induced optical scattering at the 860 nm near-IR wavelength used by real inline turbidity probes, where chlorophyll absorption is negligible.

**Shear Repair Tax** (novel, requires experimental validation):
```python
repair_factor = clip((rpm - 150.0) / 50.0, 0, 1)
repair_tax = 1.0 - (0.25 * repair_factor)
```
Models the metabolic energy diverted to membrane repair under high-shear conditions (>150 RPM). This imposes a 0–25% reduction in effective growth rate at the top of the RPM range. *Note: While membrane repair costs are biologically documented, the parametric form (linear onset, 25% max penalty) was tuned empirically and requires validation against experimental shear-stress culture data.*

**Stochastic Lysis Model** — death probability scales with nutritional stress:
```python
stress_factor = clip((m_respiration - mean_mu) / m_respiration, 0, 1)
lysis_rate = 1e-4 + 2e-3 * stress_factor²   # per hour
```
Background lysis rate of `~0.5%/day` matches reported dark-respiration death rates; stress-induced lysis of `~5%/day` at full starvation requires specific experimental confirmation.

---

## Physics & Literature References (`light_env.py`)

### Turbidity & Optical Sensing

**Mie Scattering at 860 nm (Near-IR Nephelometry)**
> Jonasz, M. & Fournier, G.R. (2007). *Light Scattering by Particles in Water: Theoretical and Experimental Foundations*. Academic Press.

Real inline turbidity sensors operate at 860 nm where chlorophyll-a and phycocyanin have negligible absorption. The `pigment_contrast = 0.7 + 0.3 × pigment` range (compressed to 0.7–1.0) reflects this reduced pigment sensitivity compared to visible-wavelength sensors.

**Multiple Scattering Saturation**
> Bohren, C.F. & Huffman, D.R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley.

Non-linear turbidity response at high cell density (`saturation_factor = 1 / (1 + 0.05 × OD)`) captures the onset of multiple-scattering regimes above OD ~5–10, where nephelometric readings plateau.

### Shear Stress & Membrane Integrity

**Shear Repair Tax (>150 RPM)**

> ⚠️ *New literature needed* — The 25% max growth penalty onset at 150 RPM is an empirical estimate. Experimental validation should reference:

> Moulton, T.P. (1990). *The biotechnology of commercially important microalgae*. In: Round, F.E. & Chapman, D.J. (eds), *Progress in Phycological Research*, Vol. 7.
> Molina Grima, E. et al. (1999). *Photobioreactors: light regime, mass transfer, and scaleup*. Journal of Biotechnology, 70(1–3), 231–247.

The repair cost model assumes that membrane lipid turnover under high-shear conditions diverts ~25% of biosynthetic capacity at 200 RPM. **Experimental measurements of specific shear thresholds and repair costs for Spirulina at these RPM ranges are not well established in the literature and this parametric form should be treated as a modelling assumption.**

### Stochastic Lysis

**Background Lysis Rate (~0.5%/day)**
> Vonshak, A. (1997). *Spirulina platensis (Arthrospira): Physiology, Cell Biology and Biotechnology*. Taylor & Francis, Chapter 3.

Background lysis at `1e-4 h⁻¹` (`~0.24%/day`) reflects natural trichome fragmentation and programmed cell death rates observed in unstressed Spirulina cultures.

**Stress-Induced Lysis (~5%/day at full starvation)**
> Markou, G. & Georgakakis, D. (2011). *Cultivation of filamentous cyanobacteria (blue-green algae) in agro-industrial wastes and wastewaters: A review*. Applied Energy, 88(10), 3389–3401.

> ⚠️ *New literature needed* — The `2e-3 × stress_factor²` stress lysis term (max `~5%/day`) extrapolates beyond well-characterised ranges. The quadratic scaling with nutritional stress intensity is a modelling choice requiring experimental calibration data on starvation-induced lysis kinetics in photobioreactor conditions.

### Osmotic Stress (Nutrient Inhibition)

**High-Concentration Nutrient Osmotic Inhibition (>2000 mg/L)**
> Borowitzka, M.A. (1998). *Limits to growth*. In: Borowitzka, M.A. & Borowitzka, L.J. (eds), *Micro-algal Biotechnology*. Cambridge University Press.
> Marquez, F.J. et al. (1995). *Growth of Spirulina platensis in open raceways at low and high cell concentrations*. Bioresource Technology, 52(1), 33–37.

The Gaussian inhibition model `f_Osmosis = exp(-0.5 × ((N - 2000)/500)²)` for nutrient concentrations above 2000 mg/L approximates the osmotic stress onset. The 500 mg/L width parameter is an estimate — **specific half-inhibition concentrations for Spirulina under nutrient enrichment are available in the cited sources but the exact Gaussian form should be validated against dose-response data.**

### Inorganic Carbon Limitation

**CO₂ Limitation (Michaelis-Menten form, Kc = 0.5 mg/L)**
> Livansky, K. (1990). *CO₂ losses in outdoor thin-layer culture units*. Algological Studies, 53/54, 131–138.
> Cornet, J.F., Dussap, C.G. & Dubertret, G. (1992). *A structured model for simulation of cultures of the cyanobacterium Spirulina platensis in photobioreactors*. Biotechnology & Bioengineering, 40(7), 817–825.

`f_CO2 = dissolved_CO2 / (0.5 + dissolved_CO2)` models the half-saturation constant for inorganic carbon at the RuBisCO active site. The Kc = 0.5 mg/L value is consistent with reported CO₂ affinities for cyanobacterial carbon concentrating mechanisms (CCMs).

### Sensor Noise & Calibration Drift

**Stochastic Sensor Jitter (±2%)**

> ⚠️ *New literature recommended* — The `±2%` per-step multiplicative jitter and per-episode calibration drift (`_sensor_drift_mult ∈ [0.98, 1.02]`) are engineering estimates for typical inline probe accuracy. Validation should reference instrument specifications:

> Endress+Hauser (2020). *Turbidity measurement in process technology*. Technical Application Guide TI01063K. *(or equivalent instrument datasheet for your specific probe model)*

This noise model exists to ensure the policy learns robustness to realistic sensor imprecision rather than fitting to noise-free observations.

---

### `evaluate_agent.py`

Benchmarks multiple trained agent types against each other on a standardised episode.

**Supported Agent Types:**
- `RecurrentPPO` (primary trained agent)
- `TD-MPC2` (model-predictive control baseline)
- `VarMPC` (variance-aware MPC baseline)
- `SAC` (off-policy actor-critic baseline)
- `Random` (smoothed random walk baseline)

**CLI Usage:**
```bash
# Evaluate RecurrentPPO on genetic env
python ppo_ibm/evaluate_agent.py --env genetic --episodes 10

# Compare all agents
python ppo_ibm/evaluate_agent.py --env heavy
```

Results are saved to `benchmark_results.csv`.

---

### `visualize_env.py`

A Pygame-based real-time visualiser for manual interaction with the environment.

**Display:**
- Particle rendering of all active cells (downsampled to 15,000 for 60 FPS)
- Color: green (healthy) → yellow (bleached pigment)
- Depth shading (Lambert exponential attenuation)
- Live sensor panel: OD, pH, Nutrients, Temperature, Conductivity, RGB

**Interactive Controls:**

| Key | Action |
|---|---|
| ↑ / ↓ | Stirring (±RPM) |
| → / ← | Light intensity |
| W / S | Nutrient flow |
| D / A | CO₂ sparging |

**CLI Usage:**
```bash
# Manual play (Genetic env, D0)
python ppo_ibm/visualize_env.py --env genetic --difficulty 0

# Record episodes to CSV
python ppo_ibm/visualize_env.py --env genetic --difficulty 1 --record-episode
```

---

## Physics & Literature References (`genetic_env.py`)

### Growth Kinetics

**Haldane (Andrews) Photoinhibition Model** — `f_I` calculation
> Andrews, J.F. (1968). *A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates*. Biotechnology & Bioengineering, 10(6), 707–723.

The model uses a split-spectrum Haldane formulation where growth is driven by red-band PAR and inhibition is driven by total irradiance:
`f_I = I_red / (Ks_light + I_red + I_total² / Kii)`

**Droop Cell Quota Model** — `f_Q` calculation
> Droop, M.R. (1968). *Vitamin B12 and marine ecology. IV. The kinetics of uptake, growth and inhibition in Monochrysis lutheri*. Journal of the Marine Biological Association UK, 48(3), 689–733.

Internal nutrient quota drives growth efficiency: `f_Q = max(0, 1 - Q_min / Q_cell)`

### pH & CO₂ Optimum

**Spirulina pH Optimum (pH 9.5)**
> Richmond, A. (1988). *Spirulina*. In: Borowitzka, M.A. & Borowitzka, L.J. (eds), *Micro-algal Biotechnology*. Cambridge University Press, pp. 85–121.
> Vonshak, A. (1997). *Spirulina platensis (Arthrospira): Physiology, Cell Biology and Biotechnology*. Taylor & Francis.
> Habib, M.A.B. et al. (2008). *A review on culture, production and use of Spirulina as food for humans and feeds for domestic animals and fish*. FAO Fisheries and Aquaculture Circular No. 1034.

The pH inhibition uses an asymmetric Gaussian (σ=1.2 acid side, σ=2.0 alkaline side) reflecting Spirulina's obligate alkaliphile physiology.

**Henry's Law CO₂ Dissolution**
> Stumm, W. & Morgan, J.J. (1996). *Aquatic Chemistry: Chemical Equilibria and Rates in Natural Waters*. 3rd ed., Wiley-Interscience.

CO₂ dissolution, degassing, and Calvin cycle consumption are modelled using equilibrium carbonate chemistry.

### Temperature

**Gaussian Temperature Optimum (T_opt ≈ 27°C)**
> Cornet, J.F., Dussap, C.G. & Dubertret, G. (1992). *A structured model for simulation of cultures of the cyanobacterium Spirulina platensis in photobioreactors*. Biotechnology & Bioengineering, 40(7), 817–825.

Temperature factor: `f_T = exp(-0.5 × ((T - T_opt) / 5.0)²)`

### Oxygen Toxicity

**Reactive Oxygen Species (ROS) Inhibition**
> Vonshak, A. & Richmond, A. (1988). *Mass production of the blue-green alga Spirulina: an overview*. Biomass, 15(4), 233–247.

Dissolved O₂ > 16 mg/L triggers ROS damage inhibition modelled as a steep quartic: `f_O2 = max(0, 1 - (DO₂/22)⁴)`

### Gas Transfer

**k_La (Volumetric Oxygen Transfer Coefficient)**
> Garcia-Ochoa, F. & Gomez, E. (2009). *Bioreactor scale-up and oxygen transfer rate in microbial processes: an overview*. Biotechnology Advances, 27(2), 153–176.

`k_La` is scaled by stirring RPM with a surface-to-volume correction for the 30L flat-panel geometry.

### Flocculation & Shear

**Kolmogorov Eddy Scale / Shear Fragmentation**
> Moulton, T.P. (1990). *The biotechnology of commercially important microalgae*. In: Round, F.E. & Chapman, D.J. (eds), *Progress in Phycological Research*, Vol. 7.

Shear breakup onset at ~80 RPM matches the Kolmogorov microscale for a 30L reactor. Clump aggregation uses a collision-probability model (`prob_stick ∝ OD × rpm_factor`).

**Cell Wall Fatigue (Membrane Integrity)**
> Rossignol, N. et al. (1999). *Membrane technology for the continuous separation microalgae/culture medium: compared performances of cross-flow microfiltration and ultrafiltration*. Aquacultural Engineering, 20(3), 191–208.

Accumulated shear stress above 80 RPM degrades membrane integrity over ~5-hour time constants, imposing a progressive growth penalty.

### Light Attenuation

**Beer-Lambert Law (Wavelength-Resolved)**
> Hu, Q. et al. (1996). *Combined effects of light intensity, light-path, and culture density on output rate of Spirulina platensis (Cyanobacteria)*. European Journal of Phycology, 31(2), 165–171.

Spectral attenuation coefficients `k_red`, `k_blue`, `k_green` are calibrated to Spirulina's phycocyanin and chlorophyll-a absorption peaks.

### Maintenance Respiration

> Raven, J.A. & Beardall, J. (2003). *Carbohydrate metabolism and respiration in algae*. In: Larkum, A.W.D. et al. (eds), *Photosynthesis in Algae*. Springer, pp. 205–224.

Maintenance respiration is modelled as a fixed fraction of `mu_max` (3%), representing the minimum metabolic cost to maintain cellular integrity in the dark.

---

## Dependencies

```
stable-baselines3
sb3-contrib
gymnasium
numpy
pygame
matplotlib
pandas
torch
tensorboard
```

---

## TensorBoard Monitoring

```bash
python -m tensorboard.main --logdir ppo_ibm/ppo_recurrent_tensorboard/ --port 6006
```

Navigate to `http://localhost:6006`. Key metrics to monitor:

| Metric | Healthy Range | Problem if... |
|---|---|---|
| `explained_variance` | 0.6–0.95 | < 0.3: Critic blind; oscillating: reward variance too high |
| `approx_kl` | 0.005–0.02 | > 0.05: reduce `n_epochs` |
| `train/std` | 0.3–0.8 | < 0.15: policy collapsed; > 1.2: too random |
| `entropy_loss` | Slowly rising | Sudden crash to 0: entropy collapse |
