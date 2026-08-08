# COMP2019 Individual Report 
# AIB Microalgae Commercialization Project
## Group 21 
### Supervisor: Dr. Yap Yee Jiun
### Date: April 29th, 2026

## Table of Contents

| | | |
|---|---|---|
| 1. | **[Environment Overview](#1-environment-overview)** | 1 |
| 2. | **[Action & Observation Space](#2-action--observation-space)** | 2 |
| | 2.1 [Action Space](#21-action-space)[BR]2.2 [Observation Space](#22-observation-space) | 2 |
| 3. | **[Physics Model](#3-physics-model)** | 3 |
| | 3.1 [Strain Randomisation](#31-strain-randomisation)[BR]3.2 [Growth Model](#32-growth-model)[BR]3.3 [Cell Transport (1D)](#33-cell-transport-1d)[BR]3.4 [Flocculation](#34-flocculation)[BR]3.5 [Gas Exchange & pH](#35-gas-exchange--ph) | 3 |
| 4. | **[Reward Function](#4-reward-function)** | 6 |
| 5. | **[Algorithm Selection: Recurrent PPO](#5-algorithm-selection-recurrent-ppo)** | 7 |
| | 5.1 [Why Recurrent PPO](#51-why-recurrent-ppo)[BR]5.2 [Hyperparameters & Entropy](#52-hyperparameters--entropy) | 7 |
| 6. | **[Training Curriculum](#6-training-curriculum)** | 9 |
| | 6.1 [Episode Start Modes](#61-episode-start-modes)[BR]6.2 [Advancement and Evaluation](#62-advancement-and-evaluation)[BR]6.3 [Infrastructure](#63-infrastructure) | 9 |
| 7. | **[Evaluation & Results](#7-evaluation--results)** | 11 |
| 8. | **[Reflection](#8-reflection)** | 12 |
| 9. | **[Limitations & Future Work](#9-limitations--future-work)** | 13 |
| | [References](#references)[BR][Appendix A](#appendix-a--full-technical-details) | 14 |


## 1. Environment Overview


`LightPhotobioreactorEnv` is a streamlined Individual-Based Model (IBM) photobioreactor simulation focused on **light-efficient Spirulina cultivation**. This variant tracks cells in **1D (vertical depth only)**, uses a simplified single-spectrum light model, and operates without a formal difficulty system. It is designed to be computationally efficient while retaining the core biological physics needed for light and nutrient management research [7].

A key design distinction is the **deliberate absence of a grace period**. The agent has full actuator authority from step 0, making early-episode stabilisation harder and demanding more careful initial control.

| Parameter | Value |
|---|---|
| Reactor volume | 30 L |
| Reactor depth | 30 cm |
| Max cells | 300,000 |
| Default initial cells | 2,000 |
| Simulation time step (dt) | 0.01 h |
| Episode length | 14,400 steps (≈ 144 h / 6 days) |
| Spatial tracking | 1D (z-axis depth only) |


## 2. Action & Observation Space

The agent controls four continuous actuators normalised to `[-1, 1]`, decoded via Exponential Moving Average (EMA) filters. An `ActionSmoothnessWrapper` applies an L2 penalty on consecutive action deltas (coefficient = 0.001) to suppress chattering; this is subtracted from the step reward.


### 2.1 Action Space

| Index | Actuator | Decoded Range | EMA Coefficient |
|---|---|---|---|
| 0 | Stirring speed | 50 – 200 RPM | α = 0.05 |
| 1 | Surface light intensity | 0 – 2000 µmol/m²/s | Instant |
| 2 | Nutrient flow rate | 0 – 100 mg/h | α = 0.06 |
| 3 | CO₂ injection rate | 0 – 120 mL/min | α = 0.15 |

The CO₂ range is broadened to 0–120 mL/min. This design change prevents the agent from "reward hacking" the pH mechanics by relying on its default Gaussian initialization (which previously mapped exactly to the optimal 2.5 mL/min biological requirement), thereby forcing it to actively learn pH regulation.


### 2.2 Observation Space

The agent receives a 6-dimensional sensor vector. Dissolved O₂ and CO₂ are tracked as full physics states but are **not included in the observation** — the agent must infer their status from indirect proxy sensors. True optical density (OD) is similarly unobserved; turbidity (NTU) serves as its noisy proxy. Observations are subject to step-level jitter drawn from U(0.98, 1.02) and a per-episode static calibration drift drawn from U(0.98, 1.02). The pH channel additionally receives RPM-coupled EMA smoothing (2–6 step lag). VecNormalize clips all values at `clip_obs = 100.0`.

| Index | Sensor | Units |
|---|---|---|
| 0 | Turbidity (NTU) | 0 – 5000 |
| 1 | pH | 0 – 14 |
| 2 | Nutrient concentration | 0 – 5000 mg/L |
| 3 | Temperature | 0 – 50 °C |
| 4 | Conductivity | 0 – 10000 µS/cm |
| 5 | RGB absorbance | 0 – 20 |



## 3. Physics Model

### 3.1 Strain Randomisation

At each episode reset, a unique algal strain is drawn from biologically-grounded Gaussian priors. Randomisation is unconditional — there is no difficulty gate:

| Parameter | Mean | Std / Range | Description |
|---|---|---|---|
| `mu_max` | 0.10 h⁻¹ | 0.05 | Maximum specific growth rate |
| `Ks` | 20.0 mg/L | 5.0 | Monod nutrient half-saturation |
| `Ki` | 120.0 µmol/m²/s | 30.0 | Photoinhibition constant |
| `T_opt` | 27.0 °C | 2.0 | Optimal growth temperature |
| `Q_min` | 1.5 | — | Droop minimum quota [5] |
| `tau_acclim` | — | U(1, 4) h | Photo-acclimation time constant |

Additionally, `mu_max` and `Ks` wander by ±1% every 500 steps throughout every episode, requiring the policy to adapt continuously to a slowly shifting strain.


### 3.2 Growth Model

Net growth uses a simplified single-spectrum photoinhibition model without explicit CO₂ toxicity (`f_CO2_tox`) or cumulative membrane fatigue (`fatigue_tax`) terms [4]:

```python
current_mu = (mu_max * f_I * f_Q * f_temp * f_shock
              * f_O2 * f_pH * f_Osmosis * f_CO2 * repair_tax)
net_mu = current_mu - m_respiration
```

The five primary factors are listed below; the full 10-factor table is in Appendix A.1.

| Factor | Description |
|---|---|
| `f_I` | Light saturation + photoinhibition |
| `f_Q` | Droop internal nutrient quota [5] |
| `f_temp` | Gaussian temperature factor (σ = 5 °C) |
| `f_O2` | Dissolved O₂ toxicity (threshold 26 mg/L) |
| `f_pH` | Asymmetric Gaussian pH inhibition (peak 9.5) |

A fouling factor accumulates proportionally to culture density every 100 steps and exponentially attenuates the surface irradiance received by the culture. Under high irradiance (>1000 µmol m⁻² s⁻¹) or nutrient stress (<100 mg/L), the model bleaches Spirulina's phycocyanin pigment, reducing per-cell light absorption until conditions recover.


### 3.3 Cell Transport (1D)

Mixing uses sinusoidal macro- and micro-turbulence along the z-axis, avoiding horizontal convection to minimise computational overhead. At low mixing intensities, cells sediment via Fickian diffusion (passive downward drift under gravity). However, at higher RPMs, a **turbulent flash-light effect** accurately replicates the biological Kok effect: stochastic mixing rapidly cycles cells between the heavily irradiated photic zone at the surface and the dark rest zone at the bottom, increasing overall photosynthetic efficiency.

Crucially, light attenuation features full **RGB spectral splitting**. Surface irradiance is divided into red (40%), blue (40%), and green (20%) bands, each attenuating at different extinction coefficients down the z-axis. Growth is driven exclusively by red PAR (Photosynthetically Active Radiation), but total PAR triggers photoinhibition. The extinction coefficients also dynamically scale with turbidity, bubble scattering (from RPM), and cell clumping. The full transport equations are in Appendix A.2.


### 3.4 Flocculation and Probabilistic Lysis

Flocculation follows mean-field Smoluchowski dynamics [6] (a collision-rate model for how particles clump) each step: sticking probability rises with optical density (OD) and drops under high-RPM shear, while Brownian breakup limits runaway aggregation when mixing is low. Clumps are particularly costly because they self-shade ($f\_clump\_shade = M^{−1/3}$), cutting light access for the cells trapped inside them.

Rather than a hard death threshold, starvation is modelled via a **probabilistic lysis model**. Background die-off runs at ~0.5% per day, but as internal quotas (`f_Q`) fall, that rate climbs to ~5%. Getting the nutrient feed wrong even briefly can compound into a die-off that is very difficult to recover from across a 6-day episode.


### 3.5 Gas Exchange & pH

The environment uses mass-balance for gas exchange. Dissolved Oxygen (DO₂) accumulates via photosynthesis (~1.2 mg O₂ per mg biomass) and drains through gas transfer governed by $k_La$ (the volumetric mass transfer coefficient). Without a DO₂ stripping actuator, the agent must use stirring to prevent DO₂ from exceeding the 26 mg/L toxicity threshold. Dissolved inorganic carbon (DIC) evolves via equilibration, direct injection, and consumption (~0.015 mg CO₂ per mg biomass).

pH dynamics, the core proxy reward signal, are modeled as a blend of carbonate chemistry and biotic drift. This state is filtered via a 95% EMA to simulate industrial probe lag. The system targets a Zarrouk-like buffer of pH 9.5 [10] — the standard alkaline growth medium for Spirulina. $k_La$ scales with stirring but is damped by viscosity changes from high OD or clumping.

Temperature is not static either: the impeller adds viscous heat scaling with RPM³ (up to ~0.5°C at 200 RPM), surface light contributes additional warming, and the reactor cools passively toward 25°C ambient. Since conductivity is temperature-scaled, stirring speed also indirectly shapes that observed signal.



## 4. Reward Function

The reward is focused on **mass productivity and oxygen dynamics** using proxy signals tuned to the simplified physics model:

```python
reward = (productivity * 1e-8) + (n_spawns * 0.1) + (mean_f_Q * 0.05) \
       + reward_do2 + reward_growth_rate \
       - penalty_do2 - penalty_clump - action_smooth_penalty
```

Stagnation and crash penalties are conditional/terminal and omitted from the inline formula above; they appear in the full table in Appendix A.2.

| Component | Signal | Purpose |
|---|---|---|
| `productivity × 1e−8` | Net mass change per step | Primary biomass signal |
| `n_spawns × 0.1` | Count of division events | Rewards successful division |
| `mean_f_Q × 0.05` | Mean internal quota | Nutrient adequacy |
| `reward_growth_rate` | max(0, dOD/dt) × 20 × pop_factor | Dense derivative growth signal |
| `penalty_do2` | −0.002 × max(0, DO₂−18)² | O₂ toxicity deterrent |
| `reward_stagnation` | −0.05 if OD High Water Mark (HWM) stale > 800 steps | Anti-plateau penalty |
| `penalty_crash` | −1000 (terminal) | Population collapse deterrent |

The reward function is designed to encourage sustained, safe productivity while discouraging risky or degenerate behaviours. Early tuning revealed policies that occasionally exploited single proxies during transient physics regimes; to counter this the final reward mixes direct productivity, division counts and anti-stagnation penalties so that short-lived proxy gains do not yield sustained reward. Coefficients were adjusted through short calibration runs and trace inspection to reduce proxy exploitation.

The full 10-component table is in Appendix A.2. A **population factor** (`pop_factor = 1 + max(0, 1 − num_active/6000)`) amplifies growth rewards as population falls, incentivising recovery from die-off events. A population below 10 active cells triggers immediate termination.



## 5. Algorithm Selection: Recurrent PPO

### 5.1 Why Recurrent PPO

`LightPhotobioreactorEnv` is **partially observable by design** [9]. Dissolved O₂, dissolved CO₂, and true cell mass are all hidden from the observation vector—intentionally, to push the agent toward strategies that would generalise to real reactors where these states are rarely measured directly.

Standard PPO [1] is Markovian: it maps each sensor snapshot to an action with no memory of what came before. That breaks down here. A memoryless agent cannot tell whether a rising pH reflects genuine photosynthetic carbon drawdown or just valve lag from a recent CO₂ injection—two situations that call for opposite responses.

RecurrentPPO [3] handles this by pairing the actor-critic with an LSTM [2]. The hidden state `h_t` builds up a running picture of the culture's trajectory across the episode, letting the policy track quantities it cannot observe directly. In practice this means it can preemptively ramp CO₂ ahead of a pH spike rather than reacting after the fact.

### 5.2 Hyperparameters & Entropy

```python
model = RecurrentPPO(
    "MlpLstmPolicy", vec_env,
    learning_rate=3e-4, n_steps=7200,
    batch_size=256,     n_epochs=4,
    ent_coef=ENTROPY_INIT, gamma=0.995,
    policy_kwargs={"lstm_hidden_size": 256, "n_lstm_layers": 1},
)
```

The full hyperparameter table is in Appendix A.3. The entropy coefficient follows a hybrid decay: an exponential baseline modulated by a std-band controller that boosts exploration if policy std < 0.20 and reduces it if std > 0.45 [8], keeping the policy in a productive exploration band throughout training.



## 6. Training Curriculum

`LightPhotobioreactorEnv` is fully compatible with the `CurriculumStartWrapper` and `PopulationStitchCallback` infrastructure from `recurrent_ppo.py` and can be used as a warm-up or standalone training environment. When used with `recurrent_ppo.py`, the curriculum varies initial population distributions and gates harder start modes using mastery thresholds computed over 100,000-step chunks. In standalone mode, as described in Section 6.2, no such gates apply.


### 6.1 Episode Start Modes

Three population initialisation modes are sampled per-episode. **Low** starts draw a log-uniform 1,000–4,000 cell population. **Mid** starts initialise at 6,000–10,000 cells to challenge mid-growth management. **Stitched** starts restore a full physical state snapshot from a previously successful high-population episode. Because `light_env` is 1D, `cells_x` is not stored or restored. Stitched starts are gated until 20 completed episodes. The full `START_DISTRIBUTION` dictionary is in Appendix A.4.


### 6.2 Advancement and Evaluation

There is no formal difficulty curriculum here—no D0/D1/D2 stages or mastery streaks. Every episode uses the same physics, and progress is judged by how well the policy maintains stability and avoids crashes across varied starting populations. Growth factor (`peak_OD / init_OD_equivalent`) is still logged but does not gate anything. The 2D predecessor to this environment did implement a staged curriculum, but the PPO never made sense of the added spatial complexity, so that design was scrapped in favour of this simpler 1D formulation.


### 6.3 Infrastructure

Five callbacks manage training infrastructure: checkpointing, episode metrics collection, warm-start state saving, live TQDM display, and TensorBoard entropy logging. Full descriptions are in Appendix A.5. The entropy coefficient and policy standard deviation are also written to TensorBoard each rollout, which proved useful for catching policy collapse early during training runs.



## 7. Evaluation & Results

I evaluated the model externally by benchmarking the trained RecurrentPPO agent against a smoothed RandomAgent baseline. The evaluation used a 30,000-step horizon—exceeding the 14,400-step training episode length—to test long-horizon generalisation beyond the trained episode boundary. Normalisation statistics were loaded and applied to maintain consistency with the training distribution.

*3 Episodes × 30,000 Steps · Light Env · Standard Configuration*

| Metric | RecurrentPPO | Random | Δ (absolute) |
|---|---|---|---|
| Total Reward | 12411.20 | 822.79 | +11588.41 |
| Peak OD | 0.0917 | 0.0074 | +0.0843 |
| Final Population | 66452 | 5347.67 | +61104.33 |
| Crash Rate | 0% | 0% | — |
| Avg CO₂ Injection (mL/min) | 56.33 | 60.45 | -4.12 |

The RecurrentPPO achieved roughly 15× the cumulative reward and 12× the final population of the random baseline while using slightly less CO₂ on average (56.33 vs 60.45 mL/min), suggesting more consistent pH regulation (and well timed CO₂ injections into the tank). Both agents avoided outright crashes, but the random baseline plateaued at low density while the PPO sustained growth across the full 30,000-step window.


## 8. Reflection

Developing `LightPhotobioreactorEnv` highlighted the trade-off between biological fidelity and training stability. A critical design choice was the removal of the "grace period," granting the agent full authority from step 0. This forced the agent to handle early-episode instability—often exacerbated by aggressive CO₂ and inconsistent lighting, necessitating the use of stitched episodes to maintain a stable and high-density reward signal.

The 1D vertical model prioritises computational speed, allowing for rapid iteration on depth-dependent light and spectral attenuation. While my decision to omit horizontal convection and airlift heterogeneity limits its use as a full "digital twin," it serves as an effective tool for exploring lighting optimisation and lysis risks.

Reward calibration was simplified by using fewer proxy signals. However, the stagnation penalty sometimes conflicted with growth rewards in physics-constrained low-light regimes. Interestingly, the agent often discovered a constant CO₂ injection strategy to stabilise pH, a behaviour that emerged with and without CO₂ toxicity.

I also learnt by trial and error, the difficulties of working with entropy control, requiring extensive testing and tuning to balance exploration and exploitation. In the end, I opted for an approach that still encouraged exploration early in the training while maintaining stability later on.

Finally, I highlight that the recurrent architecture remains essential. By integrating temporal sensor data, the agent filters noise and manages mixing-induced lags. This memory allows the policy to track hidden biological states, like dissolved oxygen accumulation or pH shifts, confirming that recurrent policies are vital for controlling biological systems where state variables are often unobserved.



## 9. Limitations & Future Work

`LightPhotobioreactorEnv` is a streamlined testbed with several known gaps. The 1D vertical model eliminates horizontal heterogeneity, so the agent never encounters the spatial gradients that real airlift or paddle-wheel reactors produce. The simulation is also single-species: competitive dynamics, contamination, and predator-prey interactions are absent. Sensor noise is limited to multiplicative jitter and static calibration drift; structured faults like probe fouling or actuator saturation are not modelled.

Future work should focus on validating policies on physical reactors and extending the model to multi-species scenarios. Thermodynamically grounded gas exchange and alternative training curricula—or model-based approaches that explicitly estimate hidden states—could also improve sample efficiency and robustness.

## References

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.


[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

[3] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 22(268), 1–8.

[4] Andrews, J. F. (1968). A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates. *Biotechnology and Bioengineering*, 10(6), 707–723.

[5] Droop, M. R. (1968). Vitamin B12 and marine ecology. IV. The kinetics of uptake, growth and inhibition in Monas lutheri. *Journal of the Marine Biological Association of the United Kingdom*, 48(3), 689–733.

[6] von Smoluchowski, M. (1917). Versuch einer mathematischen Theorie der Koagulationskinetik kolloider Lösungen. *Zeitschrift für physikalische Chemie*, 92, 129–168.

[7] Richmond, A. (Ed.). (2004). *Handbook of Microalgal Culture: Biotechnology and Applied Phycology*. Blackwell Science.

[8] Ahmed, Z., Roux, N. L., Norouzi, M., & Schuurmans, D. (2019). Understanding the Impact of Entropy on Policy Optimization. *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*, PMLR 97.

[9] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

[10] Zarrouk, C. (1966). *Contribution à l'étude d'une cyanophycée: Influence de divers facteurs physiques et chimiques sur la croissance et la photosynthèse de Spirulina maxima*. PhD thesis, Université de Paris.



## Appendix A — Full Technical Details

### A.1 Full Growth Factor Table

| Factor | Description | Key Parameters |
|---|---|---|
| `f_I` | Light saturation + photoinhibition | Monod with photoinhibition: Ki + I²/2500 |
| `f_Q` | Droop internal quota | Q_min = 1.5 |
| `f_temp` | Gaussian temperature factor | σ = 5 °C around T_opt |
| `f_shock` | Photo-acclimation mismatch penalty | Coefficient 3×10⁻⁶ |
| `f_O2` | Dissolved O₂ toxicity | Threshold 26 mg/L |
| `f_pH` | Asymmetric Gaussian pH inhibition | Peak 9.5; σ_acid=1.2, σ_base=2.0 |
| `f_osmosis` | Osmotic stress | Gaussian penalty above 2000 mg/L nutrients |
| `f_CO2` | Carbon limitation | Kc = 0.5 mg/L |
| `repair_tax` | Shear repair cost | Onset 150 RPM; max 25% penalty at 200 RPM |
| `m_respiration` | Maintenance cost (subtracted from `net_mu`, not multiplicative) | 1.0% of `mu_max` per hour |

### A.2 Cell Transport Equations & Full Reward Table

**1D Transport:**
```
v_macro = 0.005 · mix_intensity · sin(100·z − 5·t)
v_micro = 0.002 · mix_intensity · sin(500·z − 20·t + cell_index)
dz = (v_macro + v_micro) · dt_sec + √(2D) · ξ
```

**Full Reward Components:**

| Component | Signal | Description |
|---|---|---|
| `productivity × 1e−8` | Net mass change per step | Primary biomass signal |
| `n_spawns × 0.1` | Count of division events | Rewards successful division |
| `mean_f_Q × 0.05` | Mean internal quota | Nutrient adequacy |
| `reward_do2` | O₂ production × 0.05 | Photosynthesis proxy |
| `reward_growth_rate` | max(0, dOD/dt) × 20 × pop_factor | Derivative growth signal |
| `penalty_do2` | −0.002 × max(0, DO₂−18)² | O₂ toxicity deterrent |
| `penalty_clump` | −0.01 × (mean_clump−1) | Anti-aggregation signal |
| `reward_stagnation` | −0.05 if OD High Water Mark (HWM) stale > 800 steps | Anti-plateau penalty |
| `action_smooth_penalty` | −coef × Σ(Δa)² | Actuator smoothness |
| `crash penalty` | −1000 (terminal) | Crash deterrent |

### A.3 Full Hyperparameter Table

| Hyperparameter | Value | Rationale |
|---|---|---|
| Policy | MlpLstmPolicy | LSTM for POMDP handling |
| LSTM hidden size | 256 | Sufficient for 6D → 4D control |
| LSTM layers | 1 | Single layer avoids vanishing gradients |
| Learning rate | 3×10⁻⁴ | Adam default; stable for this reward scale |
| n_steps | 7,200 | ≈ 30 simulated days per update batch |
| batch_size | 256 | Standard mini-batch |
| n_epochs | 4 | Conservative PPO clip |
| γ (gamma) | 0.995 | Long-horizon discounting |
| Entropy coef (init) | 0.02 | Hybrid decay |
| VecNormalize clip | 100.0 | Bounds extreme physics excursions |
| Total budget | 40,000,000 steps | — |

### A.4 Episode Start Distribution

```python
START_DISTRIBUTION = {
    0: {"low": 0.85, "mid": 0.10, "stitched": 0.05},  # early training
    1: {"low": 0.40, "mid": 0.40, "stitched": 0.20},  # mid training
    2: {"low": 0.275, "mid": 0.275, "stitched": 0.45}, # late training
}
```

### A.5 Callback Stack

| Callback | Purpose |
|---|---|
| CheckpointCallback | Saves model every 10,000 global steps |
| EpisodeMetricsCallback | Records growth factor, crash rate, reward std per episode |
| PopulationStitchCallback | Saves warm-start states when episode-end population > 11,000 |
| TQDMActionCallback | Live display of actuator values and rolling mean OD |
| EntropyLoggingCallback | Writes entropy coefficient to TensorBoard each rollout end |
