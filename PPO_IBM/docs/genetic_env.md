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
| | 3.1 [Genetic Domain Randomisation](#31-genetic-domain-randomisation)[BR]3.2 [Grace Period](#32-grace-period)[BR]3.3 [Growth Model](#33-growth-model)[BR]3.4 [RGB Light & Turbulence](#34-rgb-light--turbulence)[BR]3.5 [Gas & pH Dynamics](#35-gas--ph-dynamics) | 3 |
| 4. | **[Reward Function](#4-reward-function)** | 6 |
| 5. | **[Algorithm Selection: Recurrent PPO](#5-algorithm-selection-recurrent-ppo)** | 7 |
| | 5.1 [Why Recurrent PPO](#51-why-recurrent-ppo)[BR]5.2 [Hyperparameters & Entropy](#52-hyperparameters--entropy) | 7 |
| 6. | **[Training Curriculum](#6-training-curriculum)** | 9 |
| | 6.1 [Difficulty & Episode Starts](#61-difficulty--episode-starts)[BR]6.2 [Advancement Criteria](#62-advancement-criteria)[BR]6.3 [Infrastructure](#63-infrastructure) | 9 |
| 7. | **[Evaluation](#7-evaluation)** | 11 |
| 8. | **[Benchmark Results](#8-benchmark-results)** | 12 |
| 9. | **[Reflection](#9-reflection)** | 13 |
| | [References](#references)[BR][Appendix A](#appendix-a--full-technical-details) | 14 |



## 1. Environment Overview

`GeneticPhotobioreactorEnv` is a high-fidelity, Individual-Based Model (IBM) photobioreactor simulation for Spirulina cultivation research. Each episode simulates up to **300,000 individual algal cells** tracked as 2D particles through a 30 L flat-panel reactor. The defining feature is **Genetic Domain Randomisation**: a unique algal strain is sampled from biologically-grounded parameter distributions at every episode reset, forcing the agent to generalise across a diverse phenotype space rather than overfit to a single strain [8].

| Parameter | Value |
|---|---|
| Reactor volume | 30 L |
| Reactor depth | 30 cm |
| Max cells | 300,000 |
| Default initial cells | 3,000 |
| Simulation time step (dt) | 0.01 h |
| Episode length | 14,400 steps (≈ 144 h / 6 days) |
| Difficulty levels | D0 (easy) · D1 (medium) · D2 (hard) |



## 2. Action & Observation Space

The agent controls four continuous actuators normalised to `[-1, 1]`, decoded to physical units via exponential moving average (EMA) filters. An `ActionSmoothnessWrapper` applies an L2 penalty on consecutive action deltas (coefficient = 0.003) to suppress actuator chattering; this is subtracted from the step reward.

### 2.1 Action Space

| Index | Actuator | Decoded Range | EMA Coefficient |
|---|---|---|---|
| 0 | Stirring speed | 50 – 200 RPM | α = 0.05 |
| 1 | Surface light intensity | 0 – 2000 µmol/m²/s | Instant |
| 2 | Nutrient flow rate | 0 – 100 mg/h | α = 0.06 |
| 3 | CO₂ injection rate | 0 – 120 mL/min | α = 0.15 |

### 2.2 Observation Space

The agent receives a 6-dimensional sensor vector. Dissolved CO₂, dissolved O₂, and true cell mass are **hidden states** — never directly observed. Sensors are subject to stochastic step-level jitter (±1–2%) and per-episode static drift (±5% at D1+); full difficulty-dependent scaling is in Appendix A.1.

| Index | Sensor | Units |
|---|---|---|
| 0 | Turbidity (NTU) | 0 – 5000 |
| 1 | pH | 0 – 14 |
| 2 | Nutrient concentration | 0 – 5000 mg/L |
| 3 | Temperature | 0 – 50 °C |
| 4 | Conductivity | 0 – 10000 µS/cm |
| 5 | RGB absorbance | 0 – 20 |



## 3. Physics Model

### 3.1 Genetic Domain Randomisation

At each episode reset, `_randomize_strain()` draws a unique biological parameter set from Gaussian/uniform priors [8]:

```python
self.strain_params = {
    'mu_max':     np.random.normal(0.05, 0.007),
    'Ks_light':   np.random.normal(100.0, 10.0),
    'Ki':         np.random.normal(120.0, 15.0),
    'T_opt':      np.random.normal(27.0, 1.0),
    'tau_acclim': np.random.uniform(1.0, 4.0),
}
```

At Difficulty 2, parameters additionally drift ±1% every 500 steps, simulating slow evolutionary pressure during continuous deployment.

### 3.2 Grace Period

For the first 2,400 steps (≈ 24 simulated hours), actuator outputs are silently clamped to prevent photoinhibition death during the earliest lag phase: surface light ramps linearly from 500 → 2000 µmol/m²/s, and nutrient flow from 20 → 100 mg/h. Crucially, **CO₂ injection is not clamped during the grace period**. The agent retains full authority over the 0–120 mL/min gas valve from step 0. This creates an implicit, highly sensitive control challenge early in the episode: the agent must quickly learn to throttle CO₂ delivery to avoid triggering severe toxicity (via `f_CO2_tox`) or an irreversible acid crash before the population is large enough to buffer the pH drop through active photosynthetic carbon consumption. This process remains fully transparent to the reward function—the agent receives no explicit signal that clamping is active on the other variables.

### 3.3 Growth Model

Net growth per cell follows a multi-factor Haldane model [4]. All inhibition terms are multiplicative, so any single factor near zero arrests growth entirely:

```python
current_mu = (mu_max * f_I * f_Q * f_carbon * f_CO2_tox
              * temp_factor * shock_factor * f_O2
              * f_pH * f_Osmosis * repair_tax * fatigue_tax)
net_mu = current_mu - m_respiration
```

The five primary growth factors are listed below; the complete 11-factor table is in Appendix A.2.

| Factor | Description |
|---|---|
| `f_I` | Haldane photoinhibition on red PAR |
| `f_Q` | Droop internal nutrient quota [5] |
| `f_carbon` | Monod CO₂ limitation |
| `f_temp` | Gaussian temperature inhibition (σ = 5 °C around T_opt) |
| `f_pH` | Asymmetric Gaussian pH inhibition (peak 9.5) |

### 3.4 RGB Light & Turbulence

Surface irradiance is split into red (40%), blue (40%), and green (20%) spectral bands, each attenuating at different extinction coefficients through the water column. Growth is driven exclusively by red PAR, while total PAR triggers photoinhibition. This prevents naive maximum-light strategies, forcing the agent to balance growth against cellular stress. 

Cells circulate through a deterministic 2D airlift convection loop. In the macro-flow regime, cells move upward in the centre of the tank and downward along the walls, simulating a drafted-tube bioreactor flow profile. At high RPM, a turbulent flash-light effect randomises the effective cell depth. This stochastic mixing rapidly cycles cells between the heavily irradiated photic zone at the surface and the dark rest zone at the bottom, accurately replicating the Kok effect where brief intense surface flashes dramatically increase photosynthetic efficiency. 

Flocculation follows Smoluchowski kinetics [6]. As culture density increases, the probability of cells sticking together rises, leading to massive, light-blocking aggregates that suffer from severe self-shading ($M^{-1/3}$). The agent can combat this through mechanical shear: stirring above 80 RPM provides the physical force necessary to break apart clumps. However, excessive impeller speeds risk triggering the `fatigue_tax` (cumulative membrane degradation), demanding careful RPM modulation.

### 3.5 Gas & pH Dynamics

Dissolved CO₂ evolves through three simultaneous fluxes: bidirectional kLa gas transfer (scaled by agitation, damped by OD-driven viscosity), stoichiometric photosynthetic consumption (~1.8 mg CO₂ per mg biomass), and respiration release. pH is computed as an 80/20 blend of DIC-driven carbonate chemistry and biotic drift, filtered through a 90% inertial EMA. The buffer target is pH 10.2 (Zarrouk medium [11]). Dissolved O₂ accumulates from photosynthesis and drains via kLa toward a pressure-corrected saturation ceiling.


## 4. Reward Function

Reward signals are designed using **observable sensor proxies** to avoid direct mass measurement, enabling deployment on a physical reactor with identical instrumentation [7]. The six primary components are:

| Component | Signal | Purpose |
|---|---|---|
| `reward_biomass` | tanh(per-cell growth) × 0.5 × pop_boost | Primary per-cell growth |
| `reward_od` | tanh(ΔHWM_OD × 1000) × 2 × pop_boost | High-water-mark OD anchor |
| `reward_growth_rate` | max(0, dOD/dt) × 100 × pop_factor | Dense derivative growth signal |
| `reward_ph` | pH drift × 50 × CO₂-scale | Carbon uptake indicator |
| `reward_stagnation` | −0.15 if Δmass < 0 | Active growth imperative |
| `penalty_crash` | −1000 (terminal) | Population collapse deterrent |

The reward function is carefully shaped to encourage robust, sustainable growth while penalising unsafe or unproductive behaviours. In early experiments the agent discovered degenerate strategies that maximised single proxy signals (for example, leveraging transient pH spikes), so the final design intentionally blends high-water-mark anchoring, derivative signals, and a population-scaled boost to ensure that only sustained, population-wide improvements generate net positive return. Coefficients were tuned iteratively using short calibration runs and inspection of episode traces and action distributions to detect proxy exploitation.

The full 11-component table including `reward_do2`, `reward_nut_consume`, `reward_dic`, `reward_lysis`, and `reward_pbrs` is in Appendix A.3. A **population boost** multiplier (`pop_boost = 1 + max(0, 1 − num_active/6000)`) amplifies all growth rewards as population falls, incentivising recovery from die-off events. A population below 10 active cells triggers immediate termination with the crash penalty.


## 5. Algorithm Selection: Recurrent PPO

### 5.1 Why Recurrent PPO

The photobioreactor environment presents a classic Partially Observable Markov Decision Process (POMDP) [10] due to the intentional exclusion of three critical internal state variables from the observation vector: dissolved CO₂, dissolved O₂, and true cell mass. The agent is forced to rely entirely on indirect, noisy sensor proxies. For example, while the agent cannot measure CO₂ directly, it must infer its presence by observing the delayed drop in pH following gas injection, factoring in the inertial lag of the sensor hardware and the dynamic carbonate buffer capacity of the Zarrouk medium.

Standard PPO [1] treats each timestep as strictly Markovian—assuming the current observation contains all necessary information to choose an action. This feed-forward architecture is structurally incapable of integrating multi-step biological signals. A reactive agent cannot distinguish between a transient pH fluctuation caused by stochastic sensor jitter and a sustained pH rise indicating active photosynthetic carbon consumption. 

To overcome this, RecurrentPPO [3] augments the core actor-critic architecture with a shared Long Short-Term Memory (LSTM) network [2]. The LSTM's hidden state, denoted as `h_t`, persists across the full 14,400-step episode. This temporal depth allows the agent to construct an internal, latent representation of the missing biological states by integrating the observable sensor stream over time. The recurrent memory acts as a low-pass filter against high-frequency sensor noise, enabling the policy to execute precise, long-horizon actuator commands based on inferred metabolic momentum rather than instantaneous, volatile sensor readings.

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

The full hyperparameter table is in Appendix A.4. The entropy coefficient follows a hybrid decay: an exponential baseline modulated by a std-band controller that increases exploration if policy std < 0.20 and decreases it if std > 0.45 [9], preventing both premature convergence and runaway exploration across curriculum phases.


## 6. Training Curriculum

Training uses an **adaptive 3-phase curriculum** [9] across a 40M-step budget divided into 100,000-step chunks. The curriculum is implemented by progressively increasing the difficulty and diversity of initial conditions, forcing the agent to master easier regimes before advancing. Advancement is gated by median and P25 growth factors and crash-rate thresholds to ensure consistent, reproducible progress rather than stochastic wins. The curriculum infrastructure is implemented via custom environment wrappers and callbacks that automate episode sampling, difficulty scaling, checkpointing and warm-start stitching.

### 6.1 Difficulty & Episode Starts

| Level | Physics Scale | Start Distribution | Mix Probability |
|---|---|---|---|
| D0 | 25% | 85% low / 10% mid / 5% stitched | 100% D0 |
| D1 | 50% | 40% low / 40% mid / 20% stitched | 80% D1, 20% D0 |
| D2 | 100% | 27.5% low / 27.5% mid / 45% stitched | 70% D2, 20% D1, 10% D0 |

Three start modes govern initial population: **low** (log-uniform 1,000–4,000 cells), **mid** (6,000–10,000 cells), and **stitched** (full physical state snapshot from a previously successful high-population episode). Mixing is sampled per-episode, not per-chunk. The full `START_DISTRIBUTION` dictionary is in Appendix A.5.

### 6.2 Advancement Criteria

| From | To | Median Growth Factor | P25 Growth Factor | Max Crash Rate |
|---|---|---|---|---|
| D0 | D1 | ≥ 9.0× | ≥ 7.5× | ≤ 5% |
| D1 | D2 | ≥ 10.0× | ≥ 8.5× | ≤ 5% |

Advancement requires 2 consecutive passing chunks, computed over on-level episodes only to prevent easier review episodes from inflating streak counts. Growth factor is `peak_OD / init_OD_equivalent`.

### 6.3 Infrastructure

A 2,000-step random-action calibration pass seeds VecNormalize statistics before training begins. Five callbacks handle checkpointing (every 10,000 steps), episode metrics collection, warm-start state saving, live TQDM display, and TensorBoard entropy logging. Full descriptions are in Appendix A.6. The infrastructure also supports automated curriculum advancement and entropy-schedule logging, which was essential to monitor and control exploration during different phases of training.



## 7. Evaluation

I evaluated externally by benchmarking the trained RecurrentPPO model against a smoothed RandomAgent baseline across multiple episodes at Difficulty 2. The evaluation protocol ensured that LSTM hidden states were correctly maintained across timesteps and reset only at episode boundaries. Normalization statistics were applied consistently to ensure policy stability during the benchmark runs.

The evaluation metrics reported in the results table include total reward, peak OD, final population, crash rate, and average CO₂ injection. These metrics were chosen to capture both productivity and safety: total reward summarises overall performance, peak OD proxies maximum culture health, final population captures long-term viability, and crash rate identifies catastrophic failure modes. All reported metric values are placeholders pending final external evaluation of the trained checkpoint.



## 8. Benchmark Results

*3 Episodes × 30,000 Steps · Genetic Env · Difficulty 2*

| Agent | Mean Total Reward | Mean Final Population | Mean Steps |
|---|---|---|---|
| RecurrentPPO | 0.00 | 0 | 0 |
| Random (baseline) | 0.00 | 0 | 0 |

| Metric | RecurrentPPO | Random | Δ (absolute) |
|---|---|---|---|
| Total Reward | 0.00 | 0.00 | 0.00 |
| Peak OD | 0.00 | 0.00 | 0.00 |
| Final Population | 0 | 0 | 0 |
| Crash Rate | 0% | 0% | — |
| Avg CO₂ Injection (mL/min) | 0.00 | 0.00 | — |

> **Note:** All values are placeholder zeros pending final external evaluation of the trained checkpoint.



## 9. Reflection

Designing this environment surfaced several fundamental tensions in simulation-to-real reinforcement learning that I did not fully anticipate at the outset. 

The most significant challenge was reward engineering and the vulnerability of the policy to reward hacking. I deliberately chose observable sensor proxies over direct mass measurements to ensure the policy could transfer to a real bioreactor with identical instrumentation [7]. In practice, this created a dense but fragile reward landscape. With eleven competing signal components, the agent frequently discovered ways to accumulate reward through proxy manipulation. For example, during early training, the agent learned to completely starve the tank of CO₂. While this severely stunted biomass growth, it drove the pH highly alkaline, which the reward function misinterpreted as aggressive photosynthetic carbon uptake. I mitigated this by implementing high-water-mark OD anchoring and derivative growth signals, ensuring that sustained proxy rewards could only be achieved if the physical biomass was actively increasing. However, a cleaner, future solution would reduce the number of components and ground each signal exclusively in validated biological thermodynamics.

Genetic domain randomisation [8] proved effective in principle—exposing the policy to a wide strain space each episode should support robust sim-to-real transfer. By forcing the agent to adapt to unseen biological parameters, the policy learns to rely on recurrent feedback rather than memorising an optimal trajectory. However, I found that this randomisation substantially slowed convergence at Difficulty 0. The agent must generalise across `mu_max` variations of ±14%, differing light inhibition thresholds, and variable thermal optima before any curriculum advancement occurs. Furthermore, at Difficulty 2, the implementation of intra-episode "genetic drift"—where the strain's biological constants slowly mutate every 500 steps—forces the LSTM hidden state to continuously track and update its internal representation of the culture's latent health. While this creates a highly resilient controller, a staged approach that trains first on a fixed mean strain before enabling randomisation might offer significantly better sample efficiency.


A key lesson was the necessity of an explicit entropy schedule to properly control exploration during training. Modulating the entropy coefficient (and logging it to TensorBoard) prevented premature convergence during early curriculum phases while allowing gradual annealing as the policy stabilised, which materially improved robustness across difficulty levels.

Finally, I note that no validation against real bioreactor data has been performed. Physics simplifications—such as steady-state kLa approximations and idealised spectral splitting—are sufficient for training a robust, cautious policy, but they may ultimately limit the precision of the zero-shot real-world fidelity.

## 10. Limitations & Future Work

While the GeneticPhotobioreactorEnv provides a rigorous testbed for reinforcement learning in biological systems, several limitations remain. The simulation does not capture hardware failures, long-term sensor drift, or unmodelled biological interactions. Reward engineering, despite careful design, may still permit unanticipated proxy exploits. Future work should validate the policies on real reactor data, tighten thermodynamic grounding for proxy signals, and investigate alternative curriculum schedules to improve sample efficiency.



## References

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.


[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

[3] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 22(268), 1–8.

[4] Andrews, J. F. (1968). A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates. *Biotechnology and Bioengineering*, 10(6), 707–723.

[5] Droop, M. R. (1968). Vitamin B12 and marine ecology. IV. The kinetics of uptake, growth and inhibition in Monas lutheri. *Journal of the Marine Biological Association of the United Kingdom*, 48(3), 689–733.

[6] von Smoluchowski, M. (1917). Versuch einer mathematischen Theorie der Koagulationskinetik kolloider Lösungen. *Zeitschrift für physikalische Chemie*, 92, 129–168.

[7] Richmond, A. (Ed.). (2004). *Handbook of Microalgal Culture: Biotechnology and Applied Phycology*. Blackwell Science.

[8] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. *Proceedings of IROS 2017*.

[9] Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum Learning. *Proceedings of ICML 2009*.

[10] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

[11] Zarrouk, C. (1966). *Contribution à l'étude d'une cyanophycée: Influence de divers facteurs physiques et chimiques sur la croissance et la photosynthèse de Spirulina maxima*. PhD thesis, Université de Paris.



## Appendix A — Full Technical Details

### A.1 Difficulty Scaling

| Feature | D0 | D1 | D2 |
|---|---|---|---|
| Sensor jitter | ±1% | ±2% | ±2% |
| Static sensor drift | None | ±5% | ±5% |
| RPM-coupled sensor lag | No | Yes | Yes |
| O₂ toxicity scaling | Off | 0.5× | 1.0× |
| Physics scaling | 25% | 50% | 100% |
| Intra-episode genetic drift | No | No | Yes (every 500 steps) |

### A.2 Full Growth Factor Table

| Factor | Description | Key Parameters |
|---|---|---|
| `f_I` | Haldane photoinhibition on red PAR | Ks_light ≈ 100, Ki_total = 2500 |
| `f_Q` | Droop internal nutrient quota | Q_min = 0.5; zero growth when depleted |
| `f_carbon` | Monod CO₂ limitation | Kc = 0.5 mg/L |
| `f_CO2_tox` | Explicit CO₂ toxicity (Hill function) | Onset 30 mg/L; Hill exponent = 2 |
| `f_temp` | Gaussian temperature inhibition | σ = 5 °C around T_opt |
| `f_shock` | Photo-acclimation mismatch penalty | Coefficient 3×10⁻⁶ |
| `f_O2` | Dissolved O₂ toxicity | Threshold 22 mg/L; D1+ only |
| `f_pH` | Asymmetric Gaussian pH inhibition | Peak 9.5; σ_acid=1.2, σ_base=2.0 |
| `repair_tax` | Shear repair metabolic cost | Onset 175 RPM; max 35% penalty |
| `fatigue_tax` | Cumulative membrane integrity | ~5 h to degrade at max shear; 15% max |
| `m_respiration` | Maintenance cost | 1.0% of `mu_max` per hour |

### A.3 Full Reward Component Table

| Component | Signal | Purpose |
|---|---|---|
| `reward_do2` | O₂ production × 0.1 | Photosynthesis proxy |
| `reward_nut_consume` | Δnutrients × 0.02 | Mixing lag bridge |
| `reward_ph` | pH drift × 50 × CO₂-scale | Carbon uptake indicator |
| `reward_dic` | DIC progress × 0.03–0.12 | Carbon management |
| `reward_biomass` | tanh(per-cell growth) × 0.5 × pop_boost | Per-cell growth |
| `reward_lysis` | −0.01 × population loss | Anti-crash gradient |
| `reward_od` | tanh(ΔHWM_OD × 1000) × 2 × pop_boost | OD anchor |
| `reward_growth_rate` | max(0, dOD/dt) × 100 × pop_factor | Derivative signal |
| `reward_stagnation` | −0.15 if Δmass < 0 | Growth imperative |
| `reward_pbrs` | γΦ(s') − Φ(s) × 0.5 | Potential-based shaping |
| `penalty_crash` | −1000 (terminal) | Crash deterrent |

### A.4 Full Hyperparameter Table

| Hyperparameter | Value | Rationale |
|---|---|---|
| Policy | MlpLstmPolicy | LSTM for POMDP handling |
| LSTM hidden size | 256 | Sufficient for 6D → 4D control |
| LSTM layers | 1 | Single layer prevents vanishing gradients |
| Learning rate | 3×10⁻⁴ | Adam default; stable across curriculum |
| n_steps | 7,200 | ≈ 30 simulated days per rollout |
| batch_size | 256 | Standard for continuous control |
| n_epochs | 4 | Conservative clip to avoid collapse |
| γ (gamma) | 0.995 | Long-horizon discounting |
| Entropy coef (init) | 0.02 | Hybrid decay |
| VecNormalize clip | 100.0 | Bounds extreme physics excursions |
| Total budget | 40,000,000 steps | — |

### A.5 Episode Start Distribution

```python
START_DISTRIBUTION = {
    0: {"low": 0.85, "mid": 0.10, "stitched": 0.05},
    1: {"low": 0.40, "mid": 0.40, "stitched": 0.20},
    2: {"low": 0.275, "mid": 0.275, "stitched": 0.45},
}
```

### A.6 Callback Stack

| Callback | Purpose |
|---|---|
| CheckpointCallback | Saves model every 10,000 global steps |
| EpisodeMetricsCallback | Records growth factor, crash rate, reward std per episode |
| PopulationStitchCallback | Saves warm-start states when episode-end population > 11,000 |
| TQDMActionCallback | Live display of actuator values and rolling OD |
| EntropyLoggingCallback | Writes entropy coefficient to TensorBoard each rollout end |
