# Known Limitations (Accepted Oversimplifications)

These are intentional simplifications that were reviewed and kept. Each entry explains the physical reality, why the simplified version is acceptable for current training, and what the trigger for fixing it is.

---

## O5 — O2 Solubility is Not Temperature-Dependent

**Reality:** O2 saturation drops ~2% per °C above 25°C (Henry's law). Over the 25–35°C operating range this is a ~20% error in `o2_sat`. Real DO2 will be lower than the model predicts at high temperature.

**Why kept:** The env's operating temperature rarely exceeds 30°C under a well-trained policy. The 10–15% error in DO2 saturation does not invert any control signal — the agent still learns to avoid DO2 extremes correctly. Fixing it requires a temperature-dependent Henry's constant lookup (trivial to add but low impact now).

**Fix trigger:** Before deploying to a real PBR where temperature routinely exceeds 30°C, or when fitting the model to real DO2 data that shows consistent negative residuals at high temperature.

**Minimum fix:**
```python
# Replace fixed o2_sat with temperature-corrected value
o2_sat = 8.0 * (o2_frac / self.ambient_o2_frac) * np.exp(-0.02 * (self.temp - 25.0))
```

---

## O6 — No Evaporative / Sparging Cooling

**Reality:** In a 30L sparged PBR at 0.3 L/min air, evaporative cooling provides approximately 0.1–0.2°C/h of passive cooling. CO2 sparging also carries latent heat out of the liquid.

**Why kept:** The existing ambient cooling term (`0.1 * (temp - ambient) * dt`) already approximates this magnitude. The combined error is < 0.3°C in steady state, which has a negligible effect on the temperature factor `f_T` (Gaussian peak at T_opt with σ=5°C).

**Fix trigger:** When fitting the temperature model to real PBR data and the steady-state temperature bias exceeds 1°C.

---

## O7 — Conductivity Absolute Calibration Gap (~3× Low) *(formula upgraded)*

**Status:** Kohlrausch molar conductance formula now implemented (σ = Σ λᵢcᵢ × temperature correction). Relative dynamics are physically correct. **Remaining gap:** absolute value is ~3× lower than real Zarrouk medium (~6400 µS/cm simulated vs ~18000–25000 µS/cm real). Root cause: NaHCO₃ (~16.8 g/L in Zarrouk, dominant ion) is not tracked as a separate state variable — its contribution is folded into the `salt` proxy (default 1000 mg/L NaCl-equivalent), which gives only ~2160 µS/cm vs the ~19000 µS/cm NaHCO₃ actually provides. The observation bound was widened to 25000 µS/cm.

**Why kept:** VecNormalize handles scale. The agent does not reason about the absolute conductivity value — only relative changes within an episode. N uptake, P uptake, and pH shifts all produce correct directional conductivity responses.

**Fix trigger:** When bootstrapping VecNormalize from real sensor data. At that point, either (a) add NaHCO₃ as a tracked state or (b) fit a linear calibration `conductivity_real = a × conductivity_sim + b` and bake it into observation scaling.

---

## O8 — Flash-Light (Kok) Effect Uses Linear Interpolation

**Reality:** The Kok effect (photosynthetic enhancement from intermittent high-intensity light flashes) operates at millisecond–second timescales. The mechanism involves the dark-reaction pool of acceptors being replenished during the dark fraction of each flash cycle. The real enhancement is non-linear and depends on flash frequency and duration.

**Why kept:** `dt = 0.02h = 72s` cannot resolve sub-second flash dynamics. The current implementation linearly interpolates between stratified depth and uniformly random depth as a function of `mix_intensity`. This is a standard aggregate approximation used in PBR modelling (e.g., Pruvost et al. 2008). The error is in the *magnitude* of the Kok enhancement, not its direction.

**Fix trigger:** If training in a flat-panel PBR geometry where flash frequencies can be mechanistically modelled (much shorter optical path length, known rotation frequency). Not applicable to the current 30L fish-tank geometry.

---

## O9 — Reward Has 10+ Overlapping Components

**Reality:** A well-designed reward should be parsimonious. Multiple terms tracking the same physical quantity (biomass growth) with different lags and scales makes gradient attribution difficult and can create reward hacking opportunities.

**Why kept:** The curriculum structure (D0 → D1 → D2) depends on dense early reward signals to bootstrap learning before the agent can produce measurable OD gains. The overlapping components (do2 proxy, nutrient proxy, pH proxy, biomass, od anchor, growth rate) serve as scaffolding at different timescales. Removing them before the agent has a stable D2 policy risks learning collapse.

**Fix trigger:** After a D2 policy achieves consistent growth (peak OD > 0.10 across > 80% of episodes), simplify by removing all components except `reward_biomass`, `reward_od`, and `reward_growth_rate`, and retrain from that checkpoint to check for performance regression. Reward simplification is a post-curriculum cleanup step.

---

## O10 — gamma = 0.995 Gives Only ~200-Step Effective Horizon (4 Sim-Hours)

**Reality:** At `gamma = 0.995` and `dt = 0.02h`, the discounted return at step 7200 (end of 144h episode) is `0.995^7200 ≈ 10^{-16}`. The effective horizon (where discount weight > 1/e) is about 200 steps ≈ 4 hours of sim time. This means the policy cannot plan growth curves that span multiple days — it optimises proximal growth rate, not final batch OD. A terminal bonus (`tanh(OD/0.05) × 10`) was added to partially anchor the value function at episode end, but does not fully replace long-horizon planning.

**Why kept:** Dense proxy rewards (DO₂, nutrient consumption, pH drift, OD growth rate) are designed to fire within the 4-hour horizon. Higher gamma without this scaffolding would produce sparse reward and unstable gradients.

**Fix trigger:** Once a stable D2 policy achieves peak OD > 0.10 consistently. At that point experiment with `gamma = 0.998–0.999` and verify the agent doesn't destabilise. Simultaneously remove or reduce proxy reward scaffolding so the higher gamma doesn't over-weight them.

---

## O11 — Single Training Environment (1x DummyVecEnv)

**Reality:** RecurrentPPO with a single environment means:
- Each rollout buffer (`n_steps = 7200`) samples from exactly one trajectory
- No diversity in LSTM hidden state initialization across rollouts
- Throughput is bounded by a single env's step rate

**Why kept:** The IBM simulation uses 7500 super-agent arrays per step (scaled from 300k; each super-agent = 2.5M real cells). Even at this reduced scale, parallel envs multiply memory and CPU proportionally. The current single-env setup is stable and produces consistent learning signals via Genetic Domain Randomization (new strain per episode).

**Fix trigger:** When env step rate drops below ~200 steps/sec or sample efficiency becomes the bottleneck. At that point, use `SubprocVecEnv` with 2–4 workers. Even 2 parallel envs with `n_steps=3600` each gives 2× trajectory diversity at similar rollout length.

**Quick win:** 2 parallel envs × `n_steps=3600` ≈ same total data, 2× strain diversity per update.

---

## O12 — reward_stagnation *(resolved: now proportional)*

**Previous issue:** Flat `-0.15` penalty fired identically for a single dying cell or a 5% culture crash, making the agent defensively prevent any OD dip.

**Current implementation:** Proportional severity — `severity = clip(|delta_mass_mg| / 0.5, 0.1, 1.0)`, penalty = `-0.15 × severity`. A single cell death gives ≈ −0.015; a 0.5 mg/step crash gives the full −0.15. The floor of 0.1 ensures any genuine decline is noticed.

**Remaining limitation:** The 0.5 mg/step full-penalty threshold is calibrated for a 3000-agent culture. At max population (7500 agents), the same threshold represents a smaller fractional decline (~0.7% vs ~1.3%). This may slightly under-penalise large absolute crashes at full density.

**Fix trigger:** If the agent at max population routinely tolerates 1–2% mass fluctuations without corrective action. Adjust threshold to `0.5 × (num_active / 3000)` for population-relative scaling.

---

## O13 — No Harvest / Dilution Mechanism

**Reality:** Real continuous PBRs dilute the culture at regular intervals (dilution rate D, typically 0.05–0.2 /day) to maintain log-phase growth and prevent self-shading. Without dilution, density-dependent inhibition eventually limits productivity even when nutrients are abundant.

**Why kept:** The current env models a batch culture. Adding a dilution action would change the action space dimension (requiring a fresh checkpoint) and substantially alter the reward landscape — the agent would need to learn a feeding vs. harvesting trade-off. This is a separate experiment from the current batch optimisation goal.

**Fix trigger:** When batch-mode OD control is mastered and the goal shifts to continuous-mode productivity (g/L/day). At that point add a 5th action (dilution rate 0–0.2 /h) and reformulate the reward around volumetric productivity rather than terminal OD.

---

## O14 — Population Cap Creates Density Ceiling Without Dilution

**Reality:** With `max_cells = 7500` super-agents and no harvest, the population hits the array ceiling within ~50–70 hours (episode midpoint). Post-ceiling, growth continues via cell mass accumulation (1.4×10⁸ → 5×10⁸ pg/cell), but the culture is in a non-physiological over-dense state. Real cultures at this density would self-shade severely and undergo stationary-phase entry.

**Why kept:** Removing the ceiling requires either larger arrays (memory) or a dilution action (scope change). Post-ceiling mass growth still produces OD rewards and is mechanistically consistent with self-shading through the Beer-Lambert attenuation model. The `reward_od` high-water-mark and `reward_growth_rate` terms continue to fire during mass accumulation, keeping the reward signal dense.

**Fix trigger:** When implementing the dilution action (O13). At that point the ceiling becomes a soft constraint enforced by the dilution rate rather than a hard array limit.
