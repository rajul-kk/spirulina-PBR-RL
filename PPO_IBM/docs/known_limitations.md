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

## O9 — Reward Has 10+ Overlapping Components *(STALE — reward was rewritten)*

**This entry describes an old reward design that no longer exists.** The current `_compute_reward()` (`environments/genetic_env.py`) has exactly **four** terms: `reward_od` (peaked target-band on instantaneous OD), `reward_biomass` (per-cell growth + flatline penalty), `reward_od_delta` (dense OD-direction signal, skipped on harvest-event steps), and `reward_harvest` (periodic harvest yield, now with a collapse penalty — see Fix #28 below). There is no separate do2/nutrient/pH proxy term, and no `reward_growth_rate` term as such — the reward history (`finalresults.md`) documents a long series of measured, single-change fixes to get from a 10+-term design down to this one, precisely because the overlapping-components problem this entry warns about was real and had to be resolved. Kept here only as a pointer: if you're looking for the reward's current design and rationale, read `_compute_reward()`'s inline comments and `finalresults.md` directly, not this entry.

---

## O10 — gamma Effective Horizon *(STALE VALUE — gamma changed)*

**This entry's `gamma = 0.995` no longer matches the code.** PPO's actual `gamma` is **0.9995** (`training/recurrent_ppo.py`), changed by Fix #13 specifically because 0.995's ~200-step (4h) effective horizon couldn't reach the reward-relevant OD dynamics — see `finalresults.md`'s run history. At 0.9995, the effective horizon is roughly 2000 steps (~40h), long enough to span multiple harvest events (`HARVEST_INTERVAL_STEPS=600`). There is no `tanh(OD/0.05) × 10` terminal bonus in the current reward — that also predates the current design. The general point (short-horizon PPO can't credit delayed OD/harvest tradeoffs) is real and still discussed in this session's reward-cliff findings (`finalresults.md`, TD-MPC2/PPO v28-31 sections), but the specific numbers here are from a prior version of the code.

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

## O13 — No Harvest / Dilution Mechanism *(RESOLVED — harvest exists and is central to the project)*

**This entry is factually wrong about the current environment.** A periodic semi-continuous harvest/dilution mechanism has existed for the entire history of this session's work and is the third action dimension (`stir, light, harvest`). Every `HARVEST_INTERVAL_STEPS` (600 steps = 12h), the interval-averaged harvest action removes a per-cell Bernoulli-sampled fraction of standing biomass (up to `F_MAX=0.5`) and dilutes dissolved-phase state toward fresh medium — see `_compute_reward()`'s harvest handling and the harvest event block in `step()`. `cumulative_harvested_mg` and `time_avg_od` (averaged over the episode's back half) are the two headline curriculum-gate metrics, and essentially all of this project's reward-design history (`finalresults.md`) is about the harvest-vs-OD tradeoff this mechanism creates — including two fixes from this session (#28, #29) directly addressing how the reward and training loop handle harvest-fraction dynamics. Left here as a historical marker only: this entry described a genuinely batch-only, pre-harvest version of the env that predates all of the project's curriculum/RL work.

---

## O14 — Population Cap Creates Density Ceiling Without Dilution *(STALE — dilution exists)*

**Also describes the pre-harvest environment.** With harvest active (see O13), the population is periodically diluted rather than growing to an unbounded ceiling; `max_cells=7500` is a computational cap on the agent-array size (matched across PPO, TD-MPC2, and the diagnostics — see this session's `MAX_CELLS` cost-probe finding, which measured actual active population staying around 2,990-3,000 cells at multiple cap sizes, far below 7500, so the cap is not typically a binding constraint in practice). The "density ceiling without dilution" scenario this entry describes does not apply to the current environment.
