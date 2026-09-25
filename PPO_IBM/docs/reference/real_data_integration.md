# Real Photobioreactor Data Integration Guide

## Overview

This document describes how to use real Spirulina PBR sensor data and harvest measurements to calibrate the simulation environment, improve training, and validate the deployed policy.

Data falls into two categories:
- **High-frequency sensor logs** — continuously recorded during culture runs
- **Low-frequency harvest measurements** — recorded once per batch at or near harvest

---

## 1. Required Data Format

### 1a. Sensor Log (per-run CSV)

One file per culture run. Minimum 1-minute sampling interval; 10–30 second preferred.

```
timestamp,           turbidity_NTU, pH,   temperature_C, DO2_mgL, conductivity_uS_cm, stir_rpm, light_umol_m2_s, co2_flow_mLmin, nut_flow_mLmin
2024-01-15 08:00:00, 142.3,         9.41, 24.8,          7.2,     3210,               120,      800,             5.2,            0.0
2024-01-15 08:01:00, 143.1,         9.43, 24.8,          7.4,     3215,               120,      800,             5.1,            0.0
...
```

**Required columns:**
| Column | Unit | Sensor / Source |
|--------|------|----------------|
| `timestamp` | ISO 8601 datetime | Data logger |
| `turbidity_NTU` | NTU | Turbidity probe |
| `pH` | — | pH electrode |
| `temperature_C` | °C | Thermocouple / PT100 |
| `DO2_mgL` | mg/L | Dissolved O2 probe |
| `conductivity_uS_cm` | µS/cm | Conductivity probe |
| `stir_rpm` | RPM | Stirrer tachometer / controller feedback |
| `light_umol_m2_s` | µmol/m²/s PAR | PAR sensor (or LED controller setpoint) |
| `co2_flow_mLmin` | mL/min | Mass flow controller |
| `nut_flow_mLmin` | mL/min | Pump controller / peristaltic log |

**Optional but valuable:**
| Column | Notes |
|--------|-------|
| `OD_680nm` | If an in-line spectrophotometer is available — direct biomass proxy |
| `chlorophyll_rfu` | Fluorescence (algae-specific sensor like AlgaeOnline) |
| `headspace_CO2_pct` | Off-gas analyser — constrains kLa model directly |
| `headspace_O2_pct` | Off-gas analyser — constrains O2 transfer model |

---

### 1b. Harvest Measurements (per-run CSV or spreadsheet row)

```
run_id, strain_id, date_start,  date_end,    duration_h, dry_weight_gL, biomass_gL, chlorophyll_a_mgL, notes
run_001, SP_WT,    2024-01-15,  2024-01-21,  144,        2.31,          12.4,       48.2,              batch
run_002, SP_WT,    2024-01-22,  2024-01-28,  144,        2.08,          11.1,       44.7,              nutrient_limited
...
```

| Column | Description |
|--------|-------------|
| `run_id` | Matches sensor log filename or a header column |
| `strain_id` | Identifies the Spirulina strain variant |
| `dry_weight_gL` | Gravimetric dry cell weight at harvest (g/L) — ground truth biomass |
| `biomass_gL` | Wet or centrifuged weight if DW unavailable |
| `chlorophyll_a_mgL` | Spectrophotometric Chl-a — used to calibrate rgb_absorbance observation |

---

## 2. Data Preprocessing Pipeline

```
raw_sensor_log.csv
    ↓  resample_to_sim_dt()     — resample to dt=0.01h (36s) bins
    ↓  impute_gaps()            — linear interpolation for gaps < 5 min; flag longer
    ↓  clip_to_obs_bounds()     — flag outliers outside observation_space bounds
    ↓  align_with_harvest()     — join harvest measurements by run_id
    ↓  compute_od_proxy()       — turbidity → OD conversion using harvest DW calibration
    ↓  export: processed_run_001.parquet
```

### Resampling to sim dt

```python
import pandas as pd

def resample_to_sim_dt(df: pd.DataFrame, dt_hours: float = 0.01) -> pd.DataFrame:
    df = df.set_index("timestamp").sort_index()
    rule = f"{int(dt_hours * 3600)}s"
    return df.resample(rule).interpolate("time")
```

### Turbidity → OD calibration

Use harvest dry weight and turbidity at harvest time to fit a linear or power-law calibration:

```python
from scipy.optimize import curve_fit
import numpy as np

def turbidity_to_od(ntu, a, b):
    return a * ntu ** b

# fit a, b from paired (turbidity_at_harvest, dry_weight_gL) across multiple runs
popt, _ = curve_fit(turbidity_to_od, turbidity_values, dw_values / 300.0)
```

This directly calibrates the `self.od = (total_mass_mg / volume_L) / 300.0` formula in the env.

---

## 3. Environment Calibration

### 3a. Bootstrap VecNormalize from Real Data

Replace the 2000-step random-action calibration with statistics computed from real sensor data. This prevents the normalizer from being anchored to unphysical random-action observations.

```python
import numpy as np
import pandas as pd

OBS_COLS = ["turbidity_NTU", "pH", "nutrients_mgL", "temperature_C",
            "conductivity_uS_cm", "rgb_absorbance"]  # must match env observation order

def bootstrap_vec_normalize_from_data(vec_env, sensor_logs: list[pd.DataFrame]):
    all_obs = pd.concat(sensor_logs)[OBS_COLS].dropna().values.astype(np.float32)
    vec_env.obs_rms.mean = all_obs.mean(axis=0)
    vec_env.obs_rms.var  = all_obs.var(axis=0).clip(min=1e-4)
    vec_env.obs_rms.count = float(len(all_obs))
    print(f"VecNormalize bootstrapped from {len(all_obs):,} real observations.")
    print(f"  mean: {vec_env.obs_rms.mean.round(2)}")
    print(f"  std:  {np.sqrt(vec_env.obs_rms.var).round(2)}")
```

Call this after creating `vec_env`, before the first `.reset()`.

---

### 3b. Fit Strain Parameters from Real Growth Curves

Use real OD time-series to fit the key Monod/Droop parameters. This replaces the arbitrary Gaussian priors in `_randomize_strain()` with distributions grounded in your actual strain's biology.

```python
from scipy.optimize import curve_fit
import numpy as np

def monod_growth(t, mu_max, Ks_light, I_mean):
    """Simplified Monod OD growth for parameter fitting."""
    f_I = I_mean / (Ks_light + I_mean)
    return np.exp(mu_max * f_I * t)  # relative OD growth from t=0

def fit_strain_params(od_timeseries: np.ndarray, time_h: np.ndarray,
                      mean_light: float) -> dict:
    """
    od_timeseries: OD values over time (normalized, starting at ~1.0)
    time_h: corresponding time in hours
    mean_light: average PAR during the run (µmol/m²/s)
    """
    def model(t, mu_max, Ks_light):
        return monod_growth(t, mu_max, Ks_light, mean_light)

    popt, pcov = curve_fit(model, time_h, od_timeseries / od_timeseries[0],
                           p0=[0.05, 100.0], bounds=([0.01, 20], [0.2, 500]))
    perr = np.sqrt(np.diag(pcov))
    return {
        "mu_max_mean": popt[0], "mu_max_std": perr[0],
        "Ks_light_mean": popt[1], "Ks_light_std": perr[1],
    }

# Run across all available batches to build prior distributions
results = [fit_strain_params(run["od"], run["time_h"], run["mean_light"])
           for run in processed_runs]

# Summary statistics for _randomize_strain() priors
mu_max_values  = [r["mu_max_mean"]  for r in results]
Ks_light_values = [r["Ks_light_mean"] for r in results]
print(f"mu_max:   mean={np.mean(mu_max_values):.4f}, std={np.std(mu_max_values):.4f}")
print(f"Ks_light: mean={np.mean(Ks_light_values):.1f}, std={np.std(Ks_light_values):.1f}")
```

Update `_randomize_strain()` with the fitted mean/std values from all runs.

---

### 3c. Calibrate kLa from Off-Gas Data (if available)

If headspace CO2/O2 measurements exist, fit kLa as a function of stir_rpm and gas flow:

```python
# kLa from O2 step-change experiment:
# Stop aeration → measure DO2 drop rate (respiration only)
# Resume aeration → measure DO2 recovery rate → kLa from slope

def fit_kLa(do2_recovery: np.ndarray, time_s: np.ndarray,
            do2_sat: float) -> float:
    """Fit kLa from a DO2 re-saturation curve."""
    from scipy.optimize import curve_fit
    def model(t, kLa):
        return do2_sat - (do2_sat - do2_recovery[0]) * np.exp(-kLa * t / 3600)
    popt, _ = curve_fit(model, time_s, do2_recovery, p0=[5.0])
    return popt[0]  # h⁻¹
```

---

### 3d. Calibrate Curriculum OD Thresholds

Replace the hardcoded `ADVANCE_TARGETS` growth factors with real performance percentiles:

```python
def compute_real_growth_factors(processed_runs: list) -> dict:
    factors = []
    for run in processed_runs:
        od_start = run["od"][0]
        od_peak  = run["od"].max()
        factors.append(od_peak / max(od_start, 1e-4))
    factors = np.array(factors)
    return {
        "p25": float(np.percentile(factors, 25)),
        "median": float(np.percentile(factors, 50)),
        "p75": float(np.percentile(factors, 75)),
    }

# Suggested curriculum targets (adjust based on your reactor's typical performance):
# D0 advance threshold ≈ p25 of real runs (easy to achieve)
# D1 advance threshold ≈ median of real runs
# D2 is considered "mastered" when policy beats p75 of real runs
```

---

## 4. Finetuning the Policy on Real Data

### 4a. Behavioral Cloning Pre-training (if operator action logs exist)

If you have logs of how a human operator controlled the reactor (stir_rpm, light, nutrient additions, CO2 flow at each timestep), you can pretrain the policy via imitation learning before PPO.

```python
from imitation.algorithms import bc
from imitation.data.types import Trajectory
import numpy as np

def build_trajectories(sensor_logs, action_logs):
    """Convert paired sensor+action logs into imitation.Trajectory objects."""
    trajectories = []
    for obs_df, act_df in zip(sensor_logs, action_logs):
        obs = obs_df[OBS_COLS].values.astype(np.float32)
        # Normalize actions to [-1, 1] to match env action space
        acts = np.stack([
            np.interp(act_df["stir_rpm"],    [50, 200],  [-1, 1]),
            np.interp(act_df["light_umol"],  [0, 2000],  [-1, 1]),
            np.interp(act_df["nut_flow"],    [0, 100],   [-1, 1]),
            np.interp(act_df["co2_flow"],    [0, 120],   [-1, 1]),
        ], axis=1).astype(np.float32)
        terminals = np.zeros(len(obs) - 1, dtype=bool)
        terminals[-1] = True
        trajectories.append(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))
    return trajectories

trajectories = build_trajectories(processed_sensor_logs, processed_action_logs)
bc_trainer = bc.BC(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    demonstrations=trajectories,
    rng=np.random.default_rng(42),
)
bc_trainer.train(n_epochs=100)

# Copy BC weights to the RecurrentPPO policy before curriculum training:
# model.policy.load_state_dict(bc_trainer.policy.state_dict(), strict=False)
```

### 4b. Fine-tuning a Trained Checkpoint with Real Data

After curriculum training, use the `finetune_recurrent_agent()` function with an env calibrated to real parameters. Key settings for real-data fine-tuning:

- Set `norm_reward=False` in VecNormalize (real data calibrated env has smaller reward variance)
- Use a lower learning rate: `1e-5` (not `1e-4`) for real-data fine-tuning to avoid forgetting curriculum knowledge
- Limit to 200k–500k steps; real-data fine-tuning is for adaptation, not re-training
- Verify that `vec_env.obs_rms` is bootstrapped from real sensor data (Section 3a) before loading

---

## 5. Sim-to-Real Validation

### Shadow Mode Protocol

Before deploying the policy on a real PBR:

1. **Run the policy in shadow mode**: log the policy's recommended actions at each step without applying them. Compare recommended vs. actual operator actions.
2. **Compare predicted OD trajectory**: use the env's `step()` with real sensor observations as input (not simulated) and compare predicted OD against measured OD.
3. **Compute residuals**: large residuals in pH or OD indicate where the env model diverges from reality. Prioritize fixing those physics first.

### Key Validation Metrics

| Metric | Acceptable Residual | If Exceeded |
|--------|--------------------|----|
| OD at 24h | < 20% MAPE | Recalibrate turbidity→OD conversion |
| pH trajectory | < 0.3 units RMSE | Recalibrate carbonate chemistry / CO2 transfer |
| DO2 trajectory | < 1.5 mg/L RMSE | Recalibrate kLa model |
| Temperature | < 1°C RMSE | Check heat balance (light → temp model) |

---

## 6. Observation Space Adaptation for Real PBR

The simulation's 6-dimensional observation space was designed for a synthetic environment where internal state is directly accessible. When connecting to a real reactor, three observations require changes and one must be replaced entirely.

### Current vs. Real-sensor mapping

| idx | Sim obs | Real sensor available? | Issue & fix |
|-----|---------|----------------------|-------------|
| 0 | Turbidity (NTU) | **Yes** — turbidity probe | Scale mismatch: sim uses a synthetic nephelometric formula; real probe reads raw NTU. **Fix**: fit `OD = a × NTU^b` from paired (turbidity_at_harvest, dry_weight) samples and apply the calibration inside `_get_obs()`. |
| 1 | pH | **Yes** — pH electrode | No issue. Direct read. |
| 2 | N-pool (mg N/L) | **No** — requires lab nitrate probe or grab samples | See detailed note below. **Fix**: replace with Δconductivity. |
| 3 | Temperature (°C) | **Yes** — thermocouple / PT100 | No issue. Direct read. |
| 4 | Conductivity (µS/cm) | **Yes** — conductivity probe | Scale mismatch: sim formula gives ~500–3000 (arbitrary); real Zarrouk medium reads 3000–7000 µS/cm. **Fix**: linear calibration `cond_real = a × cond_sim + b` fitted from real data; update obs bounds in `__init__`. |
| 5 | RGB absorbance | **Partial** — fluorescence probe or A680 | Synthetic variable (`turbidity × pigment_health`). No direct sensor equivalent on most rigs. **Recommended fix**: replace with **DO₂ (mg/L)**, which is inline-measurable and directly reflects photosynthetic rate — more informative than a synthetic absorbance. |

### The N-pool problem (obs[2]) — why it can't be directly measured

The simulation exposes `n_pool` (dissolved NO₃⁻, mg N/L) as a direct observation because the sim has omniscient access to its own state. In a real reactor, measuring dissolved nitrate continuously requires either:
- A **YSI or Hach inline nitrate probe** (~£3–8k, requires frequent calibration)
- **Periodic grab samples** sent to a lab (gives a reading every 1–24h, far too slow for RL control at 36-second timesteps)
- An **ion-selective electrode** (noisy, prone to fouling in algae culture)

None of these are practical for a typical academic PBR setup. The policy therefore **cannot observe nitrogen directly** when deployed on a real reactor.

### What to use instead: Δconductivity as N-proxy

Conductivity measures total dissolved ions. As Spirulina consumes NO₃⁻ from the Zarrouk medium, conductivity **drops** — the rate of that drop is a direct proxy for nitrogen uptake.

```
N-depletion signal = conductivity(t) - conductivity(t-1)  [µS/cm per timestep]
```

A negative Δconductivity means nutrients are being consumed. A Δconductivity near zero means either the culture has stopped growing or nutrients have been replenished by a feed addition.

**Implementation change** (when real data is available):

Replace obs[2] in `_get_obs()`:
```python
# Sim (current):
obs[2] = self.n_pool

# Real PBR adaptation:
delta_cond = self.conductivity - self._prev_conductivity
self._prev_conductivity = self.conductivity
obs[2] = float(np.clip(delta_cond, -200.0, 200.0))  # µS/cm per step
```

The LSTM will learn to integrate this signal over time to infer remaining nitrogen — the same way a human operator reads a conductivity trend to estimate nutrient depletion without a nitrate probe.

**Recommended final obs space for real PBR deployment (same 6D shape):**
```
[Turbidity_NTU_calibrated, pH, Δconductivity_uS_per_step, Temp_C, Conductivity_abs_uS, DO2_mgL]
```

This can be provided to the policy from any standard PBR sensor suite (turbidity probe + pH electrode + conductivity probe + thermocouple + DO₂ probe) — all inline, all cheap, no lab work required during operation.

---

## 7. Files Produced by This Pipeline

| File | Description |
|------|-------------|
| `data/processed_run_XXX.parquet` | Resampled, interpolated sensor log |
| `data/harvest_measurements.csv` | Cleaned harvest table |
| `data/strain_params_fitted.json` | Fitted mu_max, Ks_light, etc. per run |
| `data/obs_stats.npz` | mean/var arrays for VecNormalize bootstrap |
| `data/od_calibration.json` | Turbidity→OD power-law fit coefficients |
| `data/kLa_calibration.json` | kLa vs. RPM/flow fit coefficients (optional) |
| `data/bc_trajectories.pkl` | Imitation learning trajectory dataset (optional) |
