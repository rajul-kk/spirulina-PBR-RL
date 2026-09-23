
import os
import sys
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict

# Per-step physics trace, OFF by default. See the gate at the `[EnvDebug]` print for why:
# (full rationale: docs/decision_history.md#--environments-genetic_env-py-9)
ENV_DEBUG = os.environ.get("ENV_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _fclip(x, lo, hi):
    """Scalar clip. numpy dispatch costs ~2.5us per call and this runs ~25x per step;
    NaN propagates identically because both comparisons are False."""
    return float(lo if x < lo else (hi if x > hi else x))

class GeneticPhotobioreactorEnv(gym.Env):
    """Individual-Based Model (IBM) Photobioreactor Environment.
    (full rationale: docs/decision_history.md#--environments-genetic_env-py-14)"""
    metadata = {'render_modes': ['human']}

    def __init__(self, max_cells: int = 300000, initial_cells: int = 3000, difficulty: int = 2,
                 lights_off_hour: Optional[float] = None, lights_on_hour: float = 6.0,
                 enable_fouling: bool = True):
        super(GeneticPhotobioreactorEnv, self).__init__()
        self.difficulty = difficulty
        
        # --- CONFIGURATION ---
        self.max_cells = max_cells
        self.initial_cells = initial_cells
        self.dt = 0.02  # Time step (0.02h); 7200 steps = 144h episode
        self.reactor_depth = 0.30  # 30cm depth
        self.reactor_width = 1.0
        self.volume_L = 20.0  # 20L reactor volume (fully controlled indoor PBR target)
        
        # --- Gas-Phase / Carbonate Configuration (closed 20L PBR) ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-37)
        self.base_air_flow_lpm = 0.30        # Baseline air sparge (L/min)
        self.ambient_co2_frac = 420e-6       # Atmospheric CO2 mol fraction
        self.ambient_o2_frac = 0.209         # Atmospheric O2 mol fraction
        self.buffer_equilibrium_ph = 9.5     # Zarrouk equilibrium at ~200 mM HCO3- (Spirulina medium)
        self.co2_toxicity_Ki_mgL = 30.0      # Mild dissolved-CO2 toxicity onset
        self.co2_toxicity_hill = 2.0         # Hill exponent for toxicity curve

        # --- Automated PID setpoint (Nutrient dosing; CO2 control removed) ---
        # Thresholds rescaled to Zarrouk's much richer baseline (N=410, P=89 mg/L)
        self.N_DOSE_LOW = 150.0      # mg N/L — start dosing below this
        self.N_DOSE_HIGH = 350.0     # mg N/L — stop dosing above this
        self.N_DOSE_RATE = 50.0      # mg/h when active
        self.P_DOSE_LOW = 25.0       # mg P/L — start dosing below this
        self.P_DOSE_HIGH = 70.0      # mg P/L — stop dosing above this

        # --- Semi-continuous cycle: periodic, agent-controlled harvest events ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-57)
        self.F_MAX = 0.5                    # max fraction removed in one harvest event
        self.HARVEST_INTERVAL_STEPS = 600   # 600*0.02h = 12h between harvest decisions (12/episode)

        # Terminal crash penalty, cut -100 -> -10 (2026-09-08): at -100 it was a ~680x
        # reward outlier that destabilized the TD3 critic.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-crash-penalty)
        self.CRASH_PENALTY = 10.0
        # DECLINE_WARN_*, OD_ABOVE_SLOPE/FLOOR, OD_TAIL_COEF removed 2026-09-23: they
        # parameterized the additive shaping terms that PBRS replaced. Their values and the
        # formulas that used them are preserved in
        # docs/decision_history.md#--environments-genetic_env-reward-pre-pbrs-archive

        # --- Potential-based reward shaping (PBRS), 2026-09-23 ---------------------------
        # Replaces the four additive hand-tuned shaping terms (od level, biomass growth rate,
        # od delta, decline warning) with one potential Phi(s) whose discounted difference
        # supplies all dense guidance. Ng/Harada/Russell (1999): F = gamma*Phi(s') - Phi(s)
        # is the only shaping form that provably preserves the optimal policy for ANY Phi,
        # so scale/weight choices here cannot introduce a perverse incentive the way the
        # replaced terms repeatedly did (v48/v50/v52/v53).
        # (full rationale: docs/decision_history.md#--environments-genetic_env-reward-pre-pbrs-archive)
        # MUST match the trainer's discount (legacy/TD3.py GAMMA) or the invariance guarantee
        # is only approximate.
        self.PBRS_GAMMA = 0.9995
        self.PHI_OD_W = 1.0                 # weight on the OD-health component of Phi
        self.PHI_POP_W = 0.5                # weight on the population-health component of Phi
        self.PHI_POP_REF = 400.0            # cells at which pop health saturates toward 1
        self.PHI_SCALE = 3.0                # overall Phi gain; sets dense-signal magnitude
                                            # relative to the task reward (harvest ~0.5/event)
        # OD_DELTA_SIGN_WIDTH removed 2026-09-14: the directional delta term it configured
        # was reverted after causing full-collapse in v52/v53. See the revert note above.
        # Back-half window, shared by time_avg_od and the back-half harvest metric.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-back-half-harvest)
        self.BACK_HALF_STEP = 3600          # half of max_steps (7200)

        # Action: [stir, light, harvest fraction]; CO2 and nutrient dosing stay automated.
        # The harvest fraction only takes effect on periodic harvest-event steps.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-78)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation Space (8 Dims)
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-83)
        _low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _high = [1000.0, 14.0, 5000.0, 40000.0, 50.0, 65535.0, 1000.0, 1.0]
        self._obs_dim = 8 if self.OBS_EXTENDED else 6
        self.observation_space = spaces.Box(
            low=np.array(_low[:self._obs_dim], dtype=np.float32),
            high=np.array(_high[:self._obs_dim], dtype=np.float32),
            dtype=np.float32
        )
        
        # --- VECTORIZED STATE ARRAYS ---
        # Fixed size arrays, managed via active_mask
        self.cells_z = np.zeros(self.max_cells, dtype=np.float32) 
        self.cells_x = np.zeros(self.max_cells, dtype=np.float32) # New: Width Dimension (0-1)
        self.cells_mass = np.zeros(self.max_cells, dtype=np.float32)
        self.cells_quota = np.zeros(self.max_cells, dtype=np.float32)
        # Hysteresis: Photo-acclimation state (tracks recent light exposure)
        self.cells_acclimation = np.zeros(self.max_cells, dtype=np.float32)
        
        # Boolean mask: True = Active Living Cell
        self.active_mask = np.zeros(self.max_cells, dtype=bool)
        # Flocculation state (clumping): 1.0 = single cell, >1.0 = aggregate.
        self.clump_mass = np.ones(self.max_cells, dtype=np.float32)

        # Sensor-lag model: best case is 2 steps at high RPM; low RPM is slower.
        self._sensor_delay_min_steps = 2
        self._sensor_delay_max_steps = 8
        self.action_smooth_coef = 0.003

        # All per-episode state (pools, gas layers, temperature, counters, sensor EMAs) is
        # initialized in reset(); the env must be reset before use, as gym requires.
        self.strain_params = {}
        self.max_steps = 7200  # 7200 steps × 0.02h = 144h episode; 1 rollout = 1 episode

        # --- DAY/NIGHT CYCLE ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-200)
        self.lights_off_hour = lights_off_hour
        self.lights_on_hour  = lights_on_hour
        self.enable_fouling  = enable_fouling

    def set_difficulty(self, difficulty: int) -> None:
        """Set curriculum difficulty level (0=easy, 1=medium, 2=hard)."""
        self.difficulty = int(np.clip(int(difficulty), 0, 2))

    def get_difficulty(self) -> int:
        return int(self.difficulty)
        
    def _randomize_strain(self):
        """Generates a unique 'Strain' of algae for this episode."""
        self.strain_params = {
            # Was N(0.080, 0.015) (~8.7h doubling), above the project's own cited Spirulina range
            # (0.04-0.07 h^-1).
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-218)
            'mu_max':    np.random.normal(0.055, 0.008),   # Arthrospira (Spirulina) platensis: ~12-13h doubling
            'Ks':        np.random.normal(1.0, 0.2),       # NO3-N Ks retained (cross-species similarity)
            'Ks_light':  np.random.normal(100.0, 10.0),
            'Kii':       np.random.normal(2500.0, 250.0),
            'T_opt':     np.random.normal(36.0, 1.0),      # Spirulina optimum ~35-37C
            'Q_min':     0.5,
            'Q_max':     5.0,
            'tau_acclim': np.random.uniform(1.0, 4.0),
            'Ks_P':      float(np.random.uniform(1.0, 4.0)),  # Spirulina P half-saturation (Zarrouk-rich medium)
        }
        self.strain_params['mu_max']   = max(0.025, self.strain_params['mu_max'])  # floor rescaled with the new mean (was 0.04 @ mean 0.08, same 50% ratio)
        self.strain_params['Ks']       = max(0.3, self.strain_params['Ks'])       # floor at 0.3 mg N/L
        self.strain_params['Ks_light'] = max(50.0, self.strain_params['Ks_light'])
        self.strain_params['Kii']      = max(500.0, self.strain_params['Kii'])
        self.strain_params['T_opt']    = np.clip(self.strain_params['T_opt'], 30.0, 40.0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        # This env randomizes via the legacy global np.random, not gym's self.np_random,
        # so super().reset(seed=) alone does not make strain/initial state reproducible.
        if seed is not None:
            np.random.seed(seed)
        self._randomize_strain()
        
        # Initialize Population
        self.num_active = self.initial_cells
        self.active_mask[:] = False
        self.active_mask[:self.num_active] = True
        self._aidx_cache = None
        
        self.cells_z[:self.num_active] = np.random.uniform(0, self.reactor_depth, self.num_active)
        self.cells_x[:self.num_active] = np.random.uniform(0, self.reactor_width, self.num_active)
        # Super-Agent Scaling: 1 Agent = 2,500,000 Cells (~500pg each)
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-254)
        density_ratio = np.clip(self.initial_cells / 15000.0, 0.0, 1.0)
        starting_mass = 1.25e8 - (density_ratio * 0.45e8)
        self.cells_mass[:self.num_active] = np.random.normal(starting_mass, 1e6, self.num_active)
        self.cells_quota[:self.num_active] = np.random.uniform(3.5, 5.0, self.num_active)  # start near Q_max (replete Zarrouk)
        # Initialize acclimation to average light (approx 200)
        self.cells_acclimation[:self.num_active] = np.random.uniform(100.0, 300.0, self.num_active)
        self.clump_mass[:self.num_active] = 1.0 # Start as single cells
        
        fresh = self.FRESH_MEDIUM
        self.ext_nutrients = fresh["ext_nutrients"]
        self.n_pool = fresh["n_pool"]
        self.p_pool = fresh["p_pool"]
        self.bicarbonate = fresh["bicarbonate"]
        self.ph = self.buffer_equilibrium_ph
        self.do2 = self.do2_s = self.do2_b = fresh["do2"]
        self._f_surface_cells = 1.0 / 3.0
        self.temp = float(np.random.uniform(32.0, 38.0))  # per-episode ambient variation (DS18B20)
        self.step_count = 0
        self.time_t = 0.0
        self.fouling_factor = 0.0 # Biofouling accumulation (light path)
        self.turb_fouling_factor = 0.0  # Fix #19: nephelometer window fouling (reads HIGH)
        
        # Reset Hardware Smoothing
        self.current_stir_rpm = 50.0
        self.current_nut_flow = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)
        
        # Reset Advanced Physics
        self.salt = fresh["salt"]
        self.pigment = 1.0
        self.dissolved_co2 = self.co2_s = self.co2_b = fresh["co2"]
        
        self.dosing_integral = 0.0  # reset internal PID tracking each episode
        self.harvest_integral = 0.0  # cumulative volume harvested (L), accumulates per-step via dilution
        self.cumulative_harvested_mg = 0.0  # running total mg harvested across the episode (curriculum metric)
        self.cumulative_harvested_mg_back_half = 0.0
        self._harvest_action_sum = 0.0      # Fix #16 (v19): must reset per episode, or the
        self._harvest_action_count = 0      # first event of a new episode averages stale steps
        self.od_sum_back_half = 0.0         # for time-averaged OD (curriculum metric)
        self.od_count_back_half = 0
        self.reward_term_sums = {"harvest": 0.0, "shaping": 0.0, "potential": 0.0}
        self.I_surface = 0.0        # reset BH1750 source signal
        # --- Sim-to-Real Sensor Drift & Lag (D1+) ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-303)
        if self.difficulty >= 1:
            self._sensor_drift_mult = np.random.uniform(0.95, 1.05, size=8)
            self._sensor_drift_mult[1] = 1.0   # pH: additive bias only, no multiplicative drift
            # Fix #18: the EMA is derived from channel 0, so it inherits that channel's drift; an
            # independent draw would let the policy average out a drift real hardware can't.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-309)
            self._sensor_drift_mult[6] = self._sensor_drift_mult[0]
            # episode_phase is the controller's own clock, not a sensor: no drift, no jitter.
            self._sensor_drift_mult[7] = 1.0
            self._ph_bias = float(np.random.uniform(-0.1, 0.1))  # SEN0161 ±0.1 pH additive offset
            self._ph_obs_ema = self.ph
            self._temp_obs_ema = self.temp
        else:
            self._sensor_drift_mult = np.ones(8, dtype=np.float32)
            self._ph_bias = 0.0
            self._ph_obs_ema = self.ph
            self._temp_obs_ema = self.temp
            
        # --- Initialize derived state (prevents stale-value NaN on episode 2+) ---
        init_total_mass = np.sum(self.cells_mass[:self.num_active])
        self.last_mass = init_total_mass
        self.od = (init_total_mass * 1e-9 / self.volume_L) / 300.0
        # B6: initialize sensor state here so _get_obs() never reads stale episode values.
        # Same formula as every step, so the first observation isn't off by the bicarbonate
        # term (it was: ~12,000 vs ~22,600 µS/cm, a jump on step 1).
        self.conductivity = self._conductivity()
        self.rgb_absorbance = 0.0
        self.last_hourly_od = float(self.od)
        # Fix #18 (v21): reset the turbidity EMA per episode — carrying it across resets would
        # seed a fresh, sparse culture with the previous episode's biomass reading.
        self._turb_ema = None
        self.debug_mu = 0.0
        self.debug_stress = 0.0
        self.debug_f_I = 1.0
        self.debug_f_Q = 1.0
        self.debug_shock = 0.0
        self.debug_clump = 1.0
        self.membrane_integrity = 1.0  # 1.0 = pristine membranes, 0.0 = fully fatigued
        self.max_historical_od = float(self.od)
        # Seed the PBRS potential from the true initial state, so the first scored step
        # measures a real transition rather than differencing against itself.
        self._phi_prev = self._potential()

        return self._get_obs(), {}

    def _aidx(self):
        """Cached integer indices of active cells.
        (full rationale: docs/decision_history.md#--environments-genetic_env-active-index-cache)"""
        c = self._aidx_cache
        if c is None or len(c) != self.num_active:
            c = np.where(self.active_mask)[0]
            self._aidx_cache = c
        return c

    def _get_obs(self):
        # Calculate stats only if cells exist
        if self.num_active > 0:
            total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9
            self.od = (total_mass_mg / self.volume_L) / 300.0  # Volume-normalised OD

            # conductivity, rgb_absorbance, last_hourly_od are guaranteed by reset()

            # Turbidity sensor: realistic nephelometric model
            avg_clump = np.mean(self.clump_mass[self.active_mask])

            # Mie scattering: fixed mass total cross-section
            clump_scatter = avg_clump ** (-1.0/3.0)  # clumps reduce total surface area

            # Pigment effect (weakened for near-IR nephelometer)
            # Real turbidity sensors use 860nm where chlorophyll doesn't absorb strongly
            pigment_contrast = 0.7 + 0.3 * self.pigment  # Compressed range 0.7-1.0

            # Multiple scattering saturation (onset at OD ~ 5-10)
            # High-density cultures show non-linear turbidity response
            saturation_factor = 1.0 / (1.0 + 0.05 * self.od)

            # Base turbidity calculation
            turbidity_base = self.od * pigment_contrast * clump_scatter * saturation_factor
            # Fix #19 (v22): window biofilm scatters extra light into the detector, so the reading
            # drifts high while true biomass is unchanged.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-380)
            turbidity_base *= (1.0 + float(self.turb_fouling_factor))

            # Bubble/flow-induced noise (RPM-dependent)
            # Real sensors show high-frequency noise from bubbles and turbulent eddies
            rpm = float(self.current_stir_rpm)
            flow_noise = 1.0 + 0.03 * (rpm / 200.0) * np.random.normal(0, 1)

            # Convert to raw NTU units (0-5000 range for sim-to-real transfer)
            # Add base sensor noise floor (Difficulty scaled)
            noise_scale = 0.05
            # SEN0189 saturates at 1000 NTU — clip matches hardware ceiling
            turbidity_obs = _fclip((turbidity_base * 1000.0 * flow_noise) + np.random.normal(0, noise_scale), 0.0, 1000.0)
            self.turbidity_obs = turbidity_obs  # Store for debug logging
        else:
            self.od = 0.0
            self.conductivity = 0.0
            self.rgb_absorbance = 0.0
            turbidity_obs = 0.0
            self.turbidity_obs = turbidity_obs

        bh1750_lux = _fclip(self.I_surface * 80.0 + np.random.normal(0.0, 500.0), 0.0, 65535.0)

        # Fix #18 (v21): long-window (~600-step) EMA of turbidity, since the raw channel carries
        # stir-dependent multiplicative noise.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-406)
        _EMA_ALPHA = 2.0 / (600.0 + 1.0)
        if self._turb_ema is None:
            self._turb_ema = float(turbidity_obs)
        else:
            self._turb_ema += _EMA_ALPHA * (float(turbidity_obs) - self._turb_ema)

        base_obs = np.array([
            turbidity_obs,
            self.ph,
            np.clip(self.harvest_integral, 0.0, 5000.0),  # pump counter (L cumulative)
            np.clip(self.conductivity, 0.0, 40000.0),  # DFR0300 measurable ceiling
            self.temp,
            bh1750_lux,
            np.clip(self._turb_ema, 0.0, 1000.0),   # Fix #18: smoothed turbidity
            # Fix #21 (v22): HARVEST-CYCLE phase, replacing the episode phase used in v21.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-425)
            (np.clip(self.step_count / max(self.max_steps, 1), 0.0, 1.0)
             if self.USE_EPISODE_PHASE else
             np.clip((self.step_count % max(self.HARVEST_INTERVAL_STEPS, 1))
                     / max(self.HARVEST_INTERVAL_STEPS, 1), 0.0, 1.0)),
        ], dtype=np.float32)

        # Truncate to the configured width (6 by default, 8 with OBS_EXTENDED). Channels 6-7
        # are still computed so the EMA stays warm.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-447)
        base_obs = base_obs[:self._obs_dim]

        # Stochastic Sensor Noise: ±1% jitter at D0, ±2% at D1+
        jitter_mag = 0.02 if self.difficulty >= 1 else 0.01
        jitter = np.random.uniform(1.0 - jitter_mag, 1.0 + jitter_mag, size=(self._obs_dim,))

        # RPM-coupled EMA lag on pH and temperature (D1+)
        if self.difficulty >= 1:
            rpm = _fclip(self.current_stir_rpm, 50.0, 200.0)
            mix_quality = (rpm - 50.0) / 150.0
            lag_span = self._sensor_delay_max_steps - self._sensor_delay_min_steps
            lag_steps = int(np.clip(
                round(self._sensor_delay_max_steps - lag_span * mix_quality),
                self._sensor_delay_min_steps, self._sensor_delay_max_steps
            ))
            alpha = 2.0 / (lag_steps + 1.0)
            self._ph_obs_ema   = (1.0 - alpha) * self._ph_obs_ema   + alpha * base_obs[1]
            self._temp_obs_ema = (1.0 - alpha) * self._temp_obs_ema + alpha * base_obs[4]
            base_obs[1] = self._ph_obs_ema
            base_obs[4] = self._temp_obs_ema
            # Additive pH bias (SEN0161 ±0.1 pH calibration offset — per-episode constant)
            base_obs[1] = _fclip(base_obs[1] + self._ph_bias, 0.0, 14.0)

        noisy_obs = base_obs * jitter * self._sensor_drift_mult[:self._obs_dim]
        noisy_obs[3] = _fclip(noisy_obs[3], 0.0, 40000.0)   # DFR0300 hard ceiling (post-jitter)
        noisy_obs[5] = _fclip(noisy_obs[5], 0.0, 65535.0)  # BH1750 hard ADC ceiling (16-bit)
        if self._obs_dim > 7:
            # Fix #18: phase is the controller's own clock — restore it exactly, since the
            # jitter multiply above would otherwise corrupt a quantity known perfectly.
            noisy_obs[7] = base_obs[7]
        return noisy_obs.astype(np.float32)

    def get_privileged_state(self) -> np.ndarray:
        """4D privileged vector — sim-only, never exposed at deployment.
        (full rationale: docs/decision_history.md#--environments-genetic_env-py-386)"""
        mean_fQ = float(np.mean(np.maximum(
            0.0, 1.0 - 0.5 / (self.cells_quota[self.active_mask] + 1e-6)
        ))) if self.num_active > 0 else 0.0
        return np.array([
            getattr(self, 'dissolved_co2', 2.0),            # hidden DIC state
            mean_fQ,                                          # mean Droop quota
            self.strain_params.get('mu_max', 0.05),          # strain growth rate
            self.strain_params.get('Ks_light', 100.0) / 200.0,  # normalised
        ], dtype=np.float32)

    # Per-event harvest target for reward_harvest (mg per event; events fire every
    # HARVEST_INTERVAL_STEPS, 12 per episode), from a harvest-fraction grid sweep.
    # (full rationale: docs/decision_history.md#--environments-genetic_env-py-496)
    TARGET_MG_PER_EVENT = 12.32

    # Target standing OD: the peak of the PBRS OD-health term and the reference for the
    # harvest-collapse penalty.
    OD_TARGET = 0.012

    # Fresh Zarrouk medium: what reset() fills the tank with AND what harvest dilution refills
    # it with. One definition, so the two can't drift apart again (salt did: 2500 vs 1000).
    FRESH_MEDIUM = {
        "ext_nutrients": 300.0,  # mg/L — mineral salts (MgSO4, CaCl2, trace metals)
        "n_pool": 410.0,         # mg N/L — NaNO3 2.5 g/L
        "p_pool": 89.0,          # mg P/L — K2HPO4 0.5 g/L
        "bicarbonate": 200.0,    # mM — NaHCO3 16.8 g/L
        "salt": 2500.0,          # mg/L — NaCl + K2SO4 + trace salts ionic background
        "do2": 7.0,              # mg/L — air-equilibrated
        "co2": 6.2,              # mg/L — CO2(aq) at pH 9.5 with 200 mM HCO3-
    }

    # Light-path biofouling coefficient. NOTE: 0.0002 was calibrated for lab OD600 units,
    # ~250x larger than this sim's od, so the term is nearly inert.
    # (full rationale: docs/decision_history.md#--environments-genetic_env-py-508)
    LIGHT_FOULING_COEF = 0.0002

    # ── Switchable realism / observation options (v23) ────────────────────────────────────
    # (full rationale: docs/decision_history.md#--environments-genetic_env-py-518)
    TURB_FOULING_COEF = 0.0
    HARVEST_PUMP_ERROR = 0.0
    USE_EPISODE_PHASE = True

    # OBS_EXTENDED: False -> 6 channels (real hardware sensors; the default). True -> 8,
    # adding the turbidity EMA and phase. Changing it invalidates saved models.
    # (full rationale: docs/decision_history.md#--environments-genetic_env-py-544)
    OBS_EXTENDED = False

    def _potential(self):
        """Phi(s) for PBRS. A pure, bounded function of the CURRENT state only.

        Two components, each in [0, 1] before weighting:
          - OD health: peaks at 1.0 exactly at OD_TARGET, falls off both ways. Below target
            the culture is under-productive; above it, overgrown (light limitation, crash
            risk). Uses a log-ratio distance so the falloff is symmetric in relative terms
            and never flattens to zero gradient -- the dead zone that cost this project four
            runs was a bounded penalty reaching exactly zero slope, which cannot happen here.
          - Population health: saturating in num_active, so extinction is a deep hole and
            large populations plateau rather than paying unbounded reward for hoarding
            biomass the agent never harvests.

        Must stay a function of state alone: no deltas, no action, no episode phase. Adding
        any transition-dependent quantity here silently voids the policy-invariance guarantee.
        """
        od_x = max(float(self.od) / self.OD_TARGET, 1e-6)
        # log-ratio distance from target, squashed; 1.0 at target, ->0 far either side
        phi_od = float(np.exp(-(np.log(od_x) ** 2) / 2.0))

        phi_pop = float(np.tanh(self.num_active / self.PHI_POP_REF))

        phi = self.PHI_OD_W * phi_od + self.PHI_POP_W * phi_pop
        return self.PHI_SCALE * phi / (self.PHI_OD_W + self.PHI_POP_W)

    def _compute_reward(self, harvested_this_step_mg, is_harvest_event):
        """Task reward (harvest yield, harvest-collapse penalty) plus PBRS shaping.
        (full rationale: docs/decision_history.md#--environments-genetic_env-py-418)
        The additive shaping terms this replaced are recorded in
        docs/decision_history.md#--environments-genetic_env-reward-pre-pbrs-archive."""
        # Periodic harvest yield: nonzero only on harvest-event steps.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-706)
        reward_harvest = 0.5 * float(np.tanh(harvested_this_step_mg / self.TARGET_MG_PER_EVENT))

        # Fix #28: harvest-event OD-collapse penalty. reward_harvest saturates just past the
        # optimal harvest fraction, so over-harvesting that crashes OD would otherwise be free.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-740)
        if is_harvest_event:
            OD_SAFE_FLOOR = 0.4
            post_harvest_ratio = self.od / self.OD_TARGET
            if post_harvest_ratio < OD_SAFE_FLOOR:
                reward_harvest -= 0.3 * float(OD_SAFE_FLOOR - post_harvest_ratio)

        # PBRS shaping, the ONLY dense guidance term: F = gamma*Phi(s') - Phi(s). self.* is
        # already post-transition here, and _phi_prev holds Phi of the previous state (seeded
        # by reset(), re-seeded after stitched starts). The growth incentive falls out of the
        # telescoping; adding any separate rate term would void the policy-invariance guarantee.
        phi_now = self._potential()
        reward_shaping = self.PBRS_GAMMA * phi_now - self._phi_prev
        self._phi_prev = phi_now

        reward = reward_harvest + reward_shaping

        # Episode-accumulated breakdown for the info dict. "potential" logs the running Phi so
        # a collapse can be read directly off the state value.
        self.reward_term_sums["harvest"] += reward_harvest
        self.reward_term_sums["shaping"] += reward_shaping
        self.reward_term_sums["potential"] = phi_now

        if not np.isfinite(reward):
            reward = -10.0  # Punishment for breaking physics
        return float(reward)

    def _update_episode_stats(self, shock_factor):
        """Curriculum metrics and debug trackers. None of these feed the reward."""
        # Curriculum metric: time-averaged OD over the back half of the episode (steps
        # 3600-7200), a steady-state proxy that a brief early spike can't game.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-564)
        if self.step_count >= self.BACK_HALF_STEP:
            self.od_sum_back_half += self.od
            self.od_count_back_half += 1

        self.debug_shock = float(np.mean(shock_factor) if self.num_active > 0 else 1.0)
        self.debug_clump = float(np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0)
        if self.od > self.max_historical_od:
            self.max_historical_od = self.od
        # Next step's delta_mass_mg is measured against this.
        self.last_mass = np.sum(self.cells_mass[self.active_mask]) if self.num_active > 0 else 0

    def step(self, action):
        stir_rpm, I_surface, nut_flow = self._apply_actions(action)
        I_surface, nut_flow = self._apply_light_schedule(I_surface, nut_flow)
        self._update_temperature(I_surface, stir_rpm)
        self._update_fouling(stir_rpm)

        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-953)
        dt_sec = self.dt * 3600
        self.time_t += self.dt
        mix_intensity = stir_rpm / 200.0  # 0 to 1

        self._update_flocculation(stir_rpm)
        self._move_cells(mix_intensity, dt_sec)
        total_uptake_mg, shock_factor = self._update_biology(I_surface, stir_rpm, mix_intensity, nut_flow)

        # Transfer coefficients and the biomass change are taken on the PRE-harvest culture;
        # the gas-layer balances below then run on the post-harvest tank.
        k_La, o2_frac, co2_frac = self._gas_transfer(stir_rpm)
        delta_mass_mg = self._biomass_change()
        harvested_this_step_mg, is_harvest_event = self._apply_harvest()

        # Recompute standing mass/OD post-dilution — this is what the tank actually holds
        total_mass_mg = np.sum(self.cells_mass[self._aidx()]) * 1e-9
        self.od = (total_mass_mg / self.volume_L) / 300.0

        self._update_gas_and_carbonate(k_La, o2_frac, co2_frac, mix_intensity, delta_mass_mg)
        self._update_pigment_and_salt(I_surface, delta_mass_mg, total_uptake_mg)
        self._update_sensors()

        self._update_episode_stats(shock_factor)
        reward = self._compute_reward(harvested_this_step_mg, is_harvest_event)

        # Always increment step_count so Monitor reports correct episode length on crash
        self.step_count += 1
        # Extinction check: population OR total biomass, since a few 'zombie' cells can hover
        # above the starvation threshold with near-zero total mass.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1723)
        if self.num_active < 10 or total_mass_mg < 1.0:
            # CRASH_PENALTY history: -1000 -> -100 -> -10, each cut because the outlier
            # destabilized learning.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1729)
            reward -= self.CRASH_PENALTY
            terminated, truncated = True, False
        else:
            # Time limit is truncation, not termination: the culture is still alive, so value
            # learners must bootstrap through it. Reporting it as terminal also leaks a
            # policy-dependent gamma*Phi(s_T) bonus under PBRS (Ng et al. assume Phi(terminal)=0).
            # No terminal bonus in semi-continuous mode — harvest yield (cumulative_harvested_mg)
            # accumulates continuously via reward_harvest each step (see _compute_reward).
            terminated, truncated = False, self.step_count >= self.max_steps
        done = terminated or truncated

        # Per-step debug trace, gated behind ENV_DEBUG (default off) because it once made up
        # ~43% of every training log.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1741)
        if ENV_DEBUG and ((self.step_count % 500 == 0) or done):
            mean_x = np.mean(self.cells_x[self.active_mask]) if self.num_active > 0 else 0.0
            msg = (f"[EnvDebug] Step: {self.step_count}, Active: {self.num_active}, Mass: {total_mass_mg:.2f}, "
                   f"OD: {self.od:.4f}, Turb: {self.turbidity_obs:.4f}, pH: {self.ph:.2f}, "
                   f"Shock: {self.debug_shock:.2f}, Clump: {self.debug_clump:.2f}, MeanX: {mean_x:.2f}, "
                   f"RGB: {getattr(self, 'rgb_ratio', 0.0):.2f}, Rew: {reward:.3f}, Done: {done}")
            if 'tqdm' in sys.modules:
                from tqdm import tqdm
                tqdm.write(msg)
            else:
                print(msg)

        return self._get_obs(), float(reward), terminated, truncated, {
            "pop": self.num_active,
            "fouling": self.fouling_factor,
            "peak_od": float(self.max_historical_od),
            "od": float(self.od),
            "kLa_h-1": float(self.kLa),
            "dissolved_co2_mgL": float(self.dissolved_co2),
            "cumulative_harvested_mg": float(self.cumulative_harvested_mg),
            "harvested_mg_back_half": float(self.cumulative_harvested_mg_back_half),
            "time_avg_od": float(self.od_sum_back_half / max(self.od_count_back_half, 1)),
            "start_mode": getattr(self, 'episode_start_mode', 'low'),
            "reward_term_sums": dict(self.reward_term_sums),
        }



    def _remove_cells(self, idx):
        """Deactivate cells (lysis, starvation, harvest) and reset their slots."""
        self.active_mask[idx] = False
        self._aidx_cache = None
        self.num_active -= len(idx)
        self.cells_mass[idx] = 0.0
        self.cells_quota[idx] = 0.0
        self.cells_acclimation[idx] = 0.0
        self.clump_mass[idx] = 1.0

    def _apply_actions(self, action):
        """Decode the action, run the nutrient PID and actuator smoothing. Returns the delivered (stir_rpm, I_surface, nut_flow)."""
        # --- Safety Checks ---
        # 0. Check for invalid actions
        if np.any(np.isnan(action)):
            # If action is NaN (e.g. model output is corrupted), default to safe values
            # (no stir, no light, no dilution — safest static fallback)
            action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

        action_vec = np.asarray(action, dtype=np.float32)
        self.prev_action = action_vec.copy()

        stir_act, light_act, harvest_act = action

        # 1. Decode Target Actions (Stir, Light — every step; Harvest fraction — only
        # decoded/applied on periodic harvest-event steps, see HARVEST_INTERVAL_STEPS)
        target_stir_rpm  = np.interp(stir_act,    [-1, 1], [50, 200])
        target_I_surface = np.interp(light_act,   [-1, 1], [0, 2000])
        harvest_frac_action = float(np.interp(harvest_act, [-1, 1], [0.0, self.F_MAX]))

        # Fix #16 (v19): INTERVAL-AVERAGED HARVEST — the credit-assignment fix.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-806)
        self._harvest_action_sum += harvest_frac_action
        self._harvest_action_count += 1

        # 1b. Automated PID Controller (Nutrient N/P threshold control only)
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-838)
        if self.n_pool < self.N_DOSE_LOW or self.p_pool < self.P_DOSE_LOW:
            target_nut_flow = self.N_DOSE_RATE
        elif self.n_pool > self.N_DOSE_HIGH and self.p_pool > self.P_DOSE_HIGH:
            target_nut_flow = 0.0
        else:
            target_nut_flow = self.current_nut_flow  # hold

        # 2. Hardware Smoothing (EMA)
        # EMA alphas doubled for dt=0.02h to preserve the same physical lag time constants
        alpha_nut = 0.12   # same ~0.17h lag as 0.06 at dt=0.01h
        self.current_stir_rpm = (0.90 * self.current_stir_rpm) + (0.10 * target_stir_rpm)
        self.current_nut_flow = (1.0 - alpha_nut) * self.current_nut_flow + alpha_nut * target_nut_flow

        stir_rpm    = self.current_stir_rpm
        I_surface   = target_I_surface  # Light changes instantly
        nut_flow    = self.current_nut_flow

        # Actuator delivery noise (D1+): ±5% simulates pump calibration drift and motor imprecision
        if self.difficulty >= 1:
            stir_rpm *= float(np.random.uniform(0.95, 1.05))
            nut_flow *= float(np.random.uniform(0.95, 1.05))

        # Accumulate internal PID dosing tracker (not exposed to obs)
        self.dosing_integral += self.current_nut_flow * 0.79 * self.dt  # 79% N fraction (Zarrouk ratio)
        return stir_rpm, I_surface, nut_flow

    def _apply_light_schedule(self, I_surface, nut_flow):
        """Day/night cycle and early-episode grace limits on light and nutrient flow."""
        # --- Day/Night Cycle (enforced before grace period) ---
        if self.lights_off_hour is not None:
            current_hour = (self.step_count * self.dt) % 24.0
            # Night wraps through midnight (e.g. off=20, on=6 -> dark 20:00-06:00)
            if self.lights_off_hour > self.lights_on_hour:
                is_night = (current_hour >= self.lights_off_hour) or (current_hour < self.lights_on_hour)
            else:
                is_night = (self.lights_off_hour <= current_hour < self.lights_on_hour)
            if is_night:
                I_surface = 0.0  # Hard dark period -- agent cannot override
            self.is_night = is_night
        else:
            self.is_night = False

        # --- Grace Period (Training Wheels) ---
        if self.step_count < 1200:
            grace_factor = self.step_count / 1200.0
            max_light = 500.0 + (1500.0 * grace_factor)
            I_surface = min(I_surface, max_light)
            max_nuts = 20.0 + (80.0 * grace_factor) # Ramp 20→100 mg/h: matches real consumption at 3k-12k cells
            nut_flow = min(nut_flow, max_nuts)

        # Store effective PAR after day/night and grace period — BH1750 reads actual LED delivery
        self.I_surface = float(I_surface)
        return I_surface, nut_flow

    def _update_temperature(self, I_surface, stir_rpm):
        # --- Temperature Inertia ---
        ambient_temp = 25.0
        # Physics scale: D0=50%, D1=75%, D2=100%
        diff_level = self.difficulty
        phys_scale = 0.50 if diff_level == 0 else (0.75 if diff_level == 1 else 1.0)
        
        # Light adds heat: max 2000 umol/m2/s ~ 2.0 degrees C / hour
        heat_from_light = (I_surface * 0.001) * self.dt * phys_scale
        # Cooling to ambient: Rate of 0.1 per hour
        cooling = 0.1 * (self.temp - ambient_temp) * self.dt * phys_scale
        self.temp += heat_from_light - cooling
        # Viscous/Ohmic heating from impeller (scales with RPM^3, max ~0.5°C at 200 RPM)
        stir_heat = (stir_rpm / 200.0) ** 3 * 0.5 * self.dt * phys_scale
        self.temp += stir_heat
        
        self.temp = np.clip(self.temp, 15.0, 45.0)

    def _update_fouling(self, stir_rpm):
        # --- Biofouling Accumulation ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-913)
        if self.enable_fouling:
            # LIGHT_FOULING_COEF is a class attribute so feasibility probes can override it
            # without editing physics.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-917)
            fouling_rate = max(0.0, 1.0 - stir_rpm / 200.0) * self.od * self.LIGHT_FOULING_COEF
            self.fouling_factor += fouling_rate * self.dt
            self.fouling_factor = _fclip(self.fouling_factor, 0.0, 0.5)

            # Fix #19 (v22): NEPHELOMETER WINDOW FOULING (D1+).
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-924)
            if self.difficulty >= 1 and self.TURB_FOULING_COEF > 0.0:
                turb_foul_rate = max(0.0, 1.0 - stir_rpm / 200.0) * self.od * self.TURB_FOULING_COEF
                self.turb_fouling_factor += turb_foul_rate * self.dt
                self.turb_fouling_factor = _fclip(self.turb_fouling_factor, 0.0, 0.25)

    def _update_flocculation(self, stir_rpm):
        # --- FLOCCULATION PHYSICS (Mean-Field) ---
        if self.num_active > 0:
            # 1. Aggregation (Sticking) - Orthokinetic + Perikinetic
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-966)
            od_approx = self.od  # Use internal OD, not obs[0] (now turbidity!)
            
            # --- New Physics: 
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-972)
            rpm_factor = max(0.1, 1.0 - (stir_rpm / 250.0))
            prob_stick = (od_approx * 1e-3) * rpm_factor
            
            # Apply sticking to active cells only — O(num_active) not O(max_cells)
            # Generating 300k random numbers every step with 3k active cells was 25% of step time.
            _active_idx = self._aidx()
            _stick = np.random.uniform(0, 1, self.num_active) < prob_stick
            self.clump_mass[_active_idx[_stick]] += 1.0
            
            # 2. Breakup (Shear + Brownian)
            # Mechanical shear breakup (onset at 80 RPM)
            clump_shear = max(0.0, (stir_rpm - 80.0) / 120.0) ** 2

            # Brownian/diffusive breakup (always active, weak)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-991)
            brownian_breakup = 0.005 * (self.clump_mass[self._aidx()] - 1.0) ** 0.5

            # Total breakup rate: shear + Brownian diffusion
            breakup_rate = 0.5 * clump_shear * (self.clump_mass[self._aidx()] ** 0.5) + brownian_breakup

            # Apply breakup to all active clumps
            self.clump_mass[self._aidx()] -= breakup_rate * self.dt

            # Physical Lower Bound: 1.0 (Single Cell)
            self.clump_mass[self._aidx()] = np.maximum(self.clump_mass[self._aidx()], 1.0)

    def _move_cells(self, mix_intensity, dt_sec):
        """Advect/diffuse cells (turbulent mixing, or sedimentation at low RPM), then reflect at the walls."""
        if self.num_active > 0 and mix_intensity > 0.01:
            # --- 2D Kinematic Turbulence (Airlift / Convection Loop) ---
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1007)
            
            x_pos = self.cells_x[self._aidx()]
            z_pos = self.cells_z[self._aidx()]
            
            # 1. Vertical Velocity (Vz)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1015)
            v_max_z = 0.05 * mix_intensity   # 0.05 m/s at max RPM — realistic for 30L flat-panel airlift
            # Damping at top/bottom walls (z close to 0 or D)
            v_macro_z = v_max_z * -np.cos(2 * np.pi * (x_pos - 0.5)) 
            
            # 2. Horizontal Velocity (Vx)
            v_max_x = v_max_z * 5.0 # Aspect ratio scaling (Width >> Depth)
            v_macro_x = v_max_x * (x_pos - 0.5) * np.cos(np.pi * z_pos / self.reactor_depth)
            
            # 3. Turbulence (Random Perturbations)
            # Perlin-like noise
            indices = self._aidx()
            turb_freq = 5.0
            turb_phase = 10.0 * self.time_t
            
            v_turb_x = 0.002 * mix_intensity * np.sin(turb_freq * z_pos * 100 - turb_phase + indices)
            v_turb_z = 0.002 * mix_intensity * np.cos(turb_freq * x_pos * 10 - turb_phase + indices)
            
            # 4. Sinking & Diffusion
            r_eff = self.clump_mass[self._aidx()] ** (1.0/3.0)
            v_sink = 0.001 * (self.clump_mass[self._aidx()] ** (2.0/3.0))
            
            noise_x = np.random.normal(0, 1, self.num_active)
            noise_z = np.random.normal(0, 1, self.num_active)
            
            # Brownian displacement over the step: sqrt(2*D*dt), D = 1e-7 m^2/s scaled by clump size.
            diff_x = (np.sqrt(2 * 1e-7 * dt_sec) / r_eff) * noise_x
            diff_z = (np.sqrt(2 * 1e-7 * dt_sec) / r_eff) * noise_z

            # Integrate
            dz = (v_macro_z + v_turb_z - v_sink) * dt_sec + diff_z
            dx = (v_macro_x + v_turb_x) * dt_sec + diff_x
            
            self.cells_z[self._aidx()] += dz
            self.cells_x[self._aidx()] += dx
            
        else:
            # Low mixing -> Sedimentation
            sink_speed = 0.005 * (self.clump_mass[self._aidx()] ** (2.0/3.0))
            r_eff = self.clump_mass[self._aidx()] ** (1.0/3.0)
            
            noise = np.random.normal(0, 1, self.num_active)
            dz = -sink_speed * self.dt + ((np.sqrt(2 * 1e-7 * dt_sec)/r_eff) * noise)
            
            # X Diffusion only
            noise_x = np.random.normal(0, 1, self.num_active)
            dx = (np.sqrt(2 * 1e-7 * dt_sec)/r_eff) * noise_x
            
            self.cells_z[self._aidx()] += dz
            self.cells_x[self._aidx()] += dx
        
        # Reflecting walls. One step can carry a cell several reactor widths (velocity x
        # dt_sec, up to ~9 m in a 1 m tank), so fold with a triangle wave. Reflecting once and
        # then clipping sent every large overshoot to the wall, which piled all cells at
        # x=0, z=0 within ~50 steps and collapsed the depth-dependent light and gas-layer models.
        self.cells_z = self._reflect(self.cells_z, self.reactor_depth)
        self.cells_x = self._reflect(self.cells_x, self.reactor_width)

    @staticmethod
    def _reflect(pos, length):
        """Map positions into [0, length] as if bounced off both walls any number of times."""
        pos = np.mod(pos, 2.0 * length)
        return np.where(pos > length, 2.0 * length - pos, pos).astype(np.float32)

    def _update_biology(self, I_surface, stir_rpm, mix_intensity, nut_flow):
        """Light field, growth, lysis, nutrient uptake and division. Returns (total_uptake_mg, shock_factor)."""
        # --- BIOLOGY ---
        
        # 1. Shear Stress (RPM > 400)
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1083)

        total_uptake_mg = 0.0
        if self.num_active > 0:
            params = self.strain_params
            
            # 1. Spectral Light Field (RGB Physics)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1093)
            
            # Split Spectrum (Grow Light logic: High Red/Blue, Low Green)
            I_s_red = I_surface * 0.4
            I_s_blue = I_surface * 0.4
            I_s_green = I_surface * 0.2
            
            # Optical Density (Biomass)
            total_mass_mg = np.sum(self.cells_mass[self._aidx()]) * 1e-9
            current_od = (total_mass_mg / self.volume_L) / 300.0
            
            # Attenuation Coefficients (k)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1106)
            
            # Bubble Scattering (0.004 * RPM)
            k_scatter = stir_rpm * 0.004
            
            k_red = 0.5 + (3.5 * current_od) + k_scatter
            # Blue: Absorbed by pigments but penetrates better (3x better than Red?)
            k_blue = 0.2 + (1.5 * current_od) + k_scatter
            # Green: Reflected/Transmitted (Deep penetration)
            k_green = 0.05 + (0.5 * current_od) + k_scatter
            
            # ── Turbulent Flash-Light Effect (Biologically Accurate) ──────────
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1119)
            static_z = self.cells_z[self._aidx()]
            # 2-layer gas: surface = top 10cm (z < 0.10m), bulk = bottom 20cm
            surface_cell_mask = static_z < 0.10
            self._f_surface_cells = float(np.mean(surface_cell_mask))
            cells_local_do2 = np.where(surface_cell_mask, self.do2_s, self.do2_b).astype(np.float32)
            cells_local_co2 = np.where(surface_cell_mask, self.co2_s, self.co2_b).astype(np.float32)

            turbulent_z = np.random.uniform(0.0, self.reactor_depth, size=static_z.shape)
            # Turbulent fraction scales with mix_intensity (0 → rest, 0.95 → 500 RPM max mixing)
            turb_fraction = min(0.95, mix_intensity * 0.95)
            z_pos = (1.0 - turb_fraction) * static_z + turb_fraction * turbulent_z
            
            # Calculate Light at each cell depth
            # Apply Clump Self-Shading (Geometric) — Mass^(-1/3) (Surface/Volume Scaling)
            f_clump_shade = self.clump_mass[self._aidx()] ** (-1.0/3.0)
            
            # Biofouling Effect
            I_s_red *= np.exp(-self.fouling_factor)
            I_s_blue *= np.exp(-self.fouling_factor)
            I_s_green *= np.exp(-self.fouling_factor)
            
            I_red = I_s_red * np.exp(-k_red * z_pos) * f_clump_shade
            I_blue = I_s_blue * np.exp(-k_blue * z_pos) * f_clump_shade
            I_green = I_s_green * np.exp(-k_green * z_pos) * f_clump_shade
            
            # Total Energy (for Inhibition/Bleaching/Acclimation)
            cells_I_total = I_red + I_blue + I_green
            
            # Growth Energy (Photosynthetically Active Radiation - PUR)
            # Plants primarily use Red light for efficient growth
            cells_I_growth = I_red 
            
            # --- Photo-Acclimation (Hysteresis) ---
            # EMA lag = tau_acclim (1–4h per strain); corrected from fixed 0.1 (~0.2h, 5–20× too fast)
            alpha_accum = self.dt / max(params['tau_acclim'], 0.01)
            self.cells_acclimation[self._aidx()] += alpha_accum * (cells_I_total - self.cells_acclimation[self._aidx()])
            I_effective = self.cells_acclimation[self._aidx()]
            
            # 2. Temperature Factor (Gaussian)
            temp_factor = np.exp(-0.5 * ((self.temp - params['T_opt'])/5.0)**2)
            
            # Photo-Inhibition / Shock
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1168)
            diff = (cells_I_total - I_effective)
            shock_factor = np.exp(-0.000003 * (diff**2))  # 3e-6: 24% penalty at diff=300 (was 9%)

            # --- Oxygen Toxicity (ROS Damage) — per-cell using local layer DO2 ---
            if self.difficulty >= 1:
                phys_scale = 0.75 if self.difficulty == 1 else 1.0
                f_O2 = np.maximum(0.0, 1.0 - ((cells_local_do2 / 22.0)**4) * phys_scale)
            else:
                f_O2 = np.ones(self.num_active, dtype=np.float32)
            
            # 3. Growth Rate (Haldane)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1182)
            
            # Parameters — both now strain-specific (Kii bug fix: was hardcoded 2500)
            Ks_I = params['Ks_light']
            Ki_I = params['Kii']
            f_I_raw = cells_I_growth / (Ks_I + cells_I_growth + (cells_I_total**2 / Ki_I))
            f_I_raw = np.nan_to_num(f_I_raw)
            # Normalise to [0,1] by the Haldane theoretical maximum at I_peak = sqrt(Ks_I * Ki_I).
            # Replaces the ×1.5 hack that allowed f_I > 1.0 and effective mu > mu_max.
            I_peak    = np.sqrt(Ks_I * Ki_I)
            f_I_max   = I_peak / (2.0 * Ks_I + I_peak)
            f_I = np.clip(f_I_raw / (f_I_max + 1e-8), 0.0, 1.0)

            # DEBUG: Save RGB Ratio for observation
            avg_red = np.mean(I_red)
            mean_blue = np.mean(I_blue)
            avg_blue = mean_blue if mean_blue > 0.001 else 1.0
            self.rgb_ratio = avg_red / avg_blue
            
            # Droop Quota
            # Only update active quotas
            current_quotas = self.cells_quota[self._aidx()]
            
            f_Q = np.maximum(0.0, 1.0 - params['Q_min'] / (current_quotas + 1e-6))
            
            # pH Inhibition (Asymmetric Gaussian — Arthrospira/Spirulina platensis)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1209)
            if self.ph <= 9.3:
                f_pH = np.exp(-0.5 * ((self.ph - 9.3) / 0.7) ** 2)
            else:
                f_pH = np.exp(-0.5 * ((self.ph - 9.3) / 1.0) ** 2)
            
            # Osmotic Stress — conductivity as ionic strength proxy (all ions: N, P, HCO3-, salts)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1218)
            cond_for_osmosis = getattr(self, 'conductivity', 19000.0)
            if cond_for_osmosis > 25000.0:
                f_Osmosis = np.exp(-0.5 * ((cond_for_osmosis - 25000.0) / 15000.0) ** 2)
            else:
                f_Osmosis = 1.0
                
            # Debug stats (debug_shock/debug_clump are set every step in _update_episode_stats)
            if self.step_count % 100 == 0:
                self.debug_f_I = np.mean(f_I)
                self.debug_f_Q = np.mean(f_Q)
                self.debug_f_pH = f_pH
                self.debug_f_O2 = float(np.mean(f_O2))

            # Shear repair tax: sigmoid centred at 100 RPM; filamentous Spirulina fragments under
            # shear (max 35% penalty).
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1245)
            repair_factor = 1.0 / (1.0 + np.exp(-0.12 * (stir_rpm - 100.0)))
            repair_tax = 1.0 - (0.35 * repair_factor)

            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1253)
            shear_stress = _fclip((stir_rpm - 80.0) / 100.0, 0.0, 1.0)
            self.membrane_integrity -= shear_stress * 0.001          # slow degradation at high RPM
            self.membrane_integrity += (1.0 - self.membrane_integrity) * 0.002  # ~5h to recover
            self.membrane_integrity = _fclip(self.membrane_integrity, 0.0, 1.0)
            fatigue_tax = 1.0 - (0.15 * (1.0 - self.membrane_integrity))  # max 15% penalty

            # Phosphorus-Limited Growth — Ks_P now strain-specific (0.5–2.0 mg P/L, Spirulina range)
            Ks_P = params.get('Ks_P', 1.0)
            f_P = self.p_pool / (Ks_P + self.p_pool)
            f_P = _fclip(f_P, 0.0, 1.0)

            # Carbon-limited growth: Spirulina's bicarbonate CCM makes HCO3- the main carbon
            # source at Zarrouk levels; dissolved CO2 contributes little.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1267)
            Kc_CO2  = 0.5    # mg/L half-saturation for dissolved CO2 (unchanged, minor pathway)
            Kc_HCO3 = 0.05   # mM half-saturation for bicarbonate (high-affinity CCM)
            f_co2_term  = cells_local_co2 / (Kc_CO2 + cells_local_co2)
            f_hco3_term = float(self.bicarbonate / (Kc_HCO3 + self.bicarbonate))
            f_carbon    = 0.15 * f_co2_term + 0.85 * f_hco3_term  # 15% CO2, 85% HCO3- (Spirulina CCM)

            # CO2 toxicity — per-cell
            f_CO2_tox = 1.0 / (1.0 + (cells_local_co2 / (self.co2_toxicity_Ki_mgL + 1e-9)) ** self.co2_toxicity_hill)
            f_CO2_tox = np.clip(f_CO2_tox, 0.0, 1.0)
            self.debug_f_CO2 = float(np.mean(f_CO2_tox))

            current_mu = params['mu_max'] * f_I * f_Q * f_P * f_carbon * f_CO2_tox * temp_factor * shock_factor * f_O2 * f_pH * f_Osmosis * repair_tax * fatigue_tax
            current_mu = np.clip(current_mu, 0.0, 5.0) 
            
            # --- Maintenance Respiration ---
            # Night: 2.0× elevated dark respiration (Tomaselli et al. 1987: ~2×; Tomaselli et al. 1995: 79/39=2.03×)
            dark_factor   = 2.0 if self.is_night else 1.0
            m_respiration = 0.010 * params['mu_max'] * dark_factor
            
            # Net Growth Rate = Photosynthesis - Respiration
            # This can be negative (mass loss) if light/nutrients are insufficient!
            net_mu = current_mu - m_respiration
            
            # Grow Biomass
            growth_mult = np.exp(net_mu * self.dt)
            # Clip multiplier to avoid single-step explosion (both up and down)
            growth_mult = np.clip(growth_mult, 0.5, 2.0)
            
            self.cells_mass[self._aidx()] *= growth_mult

            # Droop quota dilution: as cells grow, intracellular quota (N/biomass) is diluted.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1301)
            q_dil = np.clip(np.maximum(0.0, net_mu) * self.dt, 0.0, 0.95)
            self.cells_quota[self._aidx()] *= (1.0 - q_dil)
            self.cells_quota[self._aidx()] = np.maximum(self.cells_quota[self._aidx()], 0.0)

            # --- PROBABILISTIC LYSIS DEATH (replaces dead-code hard starvation check) ---
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1308)
            mean_mu       = float(np.mean(current_mu))
            stress_factor = np.clip(
                (m_respiration - mean_mu) / (m_respiration + 1e-9), 0.0, 1.0
            )
            self.debug_mu = mean_mu
            self.debug_stress = stress_factor
            lysis_rate  = 5e-4 + (2e-3 * (stress_factor ** 2))  # per hour; 5e-4 ~1.2%/day healthy baseline
            death_prob  = lysis_rate * self.dt             # per step

            curr_active_indices = self._aidx()
            survival_mask = np.random.uniform(0, 1, self.num_active) > death_prob
            dying_indices = curr_active_indices[~survival_mask]

            if len(dying_indices) > 0:
                self._remove_cells(dying_indices)

            # Cap mass at upper bound; no lower floor — let starving cells lose mass naturally
            self.cells_mass[self._aidx()] = np.minimum(self.cells_mass[self._aidx()], 5e8)

            # O4: cells below the death threshold face certain lysis on this cycle.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1336)
            starving_mask = self.active_mask & (self.cells_mass < 1e7)
            if np.any(starving_mask):
                self._remove_cells(np.where(starving_mask)[0])
                
            # Nutrient Uptake (O3: Monod saturation on nitrogen pool)
            uptake_rate = 0.5 * (self.n_pool / (params['Ks'] + self.n_pool))
            uptake_amount = uptake_rate * self.dt

            # Update intracellular quota from nitrogen uptake
            self.cells_quota[self._aidx()] += uptake_amount
            # Enforce Q_max: prevents unbounded hyperaccumulation (Droop model assumption)
            self.cells_quota[self._aidx()] = np.minimum(self.cells_quota[self._aidx()], params['Q_max'])

            # N drain factor 0.01: max drain ~37.5 mg N/h at 7500 cells (calibrated; see calibration.md)
            total_uptake_mg = uptake_amount * self.num_active * 0.01
            # P uptake: Monod saturation with strain-specific Ks_P.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1360)
            p_uptake_rate = self.p_pool / (Ks_P + self.p_pool)
            total_p_uptake_mg = p_uptake_rate * self.dt * self.num_active * 0.0014
            # nut_flow dosing composition: 79% N, 16% P, 5% inorganic salts (Zarrouk stock ratio).
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1365)
            n_input_step = nut_flow * 0.79 * self.dt
            # N waste penalty removed: it caused mode collapse (early overdose, then zero dosing
            # for the rest of the episode).
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1369)
            self.n_pool        = max(0.0, self.n_pool        - total_uptake_mg    + n_input_step)
            self.p_pool        = max(0.0, self.p_pool        - total_p_uptake_mg  + (nut_flow * 0.16 * self.dt))
            # K, Mg consumed proportionally to N uptake (~2% of N uptake by mass); floor at 50 mg/L
            ext_uptake_mg = total_uptake_mg * 0.02
            self.ext_nutrients = max(50.0, self.ext_nutrients - ext_uptake_mg     + (nut_flow * 0.05 * self.dt))
            
            # --- CELL DIVISION (Reproduction) ---
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1378)
            ready_to_divide = (self.active_mask) & (self.cells_mass >= 1.4e8)
            n_dividing = np.sum(ready_to_divide)
            
            if n_dividing > 0:
                # Find empty slots
                inactive_indices = np.where(~self.active_mask)[0]
                n_slots = len(inactive_indices)
                n_spawns = min(n_dividing, n_slots)
                
                if n_spawns > 0:
                    # Indices of parents that get to spawn (capped by slots)
                    # We need precise mapping. 
                    parent_indices = np.where(ready_to_divide)[0][:n_spawns]
                    child_indices = inactive_indices[:n_spawns]
                    
                    # Split Mass
                    self.cells_mass[parent_indices] *= 0.5

                    # Activate Children
                    self.active_mask[child_indices] = True
                    self._aidx_cache = None
                    self.cells_mass[child_indices] = self.cells_mass[parent_indices]
                    self.cells_z[child_indices] = self.cells_z[parent_indices]
                    self.cells_x[child_indices] = self.cells_x[parent_indices]
                    # Quota conservation: cytoplasm splits — both daughters start at Q/2
                    # (previously child got full parent quota, violating Droop mass balance)
                    half_quota = self.cells_quota[parent_indices] * 0.5
                    self.cells_quota[parent_indices] = half_quota
                    self.cells_quota[child_indices]  = half_quota
                    self.cells_acclimation[child_indices] = self.cells_acclimation[parent_indices]
                    
                    self.clump_mass[child_indices] = 1.0 # Children start as single cells
                    
                    self.num_active += n_spawns

                # B9 removed: when slots are full, cells keep growing to the 5e8 mass cap instead of
                # stalling at the division threshold.
                # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1416)

        else:
            # shock_factor is otherwise only assigned in the num_active>0 branch above;
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1424)
            shock_factor = np.array([1.0], dtype=np.float32)
        return total_uptake_mg, shock_factor

    def _gas_transfer(self, stir_rpm):
        """Gas fractions and the kLa transfer coefficient. Returns (k_La, o2_frac, co2_frac)."""
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1435)
        od = self.od
        avg_clump = np.mean(self.clump_mass[self._aidx()]) if self.num_active > 0 else 1.0
        
        # Resistance starts at 1.0 (water-like broth) and increases with OD/clumping.
        # od/0.5: broth viscosity doubles at OD=0.5 — consistent with real PBR measurements.
        flow_resistance = 1.0 + ((od / 0.5)**2) * (avg_clump ** 0.5)
        
        co2_flow_lpm = 0.0  # no CO2 injection — ambient air sparge only (see gas-phase config)
        total_gas_lpm = max(1e-6, self.base_air_flow_lpm + co2_flow_lpm)
        co2_frac = ((self.ambient_co2_frac * self.base_air_flow_lpm) + co2_flow_lpm) / total_gas_lpm
        co2_frac = _fclip(co2_frac, self.ambient_co2_frac, 0.12)
        o2_frac = _fclip((self.ambient_o2_frac * self.base_air_flow_lpm) / total_gas_lpm, 0.05, self.ambient_o2_frac)

        # kLa correlation is stir/gas-flow driven with no volume term, so it carries over
        # unchanged from the 30L->20L resize (units: 1/h).
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1453)
        mix_term = np.clip(stir_rpm / 200.0, 0.25, 1.0)
        gas_term = np.clip(total_gas_lpm / self.base_air_flow_lpm, 0.5, 6.0)
        base_kLa = (0.6 + 5.0 * (mix_term ** 1.3)) * (gas_term ** 0.35)
        k_La = _fclip(base_kLa / flow_resistance, 0.05, 12.0)
        self.kLa = k_La
        return k_La, o2_frac, co2_frac

    def _biomass_change(self):
        """Net biomass change this step (mg), measured against last step's mass."""
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1464)
        total_mass_mg = np.sum(self.cells_mass[self._aidx()]) * 1e-9  # pg -> mg
        if self.step_count == 0:
            return 0.0
        return float(total_mass_mg - self.last_mass * 1e-9)

    def _apply_harvest(self):
        """Periodic semi-continuous harvest/dilution. Returns (harvested_mg, is_harvest_event)."""
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1484)
        harvested_this_step_mg = 0.0
        is_harvest_event = (self.step_count > 0) and (self.step_count % self.HARVEST_INTERVAL_STEPS == 0)
        # Fix #16 (v19): apply the INTERVAL MEAN of the harvest action, not the instantaneous
        # sample; the accumulator resets after each event.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1496)
        if is_harvest_event:
            harvest_frac_applied = (self._harvest_action_sum
                                    / max(self._harvest_action_count, 1))
            self._harvest_action_sum = 0.0
            self._harvest_action_count = 0
            # Fix #20 (v22): HARVEST PUMP DELIVERY ERROR (D1+).
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1504)
            if self.difficulty >= 1 and self.HARVEST_PUMP_ERROR > 0.0:
                harvest_frac_applied *= float(np.random.uniform(
                    1.0 - self.HARVEST_PUMP_ERROR, 1.0 + self.HARVEST_PUMP_ERROR))
        else:
            harvest_frac_applied = 0.0
        frac_diluted = _fclip(harvest_frac_applied, 0.0, 0.95) if is_harvest_event else 0.0
        if frac_diluted > 0.0 and self.num_active > 0:
            active_idx = self._aidx()
            # Per-cell Bernoulli removal, not round(frac*n): with small populations frac*n often
            # rounds to 0, which would silently disable dilution.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1522)
            remove_local = np.random.uniform(0.0, 1.0, len(active_idx)) < frac_diluted
            remove_idx = active_idx[remove_local]
            if len(remove_idx) > 0:
                harvested_this_step_mg = float(np.sum(self.cells_mass[remove_idx])) * 1e-9
                self._remove_cells(remove_idx)

            # The removed volume is replaced with fresh medium (FRESH_MEDIUM, the same fill as
            # reset()); dissolved gases refill air-equilibrated, not at zero.
            fresh, keep = self.FRESH_MEDIUM, 1.0 - frac_diluted
            for pool in ("ext_nutrients", "n_pool", "p_pool", "bicarbonate", "salt"):
                setattr(self, pool, getattr(self, pool) * keep + fresh[pool] * frac_diluted)
            for layer in ("do2_s", "do2_b"):
                setattr(self, layer, getattr(self, layer) * keep + fresh["do2"] * frac_diluted)
            for layer in ("co2_s", "co2_b"):
                setattr(self, layer, getattr(self, layer) * keep + fresh["co2"] * frac_diluted)

        self.cumulative_harvested_mg += harvested_this_step_mg
        if self.step_count >= self.BACK_HALF_STEP:
            self.cumulative_harvested_mg_back_half += harvested_this_step_mg
        self.harvest_integral        += frac_diluted * self.volume_L  # cumulative volume harvested (L)
        return harvested_this_step_mg, is_harvest_event

    def _update_gas_and_carbonate(self, k_La, o2_frac, co2_frac, mix_intensity, delta_mass_mg):
        """2-layer O2/CO2 balances, bicarbonate, and pH."""
        # 2-layer model: surface z<10cm (10 L), bulk z>=10cm (20 L).
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1560)
        LAYER_DEPTH = 0.10
        vol_s = self.volume_L * (LAYER_DEPTH / self.reactor_depth)   # 10 L
        vol_b = self.volume_L - vol_s                                  # 20 L
        f_s   = getattr(self, '_f_surface_cells', LAYER_DEPTH / self.reactor_depth)

        # Net O2 yield: photosynthesis 6CO2+6H2O→C6H12O6+6O2 gives 5.33 mg O2/mg C (50% C biomass → 2.67 mg O2/mg DW gross).
        # Net (after growth respiration ~40% overhead): ~1.5 mg O2 per mg net DW gained.
        o2_production = delta_mass_mg * 1.5  # mg O2 total (net stoichiometric yield)

        # Distribute O2/CO2 by productivity: surface cells are ~3× more active per cell
        surf_prod  = f_s * 3.0
        bulk_prod  = (1.0 - f_s) * 1.0
        total_prod = max(surf_prod + bulk_prod, 1e-9)
        w_s = surf_prod / total_prod
        w_b = bulk_prod / total_prod

        # Per-layer gas-atmosphere exchange; surface gets slight headspace bonus (+20%)
        kLa_s = k_La * 1.20
        kLa_b = k_La * 0.90
        o2_sat = 8.0 * (o2_frac / self.ambient_o2_frac)
        o2_xfer_s = kLa_s * (o2_sat - self.do2_s) * self.dt
        o2_xfer_b = kLa_b * (o2_sat - self.do2_b) * self.dt

        # Inter-layer mixing: first-order exchange at rate kLa_inter (1/h), mass-conserving.
        kLa_inter = k_La * mix_intensity * 0.5
        mix_s, mix_b = self._layer_exchange(self.do2_s, self.do2_b, kLa_inter, vol_s, vol_b)

        self.do2_s = _fclip(mix_s + (o2_production * w_s / vol_s) + o2_xfer_s, 0.0, 40.0)
        self.do2_b = _fclip(mix_b + (o2_production * w_b / vol_b) + o2_xfer_b, 0.0, 30.0)

        # DIC balance per layer
        # Henry's law: [CO2(aq)] = K_H * pCO2; K_H=29 mol/(L·atm), MW=44 → 1276 mg/(L·atm) at 30°C
        co2_sat = _fclip(1276.0 * co2_frac, 0.3, 60.0)
        co2_xfer_s = kLa_s * (co2_sat - self.co2_s) * self.dt
        co2_xfer_b = kLa_b * (co2_sat - self.co2_b) * self.dt
        co2_mix_s, co2_mix_b = self._layer_exchange(self.co2_s, self.co2_b, kLa_inter, vol_s, vol_b)

        # Photosynthetic stoichiometry: 6CO2 → C6H12O6; 6×44/(6×12) = 3.67 mg CO2/mg C fixed.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1600)
        co2_uptake = max(0.0, delta_mass_mg) * 1.835
        co2_release = max(0.0, -delta_mass_mg) * 1.835
        co2_bio_s = _fclip((co2_release * w_s - co2_uptake * w_s) / vol_s, -1.0, 1.0)
        co2_bio_b = _fclip((co2_release * w_b - co2_uptake * w_b) / vol_b, -1.0, 1.0)

        self.co2_s = _fclip(co2_mix_s + co2_xfer_s + co2_bio_s, 0.0, 80.0)
        self.co2_b = _fclip(co2_mix_b + co2_xfer_b + co2_bio_b, 0.0, 80.0)

        # Volume-weighted averages — used by reward, PBRS, observations
        self.do2         = (self.do2_s * vol_s + self.do2_b * vol_b) / self.volume_L
        self.dissolved_co2 = (self.co2_s * vol_s + self.co2_b * vol_b) / self.volume_L

        # Bicarbonate balance: depleted by photosynthesis (85% of DIC uptake via HCO3-,
        # matching f_carbon), replenished by CO2 sparging at a pH-dependent fraction.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1615)
        bicarb_consumed_mM = (max(0.0, delta_mass_mg) * 0.85 / 12.0) / self.volume_L  # C fixed via HCO3-
        co2_to_hco3_mg = max(0.0, co2_xfer_s * vol_s + co2_xfer_b * vol_b)  # CO2 absorbed from sparging
        # At current pH, fraction of newly dissolved CO2 that converts to HCO3- (Henderson-Hasselbalch equilibrium)
        f_to_hco3 = float(10.0 ** (self.ph - 6.35) / (1.0 + 10.0 ** (self.ph - 6.35)))
        bicarb_added_mM = (co2_to_hco3_mg / 44.0 / self.volume_L) * f_to_hco3
        # NOTE: this 5.0 ceiling is 40x below the 200 mM Zarrouk baseline set in reset(), and
        # it is load-bearing: raising it pushes pH to ~10.5 and halves yield, because the
        # carbonate constants were never calibrated to 200 mM. Known physics debt.
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1623)
        self.bicarbonate = _fclip(self.bicarbonate - bicarb_consumed_mM + bicarb_added_mM, 0.0, 5.0)

        # pH via Henderson-Hasselbalch: pH = pKa1 + log10([HCO3-]/[CO2(aq)])
        # pKa1 temperature correction: -0.002/°C (symmetric around 25°C; Stumm & Morgan 1996)
        pKa1 = 6.35 - 0.002 * (self.temp - 25.0)
        # mg/L ÷ g/mol = mM (dimensional identity: mg/L × mol/g = 10^-3 mol/L = mM)
        co2_aq_mM = max(self.co2_b, 0.001) / 44.0
        ph_eq = _fclip(pKa1 + np.log10(max(self.bicarbonate, 0.001) / co2_aq_mM), 5.0, 11.0)
        # pH tracks CO2 dissolution rate (kLa ~1.5-5/h); 2.0/h gives ~30-min response — physically correct
        self.ph = _fclip(self.ph + 2.0 * self.dt * (ph_eq - self.ph), 5.0, 11.0)

    def _layer_exchange(self, c_s, c_b, k, vol_s, vol_b):
        """One explicit step of first-order mixing between the surface and bulk layers.

        The surface layer relaxes toward the bulk at rate k (1/h); the bulk receives the same
        mass back, so total dissolved mass is conserved. The flux used to be computed as a
        concentration change and then divided by layer volume a second time, which made
        mixing 10-20x weaker than k says. Stable while k*dt < 1 (k <= ~6/h here, dt = 0.02 h).
        """
        moved_mg = k * (c_b - c_s) * self.dt * vol_s
        return c_s + moved_mg / vol_s, c_b - moved_mg / vol_b

    def _update_pigment_and_salt(self, I_surface, delta_mass_mg, total_uptake_mg):
        # Pigment dynamics (photo-inhibition & chlorosis)
        # Bleaching: High Light (>1000) or Low Nitrogen (<100) damages pigment
        avg_light = I_surface * np.exp(-0.2 * self.reactor_depth/2) # Approx mid-depth light
        is_bleached = (avg_light > 1000.0) or (self.n_pool < 75.0)  # rescaled to Zarrouk's richer N baseline
        
        if is_bleached:
            self.pigment -= 0.01 * self.dt # Slow degradation
        else:
            self.pigment += 0.01 * self.dt # Slow recovery
        self.pigment = np.clip(self.pigment, 0.2, 1.0) # Min 20% pigment
        
        # Salinity accumulation
        lysis_mg      = max(0.0, -delta_mass_mg)
        salt_inflow_mg = total_uptake_mg * 0.1  # impurity carryover from nutrient feed
        salt_decay_mg  = lysis_mg * 0.5          # ion release from lysed cells
        self.salt += (salt_inflow_mg + salt_decay_mg) / max(self.volume_L, 1e-9)

    def _conductivity(self):
        """Kohlrausch molar-conductance estimate of the medium's conductivity (µS/cm)."""
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1687)

        # n_pool as NaNO₃-N: [NO₃⁻]=[Na⁺] = n_pool/14000 mol/L (MW_N=14)
        sigma_n    = (71.4 + 50.1) * (self.n_pool / 14000.0)

        # p_pool as K₂HPO₄-P: [HPO₄²⁻]=p/30970, [K⁺]=2×[HPO₄²⁻] (MW_P=30.97)
        sigma_p    = (57.0 + 2.0 * 73.5) * (self.p_pool / 30970.0)

        # ext_nutrients as K₂SO₄ proxy: 2×K⁺ + SO₄²⁻, MW=174.3 g/mol
        sigma_ext  = (307.0 / 174300.0) * self.ext_nutrients

        # salt as NaCl-equivalent ionic background: Na⁺+Cl⁻=126.5, MW=58.44
        sigma_salt = (126.5 / 58440.0) * self.salt

        # pH: OH⁻ (198.0) and H⁺ (349.8) — replaces linear 100×|7-pH| proxy
        sigma_ph   = 198.0 * (10.0 ** (self.ph - 14.0)) + 349.8 * (10.0 ** (-self.ph))

        # NaHCO3 bicarbonate: HCO3- λ=44.5 S·cm²/mol, MW=61 g/mol — fixes O7 calibration gap
        sigma_hco3 = (44.5 / 61000.0) * (self.bicarbonate * 61.0)  # bicarbonate in mM → mg/L equiv
        # Kohlrausch temperature correction ~2%/°C
        cond_temp    = 1.0 + 0.020 * (self.temp - 25.0)
        return (sigma_n + sigma_p + sigma_ext + sigma_salt + sigma_ph + sigma_hco3) * cond_temp * 1000.0

    def _update_sensors(self):
        """Sensor-facing state: RGB absorbance, strain micro-drift, conductivity."""
        # RGB absorbance (chlorophyll proxy) = linear OD x pigment health
        self.rgb_absorbance = self.od * self.pigment
        
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1678)
        if self.difficulty >= 2 and self.step_count > 0 and self.step_count % 250 == 0:
            self.strain_params['mu_max'] *= np.random.uniform(0.99, 1.01)
            self.strain_params['Ks'] *= np.random.uniform(0.99, 1.01)
            self.strain_params['Ks_light'] *= np.random.uniform(0.99, 1.01)

        self.conductivity = self._conductivity()
