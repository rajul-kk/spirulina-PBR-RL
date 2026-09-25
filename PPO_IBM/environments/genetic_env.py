
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


# Apparent carbonate constants for a ~0.2 M NaHCO3 (Zarrouk) medium, where ionic strength
# lowers both pKs by ~0.3 from their infinite-dilution values (6.35, 10.33).
def _carbonate_constants(temp):
    return 10.0 ** -(6.1 - 0.002 * (temp - 25.0)), 10.0 ** -(10.0 - 0.009 * (temp - 25.0))


def _solve_ph(alk_mM, dic_mM, temp):
    """pH from total alkalinity (meq/L) and DIC (mM): carbonate system plus water, by bisection
    (carbonate alkalinity rises monotonically with pH)."""
    K1, K2 = _carbonate_constants(temp)
    alk, dic = alk_mM * 1e-3, dic_mM * 1e-3
    lo, hi = 4.0, 12.5
    for _ in range(40):
        ph = 0.5 * (lo + hi)
        h = 10.0 ** -ph
        a = dic * (K1 * h + 2.0 * K1 * K2) / (h * h + K1 * h + K1 * K2) + 1e-14 / h - h
        if a > alk:
            hi = ph
        else:
            lo = ph
    return 0.5 * (lo + hi)


def _air_equilibrated_dic(alk_mM, co2_aq_mM, temp):
    """DIC (mM) of a medium with this alkalinity once its CO2(aq) has equilibrated at co2_aq_mM."""
    K1, K2 = _carbonate_constants(temp)
    c0 = co2_aq_mM * 1e-3
    lo, hi = 4.0, 12.5
    for _ in range(40):
        ph = 0.5 * (lo + hi)
        h = 10.0 ** -ph
        a = c0 * (K1 / h + 2.0 * K1 * K2 / (h * h)) + 1e-14 / h - h
        if a > alk_mM * 1e-3:
            hi = ph
        else:
            lo = ph
    h = 10.0 ** -(0.5 * (lo + hi))
    return c0 * (h * h + K1 * h + K1 * K2) / (h * h) * 1e3


# Air sparge at 420 ppm, Henry 1276 mg/(L atm) at ~30C.
_CO2_AIR_SAT_MGL = 1276.0 * 420e-6

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
        # The flat panel is lit through its 1.0 x 0.30 m face, so light crosses the
        # 20 L / 0.30 m^2 = 6.7 cm thickness (a typical flat-panel light path).
        self.light_path_m = self.volume_L * 1e-3 / (self.reactor_width * self.reactor_depth)

        # --- Thermostat: heater/chiller holding T_SETPOINT. Heating capacity comfortably
        # covers ambient loss; chiller capacity is limited, so sustained full light still
        # warms the tank (~39C at 2000 umol) instead of being free.
        self.T_SETPOINT = 35.0
        self.HEAT_MAX_C_PER_H = 2.0
        self.COOL_MAX_C_PER_H = 0.6

        # --- Gas-Phase / Carbonate Configuration (closed 20L PBR) ---
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-37)
        self.base_air_flow_lpm = 0.30        # Baseline air sparge (L/min)
        self.ambient_co2_frac = 420e-6       # Atmospheric CO2 mol fraction
        self.ambient_o2_frac = 0.209         # Atmospheric O2 mol fraction
        self.co2_toxicity_Ki_mgL = 30.0      # Mild dissolved-CO2 toxicity onset
        self.co2_toxicity_hill = 2.0         # Hill exponent for toxicity curve
        # pH-stat CO2 feed, standard in Spirulina production: air alone cannot supply the
        # carbon a dense culture fixes, so CO2 is injected whenever pH rises above setpoint.
        self.PH_SETPOINT = 10.0
        self.CO2_MAX_LPM = 0.05

        # --- Automated nutrient dosing (hysteresis on N and P) ---
        self.N_DOSE_LOW = 150.0      # mg N/L — start dosing below this
        self.N_DOSE_HIGH = 350.0     # mg N/L — stop dosing above this
        self.N_DOSE_RATE = 240.0     # mg stock/h into the tank when active (~10 mg N/L/h)
        self.P_DOSE_LOW = 25.0       # mg P/L — start dosing below this
        self.P_DOSE_HIGH = 70.0      # mg P/L — stop dosing above this
        # Stock composition by mass, matched to biomass demand (N_FRAC:P_FRAC ~8:1) so neither
        # nutrient runs out while the other piles up.
        self.DOSE_N_FRAC = 0.83
        self.DOSE_P_FRAC = 0.10
        self.DOSE_EXT_FRAC = 0.07

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
            # 0.040/h (~1/day, ~17 h doubling). Was 0.055, but a light-term normalisation bug
            # capped f_I at 0.70, so the effective max was 0.038; that bug is fixed.
            'mu_max':    np.random.normal(0.040, 0.006),   # Arthrospira (Spirulina) platensis
            'Ks':        np.random.normal(1.0, 0.2),       # NO3-N Ks retained (cross-species similarity)
            'Ks_light':  np.random.normal(100.0, 10.0),
            'Kii':       np.random.normal(2500.0, 250.0),
            'T_opt':     np.random.normal(36.0, 1.0),      # Spirulina optimum ~35-37C
            'Q_min':     0.5,
            'Q_max':     5.0,
            'tau_acclim': np.random.uniform(1.0, 4.0),
            'Ks_P':      float(np.random.uniform(1.0, 4.0)),  # Spirulina P half-saturation (Zarrouk-rich medium)
        }
        self.strain_params['mu_max']   = max(0.020, self.strain_params['mu_max'])  # floor at 50% of the mean
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
        self.alkalinity = fresh["alkalinity"]
        self.dic = fresh["dic"]
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
        self._update_carbonate_speciation()

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
        self.od = (init_total_mass * self.MG_PER_MASS_UNIT / self.volume_L) / 300.0
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
            total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * self.MG_PER_MASS_UNIT
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

            # ~250 NTU per od unit (od 1 = 300 mg/L), so the SEN0189's 1000 NTU ceiling sits
            # near ~1.5 g/L instead of saturating at ~4 mg/L.
            noise_scale = 0.05
            turbidity_obs = _fclip((turbidity_base * 250.0 * flow_noise) + np.random.normal(0, noise_scale), 0.0, 1000.0)
            self.turbidity_obs = turbidity_obs  # Store for debug logging
        else:
            self.od = 0.0
            self.conductivity = 0.0
            self.rgb_absorbance = 0.0
            turbidity_obs = 0.0
            self.turbidity_obs = turbidity_obs

        # ~30 lux per umol/m2/s for a red/blue grow LED (80 suited white light and saturated
        # the 16-bit BH1750 at ~820 umol, hiding the top of the light range).
        bh1750_lux = _fclip(self.I_surface * 30.0 + np.random.normal(0.0, 200.0), 0.0, 65535.0)

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
            self.dic,                                         # hidden DIC state (mM)
            mean_fQ,                                          # mean Droop quota
            self.strain_params.get('mu_max', 0.05),          # strain growth rate
            self.strain_params.get('Ks_light', 100.0) / 200.0,  # normalised
        ], dtype=np.float32)

    # Per-event harvest target for reward_harvest (mg per event; events fire every
    # HARVEST_INTERVAL_STEPS, 12 per episode), from a harvest-fraction grid sweep.
    # (full rationale: docs/decision_history.md#--environments-genetic_env-py-496)
    TARGET_MG_PER_EVENT = 850.0     # scripted expert's cold-start median / 12 events (D0/D1)

    # Target standing OD: the peak of the PBRS OD-health term and the reference for the
    # harvest-collapse penalty.
    OD_TARGET = 0.75                # middle of the expert's flat optimum (0.58-0.97, ~225 mg/L)

    # Biomass represented by one unit of cells_mass, in mg. Each agent (~1e8 units) stands for
    # ~10 mg of dry biomass, so the 45-7500 agents the curriculum uses span ~25 mg/L to
    # ~5 g/L in 20 L: real Spirulina densities. At 1e-9 the culture ran at ~4 mg/L, where
    # self-shading, O2 build-up and flocculation never engaged.
    MG_PER_MASS_UNIT = 1e-7

    # Specific light extinction by biomass (m^2 per g dry weight, i.e. per m^-1 per mg/L),
    # per LED channel. PAR-weighted mean ~0.19 m^2/g, the range reported for A. platensis.
    EXT_RED, EXT_BLUE, EXT_GREEN = 0.20, 0.25, 0.06
    RED_FRAC, BLUE_FRAC, GREEN_FRAC = 0.4, 0.4, 0.2

    # Biomass composition (mass fraction of dry weight) that sets medium drawdown per mg grown.
    C_FRAC, N_FRAC, P_FRAC, EXT_FRAC = 0.50, 0.10, 0.012, 0.02

    # Fresh Zarrouk medium: what reset() fills the tank with AND what harvest dilution refills
    # it with. One definition, so the two can't drift apart again (salt did: 2500 vs 1000).
    # 16.8 g/L NaHCO3 gives 200 meq/L alkalinity; once air-equilibrated it holds ~140 mM DIC
    # at pH ~9.9, the range Zarrouk cultures actually run at.
    FRESH_MEDIUM = {
        "ext_nutrients": 300.0,  # mg/L — mineral salts (MgSO4, CaCl2, trace metals)
        "n_pool": 410.0,         # mg N/L — NaNO3 2.5 g/L
        "p_pool": 89.0,          # mg P/L — K2HPO4 0.5 g/L
        "alkalinity": 200.0,     # meq/L — NaHCO3 16.8 g/L
        "dic": _air_equilibrated_dic(200.0, _CO2_AIR_SAT_MGL / 44.0, 30.0),  # mM
        "salt": 2500.0,          # mg/L — NaCl + K2SO4 + trace salts ionic background
        "do2": 7.0,              # mg/L — air-equilibrated
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

        Two weighted components, each in [0, 1], scaled by a multiplicative thermal factor:
          - OD health: peaks at 1.0 exactly at OD_TARGET, falls off both ways. Below target
            the culture is under-productive; above it, overgrown (light limitation, crash
            risk). Uses a log-ratio distance so the falloff is symmetric in relative terms
            and never flattens to zero gradient -- the dead zone that cost this project four
            runs was a bounded penalty reaching exactly zero slope, which cannot happen here.
          - Population health: saturating in num_active, so extinction is a deep hole and
            large populations plateau rather than paying unbounded reward for hoarding
            biomass the agent never harvests.
          - Thermal health: the growth model's own temperature factor, ~1 near the strain's
            optimum. Without it a culture cooked to 45C by sustained max light, parked at OD
            target, scored maximal Phi while it stopped growing; the damage only reached the
            reward ~1000 steps later as lost harvest, and v58/v59 both drifted into it.

        Must stay a function of state alone: no deltas, no action, no episode phase. Adding
        any transition-dependent quantity here silently voids the policy-invariance guarantee.
        """
        od_x = max(float(self.od) / self.OD_TARGET, 1e-6)
        # log-ratio distance from target, squashed; 1.0 at target, ->0 far either side
        phi_od = float(np.exp(-(np.log(od_x) ** 2) / 2.0))

        phi_pop = float(np.tanh(self.num_active / self.PHI_POP_REF))

        phi_temp = float(np.exp(-0.5 * ((self.temp - self.strain_params['T_opt']) / 5.0) ** 2))

        phi = self.PHI_OD_W * phi_od + self.PHI_POP_W * phi_pop
        return self.PHI_SCALE * phi * phi_temp / (self.PHI_OD_W + self.PHI_POP_W)

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

        self.time_t += self.dt
        mix_intensity = stir_rpm / 200.0  # 0 to 1

        self._update_flocculation(stir_rpm)
        self._mix_cells()
        dosed_mg, shock_factor = self._update_biology(I_surface, stir_rpm, mix_intensity, nut_flow)

        # Transfer coefficients and the biomass change are taken on the PRE-harvest culture;
        # the gas-layer balances below then run on the post-harvest tank.
        k_La, o2_frac, co2_frac = self._gas_transfer(stir_rpm)
        delta_mass_mg = self._biomass_change()
        harvested_this_step_mg, is_harvest_event = self._apply_harvest()

        # Recompute standing mass/OD post-dilution — this is what the tank actually holds
        total_mass_mg = np.sum(self.cells_mass[self._aidx()]) * self.MG_PER_MASS_UNIT
        self.od = (total_mass_mg / self.volume_L) / 300.0

        self._update_gas_and_carbonate(k_La, o2_frac, co2_frac, mix_intensity, delta_mass_mg)
        self._update_pigment_and_salt(delta_mass_mg, dosed_mg)
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
        self.dosing_integral += self.current_nut_flow * self.DOSE_N_FRAC * self.dt
        return stir_rpm, I_surface, nut_flow

    def _apply_light_schedule(self, I_surface, nut_flow):
        """Day/night cycle. (An early-episode light cap used to live here too: for the first
        24 h it silently turned light actions above the cap into no-ops.)"""
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
        # Impeller heating, ~P = Np*rho*N^3*D^5 (~2 W at 200 RPM in 20 L, ~0.1 C/h).
        stir_heat = (stir_rpm / 200.0) ** 3 * 0.1 * self.dt * phys_scale
        self.temp += stir_heat
        # Thermostat (proportional heater/chiller, capacity-limited).
        u = _fclip(2.0 * (self.T_SETPOINT - self.temp), -self.COOL_MAX_C_PER_H, self.HEAT_MAX_C_PER_H)
        self.temp += u * self.dt

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

    def _mix_cells(self):
        """Well-mixed culture. At dt = 72 s every cell circulates the 0.3 m tank many times per
        step, so its position is redrawn uniformly each step. The 2-D velocity field this
        replaces moved cells up to 18 m per step, which amounted to the same thing."""
        if self.num_active > 0:
            idx = self._aidx()
            self.cells_z[idx] = np.random.uniform(0.0, self.reactor_depth, self.num_active)
            self.cells_x[idx] = np.random.uniform(0.0, self.reactor_width, self.num_active)

    def _update_biology(self, I_surface, stir_rpm, mix_intensity, nut_flow):
        """Light field, growth, lysis, nutrient uptake and division. Returns (dosed_mg, shock_factor)."""
        # --- BIOLOGY ---
        
        # 1. Shear Stress (RPM > 400)
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1083)

        dosed_mg = 0.0
        if self.num_active > 0:
            params = self.strain_params
            idx = self._aidx()

            # 1. Spectral light field across the 6.7 cm light path.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1093)
            fouling = np.exp(-self.fouling_factor)
            I_s_red = I_surface * self.RED_FRAC * fouling
            I_s_blue = I_surface * self.BLUE_FRAC * fouling
            I_s_green = I_surface * self.GREEN_FRAC * fouling

            # Attenuation: water + bubble scattering + biomass (specific extinction x mg/L).
            X_mgL = np.sum(self.cells_mass[idx]) * self.MG_PER_MASS_UNIT / self.volume_L
            k_scatter = stir_rpm * 0.004
            k_red = 0.5 + self.EXT_RED * X_mgL + k_scatter
            k_blue = 0.2 + self.EXT_BLUE * X_mgL + k_scatter
            k_green = 0.05 + self.EXT_GREEN * X_mgL + k_scatter

            # Gas layer each cell sits in this step (surface = top 10 cm).
            surface_cell_mask = self.cells_z[idx] < 0.10
            self._f_surface_cells = float(np.mean(surface_cell_mask))
            cells_local_do2 = np.where(surface_cell_mask, self.do2_s, self.do2_b).astype(np.float32)
            cells_local_co2 = self.dissolved_co2

            # Every cell crosses the light path many times per 72 s step, so it experiences the
            # whole gradient: growth averages the instantaneous light response over the path
            # (8-point midpoint rule); slow processes (acclimation, shock) see the path mean.
            zq = (np.arange(8) + 0.5) / 8.0 * self.light_path_m
            q_red = I_s_red * np.exp(-k_red * zq)
            q_tot = q_red + I_s_blue * np.exp(-k_blue * zq) + I_s_green * np.exp(-k_green * zq)

            # Clump self-shading (geometric, mass^-1/3 surface/volume scaling).
            f_clump_shade = self.clump_mass[idx] ** (-1.0 / 3.0)
            cells_I_growth = f_clump_shade[:, None] * q_red[None, :]      # (n, 8)
            cells_I_q = f_clump_shade[:, None] * q_tot[None, :]
            cells_I_total = cells_I_q.mean(axis=1)                        # path-mean light
            self.mean_cell_light = float(np.mean(cells_I_total))

            # --- Photo-Acclimation (Hysteresis) ---
            # EMA lag = tau_acclim (1–4h per strain)
            alpha_accum = self.dt / max(params['tau_acclim'], 0.01)
            self.cells_acclimation[idx] += alpha_accum * (cells_I_total - self.cells_acclimation[idx])
            I_effective = self.cells_acclimation[idx]

            # 2. Temperature Factor (Gaussian)
            temp_factor = np.exp(-0.5 * ((self.temp - params['T_opt'])/5.0)**2)

            # Photo-Inhibition / Shock (sustained light change vs the acclimated level)
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1168)
            diff = (cells_I_total - I_effective)
            shock_factor = np.exp(-0.000003 * (diff**2))  # 24% penalty at diff=300

            # --- Oxygen inhibition (D1+). Spirulina productivity falls noticeably above
            # ~25 mg/L DO and roughly halves by ~35 mg/L.
            if self.difficulty >= 1:
                phys_scale = 0.75 if self.difficulty == 1 else 1.0
                f_O2 = 1.0 / (1.0 + phys_scale * (cells_local_do2 / 35.0) ** 4)
            else:
                f_O2 = np.ones(self.num_active, dtype=np.float32)

            # 3. Growth Rate (Haldane), red light drives growth, total light inhibits.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1182)
            Ks_I = params['Ks_light']
            Ki_I = params['Kii']
            f_I_raw = cells_I_growth / (Ks_I + cells_I_growth + (cells_I_q**2 / Ki_I))
            # Normalise by the curve's true maximum. It peaks at total I* = sqrt(Ks*Ki), where the
            # red share is RED_FRAC*I*; normalising by the all-red peak (as before) capped f_I
            # at ~0.70 and silently cut the effective mu_max by 30%.
            I_peak = np.sqrt(Ks_I * Ki_I)
            f_I_max = self.RED_FRAC * I_peak / (2.0 * Ks_I + self.RED_FRAC * I_peak)
            f_I = np.clip(np.nan_to_num(f_I_raw).mean(axis=1) / (f_I_max + 1e-8), 0.0, 1.0)

            self.rgb_ratio = float(np.mean(q_red)) / max(float(np.mean(I_s_blue * np.exp(-k_blue * zq))), 1e-3)

            # Droop Quota
            current_quotas = self.cells_quota[idx]
            f_Q = np.maximum(0.0, 1.0 - params['Q_min'] / (current_quotas + 1e-6))

            # pH (Arthrospira): optimum ~9.5-10, sharper decline above ~10.5.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1209)
            if self.ph <= 9.8:
                f_pH = np.exp(-0.5 * ((self.ph - 9.8) / 0.9) ** 2)
            else:
                f_pH = np.exp(-0.5 * ((self.ph - 9.8) / 0.7) ** 2)

            # Osmotic stress. Zarrouk itself is ~30 mS/cm and is Spirulina's native medium, so
            # the onset sits above it.
            # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1218)
            cond_for_osmosis = getattr(self, 'conductivity', 30000.0)
            if cond_for_osmosis > 40000.0:
                f_Osmosis = np.exp(-0.5 * ((cond_for_osmosis - 40000.0) / 15000.0) ** 2)
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

            # Dry biomass built this step (mg); sets the medium drawdown below.
            grown_mg = float(np.sum(self.cells_mass[idx] * np.maximum(growth_mult - 1.0, 0.0))) * self.MG_PER_MASS_UNIT
            self.cells_mass[idx] *= growth_mult

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

            # Medium drawdown follows the biomass actually built (N_FRAC etc. of dry weight).
            # It used to scale with agent count, ran on at Q_max, and took mg from mg/L pools:
            # ~600x the N the biomass needed. Dosing enters as mg into the whole tank.
            dosed_mg = nut_flow * self.dt
            V = self.volume_L
            n_drain = grown_mg * self.N_FRAC / V
            self.n_pool = max(0.0, self.n_pool - n_drain + dosed_mg * self.DOSE_N_FRAC / V)
            self.p_pool = max(0.0, self.p_pool - grown_mg * self.P_FRAC / V + dosed_mg * self.DOSE_P_FRAC / V)
            self.ext_nutrients = max(50.0, self.ext_nutrients - grown_mg * self.EXT_FRAC / V
                                     + dosed_mg * self.DOSE_EXT_FRAC / V)
            # Nitrate assimilation releases OH-: +1 eq alkalinity per mol N.
            self.alkalinity += n_drain / 14.0

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
        return dosed_mg, shock_factor

    def _gas_transfer(self, stir_rpm):
        """Gas fractions and the kLa transfer coefficient. Returns (k_La, o2_frac, co2_frac)."""
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1435)
        od = self.od
        avg_clump = np.mean(self.clump_mass[self._aidx()]) if self.num_active > 0 else 1.0
        
        # Resistance starts at 1.0 (water-like broth) and rises with density/clumping. Spirulina
        # broth stays near water viscosity below a few g/L; od 10 is ~3 g/L here.
        flow_resistance = 1.0 + ((od / 10.0)**2) * (avg_clump ** 0.5)

        # pH-stat CO2 feed: proportional above PH_SETPOINT, full flow 0.3 pH units over.
        co2_flow_lpm = _fclip(self.CO2_MAX_LPM * (self.ph - self.PH_SETPOINT) / 0.3, 0.0, self.CO2_MAX_LPM)
        self.co2_flow_lpm = co2_flow_lpm
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
        total_mass_mg = np.sum(self.cells_mass[self._aidx()]) * self.MG_PER_MASS_UNIT
        if self.step_count == 0:
            return 0.0
        return float(total_mass_mg - self.last_mass * self.MG_PER_MASS_UNIT)

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
                harvested_this_step_mg = float(np.sum(self.cells_mass[remove_idx])) * self.MG_PER_MASS_UNIT
                self._remove_cells(remove_idx)

            # The removed volume is replaced with fresh medium (FRESH_MEDIUM, the same fill as
            # reset()); dissolved gases refill air-equilibrated, not at zero.
            fresh, keep = self.FRESH_MEDIUM, 1.0 - frac_diluted
            for pool in ("ext_nutrients", "n_pool", "p_pool", "alkalinity", "dic", "salt"):
                setattr(self, pool, getattr(self, pool) * keep + fresh[pool] * frac_diluted)
            for layer in ("do2_s", "do2_b"):
                setattr(self, layer, getattr(self, layer) * keep + fresh["do2"] * frac_diluted)

        self.cumulative_harvested_mg += harvested_this_step_mg
        if self.step_count >= self.BACK_HALF_STEP:
            self.cumulative_harvested_mg_back_half += harvested_this_step_mg
        self.harvest_integral        += frac_diluted * self.volume_L  # cumulative volume harvested (L)
        return harvested_this_step_mg, is_harvest_event

    def _update_gas_and_carbonate(self, k_La, o2_frac, co2_frac, mix_intensity, delta_mass_mg):
        """2-layer O2 balance, one well-mixed DIC pool, and pH from alkalinity + DIC."""
        # 2-layer O2: surface z<10cm (6.7 L), bulk (13.3 L).
        # (full rationale: docs/decision_history.md#--environments-genetic_env-py-1560)
        LAYER_DEPTH = 0.10
        vol_s = self.volume_L * (LAYER_DEPTH / self.reactor_depth)
        vol_b = self.volume_L - vol_s
        w_s = getattr(self, '_f_surface_cells', LAYER_DEPTH / self.reactor_depth)
        w_b = 1.0 - w_s

        # Net O2 yield ~1.5 mg O2 per mg DW gained (2.67 gross, less growth respiration).
        o2_production = delta_mass_mg * 1.5

        # Per-layer gas-atmosphere exchange; surface gets slight headspace bonus (+20%)
        kLa_s = k_La * 1.20
        kLa_b = k_La * 0.90
        o2_sat = 8.0 * (o2_frac / self.ambient_o2_frac)
        o2_xfer_s = kLa_s * (o2_sat - self.do2_s) * self.dt
        o2_xfer_b = kLa_b * (o2_sat - self.do2_b) * self.dt

        # Inter-layer mixing: first-order exchange at rate kLa_inter (1/h), mass-conserving.
        kLa_inter = k_La * mix_intensity * 0.5
        mix_s, mix_b = self._layer_exchange(self.do2_s, self.do2_b, kLa_inter, vol_s, vol_b)

        self.do2_s = _fclip(mix_s + (o2_production * w_s / vol_s) + o2_xfer_s, 0.0, 60.0)
        self.do2_b = _fclip(mix_b + (o2_production * w_b / vol_b) + o2_xfer_b, 0.0, 60.0)
        self.do2 = (self.do2_s * vol_s + self.do2_b * vol_b) / self.volume_L

        # DIC: gas exchange of CO2(aq) toward the sparge gas, minus carbon fixed into biomass
        # (C_FRAC of dry weight), plus carbon returned by net biomass loss.
        co2_sat = _fclip(1276.0 * co2_frac, 0.3, 160.0)          # mg/L, Henry at ~30C
        co2_flux_mM = k_La * (co2_sat - self.dissolved_co2) * self.dt / 44.0
        bio_mM = delta_mass_mg * self.C_FRAC / 12.0 / self.volume_L
        self.dic = max(1e-3, self.dic + co2_flux_mM - bio_mM)
        self._update_carbonate_speciation()

    def _update_carbonate_speciation(self):
        """pH, CO2(aq), HCO3- and CO3(2-) from alkalinity and DIC (equilibrium is fast)."""
        self.ph = _solve_ph(self.alkalinity, self.dic, self.temp)
        K1, K2 = _carbonate_constants(self.temp)
        h = 10.0 ** -self.ph
        d = h * h + K1 * h + K1 * K2
        self.dissolved_co2 = self.dic * h * h / d * 44.0     # mg/L
        self.bicarbonate = self.dic * K1 * h / d              # mM
        self.carbonate = self.dic * K1 * K2 / d               # mM

    def _layer_exchange(self, c_s, c_b, k, vol_s, vol_b):
        """One explicit step of first-order mixing between the surface and bulk layers.

        The surface layer relaxes toward the bulk at rate k (1/h); the bulk receives the same
        mass back, so total dissolved mass is conserved. The flux used to be computed as a
        concentration change and then divided by layer volume a second time, which made
        mixing 10-20x weaker than k says. Stable while k*dt < 1 (k <= ~6/h here, dt = 0.02 h).
        """
        moved_mg = k * (c_b - c_s) * self.dt * vol_s
        return c_s + moved_mg / vol_s, c_b - moved_mg / vol_b

    def _update_pigment_and_salt(self, delta_mass_mg, dosed_mg):
        # Pigment: bleaches when cells see high light on average, or under N starvation.
        is_bleached = (getattr(self, 'mean_cell_light', 0.0) > 1000.0) or (self.n_pool < 75.0)
        if is_bleached:
            self.pigment -= 0.01 * self.dt
        else:
            self.pigment += 0.01 * self.dt
        self.pigment = np.clip(self.pigment, 0.2, 1.0)

        # Salinity: impurities carried in with the nutrient stock, ions from lysed biomass.
        lysis_mg = max(0.0, -delta_mass_mg)
        self.salt += (dosed_mg * 0.02 + lysis_mg * 0.5) / self.volume_L

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

        # Carbonate system: Na+ counter-ion (= alkalinity, from NaHCO3), HCO3- and CO3(2-).
        sigma_hco3 = (50.1 * self.alkalinity + 44.5 * self.bicarbonate + 138.6 * self.carbonate) / 1000.0
        # Kohlrausch temperature correction ~2%/°C
        cond_temp    = 1.0 + 0.020 * (self.temp - 25.0)
        # Summing limiting conductances overestimates at Zarrouk's ~0.3 M ionic strength, where
        # molar conductivities run ~25% below their infinite-dilution values.
        concentration_factor = 0.75
        return (sigma_n + sigma_p + sigma_ext + sigma_salt + sigma_ph + sigma_hco3) * concentration_factor * cond_temp * 1000.0

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
