
import os
import sys
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict

# Per-step physics trace, OFF by default. See the gate at the `[EnvDebug]` print for why:
# it accounted for 43% of every training log in this project. Turn on when debugging physics:
#     ENV_DEBUG=1 python diagnostics/dynamic_profile_sweep_od.py
ENV_DEBUG = os.environ.get("ENV_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

class GeneticPhotobioreactorEnv(gym.Env):
    """
    Individual-Based Model (IBM) Photobioreactor Environment.
    Tracks N individual algal cells as particles in 1D depth (z-axis) using vectorized operations.
    Implements Genetic Domain Randomization with unique algal strains per episode.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_cells: int = 300000, initial_cells: int = 3000, difficulty: int = 2,
                 lights_off_hour: Optional[float] = None, lights_on_hour: float = 6.0,
                 enable_fouling: bool = True):
        super(GeneticPhotobioreactorEnv, self).__init__()
        self.difficulty = difficulty
        
        # --- CONFIGURATION ---
        self.max_cells = max_cells
        self.initial_cells = initial_cells
        self.num_active = initial_cells
        self.dt = 0.02  # Time step (0.02h); 7200 steps = 144h episode
        self.reactor_depth = 0.30  # 30cm depth
        self.reactor_width = 1.0
        self.volume_L = 20.0  # 20L reactor volume (fully controlled indoor PBR target)
        
        # --- Gas-Phase / Carbonate Configuration (closed 20L PBR) ---
        # No CO2 injection: validated empirically that Spirulina's Zarrouk bicarbonate
        # reservoir (~200 mM) self-buffers pH near 9.5 without any active control — across
        # 6000 random-action steps CO2 injection never fired and pH stayed in [8.54, 9.44].
        # Only the baseline ambient-air sparge remains (420ppm atmospheric CO2).
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
        self.F_MAX = 0.5                    # max fraction removed in one harvest event
        self.HARVEST_INTERVAL_STEPS = 600   # 600*0.02h = 12h between harvest decisions (12/episode)

        # Action: [Stirring, Light, Harvest fraction] — CO2 and Nutrient dosing remain
        # automated/implicit. Harvest fraction is only applied on periodic harvest-event
        # steps (see step() dilution/harvest block); ignored on other steps.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

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
        
        self.ext_nutrients = 300.0  # mg/L — Zarrouk mineral salts (MgSO4, CaCl2, trace metals)
        self.n_pool = 410.0         # mg N/L — Zarrouk NaNO3 2.5 g/L
        self.p_pool = 89.0          # mg P/L — Zarrouk K2HPO4 0.5 g/L
        self.bicarbonate = 200.0    # mM — Zarrouk: NaHCO3 16.8 g/L -> ~200 mM HCO3-
        self.ph = self.buffer_equilibrium_ph
        self.do2 = 7.0
        # 2-layer gas model: surface (z<10cm, 10L) and bulk (z>=10cm, 20L)
        self.do2_s = 7.0
        self.do2_b = 7.0
        self.co2_s = 6.2
        self.co2_b = 6.2
        self._f_surface_cells = 1.0 / 3.0  # geometric layer fraction fallback
        self.temp = 36.0 # Ambient temperature
        self.time_t = 0.0 # Continuous time for turbulence
        
        # Hardware Smoothing State (EMA)
        self.current_stir_rpm = 50.0
        self.current_nut_flow = 0.0
        # Sensor-lag model: best case is 2 steps at high RPM; low RPM is slower.
        self._sensor_delay_min_steps = 2
        self._sensor_delay_max_steps = 8
        self._ph_obs_ema = self.ph
        self._temp_obs_ema = self.temp
        self.dosing_integral = 0.0  # cumulative mg N added (internal PID tracking only)
        self.harvest_integral = 0.0  # cumulative volume harvested (L) — pump counter, obs[2] source
        self.cumulative_harvested_mg = 0.0  # running total mg harvested across the episode (per-step dilution)
        self._harvest_action_sum = 0.0      # Fix #16 (v19): interval-averaged harvest action
        self._harvest_action_count = 0
        self.od_sum_back_half = 0.0
        self.od_count_back_half = 0
        self.reward_term_sums = {"od": 0.0, "biomass": 0.0, "od_delta": 0.0, "harvest": 0.0}
        self.I_surface = 0.0        # last delivered PAR (µmol/m²/s) — BH1750 source signal
        self._ph_bias = 0.0         # per-episode additive pH calibration offset
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.action_smooth_coef = 0.003
        
        # Advanced Physics State
        self.salt = 1000.0 # mg/L
        self.pigment = 1.0 # Health 0-1
        
        # Flocculation State (Clumping)
        # 1.0 = Single Cell. >1.0 = Aggregate.
        self.clump_mass = np.ones(self.max_cells, dtype=np.float32)
        
        # --- GENETIC PARAMS (Placeholder, set in reset) ---
        self.strain_params = {}
        
        self.step_count = 0
        self.max_steps = 7200  # 7200 steps × 0.02h = 144h episode; 1 rollout = 1 episode

        # --- DAY/NIGHT CYCLE ---
        # lights_off_hour: hour of day (0-24) when lights turn off. None = always on (default).
        # lights_on_hour:  hour of day (0-24) when lights come back on.
        # Example: lights_off_hour=20, lights_on_hour=6  ->  14h light / 10h dark.
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
            # Was N(0.080, 0.015) — ~8.7h doubling. This is faster than the project's OWN
            # cited source (literature.md: Torzillo et al. 1993, mu_max 0.04-0.07 h^-1;
            # the old mean of 0.08 sat above the ENTIRE cited range). Independently
            # confirmed via fresh literature search: real Zarrouk-medium Arthrospira batch
            # studies report generation times of ~2.7-3.2 days (not ~9h), and PBR-optimal
            # cases report specific growth rates as low as ~0.12/day (~0.005/h). Recentered
            # on the cited range's midpoint (~0.055/h, ~12.6h doubling) — still on the
            # faster/optimistic end of real reported values, but no longer contradicting
            # the project's own citation.
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
        self._randomize_strain()
        
        # Initialize Population
        self.num_active = self.initial_cells
        self.active_mask[:] = False
        self.active_mask[:self.num_active] = True
        
        self.cells_z[:self.num_active] = np.random.uniform(0, self.reactor_depth, self.num_active)
        self.cells_x[:self.num_active] = np.random.uniform(0, self.reactor_width, self.num_active)
        # Super-Agent Scaling: 1 Agent = 2,500,000 Cells (~500pg each)
        # Density-dependent starting mass:
        # At 300 cells, mass is ~1.25e8. At 15,000 cells (Log Ladder limit),
        # mass drops to ~0.8e8 (starving) due to immediate shelf-shading/nutrient competition.
        density_ratio = np.clip(self.initial_cells / 15000.0, 0.0, 1.0)
        starting_mass = 1.25e8 - (density_ratio * 0.45e8)
        self.cells_mass[:self.num_active] = np.random.normal(starting_mass, 1e6, self.num_active)
        self.cells_quota[:self.num_active] = np.random.uniform(3.5, 5.0, self.num_active)  # start near Q_max (replete Zarrouk)
        # Initialize acclimation to average light (approx 200)
        self.cells_acclimation[:self.num_active] = np.random.uniform(100.0, 300.0, self.num_active)
        self.clump_mass[:self.num_active] = 1.0 # Start as single cells
        
        self.ext_nutrients = 300.0  # mg/L — Zarrouk mineral salts (MgSO4, CaCl2, trace metals)
        self.n_pool = 410.0         # mg N/L — Zarrouk NaNO3 2.5 g/L
        self.p_pool = 89.0          # mg P/L — Zarrouk K2HPO4 0.5 g/L
        self.bicarbonate = 200.0    # mM — Zarrouk: NaHCO3 16.8 g/L -> ~200 mM HCO3-
        self._phi_prev = None
        self.ph = self.buffer_equilibrium_ph
        self.do2 = 7.0
        self.do2_s = 7.0
        self.do2_b = 7.0
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
        self.salt = 2500.0          # Zarrouk: NaCl + K2SO4 + trace salts -> higher ionic background
        self.pigment = 1.0
        self.dissolved_co2 = 6.2   # equilibrium CO2(aq) at pH 9.5 with 200 mM HCO3- (Zarrouk)
        self.co2_s = 6.2
        self.co2_b = 6.2
        
        self.dosing_integral = 0.0  # reset internal PID tracking each episode
        self.harvest_integral = 0.0  # cumulative volume harvested (L), accumulates per-step via dilution
        self.cumulative_harvested_mg = 0.0  # running total mg harvested across the episode (curriculum metric)
        self._harvest_action_sum = 0.0      # Fix #16 (v19): must reset per episode, or the
        self._harvest_action_count = 0      # first event of a new episode averages stale steps
        self.od_sum_back_half = 0.0         # for time-averaged OD (curriculum metric)
        self.od_count_back_half = 0
        self.reward_term_sums = {"od": 0.0, "biomass": 0.0, "od_delta": 0.0, "harvest": 0.0}
        self.I_surface = 0.0        # reset BH1750 source signal
        # --- Sim-to-Real Sensor Drift & Lag (D1+) ---
        # 8 channels: [Turbidity, pH, Dosing_integral, Conductivity, Temperature, Light(BH1750),
        #              turbidity_EMA (Fix #18), episode_phase (Fix #18)]
        if self.difficulty >= 1:
            self._sensor_drift_mult = np.random.uniform(0.95, 1.05, size=8)
            self._sensor_drift_mult[1] = 1.0   # pH: additive bias only, no multiplicative drift
            # Fix #18: the EMA is DERIVED from channel 0, so it must inherit that channel's
            # drift rather than draw an independent one — an independent draw would let the
            # policy average the two to cancel a drift a real controller cannot cancel.
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
        # B6: initialize sensor state here so _get_obs() never reads stale episode values
        # Compute conductivity from actual reset pools so first obs matches formula
        _sigma_init = (
            (71.4 + 50.1) * (self.n_pool / 14000.0) +
            (57.0 + 2.0 * 73.5) * (self.p_pool / 30970.0) +
            (307.0 / 174300.0) * self.ext_nutrients +
            (126.5 / 58440.0) * self.salt +
            198.0 * (10.0 ** (self.ph - 14.0)) + 349.8 * (10.0 ** (-self.ph))
        )
        self.conductivity = _sigma_init * (1.0 + 0.020 * (self.temp - 25.0)) * 1000.0
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
        self._prev_od_for_rate = float(self.od)  # For OD growth rate reward
        self._phi_prev = None

        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate stats only if cells exist
        if self.num_active > 0:
            total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9
            self.od = (total_mass_mg / self.volume_L) / 300.0  # Volume-normalised OD

            # conductivity, rgb_absorbance, last_hourly_od are guaranteed by reset()

            # Turbidity sensor: realistic nephelometric model
            avg_clump = np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0

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
            # Fix #19 (v22): window biofilm scatters extra light into the detector — the
            # reading drifts HIGH while true biomass is unchanged. Monotonic within an
            # episode, so an EMA cannot filter it out (see the accumulation comment in step()).
            turbidity_base *= (1.0 + float(getattr(self, "turb_fouling_factor", 0.0)))

            # Bubble/flow-induced noise (RPM-dependent)
            # Real sensors show high-frequency noise from bubbles and turbulent eddies
            rpm = float(getattr(self, 'current_stir_rpm', 50.0))
            flow_noise = 1.0 + 0.03 * (rpm / 200.0) * np.random.normal(0, 1)

            # Convert to raw NTU units (0-5000 range for sim-to-real transfer)
            # Add base sensor noise floor (Difficulty scaled)
            noise_scale = 0.05
            # SEN0189 saturates at 1000 NTU — clip matches hardware ceiling
            turbidity_obs = float(np.clip((turbidity_base * 1000.0 * flow_noise) + np.random.normal(0, noise_scale), 0.0, 1000.0))
            self.turbidity_obs = turbidity_obs  # Store for debug logging
        else:
            self.od = 0.0
            self.conductivity = 0.0
            self.rgb_absorbance = 0.0
            turbidity_obs = 0.0
            self.turbidity_obs = turbidity_obs
            if not hasattr(self, 'max_historical_od'): self.max_historical_od = 0.0

        bh1750_lux = float(np.clip(self.I_surface * 80.0 + np.random.normal(0.0, 500.0), 0.0, 65535.0))

        # Fix #18 (v21): long-window EMA of turbidity. The raw channel carries multiplicative
        # noise that scales with the agent's own stir setting, so a single reading is a poor
        # OD estimate; averaging it over a long window recovers most of the underlying signal.
        # ALPHA corresponds to a ~600-step (one harvest-interval) effective window, matching
        # the timescale the harvest decision actually operates on.
        _EMA_ALPHA = 2.0 / (600.0 + 1.0)
        if not hasattr(self, "_turb_ema") or self._turb_ema is None:
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
            (np.clip(self.step_count / max(self.max_steps, 1), 0.0, 1.0)
             if self.USE_EPISODE_PHASE else
             np.clip((self.step_count % max(self.HARVEST_INTERVAL_STEPS, 1))
                     / max(self.HARVEST_INTERVAL_STEPS, 1), 0.0, 1.0)),
        ], dtype=np.float32)

        # Truncate to the configured width (6 by default, 8 with OBS_EXTENDED). Channels 6-7
        # are still COMPUTED above — the turbidity EMA has to keep updating regardless so it is
        # warm if the flag is switched on, and the cost is two float ops per step.
        base_obs = base_obs[:self._obs_dim]

        # Stochastic Sensor Noise: ±1% jitter at D0, ±2% at D1+
        jitter_mag = 0.02 if self.difficulty >= 1 else 0.01
        jitter = np.random.uniform(1.0 - jitter_mag, 1.0 + jitter_mag, size=(self._obs_dim,))

        # RPM-coupled EMA lag on pH and temperature (D1+)
        if self.difficulty >= 1:
            rpm = float(np.clip(getattr(self, 'current_stir_rpm', 50.0), 50.0, 200.0))
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
            base_obs[1] = float(np.clip(base_obs[1] + self._ph_bias, 0.0, 14.0))

        noisy_obs = base_obs * jitter * self._sensor_drift_mult[:self._obs_dim]
        noisy_obs[3] = float(np.clip(noisy_obs[3], 0.0, 40000.0))   # DFR0300 hard ceiling (post-jitter)
        noisy_obs[5] = float(np.clip(noisy_obs[5], 0.0, 65535.0))  # BH1750 hard ADC ceiling (16-bit)
        if self._obs_dim > 7:
            # Fix #18: phase is the controller's own clock — restore it exactly, since the
            # jitter multiply above would otherwise corrupt a quantity known perfectly.
            noisy_obs[7] = base_obs[7]
        return noisy_obs.astype(np.float32)

    def get_privileged_state(self) -> np.ndarray:
        """4D privileged vector — sim-only, never exposed at deployment.
        Returns: [dissolved_co2, mean_f_Q, mu_max, Ks_light]
        """
        mean_fQ = float(np.mean(np.maximum(
            0.0, 1.0 - 0.5 / (self.cells_quota[self.active_mask] + 1e-6)
        ))) if self.num_active > 0 else 0.0
        return np.array([
            getattr(self, 'dissolved_co2', 2.0),            # hidden DIC state
            mean_fQ,                                          # mean Droop quota
            self.strain_params.get('mu_max', 0.05),          # strain growth rate
            self.strain_params.get('Ks_light', 100.0) / 200.0,  # normalised
        ], dtype=np.float32)

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
    TARGET_MG_PER_EVENT = 12.32

    # Light-path biofouling coefficient. NOTE: 0.0002 is calibrated for conventional OD units
    # (~1-10, lab OD600), but this sim's `od` is volume-normalised as mass_mg/volume_L/300 and
    # sits at ~0.015-0.02 — roughly 250x smaller. At this value the term accumulates only
    # ~0.0003 over a full 144h episode against its own 0.5 cap, i.e. it has been INERT for
    # this project's entire history: `enable_fouling=True` has had no measurable effect.
    # Left at the historical value so runs v11-v22 remain comparable; exposed as a class
    # attribute so feasibility probes can raise it without editing physics.
    # A realistic value for this OD scale is ~0.075 (see Fix #19's turbidity-fouling note).
    LIGHT_FOULING_COEF = 0.0002

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
    TURB_FOULING_COEF = 0.0
    HARVEST_PUMP_ERROR = 0.0
    USE_EPISODE_PHASE = True

    # OBS_EXTENDED: False -> 6 channels (real hardware sensors only; the default and the
    # sim-to-real baseline). True -> 8 channels, adding Fix #18's turbidity EMA and phase.
    # Fix #18 measurably helped PPO (od +74% in v21) but its dimension change orphaned every
    # prior checkpoint, so it is opt-in. Changing this invalidates saved models in BOTH
    # directions — scripts/validate.py detects and reports the mismatch rather than crashing.
    OBS_EXTENDED = False

    def _compute_reward(self, delta_mass_mg, shock_factor, harvested_this_step_mg=0.0, is_harvest_event=False):
        """Semi-continuous reward: sustained growth + periodic dilution/harvest.

        Dense per-step components (OD, biomass — the latter also covers stagnation via its
        own curve — and OD movement) carry the agent between harvest events; reward_harvest
        fires only on harvest-event steps (every HARVEST_INTERVAL_STEPS), rewarding the
        size of that periodic yield. There is no batch "terminal harvest" — harvest happens
        repeatedly through the episode, not once at the end.

        Simplified to 4 terms (from 5) after three training attempts: extra reward-shaping
        terms proved to be a liability, not just complexity — see reward_biomass comment
        below for what was folded/removed and why.
        """
        # Curriculum metric: time-averaged OD over the back half of the episode (steps
        # 3600-7200) — a "sustained healthy steady-state" proxy that can't be gamed by a
        # brief early spike.
        if self.step_count >= 3600:
            self.od_sum_back_half += self.od
            self.od_count_back_half += 1

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
        OD_TARGET = 0.012
        od_x = self.od / OD_TARGET
        reward_od = 0.15 * float(od_x * np.exp(1.0 - od_x))

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
        per_cell_growth = (delta_mass_mg / (self.num_active + 1e-6)) * 1000
        reward_biomass = 0.20 * float(np.tanh(per_cell_growth / 5.0))
        if per_cell_growth < 0.01:
            reward_biomass -= 0.010

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
        OD_RATE_FLOOR = 1e-4
        delta_od = self.od - self._prev_od_for_rate
        rel_delta_od = delta_od / max(self._prev_od_for_rate, OD_RATE_FLOOR)
        reward_od_delta = 0.0 if is_harvest_event else 0.01 * float(np.tanh(rel_delta_od / 2e-4))
        self._prev_od_for_rate = self.od

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
        reward_harvest = 0.5 * float(np.tanh(harvested_this_step_mg / self.TARGET_MG_PER_EVENT))

        reward = reward_od + reward_biomass + reward_od_delta + reward_harvest

        # Episode-accumulated per-term breakdown, exposed via info dict for diagnostics
        # (not used in reward itself).
        self.reward_term_sums["od"] += reward_od
        self.reward_term_sums["biomass"] += reward_biomass
        self.reward_term_sums["od_delta"] += reward_od_delta
        self.reward_term_sums["harvest"] += reward_harvest

        # Tracking for debug log (not used in reward)
        mean_shock = np.mean(shock_factor) if self.num_active > 0 else 1.0
        mean_clump = np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0
        self.debug_shock = float(mean_shock)
        self.debug_clump = float(mean_clump)
        if self.od > getattr(self, 'max_historical_od', 0.0):
            self.max_historical_od = self.od

        # Keep tracking internal true mass for logging, but not for reward
        total_mass = np.sum(self.cells_mass[self.active_mask]) if self.num_active > 0 else 0
        self.last_mass = total_mass

        # Final safeguard on reward
        if not np.isfinite(reward):
            reward = -10.0  # Punishment for breaking physics

        return float(reward)

    def step(self, action):
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
        self._harvest_action_sum += harvest_frac_action
        self._harvest_action_count += 1

        # 1b. Automated PID Controller (Nutrient N/P threshold control only)
        # No CO2 control: Spirulina's Zarrouk bicarbonate reservoir self-buffers pH near
        # 9.5 without any active carbon dosing (validated empirically — see genetic_env
        # gas-phase config comment). Only ambient air sparge feeds the carbonate system.
        # Gate on EITHER N or P running low — dosing replenishes both (87% N, 8% P per
        # BG-11 ratio), but N typically depletes slower than P relative to its dose threshold.
        # Gating on N alone left P to starve silently while N sat in the hold band.
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
        co2_flow    = 0.0  # no CO2 injection — ambient air sparge only (see gas-phase config)

        # Actuator delivery noise (D1+): ±5% simulates pump calibration drift and motor imprecision
        if self.difficulty >= 1:
            stir_rpm *= float(np.random.uniform(0.95, 1.05))
            nut_flow *= float(np.random.uniform(0.95, 1.05))

        # Accumulate internal PID dosing tracker (not exposed to obs)
        self.dosing_integral += self.current_nut_flow * 0.79 * self.dt  # 79% N fraction (Zarrouk ratio)

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

        # --- Temperature Inertia ---
        ambient_temp = 25.0
        # Physics scale: D0=50%, D1=75%, D2=100%
        diff_level = getattr(self, 'difficulty', 0)
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

        # --- Biofouling Accumulation ---
        # Cells adhere to surfaces at low mixing and high biomass density.
        # exp(-0.5) ≈ 60% light transmission at full fouling (cap 0.5).
        if self.enable_fouling:
            # LIGHT_FOULING_COEF is a class attribute so it can be overridden for feasibility
            # probes without editing physics. Default 0.0002 is the historical value — see the
            # inertness note below; it is left unchanged so v11-v22 stay comparable.
            fouling_rate = max(0.0, 1.0 - stir_rpm / 200.0) * self.od * self.LIGHT_FOULING_COEF
            self.fouling_factor += fouling_rate * self.dt
            self.fouling_factor = float(np.clip(self.fouling_factor, 0.0, 0.5))

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
            if self.difficulty >= 1 and self.TURB_FOULING_COEF > 0.0:
                turb_foul_rate = max(0.0, 1.0 - stir_rpm / 200.0) * self.od * self.TURB_FOULING_COEF
                self.turb_fouling_factor += turb_foul_rate * self.dt
                self.turb_fouling_factor = float(np.clip(self.turb_fouling_factor, 0.0, 0.25))

        # --- Physics (Chaotic Turbulence) ---
        # Apply only to active cells
        # We replace simple Brownian motion with structured "Swirls"
        # Flow V(z, t) = Sum( A * sin(k*z - w*t) )
        
        dt_sec = self.dt * 3600
        self.time_t += self.dt
        
        # Base Mixing Intensity
        mix_intensity = stir_rpm / 200.0 # 0 to 1
        
        # --- FLOCCULATION PHYSICS (Mean-Field) ---
        if self.num_active > 0:
            # 1. Aggregation (Sticking) - Orthokinetic + Perikinetic
            # Orthokinetic: Stirring INCREASES collision frequency (Smoluchowski)
            # Sticking = Base (Brownian) + Shear-Induced (RPM)
            # Fix scaling bug: Use true physical OD, not an assumption based on cell count!
            od_approx = self.od  # Use internal OD, not obs[0] (now turbidity!)
            
            # --- New Physics: 
            # - Base chance: 1e-3 (Stronger - 5x boost)
            # - RPM Boost: Increases linearly with Mixing (more collisions)
            # --- Flocculation: Stirring BREAKS clumps (shear dispersal dominates at moderate RPM)
            # rpm_factor now REDUCES sticking — higher RPM = more shear = less aggregation
            # At 0 RPM: rpm_factor=1.0 (max sticking). At 200 RPM: rpm_factor=0.2 (80% less sticking)
            rpm_factor = max(0.1, 1.0 - (stir_rpm / 250.0))
            prob_stick = (od_approx * 1e-3) * rpm_factor
            
            # Apply sticking to active cells only — O(num_active) not O(max_cells)
            # Generating 300k random numbers every step with 3k active cells was 25% of step time.
            _active_idx = np.where(self.active_mask)[0]
            _stick = np.random.uniform(0, 1, self.num_active) < prob_stick
            self.clump_mass[_active_idx[_stick]] += 1.0
            
            # 2. Breakup (Shear + Brownian)
            # Mechanical shear breakup (onset at 80 RPM)
            clump_shear = max(0.0, (stir_rpm - 80.0) / 120.0) ** 2

            # Brownian/diffusive breakup (always active, weak)
            # Prevents runaway aggregation at low RPM
            # Small clumps (1-5) barely affected, large clumps (50+) slowly erode
            brownian_breakup = 0.005 * (self.clump_mass[self.active_mask] - 1.0) ** 0.5

            # Total breakup rate: shear + Brownian diffusion
            breakup_rate = 0.5 * clump_shear * (self.clump_mass[self.active_mask] ** 0.5) + brownian_breakup

            # Apply breakup to all active clumps
            self.clump_mass[self.active_mask] -= breakup_rate * self.dt

            # Physical Lower Bound: 1.0 (Single Cell)
            self.clump_mass[self.active_mask] = np.maximum(self.clump_mass[self.active_mask], 1.0)
            
        
        if self.num_active > 0 and mix_intensity > 0.01:
            # --- 2D Kinematic Turbulence (Airlift / Convection Loop) ---
            # Center (x=0.5): Upward Flow (-z)
            # Walls (x=0,1): Downward Flow (+z)
            # Top/Bottom: Turnaround (Horizontal Flow)
            
            x_pos = self.cells_x[self.active_mask]
            z_pos = self.cells_z[self.active_mask]
            
            # 1. Vertical Velocity (Vz)
            # Cosine profile: Max Up at 0.5, Max Down at 0, 1.
            # Scale: 0.01 m/s * intensity
            v_max_z = 0.05 * mix_intensity   # 0.05 m/s at max RPM — realistic for 30L flat-panel airlift
            # Damping at top/bottom walls (z close to 0 or D)
            # damp_z = 1.0 - (2.0 * (z_pos / self.reactor_depth) - 1.0)**4
            v_macro_z = v_max_z * -np.cos(2 * np.pi * (x_pos - 0.5)) 
            
            # 2. Horizontal Velocity (Vx)
            v_max_x = v_max_z * 5.0 # Aspect ratio scaling (Width >> Depth)
            v_macro_x = v_max_x * (x_pos - 0.5) * np.cos(np.pi * z_pos / self.reactor_depth)
            
            # 3. Turbulence (Random Perturbations)
            # Perlin-like noise
            indices = np.where(self.active_mask)[0]
            turb_freq = 5.0
            turb_phase = 10.0 * self.time_t
            
            v_turb_x = 0.002 * mix_intensity * np.sin(turb_freq * z_pos * 100 - turb_phase + indices)
            v_turb_z = 0.002 * mix_intensity * np.cos(turb_freq * x_pos * 10 - turb_phase + indices)
            
            # 4. Sinking & Diffusion
            r_eff = self.clump_mass[self.active_mask] ** (1.0/3.0)
            v_sink = 0.001 * (self.clump_mass[self.active_mask] ** (2.0/3.0))
            
            noise_x = np.random.normal(0, 1, self.num_active)
            noise_z = np.random.normal(0, 1, self.num_active)
            
            v_diff_x = (np.sqrt(2 * 1e-7) / r_eff) * noise_x
            v_diff_z = (np.sqrt(2 * 1e-7) / r_eff) * noise_z
            
            # Integrate
            dz = (v_macro_z + v_turb_z - v_sink) * dt_sec + v_diff_z
            dx = (v_macro_x + v_turb_x) * dt_sec + v_diff_x
            
            self.cells_z[self.active_mask] += dz
            self.cells_x[self.active_mask] += dx
            
        else:
            # Low mixing -> Sedimentation
            sink_speed = 0.005 * (self.clump_mass[self.active_mask] ** (2.0/3.0))
            r_eff = self.clump_mass[self.active_mask] ** (1.0/3.0)
            
            noise = np.random.normal(0, 1, sum(self.active_mask))
            dz = -sink_speed * self.dt + ((np.sqrt(2 * 1e-7 * dt_sec)/r_eff) * noise)
            
            # X Diffusion only
            noise_x = np.random.normal(0, 1, sum(self.active_mask))
            dx = (np.sqrt(2 * 1e-7 * dt_sec)/r_eff) * noise_x
            
            self.cells_z[self.active_mask] += dz
            self.cells_x[self.active_mask] += dx
        
        # Boundary Conditions (Reflective)
        # Z Boundary
        self.cells_z = np.abs(self.cells_z)
        over_bottom = self.cells_z > self.reactor_depth
        self.cells_z[over_bottom] = 2*self.reactor_depth - self.cells_z[over_bottom]
        self.cells_z = np.clip(self.cells_z, 0, self.reactor_depth)
        
        # X Boundary
        self.cells_x = np.abs(self.cells_x)
        over_width = self.cells_x > self.reactor_width
        self.cells_x[over_width] = 2*self.reactor_width - self.cells_x[over_width]
        self.cells_x = np.clip(self.cells_x, 0, self.reactor_width)
        
        # --- BIOLOGY ---
        
        # 1. Shear Stress (RPM > 400)
        # Random death probability for cells if mixing is too violent
        # Note: We already have this logic downstream at line 550, but let's keep the flow clean.
        # Actually, let's just fall through to the Biology block.

        total_uptake_mg = 0.0
        if self.num_active > 0:
            n_spawns = 0
            params = self.strain_params
            
            # 1. Spectral Light Field (RGB Physics)
            # Action 'light' sets Total Surface Intensity (PAR)
            # I_surface is already calculated at top of step()
            
            # Split Spectrum (Grow Light logic: High Red/Blue, Low Green)
            I_s_red = I_surface * 0.4
            I_s_blue = I_surface * 0.4
            I_s_green = I_surface * 0.2
            
            # Optical Density (Biomass)
            total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9
            current_od = (total_mass_mg / self.volume_L) / 300.0
            
            # Attenuation Coefficients (k)
            # Red: Absorbed STRONGLY by Chlorophyll (Growth)
            # k_red boosted to 3.5 (was 10.0) to allow deep biological growth past 12k cells
            
            # Bubble Scattering (0.004 * RPM)
            k_scatter = stir_rpm * 0.004
            
            k_red = 0.5 + (3.5 * current_od) + k_scatter
            # Blue: Absorbed by pigments but penetrates better (3x better than Red?)
            k_blue = 0.2 + (1.5 * current_od) + k_scatter
            # Green: Reflected/Transmitted (Deep penetration)
            k_green = 0.05 + (0.5 * current_od) + k_scatter
            
            # ── Turbulent Flash-Light Effect (Biologically Accurate) ──────────
            # In real Spirulina PBRs, turbulent mixing causes cells to cycle
            # between the photic zone (surface) and dark zone (deep) rapidly.
            # This "flash-light effect" dramatically increases photosynthetic
            # efficiency (Kok effect): brief intense surface flashes > sustained dim light.
            #
            # At 0 RPM  : cells see only their actual static depth (fully stratified).
            # At 500 RPM: cells see a near-random depth distribution each step (fully mixed).
            static_z = self.cells_z[self.active_mask]
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
            f_clump_shade = self.clump_mass[self.active_mask] ** (-1.0/3.0)
            
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
            self.cells_acclimation[self.active_mask] += alpha_accum * (cells_I_total - self.cells_acclimation[self.active_mask])
            I_effective = self.cells_acclimation[self.active_mask]
            
            # 2. Temperature Factor (Gaussian)
            temp_factor = np.exp(-0.5 * ((self.temp - params['T_opt'])/5.0)**2)
            
            # Photo-Inhibition / Shock
            # Cells experience stress when light changes suddenly
            # Scalar reduced from 0.0001 to 0.000001 to prevent startup death
            # At diff=300: Old penalty=99.99%, New penalty=9% (survivable!)
            diff = (cells_I_total - I_effective)
            shock_factor = np.exp(-0.000003 * (diff**2))  # 3e-6: 24% penalty at diff=300 (was 9%)

            # --- Oxygen Toxicity (ROS Damage) — per-cell using local layer DO2 ---
            if self.difficulty >= 1:
                phys_scale = 0.75 if self.difficulty == 1 else 1.0
                f_O2 = np.maximum(0.0, 1.0 - ((cells_local_do2 / 22.0)**4) * phys_scale)
            else:
                f_O2 = np.ones(self.num_active, dtype=np.float32)
            
            # 3. Growth Rate (Haldane)
            # Growth is driven by RED light availability
            # Inhibition is driven by TOTAL light intensity
            # f_I = I_growth / (Ks + I_growth + I_total^2/Ki)
            
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
            avg_blue = np.mean(I_blue) if np.mean(I_blue) > 0.001 else 1.0
            self.rgb_ratio = avg_red / avg_blue
            
            # Droop Quota
            # Only update active quotas
            current_quotas = self.cells_quota[self.active_mask]
            
            f_Q = np.maximum(0.0, 1.0 - params['Q_min'] / (current_quotas + 1e-6))
            
            # pH Inhibition (Asymmetric Gaussian — Arthrospira/Spirulina platensis)
            # Peak at 9.3 (Zarrouk operating range 8.5-11; native soda-lake alkaliphile)
            # Acid side: σ=0.7 — steep falloff below pH 8, intolerant of neutral pH
            # Alkaline side: σ=1.0 — tolerates up to pH 11 with moderate inhibition
            if self.ph <= 9.3:
                f_pH = np.exp(-0.5 * ((self.ph - 9.3) / 0.7) ** 2)
            else:
                f_pH = np.exp(-0.5 * ((self.ph - 9.3) / 1.0) ** 2)
            
            # Osmotic Stress — conductivity as ionic strength proxy (all ions: N, P, HCO3-, salts)
            # Spirulina is a soda-lake alkaliphile adapted to high ionic strength; Zarrouk
            # medium baseline is ~19,000 µS/cm (vs BG-11's ~3200). Onset raised accordingly.
            # Uses previous step's conductivity (one-step lag, 72s — negligible).
            cond_for_osmosis = getattr(self, 'conductivity', 19000.0)
            if cond_for_osmosis > 25000.0:
                f_Osmosis = np.exp(-0.5 * ((cond_for_osmosis - 25000.0) / 15000.0) ** 2)
            else:
                f_Osmosis = 1.0
                
            # --- DEBUG STATS CAPTURE ---
            if self.step_count % 100 == 0 and self.num_active > 0:
                  self.debug_f_I = np.mean(f_I)
                  
                  
                  self.debug_f_Q = np.mean(f_Q)
                  self.debug_f_pH = f_pH
                  self.debug_f_O2 = float(np.mean(f_O2))
                  self.debug_shock = np.mean(shock_factor) # shock_factor calculated on active_mask in line 260
                  self.debug_clump = np.mean(self.clump_mass[self.active_mask])
            # ---------------------------
            # ---------------------------
            
            # Logistic Hard Limit REMOVED
            # Natural Limits (Gas Transfer Failure) handle carrying capacity now.
            limit_factor = 1.0 
            
            # Calculate Rate
            # --- Shear Repair Tax (sigmoid, centered at 100 RPM) ---
            # Spirulina (Arthrospira) is a filamentous cyanobacterium — helical trichomes
            # fragment under shear far more readily than Chlorella's rigid unicells.
            # Max 35% penalty at sustained 200 RPM.
            repair_factor = 1.0 / (1.0 + np.exp(-0.12 * (stir_rpm - 100.0)))
            repair_tax = 1.0 - (0.35 * repair_factor)

            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Filament breakage accumulates faster than unicell wall fatigue.
            # Onset at ~80 RPM; max 15% growth penalty.
            shear_stress = float(np.clip((stir_rpm - 80.0) / 100.0, 0.0, 1.0))
            self.membrane_integrity -= shear_stress * 0.001          # slow degradation at high RPM
            self.membrane_integrity += (1.0 - self.membrane_integrity) * 0.002  # ~5h to recover
            self.membrane_integrity = float(np.clip(self.membrane_integrity, 0.0, 1.0))
            fatigue_tax = 1.0 - (0.15 * (1.0 - self.membrane_integrity))  # max 15% penalty

            # Phosphorus-Limited Growth — Ks_P now strain-specific (0.5–2.0 mg P/L, Spirulina range)
            Ks_P = params.get('Ks_P', 1.0)
            f_P = self.p_pool / (Ks_P + self.p_pool)
            f_P = float(np.clip(f_P, 0.0, 1.0))

            # Carbon-Limited Growth — Arthrospira/Spirulina has an efficient bicarbonate CCM
            # (active HCO3- transport + carbonic anhydrase), the adaptation that lets it
            # dominate alkaline soda lakes where free CO2 is scarce. HCO3- is the primary
            # DIC source at Zarrouk concentrations (~200 mM); dissolved CO2 contributes little.
            Kc_CO2  = 0.5    # mg/L half-saturation for dissolved CO2 (unchanged, minor pathway)
            Kc_HCO3 = 0.05   # mM half-saturation for bicarbonate (high-affinity CCM)
            f_co2_term  = cells_local_co2 / (Kc_CO2 + cells_local_co2)
            f_hco3_term = float(self.bicarbonate / (Kc_HCO3 + self.bicarbonate))
            f_carbon    = 0.15 * f_co2_term + 0.85 * f_hco3_term  # 15% CO2, 85% HCO3- (Spirulina CCM)

            # CO2 toxicity — per-cell
            f_CO2_tox = 1.0 / (1.0 + (cells_local_co2 / (self.co2_toxicity_Ki_mgL + 1e-9)) ** self.co2_toxicity_hill)
            f_CO2_tox = np.clip(f_CO2_tox, 0.0, 1.0)
            self.debug_f_CO2 = float(np.mean(f_CO2_tox))

            current_mu = params['mu_max'] * f_I * f_Q * f_P * f_carbon * f_CO2_tox * temp_factor * shock_factor * f_O2 * f_pH * f_Osmosis * limit_factor * repair_tax * fatigue_tax
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
            
            self.cells_mass[self.active_mask] *= growth_mult

            # Droop quota dilution: as cells grow, intracellular quota (N/biomass) is diluted.
            # dQ/dt = V(N) - µ*Q; this applies the -µ*Q term per cell.
            # Only positive net_mu dilutes (shrinking cells retain their quota concentration).
            q_dil = np.clip(np.maximum(0.0, net_mu) * self.dt, 0.0, 0.95)
            self.cells_quota[self.active_mask] *= (1.0 - q_dil)
            self.cells_quota[self.active_mask] = np.maximum(self.cells_quota[self.active_mask], 0.0)

            # --- PROBABILISTIC LYSIS DEATH (replaces dead-code hard starvation check) ---
            # Background lysis: ~0.5%/day (realistic Spirulina batch culture baseline).
            # Stress lysis: scales up to ~5%/day when mean current_mu < m_respiration.
            # Never a hard cliff — always a smooth gradient signal for the RL agent.
            mean_mu       = float(np.mean(current_mu))
            stress_factor = np.clip(
                (m_respiration - mean_mu) / (m_respiration + 1e-9), 0.0, 1.0
            )
            self.debug_mu = mean_mu
            self.debug_stress = stress_factor
            lysis_rate  = 5e-4 + (2e-3 * (stress_factor ** 2))  # per hour; 5e-4 ~1.2%/day healthy baseline
            death_prob  = lysis_rate * self.dt             # per step

            curr_active_indices = np.where(self.active_mask)[0]
            survival_mask = np.random.uniform(0, 1, self.num_active) > death_prob
            dying_indices = curr_active_indices[~survival_mask]

            if len(dying_indices) > 0:
                self.active_mask[dying_indices] = False
                self.num_active -= len(dying_indices)
                self.cells_mass[dying_indices]        = 0.0
                self.cells_quota[dying_indices]       = 0.0
                self.cells_acclimation[dying_indices] = 0.0
                self.clump_mass[dying_indices]        = 1.0  # reset dead-cell slots

            # Cap mass at upper bound; no lower floor — let starving cells lose mass naturally
            self.cells_mass[self.active_mask] = np.minimum(self.cells_mass[self.active_mask], 5e8)

            # O4: cells below the death threshold face certain lysis on this cycle
            # Threshold set to 8% of starting mass (1.25e8 pg) — the prior 5e5 floor was
            # unreachable before stochastic lysis killed the cell first (dead code).
            starving_mask = self.active_mask & (self.cells_mass < 1e7)
            if np.any(starving_mask):
                starving_idx = np.where(starving_mask)[0]
                self.active_mask[starving_idx] = False
                self.num_active -= len(starving_idx)
                self.cells_mass[starving_idx]        = 0.0
                self.cells_quota[starving_idx]       = 0.0
                self.cells_acclimation[starving_idx] = 0.0
                self.clump_mass[starving_idx]        = 1.0
                
            # Nutrient Uptake (O3: Monod saturation on nitrogen pool)
            uptake_rate = 0.5 * (self.n_pool / (params['Ks'] + self.n_pool))
            uptake_amount = uptake_rate * self.dt

            # Update intracellular quota from nitrogen uptake
            self.cells_quota[self.active_mask] += uptake_amount
            # Enforce Q_max: prevents unbounded hyperaccumulation (Droop model assumption)
            self.cells_quota[self.active_mask] = np.minimum(self.cells_quota[self.active_mask], params['Q_max'])

            # N drain factor 0.01: max drain ~37.5 mg N/h at 7500 cells (calibrated; see calibration.md)
            total_uptake_mg = uptake_amount * self.num_active * 0.01
            # P uptake: Monod saturation with strain-specific Ks_P.
            # Factor 0.0014 = 0.01 / 7.2 (Redfield N:P ratio by mass — a broadly cross-species
            # phytoplankton constant, applies to cyanobacteria as well as green algae)
            p_uptake_rate = self.p_pool / (Ks_P + self.p_pool)
            total_p_uptake_mg = p_uptake_rate * self.dt * self.num_active * 0.0014
            # nut_flow dosing composition: 79% N, 16% P, 5% inorganic salts — matches Zarrouk
            # stock ratio (NaNO3 2.5 g/L : K2HPO4 0.5 g/L ~ 5:1 N:P by mass, far richer in P
            # than BG-11's ~28:1)
            n_input_step = nut_flow * 0.79 * self.dt
            # N waste penalty removed: it caused mode collapse where agent overdosed early,
            # earned heavy penalties, then locked to zero dosing for the entire episode.
            # phi_cur N Gaussian (peak at 200 mg/L) + starvation penalty below provide the equilibrium signal.
            self.n_pool        = max(0.0, self.n_pool        - total_uptake_mg    + n_input_step)
            self.p_pool        = max(0.0, self.p_pool        - total_p_uptake_mg  + (nut_flow * 0.16 * self.dt))
            # K, Mg consumed proportionally to N uptake (~2% of N uptake by mass); floor at 50 mg/L
            ext_uptake_mg = total_uptake_mg * 0.02
            self.ext_nutrients = max(50.0, self.ext_nutrients - ext_uptake_mg     + (nut_flow * 0.05 * self.dt))
            
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e8 pg (12% above init mass of 1.25e8)
            # Use >= to avoid edge case where cells hover exactly at boundary.
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
                    # self.clump_mass[parent_indices] = 1.0 # Optional: Parents also disperse? No, let them stay stuck.
                    
                    self.num_active += n_spawns

                # B9 removed: when slots are full, let cells continue growing up to the
                # hard 5e8 cap (line ~775). Capping at the division threshold caused OD
                # to plateau at max_cells, killing all growth reward past population ceiling.

        else:
            n_spawns = 0
            current_mu = np.zeros(1) # fallback
            f_Q = np.zeros(1)
            # shock_factor is otherwise only assigned in the num_active>0 branch above;
            # defining it here guarantees it always exists by the time the reward/debug
            # section reads it (previously relied on lazy ternary short-circuit evaluation
            # at each read site — fragile if that code ever gets refactored/extracted).
            shock_factor = np.array([1.0], dtype=np.float32)

        # --- Environmental Dynamics (Macro) ---

        # 1. Shear Stress (RPM > 300) -> MOVED to Biology loop (Repair Tax only)
        # Note: True lethal shear max RPM bounded to 200, so immediate death removed.

        # 2. Gas Exchange (O2 & CO2)
        # Closed-tank model: gas composition is set by baseline air + injected pure CO2.
        # kLa scales with agitation, gas throughput, and broth resistance at high biomass.
        # Bug fix: Use cached self.od instead of calling _get_obs() which
        # would corrupt the pH lag buffer by appending mid-step.
        od = self.od
        avg_clump = np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0
        
        # Resistance starts at 1.0 (water-like broth) and increases with OD/clumping.
        # od/0.5: broth viscosity doubles at OD=0.5 — consistent with real PBR measurements.
        flow_resistance = 1.0 + ((od / 0.5)**2) * (avg_clump ** 0.5)
        
        co2_flow_lpm = co2_flow / 1000.0
        total_gas_lpm = max(1e-6, self.base_air_flow_lpm + co2_flow_lpm)
        co2_frac = ((self.ambient_co2_frac * self.base_air_flow_lpm) + co2_flow_lpm) / total_gas_lpm
        co2_frac = float(np.clip(co2_frac, self.ambient_co2_frac, 0.12))
        o2_frac = float(np.clip((self.ambient_o2_frac * self.base_air_flow_lpm) / total_gas_lpm, 0.05, self.ambient_o2_frac))

        # kLa correlation is stir/gas-flow driven, not volume-parametrized (no volume_L
        # term in the formula below), so it carries over unchanged from the 30L->20L
        # resize. Originally tuned/validated against a 30L sparged tank; re-validate at
        # 20L via the dilution/grid sweep rather than assuming. (units: 1/hour)
        # Agitation drives eddy renewal; gas throughput adds bubble interfacial area.
        mix_term = np.clip(stir_rpm / 200.0, 0.25, 1.0)
        gas_term = np.clip(total_gas_lpm / self.base_air_flow_lpm, 0.5, 6.0)
        base_kLa = (0.6 + 5.0 * (mix_term ** 1.3)) * (gas_term ** 0.35)
        k_La = float(np.clip(base_kLa / flow_resistance, 0.05, 12.0))
        self.kLa = k_La
        
        # Dissolved Oxygen Dynamics
        # Production: Proportional to Growth (approx 1.5g O2 per g Biomass)
        # Respiration: Proportional to maintenance (approx 1.0g O2 per g Biomass lost)
        # Calculate net biomass change from biology step (approx)
        total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9 # pg * 1e-9 = mg? 
        # 1 pg = 10^-12 g. 1 mg = 10^-3 g. So 1 pg = 10^-9 mg. Correct.

        # Bootstrap last_mass if this is step 0 (belt-and-suspenders over reset() init)
        if self.step_count == 0:
            self.last_mass = np.sum(self.cells_mass[self.active_mask])
        
        # Simplify: Delta Mass roughly tracks O2.
        if self.step_count > 0:
            delta_mass_mg = total_mass_mg - (self.last_mass * 1e-9)
        else:
            delta_mass_mg = 0.0
        
        # Guard: clamp delta_mass_mg to prevent NaN from stale last_mass on first step
        delta_mass_mg = float(np.clip(delta_mass_mg, -1e6, 1e6))

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
        harvested_this_step_mg = 0.0
        is_harvest_event = (self.step_count > 0) and (self.step_count % self.HARVEST_INTERVAL_STEPS == 0)
        # Fix #16 (v19): apply the INTERVAL MEAN, not the instantaneous sample. See the comment
        # at the harvest decode above for why. The accumulator is reset after each event so the
        # next interval averages only its own steps.
        if is_harvest_event:
            harvest_frac_applied = (self._harvest_action_sum
                                    / max(self._harvest_action_count, 1))
            self._harvest_action_sum = 0.0
            self._harvest_action_count = 0
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
            if self.difficulty >= 1 and self.HARVEST_PUMP_ERROR > 0.0:
                harvest_frac_applied *= float(np.random.uniform(
                    1.0 - self.HARVEST_PUMP_ERROR, 1.0 + self.HARVEST_PUMP_ERROR))
        else:
            harvest_frac_applied = 0.0
        frac_diluted = float(np.clip(harvest_frac_applied, 0.0, 0.95)) if is_harvest_event else 0.0
        if frac_diluted > 0.0 and self.num_active > 0:
            active_idx = np.where(self.active_mask)[0]
            # Per-cell Bernoulli removal (not round(frac*n)) — at realistic D and small
            # early-episode populations, frac_diluted*n is often <0.5 and would round to
            # 0 every step, silently disabling dilution. Independent per-cell removal
            # probability = frac_diluted correctly gives the right EXPECTED removal rate
            # over time regardless of population size, matching a well-mixed CSTR.
            remove_local = np.random.uniform(0.0, 1.0, len(active_idx)) < frac_diluted
            remove_idx = active_idx[remove_local]
            if len(remove_idx) > 0:
                harvested_this_step_mg = float(np.sum(self.cells_mass[remove_idx])) * 1e-9
                self.active_mask[remove_idx]       = False
                self.num_active                   -= len(remove_idx)
                self.cells_mass[remove_idx]        = 0.0
                self.cells_quota[remove_idx]       = 0.0
                self.cells_acclimation[remove_idx] = 0.0
                self.clump_mass[remove_idx]        = 1.0

            # Dilute dissolved-phase state toward fresh Zarrouk medium concentration
            # (matches reset()'s initial values — feed reservoir is full-strength medium)
            self.n_pool      = self.n_pool      * (1.0 - frac_diluted) + 410.0  * frac_diluted
            self.p_pool      = self.p_pool      * (1.0 - frac_diluted) + 89.0   * frac_diluted
            self.bicarbonate = self.bicarbonate * (1.0 - frac_diluted) + 200.0  * frac_diluted
            self.salt        = self.salt        * (1.0 - frac_diluted) + 1000.0 * frac_diluted
            # Gas-phase layers dilute toward ambient-equilibrium (fresh medium is
            # air-equilibrated, not zero) rather than 0.0
            self.do2_s = self.do2_s * (1.0 - frac_diluted) + 7.0 * frac_diluted
            self.do2_b = self.do2_b * (1.0 - frac_diluted) + 7.0 * frac_diluted
            self.co2_s = self.co2_s * (1.0 - frac_diluted) + 6.2 * frac_diluted
            self.co2_b = self.co2_b * (1.0 - frac_diluted) + 6.2 * frac_diluted

        self.cumulative_harvested_mg += harvested_this_step_mg
        self.harvest_integral        += frac_diluted * self.volume_L  # cumulative volume harvested (L)

        # Recompute standing mass/OD post-dilution — this is what the tank actually holds
        total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9
        # Update OD — normalised by volume (concentration, not total mass)
        self.od = (total_mass_mg / self.volume_L) / 300.0
        # print(f"DEBUG: Mass={total_mass_mg}, OD={self.od}")

        # 3. 2-Layer Gas Exchange (surface z<10cm = 10L, bulk z>=10cm = 20L)
        # Surface cells photosynthesize more (better light) → O2 accumulates at surface,
        # CO2 depletes there. Mixing inter-layer exchange dissipates gradients at high RPM.
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

        # Inter-layer mixing flux (mass-conservative; sign: positive = bulk→surface)
        kLa_inter = k_La * mix_intensity * 0.5
        do2_flux  = kLa_inter * (self.do2_b - self.do2_s) * self.dt

        self.do2_s = float(np.clip(self.do2_s + (o2_production * w_s / vol_s) + o2_xfer_s + do2_flux / vol_s, 0.0, 40.0))
        self.do2_b = float(np.clip(self.do2_b + (o2_production * w_b / vol_b) + o2_xfer_b - do2_flux / vol_b, 0.0, 30.0))

        # DIC balance per layer
        # Henry's law: [CO2(aq)] = K_H * pCO2; K_H=29 mol/(L·atm), MW=44 → 1276 mg/(L·atm) at 30°C
        co2_sat = float(np.clip(1276.0 * co2_frac, 0.3, 60.0))
        co2_xfer_s = kLa_s * (co2_sat - self.co2_s) * self.dt
        co2_xfer_b = kLa_b * (co2_sat - self.co2_b) * self.dt
        co2_flux   = kLa_inter * (self.co2_b - self.co2_s) * self.dt

        # Photosynthetic stoichiometry: 6CO2 → C6H12O6; 6×44/(6×12) = 3.67 mg CO2/mg C fixed.
        # Biomass is ~50% C by dry weight, so per mg DW the CO2 demand is 3.67×0.5 = 1.835 mg CO2/mg DW.
        # Both uptake and release use same ratio: decomposition re-releases the same CO2 per mass.
        co2_uptake = max(0.0, delta_mass_mg) * 1.835
        co2_release = max(0.0, -delta_mass_mg) * 1.835
        co2_bio_s = float(np.clip((co2_release * w_s - co2_uptake * w_s) / vol_s, -1.0, 1.0))
        co2_bio_b = float(np.clip((co2_release * w_b - co2_uptake * w_b) / vol_b, -1.0, 1.0))

        self.co2_s = float(np.clip(self.co2_s + co2_xfer_s + co2_bio_s + co2_flux / vol_s, 0.0, 80.0))
        self.co2_b = float(np.clip(self.co2_b + co2_xfer_b + co2_bio_b - co2_flux / vol_b, 0.0, 80.0))

        # Volume-weighted averages — used by reward, PBRS, observations
        self.do2         = (self.do2_s * vol_s + self.do2_b * vol_b) / self.volume_L
        self.dissolved_co2 = (self.co2_s * vol_s + self.co2_b * vol_b) / self.volume_L

        # Bicarbonate balance: depleted by photosynthesis (85% of DIC uptake via HCO3-),
        # replenished by CO2 sparging — fraction that equilibrates to HCO3- depends on pH.
        # 85% fraction matches f_carbon's 0.85 * f_hco3 term (Spirulina CCM, same as growth model).
        bicarb_consumed_mM = (max(0.0, delta_mass_mg) * 0.85 / 12.0) / self.volume_L  # C fixed via HCO3-
        co2_to_hco3_mg = max(0.0, co2_xfer_s * vol_s + co2_xfer_b * vol_b)  # CO2 absorbed from sparging
        # At current pH, fraction of newly dissolved CO2 that converts to HCO3- (Henderson-Hasselbalch equilibrium)
        f_to_hco3 = float(10.0 ** (self.ph - 6.35) / (1.0 + 10.0 ** (self.ph - 6.35)))
        bicarb_added_mM = (co2_to_hco3_mg / 44.0 / self.volume_L) * f_to_hco3
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
        self.bicarbonate = float(np.clip(self.bicarbonate - bicarb_consumed_mM + bicarb_added_mM, 0.0, 5.0))

        # pH via Henderson-Hasselbalch: pH = pKa1 + log10([HCO3-]/[CO2(aq)])
        # pKa1 temperature correction: -0.002/°C (symmetric around 25°C; Stumm & Morgan 1996)
        pKa1 = 6.35 - 0.002 * (self.temp - 25.0)
        # mg/L ÷ g/mol = mM (dimensional identity: mg/L × mol/g = 10^-3 mol/L = mM)
        co2_aq_mM = max(self.co2_b, 0.001) / 44.0
        ph_eq = float(np.clip(pKa1 + np.log10(max(self.bicarbonate, 0.001) / co2_aq_mM), 5.0, 11.0))
        # pH tracks CO2 dissolution rate (kLa ~1.5-5/h); 2.0/h gives ~30-min response — physically correct
        self.ph = float(np.clip(self.ph + 2.0 * self.dt * (ph_eq - self.ph), 5.0, 11.0))
        
        # --- Advanced Physics: Pigment & Salt ---
        
        # 4. Pigment Dynamics (Photo-inhibition & Chlorosis)
        # Bleaching: High Light (>1000) or Low Nitrogen (<100) damages pigment
        avg_light = I_surface * np.exp(-0.2 * self.reactor_depth/2) # Approx mid-depth light
        is_bleached = (avg_light > 1000.0) or (self.n_pool < 75.0)  # rescaled to Zarrouk's richer N baseline
        
        if is_bleached:
            self.pigment -= 0.01 * self.dt # Slow degradation
        else:
            self.pigment += 0.01 * self.dt # Slow recovery
        self.pigment = np.clip(self.pigment, 0.2, 1.0) # Min 20% pigment
        
        # 5. Salinity Accumulation
        lysis_mg      = max(0.0, -delta_mass_mg)
        salt_inflow_mg = total_uptake_mg * 0.1  # impurity carryover from nutrient feed
        salt_decay_mg  = lysis_mg * 0.5          # ion release from lysed cells
        self.salt += (salt_inflow_mg + salt_decay_mg) / max(self.volume_L, 1e-9)
        
        # --- Sensors ---
        
        # OD ~ Mass^0.8 (Self-Shading effect)
        # 1e11 cells ~ 1g/L ~ OD 1.0
        # density_gL = (total_mass_mg * 1e-9) / self.volume_L # BUG: 1e-9 is wrong units (pg->mg happened already)
        # turbidity = 1.0 * (density_gL ** 0.8)
        
        turbidity = self.od # Use the Linear Phsyics OD
        
        # 2. RGB Absorbance (Proxy for Chlorophyll)
        # Absorbance = Turbidity * Pigment_Health
        rgb_absorbance = turbidity * self.pigment
        
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # At D2, simulate slow natural evolution/mutation of the strain over weeks of deployment.
        # Every ~5 hours (250 steps at dt=0.02h), strain parameters wander by ±1%.
        # This forces the internal LMU to constantly adapt its latent state tracking.
        if self.difficulty >= 2 and self.step_count > 0 and self.step_count % 250 == 0:
            self.strain_params['mu_max'] *= np.random.uniform(0.99, 1.01)
            self.strain_params['Ks'] *= np.random.uniform(0.99, 1.01)
            self.strain_params['Ks_light'] *= np.random.uniform(0.99, 1.01)

        # 3. Conductivity — Kohlrausch molar conductance formula (µS/cm)
        # σ (mS/cm) = Σ λᵢ (S·cm²/mol) × cᵢ (mol/L), then ×1000 → µS/cm
        # λ values at 25°C (literature): NO₃⁻=71.4, Na⁺=50.1, HPO₄²⁻=57.0,
        # K⁺=73.5, SO₄²⁻=160.0, Na⁺=50.1, Cl⁻=76.4, OH⁻=198.0, H⁺=349.8

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
        conductivity = (sigma_n + sigma_p + sigma_ext + sigma_salt + sigma_ph + sigma_hco3) * cond_temp * 1000.0
        
        # Store for Observation (Overwrite OD with Turbidity, add others)
        # self.od = turbidity # Don't overwrite, maintain linear physics
        self.rgb_absorbance = rgb_absorbance
        self.conductivity = conductivity
        
        # --- Reward --- (see _compute_reward for the semi-continuous reward design)
        reward = self._compute_reward(delta_mass_mg, shock_factor, harvested_this_step_mg, is_harvest_event)

        # Always increment step_count so Monitor reports correct episode length on crash
        self.step_count += 1
        # Extinction check: population OR total biomass. Cells can hover just above the
        # per-cell starvation threshold (1e7 pg) without individually triggering death,
        # leaving a "zombie" culture of a few surviving cells with near-zero total mass —
        # this stalls the episode at flat negative reward (washout+stagnation) for
        # thousands of steps with no learning signal instead of ending the rollout.
        if self.num_active < 10 or total_mass_mg < 1.0:
            # Reduced from -1000: that scale was 300-1000x larger than typical achievable
            # per-episode reward (~10-20), so occasional exploration-driven crashes were
            # corrupting the LSTM's learned weights for millions of steps to recover from
            # (observed repeatedly as regression-recovery cycles during Spirulina training).
            # -100 still clearly signals "bad" without being catastrophically destabilizing.
            reward -= 100.0
            done = True
        else:
            done = self.step_count >= self.max_steps
            # No terminal bonus in semi-continuous mode — harvest yield (cumulative_harvested_mg)
            # accumulates continuously via reward_harvest each step (see _compute_reward).
        
        # Per-step debug trace. Gated behind ENV_DEBUG (default OFF) because it dominated every
        # log this project produced: 4,519 of 10,586 lines (43%) in a single chunk, multi-MB per
        # run, and every single grep across 25 runs needed `grep -v EnvDebug` to be readable.
        # Enable per-invocation with `ENV_DEBUG=1 python ...` when actually debugging physics.
        if ENV_DEBUG and ((self.step_count % 500 == 0) or done):
             # Use collected stats if available, else 0
             d_shock = getattr(self, 'debug_shock', 0.0)
             d_clump = getattr(self, 'debug_clump', 1.0)
             mean_x = np.mean(self.cells_x[self.active_mask]) if self.num_active > 0 else 0.0
             ratio = getattr(self, 'rgb_ratio', 0.0)
             turb = getattr(self, 'turbidity_obs', 0.0)

             if 'tqdm' in sys.modules:
                 from tqdm import tqdm
                 tqdm.write(f"[EnvDebug] Step: {self.step_count}, Active: {self.num_active}, Mass: {total_mass_mg:.2f}, OD: {self.od:.4f}, Turb: {turb:.4f}, pH: {self.ph:.2f}, Shock: {d_shock:.2f}, Clump: {d_clump:.2f}, MeanX: {mean_x:.2f}, RGB: {ratio:.2f}, Rew: {reward:.3f}, Done: {done}")
             else:
                 print(f"[EnvDebug] Step: {self.step_count}, Active: {self.num_active}, Mass: {total_mass_mg:.2f}, OD: {self.od:.4f}, Turb: {turb:.4f}, pH: {self.ph:.2f}, Shock: {d_shock:.2f}, Clump: {d_clump:.2f}, MeanX: {mean_x:.2f}, RGB: {ratio:.2f}, Rew: {reward:.3f}, Done: {done}")
             
        return self._get_obs(), float(reward), done, False, {
            "pop": self.num_active,
            "fouling": self.fouling_factor,
            "peak_od": float(getattr(self, 'max_historical_od', getattr(self, 'od', 0.0))),
            "od": float(getattr(self, 'od', 0.0)),
            "kLa_h-1": float(getattr(self, 'kLa', 0.0)),
            "dissolved_co2_mgL": float(getattr(self, 'dissolved_co2', 0.0)),
            "cumulative_harvested_mg": float(getattr(self, 'cumulative_harvested_mg', 0.0)),
            "time_avg_od": float(self.od_sum_back_half / max(self.od_count_back_half, 1)),
            "start_mode": getattr(self, 'episode_start_mode', 'low'),
            "reward_term_sums": dict(getattr(self, 'reward_term_sums', {})),
        }

