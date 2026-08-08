
import sys
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict

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
        self.reactor_depth = 0.30  # 30cm depth (30L fish-tank scale)
        self.reactor_width = 1.0
        self.volume_L = 30.0  # 30L reactor volume
        
        # --- Gas-Phase / Carbonate Configuration (closed 30L PBR) ---
        self.base_air_flow_lpm = 0.30        # Baseline air sparge (L/min)
        self.max_co2_flow_lpm = 0.12         # Max pure-CO2 injection authority (L/min)
        self.ambient_co2_frac = 420e-6       # Atmospheric CO2 mol fraction
        self.ambient_o2_frac = 0.209         # Atmospheric O2 mol fraction
        self.buffer_equilibrium_ph = 10.2    # Typical Zarrouk alkaline equilibrium
        self.co2_toxicity_Ki_mgL = 30.0      # Mild dissolved-CO2 toxicity onset
        self.co2_toxicity_hill = 2.0         # Hill exponent for toxicity curve
        
        # Action: [Stirring, Light, Nutrient, CO2]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation Space (6 Dims)
        # 1. Turbidity  2. pH  3. Nutrients  4. Temp  5. Conductivity  6. RGB
        # Note: dissolved_co2 is internal state only — agent reads it indirectly via pH
        # DO2 remains internal state in this observation design.
        self.observation_space = spaces.Box(
            low=np.array( [0.0, 0.0,    0.0, 0.0,     0.0, 0.0]),
            high=np.array([5000.0,14.0,5000.0,50.0,10000.0,20.0]),
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
        
        self.ext_nutrients = 150.0  # mg/L — inorganic salts / non-N nutrients (K, Mg)
        self.n_pool = 400.0         # mg N/L — primary nitrogen source (Zarrouk NO3-)
        self.p_pool = 80.0          # mg P/L — phosphorus (K2HPO4 in Zarrouk, N:P ~5:1 by mass)
        self.ph = self.buffer_equilibrium_ph
        self.do2 = 6.0
        self.temp = 25.0 # Ambient temperature
        self.time_t = 0.0 # Continuous time for turbulence
        
        # Hardware Smoothing State (EMA)
        self.current_stir_rpm = 50.0
        self.current_nut_flow = 0.0
        self.current_co2_flow = 0.0
        # Sensor-lag model: best case is 2 steps at high RPM; low RPM is slower.
        self._sensor_delay_min_steps = 2
        self._sensor_delay_max_steps = 8
        self._ph_obs_ema = self.ph
        self._temp_obs_ema = self.temp
        self.prev_action = np.zeros(4, dtype=np.float32)
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
            'mu_max': np.random.normal(0.05, 0.007), # Narrowed to match heavy
            'Ks': np.random.normal(20.0, 2.5), # Narrowed to match heavy
            'Ks_light': np.random.normal(100.0, 10.0), # Narrowed proportionally
            'Ki': np.random.normal(120.0, 15.0), # Narrowed to match heavy
            'Kii': np.random.normal(2000.0, 250.0), # Narrowed to match heavy
            'T_opt': np.random.normal(27.0, 1.0), # Narrowed to match heavy
            'Q_min': 0.5,    # Lowered from 1.5: f_Q was 0.40 at typical quota (~2.5), imposing 60% growth penalty
            'Q_max': 5.0,
            'tau_acclim': np.random.uniform(1.0, 4.0) # Acclimation time constant (hours)
        }
        self.strain_params['mu_max'] = max(0.05, self.strain_params['mu_max'])
        self.strain_params['Ks_light'] = max(50.0, self.strain_params['Ks_light'])
        self.strain_params['Ki'] = max(10, self.strain_params['Ki'])
        self.strain_params['T_opt'] = np.clip(self.strain_params['T_opt'], 20.0, 35.0)

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
        self.cells_quota[:self.num_active] = np.random.uniform(2.0, 4.0, self.num_active)
        # Initialize acclimation to average light (approx 200)
        self.cells_acclimation[:self.num_active] = np.random.uniform(100.0, 300.0, self.num_active)
        self.clump_mass[:self.num_active] = 1.0 # Start as single cells
        
        self.ext_nutrients = 150.0  # mg/L — inorganic salts / non-N nutrients
        self.n_pool = 400.0         # mg N/L — primary nitrogen source
        self.p_pool = 80.0          # mg P/L — phosphorus (Zarrouk K2HPO4, N:P ~5:1)
        self._prev_n_pool = self.n_pool
        self.ph = self.buffer_equilibrium_ph
        self.do2 = 7.0
        self.temp = 25.0
        self.step_count = 0
        self.time_t = 0.0
        self.fouling_factor = 0.0 # Biofouling accumulation
        
        # Reset Hardware Smoothing
        self.current_stir_rpm = 50.0
        self.current_nut_flow = 0.0
        self.current_co2_flow = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        # Reset Advanced Physics
        self.salt = 1000.0
        self.pigment = 1.0
        self.dissolved_co2 = 2.0  # mg/L dissolved inorganic carbon in alkaline medium
        
        # --- Sim-to-Real Sensor Drift & Lag (D1+) ---
        if self.difficulty >= 1:
            # Per-episode static calibration error (±5% uniform random drift)
            self._sensor_drift_mult = np.random.uniform(0.95, 1.05, size=6)
            # Sensors: [Turbidity, pH, Nutrients, Temp, Conductivity, RGB]
            # Dynamic lag state for RPM-coupled pH/temperature observation smoothing.
            self._ph_obs_ema = self.ph
            self._temp_obs_ema = self.temp
        else:
            self._sensor_drift_mult = np.ones(6, dtype=np.float32)
            self._ph_obs_ema = self.ph
            self._temp_obs_ema = self.temp
            
        # --- Initialize derived state (prevents stale-value NaN on episode 2+) ---
        init_total_mass = np.sum(self.cells_mass[:self.num_active])
        self.last_mass = init_total_mass
        self.od = (init_total_mass * 1e-9 / self.volume_L) / 300.0
        # B6: initialize sensor state here so _get_obs() never reads stale episode values
        self.conductivity = 1500.0
        self.rgb_absorbance = 0.0
        self.last_hourly_od = float(self.od)
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
        self._prev_dic_err = None
        
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

            # Bubble/flow-induced noise (RPM-dependent)
            # Real sensors show high-frequency noise from bubbles and turbulent eddies
            rpm = float(getattr(self, 'current_stir_rpm', 50.0))
            flow_noise = 1.0 + 0.03 * (rpm / 200.0) * np.random.normal(0, 1)

            # Convert to raw NTU units (0-5000 range for sim-to-real transfer)
            # Add base sensor noise floor (Difficulty scaled)
            noise_scale = 0.05
            turbidity_obs = max(0.0, (turbidity_base * 1000.0 * flow_noise) + np.random.normal(0, noise_scale))
            self.turbidity_obs = turbidity_obs  # Store for debug logging
        else:
            self.od = 0.0
            self.conductivity = 0.0
            self.rgb_absorbance = 0.0
            turbidity_obs = 0.0
            self.turbidity_obs = turbidity_obs
            if not hasattr(self, 'max_historical_od'): self.max_historical_od = 0.0

        base_obs = np.array([
            turbidity_obs,
            self.ph,
            self.n_pool,          # O3: nitrogen pool (mg N/L) replaces generic nutrients
            self.temp,
            self.conductivity,
            self.rgb_absorbance
        ], dtype=np.float32)

        # Stochastic Sensor Noise: High-frequency jitter
        # D0 gets minor ±1% jitter, D1+ gets realistic ±2% jitter
        jitter_mag = 0.02 if self.difficulty >= 1 else 0.01
        jitter = np.random.uniform(1.0 - jitter_mag, 1.0 + jitter_mag, size=(6,))
        
        # Dynamic RPM-coupled sensor lag for pH and temperature (D1+ only).
        # Better mixing (high RPM) -> faster response (~2-step effective lag).
        # Poor mixing (low RPM) -> slower response (up to ~8-step effective lag).
        if self.difficulty >= 1:
            rpm = float(np.clip(getattr(self, 'current_stir_rpm', 50.0), 50.0, 200.0))
            mix_quality = (rpm - 50.0) / 150.0
            lag_span = self._sensor_delay_max_steps - self._sensor_delay_min_steps
            lag_steps = int(round(self._sensor_delay_max_steps - (lag_span * mix_quality)))
            lag_steps = int(np.clip(lag_steps, self._sensor_delay_min_steps, self._sensor_delay_max_steps))

            # EMA coefficient corresponding to an N-step smoothing horizon.
            alpha = 2.0 / (lag_steps + 1.0)

            self._ph_obs_ema = (1.0 - alpha) * self._ph_obs_ema + alpha * base_obs[1]
            self._temp_obs_ema = (1.0 - alpha) * self._temp_obs_ema + alpha * base_obs[3]

            base_obs[1] = self._ph_obs_ema
            base_obs[3] = self._temp_obs_ema
            
        # Apply combined Jitter * Static Drift
        noisy_obs = base_obs * jitter * self._sensor_drift_mult
        
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

    def step(self, action):
        # --- Safety Checks ---
        # 0. Check for invalid actions
        if np.any(np.isnan(action)):
            # If action is NaN (e.g. model output is corrupted), default to safe values
            action = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        action_vec = np.asarray(action, dtype=np.float32)
        action_delta = action_vec - self.prev_action
        action_smooth_penalty = self.action_smooth_coef * float(np.sum(action_delta ** 2))
        self.prev_action = action_vec.copy()

        stir_act, light_act, nut_act, co2_act = action
        
        # 1. Decode Target Actions
        target_stir_rpm = np.interp(stir_act, [-1, 1], [50, 200])
        target_I_surface = np.interp(light_act, [-1, 1], [0, 2000])
        target_nut_flow  = np.interp(nut_act,   [-1, 1], [0, 100])
        target_co2_flow  = np.interp(co2_act,   [-1, 1], [0, self.max_co2_flow_lpm * 1000.0])  # mL/min pure CO2 injection
        
        # 2. Hardware Smoothing (EMA)
        # EMA alphas doubled for dt=0.02h to preserve the same physical lag time constants
        alpha_nut = 0.12   # same ~0.17h lag as 0.06 at dt=0.01h
        alpha_co2 = 0.30   # same ~0.07h lag as 0.15 at dt=0.01h
        self.current_stir_rpm = (0.90 * self.current_stir_rpm) + (0.10 * target_stir_rpm)
        self.current_nut_flow = (1.0 - alpha_nut) * self.current_nut_flow + alpha_nut * target_nut_flow
        self.current_co2_flow = (1.0 - alpha_co2) * self.current_co2_flow + alpha_co2 * target_co2_flow
        
        stir_rpm  = self.current_stir_rpm
        I_surface = target_I_surface  # Light changes instantly
        nut_flow  = self.current_nut_flow
        co2_flow  = self.current_co2_flow
        
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
            fouling_rate = max(0.0, 1.0 - stir_rpm / 200.0) * self.od * 0.0002
            self.fouling_factor += fouling_rate * self.dt
            self.fouling_factor = float(np.clip(self.fouling_factor, 0.0, 0.5))

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
            v_max_z = 0.01 * mix_intensity 
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

        prev_num_active = self.num_active
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
            # Cells adapt to the TOTAL light they see
            # k_accum = 0.1 (Fast integration)
            self.cells_acclimation[self.active_mask] += 0.1 * (cells_I_total - self.cells_acclimation[self.active_mask])
            I_effective = self.cells_acclimation[self.active_mask]
            
            # 2. Temperature Factor (Gaussian)
            temp_factor = np.exp(-0.5 * ((self.temp - params['T_opt'])/5.0)**2)
            
            # Photo-Inhibition / Shock
            # Cells experience stress when light changes suddenly
            # Scalar reduced from 0.0001 to 0.000001 to prevent startup death
            # At diff=300: Old penalty=99.99%, New penalty=9% (survivable!)
            diff = (cells_I_total - I_effective)
            shock_factor = np.exp(-0.000003 * (diff**2))  # 3e-6: 24% penalty at diff=300 (was 9%)

            # --- Oxygen Toxicity (ROS Damage) ---
            if self.difficulty >= 1:
                # Soften O2 toxicity for D1 (matches thermal phys_scale progression)
                phys_scale = 0.75 if self.difficulty == 1 else 1.0
                f_O2 = max(0.0, 1.0 - ((self.do2 / 22.0)**4) * phys_scale)  # 22 mg/L threshold: inhibition starts ~16 mg/L
            else:
                f_O2 = 1.0
            
            # 3. Growth Rate (Haldane)
            # Growth is driven by RED light availability
            # Inhibition is driven by TOTAL light intensity
            # f_I = I_growth / (Ks + I_growth + I_total^2/Ki)
            
            # Parameters
            Ks_I = params['Ks_light'] # Use dynamic param (avg 100.0)
            Ki_I = 2500.0 # Inhibition threshold (Total Light)
            
            f_I = cells_I_growth / (Ks_I + cells_I_growth + (cells_I_total**2 / Ki_I))
            f_I = np.nan_to_num(f_I)
            
            # Normalize to 0-1 approx for consistency with old model
            # Max possible ~ 0.8
            f_I = f_I * 1.5 
            f_I = np.clip(f_I, 0.0, 1.5)

            # DEBUG: Save RGB Ratio for observation
            avg_red = np.mean(I_red)
            avg_blue = np.mean(I_blue) if np.mean(I_blue) > 0.001 else 1.0
            self.rgb_ratio = avg_red / avg_blue
            
            # Droop Quota
            # Only update active quotas
            current_quotas = self.cells_quota[self.active_mask]
            
            f_Q = np.maximum(0.0, 1.0 - params['Q_min'] / (current_quotas + 1e-6))
            
            # pH Inhibition (Asymmetric Gaussian — Spirulina alkaliphile)
            # Peak at 9.5 (true optimum per Richmond 1988; Vonshak 1997; Habib FAO 2008)
            # Acid side: steep drop (σ=1.2) — Spirulina intolerant of low pH
            # Alkaline side: gentle drop (σ=2.0) — obligate alkaliphile tolerates high pH well
            if self.ph <= 9.5:
                f_pH = np.exp(-0.5 * ((self.ph - 9.5) / 1.2) ** 2)
            else:
                f_pH = np.exp(-0.5 * ((self.ph - 9.5) / 2.0) ** 2)
            
            # Osmotic Stress — total dissolved solutes (N + inorganic salts)
            total_dissolved = self.n_pool + self.ext_nutrients
            if total_dissolved > 12000.0:
                f_Osmosis = np.exp(-0.5 * ((total_dissolved - 12000.0) / 3000.0) ** 2)
            else:
                f_Osmosis = 1.0
                
            # --- DEBUG STATS CAPTURE ---
            if self.step_count % 100 == 0 and self.num_active > 0:
                  self.debug_f_I = np.mean(f_I)
                  
                  
                  self.debug_f_Q = np.mean(f_Q)
                  self.debug_f_pH = f_pH
                  self.debug_f_O2 = f_O2
                  self.debug_shock = np.mean(shock_factor) # shock_factor calculated on active_mask in line 260
                  self.debug_clump = np.mean(self.clump_mass[self.active_mask])
            # ---------------------------
            # ---------------------------
            
            # Logistic Hard Limit REMOVED
            # Natural Limits (Gas Transfer Failure) handle carrying capacity now.
            limit_factor = 1.0 
            
            # Calculate Rate
            # --- Shear Repair Tax (sigmoid, centered at 175 RPM) ---
            # Steep onset above 175 RPM matches real shear fragmentation threshold.
            # Max 35% penalty at sustained >200 RPM.
            repair_factor = 1.0 / (1.0 + np.exp(-0.12 * (stir_rpm - 175.0)))
            repair_tax = 1.0 - (0.35 * repair_factor)
            
            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Hydrodynamic shear stress on Spirulina trichomes accumulates over time.
            # Onset at ~80 RPM (Kolmogorov eddy scale for 30L tank).
            # ~5h time constant: equilibrium integrity ≈ 0.5 at sustained 150 RPM.
            # 15% max growth penalty at full degradation (0.0 integrity).
            shear_stress = float(np.clip((stir_rpm - 80.0) / 70.0, 0.0, 1.0))
            self.membrane_integrity -= shear_stress * 0.001          # ~5h to degrade at max shear
            self.membrane_integrity += (1.0 - self.membrane_integrity) * 0.002  # ~5h to recover
            self.membrane_integrity = float(np.clip(self.membrane_integrity, 0.0, 1.0))
            fatigue_tax = 1.0 - (0.15 * (1.0 - self.membrane_integrity))

            # Phosphorus-Limited Growth (Monod on p_pool, Ks_P=5 mg P/L)
            # At p_pool=80: f_P=0.94 (replete). At p_pool=5: f_P=0.50 (Ks). At p_pool=1: f_P=0.17 (severe).
            f_P = self.p_pool / (5.0 + self.p_pool)
            f_P = float(np.clip(f_P, 0.0, 1.0))

            # Carbon-Limited Growth (Monod saturation on dissolved CO2)
            # At dissolved_co2 = 0.5 mg/L: f_carbon = 0.50
            # At dissolved_co2 = 2.0 mg/L: f_carbon = 0.80
            Kc_CO2   = 0.5
            f_carbon = getattr(self, 'dissolved_co2', 2.0) / (Kc_CO2 + getattr(self, 'dissolved_co2', 2.0))

            # Explicit dissolved-CO2 toxicity (separate from pH-mediated inhibition).
            f_CO2_tox = 1.0 / (1.0 + (getattr(self, 'dissolved_co2', 2.0) / (self.co2_toxicity_Ki_mgL + 1e-9)) ** self.co2_toxicity_hill)
            f_CO2_tox = float(np.clip(f_CO2_tox, 0.0, 1.0))
            self.debug_f_CO2 = f_CO2_tox

            current_mu = params['mu_max'] * f_I * f_Q * f_P * f_carbon * f_CO2_tox * temp_factor * shock_factor * f_O2 * f_pH * f_Osmosis * limit_factor * repair_tax * fatigue_tax
            current_mu = np.clip(current_mu, 0.0, 5.0) 
            
            # --- Maintenance Respiration ---
            # 1.0% of mu_max: tighter than before but avoids catastrophic death spiral
            m_respiration = 0.010 * params['mu_max']
            
            # Net Growth Rate = Photosynthesis - Respiration
            # This can be negative (mass loss) if light/nutrients are insufficient!
            net_mu = current_mu - m_respiration
            
            # Grow Biomass
            growth_mult = np.exp(net_mu * self.dt)
            # Clip multiplier to avoid single-step explosion (both up and down)
            growth_mult = np.clip(growth_mult, 0.5, 2.0)
            
            self.cells_mass[self.active_mask] *= growth_mult
            
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
            lysis_rate  = 1e-4 + (2e-3 * (stress_factor ** 2))  # per hour
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
            starving_mask = self.active_mask & (self.cells_mass < 5e5)
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

            # 1 agent = 2.5M cells; 0.05 gives realistic episode-scale N drain
            total_uptake_mg = uptake_amount * self.num_active * 0.05
            # P uptake: Monod saturation, 0.005 scalar → ~56h depletion at 300 agents (matches N timescale)
            p_uptake_rate = self.p_pool / (5.0 + self.p_pool)
            total_p_uptake_mg = p_uptake_rate * self.dt * self.num_active * 0.005
            # nut_flow: 75% N, 15% P, 10% inorganic salts (Zarrouk N:P ~5:1 by mass)
            self.n_pool        = max(0.0, self.n_pool        - total_uptake_mg    + (nut_flow * 0.75 * self.dt))
            self.p_pool        = max(0.0, self.p_pool        - total_p_uptake_mg  + (nut_flow * 0.15 * self.dt))
            self.ext_nutrients = max(0.0, self.ext_nutrients                      + (nut_flow * 0.10 * self.dt))
            
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
                    self.cells_mass[child_indices] = self.cells_mass[parent_indices] # Start with half mass
                    self.cells_z[child_indices] = self.cells_z[parent_indices]     # Inherit position Z
                    self.cells_x[child_indices] = self.cells_x[parent_indices]     # Inherit position X
                    self.cells_quota[child_indices] = self.cells_quota[parent_indices] # Inherit quota status
                    self.cells_acclimation[child_indices] = self.cells_acclimation[parent_indices] # Inherit acclimation
                    
                    self.clump_mass[child_indices] = 1.0 # Children start as single cells
                    # self.clump_mass[parent_indices] = 1.0 # Optional: Parents also disperse? No, let them stay stuck.
                    
                    self.num_active += n_spawns

                # B9: when all slots are full, cap oversized cells at the division threshold
                # so they stop accumulating mass and wasting quota each step.
                if n_slots == 0:
                    self.cells_mass[ready_to_divide] = np.minimum(
                        self.cells_mass[ready_to_divide], 1.4e8
                    )

        else:
            n_spawns = 0
            current_mu = np.zeros(1) # fallback
            f_Q = np.zeros(1)

        # --- Environmental Dynamics (Macro) ---

        if self.step_count % 100 == 0:
            # Rebalance array occasionally if needed (not strictly necessary with this mask implementation)
            pass

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
        flow_resistance = 1.0 + ((od / 30.0)**2) * (avg_clump ** 0.5)
        
        co2_flow_lpm = co2_flow / 1000.0
        total_gas_lpm = max(1e-6, self.base_air_flow_lpm + co2_flow_lpm)
        co2_frac = ((self.ambient_co2_frac * self.base_air_flow_lpm) + co2_flow_lpm) / total_gas_lpm
        co2_frac = float(np.clip(co2_frac, self.ambient_co2_frac, 0.12))
        o2_frac = float(np.clip((self.ambient_o2_frac * self.base_air_flow_lpm) / total_gas_lpm, 0.05, self.ambient_o2_frac))

        # kLa correlation tuned for 30L sparged tank (units: 1/hour).
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
            
        # Update OD — normalised by volume (concentration, not total mass)
        self.od = (total_mass_mg / self.volume_L) / 300.0
        # print(f"DEBUG: Mass={total_mass_mg}, OD={self.od}")
            
        o2_production = delta_mass_mg * 1.2 # mg O2 produced
        
        # Gas transfer toward gas-phase equilibrium (temperature fixed at 25C baseline).
        # O2 solubility scales with O2 partial pressure in the sparge/headspace mix.
        o2_sat = 8.0 * (o2_frac / self.ambient_o2_frac)
        o2_transfer = k_La * (o2_sat - self.do2) * self.dt
        
        self.do2 += (o2_production / self.volume_L) + o2_transfer
        self.do2 = np.clip(self.do2, 0.0, 30.0) # Cap at realistic supersaturation
        
        # 3. DIC-Driven pH (alkaline carbonate model)
        #
        # O1: In alkaline Zarrouk medium (pH 9.5-11), virtually all injected CO2 converts
        # to bicarbonate/carbonate immediately.  The equilibrium DIC scales with CO2 partial
        # pressure but with square-root dampening from the high-alkalinity buffer capacity.
        # At atmospheric CO2: co2_sat ≈ 2 mg/L DIC.  At max injection (~29% CO2): ≈ 52 mg/L.
        co2_ppm_factor = co2_frac / self.ambient_co2_frac   # enrichment relative to atmosphere
        co2_sat = float(np.clip(2.0 * (co2_ppm_factor ** 0.5), 0.5, 60.0))
        co2_transfer = k_La * (co2_sat - self.dissolved_co2) * self.dt

        # Stoichiometric carbon balance: photosynthesis consumes DIC, respiration releases it.
        co2_uptake_mg = max(0.0, delta_mass_mg) * 1.8
        co2_release_mg = max(0.0, -delta_mass_mg) * 0.8
        co2_bio_delta = np.clip((co2_release_mg - co2_uptake_mg) / self.volume_L, -1.0, 1.0)

        self.dissolved_co2 = float(np.clip(
            self.dissolved_co2 + co2_transfer + co2_bio_delta, 0.0, 80.0))

        # O2: pH driven by Henderson-Hasselbalch — replaces the ad hoc blend.
        # Higher DIC → more bicarbonate → pH drops from alkaline baseline.
        # No CO2 + active photosynthesis → DIC depletes → pH rises naturally.
        ph_from_dic = self.buffer_equilibrium_ph - 0.8 * np.log10(max(self.dissolved_co2, 0.01) / 2.0)
        ph_rise     = max(0.0,  delta_mass_mg / self.volume_L) * 0.015   # photosynthesis → pH↑
        ph_drop_resp= max(0.0, -delta_mass_mg / self.volume_L) * 0.010   # respiration    → pH↓
        ph_restore  = (self.buffer_equilibrium_ph - self.ph) * 0.08 * self.dt
        new_ph = float(np.clip(
            self.ph + 0.15 * (ph_from_dic - self.ph) * self.dt + ph_rise - ph_drop_resp + ph_restore,
            7.2, 11.5
        ))
        self.ph = new_ph if np.isfinite(new_ph) else self.buffer_equilibrium_ph
        
        # --- Advanced Physics: Pigment & Salt ---
        
        # 4. Pigment Dynamics (Photo-inhibition & Chlorosis)
        # Bleaching: High Light (>1000) or Low Nitrogen (<100) damages pigment
        avg_light = I_surface * np.exp(-0.2 * self.reactor_depth/2) # Approx mid-depth light
        is_bleached = (avg_light > 1000.0) or (self.n_pool < 50.0)  # N starvation triggers chlorosis
        
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

        # 3. Conductivity (Realistic Formula)
        # C = (Salt + Nutrients + |7-pH|) * Temp_Factor
        cond_base = self.salt + self.n_pool + self.p_pool + self.ext_nutrients + 100.0 * abs(7.0 - self.ph)
        cond_temp = 1.0 + 0.02 * (self.temp - 25.0)
        conductivity = cond_base * cond_temp
        
        # Store for Observation (Overwrite OD with Turbidity, add others)
        # self.od = turbidity # Don't overwrite, maintain linear physics
        self.rgb_absorbance = rgb_absorbance
        self.conductivity = conductivity
        
        # --- Reward (Sim2Real Proxy Tuning) ---
        # In a real physical PBR, we cannot measure true mass on every step.
        # The RL agent must learn to optimize growth using *only* high-frequency sensor proxies.
        
        # 1. DO2 Production Proxy (Photosynthesis)
        # We calculated 'o2_production' on line 602 as the raw biological O2 exhaust.
        # In reality, the agent reads `self.do2` and subtracts expected `k_La` off-gassing.
        # Here we directly use the simulated O2 production to train the proxy behavior.
        reward_do2 = o2_production * 0.1  # Scale to ~0.01 to 0.1 per step during growth
        
        # 2. Nitrogen Consumption Proxy (near-instant growth signal)
        prev_nut = getattr(self, '_prev_n_pool', self.n_pool)
        reward_nut_consume = max(0.0, (prev_nut - self.n_pool)) * 0.02
        self._prev_n_pool = self.n_pool
        
        # 2. pH Drift Proxy (Carbon Uptake)
        # If CO2 is OFF, and pH rises, cells are actively growing.
        # Base pH drift without biology is 0. Biological growth adds delta_ph_bio (calculated line 629)
        # Keep legacy magnitude while making the scale explicitly volume-normalised.
        delta_ph_bio = (delta_mass_mg / max(self.volume_L, 1e-9)) * 0.02
        co2_scale = np.clip(1.0 - (co2_flow / (self.max_co2_flow_lpm * 1000.0 + 1e-9)), 0.0, 1.0)
        reward_ph = delta_ph_bio * 50.0 * co2_scale

        # 2b. Carbon transfer progress proxy (fractions-aware)
        # Gives short-horizon credit when dissolved CO2 moves toward a soft target band,
        # helping credit assignment through gas-fraction and kLa transfer delays.
        dic_target = float(np.clip(2.0 + 0.25 * co2_sat, 1.5, 12.0))
        dic_err = abs(self.dissolved_co2 - dic_target)
        prev_dic_err = self._prev_dic_err if self._prev_dic_err is not None else dic_err
        dic_progress = np.clip(prev_dic_err - dic_err, -0.2, 0.2)
        self._prev_dic_err = dic_err
        if self.difficulty == 0:
            dic_scale = 0.12
        elif self.difficulty == 1:
            dic_scale = 0.06
        else:
            dic_scale = 0.03
        reward_dic = dic_progress * dic_scale
        
        # Direct biomass term scaled by per-cell growth (population-normalized)
        # Per-cell growth: biomass per individual cell per step
        per_cell_growth = (delta_mass_mg / (self.num_active + 1e-6)) * 1000
        reward_biomass = 0.50 * np.tanh(per_cell_growth / 5.0)  # /5.0: normalised for super-agent 10× mass per particle

        # Population boost: smaller populations get higher reward multiplier for same per-cell growth
        # At 100: 1.83×, At 300: 1.5×, At 600: 1.0×, At 900+: 1.0×
        pop_boost = 1.0 + max(0.0, 1.0 - (self.num_active / 600.0))
        reward_biomass *= pop_boost

        # Low OD emphasis: extra boost in lag phase
        if self.od < 0.05:
            reward_biomass *= 1.5
        pop_loss = max(0, prev_num_active - self.num_active)
        reward_lysis = -0.01 * float(pop_loss)

        # 3. High Water Mark OD Anchor (Latency-Tolerant)
        # Guarantees dense reward only when the agent strictly breaks the all-time high OD.
        reward_od = 0.0
        if not hasattr(self, 'max_historical_od'):
            self.max_historical_od = self.od
        if not hasattr(self, 'steps_since_od_high'):
            self.steps_since_od_high = 0
            
        if self.od > self.max_historical_od:
            delta_od = self.od - self.max_historical_od
            # Population-aware OD anchor: boost reward multiplier at low pop (harder to achieve growth)
            # At 100: 2.0×, At 300: 1.33×, At 600: 1.0×, At 900: 1.0×
            pop_boost = min(2.0, max(1.0, 400.0 / (self.num_active + 1e-6)))
            reward_od = np.tanh(delta_od * 1000.0) * 2.0 * pop_boost  # ↑ 4x: breaking OD record is the primary objective
            self.max_historical_od = self.od
            self.steps_since_od_high = 0
        else:
            self.steps_since_od_high += 1
            
        # 4. OD Growth Rate Reward & Options B Stagnation
        # Rewards any positive d(OD)/dt — dense, immediate signal for active growth.
        prev_od = getattr(self, '_prev_od_for_rate', self.od)
        od_rate = (self.od - prev_od) / (self.dt + 1e-9)  # OD units per hour
        self._prev_od_for_rate = float(self.od)
        
        reward_stagnation = 0.0
        # Proportional stagnation: severity scales with how much mass is lost, not binary.
        # Floor of 0.1 ensures any decline is noticed; ceiling of 1.0 caps at -0.15 for large crashes.
        # Threshold 0.5 mg/step ≈ 1.3% of a healthy 300-agent culture at full mass.
        if delta_mass_mg < -0.0001:
            severity = float(np.clip(abs(delta_mass_mg) / 0.5, 0.1, 1.0))
            reward_stagnation = -0.15 * severity
        # Population boost for growth rate (at 100: 1.83×, at 600: 1.0×)
        pop_factor = 1.0 + max(0.0, 1.0 - (self.num_active / 600.0))
        reward_growth_rate = max(0.0, od_rate) * 100.0 * pop_factor  # ↑ 5x: active d(OD)/dt is the densest growth signal
            
        mean_shock = np.mean(shock_factor) if self.num_active > 0 else 1.0
        mean_clump = np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0
        self.debug_clump = float(mean_clump)  # expose to EnvDebug log
        
        penalty_shock = (1.0 - mean_shock) * -0.05
        penalty_clump = (mean_clump - 1.0) * -0.01
        
        # Total Proxy Reward + Metabolic Momentum
        mean_f_Q = float(np.mean(f_Q)) if self.num_active > 0 else 0.0
        reward = reward_do2 + reward_nut_consume + reward_ph + reward_dic + reward_biomass + reward_lysis + reward_od + reward_growth_rate + reward_stagnation + (mean_f_Q * 0.005)  # ↓ 10x: metabolic quota is tie-breaker, not objective
        reward -= action_smooth_penalty
        
        # 5. Potential-Based Reward Shaping (PBRS)
        # Φ(s) = future growth capacity — maintained nutrient/DIC buffers + O2 headroom + quota
        # F(s,s') = γΦ(s') - Φ(s)  — provably policy-invariant, reshapes landscape without bias
        gamma_pbrs = 0.99
        mean_f_Q_cur = float(np.mean(np.maximum(0.0, 1.0 - 0.5 / (self.cells_quota[self.active_mask] + 1e-6)))) if self.num_active > 0 else 0.0
        phi_cur = (
            0.3 * min(self.n_pool / 400.0, 1.0) +                           # nitrogen buffer
            0.3 * float(np.exp(-0.5 * ((self.dissolved_co2 - 4.0) / 3.0) ** 2)) +  # DIC in safe window (peak 4 mg/L)
            0.2 * float(np.exp(-0.5 * ((self.do2 - 8.0) / 5.0) ** 2)) +      # O2 headroom: Gaussian peak at 8 mg/L
            0.2 * mean_f_Q_cur                                                # cell quota health
        )
        phi_prev = self._phi_prev if self._phi_prev is not None else phi_cur
        reward_pbrs = gamma_pbrs * phi_cur - phi_prev
        self._phi_prev = phi_cur
        reward += reward_pbrs * 0.5  # Scaled so PBRS doesn't dominate OD anchor
        
        if self.difficulty == 0:
            reward += penalty_shock + penalty_clump

        # ── Phase 0 Soft Reward Shaping (Difficulty 0 only) ──────────────────
        # At Difficulty 0, add small Gaussian bonuses proportional to sensor
        # proximity to known Spirulina optima. This scaffolds early learning by
        # giving the agent dense micro-signal without a single correct answer.
        # These bonuses are DISABLED at Difficulty 1+ so the agent generalises.
        if self.difficulty == 0:
            # pH shaping: Gaussian peak at pH 9.5 (Spirulina optimum), σ=1.0
            # ↓ 10x scaling: D0 shaping is scaffolding only, not a primary reward source
            ph_shape = np.exp(-0.5 * ((self.ph - 9.5) / 1.0) ** 2) * 0.015

            # DO2 shaping: Reward keeping O2 in the aerobic sweet spot (5-15 mg/L)
            # Peak at 10mg/L, penalises anoxia (<5) and O2 toxicity (>20)
            do2_shape = np.exp(-0.5 * ((self.do2 - 10.0) / 4.0) ** 2) * 0.010

            # Temperature shaping: Gaussian peak at T_opt (strain-specific)
            t_opt = self.strain_params.get('T_opt', 27.0)
            temp_shape = np.exp(-0.5 * ((self.temp - t_opt) / 3.0) ** 2) * 0.010

            # Fade out shaping linearly over 3600 steps (72h at dt=0.02h) to force transition to biomass growth
            fade_multiplier = max(0.0, 1.0 - (getattr(self, 'step_count', 0) / 3600.0))
            reward += (ph_shape + do2_shape + temp_shape) * fade_multiplier

        elif self.difficulty in (1, 2):
            # Weaker pH shaping persists at D1/D2 so agent never fully loses
            # the pH→growth gradient signal across curriculum phases.
            # DO2 and temp shaping are omitted: O2 toxicity (f_O2) and f_pH
            # in the growth model already provide implicit signals for those.
            ph_shape = np.exp(-0.5 * ((self.ph - 9.5) / 1.0) ** 2) * 0.005  # ↓ 10x: weak tie-breaker gradient only
            reward += ph_shape
        # ──────────────────────────────────────────────────────────────────────

        # Safety/Stability Penalties
        if self.ph > 11.0 or self.ph < 7.0:
            reward -= 0.1 # Small continuous penalty for toxic pH bounds
        
        # Keep tracking internal true mass for logging, but not for reward
        total_mass = np.sum(self.cells_mass[self.active_mask]) if self.num_active > 0 else 0
        self.last_mass = total_mass
        
        # Final safeguard on reward
        if not np.isfinite(reward):
            reward = -10.0 # Punishment for breaking physics
        
        # Always increment step_count so Monitor reports correct episode length on crash
        self.step_count += 1
        if self.num_active < 10:
            reward -= 1000.0
            done = True
        else:
            done = self.step_count >= self.max_steps
            if done:
                # Terminal bonus anchors the value function to end-of-episode OD.
                # Not given for crash episodes — only for completing the full 144h batch.
                reward += np.tanh(self.od / 0.05) * 10.0
        
        # Debug Print
        if (self.step_count % 500 == 0) or done:
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
            "co2_flow_ml_min": float(getattr(self, 'current_co2_flow', 0.0)),
            "start_mode": getattr(self, 'episode_start_mode', 'low'),
        }

