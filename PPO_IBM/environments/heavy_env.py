import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict

class HeavyPhotobioreactorEnv(gym.Env):
    """
    Individual-Based Model (IBM) Photobioreactor Environment (Simplified 1D/Heavy).
    Tracks N individual algal cells as particles in 1D depth (z-axis).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_cells: int = 300000, initial_cells: int = 3000, difficulty: int = 2):
        super(HeavyPhotobioreactorEnv, self).__init__()
        self.difficulty = difficulty
        
        # --- CONFIGURATION ---
        self.max_cells = max_cells
        self.initial_cells = initial_cells
        self.num_active = initial_cells
        self.dt = 0.01  # Time step (0.01h) determines simulation resolution
        self.reactor_depth = 0.30  # 30cm depth (30L fish-tank scale)
        self.volume_L = 30.0       # 30L reactor volume
        
        # Action: [Stirring, Light, Nutrient, CO2]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation Space (6 Dims)
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-28)
        self.observation_space = spaces.Box(
            low=np.array( [0.0, 0.0,    0.0, 0.0,     0.0, 0.0]),
            high=np.array([5000.0,14.0,5000.0,50.0,10000.0,20.0]),
            dtype=np.float32
        )
        
        # --- VECTORIZED STATE ARRAYS ---
        # Fixed size arrays, managed via active_mask
        self.cells_z = np.zeros(self.max_cells, dtype=np.float32) 
        self.cells_mass = np.zeros(self.max_cells, dtype=np.float32)
        self.cells_quota = np.zeros(self.max_cells, dtype=np.float32)
        # Hysteresis: Photo-acclimation state (tracks recent light exposure)
        self.cells_acclimation = np.zeros(self.max_cells, dtype=np.float32)
        
        # Boolean mask: True = Active Living Cell
        self.active_mask = np.zeros(self.max_cells, dtype=bool)
        
        self.ext_nutrients = 500.0
        self.ph = 8.5
        self.do2 = 6.0
        self.temp = 25.0 # Ambient temperature
        self.time_t = 0.0 # Continuous time for turbulence
        
        # Advanced Physics State
        self.salt = 1000.0 # mg/L
        self.pigment = 1.0 # Health 0-1
        self.membrane_integrity = 1.0
        
        # Flocculation State (Clumping)
        self.clump_mass = np.ones(self.max_cells, dtype=np.float32)
        
        # Debug Stats
        self.debug_mu = 0.0
        self.debug_stress = 0.0
        self.debug_shock = 0.0
        self.debug_clump = 1.0
        
        # --- GENETIC PARAMS (Placeholder, set in reset) ---
        self.strain_params = {}
        
        self.step_count = 0
        self.max_steps = 24000 # 240h (approx 10 days) to see full growth curve
        # Actually with dt=0.01h, 14400 steps = 144 hours.

    def set_difficulty(self, difficulty: int) -> None:
        """Set curriculum difficulty level (0=easy, 1=medium, 2=hard)."""
        self.difficulty = int(np.clip(int(difficulty), 0, 2))

    def get_difficulty(self) -> int:
        return int(self.difficulty)
        
    def _randomize_strain(self):
        """Generates a unique 'Strain' of algae for this episode."""
        self.strain_params = {
            'mu_max': np.random.normal(0.05, 0.007), # Genetic is 0.01
            'Ks': np.random.normal(20.0, 2.5),      # Genetic is 5.0
            'Ki': np.random.normal(120.0, 15.0),    # Genetic is 30.0
            'Kii': np.random.normal(2000.0, 250.0), # Genetic is 500.0
            'T_opt': np.random.normal(27.0, 1.0),   # Genetic is 2.0
            'Q_min': 0.5,    
            'Q_max': 5.0,
            'tau_acclim': np.random.uniform(1.0, 4.0) 
        }
        self.strain_params['mu_max'] = max(0.05, self.strain_params['mu_max'])
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
        # Super-Agent Scaling: 1 Agent = 250,000 Cells (~50pg each)
        # Density-dependent starting mass:
        density_ratio = np.clip(self.initial_cells / 150000.0, 0.0, 1.0)
        starting_mass = 1.25e7 - (density_ratio * 0.45e7) 
        self.cells_mass[:self.num_active] = np.random.normal(starting_mass, 1e5, self.num_active)
        self.cells_quota[:self.num_active] = np.random.uniform(2.0, 4.0, self.num_active)
        # Initialize acclimation to average light (approx 200)
        self.cells_acclimation[:self.num_active] = np.random.uniform(100.0, 300.0, self.num_active)
        
        self.ext_nutrients = 500.0 # Enough for episode-long run
        self.membrane_integrity = 1.0  # 1.0 = pristine membranes, 0.0 = fully fatigued
        self.ph = 8.5 # Standardized starting pH
        self.do2 = 7.0
        self.temp = 25.0
        self.time_t = 0.0
        self.clump_mass[:] = 1.0

        self.debug_mu = 0.0
        self.debug_stress = 0.0
        self.debug_shock = 0.0
        self.debug_clump = 1.0
        self._prev_od_for_rate = float(self.od) if hasattr(self, 'od') else 0.0

        # Hardware Smoothing State (EMA)
        self.current_stir_rpm = 50.0
        self.current_nut_flow = 0.0
        self.current_co2_flow = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.action_smooth_coef = 0.001
        self.step_count = 0
        self.fouling_factor = 0.0 # Biofouling accumulation
        
        # Reset Advanced Physics
        self.salt = 1000.0
        self.pigment = 1.0
        
        self.dissolved_co2 = 2.0  # mg/L — atmospheric equilibrium

        # --- Sim-to-Real Sensor Drift (D1+) ---
        # Must be initialized before _get_obs() on reset.
        if self.difficulty >= 1:
            # Per-episode static calibration error (±5% uniform random drift)
            self._sensor_drift_mult = np.random.uniform(0.95, 1.05, size=6)
        else:
            self._sensor_drift_mult = np.ones(6, dtype=np.float32)
        
        initial_obs = self._get_obs()
        self.max_historical_od = float(getattr(self, 'od', 0.0))
        self.steps_since_od_high = 0
        self._prev_ext_nutrients = 500.0
        self._phi_prev = None
        self.episode_init_cells = int(self.num_active)
            
        return initial_obs, {}

    def _get_obs(self):
        # Calculate stats only
        if self.num_active > 0:
            # Use pre-calculated physics states from step() if available
            # If called during reset(), calculate initial states
            if not hasattr(self, 'od'):
                total_mass_mg = float(np.sum(self.cells_mass[self.active_mask]) * 1e-9)
                self.od = (total_mass_mg / self.volume_L) / 300.0
            if not hasattr(self, 'conductivity'): self.conductivity = 1500.0
            if not hasattr(self, 'rgb_absorbance'): self.rgb_absorbance = 0.05

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

        base_obs = np.array([
            turbidity_obs,
            self.ph,
            self.ext_nutrients,
            self.temp,
            self.conductivity,
            self.rgb_absorbance
        ], dtype=np.float32)

        # Stochastic Sensor Noise: High-frequency jitter
        # D0 gets minor ±1% jitter, D1+ gets realistic ±2% jitter
        jitter_mag = 0.02 if self.difficulty >= 1 else 0.01
        jitter = np.random.uniform(1.0 - jitter_mag, 1.0 + jitter_mag, size=(6,))
        
        # Apply combined Jitter * Static Drift
        noisy_obs = base_obs * jitter * self._sensor_drift_mult
        
        return noisy_obs.astype(np.float32)

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
        target_stir_rpm = np.interp(stir_act, [-1, 1], [50, 200])   # Standardized RPM range constraint
        target_I_surface = np.interp(light_act, [-1, 1], [0, 2000]) # High-intensity lights
        target_nut_flow  = np.interp(nut_act,   [-1, 1], [0, 100])  # High nutrients
        target_co2_flow  = np.interp(co2_act,   [-1, 1], [0, 5])    # mL/min sparging
        
        # 2. Hardware Smoothing (EMA)
        # Nutrients: target ~10 minute actuator horizon at dt=0.01h
        alpha_nut = 0.06
        alpha_co2 = 0.15  # fast gas-valve response with mild actuator lag
        self.current_stir_rpm = (0.95 * self.current_stir_rpm) + (0.05 * target_stir_rpm)
        self.current_nut_flow = (1.0 - alpha_nut) * self.current_nut_flow + alpha_nut * target_nut_flow
        self.current_co2_flow = (1.0 - alpha_co2) * self.current_co2_flow + alpha_co2 * target_co2_flow
        
        stir_rpm  = self.current_stir_rpm
        I_surface = target_I_surface  # Light changes instantly
        nut_flow  = self.current_nut_flow
        co2_flow  = self.current_co2_flow

        # --- Grace Period (Training Wheels) ---
        if self.step_count < 2400:
            grace_factor = self.step_count / 2400.0
            max_light = 500.0 + (1500.0 * grace_factor)
            I_surface = min(I_surface, max_light)
            max_nuts = 20.0 + (80.0 * grace_factor) # Ramp 20→100 mg/h: matches real consumption at 3k-12k cells
            nut_flow = min(nut_flow, max_nuts)

        # --- Temperature Inertia ---
        ambient_temp = 25.0
        # Physics scale: D0=25%, D1=50%, D2=100%
        diff_level = getattr(self, 'difficulty', 0)
        phys_scale = 0.25 if diff_level == 0 else (0.5 if diff_level == 1 else 1.0)
        
        # Light adds heat: max 2000 umol/m2/s ~ 2.0 degrees C / hour
        heat_from_light = (I_surface * 0.001) * self.dt * phys_scale
        # Cooling to ambient: Rate of 0.1 per hour
        cooling = 0.1 * (self.temp - ambient_temp) * self.dt * phys_scale
        self.temp += heat_from_light - cooling
        # Viscous/Ohmic heating from impeller (scales with RPM^3, max ~0.5°C at 200 RPM)
        stir_heat = (stir_rpm / 200.0) ** 3 * 0.5 * self.dt * phys_scale
        self.temp += stir_heat
        
        self.temp = np.clip(self.temp, 15.0, 45.0)

        # --- Physics (Chaotic Turbulence) ---
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-285)
        
        dt_sec = self.dt * 3600
        self.time_t += self.dt
        # Bubble Scattering (0.004 * RPM -> 0.8 at 200 RPM)
        k_scatter = stir_rpm * 0.004 # 0 to 1
        
        # Base Mixing Intensity
        mix_intensity = stir_rpm / 200.0 # 0 to 1
        
        if self.num_active > 0 and mix_intensity > 0.01:
            # Coherent structure (Large Eddy)
            v_macro = 0.005 * mix_intensity * np.sin(100.0 * self.cells_z[self.active_mask] - 5.0 * self.time_t)
            
            # Turbulent fluctuations (Small Eddies)
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-302)
            indices = np.where(self.active_mask)[0]
            v_micro = 0.002 * mix_intensity * np.sin(500.0 * self.cells_z[self.active_mask] - 20.0 * self.time_t + indices)
            
            # Diffusion (Random Walk) still exists but is smaller
            noise = np.random.normal(0, 1, self.num_active)
            v_diff = np.sqrt(2 * 1e-7) * noise # reduced noise
            
            dz = (v_macro + v_micro) * dt_sec + v_diff
            
            self.cells_z[self.active_mask] += dz
        else:
            # Low mixing -> Sedimentation (Sinking) + diffusion
            sinking_rate = 0.001 # m/h
            noise = np.random.normal(0, 1, sum(self.active_mask))
            dz = -sinking_rate * self.dt + (np.sqrt(2 * 1e-7 * dt_sec) * noise)
            self.cells_z[self.active_mask] += dz
        
        # Boundary Conditions
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-322)
        self.cells_z = np.abs(self.cells_z)
        over_bottom = self.cells_z > self.reactor_depth
        self.cells_z[over_bottom] = 2*self.reactor_depth - self.cells_z[over_bottom]
        self.cells_z = np.clip(self.cells_z, 0, self.reactor_depth)

        # --- Biology ---
        prev_num_active = self.num_active
        total_uptake_mg = 0.0
        if self.num_active > 0:
            n_spawns = 0
            params = self.strain_params
            
            # 1. Light Field (Self-Shading + Biofouling)
            od = self._get_obs()[0]
            # Safety clamp for OD (User Limit = 10.0)
            od = np.nan_to_num(od, nan=0.0, posinf=10.0)
            od = np.clip(od, 0.0, 10.0)
            
            # Update Biofouling (Slow accumulation based on biomass)
            # Rate ~ 0.01 per 72h if full biomass
            if self.step_count % 10 == 0:
                self.fouling_factor += (od * 1e-5) * self.dt 
            
            # Effective Light Reduced by Fouling (Beer-Lambert for wall coating)
            I_effective = I_surface * np.exp(-self.fouling_factor)
            
            # Spirulina Extinction: Gentler attenuation matching genetic_env
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-351)
            k_ext = 0.5 + (3.5 * od) + k_scatter 
            # prevent overflow in exp if z is huge (unlikely but safe)
            exponent = -k_ext * self.cells_z
            exponent = np.clip(exponent, -50, 0) # e^-50 is basically 0
            cells_I = I_effective * np.exp(exponent)
            
            # 2. Temperature Factor (Gaussian)
            # T_opt randomization test
            temp_factor = np.exp(-0.5 * ((self.temp - params['T_opt'])/5.0)**2)
            
            # --- Photo-Acclimation (Hysteresis) ---
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-364)
            current_acclim = self.cells_acclimation[self.active_mask]
            tau_acclim = params['tau_acclim']
            
            # Update Acclimation State
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-370)
            cells_I_active = cells_I[self.active_mask]
            
            d_acclim = (cells_I_active - current_acclim) * (self.dt / tau_acclim)
            self.cells_acclimation[self.active_mask] += d_acclim
            
            # Photo-Inhibition / Shock
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-378)
            diff = (cells_I_active - current_acclim)
            shock_factor = np.exp(-0.000003 * (diff**2))  # 3e-6: matches genetic_env/total_env
            # Asymmetrical: Moving to dark is just low energy (handled by f_I), 
            # but moving to high light when acclimated to dark is DAMAGE.
            
            # Refined Shock: Only penalize if I > A (High light shock)
            # mild penalty for A > I (Dark adaptation lag)
            
            # --- Oxygen Toxicity (ROS Damage) ---
            if self.difficulty >= 1:
                # Inhibition starts ~16 mg/L with ^4 curve
                f_O2 = max(0.0, 1.0 - (self.do2 / 22.0)**4)
            else:
                f_O2 = 1.0
            
            # 3. Growth Rate
            f_I = cells_I / (params['Ki'] + cells_I + (cells_I**2 / params['Kii']))
            f_I = np.nan_to_num(f_I)
            
            # Droop Quota
            # Only update active quotas
            current_quotas = self.cells_quota[self.active_mask]
            
            f_Q = np.maximum(0.0, 1.0 - params['Q_min'] / (current_quotas + 1e-6))
            
            # pH Inhibition (Asymmetric Gaussian — Spirulina alkaliphile)
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-407)
            if self.ph <= 9.5:
                f_pH = np.exp(-0.5 * ((self.ph - 9.5) / 1.2) ** 2)
            else:
                f_pH = np.exp(-0.5 * ((self.ph - 9.5) / 2.0) ** 2)
            
            # Osmotic Stress (Nutrient Overdose)
            # Safe zone: 0 - 2500 mg/L for Medium (Genetic: 2000)
            if self.ext_nutrients > 2500.0:
                f_Osmosis = np.exp(-0.5 * ((self.ext_nutrients - 2500.0)/700.0)**2)
            else:
                f_Osmosis = 1.0
                
            # --- DEBUG STATS CAPTURE ---
            if self.step_count % 100 == 0 and self.num_active > 0:
                  self.debug_f_I = np.mean(f_I[self.active_mask])
                  self.debug_f_Q = np.mean(f_Q)
                  self.debug_f_pH = f_pH
                  self.debug_f_O2 = f_O2
                  self.debug_shock = np.mean(shock_factor)
            # ---------------------------
            
            # Clamp mu to prevent explosion
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-432)
            repair_factor = 1.0 / (1.0 + np.exp(-0.12 * (stir_rpm - 195.0)))
            repair_tax = 1.0 - (0.25 * repair_factor)
            
            # --- Cell Wall Fatigue (Accumulative Membrane Integrity) ---
            # Reduced decay for Medium difficulty
            shear_stress = float(np.clip((stir_rpm - 80.0) / 70.0, 0.0, 1.0))
            self.membrane_integrity -= shear_stress * 0.0005 # Half decay rate
            self.membrane_integrity += (1.0 - self.membrane_integrity) * 0.002
            self.membrane_integrity = float(np.clip(self.membrane_integrity, 0.0, 1.0))
            fatigue_tax = 1.0 - (0.10 * (1.0 - self.membrane_integrity))

            # Carbon-limited growth (Monod saturation on dissolved CO2).
            # At dissolved_co2 = 0.5 mg/L, f_carbon = 0.5.
            Kc_CO2 = 0.5
            dissolved_co2 = getattr(self, 'dissolved_co2', 2.0)
            f_carbon = dissolved_co2 / (Kc_CO2 + dissolved_co2)

            # Apply shock factor and repair/fatigue tax to growth
            current_mu = params['mu_max'] * f_I[self.active_mask] * f_Q * f_carbon * temp_factor * shock_factor * f_O2 * f_pH * f_Osmosis * repair_tax * fatigue_tax
            current_mu = np.clip(current_mu, 0.0, 5.0) 
            
            # --- Maintenance Respiration ---
            # 1.0% of mu_max — matches genetic_env/total_env
            m_respiration = 0.010 * params['mu_max']
            
            # Net Growth Rate = Photosynthesis - Respiration
            # This can be negative (mass loss) if light/nutrients are insufficient!
            net_mu = current_mu - m_respiration
            
            # Grow Biomass
            growth_mult = np.exp(net_mu * self.dt)
            # Clip multiplier to avoid single-step explosion (both up and down)
            growth_mult = np.clip(growth_mult, 0.5, 2.0)
            
            self.cells_mass[self.active_mask] *= growth_mult
            
            # --- PROBABILISTIC LYSIS DEATH (replaces hard starvation threshold) ---
            # Background lysis: ~0.5%/day. Stress lysis: up to ~5%/day when starved.
            mean_mu       = float(np.mean(current_mu))
            stress_factor = np.clip(
                (m_respiration - mean_mu) / (m_respiration + 1e-9), 0.0, 1.0
            )
            # Lysis penalty reduced by 25% for Medium
            lysis_rate  = 0.75 * (1e-4 + (2e-3 * (stress_factor ** 2)))  # per hour
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

            # Mass clamp for numerical safety on survivors only (floor 1e5, cap 5e7)
            self.cells_mass[self.active_mask] = np.clip(self.cells_mass[self.active_mask], 1e5, 5e7)
                
            # Nutrient Uptake
            uptake_rate = 0.5 * (self.ext_nutrients / (params['Ks'] + self.ext_nutrients))
            uptake_amount = uptake_rate * self.dt
            
            # Update Quota & External Nutrients
            self.cells_quota[self.active_mask] += uptake_amount
            
            # Bug fix: uptake_amount is a per-cell scalar, BUT 1 agent = 250k cells.
            # Using 0.005 provides a realistic episode-scale nutrient drain for RL.
            total_uptake_mg = uptake_amount * self.num_active * 0.005
            # Fix: Scale nut_flow by dt (Rate -> Amount)
            self.ext_nutrients = max(0, self.ext_nutrients - total_uptake_mg + (nut_flow * self.dt))
            
            # --- CELL DIVISION (Reproduction) ---
            # (full rationale: docs/decision_history.md#--environments-heavy_env-py-508)
            ready_to_divide = (self.active_mask) & (self.cells_mass >= 1.4e7)
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
                    self.cells_z[child_indices] = self.cells_z[parent_indices]     # Inherit position
                    self.cells_quota[child_indices] = self.cells_quota[parent_indices] # Inherit quota status
                    self.cells_acclimation[child_indices] = self.cells_acclimation[parent_indices] # Inherit acclimation
                    
                    self.num_active += n_spawns
                    
                    # Bug fix: Reset clump_mass for new children — they start as single cells.
                    # Without this, children inherit stale clump data from dead-cell slots.
                    self.clump_mass[child_indices] = 1.0
            
        else:
            n_spawns = 0
            current_mu = np.zeros(1) # fallback
            f_Q = np.zeros(1)

        # --- Environmental Dynamics (Macro) ---

        # 2. Gas Exchange (O2 & CO2)
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-549)
        od = self.od
        viscosity_penalty = max(0.1, 1.0 - (od * 0.01)) # Drops to 0.1 at OD=90
        
        # k_La: scaled by surface-to-volume ratio for 30L reactor (S/V ∝ volume^(-1/3))
        sv_ratio  = (1.0 / self.volume_L) ** (1.0 / 3.0)  # 0.32 at 30L
        base_kLa  = sv_ratio * (0.5 + (4.0 * ((stir_rpm / 200.0)**1.8)))
        k_La = base_kLa * viscosity_penalty

        # Dissolved Oxygen Dynamics
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-564)
        total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9  # pg to mg

        # Simplify: Delta Mass roughly tracks O2
        if self.step_count > 0:
            delta_mass_mg = total_mass_mg - (self.last_mass * 1e-9)
        else:
            delta_mass_mg = 0.0
            
        # Update OD — volume-normalised (concentration, not total mass)
        self.od = (total_mass_mg / self.volume_L) / 300.0
        # print(f"DEBUG: Mass={total_mass_mg}, OD={self.od}")
            
        o2_production = delta_mass_mg * 1.2 # mg O2 produced
        
        # Gas Transfer
        # O2 Saturation at 25C is approx 8.0 mg/L
        o2_sat = 8.0 
        o2_transfer = k_La * (o2_sat - self.do2) * self.dt
        
        self.do2 += (o2_production / self.volume_L) + o2_transfer
        self.do2 = np.clip(self.do2, 0.0, 30.0) # Cap at realistic supersaturation
        
        # 3. DIC-Driven pH (Full Henry's Law CO2 Physics)
        # CO2 dissolution: inject + offgassing equilibrium + Calvin cycle consumption
        co2_sat      = 0.6   # mg/L — atmospheric equilibrium
        co2_inject   = (co2_flow * 0.44 / self.volume_L) * self.dt
        co2_degas    = k_La * max(0.0, self.dissolved_co2 - co2_sat) * self.dt
        co2_consumed = delta_mass_mg * 0.015   # Calvin cycle consumption
        self.dissolved_co2 = float(np.clip(
            self.dissolved_co2 + co2_inject - co2_degas - co2_consumed, 0.0, 20.0))
        
        # pH driven by DIC concentration (log-linear carbonate chemistry)
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-599)
        dic_buffer_reserve = 0.03  # mg/L effective DIC from alkalinity buffer
        effective_dic = self.dissolved_co2 + dic_buffer_reserve
        ph_from_co2 = 8.5 - 1.2 * np.log10(effective_dic / 0.6)
        ph_rise      = (o2_production / self.volume_L) * 0.1
        ph_drop_resp = (total_mass_mg / self.volume_L) * 0.0001
        ph_restore   = (9.5 - self.ph) * 0.15 * self.dt  # Increased buffer for Medium
        ph_biotic = self.ph + ph_rise - ph_drop_resp + ph_restore
        
        # Blend: 50% DIC chemistry, 50% biological drift (Softer than 70/30)
        new_ph = float(np.clip(0.95 * self.ph + 0.05 * (0.5 * ph_from_co2 + 0.5 * ph_biotic), 6.0, 11.5))
        self.ph = new_ph if np.isfinite(new_ph) else 8.5
        
        # --- Advanced Physics: Pigment & Salt ---
        
        # 4. Pigment Dynamics (Photo-inhibition & Chlorosis)
        # Bleaching: High Light (>1000) or Low Nitrogen (<100) damages pigment
        avg_light = I_surface * np.exp(-0.2 * self.reactor_depth/2) # Approx mid-depth light
        is_bleached = (avg_light > 1000.0) or (self.ext_nutrients < 100.0)
        
        if is_bleached:
            self.pigment -= 0.01 * self.dt # Slow degradation
        else:
            self.pigment += 0.01 * self.dt # Slow recovery
        self.pigment = np.clip(self.pigment, 0.2, 1.0) # Min 20% pigment
        
        # 5. Salinity Accumulation
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-627)
        nutrient_uptake = total_uptake_mg  # mg consumed this step
        salt_inflow_mg = nutrient_uptake * 0.1  # impurity carryover from feed salts
        lysis_mg = max(0.0, -delta_mass_mg)
        salt_decay_mg = lysis_mg * 0.5
        self.salt += (salt_inflow_mg + salt_decay_mg) / max(self.volume_L, 1e-9)
        
        # --- Sensors ---
        
        # OD ~ Mass^0.8 (Self-Shading effect)
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-639)
        
        turbidity = self.od # Use the Linear Physics OD
        
        # 2. RGB Absorbance (Proxy for Chlorophyll)
        # Absorbance = Turbidity * Pigment_Health
        rgb_absorbance = turbidity * self.pigment
        
        # 3. Conductivity (Realistic Formula)
        # C = (Salt + Nutrients + |7-pH|) * Temp_Factor
        cond_base = self.salt + self.ext_nutrients + 100.0 * abs(7.0 - self.ph)
        cond_temp = 1.0 + 0.02 * (self.temp - 25.0)
        conductivity = cond_base * cond_temp

        # Store for Observation
        self.rgb_absorbance = rgb_absorbance
        self.conductivity = conductivity
        
        # --- Reward ---
        # Reward is biomass growth minus penalty for crash
        total_mass = np.sum(self.cells_mass[self.active_mask]) if self.num_active > 0 else 0
        if self.step_count == 0: self.last_mass = total_mass
        
        # Calculate Productivity
        productivity = total_mass - self.last_mass
        if not np.isfinite(productivity):
            productivity = 0.0
            
        self.last_mass = total_mass
        
        # --- Sim-to-Real: Intra-Episode Genetic Micro-Drift ---
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-672)
        if self.difficulty >= 2 and self.step_count > 0 and self.step_count % 500 == 0:
            self.strain_params['mu_max'] *= np.random.uniform(0.99, 1.01)
            self.strain_params['Ks'] *= np.random.uniform(0.99, 1.01)
        
        # 1. Productivity: Scale by 1e-8 (1e8 pg = 0.1 mg growth -> Reward 1.0)
        # (full rationale: docs/decision_history.md#--environments-heavy_env-py-679)
        mean_shock = np.mean(shock_factor) if self.num_active > 0 else 1.0
        mean_clump = np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0
        penalty_shock = (1.0 - mean_shock) * -0.05
        penalty_clump = (mean_clump - 1.0) * -0.01
        
        # 1. DO2 Production Proxy (Photosynthesis)
        # Provides a high-frequency growth signal before biomass accumulates.
        reward_do2 = o2_production * 0.1 
        
        # 2. Nutrient Consumption Proxy (near-instant growth signal)
        reward_nut_consume = max(0.0, (self._prev_ext_nutrients - self.ext_nutrients)) * 0.02
        self._prev_ext_nutrients = self.ext_nutrients
        
        # 3. pH Drift Proxy (Carbon Uptake)
        # Biological growth adds delta_ph_bio (ph_rise in code)
        co2_scale = np.clip(1.0 - (co2_flow / 5.0), 0.0, 1.0)
        reward_ph = ph_rise * 50.0 * co2_scale
        
        # Direct biomass term scaled by per-cell growth (population-normalized)
        # Per-cell growth: biomass per individual cell per step
        per_cell_growth = (delta_mass_mg / (self.num_active + 1e-6)) * 1000
        reward_biomass = 0.15 * np.tanh(per_cell_growth / 0.5)

        # Population boost: smaller populations get higher reward multiplier for same per-cell growth
        # At 1k: 1.83×, At 3k: 1.5×, At 6k: 1.0×, At 9k+: 1.0×
        pop_boost = 1.0 + max(0.0, 1.0 - (self.num_active / 6000.0))
        reward_biomass *= pop_boost

        # Low OD emphasis: extra boost in lag phase
        if self.od < 0.05:
            reward_biomass *= 1.5
        pop_loss = max(0, prev_num_active - self.num_active)
        reward_lysis = -0.01 * float(pop_loss)

        # 4. High Water Mark OD Anchor
        # Guarantees dense reward only when the agent strictly breaks the all-time high OD.
        reward_od = 0.0
        if not hasattr(self, 'steps_since_od_high'):
            self.steps_since_od_high = 0
            
        if self.od > self.max_historical_od:
            delta_od = self.od - self.max_historical_od
            # Population-aware OD anchor: boost reward multiplier at low pop (harder to achieve growth)
            # At 1k: 2.0×, At 3k: 1.33×, At 6k: 1.0×, At 9k: 1.0×
            pop_boost = min(2.0, max(1.0, 4000.0 / (self.num_active + 1e-6)))
            od_gain_scale = (8000.0 if self.od < 0.05 else 5000.0) * pop_boost
            reward_od = np.tanh(delta_od * 1000.0) * 0.5 * pop_boost
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
        # If the biological mass of the culture is actively dropping (lysis/death), bleed reward!
        # We use delta_mass_mg instead of OD to avoid tiny floating-point rounding blindspots
        if delta_mass_mg < -0.0001:  
            reward_stagnation = -0.15
        # Population boost for growth rate (at 1k: 1.83×, at 6k: 1.0×)
        pop_factor = 1.0 + max(0.0, 1.0 - (self.num_active / 6000.0))
        reward_growth_rate = max(0.0, od_rate) * 20.0 * pop_factor

        # Total Proxy Reward + Metabolic Momentum
        mean_f_Q = float(np.mean(f_Q)) if self.num_active > 0 else 0.0
        reward = reward_do2 + reward_nut_consume + reward_ph + reward_biomass + reward_lysis + reward_od + reward_growth_rate + reward_stagnation + (mean_f_Q * 0.05)
        reward -= action_smooth_penalty

        # Potential-Based Reward Shaping (PBRS)
        gamma_pbrs = 0.99
        mean_f_Q_cur = float(np.mean(np.maximum(
            0.0, 1.0 - 0.5 / (self.cells_quota[self.active_mask] + 1e-6)
        ))) if self.num_active > 0 else 0.0
        phi_cur = (
            0.3 * min(self.ext_nutrients / 500.0, 1.0) +
            0.3 * min(self.dissolved_co2 / 3.0, 1.0) +
            0.2 * max(0.0, 1.0 - self.do2 / 22.0) +
            0.2 * mean_f_Q_cur
        )
        phi_prev = self._phi_prev if self._phi_prev is not None else phi_cur
        reward_pbrs = gamma_pbrs * phi_cur - phi_prev
        self._phi_prev = phi_cur
        reward += reward_pbrs * 0.5

        if self.difficulty == 0:
            reward += penalty_shock + penalty_clump

        # Phase-based reward shaping for curriculum consistency with genetic_env.
        if self.difficulty == 0:
            ph_shape = np.exp(-0.5 * ((self.ph - 9.5) / 1.0) ** 2) * 0.15
            do2_shape = np.exp(-0.5 * ((self.do2 - 10.0) / 4.0) ** 2) * 0.10
            t_opt = self.strain_params.get('T_opt', 27.0)
            temp_shape = np.exp(-0.5 * ((self.temp - t_opt) / 3.0) ** 2) * 0.10
            # Fade out shaping linearly over 3000 steps to force transition to biomass growth
            fade_multiplier = max(0.0, 1.0 - (getattr(self, 'step_count', 0) / 7200.0))
            reward += (ph_shape + do2_shape + temp_shape) * fade_multiplier
        elif self.difficulty in (1, 2):
            ph_shape = np.exp(-0.5 * ((self.ph - 9.5) / 1.0) ** 2) * 0.05
            reward += ph_shape

        # Safety/Stability Penalties
        if self.ph > 11.0 or self.ph < 7.0:
            reward -= 0.1
        
        # Final safeguard on reward
        if not np.isfinite(reward):
            reward = -10.0 # Punishment for breaking physics
        
        # Penalty for crashing population
        if self.num_active < 10:
            reward -= 1000.0
            done = True
        else:
            self.step_count += 1
            done = self.step_count >= self.max_steps
        
        # Debug Print
        if (self.step_count % 100 == 0) or done:
             # Use collected stats if available, else 0
             d_fI = getattr(self, 'debug_f_I', 0.0)
             d_fQ = getattr(self, 'debug_f_Q', 0.0)
             d_shock = getattr(self, 'debug_shock', 0.0)
             turb = getattr(self, 'turbidity_obs', 0.0)
             print(f"[EnvDebug] Step: {self.step_count}, Active: {self.num_active}, Mass: {total_mass_mg:.2f}, OD: {self.od:.4f}, Turb: {turb:.4f}, pH: {self.ph:.2f}, f_I: {d_fI:.2f}, f_Q: {d_fQ:.2f}, Shock: {d_shock:.2f}, Rew: {reward:.3f}, Done: {done}")
             
        return self._get_obs(), float(reward), done, False, {
            "pop": self.num_active,
            "init_pop": int(getattr(self, 'episode_init_cells', self.initial_cells)),
            "fouling": self.fouling_factor,
            "peak_od": float(getattr(self, 'max_historical_od', getattr(self, 'od', 0.0))),
            "od": float(getattr(self, 'od', 0.0)),
            "start_mode": getattr(self, 'episode_start_mode', 'low'),
        }

