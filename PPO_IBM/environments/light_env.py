
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict

class LightPhotobioreactorEnv(gym.Env):
    """
    Individual-Based Model (IBM) Photobioreactor Environment (Light Efficient).
    Tracks N individual algal cells as particles in 1D depth (z-axis).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_cells: int = 300000, initial_cells: int = 2000):
        super(LightPhotobioreactorEnv, self).__init__()
        
        # --- CONFIGURATION ---
        self.max_cells = max_cells
        self.initial_cells = initial_cells
        self.num_active = initial_cells
        self.dt = 0.01  # Time step (0.01h) determines simulation resolution
        self.reactor_depth = 0.30
        self.volume_L = 30.0
        
        # Action: [Stirring, Light, Nutrient, CO2]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation Space (6 Dims): OD, pH, Nutrients, DO2, Temp, Conductivity, RGB
     
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([5000.0,14.0,5000.0,50.0,10000.0,20.0]),
            dtype=np.float32
        )
        
        # --- VECTORIZED STATE ARRAYS ---
        self.cells_z = np.zeros(self.max_cells, dtype=np.float32) 
        self.cells_mass = np.zeros(self.max_cells, dtype=np.float32)
        self.cells_quota = np.zeros(self.max_cells, dtype=np.float32)
        self.cells_acclimation = np.zeros(self.max_cells, dtype=np.float32)
        self.clump_mass = np.ones(self.max_cells, dtype=np.float32)  # 1.0 = single cell
        self.active_mask = np.zeros(self.max_cells, dtype=bool)
        
        self.ext_nutrients = 500.0
        self.ph = 8.5
        self.do2 = 6.0
        self.temp = 25.0 
        self.time_t = 0.0
        
        # Hardware Smoothing State (EMA)
        self.current_stir_rpm = 50.0
        self.current_I_surface = 0.0
        self.current_nut_flow = 0.0
        self.current_co2_flow = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.action_smooth_coef = 0.001
        
        # Advanced Physics State
        self.salt = 1000.0
        self.pigment = 1.0
        self.dissolved_co2 = 2.0
        
        self.strain_params = {}
        self.step_count = 0
        self.max_steps = 30000 
        
    def _randomize_strain(self):
        """Generates a unique 'Strain' of algae for this episode."""
        self.strain_params = {
            'mu_max': np.random.normal(0.10, 0.05),
            'Ks': np.random.normal(20.0, 5.0),
            'Ki': np.random.normal(120.0, 30.0),
            'Kii': np.random.normal(2000.0, 500.0),
            'T_opt': np.random.normal(27.0, 2.0),
            'Q_min': 1.5,
            'Q_max': 5.0,
            'tau_acclim': np.random.uniform(1.0, 4.0)
        }
        self.strain_params['mu_max'] = max(0.05, self.strain_params['mu_max'])
        self.strain_params['Ki'] = max(10, self.strain_params['Ki'])
        self.strain_params['T_opt'] = np.clip(self.strain_params['T_opt'], 20.0, 35.0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._randomize_strain()
        
        self.num_active = self.initial_cells
        self.active_mask[:] = False
        self.active_mask[:self.num_active] = True
        
        self.cells_z[:self.num_active] = np.random.uniform(0, self.reactor_depth, self.num_active)
        self.cells_mass[:self.num_active] = np.random.normal(1.25e7, 1e6, self.num_active)
        self.cells_quota[:self.num_active] = np.random.uniform(2.0, 4.0, self.num_active)
        self.cells_acclimation[:self.num_active] = np.random.uniform(100.0, 300.0, self.num_active)
        self.clump_mass[:] = 1.0  # all cells start as single cells
        
        self.ext_nutrients = 300.0
        self.ph = 8.5
        self.do2 = 7.0
        self.temp = 25.0
        self.step_count = 0
        self.time_t = 0.0
        self.fouling_factor = 0.0
        self.last_mass = 0.0
        
        # Reset Hardware Smoothing
        self.current_stir_rpm = 50.0
        self.current_I_surface = 0.0
        self.current_nut_flow = 0.0
        self.current_co2_flow = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        # Advanced Physics State
        self.salt = 1000.0
        self.pigment = 1.0
        self.dissolved_co2 = 2.0
        # Mild per-episode calibration drift for robustness tests.
        self._sensor_drift_mult = np.random.uniform(0.98, 1.02, size=6)
        # RPM-coupled EMA lag state for pH and DO2
        self._ph_obs_ema = self.ph
        self._do2_obs_ema = self.do2
        self._prev_od_for_rate = float(self.od) if hasattr(self, 'od') else 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.num_active > 0:
            if not hasattr(self, 'od'): self.od = 0.05
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

        # RPM-coupled EMA sensor lag for pH (idx 1) and DO2 (idx 3).
        # High RPM -> fast mixing -> short lag (2 steps).
        # Low RPM  -> slow mixing -> long lag  (6 steps).
        rpm = float(np.clip(getattr(self, 'current_stir_rpm', 50.0), 50.0, 200.0))
        mix_quality = (rpm - 50.0) / 150.0
        lag_steps = max(2, int(round(6 - (4 * mix_quality))))
        alpha = 2.0 / (lag_steps + 1.0)
        self._ph_obs_ema  = (1.0 - alpha) * getattr(self, '_ph_obs_ema', base_obs[1]) + alpha * base_obs[1]
        base_obs[1] = self._ph_obs_ema

        # Stochastic sensor jitter: ±2% high-frequency noise
        noise_factor = np.random.uniform(0.98, 1.02, size=(6,))
        noisy_obs = base_obs * noise_factor * self._sensor_drift_mult
        
        return noisy_obs.astype(np.float32)

    def step(self, action):
        action_vec = np.asarray(action, dtype=np.float32)
        action_delta = action_vec - self.prev_action
        action_smooth_penalty = self.action_smooth_coef * float(np.sum(action_delta ** 2))
        self.prev_action = action_vec.copy()

        stir_act, light_act, nut_act, co2_act = action
        
        # 1. Decode Target Actions
        target_stir_rpm = np.interp(stir_act, [-1, 1], [50, 200])   # Standardized RPM range constraint
        target_I_surface = np.interp(light_act, [-1, 1], [0, 2000]) # Moderated light range
        target_nut_flow = np.interp(nut_act, [-1, 1], [0, 100]) # Moderated nutrient range
        target_co2_flow = np.interp(co2_act, [-1, 1], [0, 120])
        
        # 2. Hardware Smoothing (EMA)
        alpha_nut = 0.06
        alpha_co2 = 0.15  # fast gas-valve response with mild actuator lag
        self.current_stir_rpm = (0.95 * self.current_stir_rpm) + (0.05 * target_stir_rpm)
        self.current_nut_flow = (1.0 - alpha_nut) * self.current_nut_flow + alpha_nut * target_nut_flow
        self.current_co2_flow = (1.0 - alpha_co2) * self.current_co2_flow + alpha_co2 * target_co2_flow
        
        # Use smoothed
        stir_rpm = self.current_stir_rpm
        I_surface = target_I_surface # Light changes instantly
        nut_flow = self.current_nut_flow
        co2_flow = self.current_co2_flow
        
        if np.any(np.isnan(action)):
            stir_rpm = 50.0; I_surface = 0.0; nut_flow = 0.0; co2_flow = 0.0

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

        # --- Physics ---
        dt_sec = self.dt * 3600
        self.time_t += self.dt
        mix_intensity = stir_rpm / 200.0
        # Bubble Scattering (0.025 * RPM -> 5.0 at 200 RPM)
        k_scatter = stir_rpm * 0.025
        
        if self.num_active > 0 and mix_intensity > 0.01:
            v_macro = 0.005 * mix_intensity * np.sin(100.0 * self.cells_z[self.active_mask] - 5.0 * self.time_t)
            indices = np.where(self.active_mask)[0]
            v_micro = 0.002 * mix_intensity * np.sin(500.0 * self.cells_z[self.active_mask] - 20.0 * self.time_t + indices)
            noise = np.random.normal(0, 1, self.num_active)
            v_diff = np.sqrt(2 * 1e-7) * noise 
            dz = (v_macro + v_micro) * dt_sec + v_diff
            self.cells_z[self.active_mask] += dz
        else:
            sinking_rate = 0.001 
            noise = np.random.normal(0, 1, sum(self.active_mask))
            dz = -sinking_rate * self.dt + (np.sqrt(2 * 1e-7 * dt_sec) * noise)
            self.cells_z[self.active_mask] += dz
        
        self.cells_z = np.abs(self.cells_z)
        over_bottom = self.cells_z > self.reactor_depth
        self.cells_z[over_bottom] = 2*self.reactor_depth - self.cells_z[over_bottom]
        self.cells_z = np.clip(self.cells_z, 0, self.reactor_depth)

        # --- Biology ---
        if self.num_active > 0:
            od_approx = self.od if hasattr(self, 'od') else 0.001

            # --- Flocculation & Shear Breakup ---
            # High RPM = more shear = less sticking (but more breakup)
            rpm_factor = max(0.1, 1.0 - (stir_rpm / 250.0))
            prob_stick = (od_approx * 1e-3) * rpm_factor
            stick_mask = (np.random.uniform(0, 1, self.max_cells) < prob_stick) & self.active_mask
            self.clump_mass[stick_mask] += 1.0

            # Mechanical shear breakup (onset at 80 RPM)
            clump_shear = max(0.0, (stir_rpm - 80.0) / 120.0) ** 2

            # Brownian/diffusive breakup (always active, prevents runaway aggregation)
            brownian_breakup = 0.005 * (self.clump_mass[self.active_mask] - 1.0) ** 0.5

            # Total breakup rate
            breakup_rate = 0.5 * clump_shear * (self.clump_mass[self.active_mask] ** 0.5) + brownian_breakup
            self.clump_mass[self.active_mask] -= breakup_rate * self.dt
            self.clump_mass[self.active_mask] = np.maximum(self.clump_mass[self.active_mask], 1.0)

            n_spawns = 0
            params = self.strain_params

            od_for_fouling = np.nan_to_num(od_approx, nan=0.0, posinf=10.0)
            od_for_fouling = np.clip(od_for_fouling, 0.0, 10.0)
            # Slow biofouling accumulation.
            if self.step_count % 100 == 0:
                self.fouling_factor += (od_for_fouling * 1e-5) * self.dt
                
            # Intra-episode genetic micro-drift (±1% every 500 steps).
            # Simulates slow natural mutation/evolution of the strain.
            if self.step_count > 0 and self.step_count % 500 == 0:
                self.strain_params['mu_max'] *= np.random.uniform(0.99, 1.01)
                self.strain_params['Ks'] *= np.random.uniform(0.99, 1.01)
            
            # Light attenuation with clump self-shading (geometric S/V scaling)
            f_clump_shade = self.clump_mass[self.active_mask] ** (-1.0 / 3.0)
            I_effective_surf = I_surface * np.exp(-self.fouling_factor)
            k_ext = 0.2 + (4.5 * od_approx) + k_scatter
            exponent = np.clip(-k_ext * self.cells_z[self.active_mask], -50, 0)
            cells_I = I_effective_surf * np.exp(exponent) * f_clump_shade
            
            temp_factor = np.exp(-0.5 * ((self.temp - params['T_opt'])/5.0)**2)
            
            current_acclim = self.cells_acclimation[self.active_mask]
            cells_I_active = cells_I  # already sliced to active cells above
            
            # Fixed EMA coefficient (matches genetic_env strength: 0.1)
            self.cells_acclimation[self.active_mask] += 0.1 * (cells_I_active - current_acclim)
            
            diff = (cells_I_active - current_acclim)
            shock_factor = np.exp(-0.000003 * (diff**2))  # 3e-6 matches genetic_env
            
            f_O2 = max(0.0, 1.0 - (self.do2 / 26.0)**4)
            
            f_I = np.nan_to_num(cells_I / (params['Ki'] + cells_I + (cells_I**2 / 2500.0)))
            
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
            
            # --- Nutrient Inhibition (Osmotic Stress) ---
            # Safe zone: 0 - 2000 mg/L
            # Penalty starts above 2000. Simplified, no salinity tracking.
            if self.ext_nutrients > 2000.0:
                f_Osmosis = np.exp(-0.5 * ((self.ext_nutrients - 2000.0)/500.0)**2)
            else:
                f_Osmosis = 1.0

            # Explicit inorganic carbon limitation for photosynthesis.
            f_CO2 = self.dissolved_co2 / (0.5 + self.dissolved_co2)  # Kc_CO2=0.5 matches genetic_env calibration
            
            # Clamp mu to prevent explosion
            # --- Shear Repair Tax (150 to 200 RPM) ---
            # 0.0 at 150 RPM, 1.0 at 200 RPM
            # Models the metabolic cost of cellular repair under mechanical stress
            repair_factor = np.clip((stir_rpm - 150.0) / 50.0, 0.0, 1.0)
            repair_tax = 1.0 - (0.25 * repair_factor)

            # Apply shock factor and repair tax to growth
            current_mu = params['mu_max'] * f_I * f_Q * temp_factor * shock_factor * f_O2 * f_pH * f_Osmosis * f_CO2 * repair_tax  # f_I already active-cells-only
            current_mu = np.clip(current_mu, 0.0, 5.0)
            # Save debug stats (computed here where the per-cell arrays exist)
            self.debug_shock = float(np.mean(shock_factor))
            self.debug_fI    = float(np.mean(f_I))
            
            # --- Maintenance Respiration ---
            m_respiration = 0.010 * params['mu_max']
            net_mu = current_mu - m_respiration
            
            growth_mult = np.clip(np.exp(net_mu * self.dt), 0.5, 2.0)
            self.cells_mass[self.active_mask] *= growth_mult
            # --- PROBABILISTIC LYSIS DEATH (replaces hard starvation threshold) ---
            # Background lysis: ~0.5%/day. Stress lysis: up to ~5%/day when starved.
            mean_mu       = float(np.mean(current_mu))
            stress_factor = np.clip(
                (m_respiration - mean_mu) / (m_respiration + 1e-9), 0.0, 1.0
            )
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

            # Mass clamp for numerical safety on survivors only
            self.cells_mass[self.active_mask] = np.clip(self.cells_mass[self.active_mask], 1e5, 5e7)
                
            uptake_rate = 0.5 * (self.ext_nutrients / (params['Ks'] + self.ext_nutrients))
            uptake_amount = uptake_rate * self.dt
            self.cells_quota[self.active_mask] += uptake_amount

            # Super-agent scaling: keep nutrient drain consistent with genetic_env.
            total_uptake_mg = (np.sum(uptake_amount) * self.num_active) * 0.005
            self.ext_nutrients = max(0.0, self.ext_nutrients - total_uptake_mg + (nut_flow * self.dt))
            # --- CELL DIVISION (Reproduction) ---
            # Threshold: 1.4e7 pg (12% above init mass of 1.25e7)
            # Rationale: Doubling (2.5e7) was too slow; deaths outpaced births.
            # 12% growth allows cells to rapidly divide before respiration starvation.
            ready_to_divide = (self.active_mask) & (self.cells_mass > 1.4e7)
            n_dividing = np.sum(ready_to_divide)
            
            if n_dividing > 0:
                inactive_indices = np.where(~self.active_mask)[0]
                n_spawns = min(n_dividing, len(inactive_indices))
                if n_spawns > 0:
                    parent_indices = np.where(ready_to_divide)[0][:n_spawns]
                    child_indices = inactive_indices[:n_spawns]
                    self.cells_mass[parent_indices] *= 0.5
                    self.active_mask[child_indices] = True
                    self.cells_mass[child_indices] = self.cells_mass[parent_indices]
                    self.cells_z[child_indices] = self.cells_z[parent_indices]
                    self.cells_quota[child_indices] = self.cells_quota[parent_indices]
                    self.cells_acclimation[child_indices] = self.cells_acclimation[parent_indices]
                    self.clump_mass[child_indices] = 1.0  # new children start as single cells
                    self.num_active += n_spawns
        else:
            n_spawns = 0
            f_Q = np.zeros(1)
            f_CO2 = 0.0

        # k_La: clump-coupled flow resistance (matches genetic_env).
        # Bug fix: use cached od_approx — calling _get_obs() here corrupts the EMA lag buffer.
        avg_clump_kLa = np.mean(self.clump_mass[self.active_mask]) if self.num_active > 0 else 1.0
        flow_resistance = 1.0 + ((od_approx / 30.0) ** 2) * (avg_clump_kLa ** 0.5)
        sv_ratio = (1.0 / self.volume_L) ** (1.0 / 3.0)
        base_kLa = sv_ratio * (0.5 + (4.0 * (stir_rpm / 200.0)))
        k_La = base_kLa / flow_resistance
        
        total_mass_mg = np.sum(self.cells_mass[self.active_mask]) * 1e-9
        if self.step_count > 0:
            delta_mass_mg = total_mass_mg - (self.last_mass * 1e-9)
        else:
            delta_mass_mg = 0.0
            
        self.od = (total_mass_mg / self.volume_L) / 300.0
            
        o2_production = delta_mass_mg * 1.2
        o2_sat = 8.0 
        o2_transfer = k_La * (o2_sat - self.do2) * self.dt
        
        self.do2 += (o2_production / self.volume_L) + o2_transfer
        self.do2 = np.clip(self.do2, 0.0, 30.0)
        
        # CO2 dissolution: inject + offgassing equilibrium + Calvin cycle consumption
        co2_sat      = 0.6   # mg/L — atmospheric equilibrium
        co2_inject   = (co2_flow * 0.44 / self.volume_L) * self.dt
        co2_equil    = k_La * (co2_sat - self.dissolved_co2) * self.dt
        co2_consumed = delta_mass_mg * 0.015   # Calvin cycle consumption
        self.dissolved_co2 = float(np.clip(
            self.dissolved_co2 + co2_inject + co2_equil - co2_consumed, 0.0, 20.0))
        
        ph_rise = (o2_production / self.volume_L) * 0.1   # Growth raises pH
        ph_drop_resp = (total_mass_mg / self.volume_L) * 0.0001  # Respiration drops pH
        # ph_drop_co2 removed: CO2 acid effect is now captured entirely via ph_from_co2 carbonate model
        ph_restore = (9.5 - self.ph) * 0.10 * self.dt     # Zarrouk buffer equilibrium (matches genetic_env)

        ph_biotic = self.ph + ph_rise - ph_drop_resp + ph_restore
        # Carbonate chemistry proxy (matches genetic_env).
        effective_dic = self.dissolved_co2 + 0.03
        ph_from_co2 = 8.5 - 1.2 * np.log10(effective_dic / 0.6)
        # Blend: 70% DIC chemistry, 30% biological drift, with 95% inertial EMA (matches genetic_env)
        new_ph = float(np.clip(0.95 * self.ph + 0.05 * (0.7 * ph_from_co2 + 0.3 * ph_biotic), 6.0, 11.5))
        self.ph = new_ph if np.isfinite(new_ph) else 8.0

        # --- Advanced Physics: Pigment & Salt ---
        avg_light = I_surface * np.exp(-0.2 * self.reactor_depth/2)
        is_bleached = (avg_light > 1000.0) or (self.ext_nutrients < 100.0)
        if is_bleached:
            self.pigment -= 0.01 * self.dt
        else:
            self.pigment += 0.01 * self.dt
        self.pigment = np.clip(self.pigment, 0.2, 1.0)
        
        salt_inflow = total_uptake_mg * 0.1 if self.num_active > 0 else 0.0
        salt_decay = (total_mass_mg * 1e-9) * 0.05
        self.salt += (salt_inflow + salt_decay) * self.dt
        
        # --- Sensors ---
        turbidity = self.od
        rgb_absorbance = turbidity * self.pigment
        cond_base = self.salt + self.ext_nutrients + 100.0 * abs(7.0 - self.ph)
        cond_temp = 1.0 + 0.02 * (self.temp - 25.0)
        conductivity = cond_base * cond_temp
        
        self.rgb_absorbance = rgb_absorbance
        self.conductivity = conductivity
        
        # --- Reward ---
        total_mass = np.sum(self.cells_mass[self.active_mask]) if self.num_active > 0 else 0
        if self.step_count == 0: self.last_mass = total_mass
        
        productivity = total_mass - self.last_mass
        if not np.isfinite(productivity): productivity = 0.0
        self.last_mass = total_mass
        
        # Metabolic Momentum
        mean_f_Q = float(np.mean(f_Q)) if self.num_active > 0 else 0.0
        mean_clump = float(np.mean(self.clump_mass[self.active_mask])) if self.num_active > 0 else 1.0
        self.debug_clump = mean_clump
        reward_do2 = o2_production * 0.05
        do2_excess = max(0.0, self.do2 - 18.0)
        penalty_do2 = 0.002 * (do2_excess ** 2)
        penalty_clump = (mean_clump - 1.0) * -0.01  # penalise aggregation
        
        # OD Growth Rate Reward (matches all other envs)
        # Population-aware: boost signal at low population (harder to achieve same growth rate)
        prev_od = getattr(self, '_prev_od_for_rate', self.od)
        od_rate = (self.od - prev_od) / (self.dt + 1e-9)
        self._prev_od_for_rate = float(self.od)
        
        if not hasattr(self, 'max_historical_od'):
            self.max_historical_od = self.od
            self.steps_since_od_high = 0
            
        if self.od > self.max_historical_od:
            self.max_historical_od = self.od
            self.steps_since_od_high = 0
        else:
            self.steps_since_od_high += 1
            
        reward_stagnation = 0.0
        if self.steps_since_od_high > 800:
            reward_stagnation = -0.05

        # Population boost for growth rate (at 1k: 1.83×, at 6k: 1.0×)
        pop_factor = 1.0 + max(0.0, 1.0 - (self.num_active / 6000.0))
        reward_growth_rate = max(0.0, od_rate) * 20.0 * pop_factor
        
        reward = (productivity * 1e-8) + (n_spawns * 0.1) + (mean_f_Q * 0.05) + reward_do2 + reward_growth_rate + reward_stagnation
        reward -= penalty_do2
        reward += penalty_clump
        reward -= action_smooth_penalty
        if not np.isfinite(reward): reward = -10.0
        
        if self.num_active < 10:
            reward -= 1000.0
            done = True
        else:
            self.step_count += 1
            done = self.step_count >= self.max_steps
            
        # Debug Print
        if (self.step_count % 100 == 0) or done:
             d_shock = getattr(self, 'debug_shock', 1.0)
             d_clump = getattr(self, 'debug_clump', 1.0)
             d_fI    = getattr(self, 'debug_fI', 0.0)
             turb = getattr(self, 'turbidity_obs', 0.0)
             print(f"[EnvDebug] Step: {self.step_count}, Active: {self.num_active}, Mass: {total_mass_mg:.2f}, OD: {self.od:.4f}, Turb: {turb:.4f}, pH: {self.ph:.2f}, Shock: {d_shock:.2f}, Clump: {d_clump:.2f}, fI: {d_fI:.3f}, Rew: {reward:.3f}, Done: {done}")
        
        return self._get_obs(), float(reward), done, False, {
            "pop": self.num_active,
            "fouling": self.fouling_factor,
            "co2_lim": float(f_CO2),
        }
