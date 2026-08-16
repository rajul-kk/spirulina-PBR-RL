import gymnasium as gym
from gymnasium import spaces

import numpy as np
from experimental_reward import calculated_reward_function

class PhotobioreactorExperimentEnv(gym.Env):
    """
    Advanced Gymnasium Environment for Photobioreactor Control.
    
    Mechanisms:
    1. Monod-Haldane Hybrid (Growth Kinetics)
    2. Huisman Light Attenuation (Depth-integrated growth)
    3. Redfield Ratio (Nutrient uptake)
    4. CO2 & pH Dynamics (Mass transfer)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(PhotobioreactorExperimentEnv, self).__init__()
        
        # --- ACTION SPACE ---
        # [Light_Set, Temp_Set, CO2_Flow, Nutrient_Flow]
        # Normalized inputs [-1, 1] for PPO stability
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # --- OBSERVATION SPACE ---
        # [s1:Water, s2:Light, s3:pH, s4:Temp, s5:Nutrients, s6:Turbidity]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]), 
            high=np.array([100, 3000, 14, 50, 1000, 10]), 
            dtype=np.float32
        )

        # --- BIOLOGICAL CONSTANTS (Chlorella sorokiniana CY-1) ---
        self.mu_max = 0.24       # Max growth rate (h^-1)
        self.Ks = 20.0           # Half-saturation constant for nutrients (mg/L)
        self.Ki = 150.0          # Light saturation (umol/m2/s)
        self.Kii = 2000.0        # Photoinhibition constant (umol/m2/s)
        self.z_max = 0.03        # NPBR thickness (3cm)
        self.k_attenuation = 0.5 # Specific light attenuation (m^-1 per OD unit approx?) 
                                 # note: User provided 0.5.
        self.kLa = 15.0          # CO2 mass transfer coefficient
        self.CO2_sat_coeff = 1.5 # Solubility constant
        
        self.state = None
        self.step_count = 0
        self.max_steps_per_episode = 720 # 30 days
        self.dt = 1.0 # 1 hour

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # s1:Water, s2:Light, s3:pH, s4:Temp, s5:Nutrients, s6:Turbidity(OD)
        # Randomize slightly for robustness
        self.state = np.array([45.0, 0.0, 7.5, 25.0, 200.0, 0.1], dtype=np.float32)
        
        self.step_count = 0
        return self._add_noise(self.state), {}

    def _unscale_actions(self, action):
        """Maps normalized PPO actions [-1, 1] to physical values."""
        light_in = np.interp(action[0], [-1, 1], [0, 1500])
        temp_set = np.interp(action[1], [-1, 1], [15, 35])
        co2_flow = np.interp(action[2], [-1, 1], [0, 1])   # L/min
        nute_flow = np.interp(action[3], [-1, 1], [0, 50]) # mL/min
        return light_in, temp_set, co2_flow, nute_flow

    def _add_noise(self, state):
        """
        Adds realistic sensor noise to the state observations.
        Does not affect the underlying physics, only what the agent sees.
        """
        # s1:Water, s2:Light, s3:pH, s4:Temp, s5:Nutrients, s6:Turbidity
        # pH jitters ~0.02
        # Temp jitters ~0.1
        # Turbidity jitters ~0.005
        # Nutrients jitters ~0.5
        noise = np.array([
            0.0,              # Water (Clean)
            5.0,              # Light (Lux fluctuation)
            0.02,             # pH
            0.1,              # Temp
            0.5,              # Nutrients
            0.005             # Turbidity
        ], dtype=np.float32)
        
        # Apply Gaussian noise
        noisy_state = state + np.random.normal(0, noise)
        
        # Clip to valid ranges
        noisy_state = np.clip(noisy_state, self.observation_space.low, self.observation_space.high)
        return noisy_state

    def _calculate_physics(self, action):
        """
        Computes the next state based on differential equations.
        Separated for cleanliness.
        """
        # Unpack State
        s1, s2, s3, s4, s5, s6 = self.state
        # Unpack Action
        light_in, temp_set, co2_flow, nute_flow = self._unscale_actions(action)
        
        # --- MECHANISM 1: Monod-Haldane Hybrid ---
        # Haldane for light inhibition (keeps mu lower at very high light)
        # Avoid div by zero
        denom_light = (self.Ki + light_in + (light_in**2 / self.Kii))
        haldane_light = light_in / denom_light if denom_light > 0 else 0
        
        # Monod for Nutrients
        monod_nutes = s5 / (self.Ks + s5)
        
        # Temp Factor (Gaussian)
        temp_factor = np.exp(-0.5 * ((s4 - 27.0)/5.0)**2)
        
        # Effective specific growth rate contribution
        current_mu = self.mu_max * haldane_light * monod_nutes * temp_factor
        
        # --- Day/Night Cycle Mechanism ---
        hour_of_day = (self.step_count % 24)
        is_day = 6 <= hour_of_day <= 18
        metabolic_mult = 1.0 if is_day else 0.8
        
        current_mu *= metabolic_mult
        
        # --- MECHANISM 2: Huisman Light Attenuation ---
        # k_total = Specific Attenuation * Biomass(OD)
        k_total = max(1e-6, self.k_attenuation * s6)
        
        # Calculate outgoing light
        # I_out = I_in * exp(-k_total * z)
        # However, for depth integration, we need the integral of mu(I) over depth.
        # Analytic solution for Monod-type light response:
        # avg_mu = (mu_max / (k_total * z)) * ln( (H + I_in) / (H + I_out) )
        
        light_out = light_in * np.exp(-k_total * self.z_max)
        
        # Integrated Growth Rate over depth
        if light_in > 1.0:
            # We must use current_mu (which includes Nutes/Temp factors) instead of just mu_max
            integral_factor = np.log((self.Ki + light_in)/(self.Ki + light_out))
            avg_growth_rate = (current_mu / (self.z_max * k_total)) * integral_factor
        else:
            avg_growth_rate = 0.0
            
        # Delta Turbidity (OD)
        # Change in Biomass = Avg Growth Rate * Biomass - Death * Biomass
        # This matches dW/dt = (avg_mu - D) * W
        
        delta_od = (avg_growth_rate * s6) - (0.01 * s6)
        new_od = max(0.01, s6 + delta_od)
        
        # --- MECHANISM 3: Redfield Ratio (Biomass & Nutrients) ---
        # Convert OD to Dry Weight (g/L)
        old_dw = 9.904 * s6 + 0.744
        new_dw = 9.904 * new_od + 0.744
        
        delta_biomass = max(0.0, new_dw - old_dw)
        
        # Nitrogen Uptake (0.07g N per 1g Biomass)
        nitrogen_uptake = delta_biomass * 0.07
        # Update Nutrients (mg/L): Current - Uptake + Inflow
        # Uptake is g/L, convert to mg/L (*1000)
        # Inflow is mL/min? Assuming concentration is high in inflow or this is mass?
        # User prompt: "nute_flow * 0.1". 
        # If flow is 50 mL/min, 0.1 factor implies 5 mg addition? 
        # Let's stick to user's formula.
        new_nutes = max(0, s5 - (nitrogen_uptake * 1000.0) + (nute_flow * 0.1))

        # --- MECHANISM 4: CO2 & pH Dynamics ---
        # CO2 dissolution
        # co2_concentration = co2_flow * self.CO2_sat_coeff 
        
        # pH Change Equation
        # 1. Base increase tendency (kLa towards neutral/equilibrium?)
        # 2. Acidification from CO2 flow
        # 3. Alkalization from Photosynthesis (avg_growth_rate)
        ph_change = (self.kLa * (7.0 - s3) * 0.01) - (co2_flow * 0.5) + (avg_growth_rate * 0.2)
        new_ph = np.clip(s3 + ph_change, 4.0, 10.0)
        
        # --- UPDATE STATE VARIABLES ---
        new_s1 = s1 + (nute_flow * 0.001) - 0.01 # Water Level (Inflow - Evap)
        new_s2 = light_in                        # Light sensor sees input (or average?) usually avg but code said light_in
        new_s3 = new_ph
        new_s4 = s4 * 0.92 + temp_set * 0.08       # Temp inertia (0.08)
        new_s5 = new_nutes
        new_s6 = new_od
        
        next_state = np.array([new_s1, new_s2, new_s3, new_s4, new_s5, new_s6], dtype=np.float32)
        
        return next_state, delta_biomass, light_in, new_ph

    def step(self, action):
        # 1. Calculate Physics
        next_state, delta_biomass, light_in, new_ph = self._calculate_physics(action)
        self.state = next_state
        
        # 2. Calculate Reward
        productivity = delta_biomass # g/L per hour
        # Delegate to external reward logic
        # Pass nutrients state for resource management penalty
        nutrients = next_state[4]
        new_od = next_state[5]
        reward = calculated_reward_function(productivity, new_ph, light_in, nutrients, new_od)
            
        # 3. Check Termination
        self.step_count += 1
        # Done if max steps reached or culture crashes (OD < 0.04)
        # Using 0.04 threshold as requested
        done = self.step_count >= self.max_steps_per_episode or next_state[5] < 0.04
        
        truncated = False # Gym definition
        if self.step_count >= self.max_steps_per_episode:
            truncated = True
            
        info = {
            "productivity": productivity,
            "biomass": (9.904 * next_state[5] + 0.744),
            "pH": new_ph,
            "nutrients": next_state[4]
        }
        
        # Return NOISY state to agent, but keep TRUE state for physics
        obs = self._add_noise(self.state)
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        s = self.state
        print(f"Step {self.step_count} | OD: {s[5]:.4f} | pH: {s[2]:.2f} | Nutes: {s[4]:.2f} | Light: {s[1]:.0f}")

    def close(self):
        pass
