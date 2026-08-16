import numpy as np
from experiment_env import PhotobioreactorExperimentEnv

def debug_step():
    env = PhotobioreactorExperimentEnv()
    env.reset()
    
    # Force a reasonable action
    # Light=0 (mapped to ~750), Temp=0 (~25), CO2=0 (~0.5), Nutrients=0 (~25)
    # Action range is -1 to 1.
    # Light: -1->0, 1->1500. 0 -> 750.
    # Temp: -1->15, 1->35. 0 -> 25.
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    print("--- DEBUGGING PHYSICS STEP ---")
    
    # We will manually run _calculate_physics and print intermediates
    # Note: _calculate_physics returns (next_state, delta_biomass, light_in, new_ph)
    # But we want internal variables inside it.
    # I will stick to calling it and deducing, OR I can modify the env temporarily.
    # Creating a subclass to expose internals is cleaner.
    
    class DebugEnv(PhotobioreactorExperimentEnv):
        def _calculate_physics(self, action):
            # Unpack State
            s1, s2, s3, s4, s5, s6 = self.state
            # Unpack Action
            light_in, temp_set, co2_flow, nute_flow = self._unscale_actions(action)
            
            print(f"State: OD={s6:.4f}, Nutes={s5:.2f}, Temp={s4:.2f}, LightIn={light_in:.2f}")
            
            # --- MECHANISM 1: Monod-Haldane Hybrid ---
            denom_light = (self.Ki + light_in + (light_in**2 / self.Kii))
            haldane_light = light_in / denom_light if denom_light > 0 else 0
            
            monod_nutes = s5 / (self.Ks + s5)
            
            temp_factor = np.exp(-0.5 * ((s4 - 27.0)/5.0)**2)
            
            current_mu = self.mu_max * haldane_light * monod_nutes * temp_factor
            
            print(f"Factors: Haldane(Light)={haldane_light:.4f}, Monod(Nutes)={monod_nutes:.4f}, Temp={temp_factor:.4f}")
            print(f"Current Mu = {current_mu:.4f} (Max {self.mu_max})")
            
            # --- MECHANISM 2: Huisman Light Attenuation ---
            light_out = light_in * np.exp(-self.k_attenuation * s6 * self.z_max)
            print(f"Light Out = {light_out:.4f}")
            
            if light_in > 1.0:
                growth_term = (current_mu / self.z_max) * np.log((self.Ki + light_in)/(self.Ki + light_out))
            else:
                growth_term = 0.0
                
            print(f"Growth Term (Integral) = {growth_term:.4f}")
            
            death_term = 0.01 * s6
            growth_contribution = growth_term * s6
            
            print(f"Delta Calculation: Growth({growth_contribution:.6f}) - Death({death_term:.6f}) = {growth_contribution - death_term:.6f}")
            
            delta_od = growth_contribution - death_term
            new_od = max(0.01, s6 + delta_od)
            
            # Continue with rest for completeness (but we found the interesting part)
            return super()._calculate_physics(action)

    debug_env = DebugEnv()
    debug_env.reset()
    debug_env.step(action)

if __name__ == "__main__":
    debug_step()
