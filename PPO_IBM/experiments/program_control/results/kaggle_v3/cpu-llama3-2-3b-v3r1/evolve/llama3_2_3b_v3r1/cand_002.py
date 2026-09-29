import numpy as np
from process_state import ProcessState


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Use a more sophisticated light control based on lux and light umol
        if 'lux' in obs:
            light_control = np.clip((obs['lux'] / 30) * (p["light_umol"] - 1000), 0, p["light_umol"])
            light_control = np.clip(light_control, 0, p["light_umol"])
            light_control = np.clip(light_control * 0.7, 0, p["light_umol"])  # Reduce light control to prevent overheating
        else:
            light_control = np.clip((p["lux"] / 30) * (p["light_umol"] - 1000), 0, p["light_umol"])
            light_control = np.clip(light_control, 0, p["light_umol"])
            light_control = np.clip(light_control * 0.7, 0, p["light_umol"])  # Reduce light control to prevent overheating

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip((p["conductivity"] / 50) * p["stir_threshold"], 0, p["conductivity_threshold"])
        stir_control = np.clip(stir_control * p["stir"], 0, p["conductivity_threshold"])

        # Use a more sophisticated harvest control based on pump volume and harvest fraction
        harvest_control = np.clip((p["pump_L"] / 10) * p["harvest_frac"], 0, p["harvest_frac"])
        harvest_control = np.clip(harvest_control * 0.5, 0, p["harvest_frac"])  # Reduce harvest control to prevent over-harvesting

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control
        frac = frac * harvest_control

        return p["stir"], light_control, frac


DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7,
            "harvest_frac": 0.4, "pump_L_threshold": 5, "harvest_frac_threshold": 0.5}
