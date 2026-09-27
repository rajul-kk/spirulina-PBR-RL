"""Reference only: the TD3 demo expert's law, reading the simulator's TRUE OD (privileged).
Run with --privileged. No real reactor can do this; it bounds what the sensor-only
controllers lose to measurement."""
import numpy as np

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30}


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))

    def act(self, obs):
        p = self.p
        frac = float(np.clip(p["gain"] * (obs["true_od"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        return p["stir"], p["light"], frac
