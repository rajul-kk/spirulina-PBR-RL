import numpy as np
from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.3,
            "turb_per_od": 250.0}

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.t = 0
        self.harvested_mg = 0
        self.steps_since_harvest = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip((card["od_est"] - p["setpoint"]) / (card["od_est"] + 1), 0.0, p["cap"]))
        if self.t % 600 == 0 and self.steps_since_harvest >= 720:
            self.harvested_mg = self.get_harvested_mg()
            self.t = 0
            self.steps_since_harvest = 0
        stir = np.clip(p["stir"] + self.t * 1, 50, 200)
        light = np.clip(p["light"] + self.t * 10, 0, 2000)
        self.steps_since_harvest += 1
        return stir, light, frac

    def get_harvested_mg(self):
        if self.harvested_mg == 0:
            return np.mean([x["harvested_mg"] for x in self.get_harvest_episodes()])
        return np.mean([x["harvested_mg"] for x in self.get_harvest_episodes()])
    def get_harvest_episodes(self):
        return [x for x in self.get_all_episodes() if x["t_h"] % 600 == 0]
    def get_all_episodes(self):
        return [{"t_h": self.t, "turb": x["turb"], "temp_obs": x["temp_obs"], "ph": x["ph"], "stir": x["stir"], "light": x["light"], "frac": x["frac"], "true_od_over_target": x["true_od_over_target"], "true_temp": x["true_temp"], "cells": x["cells"], "harvested_mg": x["harvested_mg"]} for self.t in range(7200)]
