"""Adapter so a TD3 actor checkpoint runs in the program harness, on the same episodes as every
program. params: {"actor_path": ..., "reset": 600}. Mirrors run_td3_eval_episode: the recurrent
state is zeroed every `reset` steps."""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, "td3"))

import torch  # noqa: E402

torch.set_num_threads(1)
from actor_io import load_actor  # noqa: E402

_CACHE = {}


class Controller:
    def __init__(self, params=None):
        p = params or {}
        path = p["actor_path"]
        if path not in _CACHE:
            _CACHE[path] = load_actor(path, device=torch.device("cpu"))[0]
        self.actor = _CACHE[path]
        self.reset = int(p.get("reset", 600))
        self.hidden = self.actor.initial_hidden(batch=1)
        self.since = 0

    def act(self, obs):
        if self.since >= self.reset:
            self.hidden = self.actor.initial_hidden(batch=1)
            self.since = 0
        x = torch.tensor([obs["turbidity_ntu"], obs["ph"], obs["pump_L"], obs["conductivity"],
                          obs["temp_c"], obs["lux"]], dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            a, self.hidden = self.actor(x, self.hidden)
        a = a.view(-1).numpy()
        self.since += 1
        return (float(np.interp(a[0], [-1, 1], [50, 200])), float(np.interp(a[1], [-1, 1], [0, 2000])),
                float(np.interp(a[2], [-1, 1], [0, 0.5])))
