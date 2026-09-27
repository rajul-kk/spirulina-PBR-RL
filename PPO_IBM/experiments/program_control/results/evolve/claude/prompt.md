# Task: write a photobioreactor controller program

You are designing a controller for a simulated Spirulina photobioreactor (150 L-scale, 144 h
batch, 7200 control steps of 72 s). The goal is to HARVEST AS MUCH BIOMASS AS POSSIBLE over the
batch WITHOUT the culture dying out (extinction = crash, heavily penalised). Starting
populations vary widely (30 to 5000 cells-equivalent), strains vary per episode (max growth
rate, temperature optimum), and at the evaluated difficulty sensors are noisy, drift and lag.

## Interface (the whole file is your answer)

```python
class Controller:
    def __init__(self, params=None): ...
    def act(self, obs: dict) -> tuple:   # (stir_rpm, light_umol, harvest_frac)
```
- stir_rpm 50-200 (mixing; more stir = better gas exchange but more shear/heat)
- light_umol 0-2000 (LED intensity; drives growth but saturates, photo-inhibits and heats the
  tank; the thermostat can only remove limited heat)
- harvest_frac 0-0.5

obs keys (all sensor readings carry noise; at D1+ also drift, lag and a pH bias):
  turbidity_ntu   nephelometer, ~250 NTU per OD unit at the start of a batch; reads low as cells
                  clump over days (up to ~30%), saturates at 1000
  ph              pH probe (Zarrouk medium, pH-stat CO2 keeps it near 10)
  pump_L          cumulative harvest pump volume, litres
  conductivity    uS/cm
  temp_c          broth temperature, C (thermostat setpoint 35; strong light heats the tank)
  lux             BH1750 reading of the LED panel, ~30 lux per umol/m2/s (echoes your light)
  t               step index (1 step = 0.02 h = 72 s; episode = 7200 steps = 144 h)
  dt_h            0.02
Harvest: every 600 steps (12 h) the env removes the MEAN of the harvest_frac values you output
over the preceding 600 steps (cap 0.5) and refills with fresh medium. Nutrients are dosed
automatically. OD target ~0.75 (1 OD unit = 300 mg/L dry weight).

Allowed imports: numpy, math, collections, process_state. `process_state.ProcessState` is an
optional helper: `card = ProcessState().update(obs)` returns od_est (turbidity/250),
od_est_slow, growth_per_h, temp_c (smoothed), hours, hours_to_harvest, harvests_done, pump_L,
ph, light_obs_umol. You may keep any state you like on self. No file, network or OS access.
act() is called 7200 times per episode, so keep it cheap (well under 1 ms).

## Scoring
12 episodes per candidate. fitness = 0.5*median_harvest_mg + 0.5*p25_harvest_mg
- 10000*crash_rate (median/p25 over episodes starting above 80 cells; tiny starts are scored
on survival). A program that raises an exception loses the rest of that episode.

Reply with ONE complete Python file in a single ```python block, then at most 3 lines on what
you changed and why.


## Results so far (search split, best first)

- cand_001.py: fitness 10126, median 11544 mg, p25 8708 mg, crash 0% - calibrated K(h,turb); predictive harvest to od_post 0.55; cap 0.5
- cand_000.py: fitness 9186, median 10387 mg, p25 7985 mg, crash 0% - seed: hand-tuned sensor expert

## Program cand_001.py (fitness 10126)
```python
"""cand_001 (Claude): calibrated OD estimate + predictive harvest to a post-harvest target.

- Turbidity-per-OD drifts from ~250 to ~130 NTU as cells clump and the sensor saturates, so
  the fixed 250 under-reads OD and the culture runs 1.3-1.9x target. K(h, turb) below was fitted
  offline on calibration batches (the simulator stand-in for dry-weight samples).
- Harvest removes the mean frac over the interval, so aim each step's frac at the OD predicted
  for the next event, sized to land at OD_POST after the harvest.
- Dense starts can use the full 0.5 fraction instead of the old 0.3 cap.
"""
import math

import numpy as np

from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "od_post": 0.55, "cap": 0.5, "min_od_to_harvest": 0.6}


def ntu_per_od(hours, turb):
    return math.exp(6.787 - 0.0018 * hours - 0.2943 * math.log(max(turb, 1.0)))


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState()

    def act(self, obs):
        p = self.p
        c = self.ps.update(obs)
        turb = self.ps.turb_fast
        od = turb / ntu_per_od(c["hours"], turb)
        g = float(np.clip(c["growth_per_h"], -0.05, 0.08))
        od_at_event = od * math.exp(g * c["hours_to_harvest"])
        if od_at_event < p["min_od_to_harvest"]:
            frac = 0.0
        else:
            frac = float(np.clip(1.0 - p["od_post"] / od_at_event, 0.0, p["cap"]))
        return p["stir"], p["light"], frac

```
Per-episode results:
  init=  214 harvested=   11876 mg steps= 7200 time_avg_od=0.64 
  init=  144 harvested=    8106 mg steps= 7200 time_avg_od=0.60 
  init=  101 harvested=    6660 mg steps= 7200 time_avg_od=0.60 
  init=  163 harvested=   10513 mg steps= 7200 time_avg_od=0.65 
  init=  164 harvested=    6843 mg steps= 7200 time_avg_od=0.60 
  init=   66 harvested=    5825 mg steps= 7200 time_avg_od=0.60 
  init=   39 harvested=    4396 mg steps= 7200 time_avg_od=0.58 
  init=  230 harvested=   16461 mg steps= 7200 time_avg_od=0.70 
  init=  242 harvested=   11212 mg steps= 7200 time_avg_od=0.63 
  init= 2674 harvested=   37709 mg steps= 7200 time_avg_od=0.74 
  init= 1646 harvested=   29385 mg steps= 7200 time_avg_od=0.68 
  init= 1678 harvested=   27472 mg steps= 7200 time_avg_od=0.66 
  trace init=101 (every 12 h; true_* fields are simulator truth, not visible to the controller):
    {"t_h": 0.0, "turb": 50.9, "temp_obs": 36.13, "ph": 9.77, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.28, "true_temp": 37.92, "cells": 101, "harvested_mg": 0.0}
    {"t_h": 12.0, "turb": 65.0, "temp_obs": 33.54, "ph": 9.95, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.357, "true_temp": 35.18, "cells": 198, "harvested_mg": 0.0}
    {"t_h": 24.0, "turb": 83.9, "temp_obs": 34.63, "ph": 9.75, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.476, "true_temp": 35.18, "cells": 197, "harvested_mg": 0.0}
    {"t_h": 36.0, "turb": 110.9, "temp_obs": 34.84, "ph": 9.84, "stir": 65.0, "light": 1400.0, "frac": 0.317, "true_od_over_target": 0.627, "true_temp": 35.18, "cells": 391, "harvested_mg": 0.0}
    {"t_h": 48.0, "turb": 134.0, "temp_obs": 33.9, "ph": 10.07, "stir": 65.0, "light": 1400.0, "frac": 0.335, "true_od_over_target": 0.812, "true_temp": 35.18, "cells": 389, "harvested_mg": 0.0}
    {"t_h": 60.0, "turb": 136.3, "temp_obs": 34.32, "ph": 9.94, "stir": 65.0, "light": 1400.0, "frac": 0.37, "true_od_over_target": 0.86, "true_temp": 35.18, "cells": 319, "harvested_mg": 620.3}
    {"t_h": 72.0, "turb": 151.1, "temp_obs": 34.07, "ph": 10.19, "stir": 65.0, "light": 1400.0, "frac": 0.411, "true_od_over_target": 0.941, "true_temp": 35.18, "cells": 548, "harvested_mg": 1165.9}
    {"t_h": 84.0, "turb": 141.1, "temp_obs": 34.02, "ph": 10.22, "stir": 65.0, "light": 1400.0, "frac": 0.418, "true_od_over_target": 0.917, "true_temp": 35.18, "cells": 417, "harvested_mg": 2155.4}
    {"t_h": 96.0, "turb": 128.5, "temp_obs": 34.6, "ph": 10.18, "stir": 65.0, "light": 1400.0, "frac": 0.371, "true_od_over_target": 0.895, "true_temp": 35.18, "cells": 317, "harvested_mg": 3125.6}
    {"t_h": 108.0, "turb": 140.3, "temp_obs": 33.95, "ph": 10.01, "stir": 65.0, "light": 1400.0, "frac": 0.383, "true_od_over_target": 0.904, "true_temp": 35.18, "cells": 502, "harvested_mg": 3928.6}
    {"t_h": 120.0, "turb": 127.5, "temp_obs": 33.62, "ph": 10.1, "stir": 65.0, "light": 1400.0, "frac": 0.406, "true_od_over_target": 0.876, "true_temp": 35.18, "cells": 376, "harvested_mg": 4951.2}
    {"t_h": 132.0, "turb": 132.0, "temp_obs": 33.86, "ph": 10.26, "stir": 65.0, "light": 1400.0, "frac": 0.433, "true_od_over_target": 0.915, "true_temp": 35.18, "cells": 306, "harvested_mg": 5675.3}
  trace init=2674 (every 12 h; true_* fields are simulator truth, not visible to the controller):
    {"t_h": 0.0, "turb": 959.5, "temp_obs": 34.56, "ph": 9.81, "stir": 65.0, "light": 1400.0, "frac": 0.5, "true_od_over_target": 6.951, "true_temp": 35.65, "cells": 2674, "harvested_mg": 0.0}
    {"t_h": 12.0, "turb": 693.1, "temp_obs": 33.64, "ph": 9.88, "stir": 65.0, "light": 1400.0, "frac": 0.5, "true_od_over_target": 7.161, "true_temp": 35.18, "cells": 2658, "harvested_mg": 0.0}
    {"t_h": 24.0, "turb": 372.0, "temp_obs": 33.95, "ph": 9.84, "stir": 65.0, "light": 1400.0, "frac": 0.5, "true_od_over_target": 3.806, "true_temp": 35.18, "cells": 1324, "harvested_mg": 16071.1}
    {"t_h": 36.0, "turb": 231.2, "temp_obs": 34.09, "ph": 9.66, "stir": 65.0, "light": 1400.0, "frac": 0.5, "true_od_over_target": 2.084, "true_temp": 35.18, "cells": 1286, "harvested_mg": 24831.5}
    {"t_h": 48.0, "turb": 154.1, "temp_obs": 34.47, "ph": 9.97, "stir": 65.0, "light": 1400.0, "frac": 0.387, "true_od_over_target": 1.359, "true_temp": 35.18, "cells": 698, "harvested_mg": 29067.7}
    {"t_h": 60.0, "turb": 136.4, "temp_obs": 34.05, "ph": 9.78, "stir": 65.0, "light": 1400.0, "frac": 0.354, "true_od_over_target": 1.225, "true_temp": 35.18, "cells": 518, "harvested_mg": 30627.4}
    {"t_h": 72.0, "turb": 129.1, "temp_obs": 33.9, "ph": 9.97, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 1.244, "true_temp": 35.18, "cells": 437, "harvested_mg": 31482.4}
    {"t_h": 84.0, "turb": 154.7, "temp_obs": 34.39, "ph": 10.04, "stir": 65.0, "light": 1400.0, "frac": 0.465, "true_od_over_target": 1.248, "true_temp": 35.18, "cells": 726, "harvested_mg": 32388.6}
    {"t_h": 96.0, "turb": 126.3, "temp_obs": 34.01, "ph": 9.92, "stir": 65.0, "light": 1400.0, "frac": 0.302, "true_od_over_target": 1.044, "true_temp": 35.18, "cells": 498, "harvested_mg": 34141.2}
    {"t_h": 108.0, "turb": 131.2, "temp_obs": 34.41, "ph": 9.82, "stir": 65.0, "light": 1400.0, "frac": 0.355, "true_od_over_target": 1.11, "true_temp": 35.18, "cells": 434, "harvested_mg": 34726.7}
    {"t_h": 120.0, "turb": 136.9, "temp_obs": 33.87, "ph": 9.74, "stir": 65.0, "light": 1400.0, "frac": 0.5, "true_od_over_target": 1.111, "true_temp": 35.18, "cells": 567, "harvested_mg": 35611.3}
    {"t_h": 132.0, "turb": 127.7, "temp_obs": 33.52, "ph": 9.97, "stir": 65.0, "light": 1400.0, "frac": 0.396, "true_od_over_target": 1.021, "true_temp": 35.18, "cells": 535, "harvested_mg": 36830.9}

## Program cand_000.py (fitness 9186)
```python
"""The demo expert's proportional-harvest law on sensors only: OD is estimated from turbidity.
DEFAULTS are the hand-tuned values; cmaes_tune.py searches over exactly these keys."""
import numpy as np

from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0}


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        return p["stir"], p["light"], frac

```
Per-episode results:
  init=  214 harvested=   10565 mg steps= 7200 time_avg_od=0.99 
  init=  144 harvested=    7457 mg steps= 7200 time_avg_od=0.95 
  init=  101 harvested=    5302 mg steps= 7200 time_avg_od=0.94 
  init=  163 harvested=    9572 mg steps= 7200 time_avg_od=1.04 
  init=  164 harvested=    5554 mg steps= 7200 time_avg_od=0.92 
  init=   66 harvested=    4246 mg steps= 7200 time_avg_od=0.91 
  init=   39 harvested=    2669 mg steps= 7200 time_avg_od=0.81 
  init=  230 harvested=   16264 mg steps= 7200 time_avg_od=1.11 
  init=  242 harvested=   10209 mg steps= 7200 time_avg_od=1.03 
  init= 2674 harvested=   35117 mg steps= 7200 time_avg_od=1.28 
  init= 1646 harvested=   27532 mg steps= 7200 time_avg_od=1.19 
  init= 1678 harvested=   24272 mg steps= 7200 time_avg_od=1.19 
  trace init=101 (every 12 h; true_* fields are simulator truth, not visible to the controller):
    {"t_h": 0.0, "turb": 50.9, "temp_obs": 36.13, "ph": 9.77, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.28, "true_temp": 37.92, "cells": 101, "harvested_mg": 0.0}
    {"t_h": 12.0, "turb": 65.0, "temp_obs": 33.54, "ph": 9.95, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.357, "true_temp": 35.18, "cells": 198, "harvested_mg": 0.0}
    {"t_h": 24.0, "turb": 83.9, "temp_obs": 34.63, "ph": 9.75, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.476, "true_temp": 35.18, "cells": 197, "harvested_mg": 0.0}
    {"t_h": 36.0, "turb": 110.9, "temp_obs": 34.84, "ph": 9.84, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.627, "true_temp": 35.18, "cells": 391, "harvested_mg": 0.0}
    {"t_h": 48.0, "turb": 131.4, "temp_obs": 33.96, "ph": 9.98, "stir": 65.0, "light": 1400.0, "frac": 0.0, "true_od_over_target": 0.807, "true_temp": 35.18, "cells": 386, "harvested_mg": 0.0}
    {"t_h": 60.0, "turb": 157.6, "temp_obs": 33.84, "ph": 9.97, "stir": 65.0, "light": 1400.0, "frac": 0.039, "true_od_over_target": 1.021, "true_temp": 35.18, "cells": 385, "harvested_mg": 0.0}
    {"t_h": 72.0, "turb": 189.5, "temp_obs": 34.75, "ph": 10.18, "stir": 65.0, "light": 1400.0, "frac": 0.276, "true_od_over_target": 1.238, "true_temp": 35.18, "cells": 762, "harvested_mg": 23.9}
    {"t_h": 84.0, "turb": 182.9, "temp_obs": 34.85, "ph": 10.11, "stir": 65.0, "light": 1400.0, "frac": 0.241, "true_od_over_target": 1.3, "true_temp": 35.18, "cells": 655, "harvested_mg": 784.5}
    {"t_h": 96.0, "turb": 176.2, "temp_obs": 33.92, "ph": 10.2, "stir": 65.0, "light": 1400.0, "frac": 0.166, "true_od_over_target": 1.282, "true_temp": 35.18, "cells": 529, "harvested_mg": 1865.5}
    {"t_h": 108.0, "turb": 177.1, "temp_obs": 34.48, "ph": 10.1, "stir": 65.0, "light": 1400.0, "frac": 0.189, "true_od_over_target": 1.372, "true_temp": 35.18, "cells": 470, "harvested_mg": 2466.7}
    {"t_h": 120.0, "turb": 196.5, "temp_obs": 34.53, "ph": 10.03, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 1.445, "true_temp": 35.18, "cells": 830, "harvested_mg": 3138.8}
    {"t_h": 132.0, "turb": 177.9, "temp_obs": 34.22, "ph": 10.15, "stir": 65.0, "light": 1400.0, "frac": 0.195, "true_od_over_target": 1.352, "true_temp": 35.18, "cells": 641, "harvested_mg": 4569.6}
  trace init=2674 (every 12 h; true_* fields are simulator truth, not visible to the controller):
    {"t_h": 0.0, "turb": 959.5, "temp_obs": 34.56, "ph": 9.81, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 6.951, "true_temp": 35.65, "cells": 2674, "harvested_mg": 0.0}
    {"t_h": 12.0, "turb": 693.1, "temp_obs": 33.64, "ph": 9.88, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 7.161, "true_temp": 35.18, "cells": 2658, "harvested_mg": 0.0}
    {"t_h": 24.0, "turb": 462.6, "temp_obs": 33.86, "ph": 9.79, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 5.212, "true_temp": 35.18, "cells": 1848, "harvested_mg": 9726.6}
    {"t_h": 36.0, "turb": 325.5, "temp_obs": 33.92, "ph": 9.71, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 3.857, "true_temp": 35.18, "cells": 1286, "harvested_mg": 16696.1}
    {"t_h": 48.0, "turb": 287.3, "temp_obs": 34.58, "ph": 9.86, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 2.869, "true_temp": 35.18, "cells": 1760, "harvested_mg": 22133.2}
    {"t_h": 60.0, "turb": 218.9, "temp_obs": 34.54, "ph": 9.97, "stir": 65.0, "light": 1400.0, "frac": 0.3, "true_od_over_target": 2.237, "true_temp": 35.18, "cells": 1226, "harvested_mg": 25976.0}
    {"t_h": 72.0, "turb": 169.3, "temp_obs": 33.55, "ph": 9.68, "stir": 65.0, "light": 1400.0, "frac": 0.144, "true_od_over_target": 1.785, "true_temp": 35.18, "cells": 850, "harvested_mg": 28997.7}
    {"t_h": 84.0, "turb": 171.8, "temp_obs": 34.13, "ph": 9.94, "stir": 65.0, "light": 1400.0, "frac": 0.165, "true_od_over_target": 1.847, "true_temp": 35.18, "cells": 769, "harvested_mg": 29748.5}
    {"t_h": 96.0, "turb": 171.0, "temp_obs": 34.59, "ph": 10.03, "stir": 65.0, "light": 1400.0, "frac": 0.135, "true_od_over_target": 1.846, "true_temp": 35.18, "cells": 676, "harvested_mg": 30731.0}
    {"t_h": 108.0, "turb": 190.7, "temp_obs": 34.71, "ph": 9.96, "stir": 65.0, "light": 1400.0, "frac": 0.266, "true_od_over_target": 1.925, "true_temp": 35.18, "cells": 916, "harvested_mg": 31346.2}
    {"t_h": 120.0, "turb": 195.3, "temp_obs": 34.29, "ph": 9.79, "stir": 65.0, "light": 1400.0, "frac": 0.297, "true_od_over_target": 1.897, "true_temp": 35.18, "cells": 1077, "harvested_mg": 32424.2}
    {"t_h": 132.0, "turb": 168.7, "temp_obs": 34.08, "ph": 10.02, "stir": 65.0, "light": 1400.0, "frac": 0.136, "true_od_over_target": 1.677, "true_temp": 35.18, "cells": 838, "harvested_mg": 34368.3}

Write an improved controller. Improve on the best program; do not just resubmit it.