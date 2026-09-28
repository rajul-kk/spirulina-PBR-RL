"""Volumetric productivity (g/L/day) and photon use (g biomass per mol photons) of the scripted
expert's proportional harvest at D2, on this env vs another env file.

  python experiments/env_diagnosis/fidelity/productivity.py [path/to/other_genetic_env.py]
"""
import sys

import numpy as np

from common import MAX_CELLS, act
from expert_impact import load
from genetic_env import GeneticPhotobioreactorEnv


def run(cls, stir, light, setpoint, seed, init=700):
    np.random.seed(seed)
    env = cls(max_cells=MAX_CELLS, initial_cells=init, difficulty=2)
    env.reset(seed=seed)
    m0 = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    done = False
    while not done:
        frac = float(np.clip(env.od / setpoint - 1.0, 0.0, 0.5))
        _, _, term, trunc, _ = env.step(act(stir, light, frac))
        done = term or trunc
    m1 = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    days = env.step_count * env.dt / 24.0
    grown_g = (env.cumulative_harvested_mg + m1 - m0) / 1000.0
    area = env.volume_L / 1000.0 / env.light_path_m                     # lit face, m^2
    photons = light * 1e-6 * 86400.0 * days * area                     # mol
    return grown_g / (env.volume_L * days), grown_g / photons, env.cumulative_harvested_mg / 1000.0, float(env.temp)


if __name__ == "__main__":
    cls = load(sys.argv[1]) if len(sys.argv) > 1 else GeneticPhotobioreactorEnv
    print(f"env: {sys.argv[1] if len(sys.argv) > 1 else 'current'}")
    for stir in (50, 100, 150, 200):
        for light, sp in ((1400, 0.6), (1400, 2.0), (300, 0.6)):
            r = np.array([run(cls, stir, light, sp, s) for s in (11, 12)])
            print(f"  stir {stir:3d} light {light:4d} setpoint {sp}: {r[:,0].mean():.3f} g/L/day, "
                  f"{r[:,1].mean():.3f} g/mol photons, harvest {r[:,2].mean():.1f} g, T {r[:,3].mean():.1f} C", flush=True)
