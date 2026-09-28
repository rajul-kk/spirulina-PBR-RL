"""Photo-shock direction: a culture acclimated to 2000 umol that is turned DOWN to 1000 should grow
about as fast as one that sat at 1000 all along (lower light is not photoinhibitory). Also the
upward case (1000 -> 2000), where a transient penalty is physiological.

  python experiments/env_diagnosis/fidelity/photoshock.py
"""
import numpy as np

from common import MAX_CELLS, act
from genetic_env import GeneticPhotobioreactorEnv


def growth_after_switch(first, second, init=300, seed=11, pre_h=24.0, post_h=4.0):
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init, difficulty=0)
    env.reset(seed=seed)
    for _ in range(int(pre_h / env.dt)):
        env.step(act(65, first, 0.0))
    B0 = float(np.sum(env.cells_mass[env.active_mask]))
    shocks = []
    for _ in range(int(post_h / env.dt)):
        env.step(act(65, second, 0.0))
        shocks.append(env.debug_shock)
    B1 = float(np.sum(env.cells_mass[env.active_mask]))
    return np.log(B1 / B0) / post_h, float(np.min(shocks)), float(np.mean(shocks))


if __name__ == "__main__":
    for a, b in ((1000, 1000), (2000, 1000), (2000, 500), (500, 500), (1000, 2000), (2000, 2000)):
        res = [growth_after_switch(a, b, seed=s) for s in (11, 12, 13)]
        mu, smin, smean = (np.mean([r[i] for r in res]) for i in range(3))
        print(f"  {a:4d} -> {b:4d} umol: net mu over next 4 h {mu:+.4f}/h, shock factor min {smin:.2f} mean {smean:.2f}")
