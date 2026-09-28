"""Emergent kinetics for comparison with published Spirulina values: specific growth rate vs
light and temperature in a thin culture, dark biomass loss, and the implied quantum requirement.

  python experiments/env_diagnosis/fidelity/kinetics.py
"""
import numpy as np

from common import MAX_CELLS, act
from genetic_env import GeneticPhotobioreactorEnv


def mu_thin(light, temp=None, stir=65.0, seed=5, init=60, h0=6.0, h1=30.0, difficulty=0):
    """Net specific growth rate (1/h) of total biomass between h0 and h1, no harvest, thin culture."""
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init, difficulty=difficulty)
    env.reset(seed=seed)
    if temp is not None:
        env.T_SETPOINT, env.temp = temp, temp
        env.COOL_MAX_C_PER_H = env.HEAT_MAX_C_PER_H = 50.0   # hold temperature exactly
    a = act(stir, light, 0.0)
    B = {}
    for t in range(int(h1 / env.dt)):
        env.step(a)
        h = round((t + 1) * env.dt, 4)
        if h in (h0, h1):
            B[h] = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    X = B[h1] / env.volume_L
    return np.log(B[h1] / B[h0]) / (h1 - h0), X, env.strain_params


if __name__ == "__main__":
    print("=== P-I: net mu (1/h) in a thin culture (60 agents, ~30-60 mg/L), D0, mean of 3 strains")
    for I in (0, 50, 100, 200, 400, 700, 1000, 1400, 2000):
        vals = [mu_thin(I, temp=35.0, seed=s) for s in (5, 6, 7)]
        mu = np.mean([v[0] for v in vals])
        print(f"   I {I:5d} umol  mu {mu:+.4f}/h  ({mu * 24:+.2f}/day, doubling "
              f"{(np.log(2) / mu if mu > 0 else float('nan')):.0f} h)  end X ~{np.mean([v[1] for v in vals]):.0f} mg/L")
    print("\n=== temperature: net mu at 700 umol, temperature held")
    for T in (15, 20, 25, 30, 33, 35, 37, 39, 41, 43, 45):
        vals = [mu_thin(700, temp=float(T), seed=s) for s in (5, 6, 7)]
        print(f"   T {T:4.0f} C  mu {np.mean([v[0] for v in vals]):+.4f}/h")
    print("\n=== dark loss (light 0, 35 C): net mu over 6-30 h")
    vals = [mu_thin(0, temp=35.0, seed=s, init=700) for s in (5, 6, 7)]
    mu = np.mean([v[0] for v in vals])
    print(f"   {mu:+.4f}/h = {(1 - np.exp(mu * 24)):.1%} of biomass per day, "
          f"{(1 - np.exp(mu * 12)):.1%} per 12 h night")

    # Implied quantum requirement in the light-limited limit, from the model's own constants.
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=60, difficulty=0)
    np.random.seed(5); env.reset(seed=5)
    p = env.strain_params
    mu_eff = 0.040 * 0.9 * 0.95 * 0.93            # mean mu_max x f_Q x f_P x f_carbon at Zarrouk
    Ks, Ki = 100.0, 2500.0
    f_max = env.RED_FRAC * np.sqrt(Ks * Ki) / (2 * Ks + env.RED_FRAC * np.sqrt(Ks * Ki))
    alpha = mu_eff * env.RED_FRAC / Ks / f_max    # 1/h per umol/m2/s total PAR, low-light slope
    a_PAR = (env.RED_FRAC * env.EXT_RED + env.BLUE_FRAC * env.EXT_BLUE + env.GREEN_FRAC * env.EXT_GREEN)
    photons = 1e-6 * a_PAR * 3600.0               # mol photons absorbed per g DW per h per umol/m2/s
    carbon = alpha * env.C_FRAC / 12.0            # mol C fixed per g DW per h per umol/m2/s
    print(f"\n=== light-limited efficiency: slope alpha {alpha:.2e} /h per umol/m2/s, specific extinction "
          f"{a_PAR:.3f} m2/g")
    print(f"   quantum requirement at low light ~ {photons / carbon:.0f} photons per C fixed "
          f"(if all extinction were absorption; physiological minimum ~8-10, measured 10-20)")
