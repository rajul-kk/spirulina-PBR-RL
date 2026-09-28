"""Nutrient and carbon extremes: N starvation and P starvation with dosing disabled, and a
culture with the pH-stat CO2 feed switched off. Books the N that enters biomass against the N
that actually left the medium.

  python experiments/env_diagnosis/fidelity/starvation.py
"""
import numpy as np

from common import const, run


def n_starve(env):
    env.N_DOSE_RATE = 0.0
    env.n_pool = 20.0


def p_starve(env):
    env.N_DOSE_RATE = 0.0
    env.p_pool = 2.0


def no_co2(env):
    env.CO2_MAX_LPM = 0.0


def report(name, setup, init=700, hours=(0, 12, 24, 48, 96, 143)):
    r = run(const(65, 1400), init_cells=init, difficulty=0, seed=4, setup=setup, record_every=50)
    env, L = r["env"], r["ledger"]
    print(f"\n=== {name} (init {init}, 1400 umol, no harvest, D0)")
    print("   hour    X mg/L   n_pool  p_pool    pH   mu(mean gross)")
    for row in r["trace"]:
        if int(round(row["t_h"])) in hours and abs(row["t_h"] - round(row["t_h"])) < 1e-6:
            print(f"   {row['t_h']:5.0f}  {row['od'] * 300:8.0f} {row['n_pool']:8.1f} {row['p_pool']:7.1f} "
                  f"{row['ph']:6.2f}  {row['mu']:.4f}")
    q = env.cells_quota[env.active_mask]
    B_end = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    gained = B_end - L["B0"] + L["lysed"]
    print(f"   biomass gained (incl. later lysed) {gained:.0f} mg needs {env.N_FRAC * gained:.0f} mg N at N_FRAC; "
          f"medium supplied {(L['N0_pool'] - env.n_pool * env.volume_L) + L['n_dosed']:.0f} mg N")
    print(f"   final quota mean {q.mean():.2f} (Q_min {env.strain_params['Q_min']}, Q_max {env.strain_params['Q_max']}), "
          f"crash_t {r['crash_t']}, NaN/neg flags {r['n_bad']}")


if __name__ == "__main__":
    report("N starvation (n_pool 20 mg/L, dosing off)", n_starve)
    report("P starvation (p_pool 2 mg/L, dosing off)", p_starve)
    report("no pH-stat CO2 (air sparge only)", no_co2)
    report("no pH-stat CO2, dense (3000 cells)", no_co2, init=3000)
