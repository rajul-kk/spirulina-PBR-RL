"""Biomass and nitrogen conservation, and determinism under a fixed seed.

Biomass ledger: B_end = B0 + net biology (growth - respiration - lysis - mass cap) - harvested.
Nitrogen: N_tot = n_pool*V + N_FRAC*B. Every mg of N entering (dosing, refill) or leaving (harvested
liquid, harvested biomass) is booked; the residual is N that vanishes inside the model.

  python experiments/env_diagnosis/fidelity/balance.py
"""
import numpy as np

from common import LedgerEnv, const, expert, run


def n_report(name, r):
    env, L = r["env"], r["ledger"]
    V, NF = env.volume_L, env.N_FRAC
    B_end = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    B_pred = L["B0"] + L["bio_net"] - L["harvested"]
    N0 = L["N0_pool"] + NF * L["B0"]
    N_end = env.n_pool * V + NF * B_end
    N_pred = N0 + L["n_dosed"] + L["n_refill"] - L["n_harv_liquid"] - NF * L["harvested"]
    grown_N = L["n_drawn"]                    # N drawn from the medium into biomass
    print(f"\n=== {name}: {r['steps']} steps, harvested {L['harvested']:.0f} mg, lysed {L['lysed']:.0f} mg")
    print(f"  biomass: B0 {L['B0']:.0f}  B_end {B_end:.0f}  ledger {B_pred:.0f}  (closing error {B_end - B_pred:+.2f} mg)")
    print(f"  N: start {N0:.0f} mg, dosed {L['n_dosed']:.0f}, refill {L['n_refill']:.0f}, out liquid "
          f"{L['n_harv_liquid']:.0f}, out biomass {NF * L['harvested']:.0f}")
    print(f"     end {N_end:.0f} vs conserved {N_pred:.0f}: {N_pred - N_end:.0f} mg N vanished "
          f"({(N_pred - N_end) / max(grown_N, 1):.0%} of the {grown_N:.0f} mg N taken up)")
    print(f"     N drawn {grown_N:.0f} mg vs N_FRAC x (biomass net gain + harvested + lysed) "
          f"{NF * (B_end - L['B0'] + L['harvested'] + L['lysed']):.0f} mg")
    print(f"  agent-steps at the 5e8 per-agent mass cap: {L['cap_agent_steps']}")


def determinism():
    outs = []
    for _ in range(2):
        r = run(expert(), init_cells=300, difficulty=2, seed=42, steps=1500, record_every=100)
        e = r["env"]
        outs.append((e.cumulative_harvested_mg, e.num_active, float(np.sum(e.cells_mass)), e.ph, e.temp,
                     [row["od"] for row in r["trace"]]))
    same = outs[0] == outs[1]
    print(f"\n=== determinism (seed 42, 1500 steps, D2): identical = {same}")
    if not same:
        print("   ", outs[0][:5], "\n   ", outs[1][:5])
    # reset(seed) alone, with the global RNG left in a different state beforehand
    res = []
    for pre in (0, 999):
        np.random.seed(pre)
        env = LedgerEnv(max_cells=7500, initial_cells=300, difficulty=2)
        env.reset(seed=42)
        res.append(dict(env.strain_params))
    print(f"    reset(seed) alone reproduces the strain regardless of prior RNG state: {res[0] == res[1]}")


if __name__ == "__main__":
    n_report("expert harvest, 700 cells, D2", run(expert(), init_cells=700, difficulty=2, seed=3))
    n_report("no harvest, 1400 umol, 700 cells, D2", run(const(65, 1400), init_cells=700, difficulty=2, seed=3))
    n_report("dark, 700 cells, D2", run(const(65, 0), init_cells=700, difficulty=2, seed=3))
    n_report("no harvest, 7500 cells, D2", run(const(65, 1400), init_cells=7500, difficulty=2, seed=3))
    determinism()
