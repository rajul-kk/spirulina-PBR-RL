"""
growth_dynamics_check.py — independent sanity check of GeneticPhotobioreactorEnv's
core biology: does population/OD grow the way a Spirulina culture actually should?

Three checks:
1. NO-HARVEST GROWTH CURVE (D0, favorable stir/light, no harvest): full 144h episode.
   Expect near-exponential growth while nutrients/light are non-limiting, then a
   slowdown as N/P/light self-shading kick in. Fits an exponential to the early phase
   and compares the fitted rate to the strain's own mu_max.
2. SEMI-CONTINUOUS HARVEST CYCLE (fixed sustainable harvest fraction every 12h):
   expect a repeating sawtooth in OD/biomass, not a monotonic trend.
3. LIGHT-RESPONSE SWEEP (short runs at fixed light levels, harvest=0): expect a
   unimodal (Haldane) response -- growth rate rising then falling as light increases,
   not monotonic.

Usage (from repo root, PPO_IBM/):
    python experiments/env_diagnosis/growth_dynamics_check.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_env import GeneticPhotobioreactorEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def make_action(stir, light, frac, f_max=0.5):
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run_fixed_policy(init_cells, difficulty, stir, light, frac, seed, max_steps=None, harvest_every=None):
    # env.reset(seed=...) alone does NOT reproducibly seed this env -- strain randomization
    # (_randomize_strain) and initial cell state use the legacy global np.random module, not
    # gym's self.np_random. Must seed the global RNG explicitly first (same fix applied to
    # legacy/TD3.py's run_td3_eval_episode, see docs/decision_history.md).
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=50_000, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    steps = max_steps or env.max_steps
    f_max = float(getattr(env, "F_MAX", 0.5))

    t_h, od, active, mass_mg, n_pool, p_pool, ph = [], [], [], [], [], [], []
    # Harvest action is interval-averaged by the env (_harvest_action_sum /
    # _harvest_action_count), applied as a pulse when HARVEST_INTERVAL_STEPS fires --
    # holding frac constant every step makes the interval average equal frac, matching
    # how a real policy would need to sustain it across the interval.
    action_run = make_action(stir, light, 0.0, f_max)
    action_harvest = make_action(stir, light, frac, f_max)

    for step in range(steps):
        action = action_harvest if harvest_every else action_run
        obs, reward, terminated, truncated, info = env.step(action)
        t_h.append(step * env.dt)
        od.append(env.od)
        active.append(env.num_active)
        mass_mg.append(np.sum(env.cells_mass[env.active_mask]) * 1e-9)
        n_pool.append(env.n_pool)
        p_pool.append(env.p_pool)
        ph.append(env.ph)
        if terminated or truncated:
            break

    return {
        "t_h": np.array(t_h), "od": np.array(od), "active": np.array(active),
        "mass_mg": np.array(mass_mg), "n_pool": np.array(n_pool), "p_pool": np.array(p_pool),
        "ph": np.array(ph), "mu_max": env.strain_params["mu_max"],
    }


def sweep_light_response(n_light=13, steps=300, seed=42):
    lights = np.linspace(50, 2000, n_light)
    mean_mu = []
    for light in lights:
        # Same strain (same seed) at every light level, so light is the only variable.
        np.random.seed(seed)
        env = GeneticPhotobioreactorEnv(max_cells=50_000, initial_cells=2000, difficulty=0)
        env.reset(seed=seed)
        action = make_action(70.0, light, 0.0)
        mass_before = np.sum(env.cells_mass[env.active_mask])
        for _ in range(steps):
            env.step(action)
        mass_after = np.sum(env.cells_mass[env.active_mask])
        # effective net growth rate over the run (per hour), from mass ratio
        dt_h = steps * env.dt
        mu_eff = np.log(max(mass_after, 1e-9) / max(mass_before, 1e-9)) / dt_h
        mean_mu.append(mu_eff)
    return lights, np.array(mean_mu)


def main():
    print("=" * 78)
    print("  CHECK 1: NO-HARVEST GROWTH CURVE (D0, stir=70, light=950, frac=0.0)")
    print("=" * 78)
    r1 = run_fixed_policy(init_cells=300, difficulty=0, stir=70, light=950, frac=0.0, seed=1)
    print(f"  strain mu_max = {r1['mu_max']:.4f} /h  (expected ~0.055, Spirulina ~12-13h doubling)")

    # Fit exponential to the early phase (before growth visibly slows).
    # Use the first 30h (1500 steps) where nutrients/light are least limiting.
    mask = r1["t_h"] <= 30.0
    t_fit, mass_fit = r1["t_h"][mask], r1["mass_mg"][mask]
    valid = mass_fit > 0
    log_mass = np.log(mass_fit[valid])
    A = np.vstack([t_fit[valid], np.ones_like(t_fit[valid])]).T
    slope, intercept = np.linalg.lstsq(A, log_mass, rcond=None)[0]
    pred = A @ [slope, intercept]
    ss_res = np.sum((log_mass - pred) ** 2)
    ss_tot = np.sum((log_mass - log_mass.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"  fitted early-phase growth rate = {slope:.4f} /h  (R^2 = {r2:.4f} on log-mass vs time)")
    print(f"  ratio fitted/mu_max = {slope / r1['mu_max']:.2f}  (expect <=1.0, close to 1.0 if non-limiting)")
    print(f"  final: active={r1['active'][-1]}, OD={r1['od'][-1]:.4f}, mass={r1['mass_mg'][-1]:.1f}mg,"
          f" N={r1['n_pool'][-1]:.1f}, P={r1['p_pool'][-1]:.1f}, pH={r1['ph'][-1]:.2f}")

    print("\n" + "=" * 78)
    print("  CHECK 2: SEMI-CONTINUOUS HARVEST CYCLE (harvest frac=0.15 every 600 steps)")
    print("=" * 78)
    r2run = run_fixed_policy(init_cells=300, difficulty=0, stir=70, light=950, frac=0.15,
                              seed=2, harvest_every=600)
    # Detect sawtooth: OD should drop at each harvest_every boundary, then climb back.
    drops = 0
    for i in range(600, len(r2run["od"]), 600):
        if i < len(r2run["od"]) and r2run["od"][i] < r2run["od"][i - 1] - 1e-4:
            drops += 1
    n_boundaries = len(r2run["od"]) // 600
    print(f"  harvest boundaries crossed: {n_boundaries}, OD-drop detected at: {drops}/{n_boundaries}")
    print(f"  final: active={r2run['active'][-1]}, OD={r2run['od'][-1]:.4f}")

    print("\n" + "=" * 78)
    print("  CHECK 3: LIGHT-RESPONSE SWEEP (Haldane -- expect unimodal, not monotonic)")
    print("=" * 78)
    lights, mu_eff = sweep_light_response()
    peak_idx = int(np.argmax(mu_eff))
    is_unimodal = (
        np.all(np.diff(mu_eff[:peak_idx + 1]) >= -1e-6) and
        np.all(np.diff(mu_eff[peak_idx:]) <= 1e-6)
    ) if 0 < peak_idx < len(mu_eff) - 1 else False
    for l, m in zip(lights, mu_eff):
        print(f"  light={l:7.1f}  mu_eff={m:+.4f}/h")
    print(f"  peak at light={lights[peak_idx]:.0f} -- unimodal shape: {'YES' if is_unimodal else 'NO (check for noise/bugs)'}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(r1["t_h"], r1["mass_mg"], label="biomass (mg)")
    ax.set_yscale("log")
    ax.set_xlabel("time (h)"); ax.set_ylabel("biomass (mg, log scale)")
    ax.set_title(f"Check 1: no-harvest growth (fit rate={slope:.4f}/h, mu_max={r1['mu_max']:.4f}/h, R2={r2:.3f})")
    ax.axvline(30.0, color="gray", linestyle="--", linewidth=1, label="fit window end")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(r1["t_h"], r1["od"], label="OD")
    ax2 = ax.twinx()
    ax2.plot(r1["t_h"], r1["n_pool"], color="green", alpha=0.6, label="N pool (mg/L)")
    ax2.plot(r1["t_h"], r1["p_pool"], color="orange", alpha=0.6, label="P pool (mg/L)")
    ax.set_xlabel("time (h)"); ax.set_ylabel("OD")
    ax2.set_ylabel("nutrient pool (mg/L)")
    ax.set_title("Check 1: OD and nutrient depletion over the episode")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    ax = axes[1, 0]
    ax.plot(r2run["t_h"], r2run["od"])
    for i in range(600, len(r2run["od"]), 600):
        ax.axvline(i * r2run["t_h"][1], color="red", linestyle=":", linewidth=0.8)
    ax.set_xlabel("time (h)"); ax.set_ylabel("OD")
    ax.set_title(f"Check 2: semi-continuous harvest cycle (drops at {drops}/{n_boundaries} boundaries)")

    ax = axes[1, 1]
    ax.plot(lights, mu_eff, marker="o")
    ax.axvline(lights[peak_idx], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("light (µmol/m²/s equiv, arbitrary units)"); ax.set_ylabel("effective net growth rate (/h)")
    ax.set_title(f"Check 3: light response (peak at {lights[peak_idx]:.0f}, unimodal={is_unimodal})")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "growth_dynamics_check.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()
