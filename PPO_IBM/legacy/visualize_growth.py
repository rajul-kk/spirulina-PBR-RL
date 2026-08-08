
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from environments.genetic_env import GeneticPhotobioreactorEnv

DT_H = 0.02  # hours per step

# ── Growth models ─────────────────────────────────────────────────────────────

def gompertz(t, K, mu_max, lam):
    """Gompertz OD(t): asymmetric sigmoid, standard for batch culture."""
    inner = (mu_max * np.e / K) * (lam - t) + 1.0
    return K * np.exp(-np.exp(inner))

def logistic(t, K, r, t_mid):
    """Standard logistic OD(t)."""
    return K / (1.0 + np.exp(-r * (t - t_mid)))

def fit_growth_curve(time_h, od, model="gompertz"):
    """Fit a sigmoid growth model. Returns (params, pcov, fitted_od)."""
    od = np.asarray(od, dtype=float)
    K0 = float(od.max())
    if model == "gompertz":
        # p0: K, mu_max, lambda
        p0 = [K0, 0.05, time_h[np.argmax(np.gradient(od))] * 0.5]
        bounds = ([0.0, 0.0, 0.0], [K0 * 3, 1.0, float(time_h[-1])])
        try:
            popt, pcov = curve_fit(gompertz, time_h, od, p0=p0,
                                   bounds=bounds, maxfev=8000)
            return popt, pcov, gompertz(time_h, *popt), "Gompertz"
        except Exception:
            pass
    # Fallback: logistic
    t_mid0 = time_h[np.argmin(np.abs(od - K0 / 2))]
    try:
        popt, pcov = curve_fit(logistic, time_h, od,
                               p0=[K0, 0.1, t_mid0], maxfev=8000)
        return popt, pcov, logistic(time_h, *popt), "Logistic"
    except Exception:
        return None, None, od * np.nan, "Fit failed"

def specific_growth_rate(od, time_h, window=51):
    """μ(t) = d(ln OD)/dt, smoothed with Savitzky-Golay."""
    ln_od = np.log(np.maximum(od, 1e-9))
    raw   = np.gradient(ln_od, time_h)
    if len(raw) >= window:
        return savgol_filter(raw, window_length=window, polyorder=3)
    return raw

def identify_phases(mu, time_h, lag_mu_thresh=0.005, stat_mu_thresh=0.002):
    """Return (lag_end_h, exp_end_h, stat_start_h) based on μ thresholds."""
    lag_end  = time_h[np.argmax(mu > lag_mu_thresh)] if np.any(mu > lag_mu_thresh) else None
    peak_idx = int(np.argmax(mu))
    # stationary: after peak, μ drops below threshold
    post     = mu[peak_idx:]
    t_post   = time_h[peak_idx:]
    stat_idx = np.argmax(post < stat_mu_thresh) if np.any(post < stat_mu_thresh) else len(post) - 1
    stat_start = float(t_post[stat_idx])
    return lag_end, float(time_h[peak_idx]), stat_start

# ── Policy definitions ────────────────────────────────────────────────────────

def run_policy(policy_type, max_steps=40000):
    """Run one episode and return data arrays."""
    env = GeneticPhotobioreactorEnv(max_cells=300_000, initial_cells=500)
    env.max_steps = max_steps
    obs, _ = env.reset()

    strain = getattr(env, "strain_params", {})

    steps, od, n_pool, ph, temp, do2, pop, rew = [], [], [], [], [], [], [], []

    try:
        for t in range(max_steps):
            if policy_type == "fixed":
                # Constant mid-range control — baseline
                action = np.array([0.1, -0.5, 0.8, -1.0], dtype=np.float32)

            elif policy_type == "ideal":
                cur_od  = float(obs[0])
                cur_ph  = float(obs[1])
                cur_n   = float(obs[2])

                # Light: compensate for self-shading
                k_att   = 0.5 + 10.0 * cur_od
                target  = min(2000.0, 500.0 / np.exp(-k_att * 0.05))
                light_a = float(np.interp(target, [0, 2000], [-1.0, 1.0]))

                # Stir: scale with density, cap below 300 RPM
                stir_a  = float(np.clip(0.1 + 0.01 * cur_od, 0.1, 0.19))

                # Nutrients: feed to keep n_pool at 250–350 mg/L
                if cur_n < 200:
                    nut_a = 1.0
                elif cur_n < 300:
                    nut_a = 0.2
                else:
                    nut_a = -1.0

                # CO2: more injection when pH is too high (CO2 is acid in alkaline medium)
                if cur_ph > 10.5:
                    co2_a = 1.0
                elif cur_ph > 9.5:
                    co2_a = 0.0
                else:
                    co2_a = -1.0

                action = np.array([stir_a, light_a, nut_a, co2_a], dtype=np.float32)

            elif policy_type == "aggressive":
                # High stir, max light, heavy feeding — tests culture limits
                action = np.array([0.5, 1.0, 1.0, 0.5], dtype=np.float32)

            else:  # random
                action = env.action_space.sample()

            o, r, done, _, info = env.step(action)
            steps.append(t)
            od.append(float(o[0]))
            n_pool.append(float(o[2]))
            ph.append(float(o[1]))
            temp.append(float(o[3]))
            do2.append(float(getattr(env, "dissolved_o2", 8.0)))
            pop.append(int(info.get("pop", env.num_active)))
            rew.append(float(r))
            obs = o
            if done:
                break
    except KeyboardInterrupt:
        print(f"\n  [{policy_type}] interrupted at step {t}")

    t_h = np.array(steps) * DT_H
    return {
        "label":    policy_type,
        "time_h":   t_h,
        "od":       np.array(od),
        "n_pool":   np.array(n_pool),
        "ph":       np.array(ph),
        "temp":     np.array(temp),
        "do2":      np.array(do2),
        "pop":      np.array(pop),
        "reward":   np.array(rew),
        "strain":   strain,
    }

# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "fixed":      "#5080ff",
    "ideal":      "#30d878",
    "aggressive": "#ff9020",
    "random":     "#bb60ff",
}

def plot_results(runs):
    fig = plt.figure(figsize=(16, 14), facecolor="#0c1520")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    ax_od   = fig.add_subplot(gs[0, :])   # OD growth curves (full width)
    ax_mu   = fig.add_subplot(gs[1, 0])   # specific growth rate
    ax_n    = fig.add_subplot(gs[1, 1])   # N-pool
    ax_ph   = fig.add_subplot(gs[2, 0])   # pH
    ax_rew  = fig.add_subplot(gs[2, 1])   # cumulative reward

    style = dict(facecolor="#0d1a2e", edgecolor="#1e3050")
    for ax in (ax_od, ax_mu, ax_n, ax_ph, ax_rew):
        ax.set_facecolor("#0d1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3050")
        ax.tick_params(colors="#a0b8d0", labelsize=10)
        ax.xaxis.label.set_color("#a0b8d0")
        ax.yaxis.label.set_color("#a0b8d0")
        ax.title.set_color("#c8d8f0")
        ax.grid(True, alpha=0.12, color="white")

    ax_od.set_title("Biomass Growth Curves (OD)  +  Gompertz/Logistic Fits",
                    fontsize=13, fontweight="bold")
    ax_od.set_xlabel("Time (hours)")
    ax_od.set_ylabel("Optical Density")

    ax_mu.set_title("Specific Growth Rate  μ(t)")
    ax_mu.set_xlabel("Time (hours)")
    ax_mu.set_ylabel("μ  (h⁻¹)")
    ax_mu.axhline(0, color="#ff4444", linewidth=0.8, alpha=0.6)

    ax_n.set_title("N-Pool  (dissolved nitrogen)")
    ax_n.set_xlabel("Time (hours)")
    ax_n.set_ylabel("N-pool  (mg N/L)")
    ax_n.axhline(50, color="#ff8844", linewidth=1.0, linestyle="--",
                 alpha=0.7, label="N-starvation threshold (50 mg/L)")

    ax_ph.set_title("pH Trajectory")
    ax_ph.set_xlabel("Time (hours)")
    ax_ph.set_ylabel("pH")
    ax_ph.axhspan(9.0, 10.5, alpha=0.07, color="#30d878", label="Optimal pH zone")
    ax_ph.axhline(11.5, color="#ff4444", linewidth=0.8, linestyle=":", alpha=0.6)

    ax_rew.set_title("Cumulative Reward")
    ax_rew.set_xlabel("Time (hours)")
    ax_rew.set_ylabel("Cumulative reward")

    fit_table_lines = []

    for run in runs:
        t    = run["time_h"]
        od   = run["od"]
        col  = COLORS.get(run["label"], "#aaaaaa")
        lbl  = run["label"].capitalize()

        if len(t) < 10:
            continue

        # ── OD raw ──────────────────────────────────────────────────────
        ax_od.plot(t, od, color=col, linewidth=1.4, alpha=0.85, label=lbl)

        # ── Sigmoid fit ─────────────────────────────────────────────────
        if len(t) > 50 and od.max() > 1e-5:
            popt, pcov, od_fit, fname = fit_growth_curve(t, od)
            if popt is not None and not np.any(np.isnan(od_fit)):
                ax_od.plot(t, od_fit, color=col, linewidth=2.2,
                           linestyle="--", alpha=0.9,
                           label=f"{lbl} {fname} fit")
                if fname == "Gompertz":
                    K, mu_m, lam = popt
                    fit_table_lines.append(
                        f"{lbl:<12}  K={K:.4f}  μ_max={mu_m:.4f} h⁻¹  λ={lam:.1f} h")
                else:
                    K, r, _ = popt
                    fit_table_lines.append(f"{lbl:<12}  K={K:.4f}  r={r:.4f} h⁻¹  (logistic)")

        # ── Growth phases on OD axis ─────────────────────────────────────
        mu = specific_growth_rate(od, t)
        if np.any(mu > 0.002):
            lag_end, exp_peak, stat_start = identify_phases(mu, t)
            if lag_end is not None:
                ax_od.axvline(lag_end, color=col, linewidth=0.8,
                              linestyle=":", alpha=0.45)
            ax_od.axvline(stat_start, color=col, linewidth=0.8,
                          linestyle="-.", alpha=0.45)

        # ── μ(t) ────────────────────────────────────────────────────────
        ax_mu.plot(t, mu, color=col, linewidth=1.2, alpha=0.80, label=lbl)

        # ── N-pool ──────────────────────────────────────────────────────
        ax_n.plot(t, run["n_pool"], color=col, linewidth=1.3, alpha=0.80, label=lbl)

        # ── pH ──────────────────────────────────────────────────────────
        ax_ph.plot(t, run["ph"], color=col, linewidth=1.3, alpha=0.80, label=lbl)

        # ── Cumulative reward ───────────────────────────────────────────
        ax_rew.plot(t, np.cumsum(run["reward"]), color=col,
                    linewidth=1.3, alpha=0.80, label=lbl)

    # ── Legend & annotation ────────────────────────────────────────────
    for ax in (ax_od, ax_mu, ax_n, ax_ph, ax_rew):
        leg = ax.legend(fontsize=9, facecolor="#0d1a2e", edgecolor="#1e3050",
                        labelcolor="#c0d4f0")

    if fit_table_lines:
        fit_text = "Fit parameters:\n" + "\n".join(fit_table_lines)
        ax_od.text(0.01, 0.97, fit_text, transform=ax_od.transAxes,
                   fontsize=8.5, verticalalignment="top",
                   color="#9ab8d4", fontfamily="monospace",
                   bbox=dict(facecolor="#0c1520", edgecolor="#1e3050",
                             alpha=0.85, boxstyle="round,pad=0.4"))

    # ── Phase legend ────────────────────────────────────────────────────
    phase_note = "Phase markers:  ···  lag end     –·–  stationary start"
    fig.text(0.5, 0.965, phase_note, ha="center", fontsize=9,
             color="#7090a8", style="italic")

    # ── Strain params (first run) ────────────────────────────────────────
    if runs and runs[0]["strain"]:
        s = runs[0]["strain"]
        sp_text = (f"Strain (ep1):  "
                   f"μ_max={s.get('mu_max',0):.4f}  "
                   f"T_opt={s.get('T_opt',28):.1f}°C  "
                   f"Ks_I={s.get('Ks_light', s.get('Ks_I',80)):.0f}")
        fig.text(0.5, 0.01, sp_text, ha="center", fontsize=9,
                 color="#7090a8", fontfamily="monospace")

    fig.suptitle("Spirulina PBR  —  Growth Analysis", fontsize=15,
                 fontweight="bold", color="#d0e8ff", y=0.99)

    plt.savefig("growth_analysis.png", dpi=150, facecolor="#0c1520",
                bbox_inches="tight")
    print("Saved → growth_analysis.png")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+",
                        default=["fixed", "ideal"],
                        choices=["fixed", "ideal", "aggressive", "random"])
    parser.add_argument("--steps", type=int, default=40000,
                        help="Max steps per run (default 40000 = ~167 h)")
    args = parser.parse_args()

    print(f"Running {len(args.policies)} policies for up to {args.steps} steps each...")
    runs = []
    for p in args.policies:
        print(f"  → {p} ...", flush=True)
        run = run_policy(p, max_steps=args.steps)
        print(f"     steps={len(run['time_h'])}  "
              f"max_OD={run['od'].max():.5f}  "
              f"final_N={run['n_pool'][-1]:.1f}")
        runs.append(run)

    plot_results(runs)
