"""
test_actions.py — inspect or compare trained RecurrentPPO action outputs.

Single model:
    python test_actions.py --model model_data/recurrent_ppo_ibm_8.5_env
    python test_actions.py --interval 200 --difficulty 0 --plot

Compare two models (same env seed):
    python test_actions.py --model model_data/recurrent_ppo_ibm_8.5_env \\
                           --compare model_data/recurrent_ppo_ibm_6.5_env
    python test_actions.py --model A --norm model_data/norm_A.pkl \\
                           --compare B --norm-b model_data/norm_B.pkl --seed 42
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-test_actions-py-15)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments"))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from genetic_env import GeneticPhotobioreactorEnv

# ── Action decoding constants ─────────────────────────────────────────────────
# (full rationale: docs/decision_history.md#--diagnostics-test_actions-py-42)
ACT_RANGES = {
    "Stir    ": (50.0,  200.0,  "RPM"),
    "Light   ": (0.0,   2000.0, "umol/m2/s"),
    "Harvest ": (0.0,   0.5,    "frac"),
}
ACT_KEYS  = list(ACT_RANGES.keys())
ACT_LOW   = np.array([v[0] for v in ACT_RANGES.values()], dtype=np.float32)
ACT_HIGH  = np.array([v[1] for v in ACT_RANGES.values()], dtype=np.float32)
ACT_UNITS = [v[2] for v in ACT_RANGES.values()]


def decode_action(raw: np.ndarray) -> np.ndarray:
    return np.interp(raw, [-1.0, 1.0], [0.0, 1.0]) * (ACT_HIGH - ACT_LOW) + ACT_LOW


def find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    zips = [f for f in os.listdir(checkpoint_dir) if f.endswith("_steps.zip")]
    if not zips:
        return None
    def _step(name):
        try:
            return int(name.split("_")[-2])
        except (IndexError, ValueError):
            return 0
    latest = sorted(zips, key=_step)[-1]
    return os.path.join(checkpoint_dir, latest[:-4])


def strip_zip(path: str) -> str:
    return path[:-4] if path.endswith(".zip") else path


def load_env(difficulty: int, initial_cells: int, norm_path: str, seed: int):
    np.random.seed(seed)
    def _make():
        env = GeneticPhotobioreactorEnv(
            max_cells=7_500,
            initial_cells=initial_cells,
            difficulty=difficulty,
        )
        return Monitor(env)
    base = DummyVecEnv([_make])
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, venv=base)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"  [WARN] Norm not found at '{norm_path}' — unnormalised env.")
        vec_env = VecNormalize(base, norm_obs=False, norm_reward=False)
    return vec_env


def get_raw_env(vec_env):
    env = vec_env
    if hasattr(env, "venv"):
        env = env.venv
    env = env.envs[0]
    if hasattr(env, "env"):
        env = env.env
    return env


def run_single_episode(model, vec_env, seed: int):
    """Run one full episode. Returns (raw_actions, decoded_actions, rewards, states)."""
    np.random.seed(seed)
    obs = vec_env.reset()
    raw_env = get_raw_env(vec_env)
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)

    raw_acts, decoded_acts, cumrew = [], [], 0.0
    states = []   # snapshot every step: (od, ph, n_pool, p_pool, temp, cond, do2, num_active, cumrew)
    done = False

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_starts, deterministic=True
        )
        ep_starts = np.zeros((1,), dtype=bool)
        obs, reward, done_vec, _ = vec_env.step(action)
        done = bool(done_vec[0])
        raw = np.array(action[0], dtype=np.float32)
        raw_acts.append(raw.copy())
        decoded_acts.append(decode_action(raw))
        cumrew += float(reward[0])
        states.append((
            float(raw_env.od),
            float(raw_env.ph),
            float(raw_env.n_pool),
            float(raw_env.p_pool),
            float(raw_env.temp),
            float(raw_env.conductivity),
            float(raw_env.do2),
            int(raw_env.num_active),
            cumrew,
        ))

    return (
        np.array(raw_acts, dtype=np.float32),
        np.array(decoded_acts, dtype=np.float32),
        states,
    )


# ── Single-model printing ─────────────────────────────────────────────────────

def print_banner(step_lo, step_hi, dt):
    print(f"\n{'-'*62}")
    print(f"  STEPS {step_lo:>5} – {step_hi:<5}   (t = {step_lo*dt:.1f} – {step_hi*dt:.1f} h)")
    print(f"{'-'*62}")


def print_action_block(raw_buf: np.ndarray):
    dec = np.apply_along_axis(decode_action, 1, raw_buf)
    print(f"  {'Action':<10}  {'Raw mean':>9}  {'Raw std':>8}  |  {'Dec mean':>10}  {'Dec std':>9}  Unit")
    print(f"  {'-'*10}  {'-'*9}  {'-'*8}  |  {'-'*10}  {'-'*9}  {'-'*12}")
    for i, key in enumerate(ACT_KEYS):
        print(f"  {key:<10}  {np.mean(raw_buf[:,i]):>+9.3f}  {np.std(raw_buf[:,i]):>8.3f}  |"
              f"  {np.mean(dec[:,i]):>10.2f}  {np.std(dec[:,i]):>9.2f}  {ACT_UNITS[i]}")


def print_env_state(raw_env, cumrew):
    print(f"\n  Env state:")
    print(f"    OD={raw_env.od:.4f}  pH={raw_env.ph:.2f}  "
          f"N={raw_env.n_pool:.1f} mg/L  P={raw_env.p_pool:.1f} mg/L  T={raw_env.temp:.1f}°C")
    print(f"    Cond={raw_env.conductivity:.0f} µS/cm  "
          f"DO2_s={raw_env.do2_s:.2f}  DO2_b={raw_env.do2_b:.2f} mg/L")
    print(f"    Pop={raw_env.num_active}  Cumulative reward: {cumrew:.3f}")


def run_single(model, vec_env, interval, plot, dt, label=""):
    raw_env = get_raw_env(vec_env)
    obs = vec_env.reset()
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    buf, cumrew, step, done = [], 0.0, 0, False
    all_raw, all_dec = [], []

    tag = f" [{label}]" if label else ""
    print(f"\n  Running episode{tag} (max {raw_env.max_steps} steps, dt={dt}h) ...")

    # Snapshot of terminal state (DummyVecEnv auto-resets on done, so capture before step)
    terminal_snap = {}

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_starts, deterministic=True
        )
        ep_starts = np.zeros((1,), dtype=bool)
        # Capture state snapshot before step (becomes terminal state when done fires)
        snap = {k: getattr(raw_env, k) for k in
                ("od","ph","n_pool","p_pool","temp","conductivity","do2_s","do2_b","num_active")}
        obs, reward, done_vec, _ = vec_env.step(action)
        done = bool(done_vec[0])
        raw = np.array(action[0], dtype=np.float32)
        buf.append(raw.copy())
        cumrew += float(reward[0])
        step += 1
        if done:
            terminal_snap = snap
        if plot:
            all_raw.append(raw.copy())
            all_dec.append(decode_action(raw))
        if step % interval == 0 or done:
            print_banner(step - len(buf), step, dt)
            print_action_block(np.array(buf))
            if done and terminal_snap:
                # Print pre-reset terminal state, not the auto-reset initial state
                print(f"\n  Env state (terminal):")
                print(f"    OD={terminal_snap['od']:.4f}  pH={terminal_snap['ph']:.2f}  "
                      f"N={terminal_snap['n_pool']:.1f} mg/L  P={terminal_snap['p_pool']:.1f} mg/L  "
                      f"T={terminal_snap['temp']:.1f}C")
                print(f"    Cond={terminal_snap['conductivity']:.0f} uS/cm  "
                      f"DO2_s={terminal_snap['do2_s']:.2f}  DO2_b={terminal_snap['do2_b']:.2f} mg/L")
                print(f"    Pop={terminal_snap['num_active']}  Cumulative reward: {cumrew:.3f}")
            else:
                print_env_state(raw_env, cumrew)
            buf = []

    fin_od = terminal_snap.get("od", raw_env.od)
    print(f"\n{'='*62}")
    print(f"  Episode done  step={step}  OD={fin_od:.4f}  reward={cumrew:.3f}")
    print(f"{'='*62}\n")
    if plot and all_raw:
        _save_plot(all_raw, all_dec, interval, dt, suffix=label.replace(" ", "_") if label else "")


# ── Comparison mode ───────────────────────────────────────────────────────────

def run_comparison(model_a, vec_a, model_b, vec_b, seed, interval, dt, plot, label_a, label_b):
    print(f"\n  Running {label_a} (seed={seed}) ...")
    raw_a, dec_a, states_a = run_single_episode(model_a, vec_a, seed)
    print(f"  Running {label_b} (seed={seed}) ...")
    raw_b, dec_b, states_b = run_single_episode(model_b, vec_b, seed)

    n = min(len(raw_a), len(raw_b))
    W = 30  # column width

    def hdr(title):
        print(f"\n{'-'*80}")
        print(f"  {title}")
        print(f"{'-'*80}")

    # Per-interval action comparison
    for i_start in range(0, n, interval):
        i_end = min(i_start + interval, n)
        sl_a = raw_a[i_start:i_end]
        sl_b = raw_b[i_start:i_end]
        dec_sl_a = dec_a[i_start:i_end]
        dec_sl_b = dec_b[i_start:i_end]
        t_lo, t_hi = i_start * dt, i_end * dt
        print(f"\n{'-'*80}")
        print(f"  STEPS {i_start:>5}–{i_end:<5}   (t = {t_lo:.1f}–{t_hi:.1f} h)")
        print(f"{'-'*80}")
        la = label_a[:W]
        lb = label_b[:W]
        print(f"  {'Action':<10}  {la:>{W}}  {lb:>{W}}")
        print(f"  {'-'*10}  {'-'*W}  {'-'*W}")
        for i, key in enumerate(ACT_KEYS):
            ma = np.mean(dec_sl_a[:, i]);  sa = np.std(dec_sl_a[:, i])
            mb = np.mean(dec_sl_b[:, i]);  sb = np.std(dec_sl_b[:, i])
            ca = f"{ma:.1f}±{sa:.1f} {ACT_UNITS[i]}"
            cb = f"{mb:.1f}±{sb:.1f} {ACT_UNITS[i]}"
            better = "◀" if ma > mb else ("▶" if mb > ma else " ")
            print(f"  {key:<10}  {ca:>{W}}  {cb:>{W}}  {better}")

        # Env state at end of interval
        idx = i_end - 1
        od_a, ph_a, n_a, p_a, _, _, do2_a, pop_a, cr_a = states_a[idx]
        od_b, ph_b, n_b, p_b, _, _, do2_b, pop_b, cr_b = states_b[idx]
        print(f"\n  {'Metric':<16}  {label_a[:20]:>20}  {label_b[:20]:>20}")
        print(f"  {'-'*16}  {'-'*20}  {'-'*20}")
        rows = [
            ("OD",       f"{od_a:.4f}",  f"{od_b:.4f}"),
            ("pH",       f"{ph_a:.2f}",  f"{ph_b:.2f}"),
            ("N (mg/L)", f"{n_a:.1f}",   f"{n_b:.1f}"),
            ("P (mg/L)", f"{p_a:.1f}",   f"{p_b:.1f}"),
            ("DO2",      f"{do2_a:.2f}", f"{do2_b:.2f}"),
            ("Pop",      f"{pop_a}",     f"{pop_b}"),
            ("Cum Rew",  f"{cr_a:.2f}",  f"{cr_b:.2f}"),
        ]
        for name, va, vb in rows:
            print(f"  {name:<16}  {va:>20}  {vb:>20}")

    # Episode summary
    fin_a = states_a[-1]
    fin_b = states_b[-1]
    peak_od_a = max(s[0] for s in states_a)
    peak_od_b = max(s[0] for s in states_b)

    print(f"\n{'='*80}")
    print(f"  EPISODE SUMMARY  (seed={seed})")
    print(f"{'='*80}")
    print(f"  {'Metric':<20}  {label_a[:22]:>22}  {label_b[:22]:>22}  Winner")
    print(f"  {'-'*20}  {'-'*22}  {'-'*22}  {'-'*8}")
    summary = [
        ("Total reward",  f"{fin_a[8]:.2f}",    f"{fin_b[8]:.2f}",    fin_a[8] > fin_b[8]),
        ("Final OD",      f"{fin_a[0]:.4f}",    f"{fin_b[0]:.4f}",    fin_a[0] > fin_b[0]),
        ("Peak OD",       f"{peak_od_a:.4f}",   f"{peak_od_b:.4f}",   peak_od_a > peak_od_b),
        ("Final N mg/L",  f"{fin_a[2]:.1f}",    f"{fin_b[2]:.1f}",    None),
        ("Final P mg/L",  f"{fin_a[3]:.1f}",    f"{fin_b[3]:.1f}",    None),
        ("Final pop",     f"{fin_a[7]}",         f"{fin_b[7]}",         fin_a[7] > fin_b[7]),
        ("Steps",         f"{len(states_a)}",    f"{len(states_b)}",    None),
    ]
    for name, va, vb, a_wins in summary:
        if a_wins is None:
            winner = "—"
        elif a_wins:
            winner = label_a[:8]
        else:
            winner = label_b[:8]
        print(f"  {name:<20}  {va:>22}  {vb:>22}  {winner}")
    print(f"{'='*80}\n")

    if plot:
        _save_comparison_plot(raw_a, dec_a, raw_b, dec_b, dt, label_a, label_b)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_plot(raw_actions, decoded_actions, interval, dt, suffix=""):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not found — skipping.")
        return
    raw = np.array(raw_actions)
    dec = np.array(decoded_actions)
    T   = np.arange(len(raw)) * dt
    fig, axes = plt.subplots(len(ACT_KEYS), 2, figsize=(14, 10), sharex=True)
    fig.suptitle("RecurrentPPO Action Trajectory", fontsize=13)
    labels = [k.strip() for k in ACT_KEYS]
    for i in range(len(ACT_KEYS)):
        axes[i, 0].plot(T, raw[:, i], lw=0.8, color=f"C{i}")
        axes[i, 0].axhline(0, color="k", lw=0.4, ls="--")
        axes[i, 0].set_ylabel(f"{labels[i]}\n(raw)", fontsize=8)
        axes[i, 0].set_ylim(-1.1, 1.1)
        axes[i, 1].plot(T, dec[:, i], lw=0.8, color=f"C{i}")
        axes[i, 1].set_ylabel(f"{labels[i]}\n({ACT_UNITS[i]})", fontsize=8)
        for t_m in np.arange(0, T[-1], interval * dt):
            axes[i, 0].axvline(t_m, color="grey", lw=0.3, alpha=0.5)
            axes[i, 1].axvline(t_m, color="grey", lw=0.3, alpha=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Simulated time (h)")
    axes[0, 0].set_title("Raw [-1,1]")
    axes[0, 1].set_title("Decoded (physical units)")
    plt.tight_layout()
    tag = f"_{suffix}" if suffix else ""
    out = f"action_trajectory{tag}.png"
    plt.savefig(out, dpi=150)
    print(f"  [plot] Saved → {out}")
    plt.close()


def _save_comparison_plot(raw_a, dec_a, raw_b, dec_b, dt, label_a, label_b):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not found — skipping.")
        return
    T_a = np.arange(len(raw_a)) * dt
    T_b = np.arange(len(raw_b)) * dt
    labels = [k.strip() for k in ACT_KEYS]
    fig, axes = plt.subplots(len(ACT_KEYS), 2, figsize=(15, 11), sharex=False)
    fig.suptitle(f"Action Comparison: {label_a} vs {label_b}", fontsize=12)
    for i in range(len(ACT_KEYS)):
        # Raw
        axes[i, 0].plot(T_a, raw_a[:, i], lw=0.9, label=label_a, color="C0")
        axes[i, 0].plot(T_b, raw_b[:, i], lw=0.9, label=label_b, color="C1", alpha=0.8)
        axes[i, 0].axhline(0, color="k", lw=0.3, ls="--")
        axes[i, 0].set_ylabel(f"{labels[i]}\n(raw)", fontsize=8)
        axes[i, 0].set_ylim(-1.1, 1.1)
        if i == 0:
            axes[i, 0].legend(fontsize=7, loc="upper right")
        # Decoded
        axes[i, 1].plot(T_a, dec_a[:, i], lw=0.9, label=label_a, color="C0")
        axes[i, 1].plot(T_b, dec_b[:, i], lw=0.9, label=label_b, color="C1", alpha=0.8)
        axes[i, 1].set_ylabel(f"{labels[i]}\n({ACT_UNITS[i]})", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Simulated time (h)")
    axes[0, 0].set_title("Raw output [-1,1]")
    axes[0, 1].set_title("Decoded (physical units)")
    plt.tight_layout()
    out = "action_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"  [plot] Saved → {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect or compare RecurrentPPO actions.")
    parser.add_argument("--model",         type=str, default=None,
                        help="Model A path (no .zip). Default: latest checkpoint.")
    parser.add_argument("--norm",          type=str, default="model_data/recurrent_vec_normalize.pkl",
                        help="VecNormalize pkl for model A.")
    parser.add_argument("--compare",       type=str, default=None,
                        help="Model B path for comparison mode.")
    parser.add_argument("--norm-b",        type=str, default=None,
                        help="VecNormalize pkl for model B (defaults to --norm).")
    parser.add_argument("--seed",          type=int, default=0,
                        help="Random seed for reproducible comparison (default: 0).")
    parser.add_argument("--interval",      type=int, default=500,
                        help="Print summary every N steps (default: 500).")
    parser.add_argument("--difficulty",    type=int, default=2)
    parser.add_argument("--initial-cells", type=int, default=300)
    parser.add_argument("--plot",          action="store_true",
                        help="Save action trajectory / comparison plot.")
    args = parser.parse_args()

    checkpoint_dir = "model_data/recurrent_checkpoints"
    final_model    = "model_data/recurrent_ppo_genetic_ibm"

    # Resolve model A
    model_a_path = args.model
    if model_a_path is None:
        model_a_path = find_latest_checkpoint(checkpoint_dir)
        if model_a_path is None and os.path.exists(final_model + ".zip"):
            model_a_path = final_model
        if model_a_path is None:
            print("[ERROR] No model found. Run training first or pass --model.")
            sys.exit(1)
    model_a_path = strip_zip(model_a_path)
    norm_b_path  = args.norm_b if args.norm_b else args.norm

    label_a = os.path.basename(model_a_path)
    label_b = os.path.basename(strip_zip(args.compare)) if args.compare else ""

    print(f"  Model A : {model_a_path}")
    print(f"  Norm  A : {args.norm}")
    if args.compare:
        print(f"  Model B : {args.compare}")
        print(f"  Norm  B : {norm_b_path}")
    print(f"  D{args.difficulty}  |  initial_cells={args.initial_cells}  |  seed={args.seed}  |  interval={args.interval}")

    vec_a = load_env(args.difficulty, args.initial_cells, args.norm, args.seed)
    _obs_override = {"observation_space": vec_a.observation_space}
    model_a = RecurrentPPO.load(model_a_path, env=vec_a, device="cpu", custom_objects=_obs_override)
    print(f"  Model A loaded.")

    dt = float(getattr(get_raw_env(vec_a), "dt", 0.02))

    if args.compare:
        model_b_path = strip_zip(args.compare)
        vec_b = load_env(args.difficulty, args.initial_cells, norm_b_path, args.seed)
        model_b = RecurrentPPO.load(model_b_path, env=vec_b, device="cpu", custom_objects={"observation_space": vec_b.observation_space})
        print(f"  Model B loaded.\n")
        run_comparison(
            model_a, vec_a, model_b, vec_b,
            seed=args.seed, interval=args.interval, dt=dt,
            plot=args.plot, label_a=label_a, label_b=label_b,
        )
    else:
        print(f"  Model loaded.\n")
        run_single(model_a, vec_a, interval=args.interval, plot=args.plot, dt=dt)


if __name__ == "__main__":
    main()
