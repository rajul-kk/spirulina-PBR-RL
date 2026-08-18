"""
bc_pretrain.py — behaviour-cloning warm start for the RecurrentPPO curriculum trainer.

WHY THIS EXISTS
---------------
Across v4, v14, v15 and v16b, PPO consistently failed to find and HOLD the harvest
setpoint that the reward function itself ranks highest. Measured directly
(dynamic_profile_sweep_od.py, D1 physics, 4 seeds/point):

    constant-action controller, stir=60 light=900 frac=0.18
        -> reward 1116.8, harvest 139.9mg, time_avg_od 0.0131, 0% crash
           (clears not just the D1 gate but the D2 gate)

    v16b's learned policy (8M steps, full budget)
        -> reward 957-1028, harvest ~22mg median on a 40-seed held-out sweep

So the correct behaviour is not merely acceptable under the current reward, it is
reward-SUPERIOR by ~10%. The learned policy sits in a nearby, worse local optimum
("throttle light, coast at OD~0.012, never harvest"). This is an exploration/
convergence failure, not a reward-ranking failure.

This script therefore initialises the policy AT the known-good setpoint by supervised
learning on scripted expert rollouts, then hands off to the normal PPO curriculum. The
reward function and environment are deliberately left UNTOUCHED, so v17 remains directly
comparable to v15/v16b and so the experiment cleanly distinguishes two hypotheses:

    - BC holds the setpoint  -> the problem was exploration; nothing else needs changing.
    - BC drifts back to coasting -> the dense/sparse reward imbalance (reward_od's ~1080
      per-episode ceiling vs reward_harvest's 6) is the real cause, and a structural
      reward change is then justified by evidence rather than by inference.

Precedent: arXiv 2509.06853 bootstraps an industrial photobioreactor RL agent from PID
controller trajectories before online RL, for essentially these reasons.

NOTE ON THE LSTM: the expert is a CONSTANT action, so its target does not depend on
observation history at all. BC can therefore be done per-timestep with a zeroed LSTM
state rather than by unrolling whole 7200-step sequences — far faster and equivalent for
this target. The LSTM's recurrent capacity is left for PPO fine-tuning to develop.

Usage:
    python bc_pretrain.py                  # generate demos + train + save warm start
    python bc_pretrain.py --episodes 32 --epochs 12
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments"))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from curriculum_schedule import CurriculumStartController
from env_factory import make_env
from env_utils import unwrap_raw_env as _unwrap_raw_env
from entropy_schedule import ENTROPY_INIT
from training_state import save_state

OUT_DIR = "model_data/bc_warmstart"
MODEL_PATH = os.path.join(OUT_DIR, "recurrent_ppo_genetic_ibm")
NORM_PATH = os.path.join(OUT_DIR, "recurrent_vec_normalize.pkl")
STATE_PATH = os.path.join(OUT_DIR, "recurrent_training_state.pkl")

# Expert stir/light band. These two are genuinely fine as near-constants — the sweep found
# the outcome insensitive to them across 60-80rpm / 900-1000umol.
EXPERT_STIR_RANGE = (60.0, 80.0)
EXPERT_LIGHT_RANGE = (900.0, 1000.0)

# HARVEST IS A FEEDBACK LAW, NOT A CONSTANT.
#
# A constant harvest fraction was the original plan, based on a sweep that scored
# frac=0.18 at 139.9mg / time_avg_od 0.0131 / reward 1116.8. That sweep ran on the BARE
# env at a FIXED initial_cells=300. Measured through the real training stack — where
# CurriculumStartWrapper draws initial_cells log-uniformly (100-400 "low", 600-1500 "mid",
# 2000-5000 "high") — the identical constant action produced:
#
#     30.7, 41.3, 46.3, 57.3, 63.4, 93.3, 95.8, 186.0, 248.3, 347.8, 368.2  mg
#     (median ~93mg, a 12x spread, time_avg_od 0.0025-0.0326)
#
# i.e. the episode outcome is dominated by the cold-start draw, not by the action. A
# constant fraction strips a small starting culture before it can establish (46mg at
# od 0.0054) while under-harvesting a large one (368mg at od 0.0326). Cloning a
# state-independent constant would therefore teach the policy to IGNORE its observation
# on roughly half of all episodes — actively wrong, and a poor foundation for fine-tuning.
#
# Instead the expert harvests the SURPLUS above an OD setpoint, proportionally:
#     frac = clip(GAIN * (od / OD_SETPOINT - 1), 0, FRAC_CAP)
# Below setpoint it harvests nothing and lets the culture build; above it, it removes
# roughly the excess. This drives time_avg_od toward OD_SETPOINT regardless of where the
# episode started, which is exactly what the curriculum's time_avg_od criterion rewards.
#
# OD_SETPOINT sits above the D2 gate's time_avg_od>=0.011 with margin, and above
# genetic_env's OD_TARGET=0.012 (the peak of reward_od), so the controller holds the
# culture in the band the reward function itself pays most for.
EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.05
EXPERT_FRAC_CAP = 0.30

# Must match recurrent_ppo.py's gamma (Fix #13, v18) — returns computed at a different
# discount than the trainer uses would hand PPO a systematically miscalibrated critic.
BC_GAMMA = 0.9995
# Down-weights the value objective relative to the action objective during BC. Returns are
# O(100s) and action targets O(1), so an unweighted sum lets the critic dominate the shared
# trunk and degrade the actor clone.
VALUE_LOSS_COEF = 0.001

# Difficulty mix for the demonstration set. Weighted toward D0/D1 since the curriculum
# starts at D0, but includes D2 so the warm start is not actively wrong at full difficulty.
DEMO_DIFFICULTY_WEIGHTS = {0: 0.4, 1: 0.4, 2: 0.2}


def expert_raw_action(stir, light, frac, f_max):
    """Encode physical setpoints into the env's raw [-1, 1] action space.

    Mirrors genetic_env.step()'s decode exactly (np.interp on each dim).
    """
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def expert_harvest_frac(od):
    """Proportional surplus-harvest law. See the EXPERT_* comment block above."""
    surplus = (float(od) / EXPERT_OD_SETPOINT) - 1.0
    return float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))


def collect_demonstrations(vec_env, n_episodes, rng):
    """Roll the scripted expert through the normalized env, recording (obs, action).

    VecNormalize stays in training mode here so obs_rms adapts to the state distribution
    the expert actually visits — that is the distribution PPO will see on handoff. Using
    calibration-only stats would leave a train/BC observation mismatch.
    """
    raw_env = _unwrap_raw_env(vec_env)
    f_max = float(getattr(raw_env, "F_MAX", 0.5))

    diffs = list(DEMO_DIFFICULTY_WEIGHTS.keys())
    probs = np.array([DEMO_DIFFICULTY_WEIGHTS[d] for d in diffs], dtype=np.float64)
    probs /= probs.sum()

    obs_buf, act_buf, ret_buf, results = [], [], [], []
    for ep in range(n_episodes):
        rew_ep = []
        difficulty = int(rng.choice(diffs, p=probs))
        if hasattr(raw_env, "set_difficulty"):
            raw_env.set_difficulty(difficulty)
        else:
            raw_env.difficulty = difficulty

        stir = float(rng.uniform(*EXPERT_STIR_RANGE))
        light = float(rng.uniform(*EXPERT_LIGHT_RANGE))

        obs = vec_env.reset()
        done = False
        steps = 0
        info = {}
        fracs = []
        while not done:
            # Feedback law: read the CURRENT od off the raw env each step, so the demo
            # adapts to how the culture is actually developing rather than replaying a
            # fixed number. This is the whole point of the redesign.
            frac = expert_harvest_frac(getattr(raw_env, "od", 0.0))
            action = expert_raw_action(stir, light, frac, f_max)
            fracs.append(frac)

            obs_buf.append(np.asarray(obs[0], dtype=np.float32))
            act_buf.append(action)
            obs, reward, done_vec, info_list = vec_env.step(action.reshape(1, -1))
            # Fix #14: record the NORMALIZED reward stream so discounted returns can be
            # computed for value-function pretraining. vec_env has norm_reward=True, so
            # reward[0] is already on the same scale the critic will be trained against
            # during PPO — using raw env rewards here would produce a critic calibrated to
            # the wrong magnitude, which is worse than no pretraining at all.
            rew_ep.append(float(reward[0]))
            done = bool(done_vec[0])
            info = info_list[0] if info_list else {}
            steps += 1
        frac = float(np.mean(fracs))

        # Discounted return-to-go for each timestep of this episode, at the SAME gamma the
        # trainer uses. Episode ends at truncation (7200 steps), so bootstrap value is 0.
        ret_ep = np.zeros(len(rew_ep), dtype=np.float32)
        running = 0.0
        for t in range(len(rew_ep) - 1, -1, -1):
            running = rew_ep[t] + BC_GAMMA * running
            ret_ep[t] = running
        ret_buf.extend(ret_ep.tolist())

        # Read the episode outcome from the terminal INFO dict, not off the raw env:
        # DummyVecEnv auto-resets on done, which zeroes cumulative_harvested_mg before
        # any post-loop attribute read can see it. Same convention held_out_sweep.py and
        # deterministic_eval.py already use.
        harvested = float(info.get("cumulative_harvested_mg", 0.0))
        time_avg_od = float(info.get("time_avg_od", 0.0))
        results.append((harvested, time_avg_od, steps))
        print(f"  demo ep {ep + 1:>3}/{n_episodes}  D{difficulty}  "
              f"stir={stir:6.1f} light={light:7.1f} mean_frac={frac:5.3f}  steps={steps}  "
              f"harvested={harvested:7.1f}mg  time_avg_od={time_avg_od:.4f}")

    return (np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32),
            np.asarray(ret_buf, dtype=np.float32), results)


def summarise_expert(results):
    """Score the scripted expert against the curriculum gates it must clear.

    This is the gate that decides whether cloning is even worth doing: if the expert
    itself cannot pass D1/D2 on the real training start distribution, a clone of it
    certainly will not, and an 8M-step run would be wasted confirming that.
    """
    harvests = np.array([r[0] for r in results], dtype=np.float64)
    ods = np.array([r[1] for r in results], dtype=np.float64)
    steps = np.array([r[2] for r in results], dtype=np.float64)

    med_h, p25_h = float(np.median(harvests)), float(np.percentile(harvests, 25))
    med_od = float(np.median(ods))
    crash = float(np.mean(steps < 7200))

    print(f"\n  {'='*76}")
    print(f"  EXPERT PERFORMANCE on the real training start distribution (n={len(results)})")
    print(f"  {'='*76}")
    print(f"    median harvested_mg : {med_h:8.1f}   p25: {p25_h:7.1f}   "
          f"min: {harvests.min():.1f}  max: {harvests.max():.1f}")
    print(f"    median time_avg_od  : {med_od:8.4f}")
    print(f"    crash_rate          : {crash*100:7.1f}%")
    for tier, mh, mp, mo, mc in (("D1", 60.0, 30.0, 0.008, 0.10), ("D2", 90.0, 50.0, 0.011, 0.08)):
        ok = (med_h >= mh) and (p25_h >= mp) and (med_od >= mo) and (crash <= mc)
        marks = (f"h{'+' if med_h >= mh else '-'} p25{'+' if p25_h >= mp else '-'} "
                 f"od{'+' if med_od >= mo else '-'} crash{'+' if crash <= mc else '-'}")
        print(f"    vs {tier} gate: {'PASS' if ok else 'FAIL'}   [{marks}]")
    print(f"  {'='*76}\n")
    return med_h, p25_h, med_od, crash


def behaviour_clone(model, obs_arr, act_arr, ret_arr, epochs, batch_size, lr, rng,
                    critic_epochs=20):
    """Joint supervised pretraining of the actor (action mean) AND critic (value head).

    Per-timestep with a zeroed LSTM state (see module docstring).

    Fix #14 (v18): the critic is now pretrained too. v17 cloned ONLY the actor, leaving a
    randomly-initialised value head. PPO's first updates therefore computed advantages from
    a meaningless baseline, and v17's deterministic performance decayed steadily from the
    handoff onward (harvest 113 -> 103 -> 113 -> 98.9mg over the first four chunks, ending at
    72-80mg; time_avg_od 0.0215 -> 0.0022). A garbage critic producing large, wrongly-signed
    advantages against a good actor is a well-known way to destroy a cloned policy, and it is
    the leading explanation for that decay now that the reward-exploit and exploration-noise
    hypotheses have both been measured and refuted (see recurrent_ppo.py's Fix #13 comment).
    Regressing the value head onto discounted returns-to-go from the expert's own rollouts
    gives PPO a calibrated baseline from step one.
    """
    policy = model.policy
    device = policy.device
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs_arr, device=device)
    act_t = torch.as_tensor(act_arr, device=device)
    ret_t = torch.as_tensor(ret_arr, device=device)
    n = obs_t.shape[0]

    lstm = policy.lstm_actor
    n_layers, hidden = lstm.num_layers, lstm.hidden_size

    for epoch in range(epochs):
        perm = torch.as_tensor(rng.permutation(n), device=device)
        total, total_v, batches = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs_b, act_b, ret_b = obs_t[idx], act_t[idx], ret_t[idx]
            bs = obs_b.shape[0]

            # Fresh (zero) recurrent state for every sample, flagged as an episode start.
            states = (
                torch.zeros(n_layers, bs, hidden, device=device),
                torch.zeros(n_layers, bs, hidden, device=device),
            )
            episode_starts = torch.ones(bs, dtype=torch.float32, device=device)

            distribution, _ = policy.get_distribution(obs_b, states, episode_starts)
            pred_mean = distribution.distribution.mean
            actor_loss = torch.nn.functional.mse_loss(pred_mean, act_b)

            # Critic head shares the same recurrent-state convention as the actor.
            v_states = (
                torch.zeros(n_layers, bs, hidden, device=device),
                torch.zeros(n_layers, bs, hidden, device=device),
            )
            # NB: predict_values returns a BARE tensor (sb3_contrib policies.py:280-285), not
            # the (value, states) tuple that get_distribution returns — unpacking it would
            # silently split along the batch dimension instead.
            pred_v = policy.predict_values(obs_b, v_states, episode_starts)
            critic_loss = torch.nn.functional.mse_loss(pred_v.reshape(-1), ret_b)

            # VALUE_LOSS_COEF keeps the critic from dominating the shared trunk: returns are
            # O(100s) while action targets are O(1), so an unweighted sum would let the value
            # objective swamp the actor gradients and degrade the clone we actually need.
            loss = actor_loss + VALUE_LOSS_COEF * critic_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total += float(actor_loss.item())
            total_v += float(critic_loss.item())
            batches += 1

        print(f"  BC epoch {epoch + 1:>2}/{epochs}  action MSE = {total / max(batches, 1):.6f}"
              f"   value MSE = {total_v / max(batches, 1):.2f}")

    # ── Critic-only refinement ────────────────────────────────────────────────────────
    # The joint phase above deliberately throttles the value gradient by VALUE_LOSS_COEF
    # (0.001) to stop the O(100s) return targets from dragging the shared trunk and degrading
    # the actor clone. The side effect is an under-fit critic: the joint phase ends with value
    # MSE still falling steeply. Since an uncalibrated critic is exactly what Fix #14 exists to
    # prevent, refine it here at FULL weight with the actor's own parameters frozen — the value
    # head cannot damage what it can no longer move.
    actor_params = []
    for name, p in policy.named_parameters():
        # Freeze everything the actor path uses; leave only the value head + critic LSTM free.
        if not (name.startswith("value_net") or name.startswith("lstm_critic")
                or name.startswith("mlp_extractor.value_net")
                or name.startswith("vf_features_extractor")):
            if p.requires_grad:
                actor_params.append(p)
                p.requires_grad_(False)

    critic_params = [p for p in policy.parameters() if p.requires_grad]
    if critic_params:
        v_opt = torch.optim.Adam(critic_params, lr=lr)
        policy.train()
        for epoch in range(critic_epochs):
            perm = torch.as_tensor(rng.permutation(n), device=device)
            tv, nb = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                obs_b, ret_b = obs_t[idx], ret_t[idx]
                bs = obs_b.shape[0]
                v_states = (
                    torch.zeros(n_layers, bs, hidden, device=device),
                    torch.zeros(n_layers, bs, hidden, device=device),
                )
                episode_starts = torch.ones(bs, dtype=torch.float32, device=device)
                pred_v = policy.predict_values(obs_b, v_states, episode_starts)
                v_loss = torch.nn.functional.mse_loss(pred_v.reshape(-1), ret_b)
                v_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, 0.5)
                v_opt.step()
                tv += float(v_loss.item()); nb += 1
            rmse = (tv / max(nb, 1)) ** 0.5
            print(f"  critic-only epoch {epoch + 1:>2}/{critic_epochs}  "
                  f"value MSE = {tv / max(nb, 1):9.2f}  (RMSE {rmse:6.2f} on returns "
                  f"{float(ret_t.min()):.1f}..{float(ret_t.max()):.1f})")
    else:
        print("  [WARN] critic-only refinement skipped: no unfrozen value parameters found. "
              "Check the parameter-name prefixes against this SB3-contrib version.")

    # Restore actor gradients so the saved model is fully trainable by PPO.
    for p in actor_params:
        p.requires_grad_(True)

    policy.eval()


def verify(model, vec_env, n_episodes, f_max):
    """Deterministic rollouts of the cloned policy — the honest check that BC worked.

    Reports the decoded harvest fraction, which is the quantity every previous run got
    wrong, alongside harvest yield and time_avg_od.
    """
    raw_env = _unwrap_raw_env(vec_env)
    if hasattr(raw_env, "set_difficulty"):
        raw_env.set_difficulty(1)
    else:
        raw_env.difficulty = 1

    # Expected shape for the FEEDBACK expert: episode-mean frac ~0.04-0.15 (it varies with
    # how big the culture gets), first600 near ZERO while the culture establishes, rising
    # thereafter. A high-then-decaying profile would be v16b's failure mode returning.
    print("\n  Verification — deterministic rollouts at D1 "
          "(expect mean frac ~0.04-0.15, first600 ~0, last600 higher):")
    for ep in range(n_episodes):
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        fracs, done = [], False
        info = {}
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
            obs, _reward, done_vec, info_list = vec_env.step(action)
            done = bool(done_vec[0])
            info = info_list[0] if info_list else {}
            fracs.append(float(np.interp(action[0][2], [-1, 1], [0.0, f_max])))

        fracs = np.asarray(fracs)
        print(f"    ep {ep + 1}: frac mean={fracs.mean():.3f} std={fracs.std():.3f} "
              f"first600={fracs[:600].mean():.3f} last600={fracs[-600:].mean():.3f} | "
              f"harvested={float(info.get('cumulative_harvested_mg', 0.0)):7.1f}mg "
              f"time_avg_od={float(info.get('time_avg_od', 0.0)):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Behaviour-cloning warm start for RecurrentPPO.")
    parser.add_argument("--episodes", type=int, default=24, help="expert demo episodes")
    parser.add_argument("--epochs", type=int, default=10, help="joint actor+critic BC epochs")
    parser.add_argument("--critic-epochs", type=int, default=20,
                        help="critic-only refinement epochs after the joint phase (actor frozen)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--verify-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true",
                        help="Score the scripted expert and stop — no cloning, no model saved. "
                             "Use this to tune the controller cheaply before spending a run.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("─── Behaviour-Cloning Warm Start ───")
    print(f"Expert      : stir{EXPERT_STIR_RANGE} light{EXPERT_LIGHT_RANGE}, "
          f"harvest = clip({EXPERT_GAIN}*(od/{EXPERT_OD_SETPOINT} - 1), 0, {EXPERT_FRAC_CAP})")
    print(f"Demo mix    : {DEMO_DIFFICULTY_WEIGHTS}")
    print(f"Episodes    : {args.episodes} | BC epochs: {args.epochs}"
          f"{' | EVAL ONLY' if args.eval_only else ''}\n")

    controller = CurriculumStartController()
    controller.log_episode_starts = False
    base_env = DummyVecEnv([make_env(difficulty=2, controller=controller, initial_cells=300)])
    vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Same bootstrap calibration the trainer performs on a fresh start, so obs_rms starts
    # from a comparable place before the expert rollouts refine it.
    raw_env = _unwrap_raw_env(vec_env)
    if hasattr(raw_env, "set_difficulty"):
        raw_env.set_difficulty(0)
    else:
        raw_env.difficulty = 0
    controller.train_diff = 0
    controller.mastery_diff = 0
    print("  Calibrating VecNormalize with 2000 random steps...")
    cal_obs = vec_env.reset()
    for _ in range(2000):
        cal_obs, _, cal_done, _ = vec_env.step([vec_env.action_space.sample()])
        if cal_done[0]:
            cal_obs = vec_env.reset()
    vec_env.reset()
    print(f"  Calibration complete. Obs mean: {vec_env.obs_rms.mean.round(2)}\n")

    print("  Collecting expert demonstrations...")
    obs_arr, act_arr, ret_arr, results = collect_demonstrations(vec_env, args.episodes, rng)
    med_h, p25_h, med_od, crash = summarise_expert(results)
    print(f"  Collected {obs_arr.shape[0]:,} (obs, action) pairs.\n")

    if args.eval_only:
        print("  --eval-only: expert scored above, no cloning performed.")
        return

    # Refuse to clone an expert that cannot clear the tier the curriculum starts working
    # toward. Cloning a sub-D1 expert guarantees a sub-D1 policy, and the 8M-step run that
    # would follow could only confirm that at great cost.
    if med_h < 60.0 or med_od < 0.008:
        raise SystemExit(
            f"\n  ABORT: expert fails the D1 gate on the real start distribution "
            f"(median harvest {med_h:.1f} vs 60.0, median time_avg_od {med_od:.4f} vs 0.0080).\n"
            f"  Tune EXPERT_OD_SETPOINT / EXPERT_GAIN / EXPERT_FRAC_CAP and re-run with\n"
            f"  --eval-only until the expert clears D1 before cloning it.\n"
        )

    # Hyperparameters MUST match recurrent_ppo.py's fresh-model construction, or the
    # handoff silently rebuilds a differently-shaped policy.
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=7200,
        batch_size=240,
        n_epochs=4,
        ent_coef=ENTROPY_INIT,
        gamma=BC_GAMMA,
        gae_lambda=0.98,
        target_kl=None,
    )

    print(f"  Behaviour cloning (actor + critic, gamma={BC_GAMMA}, "
          f"return range {ret_arr.min():.1f}..{ret_arr.max():.1f})...")
    behaviour_clone(model, obs_arr, act_arr, ret_arr, args.epochs, args.batch_size, args.lr, rng,
                    critic_epochs=args.critic_epochs)

    f_max = float(getattr(_unwrap_raw_env(vec_env), "F_MAX", 0.5))
    verify(model, vec_env, args.verify_episodes, f_max)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    vec_env.save(NORM_PATH)
    # Minimal curriculum bookkeeping so the trainer's --resume path has a state file to
    # read. All counters start at zero: this is a warm-started policy, not a resumed run.
    save_state(STATE_PATH, {
        "steps_done": 0,
        "chunk_idx": 0,
        "current_difficulty": 0,
        "mastery_streak": 0,
        "demotion_streak": 0,
        "plateau_counter": 0,
        "plateau_kicks_this_difficulty": 0,
        "entropy_multiplier": 1.0,
        "completed_episodes": 0,
        "saved_population_state": None,
    })

    print(f"\n  Saved warm start:")
    print(f"    model : {MODEL_PATH}.zip")
    print(f"    norm  : {NORM_PATH}")
    print(f"    state : {STATE_PATH}")
    print(f"\n  Hand off with:")
    print(f"    python recurrent_ppo.py --resume {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
