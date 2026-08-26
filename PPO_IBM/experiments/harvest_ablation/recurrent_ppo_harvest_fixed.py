
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------

import os
import sys
from collections import defaultdict, deque
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Re-exported here (not just used internally) so that:
# (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-25)
from env_utils import unwrap_raw_env as _unwrap_raw_env
from wrappers import ActionSmoothnessWrapper, ACTION_SMOOTH_WRAPPER_PENALTY
from entropy_schedule import (
    ENTROPY_INIT, ENTROPY_DECAY, ENTROPY_MIN, ENTROPY_MAX,
    STD_BAND_LOW, STD_BAND_HIGH, STD_HARD_CAP, STD_CONTROL_EVERY_EPISODES,
    ENTROPY_ADJUST_UP, ENTROPY_ADJUST_DOWN, ENTROPY_RELAX_STEP,
    ENTROPY_MULT_MIN, ENTROPY_MULT_MAX, ENTROPY_PLATEAU_CAP, STD_LOW_PUSH_MIN_ENT_COEF,
    annealed_std_cap,
    entropy_decay_value, entropy_hybrid_value, clamp_policy_std,
)
from curriculum_schedule import (
    TOTAL_TRAINING_STEPS, CHUNK_STEPS, MASTERY_WINDOW, MASTERY_MIN_EPISODES,
    MASTERY_REQUIRED_STREAK, DEMOTION_CRASH_RATE, DEMOTION_STREAK_REQUIRED,
    PLATEAU_CHUNKS, MAX_PLATEAU_KICKS_PER_DIFFICULTY, CAPABILITY_DEMOTION_CHUNKS,
    ADVANCE_TARGETS, MIXING_PROBS,
    DET_EVAL_EPISODES_PER_CHUNK, DET_EVAL_WINDOW, DET_MASTERY_MIN_EPISODES,
    _sample_init_cells, _sample_training_difficulty, _compute_curriculum_stats,
    CurriculumStartController, CurriculumStartWrapper,
)
from callbacks import (
    TQDMActionCallback, EntropyLoggingCallback, PopulationStitchCallback, EpisodeMetricsCallback,
)
from training_state import find_latest_checkpoint, load_state, save_state

# --- harvest-ablation wiring: forked env_factory/deterministic_eval (see README.md) ---
import sys as _sys2, os as _os2
_sys2.path.insert(0, _os2.path.dirname(_os2.path.abspath(__file__)))
from env_factory_harvest_fixed import make_env
from deterministic_eval_harvest_fixed import run_deterministic_eval_episode
# ---------------------------------------------------------------------------------------

# Linear LR decay, driven by OUR OWN steps_done/TOTAL_TRAINING_STEPS tracking rather than
# (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-62)
LR_MAX = 5e-4
LR_MIN = 5e-5
_lr_state = {"value": LR_MAX}

# Fix #17 (v20): the behaviour-cloned controller's held-out D2-passing scores, printed beside
# (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-79)
BC_REFERENCE = {"harvest": 109.4, "p25": 63.8, "time_avg_od": 0.0191, "crash": 0.0}

# Fix #23 (v25): which policy the curriculum gate advances on. "dual" = stochastic AND
# (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-87)
GATE_MODE = os.environ.get("GATE_MODE", "dual").strip().lower()
if GATE_MODE not in ("dual", "stochastic"):
    raise SystemExit(f"GATE_MODE must be 'dual' or 'stochastic', got {GATE_MODE!r}")

# Fix #24 (v26): explicit, RECORDED seed. Until now nothing seeded numpy/torch/the env, so two
# (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-96)
RUN_SEED = int(os.environ.get("RUN_SEED", "0"))


def _lr_schedule_fn(_progress_remaining_ignored: float) -> float:
    """Passed to RecurrentPPO as `learning_rate`. Ignores SB3's own progress_remaining
    argument (meaningless here per the chunked-call issue above) and returns whatever
    the training loop last wrote to _lr_state, based on true overall progress."""
    return _lr_state["value"]


def train_recurrent_agent(resume=False):
    print("─── Recurrent PPO Curriculum Training ───")
    print("Algorithm : RecurrentPPO (MlpLstmPolicy)")
    print("Env       : GeneticPhotobioreactorEnv (genetic_env)")
    print(f"Mode      : Adaptive Curriculum | Total budget: {TOTAL_TRAINING_STEPS:,} steps")
    print(f"Gate      : {GATE_MODE} | seed: {RUN_SEED}")

    # Fix #24 (v26): seed numpy, torch and Python's RNG before anything samples. SB3's
    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-120)
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(RUN_SEED)

    model_name = "model_data/harvest_fixed_ppo/recurrent_ppo_genetic_ibm"
    checkpoint_dir = "./model_data/harvest_fixed_ppo/checkpoints/"
    state_path = "model_data/harvest_fixed_ppo/training_state.pkl"
    norm_path = "model_data/harvest_fixed_ppo/vec_normalize.pkl"
    tensorboard_log = "./ppo_harvest_fixed_tensorboard/"
    start_controller = CurriculumStartController()
    start_controller.log_episode_starts = False

    # Checkpoint callback — always active, saves every 10k global steps
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="recurrent_ppo_ibm",
    )

    # Single persistent env so VecNormalize statistics remain stable.
    base_env = DummyVecEnv([make_env(difficulty=2, controller=start_controller, initial_cells=300)])

    steps_done = 0
    chunk_idx = 0
    current_difficulty = 0
    mastery_streak = 0
    demotion_streak = 0
    plateau_counter = 0
    plateau_kicks_this_difficulty = 0  # Fix #12 (v16): per-difficulty entropy-kick budget
    capability_fail_streak = 0         # Fix #15 (v18): consecutive det-gate failures at tier
    # Fix #17 (v20): preserve the run's best DEPLOYABLE (deterministic) policy, so a run that
    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-152)
    best_det_score = -1.0
    best_det_dir = "model_data/harvest_fixed_ppo/best_det_checkpoint"
    d2_mastery_achieved = False  # early-stop signal once D2 (terminal tier) is mastered
    # Fix #29: early-stop signal for a sustained deterministic-gate failure streak AT D0.
    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-158)
    d0_capability_abort = False
    entropy_multiplier = 1.0

    latest_checkpoint, latest_step = (None, None)
    vec_env = None

    if resume:
        if isinstance(resume, str):
            latest_checkpoint = resume
            latest_step = 0
            print(f"  [CONTINUE] Directed to specific model: {latest_checkpoint}")
        else:
            latest_checkpoint, latest_step = find_latest_checkpoint(
                checkpoint_dir, "recurrent_ppo_ibm_", "_steps.zip"
            )
        saved_state = load_state(state_path)

        if latest_checkpoint is not None:
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, venv=base_env)
                vec_env.training = True
                vec_env.norm_reward = True
            else:
                vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            custom_objects = {
                "n_steps": 7200,
                "batch_size": 240,
                "n_epochs": 4,
                "learning_rate": _lr_schedule_fn,
                "ent_coef": ENTROPY_INIT,
                # Fix #13 (v18): must match the fresh-construction value below, or a resumed
                # run silently trains under a different discount than the run it continues.
                "gamma": 0.9995,
                "gae_lambda": 0.98,
                "target_kl": None,  # see v12 finding below — disabled pending isolation test
                # Override stale obs-space stored in checkpoint (conductivity bound 10000→25000)
                "observation_space": vec_env.observation_space,
            }
            if saved_state is not None:
                print(f"  [CONTINUE] Loading checkpoint: {latest_checkpoint}")
            model = RecurrentPPO.load(latest_checkpoint, env=vec_env, device="auto", custom_objects=custom_objects)
            # Belt-and-suspenders: custom_objects above should already install this, but
            # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-212)
            model.lr_schedule = _lr_schedule_fn
            steps_done         = int(saved_state.get("steps_done", latest_step or 0))
            chunk_idx          = int(saved_state.get("chunk_idx", 0))
            current_difficulty = int(saved_state.get("current_difficulty", 0))
            mastery_streak     = int(saved_state.get("mastery_streak", 0))
            demotion_streak    = int(saved_state.get("demotion_streak", 0))
            plateau_counter    = int(saved_state.get("plateau_counter", 0))
            # Fix #12 (v16): persisted so a resume can't silently hand the run a fresh kick budget.
            # Defaults to 0 when resuming a pre-v16 state file (which predates this key).
            plateau_kicks_this_difficulty = int(saved_state.get("plateau_kicks_this_difficulty", 0))
            capability_fail_streak = int(saved_state.get("capability_fail_streak", 0))
            start_controller.completed_episodes = int(saved_state.get("completed_episodes", 0))
            start_controller.saved_state = saved_state.get("saved_population_state")
            entropy_multiplier = float(saved_state.get("entropy_multiplier", 1.0))
            # Clamp loaded multiplier to new plateau cap — prevents resuming into a stuck high-mult state
            entropy_multiplier = float(np.clip(entropy_multiplier, ENTROPY_MULT_MIN, ENTROPY_PLATEAU_CAP))
            print(
                f"  [CONTINUE] steps={steps_done:,} | D{current_difficulty} | "
                f"streak={mastery_streak} | completed_eps={start_controller.completed_episodes}"
            )
        else:
            if resume:
                print("  [CONTINUE] No matching saved state found. Starting fresh.")
            latest_checkpoint = None

    if not resume or latest_checkpoint is None:
        if vec_env is None:
            vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Calibration is a bootstrap pass, not curriculum training.
        # Force D0 semantics and keep episode-start logs quiet here.
        raw_env_for_cal = _unwrap_raw_env(vec_env)
        if hasattr(raw_env_for_cal, "set_difficulty"):
            raw_env_for_cal.set_difficulty(0)
        else:
            raw_env_for_cal.difficulty = 0
        start_controller.train_diff = 0
        start_controller.mastery_diff = 0
        print("  Calibrating VecNormalize with 2000 random steps...")
        cal_obs = vec_env.reset()
        for _ in range(2000):
            random_act = [vec_env.action_space.sample()]
            cal_obs, _, cal_done, _ = vec_env.step(random_act)
            if cal_done[0]:
                cal_obs = vec_env.reset()
        vec_env.reset()
        print(f"  Calibration complete. Obs mean: {vec_env.obs_rms.mean.round(2)}")

        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            verbose=1,
            learning_rate=_lr_schedule_fn,
            n_steps=7200,
            batch_size=240,  # 7200/240=30 even batches; 256 leaves a 32-sample tail batch
            n_epochs=4,
            ent_coef=ENTROPY_INIT,
            # Fix #13 (v18): gamma 0.995 -> 0.9995. This is a CREDIT-ASSIGNMENT fix, arrived at
            # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-273)
            gamma=0.9995,
            gae_lambda=0.98,
            # target_kl=0.02 tested in v12 and DISABLED after a clear regression: det crash
            # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-309)
            target_kl=None,
            policy_kwargs={
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
            },
            tensorboard_log=tensorboard_log,
            device="auto",
        )

    print("\nResume Summary")
    print(f"  mode            : {'continue' if (resume and latest_checkpoint is not None) else 'fresh'}")
    print(f"  checkpoint      : {latest_checkpoint if latest_checkpoint else 'none'}")
    print(f"  steps_done      : {steps_done:,}")
    print(f"  chunk_idx       : {chunk_idx}")
    print(f"  difficulty      : D{current_difficulty}")
    print(f"  mastery_streak  : {mastery_streak}")
    print(f"  completed_eps   : {start_controller.completed_episodes}")
    print(f"  saved_warmstart : {'yes' if start_controller.saved_state is not None else 'no'}")
    start_controller.log_episode_starts = True

    action_log_cb = TQDMActionCallback()
    entropy_log_cb = EntropyLoggingCallback()
    stitch_cb = PopulationStitchCallback(
        controller=start_controller, pop_threshold=1_100, difficulty_min=1, verbose=1
    )
    raw_env = _unwrap_raw_env(vec_env)
    episodes_since_std_control = 0
    did_first_std_control = False

    # Instantiated once and reused across the whole chunk loop (not recreated per chunk)
    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-352)
    metrics_cb = EpisodeMetricsCallback(window_size=MASTERY_WINDOW)

    # Persistent, per-difficulty rolling history of DETERMINISTIC evaluation episodes —
    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-357)
    det_eval_history = defaultdict(lambda: deque(maxlen=DET_EVAL_WINDOW))
    det_eval_seed_counter = 0

    while steps_done < TOTAL_TRAINING_STEPS and not d2_mastery_achieved and not d0_capability_abort:
        chunk_idx += 1
        # Difficulty is now sampled per-episode in CurriculumStartWrapper.reset().
        # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-365)
        train_diff = current_difficulty
        start_controller.mastery_diff = current_difficulty
        # Seed chunk header context without double-printing episode start.
        start_controller.log_episode_starts = False
        vec_env.reset()
        start_controller.log_episode_starts = True

        metrics_cb.start_new_chunk()
        this_chunk = min(CHUNK_STEPS, TOTAL_TRAINING_STEPS - steps_done)
        print(
            f"\n[Chunk {chunk_idx}] train_diff=D{train_diff} | "
            f"mastery_diff=D{current_difficulty} | start={getattr(raw_env, 'episode_start_mode', 'low')} | "
            f"init={getattr(raw_env, 'initial_cells', 0):,} | steps={this_chunk:,}"
        )
        scheduled_ent_coef = entropy_hybrid_value(chunk_idx, entropy_multiplier)
        model.ent_coef = scheduled_ent_coef
        print(
            f"  Entropy coef (hybrid): {model.ent_coef:.6f} "
            f"(mult={entropy_multiplier:.2f}x)"
        )

        # Linear LR decay across the TRUE 4M-step budget (steps_done, not SB3's per-call
        # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-389)
        true_progress_remaining = 1.0 - (steps_done / TOTAL_TRAINING_STEPS)
        _lr_state["value"] = LR_MIN + (LR_MAX - LR_MIN) * true_progress_remaining
        print(f"  Learning rate: {_lr_state['value']:.6f} (progress_remaining={true_progress_remaining:.3f})")

        # Fix #22 (v24): anneal a hard cap on actor std over the back half of training, so the
        # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-396)
        _std_cap_now = annealed_std_cap(1.0 - true_progress_remaining)
        _anneal_res = clamp_policy_std(model, _std_cap_now)
        if _anneal_res is not None:
            _sb, _sa = _anneal_res
            if _sa < _sb - 1e-6:
                print(f"  [STD ANNEAL] cap={_std_cap_now:.3f} applied: std {_sb:.3f} -> {_sa:.3f}")
            else:
                print(f"  [STD ANNEAL] cap={_std_cap_now:.3f} (std {_sb:.3f}, no clamp needed)")

        action_log_cb.train_diff = train_diff
        action_log_cb.mastery_diff = current_difficulty

        model.learn(
            total_timesteps=this_chunk,
            reset_num_timesteps=False,
            tb_log_name="recurrent_ppo_curriculum",
            progress_bar=True,
            callback=[checkpoint_cb, action_log_cb, stitch_cb, metrics_cb, entropy_log_cb],
        )

        chunk_episodes = len(metrics_cb.episode_metrics)
        episodes_since_std_control += chunk_episodes
        should_run_std_control = (
            (not did_first_std_control)
            or (episodes_since_std_control >= STD_CONTROL_EVERY_EPISODES)
        )

        if should_run_std_control:
            latest_std = model.logger.name_to_value.get("train/std", None)
            if latest_std is not None:
                latest_std = float(latest_std)
                clamp_res = None
                if latest_std > STD_BAND_HIGH:
                    band_state = "high"
                    entropy_multiplier = max(
                        ENTROPY_MULT_MIN,
                        entropy_multiplier * ENTROPY_ADJUST_DOWN,  # proportional decay: ×0.25/check
                    )
                    clamp_res = clamp_policy_std(model, STD_HARD_CAP)
                elif latest_std < STD_BAND_LOW and float(model.ent_coef) > STD_LOW_PUSH_MIN_ENT_COEF:
                    band_state = "low"
                    entropy_multiplier = min(
                        ENTROPY_MULT_MAX,
                        entropy_multiplier + ENTROPY_ADJUST_UP,
                    )
                else:
                    band_state = "in-band"
                    if entropy_multiplier > 1.0:
                        entropy_multiplier = max(
                            1.0,
                            entropy_multiplier - ENTROPY_RELAX_STEP,
                        )
                    elif entropy_multiplier < 1.0:
                        entropy_multiplier = min(
                            1.0,
                            entropy_multiplier + ENTROPY_RELAX_STEP,
                        )
                print(
                    f"  Entropy feedback: train/std={latest_std:.3f}, "
                    f"next_mult={entropy_multiplier:.2f}x, "
                    f"band={band_state}"
                )
                if clamp_res is not None:
                    std_before, std_after = clamp_res
                    model.logger.record("train/std_clamp_before", std_before)
                    model.logger.record("train/std_clamp_after", std_after)
                    if std_after < std_before:
                        print(
                            "  Std hard-cap applied: "
                            f"{std_before:.3f} -> {std_after:.3f} "
                            f"(cap={STD_HARD_CAP:.2f})"
                        )
                did_first_std_control = True
                episodes_since_std_control = 0

        steps_done += this_chunk

        # Gate on the persistent rolling window for this difficulty (up to MASTERY_WINDOW
        # episodes, survives chunk boundaries) rather than only this chunk's ~14 episodes.
        stats = _compute_curriculum_stats(
            metrics_cb.history_for_difficulty(current_difficulty), mastery_diff=current_difficulty
        )

        # Deterministic evaluation pass (see deterministic_eval.py): a handful of genuinely
        # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-484)
        for _ in range(DET_EVAL_EPISODES_PER_CHUNK):
            det_eval_seed_counter += 1
            rec = run_deterministic_eval_episode(
                model, vec_env.obs_rms, current_difficulty, seed=100_000 + det_eval_seed_counter
            )
            det_eval_history[current_difficulty].append(rec)
        det_stats = _compute_curriculum_stats(
            list(det_eval_history[current_difficulty]), mastery_diff=current_difficulty
        )
        print(
            f"  [Det] eps={det_stats['episodes']} harvest_mg={det_stats['median_harvested_mg']:.1f} "
            f"p25={det_stats['p25_harvested_mg']:.1f} time_avg_od={det_stats['median_time_avg_od']:.4f} "
            f"crash={det_stats['crash_rate']*100:.1f}%"
            f"  | BC ref: {BC_REFERENCE['harvest']:.1f}mg od={BC_REFERENCE['time_avg_od']:.4f}"
        )

        # ── Fix #17 (v20): best-deterministic checkpoint tracking ─────────────────────────
        # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-505)
        _det_od = det_stats["median_time_avg_od"]
        _det_target_od = ADVANCE_TARGETS.get(current_difficulty, {}).get("min_median_time_avg_od", 0.008)
        _od_ratio = min(1.0, _det_od / max(_det_target_od, 1e-9))
        det_score = det_stats["median_harvested_mg"] * _od_ratio * (1.0 if det_stats["crash_rate"] <= 0.0 else 0.0)
        if det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES and det_score > best_det_score:
            best_det_score = det_score
            os.makedirs(best_det_dir, exist_ok=True)
            model.save(os.path.join(best_det_dir, "recurrent_ppo_genetic_ibm"))
            vec_env.save(os.path.join(best_det_dir, "recurrent_vec_normalize.pkl"))
            with open(os.path.join(best_det_dir, "best_det_info.txt"), "w", encoding="utf-8") as fh:
                fh.write(
                    f"step={steps_done}\nchunk={chunk_idx}\ndifficulty=D{current_difficulty}\n"
                    f"score={det_score:.2f}\nmedian_harvested_mg={det_stats['median_harvested_mg']:.1f}\n"
                    f"p25_harvested_mg={det_stats['p25_harvested_mg']:.1f}\n"
                    f"median_time_avg_od={_det_od:.4f}\ncrash_rate={det_stats['crash_rate']:.4f}\n"
                )
            print(f"  [BEST-DET] new best deterministic policy (score {det_score:.1f}) saved -> {best_det_dir}")

        # ── Advancement & Demotion ────────────────────────────────────────────
        target = ADVANCE_TARGETS.get(current_difficulty)
        criteria_passed = False
        det_criteria_passed = False
        criteria_detail = {}
        if target is not None and stats["episodes"] >= MASTERY_MIN_EPISODES:
            criteria_detail = {
                "harvest": (stats["median_harvested_mg"],   target["min_median_harvested_mg"],
                            stats["median_harvested_mg"]    >= target["min_median_harvested_mg"]),
                "p25":     (stats["p25_harvested_mg"],       target["min_p25_harvested_mg"],
                            stats["p25_harvested_mg"]        >= target["min_p25_harvested_mg"]),
                "crash":   (stats["crash_rate"],             target["max_crash_rate"],
                            stats["crash_rate"]              <= target["max_crash_rate"]),
                "time_od": (stats["median_time_avg_od"],     target["min_median_time_avg_od"],
                            stats["median_time_avg_od"]      >= target["min_median_time_avg_od"]),
            }
            criteria_passed = all(v[2] for v in criteria_detail.values())
        if target is not None and det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES:
            det_criteria_passed = (
                det_stats["median_harvested_mg"]  >= target["min_median_harvested_mg"]
                and det_stats["p25_harvested_mg"]   >= target["min_p25_harvested_mg"]
                and det_stats["crash_rate"]         <= target["max_crash_rate"]
                and det_stats["median_time_avg_od"] >= target["min_median_time_avg_od"]
            )
        # Fix #23 (v25): GATE_MODE selects which policy the curriculum advances on.
        # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-559)
        if GATE_MODE == "stochastic":
            pass  # criteria_passed already holds the stochastic verdict
        else:
            criteria_passed = criteria_passed and det_criteria_passed

        next_difficulty = current_difficulty
        if stats["episodes"] >= MASTERY_MIN_EPISODES:
            # Advancement
            if criteria_passed:
                mastery_streak += 1
                demotion_streak = 0
            else:
                mastery_streak = 0

            if mastery_streak >= MASTERY_REQUIRED_STREAK:
                if current_difficulty == 2:
                    # Terminal tier — nowhere further to advance, but this is the
                    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-597)
                    d2_mastery_achieved = True
                next_difficulty = min(2, current_difficulty + 1)
                mastery_streak  = 0
                demotion_streak = 0

            # Demotion: sustained high crash rate at D1/D2 drops back one level.
            # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-605)
            capability_failing = (
                det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES
                and not det_criteria_passed
            )
            if capability_failing:
                capability_fail_streak += 1
            else:
                capability_fail_streak = 0

            if current_difficulty > 0 and stats["crash_rate"] >= DEMOTION_CRASH_RATE:
                demotion_streak += 1
            else:
                demotion_streak = 0

            if capability_fail_streak >= CAPABILITY_DEMOTION_CHUNKS:
                if current_difficulty > 0:
                    print(f"  [CAPABILITY DEMOTION] deterministic gate failed "
                          f"{capability_fail_streak} consecutive chunks at D{current_difficulty} "
                          f"(crash rate {stats['crash_rate']:.2%}, so crash-based demotion never "
                          f"fired) — dropping a tier to restore a solvable task")
                    next_difficulty = max(0, current_difficulty - 1)
                    mastery_streak = 0
                    demotion_streak = 0
                    capability_fail_streak = 0
                elif stats["crash_rate"] >= DEMOTION_CRASH_RATE:
                    # Fix #29 correction (v30 retry): capability_failing alone means "hasn't
                    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-652)
                    print(f"  [CAPABILITY ABORT] deterministic gate failed "
                          f"{capability_fail_streak} consecutive chunks at D0 (crash rate "
                          f"{stats['crash_rate']:.2%}) — no tier to demote to, stopping run. "
                          f"Best deterministic checkpoint so far (score {best_det_score:.1f}) "
                          f"is preserved at {best_det_dir}; this run's remaining budget is not "
                          f"worth spending on a policy that has structurally failed its own "
                          f"gate for {capability_fail_streak} straight chunks.")
                    d0_capability_abort = True
                # else: sustained det-gate failure at D0 but crash rate is healthy — this is
                # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-674)

            if demotion_streak >= DEMOTION_STREAK_REQUIRED:
                next_difficulty = max(0, current_difficulty - 1)
                mastery_streak  = 0
                demotion_streak = 0

        # Plateau detection: consecutive chunks with no streak progress → entropy kick
        if next_difficulty != current_difficulty:
            plateau_counter = 0
            # Fix #12 (v16): fresh kick budget on every difficulty change (advance or demotion).
            plateau_kicks_this_difficulty = 0
            # Fix #15 (v18): the streak is per-tier — a tier change makes prior failures moot.
            capability_fail_streak = 0
        elif mastery_streak > 0:
            plateau_counter = 0
        else:
            plateau_counter += 1
            if plateau_counter >= PLATEAU_CHUNKS:
                # Cap the plateau multiplier so ent_coef stays below 50% of ENTROPY_MAX.
                # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-696)
                if plateau_kicks_this_difficulty >= MAX_PLATEAU_KICKS_PER_DIFFICULTY:
                    print(f"  [PLATEAU] {PLATEAU_CHUNKS} chunks with no streak — kick budget "
                          f"exhausted ({plateau_kicks_this_difficulty}/"
                          f"{MAX_PLATEAU_KICKS_PER_DIFFICULTY} used at D{current_difficulty}), "
                          f"allowing convergence (mult held at {entropy_multiplier:.2f}x)")
                elif entropy_multiplier < ENTROPY_PLATEAU_CAP:
                    entropy_multiplier = min(entropy_multiplier * 2.0, ENTROPY_PLATEAU_CAP)
                    plateau_kicks_this_difficulty += 1
                    print(f"  [PLATEAU] {PLATEAU_CHUNKS} chunks with no streak — "
                          f"boosting entropy multiplier to {entropy_multiplier:.2f}x "
                          f"(hard cap={ENTROPY_PLATEAU_CAP:.1f}x, kick "
                          f"{plateau_kicks_this_difficulty}/{MAX_PLATEAU_KICKS_PER_DIFFICULTY} "
                          f"at D{current_difficulty})")
                else:
                    print(f"  [PLATEAU] {PLATEAU_CHUNKS} chunks with no streak — "
                          f"mult already at cap ({entropy_multiplier:.2f}x), skipping boost")
                plateau_counter = 0

        m_hrv = stats.get("median_harvested_mg", 0.0)
        p_hrv = stats.get("p25_harvested_mg", 0.0)
        c_rt  = stats.get("crash_rate", 0.0)
        m_tod = stats.get("median_time_avg_od", 0.0)

        # Per-criterion breakdown
        if criteria_detail:
            def _cfmt(name, val, thr, passed):
                sym = "+" if passed else "-"
                if name == "crash":
                    return f"[{sym}]{name}:{val:.2%}/{thr:.2%}"
                elif name == "time_od":
                    return f"[{sym}]{name}:{val:.4f}/{thr:.4f}"
                else:
                    return f"[{sym}]{name}:{val:.1f}/{thr:.1f}"
            parts = [_cfmt(k, *v) for k, v in criteria_detail.items()]
            n_met = sum(1 for v in criteria_detail.values() if v[2])
            print(f"  [D{current_difficulty}->D{min(2,current_difficulty+1)}]  "
                  + "  ".join(parts)
                  + f"  ({n_met}/{len(parts)} met)")

        if next_difficulty != current_difficulty:
            direction = "ADVANCED" if next_difficulty > current_difficulty else "DEMOTED"
            print(
                f"  Curriculum {direction}: D{current_difficulty} -> D{next_difficulty} "
                f"| eps={chunk_episodes} harvest_mg={m_hrv:.1f} p25={p_hrv:.1f} "
                f"time_avg_od={m_tod:.4f} crash={c_rt:.2%}"
            )
        else:
            print(
                f"  Curriculum hold D{current_difficulty} "
                f"| eps={stats.get('episodes',0)} adv={mastery_streak}/{MASTERY_REQUIRED_STREAK} "
                f"dem={demotion_streak}/{DEMOTION_STREAK_REQUIRED} plateau={plateau_counter}/{PLATEAU_CHUNKS}"
            )

        print(f"| D{current_difficulty} | eps={chunk_episodes} "
              f"| harvest_mg={m_hrv:.1f} p25={p_hrv:.1f} time_avg_od={m_tod:.4f} "
              f"crash={c_rt*100:.1f}% "
              f"| adv={mastery_streak}/{MASTERY_REQUIRED_STREAK} "
              f"dem={demotion_streak}/{DEMOTION_STREAK_REQUIRED} "
              f"plateau={plateau_counter}/{PLATEAU_CHUNKS} "
              f"capfail={capability_fail_streak}/{CAPABILITY_DEMOTION_CHUNKS} |")
        current_difficulty = next_difficulty

        if d2_mastery_achieved:
            print(
                f"\n[EARLY STOP] D2 mastery confirmed ({MASTERY_REQUIRED_STREAK} consecutive "
                f"passing chunks at full difficulty) at step {steps_done:,} — stopping before "
                f"the {TOTAL_TRAINING_STEPS:,}-step budget. Saving final checkpoint below."
            )
        if d0_capability_abort:
            print(
                f"\n[EARLY STOP] D0 capability abort at step {steps_done:,} — see "
                f"[CAPABILITY ABORT] above. Saving final checkpoint below (not the deployable "
                f"artifact — use {best_det_dir} for that)."
            )

        checkpoint_path = os.path.join(checkpoint_dir, f"recurrent_ppo_ibm_{steps_done}_steps")
        model.save(checkpoint_path)
        vec_env.save(norm_path)
        save_state(
            state_path,
            {
                "steps_done": steps_done,
                "chunk_idx": chunk_idx,
                "current_difficulty": current_difficulty,
                "mastery_streak": mastery_streak,
                "demotion_streak": demotion_streak,
                "plateau_counter": plateau_counter,
                "plateau_kicks_this_difficulty": plateau_kicks_this_difficulty,
                "capability_fail_streak": capability_fail_streak,
                "entropy_multiplier": entropy_multiplier,
                "completed_episodes": start_controller.completed_episodes,
                "saved_population_state": start_controller.saved_state,
            },
        )

    # ── Save final model and normalisation stats ────────────────────────────
    print("\nTraining Complete.")
    model.save(model_name)
    vec_env.save(norm_path)
    print(f"Model saved     → {model_name}.zip")
    print(f"Norm stats saved → {norm_path}")
    return model


def finetune_recurrent_agent(extra_steps: int = 500_000):
    """
    Continue training from a previously saved checkpoint on Difficulty 2 (Full Physics).
    Loads the model weights AND the VecNormalize running statistics so the agent
    doesn't lose its calibrated observation normalisation.
    Uses a lower learning rate (1e-4) to consolidate long-horizon strategies
    without catastrophically forgetting the curriculum knowledge.
    """
    model_path   = "model_data/harvest_fixed_ppo/recurrent_ppo_genetic_ibm"
    norm_path    = "model_data/harvest_fixed_ppo/vec_normalize.pkl"
    model_zip    = model_path + ".zip"

    if not os.path.exists(model_zip):
        print(f"  [ERROR] No saved model found at {model_zip}")
        print("  Run 'python recurrent_ppo.py' first to complete the curriculum.")
        return

    print("─── Recurrent PPO Fine-Tune (Difficulty 2, Full Physics) ───")
    print(f"  Loading weights from : {model_zip}")
    print(f"  Loading norm stats   : {norm_path}")
    print(f"  Extra steps          : {extra_steps:,}")

    # Re-create environment at Difficulty 2
    start_controller = CurriculumStartController()
    base_env = DummyVecEnv([make_env(difficulty=2, controller=start_controller, initial_cells=300)])

    # Load saved normalisation statistics, or create a fresh VecNormalize
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, venv=base_env)
        vec_env.training = True  # Keep updating stats during fine-tune
        print("  ✔ VecNormalize stats loaded")
    else:
        # Fall back to a fresh calibration if pkl is missing
        print("  ⚠ Norm stats missing — recalibrating with 2000 random steps...")
        vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        for _ in range(2000):
            vec_env.step([vec_env.action_space.sample()])
        vec_env.reset()

    # Load the saved model and hot-swap to the new env
    model = RecurrentPPO.load(model_path, env=vec_env, device="auto")
    # Lower LR for fine-tuning: avoids catastrophic forgetting of curriculum knowledge.
    # (full rationale: docs/decision_history.md#--experiments-harvest_ablation-recurrent_ppo_harvest_fixed-py-848)
    _lr_state["value"] = 1e-4
    model.lr_schedule = _lr_schedule_fn
    print("  ✔ Model loaded — learning_rate set to 1e-4 for fine-tune")

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path="./model_data/harvest_fixed_ppo/checkpoints/",
        name_prefix="recurrent_ppo_finetune",
    )

    action_log_cb = TQDMActionCallback()
    stitch_cb = PopulationStitchCallback(
        controller=start_controller, pop_threshold=1_100, difficulty_min=1, verbose=1
    )

    print(f"\n  Starting fine-tune for {extra_steps:,} steps...")
    model.learn(
        total_timesteps=extra_steps,
        reset_num_timesteps=False,  # Continue global step counter
        progress_bar=True,
        callback=[checkpoint_cb, action_log_cb, stitch_cb],
    )

    model.save(model_path)                              # Overwrite with the improved weights
    vec_env.save(norm_path)                             # Update norm stats too
    print(f"\n  Fine-tune complete. Model saved → {model_zip}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recurrent PPO for GeneticPBR")
    parser.add_argument("--finetune", action="store_true",
                        help="Load saved model and continue training for 500K extra steps")
    parser.add_argument("--resume", "--continue", dest="resume", nargs='?', const=True, default=False,
                        help="Continue curriculum training. Optionally provide a specific .zip file path to load.")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Number of extra steps for --finetune mode (default: 500000)")
    parser.add_argument("--reset-training", action="store_true",
                        help="Delete all checkpoints and training state to start fresh on genetic_env")
    args = parser.parse_args()

    if args.reset_training:
        import shutil
        paths_to_clear = [
            "model_data/harvest_fixed_ppo/vec_normalize.pkl",
            "model_data/harvest_fixed_ppo/training_state.pkl",
            "model_data/harvest_fixed_ppo/checkpoints/"
        ]
        print("\n⚠ WARNING: This will delete all PPO training progress!")
        print("Files/directories to be deleted:")
        for path in paths_to_clear:
            if os.path.exists(path):
                print(f"  - {path}")

        confirm = input("\nProceed? (yes/no): ").strip().lower()
        if confirm == 'yes':
            for path in paths_to_clear:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"✓ Deleted directory: {path}")
                    else:
                        os.remove(path)
                        print(f"✓ Deleted file: {path}")
            print("\n✓ Reset complete. Run 'python recurrent_ppo.py' to train from scratch on genetic_env.")
        else:
            print("Reset cancelled.")
        sys.exit(0)

    if args.finetune:
        finetune_recurrent_agent(extra_steps=args.steps)
    else:
        train_recurrent_agent(resume=args.resume)
