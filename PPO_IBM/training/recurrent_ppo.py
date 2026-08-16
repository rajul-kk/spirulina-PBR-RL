
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

import os
import sys
from collections import defaultdict, deque
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Re-exported here (not just used internally) so that:
#  1) old checkpoints pickled while this file was a monolith and recorded
#     `__main__.ActionSmoothnessWrapper` etc. can still resolve those names
#     when this script is run directly as `python recurrent_ppo.py`.
#  2) other scripts (e.g. evaluate_agent.py) that import these names from
#     `recurrent_ppo` keep working unchanged.
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
from deterministic_eval import run_deterministic_eval_episode
from callbacks import (
    TQDMActionCallback, EntropyLoggingCallback, PopulationStitchCallback, EpisodeMetricsCallback,
)
from env_factory import make_env
from training_state import find_latest_checkpoint, load_state, save_state

# Linear LR decay, driven by OUR OWN steps_done/TOTAL_TRAINING_STEPS tracking rather than
# SB3's built-in progress_remaining. SB3 computes progress_remaining from the
# total_timesteps argument passed to THIS model.learn() call, not the grand training
# budget — and this codebase calls learn() once per 100k-step chunk with
# reset_num_timesteps=False. Verified against SB3's source before wiring this in: each
# chunk call re-derives its own "total_timesteps" as num_timesteps-so-far + this chunk's
# size, so progress_remaining restarts near 1.0 at the start of every chunk and always
# hits 0 by the end of that same chunk — a 40-chunk sawtooth, not a smooth 4M-step decay.
# A naive SB3 schedule would have silently cut the LR toward its floor inside nearly every
# chunk rather than only near the true end of training. Sidestepped the same way this file
# already handles ent_coef (see model.ent_coef assignment in the chunk loop): an external,
# manually-updated value the schedule function reads from, refreshed once per chunk from
# the real steps_done/TOTAL_TRAINING_STEPS ratio.
LR_MAX = 5e-4
LR_MIN = 5e-5
_lr_state = {"value": LR_MAX}

# Fix #17 (v20): the behaviour-cloned controller's held-out D2-passing scores, printed beside
# every deterministic eval as a fixed reference line. That policy (model_data/
# BEST_bc_clone_D2_validated/) is the best this project has produced — median harvest 109.4mg,
# p25 63.8, time_avg_od 0.0191, 0% crash over 40 held-out seeds — and it required no RL at all.
# Showing it inline makes "is PPO anywhere near the thing we already have?" answerable at a
# glance instead of by cross-referencing a separate document mid-run.
BC_REFERENCE = {"harvest": 109.4, "p25": 63.8, "time_avg_od": 0.0191, "crash": 0.0}

# Fix #23 (v25): which policy the curriculum gate advances on. "dual" = stochastic AND
# deterministic (default; see the conjunction site below for the full rationale).
# "stochastic" = stochastic only, for a self-consistent stochastic-deployment experiment.
# Override per run with the GATE_MODE environment variable so no source edit is needed:
#     GATE_MODE=stochastic python training/recurrent_ppo.py
GATE_MODE = os.environ.get("GATE_MODE", "dual").strip().lower()
if GATE_MODE not in ("dual", "stochastic"):
    raise SystemExit(f"GATE_MODE must be 'dual' or 'stochastic', got {GATE_MODE!r}")

# Fix #24 (v26): explicit, RECORDED seed. Until now nothing seeded numpy/torch/the env, so two
# runs of the same configuration differed by an unknown mixture of config effect and RNG draw.
# That directly weakens a conclusion already reported: "v21's time_avg_od 0.0094 was not
# reproducible" rested on v23 (same config) returning 0.0066 — but with no seed control, that
# spread cannot be attributed to the configuration rather than the seed. With the seed pinned
# and logged, a replication isolates config effects, and a deliberate seed sweep measures RNG
# variance separately. Set RUN_SEED to compare configurations; vary it to measure variance.
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
    # set_random_seed covers all three plus CUDA; the env and action space are seeded separately
    # below since they draw from their own generators.
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(RUN_SEED)

    model_name = "model_data/recurrent_ppo_genetic_ibm"
    checkpoint_dir = "./model_data/recurrent_checkpoints/"
    state_path = "model_data/recurrent_training_state.pkl"
    norm_path = "model_data/recurrent_vec_normalize.pkl"
    tensorboard_log = "./ppo_recurrent_tensorboard/"
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
    # degrades still yields its peak rather than its final weights. See the scoring comment
    # at the [BEST-DET] block below.
    best_det_score = -1.0
    best_det_dir = "model_data/best_det_checkpoint"
    d2_mastery_achieved = False  # early-stop signal once D2 (terminal tier) is mastered
    # Fix #29: early-stop signal for a sustained deterministic-gate failure streak AT D0.
    # capability_fail_streak below is gated on current_difficulty>0 for the DEMOTION branch
    # (there is no tier below D0 to demote to), but that guard also meant D0 had NO active
    # response to an in-place policy collapse at all. Confirmed live in v29: PPO's det crash
    # rate climbed 0%->80% over 5 chunks while capability_fail_streak sat structurally stuck
    # at 0 (the guard prevented it from ever incrementing) and the already-active plateau-kick
    # mechanism (entropy bumps, unrelated to this failure mode) kept firing on its own schedule
    # without arresting the decline. Rather than attempt a live mid-training weight reload
    # (risky: SB3 optimizer/rollout-buffer state can desync from a hot-swapped policy), this
    # stops the run cleanly with a clear diagnostic once the same failure signal that would
    # demote at D1/D2 sustains at D0 — burning the rest of an 8M-step budget on a policy known
    # to be failing its own gate is worse than stopping and deciding explicitly what to do next.
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
            # explicitly re-assign in case a saved checkpoint's pickled schedule survives
            # deserialization instead of being overridden — the sawtooth bug this schedule
            # exists to avoid would otherwise silently return on any future resume.
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
            # only after two competing hypotheses were measured and REFUTED:
            #
            #   (a) reward-structure exploit — refuted by reward_ab.py: on 8 identical episodes
            #       the reward function ranks the scripted expert +313 ABOVE v17 (1079 vs 766),
            #       entirely via reward_od. reward_biomass contributed 11.8, not the ~1440 its
            #       theoretical ceiling suggested (tanh(per_cell_growth/5) is tiny at realistic
            #       growth rates, and the flat -0.010 penalty offsets most of the rest), and
            #       differed between the two policies by 0.4. The reward is NOT exploitable.
            #   (b) exploration noise making the expert's strategy unachievable — refuted by
            #       noise_sensitivity.py: the expert keeps 94.8% of its noise-free reward at
            #       sigma=0.50 (exactly the train/std v15/v16b/v17 all sat at) and dominates
            #       v17 at EVERY sigma from 0.0 to 0.70. No crossover. Entropy left untouched.
            #
            # What the evidence does point at: v17 learned stir and light CORRECTLY (light
            # settled at ~1000umol, the sweep optimum) and only harvest incorrectly. The
            # distinguishing feature is credit frequency. Stir and light act on all 7200 steps;
            # the harvest action is applied only on the 12 event steps
            # (HARVEST_INTERVAL_STEPS=600), so on 7188 of 7200 steps the policy emits a harvest
            # value the env ignores while PPO still assigns it advantage — 599 of every 600
            # gradient samples on that dimension are spurious credit.
            #
            # gamma compounds this. At 0.995 the effective horizon is 1/(1-gamma) = 200 steps,
            # while a harvest decision's consequence unfolds over the following 600+ steps and
            # compounds for thousands. 0.995^600 = 0.049, so the immediate harvest reward is
            # undiscounted while its OD cost is ~95% invisible — the agent cannot see past the
            # current harvest cycle, which is precisely the trade-off the expert exploits
            # (forgo harvest now, hold OD, harvest more across the remaining ~100h).
            # At 0.9995: horizon 2000 steps (~3.3 harvest cycles) and 0.9995^600 = 0.741, a 15x
            # improvement in the visibility of the next cycle. Not pushed to 0.9999 (horizon
            # 10000 steps, beyond the 7200-step episode) to avoid the value-variance blowup that
            # very-near-1 discounting causes; 0.9995 matches the task's actual causal timescale.
            # Consistent with every failure this session having been in the harvest dimension
            # specifically, in whichever direction the local gradient happened to favour.
            gamma=0.9995,
            gae_lambda=0.98,
            # target_kl=0.02 tested in v12 and DISABLED after a clear regression: det crash
            # rate climbed to 73.3% by chunk 7 (vs. v11's clean 0% crash at a comparable
            # point under the identical reward config, no other change), ep_rew_mean sat
            # deeply negative and flat (~-50) instead of the healthy early climb v11 showed,
            # and an entropy plateau-kick made it worse, not better. Hypothesis: at 0.02,
            # target_kl's per-minibatch early-stopping fired often enough (observed
            # "Early stopping ... max kl: 0.03-0.09" on most iterations) to cut PPO's 4
            # nominal epochs short most of the time, starving the policy of the gradient
            # steps needed to correct crash-prone behavior during early, fast-changing
            # training — exactly when full updates matter most. Not re-tried at a looser
            # value yet; disabled (None) so the concurrent LR-decay change could be tested
            # in isolation as v13. Re-enable only as a deliberate, isolated test, not
            # bundled with other changes — same lesson as the reward-weight guessing
            # earlier this session: change one variable at a time.
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
    # so its per-difficulty rolling history (maxlen=MASTERY_WINDOW) survives chunk
    # boundaries — see EpisodeMetricsCallback docstring in callbacks.py.
    metrics_cb = EpisodeMetricsCallback(window_size=MASTERY_WINDOW)

    # Persistent, per-difficulty rolling history of DETERMINISTIC evaluation episodes —
    # see deterministic_eval.py. Separate from metrics_cb's stochastic history; advancement
    # requires both gates to pass, closing the exploration-noise loophole found this session.
    det_eval_history = defaultdict(lambda: deque(maxlen=DET_EVAL_WINDOW))
    det_eval_seed_counter = 0

    while steps_done < TOTAL_TRAINING_STEPS and not d2_mastery_achieved and not d0_capability_abort:
        chunk_idx += 1
        # Difficulty is now sampled per-episode in CurriculumStartWrapper.reset().
        # train_diff here equals mastery level and is used only for the streak
        # accounting check (criteria_passed and train_diff == current_difficulty).
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
        # progress_remaining — see _lr_schedule_fn comment for why). Computed from
        # steps_done at the START of this chunk, same timing convention as ent_coef above.
        true_progress_remaining = 1.0 - (steps_done / TOTAL_TRAINING_STEPS)
        _lr_state["value"] = LR_MIN + (LR_MAX - LR_MIN) * true_progress_remaining
        print(f"  Learning rate: {_lr_state['value']:.6f} (progress_remaining={true_progress_remaining:.3f})")

        # Fix #22 (v24): anneal a hard cap on actor std over the back half of training, so the
        # MEAN policy converges toward the sampled policy. The deterministic gate and any real
        # deployment use the mean; PPO optimises the samples. `train/std` sat at ~0.50-0.54 in
        # every run to date and the reactive std-band controller never brought it down, so this
        # is applied as an explicit schedule. See entropy_schedule.annealed_std_cap.
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
        # deterministic episodes per chunk, gated the same way as the stochastic rollout
        # above. Closes the exploration-noise loophole confirmed this session — a policy
        # whose deterministic (mean) action never harvests can still look fine under the
        # stochastic gate purely from action-sampling noise around that mean.
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
        # v17, v18 and v19 ALL ended at or near their worst deterministic policy of the run,
        # because training simply stops at the budget with whatever weights it currently has.
        # v19 is the clearest case: its deterministic policy was 149.1mg / od 0.0203 / 0% crash
        # at chunk 1 and 28.3mg / od 0.0002 / 80-93% crash at chunk 80 — the run PRODUCED a good
        # policy and then threw it away. Nothing in the loop preserved it.
        #
        # Score = median harvested_mg scaled by how well time_avg_od meets the CURRENT tier's
        # threshold, hard-zeroed on any crash. Crash-zeroing is deliberate: a policy that
        # crashes is unusable regardless of yield, and crash rate is the metric that exposed
        # v19's collapse while the stochastic gate stayed clean. Tracked across the whole run
        # (not per-tier) so the artifact is simply "the best deployable policy this run found".
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
        #
        # "dual" (default, v5-v24): the stochastic-rollout gate AND the deterministic gate must
        #   both pass. Added because v4 declared D2 mastery on stochastic metrics inflated by
        #   exploration noise, then scored median 0.4mg against a 90mg gate on held-out data.
        #
        # "stochastic" (v25): advance on the stochastic gate alone. The point is NOT that the
        #   deterministic check was wrong — v24 proved the deterministic policy really is far
        #   worse, because the harvest action is clipped at 0 and E[clip(x)] != clip(E[x]) when
        #   the mean sits near that floor, so the sampled policy gets a systematic upward
        #   harvest bias that survives interval-averaging. The point is CONSISTENCY: the real
        #   error in v14/v17 was gating on one policy while validating with another
        #   (held_out_sweep.py is deterministic). If a stochastic controller is acceptable to
        #   deploy, then gating stochastically is legitimate — provided validation is ALSO
        #   stochastic. `held_out_sweep.py --stochastic` exists for exactly that.
        #   Two caveats that remain true in this mode and must be stated with any result:
        #     * the criterion's difficulty drifts, because stochastic metrics depend on
        #       train/std, which the entropy schedule moves during the run;
        #     * these are the rollouts being trained on, so the metric is optimistically
        #       biased in the same way training accuracy is.
        #   The deterministic eval still RUNS and is still logged, so the gap stays visible; it
        #   just no longer blocks advancement.
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
                    # "sustained mastery at full difficulty" signal used for early
                    # stopping (see the while-loop condition below).
                    d2_mastery_achieved = True
                next_difficulty = min(2, current_difficulty + 1)
                mastery_streak  = 0
                demotion_streak = 0

            # Demotion: sustained high crash rate at D1/D2 drops back one level.
            #
            # Fix #15 (v18): ALSO demote on sustained CAPABILITY failure, not just crashes.
            # v17 exposed the gap concretely: it advanced to D2 with a genuinely good policy
            # (det harvest 113mg, time_avg_od 0.0215), then degraded across the following 48
            # D2 chunks to harvest 72-80mg / time_avg_od 0.0022 — failing the SAME criterion
            # (time_avg_od) on all 48 of them — while crash rate stayed at exactly 0.00%.
            # Because demotion keyed only on crash_rate, nothing ever walked it back down, and
            # it burned ~4.8M steps sitting at a tier it could no longer do. Held-out validation
            # then failed at BOTH D1 and D2.
            #
            # A tier the policy cannot satisfy is not a useful training distribution: dropping
            # back one level restores a solvable task and lets it re-earn the advance. Keyed on
            # the DETERMINISTIC gate (det_criteria_passed) rather than the stochastic one, since
            # deterministic behaviour is what the held-out validation and any real deployment
            # actually use. Threshold is deliberately long (CAPABILITY_DEMOTION_CHUNKS) so
            # ordinary chunk-to-chunk noise or a normal pre-advance plateau cannot trigger it —
            # only a sustained inability to perform at the current tier.
            # Fix #29: no longer gated on current_difficulty>0 — see d0_capability_abort's
            # definition above for why. The counter now tracks sustained det-gate failure at
            # ANY tier, including D0; only the RESPONSE differs below (demote vs. abort),
            # since D0 has no tier to demote to.
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
                    # cleared the det gate yet", not "is collapsing" — live-verified this fires
                    # on a run with 0.00% crash rate and steadily growing harvest (38.8/30.0,
                    # p25 29.8/15.0, both passing; only time_avg_od lagging, 0.0007/0.0040) just
                    # as readily as on genuine collapse (the original v29 trigger: crash rate
                    # climbing 0%->80% with harvest/od both declining). The D1/D2 demotion
                    # branch above deliberately does NOT require a crash floor (Fix #15 exists
                    # specifically to catch v17-style quality regression at 0% crash), but that
                    # rationale doesn't transfer to D0: D1/D2 demotion had a proven prior-good
                    # baseline to fall back to, while a D0 run stuck below the OD bar from the
                    # start has no such baseline to compare against — "never yet passed" and
                    # "regressed from passing" are not the same signal at the floor tier. Only
                    # abort D0 when crash rate is ALSO elevated, the same threshold and
                    # rationale used for D1/D2's crash-based demotion_streak.
                    print(f"  [CAPABILITY ABORT] deterministic gate failed "
                          f"{capability_fail_streak} consecutive chunks at D0 (crash rate "
                          f"{stats['crash_rate']:.2%}) — no tier to demote to, stopping run. "
                          f"Best deterministic checkpoint so far (score {best_det_score:.1f}) "
                          f"is preserved at {best_det_dir}; this run's remaining budget is not "
                          f"worth spending on a policy that has structurally failed its own "
                          f"gate for {capability_fail_streak} straight chunks.")
                    d0_capability_abort = True
                # else: sustained det-gate failure at D0 but crash rate is healthy — this is
                # "slow but not broken" (the v30 case above), not the collapse pattern this
                # exists to catch. Keep training; capability_fail_streak stays pinned at/above
                # threshold and is silently re-checked each subsequent chunk at no real cost.

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
                # Absolute cap = 0.5 * ENTROPY_MAX / decayed_base. Prevents runaway regardless
                # of how far the base has decayed. High entropy ≠ useful exploration when stuck.
                #
                # Fix #12 (v16): the boost is also budgeted — MAX_PLATEAU_KICKS_PER_DIFFICULTY
                # kicks per difficulty, then plateau chunks stop touching entropy at all so the
                # policy is allowed to converge. See curriculum_schedule.py for the rationale.
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
    model_path   = "model_data/recurrent_ppo_genetic_ibm"
    norm_path    = "model_data/recurrent_vec_normalize.pkl"
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
    # Lower LR for fine-tuning: avoids catastrophic forgetting of curriculum knowledge
    model.learning_rate = 1e-4
    print("  ✔ Model loaded — learning_rate set to 1e-4 for fine-tune")

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path="./model_data/recurrent_checkpoints/",
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
            "model_data/recurrent_vec_normalize.pkl",
            "model_data/recurrent_training_state.pkl",
            "model_data/recurrent_checkpoints/"
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
