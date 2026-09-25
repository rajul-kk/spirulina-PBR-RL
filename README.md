# PPO_IBM — Spirulina photobioreactor control

Reinforcement learning control of a simulated 20L Spirulina photobioreactor
(`GeneticPhotobioreactorEnv`), gated on a three-tier difficulty curriculum (D0 → D1 → D2)
that requires both stochastic rollouts and deterministic evaluation to pass before advancing.

**Current primary track: TD3+BC** (Fujimoto & Gu 2021), recurrent (LSTM or diagonal-LRU
core), demo-seeded replay. This is the project's active line of work and the source of its
best held-out results. **RecurrentPPO** (SB3-contrib) and **TD-MPC2** (model-based, MPPI
planning) are earlier algorithm tracks, both parked — neither ever produced a held-out-
validated D2 policy; see "Historical: PPO and TD-MPC2" below.

All code, scripts, and supporting docs referenced below live in [`PPO_IBM/`](PPO_IBM/), the
project directory this README describes.

## Results and history — where to look

- **[`docs/decision_history.md`](PPO_IBM/docs/decision_history.md)** — the primary engineering
  log. Every non-trivial decision, bug, fix, and run result across the whole project (PPO,
  TD-MPC2, and TD3+BC) is recorded here verbatim as it happened, anchored by a `#--slug` per
  entry. This is the most current and most detailed record — read it first for anything about
  *why* the code looks the way it does.
- **[`model_data/runs_registry.csv`](PPO_IBM/model_data/runs_registry.csv)** — the structured
  run ledger: one row per run (`v11`...) with its config, held-out numbers, and a compact
  narrative of what happened and why. Faster to scan than the decision history for "what has
  been tried."
- **[`finalresults.md`](PPO_IBM/finalresults.md)** (uncondensed version in
  git history, `8020ed1`) — covers the **PPO era only**, up to v31. Its "bottom line"
  (a behaviour-cloned controller beating every RL run) **predates TD3+BC and is now out of
  date** — see "Current status" below for the current best result.

## Current status: TD3+BC

TD3+BC is the first algorithm in this project to produce a **held-out-validated D2 policy**
(PPO and TD-MPC2 never did — see below). Entry points: `td3/TD3.py` (LSTM core) and
`td3/TD3_lru.py` (diagonal-LRU core, patches `TD3.py`'s module globals and reuses its
training loop). Run directly from `PPO_IBM/`:

```
python td3/TD3.py                       # LSTM core, fresh run
python td3/TD3.py --resume              # resume from model_data/td3_checkpoints/
python td3/TD3_lru.py                   # diagonal-LRU core, fresh run
python td3/TD3_lru.py --resume          # resume from model_data/td3_lru_checkpoints/

# held-out validation (required before any mastery claim — see below)
python experiments/bc_scaffold/scripts/td3_held_out_sweep.py \
    --actor-path model_data/td3_checkpoints_best/actor.pth --difficulty 2 --high-pop 12
```

Key env vars (all default to the historically-validated behavior; see `td3/TD3.py` for the
full list): `TD3_BC_COEF` (behaviour-cloning anchor strength), `TD3_DEMO_FRACTION`,
`TD3_HIDDEN_RESET_INTERVAL` (rollout/det-eval recurrent-state reset cadence, independent of the
training window `SEQ_LEN`), `TD3_THREADS`.

**Best result: v49 (LSTM core)** — held D2 mastery for 15 sustained chunks through the full
2,000,000-step budget, 0% crash throughout. 40-seed held-out sweep: harvest 135.8mg median /
83.9 p25 / 0% crash, 4/4 gate criteria pass (reset-matched sweep mode; see
`docs/decision_history.md` for the historical vs. corrected numbers). This remains the
project's standing best single result.

**The recurring high-population collapse, and what fixed it for the LRU.** Multiple LRU-core
runs (v48, v50, v52, v53) suffered a repeated failure mode: harvest yield at high starting
population (≥1500 cells) would collapse toward zero after several D2 chunks, even as low/mid
population stayed healthy. A sequence of hypotheses were tested and ruled out (a reward
dead-zone, an unconditional OD-delta sign, core-vs-reward interaction) before landing on the
actual cause: `HIDDEN_RESET_INTERVAL=60` — a correctness fix specific to the LSTM's *unbounded*
cell-state growth — was inherited by the LRU (whose state is bounded by construction) without
independent justification, and the resulting artificial 60-step cold start destabilized it at
high population. **v54** decoupled the rollout reset cadence from the training window
(`TD3_HIDDEN_RESET_INTERVAL=600` vs `SEQ_LEN=60`, no `O(T²)` training-cost blowup since rollout
calls the actor one step at a time) and completed the full 2M-step budget with **zero
collapse** — the first LRU run to do so. **v55** replicated this cleanly. Both are documented
in full, including two unrelated infrastructure incidents encountered live (a silent process
death, and laptop lid-close Modern Standby cycling), in `docs/decision_history.md`.

**Caveats on the LRU result:** the fix targets the high-population regime specifically — v54/
v55's low/mid-population held-out numbers sit slightly below v49's. A proactive critic-
divergence check mid-run found genuine (but so far behaviorally harmless) Q-overestimation at
high population; whether this would eventually matter given a longer budget is untested. The
LSTM baseline (v49) used its own natural `HIDDEN_RESET_INTERVAL=60`, not the LRU's 600, so this
is not yet a fully single-variable LSTM-vs-LRU comparison — a matched-reset-interval LSTM run
is a natural follow-up.

## Run everything from PPO_IBM/

Scripts hardcode relative paths (`model_data/...`), which resolve against the **working
directory**, not the file location — `cd PPO_IBM` before running anything. Modules in
subdirectories carry a small `sys.path` bootstrap so flat imports keep working, but the cwd
requirement stands.

Always prefix `PYTHONIOENCODING=utf-8` for direct invocations — the Windows console default
(cp1252) crashes on box-drawing characters in the log output.

**No mastery claim is final without an independent held-out sweep** — every algorithm family in
this project (PPO, TD-MPC2, TD3) has had at least one run pass its in-training gate and then
fail (or narrowly miss) held-out validation. For TD3, use
`experiments/bc_scaffold/scripts/td3_held_out_sweep.py`; for PPO, `scripts/validate.py` /
`diagnostics/held_out_sweep.py`; for TD-MPC2, `diagnostics/tdmpc2_held_out_sweep.py`.

## Layout

| folder | contents |
|---|---|
| `environments/` | `genetic_env.py` — the simulator (physics, reward, observation) |
| `legacy/` | `TD3.py` (LSTM-core TD3+BC trainer, current primary algorithm), `TD3_lru.py` + `lru_core.py` (diagonal-LRU core variant), `actor_io.py` (checkpoint core-detection/loading), `TD_MPC2.py` (parked second algorithm), `SAC_policy.py`/`recurrent_sac.py`/`Var_MPC.py` (unused one-offs) |
| `experiments/` | `bc_scaffold/` — `td3_held_out_sweep.py` and the BC-clone scaffold; `env_diagnosis/` — read-only diagnostics (`q_magnitude_check.py`, `growth_diagnosis.py`, `population_range_check.py`, …); `harvest_ablation/` |
| `training/` | PPO trainer + curriculum: `recurrent_ppo.py`, `curriculum_schedule.py` (also imported by TD3), `entropy_schedule.py`, `deterministic_eval.py`, `callbacks.py`, … |
| `bc/` | `bc_pretrain.py` — behaviour cloning from the scripted OD-feedback expert |
| `diagnostics/` | PPO/TD-MPC2 read-only probes: `held_out_sweep.py`, `tdmpc2_held_out_sweep.py`, `test_actions.py`, `reward_ab.py`, `noise_sensitivity.py`, sweeps |
| `scripts/` | PPO operational tooling (below) |
| `model_data/` | checkpoints (`td3_checkpoints*`, `td3_lru_checkpoints*`, PPO's `best_det_checkpoint/`), `archive_*/` per run, `runs_registry.csv` |
| `logs/` | training logs, per-run config snapshots, `validation/`, `scratch/` |
| `docs/` | `decision_history.md` (primary engineering log — see above), `known_limitations.md`, `calibration.md`, … |
| `artifacts/` | plots, generated documents |

## Tooling

**`td3/actor_io.py`** — loads a TD3 actor/critic checkpoint and detects its recurrent core
(LSTM vs LRU) from the parameter names, so the same downstream script (held-out sweep,
diagnostics) scores either without a flag.

**`experiments/bc_scaffold/scripts/td3_held_out_sweep.py`** — TD3's held-out validation: 40
seeds (lognormal(100,400) initial population + 10% adversarial cold starts, survival-scored),
plus an optional `--high-pop N` block (log-uniform 600–5000) probing the regime the main sweep
never samples and where policies have historically diverged most. Reset-interval matched to
training by default (`--free-run` reproduces the older, out-of-distribution free-running mode).

**`experiments/env_diagnosis/q_magnitude_check.py`** — distinguishes critic divergence
(twin-Q disagreement, Q overestimation vs. actual Monte-Carlo return) from genuine
environmental negative returns at a given population level. Run against a live or archived
checkpoint whenever a collapse (or a suspicious training-loss trend) needs diagnosing.

**`scripts/run_training.py`** (PPO) — launcher. Refuses to start if a trainer is already
running, moves the previous checkpoint dir aside (never deletes), pairs `norm`+`state` from an
explicit `--resume` directory, snapshots every outcome-relevant constant to
`logs/config_<tag>.json`, and confirms **exactly one** startup banner appears before declaring
success. Each guard exists because the corresponding failure happened: a dual-process launch
invalidated ~20h of v16, a bare `--resume` loaded an unrelated checkpoint at v14, and a `grep -c`
exit code once short-circuited a launch that never ran.

**`scripts/finish_run.py`** (PPO) — scores a run's best deterministic checkpoint against all
three tiers, prints the BC reference alongside, and writes the result into `runs_registry.csv`.

**`scripts/validate.py`** (PPO) — held-out sweeps at D1 and D2 plus action traces, in one
command. Also prints the harvest-fraction profile, because the *shape* has diagnosed every
failure mode here (never-harvest, decay-to-zero, over-harvest-early) where aggregate scores did
not.

**`diagnostics/tdmpc2_cost_probe.py`** — TD-MPC2's pre-flight cost check (correctness smoke
test + full wall-clock cost measurement projected against the configured step budget). Run
before trusting any new TD-MPC2 configuration's projected runtime — its first two cost
estimates for this file were wrong by ~20x and ~1.8x respectively.

## Historical: PPO and TD-MPC2 (parked)

Both were the project's primary tracks before TD3+BC. Neither ever produced a held-out-
validated D2 policy. Summary (full detail in `finalresults.md` and `docs/decision_history.md`):

- **PPO** reached D2 in training twice (v14, v17) and passed a stochastic-only gate once (v26);
  all three failed independent held-out validation, at large margins (v14: 0.4mg vs a 90mg
  gate; v26: ~44mg/~0.0025 od vs 90mg/0.011). The best PPO-family artifact overall was a pure
  behaviour-cloned controller with no RL applied (`model_data/BEST_bc_clone_D2_validated/`,
  109.4mg median / 63.8 p25, 0% crash) — RL fine-tuning was, at the time, net-destructive.
- **TD-MPC2** (`legacy/TD_MPC2.py`) was rewritten mid-project to a genuine spec (3D action
  space, 600-raw-step macro-timestep world model matching the harvest interval, 5-critic
  ensemble, two-hot regression, wired to the same project curriculum gate). v27 advanced
  D0→D1 over the full 8M-step budget, but its D0 claim missed held-out validation narrowly
  (`time_avg_od` 0.0036 vs a 0.004 gate, ~10% short) — same failure class as PPO's v14/v17/v26.
  Not developed further once TD3+BC's v45 became the project's first algorithm to clear the D2
  held-out gate outright.

## Known issue: observation-space versioning

The observation went 6 → 8 channels in the PPO era (Fix #18), then was **reverted to 6 as the
default** (`OBS_EXTENDED = False` on `GeneticPhotobioreactorEnv`; TD3's `OBS_DIM = 6` matches
this). The 8-channel extension remains available via `OBS_EXTENDED = True` but is opt-in. Any
change to this flag orphans checkpoints saved under the other setting, in either direction.
`scripts/validate.py` (PPO) detects and reports the mismatch explicitly. Any future observation
change orphans checkpoints the same way, so bump a version marker and note it here.

## Env configuration flags

`GeneticPhotobioreactorEnv` class attributes, so a configuration can be isolated rather than
bundled (a PPO run once changed three things at once and its regression could not be
attributed):

| flag | default | effect |
|---|---|---|
| `LIGHT_FOULING_COEF` | `0.0002` | light-path biofouling. **Inert at this value** — calibrated for lab OD600 (~1–10) while this sim's `od` is ~0.018, so it accumulates ~0.0003 against a 0.5 cap. A realistic value is ~0.075, but a probe showed the culture is light-saturated so it changes no behaviour. |
| `TURB_FOULING_COEF` | `0.0` | nephelometer window fouling (biases the reading high). Realistic, but works **against** the OD-reward fix it would otherwise improve. |
| `HARVEST_PUMP_ERROR` | `0.0` | ±fraction harvest delivery error. Forces closed-loop harvest control. |
| `USE_EPISODE_PHASE` | `True` | `True`: obs channel 7 = `step/max_steps` (only present when `OBS_EXTENDED=True`). Not sim-to-real transferable, and reveals when the `time_avg_od` scoring window opens — a gaming hazard held-out sweeps cannot detect by score alone (an action trace showed no gaming in practice). `False`: periodic harvest-cycle phase, transferable and non-gameable. Set `False` for anything intended for deployment. |
