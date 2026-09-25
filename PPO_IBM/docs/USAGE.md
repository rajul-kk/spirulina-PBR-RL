# Usage guide

Run everything from `PPO_IBM/`: scripts resolve `model_data/...` and `logs/...` against the
working directory. On Windows, prefix direct invocations with `PYTHONIOENCODING=utf-8` (the
console default cp1252 crashes on the box-drawing characters in log output).

---

## 1. TD3+BC (primary)

```
python td3/TD3.py                  # LSTM core, fresh run
python td3/TD3.py --resume         # resume from model_data/td3_checkpoints/
python td3/TD3_lru.py              # diagonal-LRU core, fresh run
python td3/TD3_lru.py --resume     # resume from model_data/td3_lru_checkpoints/
```

A fresh run starts from scratch only if `model_data/td3_training_state.pkl` (or
`td3_lru_training_state.pkl`) is absent; move the previous run's state and checkpoint dirs into
`model_data/archive_<tag>/` first.

| env var | default | effect |
|---|---|---|
| `TD3_HIDDEN_RESET_INTERVAL` | `SEQ_LEN` (60) | rollout/det-eval hidden-state reset cadence; current runs use 600 |
| `TD3_THREADS` | LRU: 6, LSTM: all cores | torch CPU threads |
| `TD3_BC_COEF` | 1.0 | behaviour-cloning anchor strength |
| `TD3_DEMO_FRACTION` | 0.25 | share of each batch drawn from the expert demo buffer |
| `TD3_STEPS` | 2,000,000 | step budget |

Record every run in `model_data/runs_registry.csv`.

## 2. Validation and checks

```
# held-out validation, required before any mastery claim
python experiments/bc_scaffold/scripts/td3_held_out_sweep.py \
    --actor-path model_data/td3_checkpoints_best/actor.pth --difficulty 2 --high-pop 12

# env/trainer regression suite: run after any change to the env, curriculum or TD3
python experiments/env_diagnosis/core_audit_check.py
```

Other current-env diagnostics are in `experiments/env_diagnosis/` (see its README).

## 3. RecurrentPPO (parked)

Launch through the guarded launcher rather than the trainer directly:

```
python scripts/run_training.py --tag <tag> [--resume <dir>] [--archive-prev <name>] [--note "..."] [--dry-run]
python scripts/finish_run.py --tag <tag> --result "<one-line verdict>"
python scripts/validate.py --model <checkpoint-without-.zip> [--n 40]
```

The trainer itself is `training/recurrent_ppo.py` (`--resume [dir]`, `--finetune --steps N`,
`--reset-training`). TensorBoard: `tensorboard --logdir ppo_recurrent_tensorboard`.

## 4. TD-MPC2 (retired)

```
python legacy/TD_MPC2.py [--resume] [--finetune [N]] [--priv-distill] [--steps N]
```

Not re-run since physics v2; see `legacy/README.md`.

## 5. Curriculum

All trainers share `training/curriculum_schedule.py`. D0 → D1 → D2 raise sensor noise, drift and
lag, actuator error and the physics scaling. Advancing requires **both** gates for 2
consecutive chunks: the stochastic gate on training episodes and the deterministic gate on a
fixed 9-episode evaluation set. Thresholds (`ADVANCE_TARGETS`, physics v2 scale):

| to leave | median harvest | p25 harvest | median time-avg od | max crash rate |
|---|---|---|---|---|
| D0 | 2060 mg | 1310 mg | 0.16 | 15% |
| D1 | 4030 mg | 2620 mg | 0.32 | 10% |
| D2 (mastery) | 5410 mg | 4780 mg | 0.45 | 8% |

Demotion: training crash rate ≥ 35% for 2 chunks, or the deterministic gate failing 12
consecutive chunks. At D0 the latter aborts the run.

## 6. Stale bytecode

After pulling or editing, clear cached bytecode if behaviour looks older than the source:

```powershell
Get-ChildItem -Path . -Filter __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
```
