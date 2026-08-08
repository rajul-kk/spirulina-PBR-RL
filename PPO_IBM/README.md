# PPO_IBM — Spirulina photobioreactor control

RecurrentPPO (SB3-contrib) on `GeneticPhotobioreactorEnv`, with a difficulty curriculum
(D0 → D1 → D2) gated on both stochastic rollouts and deterministic evaluation.

**Results and history: [`finalresults.md`](finalresults.md)** (full detail in
`finalresults_full_archive.md`). Read that before changing anything — it records 24 runs, the
fixes that worked, and several that were measured and refuted.

## Run everything from the repo root

Scripts hardcode relative paths (`model_data/...`), which resolve against the **working
directory**, not the file location. Modules in subdirectories carry a small `sys.path`
bootstrap so flat imports keep working, but the cwd requirement stands.

```
python scripts/run_training.py --tag v25_my_change --note "what this tests"
python scripts/finish_run.py  --tag v25_my_change
python scripts/validate.py    --model model_data/best_det_checkpoint/recurrent_ppo_genetic_ibm
```

Always prefix `PYTHONIOENCODING=utf-8` for direct invocations — the Windows console default
(cp1252) crashes on box-drawing characters in the log output.

## Layout

| folder | contents |
|---|---|
| `environments/` | `genetic_env.py` — the simulator (physics, reward, observation) |
| `training/` | trainer + curriculum: `recurrent_ppo.py`, `curriculum_schedule.py`, `entropy_schedule.py`, `deterministic_eval.py`, `callbacks.py`, … |
| `bc/` | `bc_pretrain.py` — behaviour cloning from the scripted OD-feedback expert |
| `diagnostics/` | read-only probes: `held_out_sweep.py`, `test_actions.py`, `reward_ab.py`, `noise_sensitivity.py`, sweeps |
| `scripts/` | operational tooling (below) |
| `model_data/` | checkpoints, `archive_*/` per run, `best_det_checkpoint/`, `runs_registry.csv` |
| `logs/` | training logs, per-run config snapshots, `validation/`, `scratch/` |
| `docs/` | supporting notes |
| `artifacts/` | plots, generated documents |
| `legacy/` | unused alternative algorithms (SAC, TD-MPC2) and one-off utilities |

## Tooling

**`scripts/run_training.py`** — launcher. Refuses to start if a trainer is already running,
moves the previous checkpoint dir aside (never deletes), pairs `norm`+`state` from an explicit
`--resume` directory, snapshots every outcome-relevant constant to `logs/config_<tag>.json`,
and confirms **exactly one** startup banner appears before declaring success. Each guard exists
because the corresponding failure happened: a dual-process launch invalidated ~20h of v16, a
bare `--resume` loaded an unrelated checkpoint at v14, and a `grep -c` exit code once
short-circuited a launch that never ran.

**`scripts/finish_run.py`** — scores a run's best deterministic checkpoint against all three
tiers, prints the BC reference alongside, and writes the result into `runs_registry.csv`.
Exists because comparing runs meant grepping five multi-MB logs by hand, which is how v21's
`od 0.0094` was briefly mistaken for a reproducible level rather than the top of a
0.0054–0.0094 spread.

**`scripts/validate.py`** — held-out sweeps at D1 and D2 plus action traces, in one command.
**No mastery claim is final without this**: v14 and v17 both passed the in-training gates and
then failed held-out validation. It also prints the harvest-fraction profile, because the
*shape* has diagnosed every failure mode here (never-harvest, decay-to-zero, over-harvest-early)
where aggregate scores did not.

## Known issue: observation-space versioning

The observation went 6 → 8 channels (Fix #18). **Every checkpoint saved before that — including
`model_data/BEST_bc_clone_D2_validated/`, the best artefact this project produced — cannot be
loaded against the current env.** `validate.py` detects and reports this explicitly. To use
those checkpoints, either regenerate them under the current env
(`python bc/bc_pretrain.py` takes ~15 min) or check out the matching env revision. Any future
observation change orphans checkpoints the same way, so bump a version marker and note it here.

## Env configuration flags

`GeneticPhotobioreactorEnv` class attributes, so a configuration can be isolated rather than
bundled (v22 changed three things at once and its regression could not be attributed):

| flag | default | effect |
|---|---|---|
| `LIGHT_FOULING_COEF` | `0.0002` | light-path biofouling. **Inert at this value** — calibrated for lab OD600 (~1–10) while this sim's `od` is ~0.018, so it accumulates ~0.0003 against a 0.5 cap. A realistic value is ~0.075, but a probe showed the culture is light-saturated so it changes no behaviour. |
| `TURB_FOULING_COEF` | `0.0` | nephelometer window fouling (biases the reading high). Realistic, but works **against** Fix #18, which improved that same signal. |
| `HARVEST_PUMP_ERROR` | `0.0` | ±fraction harvest delivery error. Forces closed-loop harvest control. |
| `USE_EPISODE_PHASE` | `True` | `True`: obs channel 7 = `step/max_steps`. Not sim-to-real transferable, and reveals when the `time_avg_od` scoring window opens at step 3600 — a gaming hazard `held_out_sweep.py` cannot detect (an action trace of v21 showed no gaming). `False`: periodic harvest-cycle phase, transferable and non-gameable. Set `False` for anything intended for deployment. |
