# Best RL-trained PPO checkpoints (preserved before the 93GB legacy prune)

Neither of these was ever given its own `archive_v*`/`best_det_checkpoint` dir; both only
existed as full-run intermediate snapshots inside `recurrent_checkpoints_pre_v22` /
`recurrent_checkpoints_pre_v24` (the `run_training.py` "move previous checkpoint dir aside,
never delete" guard). Extracted their final (8,060,000-step, end-of-8M-budget) checkpoint
before that bulk was deleted. See `docs/reports/finalresults.md` for full context.

No `vec_normalize.pkl` / `training_state.pkl` accompanies either snapshot -- these dirs held
policy weights only, no normalization stats. Usable to load/inspect the policy, not to
resume training or reproduce exact deterministic eval without re-deriving normalization.

## v21_D1pass_8060000_steps.zip

Fix #18 (8-channel obs). **The only PPO run to PASS D1 held-out validation.**

| metric | value | D1 gate |
|---|---|---|
| best-det harvest | 101.9mg | |
| p25 | 77.3mg | |
| median time_avg_od | **0.0094** | ≥ (passes) |
| verdict | **PASS** | |

## v23_bestyield_8060000_steps.zip

Replicate of v21's config. **Best raw yield of any PPO run** -- clears even the D2
harvest/p25 bars outright -- but fails D1/D2 on time_avg_od.

| metric | value | D2 gate |
|---|---|---|
| best-det harvest | **121.0mg** | ≥90 (passes) |
| p25 | **98.7mg** | ≥50 (passes) |
| median time_avg_od | 0.0066 | ≥0.011 (FAILS) |
| verdict | fail (od only) | |

`docs/reports/finalresults.md`'s own read: "Harvest yield is solved... `time_avg_od` is the entire
obstacle -- five runs [v20-v24] span 0.0035-0.0094... against D2's requirement of 0.011."
v21 and v23 (same architecture, different training-seed outcome) bracket that finding: one
clears od on D1, the other clears yield but not od -- illustrating the ~30% seed-driven
variance in this metric documented elsewhere in this project (`docs/reports/statistical_validation.md`).

## Standing recommendation, unchanged by this preservation

`docs/reports/finalresults.md` and `README.md` both name `model_data/BEST_bc_clone_D2_validated/` (a pure
behaviour-cloned controller, no RL) as the project's actual recommended PPO-track artifact:
109.4mg median / 63.8 p25 / time_avg_od 0.0191 / 0% crash, the only PPO-track policy to PASS
the full D2 held-out gate on every criterion at once. v21/v23 above are preserved as the
best *RL-trained* checkpoints for reference, not as a replacement recommendation.
