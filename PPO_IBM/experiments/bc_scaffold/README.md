# bc_scaffold — isolating the `time_avg_od` bottleneck

PPO v31 and TD-MPC2 v32 both stalled on `time_avg_od` (D0 never cleared 0.004; D1
reached then demoted back to D0). This tests whether the task is solvable at all by a
non-learned controller, before treating it as an environment-difficulty ceiling.

## What's in here

- `scripts/expert_sweep.py` — scripted proportional control law directly against
  `genetic_env.py`, no NN, no training, on the curriculum's own held-out cold-start
  distribution (90% lognormal(100,400), 10% adversarial 30–80).
- `scripts/run_bc_clone_and_eval.py` — clones the same law into the real
  `RecurrentPPO` policy (`bc/bc_pretrain.py`, supervised only, zero RL steps), then
  evaluates at D0/D1/D2 with `diagnostics/held_out_sweep.py`.
- `scripts/td3_held_out_sweep.py` — same held-out methodology for TD3 checkpoints.
- `results/` — CSVs/logs, one per tier.

## Control law

```
frac = clip(GAIN * (od / OD_SETPOINT - 1), 0, FRAC_CAP)
GAIN = 1.0   OD_SETPOINT = 0.015   FRAC_CAP = 0.30
stir, light near-constant (60-80 rpm, 900-1000 umol)
```
Below setpoint it harvests nothing; above it, removes roughly the surplus.

## Results (2026-08-19, n=40 held-out per tier)

### Scripted expert alone

| tier | median harvest [gate] | p25 [gate] | median time_avg_od [gate] | crash | verdict |
|---|---|---|---|---|---|
| D0 | 138.1 [>=30] | 79.1 [>=15] | 0.0184 [>=0.004] | 0% | PASS, 4.6x over gate |
| D1 | 101.5 [>=60] | 61.4 [>=30] | 0.0175 [>=0.008] | 0% | PASS, 2.2x over gate |
| D2 | 124.3 [>=90] | 73.9 [>=50] | 0.0179 [>=0.011] | 0% | PASS, 1.6x over gate |

### BC clone into the real policy network

Evaluated against the D2 gate at each environment-difficulty setting:

| env difficulty | median harvest | p25 | median time_avg_od | crash | verdict |
|---|---|---|---|---|---|
| D0 physics | 100.1 | 65.5 | 0.0195 | 0% | PASS |
| D1 physics | 100.2 | 54.7 | 0.0175 | 0% | PASS |
| D2 physics | 102.0 | 53.2 | 0.0177 | 0% | PASS |

The clone loses essentially nothing vs. the bare law. Reproduces and extends the
earlier v19 result (`model_data/BEST_bc_clone_D2_validated`: harvest 109.4mg, od
0.0191) to D0/D1 physics.

## Implication

Both the bare law and its NN clone clear D0/D1/D2 with zero RL — the `time_avg_od`
bottleneck is not an environment-difficulty ceiling, it's that PPO/TD-MPC2's
exploration never finds/holds this setpoint-tracking behavior. Options: (1) warm-start
from the clone again, fixing why it decayed post-handoff last time (v17: a
miscalibrated critic, already fixed for PPO but not tried for TD-MPC2); (2) constrain
early fine-tuning (small LR, tight clip range, KL-to-clone penalty) so gradients can't
unlearn the setpoint before mastery streak logic locks it in; (3) if warm-starting
keeps decaying regardless, that points at the reward/exploration side (novelty_report's
C2 finding — exploration noise masking a superior deterministic policy), not init.
