# env_diagnosis — actions / reward / environment sweep

## Why this exists

TD3+BC (v35) held D2 cleanly for ~2M steps (0% crash, harvest well over gate), then
diverged sharply after being resumed: critic loss jumped 7.8 → 543.6 within one chunk,
and deterministic crash rate climbed 0% → 23% → 33% → 43% → 50% → 100% over the next few
chunks (`runs_registry.csv`, `v35_td3bc`). Before proposing a fix, this runs a systematic
sweep across the three things that could explain it — the reward function, the
environment's crash dynamics, and the action space — using `diagnose.py`.

## Sweep 1 — Reward magnitude audit

**Finding: the crash/extinction penalty is a severe outlier.** `genetic_env.py`'s
extinction branch applies `reward -= 100.0`. Measured against 72,000 real per-step reward
samples from the validated scripted expert (D2, 10 episodes):

| | value |
|---|---|
| mean per-step reward | 0.1494 |
| best single-step reward observed | 0.6429 |
| worst non-crash single-step reward (p1) | 0.1314 |
| crash penalty | **-100.0** |
| crash penalty vs mean | **669x** |
| crash penalty vs best single step | **156x** |
| crash penalty vs worst non-crash step | **761x** |

TD3 trains with `GAMMA=0.9995` (effective bootstrap horizon ~2,000 steps). At that gamma,
36.8% of the crash penalty's magnitude (`gamma^2000`) is still present at the far edge of
the horizon — so a single crash episode entering the replay buffer doesn't just corrupt
the transitions right around it; it distorts the Q-target for any training window the
critic samples within roughly the next 2,000 steps of (bootstrapped) value. This is not a
new failure mode in this codebase — the same line in `genetic_env.py` carries a comment
noting the penalty was already reduced from -1000 to -100 specifically because
"occasional exploration-driven crashes were corrupting the LSTM's learned weights for
millions of steps to recover from" — documented for PPO. v35's critic-loss spike right
before its stochastic crash rate started climbing is consistent with the identical
mechanism now hitting TD3's Q-function instead of PPO's LSTM.

## Sweep 2 — Crash rate by initial-population bucket × difficulty

**Finding: my working hypothesis (large "high"-bucket cold starts triggering the initial
crash wave) was wrong — checked against data, not assumed.** Crash rate under the
scripted expert is 0% everywhere (as already established). Under **random** actions (a
rough proxy for what a perturbed/undertrained actor might do):

| bucket | D0 | D1 | D2 |
|---|---|---|---|
| low (100-400 cells) | 12% | 0% | 12% |
| mid (600-1500 cells) | 0% | 0% | 0% |
| high (2000-5000 cells) | 0% | 0% | 0% |

Small cold starts are mildly *more* fragile under bad actions, not large ones — a small
culture has less population buffer before crossing the `num_active < 10` extinction floor.
The "high" cold starts observed near v35's crash chunks in the log were most likely
coincidental, not causal. This matters because it rules out "a specific cold-start bucket
is a crash trap the curriculum keeps re-exposing the policy to" as the trigger — the more
likely story is a small number of ordinary bad-luck crashes (any bucket can produce one
under a sufficiently perturbed action) whose reward magnitude, not their frequency or
origin, is what did the damage once one landed in the buffer.

## Sweep 3 — Harvest-fraction crash boundary

**Finding: confirms the "washout cliff" bc_pretrain.py's comments reference, re-verified
against the current env version rather than assumed from an old comment.**

| harvest frac (constant all episode) | crash% | median harvest_mg | median time_avg_od |
|---|---|---|---|
| 0.00 | 0% | 0.0 | 0.0583 |
| 0.05 | 0% | 89.6 | 0.0412 |
| 0.10 | 0% | 115.5 | 0.0246 |
| 0.15 | 0% | 141.2 | 0.0164 |
| 0.20 | 0% | 110.5 | 0.0077 |
| 0.25 | 0% | 101.4 | 0.0048 |
| 0.30 | 0% | 102.8 | 0.0032 |
| 0.35 | 0% | 94.9 | 0.0019 |
| 0.40 | 0% | 71.6 | 0.0007 |
| **0.45** | **80%** | 65.9 | 0.0004 |
| **0.50 (=F_MAX)** | **100%** | 64.5 | 0.0004 |

The cliff sits between 0.40 and 0.45 — sharp, not gradual. Harvest yield peaks around
frac≈0.15 and *declines* at higher fractions even before the cliff (141.2mg at 0.15 down
to 71.6mg at 0.40), while `time_avg_od` degrades monotonically the entire way — both
metrics agree the safe, productive zone is well under the cliff, consistent with
`EXPERT_FRAC_CAP=0.30`'s margin and the proportional law's typical operating range
(0.05–0.20).

## Conclusion — what actually explains v35's collapse

The reward-magnitude audit is the strongest lead: a -100 outlier at 156-761x normal
per-step reward, combined with a 2,000-step effective bootstrap horizon, is a textbook
setup for a Q-function to develop large, unstable targets the moment a crash transition
enters the buffer — independent of which cold-start bucket produced it. `td3_update`'s
critic loss (`legacy/TD3.py`) currently uses plain MSE, which is exactly the loss most
sensitive to this kind of outlier (squared error on a target 100+ units off amplifies the
gradient contribution of that one transition relative to everything else in the batch).

**Recommended fix:** switch the critic loss from MSE to Huber/smooth-L1, which behaves
identically to MSE for small (normal) TD-errors but grows only linearly, not
quadratically, past a threshold — capping how much a single crash-outlier transition can
dominate a gradient step, without touching the reward function itself (which is tuned,
documented infrastructure this project has stability-tested extensively for PPO/TD-MPC2
already, and changing it would break comparability with every other run in
`finalresults.md`).
