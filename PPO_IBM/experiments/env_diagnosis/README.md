# env_diagnosis — actions / reward / environment sweep

TD3+BC (v35) held D2 cleanly for ~2M steps then diverged after resuming: critic loss
jumped 7.8 → 543.6 in one chunk, crash rate climbed 0% → 100% over the next few
(`runs_registry.csv`, `v35_td3bc`). `diagnose.py` sweeps the reward function, the
environment's crash dynamics, and the action space to find why.

## Sweep 1 — Reward magnitude audit

`genetic_env.py`'s crash/extinction penalty is `-100.0`. Measured against 72,000
per-step reward samples from the scripted expert (D2, 10 episodes):

| | value |
|---|---|
| mean per-step reward | 0.1494 |
| best single-step reward | 0.6429 |
| worst non-crash step (p1) | 0.1314 |
| crash penalty vs mean | **669x** |
| crash penalty vs best step | **156x** |
| crash penalty vs worst non-crash step | **761x** |

TD3's `GAMMA=0.9995` gives an effective bootstrap horizon of ~2,000 steps; 36.8%
(`gamma^2000`) of the crash penalty's magnitude survives that far, so one crash episode
in the replay buffer distorts Q-targets for a wide window of training data. Same failure
mode `genetic_env.py`'s own comment already documents for PPO (penalty reduced from
-1000 to -100 for this reason) — v35's critic-loss spike is consistent with it hitting
TD3's Q-function instead.

## Sweep 2 — Crash rate by initial-population bucket × difficulty

Working hypothesis (large cold starts triggering the crash wave) was wrong. Under random
actions:

| bucket | D0 | D1 | D2 |
|---|---|---|---|
| low (100-400) | 12% | 0% | 12% |
| mid (600-1500) | 0% | 0% | 0% |
| high (2000-5000) | 0% | 0% | 0% |

Small cold starts are mildly *more* fragile (less population buffer above the
`num_active < 10` extinction floor), not large ones. The "high" cold starts seen near
v35's crash chunks were coincidental — any bucket can produce a bad-luck crash under a
perturbed action; it's the crash's reward magnitude that did the damage, not its origin.

## Sweep 3 — Harvest-fraction crash boundary

| frac (constant) | crash% | median harvest_mg | median time_avg_od |
|---|---|---|---|
| 0.00-0.40 | 0% | 0-141.2 (peaks at 0.15) | 0.0583→0.0007 |
| 0.45 | 80% | 65.9 | 0.0004 |
| 0.50 (=F_MAX) | 100% | 64.5 | 0.0004 |

Confirms the washout cliff bc_pretrain.py references, re-verified against the current
env: sharp, between 0.40 and 0.45, not gradual.

## Conclusion

The reward-magnitude audit is the strongest lead: a -100 outlier at 156-761x normal
per-step reward, plus a 2,000-step bootstrap horizon, is a textbook setup for Q-function
instability once a crash transition enters the buffer. `td3_update`'s critic loss used
plain MSE, which amplifies that outlier's gradient contribution quadratically.

**Fix applied (v36):** critic loss MSE → Huber — identical for normal TD-errors, caps
outlier-transition gradient contribution to linear. Left the reward function itself
untouched (tuned, shared infrastructure other runs depend on for comparability).
