# harvest_ablation — does removing harvest control simplify the problem, or the interesting part?

## The question

Every RL run in this project has struggled specifically with the harvest dimension
(`finalresults.md`: stir/light learned correctly every run, harvest failed every run —
the only 1-in-600-step-credit dimension). Fixing harvest at a good constant tests whether
that's the whole bottleneck, or whether the environment has other independent sources of
difficulty (biofouling, actuator noise, day/night cycling).

## What this does

`recurrent_ppo_harvest_fixed.py` is PPO's exact training loop
(`training/recurrent_ppo.py`, copied not edited), pointed at the environment with harvest
overridden at a fixed constant (`HarvestFixedWrapper` in `training/wrappers.py`) before
it reaches `genetic_env.py`. The policy network still outputs a 3rd action value, it's
just discarded — no architecture/hyperparameter change.

**Fixed value: frac=0.15** (raw -0.4) — this project's own physics sweep and
`experiments/env_diagnosis/`'s harvest-fraction sweep both found it the best sustainable
constant (median harvest 141.2mg, time_avg_od 0.0164, 0% crash at D2).

## Isolation from the main pipeline

Separate paths, no collision with shared/tracked artifacts:
`model_data/harvest_fixed_ppo/`, `ppo_harvest_fixed_tensorboard/`,
`logs/training_run_v37_ppo_harvest_fixed_ablation.log`.

Two files are forked, not edited in place, since they construct the env directly and
are shared by every other run: `env_factory_harvest_fixed.py` (mirrors
`training/env_factory.py`) and `deterministic_eval_harvest_fixed.py` (mirrors
`training/deterministic_eval.py` — the dual gate's det-eval side builds its own env
independently of `env_factory.py`, so it needed the same wrapper).

## Reading the result

If PPO clears D2's held-out gate here (no PPO run ever has), harvest was the whole
bottleneck. If it still can't hold `time_avg_od`/crash rate cleanly, the environment has
real difficulty independent of harvest — D1/D2's actuator noise/biofouling/day-night
cycling would be next to isolate.
