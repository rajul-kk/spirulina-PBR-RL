# harvest_ablation — does removing harvest control simplify the problem, or the interesting part?

## The question

Every RL run in this project (PPO v4 through v31, TD-MPC2, TD3) has struggled
specifically with the harvest dimension — `finalresults.md`'s own analysis: "stir and
light... learned correctly EVERY time... Harvest, the only dimension with 1-in-600
credit, failed EVERY time." Would removing harvest control (fixing it at a good constant
instead of letting the agent choose it) let PPO cleanly solve the rest of the problem —
or does the environment have enough other sources of difficulty (biofouling, actuator
noise, pH drift, day/night cycling at D1/D2) that harvest was never the sole bottleneck?

## What this does

`recurrent_ppo_harvest_fixed.py` is PPO's exact, unmodified training loop
(`training/recurrent_ppo.py`, copied not edited) pointed at a version of the environment
where the harvest action dimension is overridden with a fixed constant
(`HarvestFixedWrapper`, added to `training/wrappers.py`) before it ever reaches
`genetic_env.py` — the policy network still outputs a 3rd action value, it's just
discarded, so nothing about PPO's architecture or hyperparameters changes, only what the
harvest dimension's output *does*.

**Fixed value: frac=0.15** (raw action -0.4). Not a guess — this project's own physics
sweep already found it the best sustainable constant fraction (`bc/bc_pretrain.py`:
147.9mg/144h, 0% crash), and `experiments/env_diagnosis/`'s harvest-fraction sweep
independently reconfirmed it against the current env version (median harvest 141.2mg,
`time_avg_od` 0.0164, 0% crash at D2). With harvest fixed there, the harvest/p25
curriculum-gate criteria should clear on physics alone — what remains to test is whether
PPO can now cleanly hold `time_avg_od` and crash rate under real stir/light control,
which is the part of the task that was never diagnosed as the actual bottleneck.

## Isolation from the main pipeline

Runs entirely under separate paths so it cannot collide with or overwrite any
shared/tracked artifact:
- `model_data/harvest_fixed_ppo/` (checkpoints, state, norm stats, best-det checkpoint)
- `ppo_harvest_fixed_tensorboard/`
- `logs/training_run_v37_ppo_harvest_fixed_ablation.log`

Two files are forked (not edited in place) because they construct the env directly and
are shared by every other run in the project:
- `env_factory_harvest_fixed.py` — mirrors `training/env_factory.py`, adds
  `HarvestFixedWrapper` innermost.
- `deterministic_eval_harvest_fixed.py` — mirrors `training/deterministic_eval.py`,
  same reason: the dual gate's deterministic side builds its own env independently of
  `env_factory.py`, so it needed the same wrapper or det-eval would score the agent's
  discarded harvest output against a fully-controllable dimension — a mismatch with what
  was actually trained.

## Reading the result

If PPO reaches D2 and clears the held-out gate here (something no PPO run has ever done,
`finalresults.md`), that's strong evidence harvest genuinely was the whole bottleneck —
the "chaoticness" was concentrated in exactly the one dimension with sparse,
1-in-600-step credit assignment, not spread across the environment generally.

If PPO *still* can't hold `time_avg_od`/crash rate cleanly even with harvest fixed at a
proven-good constant, that's evidence the environment has real difficulty independent of
harvest — D1/D2's actuator noise, biofouling, and day/night cycling would be the next
things to isolate.

Either way this is informative, not a null result to avoid — it directly answers the
question asked, not just confirms a hypothesis.
