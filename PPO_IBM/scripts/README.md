# scripts/ — PPO run operations

| file | what it does |
|---|---|
| `run_training.py` | Safe launcher for a PPO run: refuses a second trainer, sets the previous checkpoint aside, snapshots constants to `logs/config_<tag>.json`, confirms exactly one startup banner. |
| `finish_run.py` | Scores a finished run's best det checkpoint on all tiers and writes it to `model_data/runs_registry.csv`. |
| `validate.py` | Held-out sweeps at D1/D2 plus the harvest-fraction profile, in one command. |

TD3 runs are launched directly (`python td3/TD3.py`, see `td3/README.md`).
