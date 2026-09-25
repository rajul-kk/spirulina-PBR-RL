# td3/ — primary algorithm (TD3+BC, recurrent)

| file | role |
|---|---|
| `TD3.py` | Trainer: LSTM actor/critic, demo-seeded replay, dual curriculum gate, det-eval. Holds all hyperparameters and env-var knobs. |
| `TD3_lru.py` | Swaps in the diagonal-LRU core by patching `TD3`'s module globals, then runs the same training loop. |
| `lru_core.py` | The diagonal linear recurrent unit. |
| `actor_io.py` | Loads an actor checkpoint and detects its core (LSTM vs LRU) from parameter names. |

Run from `PPO_IBM/`:

```
python td3/TD3.py [--resume]         # checkpoints: model_data/td3_checkpoints*/
python td3/TD3_lru.py [--resume]     # checkpoints: model_data/td3_lru_checkpoints*/
```

Env vars: `TD3_HIDDEN_RESET_INTERVAL` (rollout/det-eval hidden-state reset; default `SEQ_LEN`),
`TD3_BC_COEF`, `TD3_DEMO_FRACTION`, `TD3_STEPS`, `TD3_THREADS`.

Shared curriculum logic lives in `training/curriculum_schedule.py` and
`training/curriculum_starts.py`. Held-out validation:
`experiments/bc_scaffold/scripts/td3_held_out_sweep.py`.
