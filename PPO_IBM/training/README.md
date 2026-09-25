# training/ — PPO trainer and the shared curriculum

**Shared by every trainer (TD3 imports these):**

| file | role |
|---|---|
| `curriculum_schedule.py` | Difficulty gates (`ADVANCE_TARGETS`), mixing, demotion, the fixed det-eval set. |
| `curriculum_starts.py` | Initial-population sampling and stitched starts (snapshot/restore a grown culture). |
| `training_state.py` | Save/load of resumable training state. |

**RecurrentPPO pipeline (parked track):**

| file | role |
|---|---|
| `recurrent_ppo.py` | Trainer (SB3-contrib RecurrentPPO). Launch through `scripts/run_training.py`. |
| `env_factory.py`, `wrappers.py`, `env_utils.py` | VecEnv construction and wrappers. |
| `callbacks.py` | Episode metrics, stitching, progress bar. |
| `deterministic_eval.py` | Noise-free eval episodes for the det gate. |
| `entropy_schedule.py` | Entropy/std control. |
