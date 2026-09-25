# legacy/ — retired algorithms

Kept for reference and reproducibility; none is on the active track, and none has been re-run
since physics v2 (2026-09-26), so expect them to need updates before they work again.

| file | what it was |
|---|---|
| `TD_MPC2.py` | Model-based TD-MPC2 with MPPI planning (v27). Parked once TD3+BC cleared D2. |
| `Var_MPC.py` | An earlier TD-MPC2 implementation, before the rewrite in `TD_MPC2.py`. |
| `ssm_core.py` | Linear state-space sequence block; nothing imports it. |
| `SAC_policy.py`, `recurrent_sac.py` | Early SAC attempts. |

The active TD3 trainers moved to `td3/` on 2026-09-26.
