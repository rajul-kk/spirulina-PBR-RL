# legacy/ — retired algorithm

Kept because its v27 results are a documented baseline and `diagnostics/tdmpc2_*.py` load it.
Not re-run since physics v2 (2026-09-26); expect it to need updates before it works again.
Earlier one-offs (SAC variants, `Var_MPC.py`, `ssm_core.py`) were deleted 2026-09-26; they are in
git history.

| file | what it was |
|---|---|
| `TD_MPC2.py` | Model-based TD-MPC2 with MPPI planning (v27). Parked once TD3+BC cleared D2. |

The active TD3 trainers moved to `td3/` on 2026-09-26.
