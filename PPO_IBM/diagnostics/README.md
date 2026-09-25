# diagnostics/ — ad-hoc probes from the PPO and TD-MPC2 eras

Read-only scripts written to answer one question each. All predate physics v2 (2026-09-26), so
their thresholds and expected numbers are in the old ~4 mg/L scale. For TD3 and current-env
diagnostics use `experiments/env_diagnosis/`.

| file | purpose |
|---|---|
| `held_out_sweep.py` | Held-out robustness sweep for a RecurrentPPO checkpoint (also used by `scripts/validate.py`). |
| `test_actions.py` | Inspect/compare trained PPO action outputs. |
| `reward_ab.py`, `reward_breakdown.py` | Per-term reward comparison: policy vs scripted expert. |
| `noise_sensitivity.py` | How much action noise degrades each policy. |
| `curriculum_gate_sweep.py` | Validates `ADVANCE_TARGETS` thresholds. |
| `dynamic_profile_sweep.py`, `dynamic_profile_sweep_od.py` | Harvest-profile sweeps for the semi-continuous harvest. |
| `fouling_feasibility.py` | Whether real biofouling would make D2 unreachable. |
| `zombie_diagnosis.py` | The "zombie culture" failure mode. |
| `tdmpc2_cost_probe.py`, `tdmpc2_held_out_sweep.py` | TD-MPC2 cost check and held-out sweep (need `legacy/TD_MPC2.py`). |
| `test_heavy.py` | Smoke test for `environments/heavy_env.py`. |
| `test_why_die.py` | Early culture-death probe. |
| `check_training.py` | Reads a hard-coded PPO TensorBoard run. |
| `_verify_envs.py` | **Broken:** sends a 4-D action; the env has taken 3-D actions since the CO2 channel was removed. |
