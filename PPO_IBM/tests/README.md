# Tests

Run from `PPO_IBM/` (`pytest.ini` lives there): `python -m pytest`. Everything is CPU-only and
offline. The default run excludes `@pytest.mark.slow`; `-m slow` runs only those, `-m ""` runs all.

Install: CPU torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then
`pip install -r requirements-dev.txt`.

| file | what it pins |
|---|---|
| `test_stats.py` | Holm, exact permutation (5 v 5 separated = 2/252), Fisher vs scipy, hierarchical-bootstrap CI on synthetic shifts, `td3_secondary.parse_run` on synthetic logs (`[RESUME]` rewinds, ADVANCED/DEMOTED, abort, complete) |
| `test_golden.py` | re-runs `program_control` / `pcgym_control` `compare.py` and `td3_secondary.py` on the committed jsons/logs and diffs against the committed `.txt` (child process; writes redirected to `tmp_path`). Fast layer masks the bootstrap CI / `p_holm` fields (bootstrap size cut to 300); slow layer is byte-exact |
| `test_splits.py` | search / pilot / test / final / SAC-training seed ranges are disjoint; final = 160 + 40 (photobioreactor) and 200 (PC-Gym); every committed final-split json ran on exactly that split |
| `test_env_invariants.py` | simulator regressions: cell positions stay in the reactor and spread out, finite obs/reward, fixed obs shape, one fresh-medium definition for reset and harvest refill, stitched starts keep `cells_acclimation`, seeded prefix determinism |
| `test_td3_cores.py` | `detect_core` / `load_actor` round trip for lstm, lru, gru, rtu; hidden-reset cadence (`TD3_HIDDEN_RESET_INTERVAL`); LRU/RTU state bounds; chunked == whole-sequence; `controllers/td3_actor.py` honours `reset` |
| `test_pcgym_task.py` | CSTR cost = mean(((Ca - sp)/0.01)^2), runaway at T > 335 K, scenario generator, failing-controller handling, golden vs committed reference PID |
| `test_harness_smoke.py` | program harness scores trivial/broken controllers on short episodes; slow: one full-length episode vs committed reference |
| `test_summarize_log.py` | `experiments/program_control/kaggle/summarize_log.py` (train.log -> train_summary.log) |

Golden tests compare against files that are committed under `results/`; they never write there.
`results/rl_v3/**/train.log` is git-ignored, so the check against committed summaries skips on a
fresh clone.
