# PPO_IBM — Spirulina photobioreactor control

RecurrentPPO (SB3-contrib) on `GeneticPhotobioreactorEnv`, with a difficulty curriculum
(D0 → D1 → D2) gated on both stochastic rollouts and deterministic evaluation. TD-MPC2
(model-based, MPPI planning) is a second algorithm under active development in `legacy/`,
upgraded this session to a genuine TD-MPC2 spec (Q-ensemble, two-hot regression,
macro-timestep world model) and wired to the same project gate.

All code, scripts, and supporting docs referenced below live in [`PPO_IBM/`](PPO_IBM/), the
project directory this README describes.

**Results and history: [`finalresults.md`](PPO_IBM/finalresults.md)** (full detail in
`finalresults_full_archive.md`). Read that before changing anything — it records 24 PPO runs
through v24, the fixes that worked, and several that were measured and refuted. **v25, v26,
and the TD-MPC2 upgrade postdate that writeup and are summarized below** until they're folded
in.

## Run everything from PPO_IBM/

Scripts hardcode relative paths (`model_data/...`), which resolve against the **working
directory**, not the file location — `cd PPO_IBM` before running anything below. Modules in
subdirectories carry a small `sys.path` bootstrap so flat imports keep working, but the cwd
requirement stands.

```
python scripts/run_training.py --tag v25_my_change --note "what this tests"
python scripts/finish_run.py  --tag v25_my_change
python scripts/validate.py    --model model_data/best_det_checkpoint/recurrent_ppo_genetic_ibm
```

Always prefix `PYTHONIOENCODING=utf-8` for direct invocations — the Windows console default
(cp1252) crashes on box-drawing characters in the log output.

## Layout

| folder | contents |
|---|---|
| `environments/` | `genetic_env.py` — the simulator (physics, reward, observation) |
| `training/` | trainer + curriculum: `recurrent_ppo.py`, `curriculum_schedule.py`, `entropy_schedule.py`, `deterministic_eval.py`, `callbacks.py`, … |
| `bc/` | `bc_pretrain.py` — behaviour cloning from the scripted OD-feedback expert |
| `diagnostics/` | read-only probes: `held_out_sweep.py`, `test_actions.py`, `reward_ab.py`, `noise_sensitivity.py`, sweeps |
| `scripts/` | operational tooling (below) |
| `model_data/` | checkpoints, `archive_*/` per run, `best_det_checkpoint/`, `runs_registry.csv` |
| `logs/` | training logs, per-run config snapshots, `validation/`, `scratch/` |
| `docs/` | supporting notes |
| `artifacts/` | plots, generated documents |
| `legacy/` | `TD_MPC2.py` — actively developed second algorithm (see below); SAC and other one-off utilities, unused |

## Tooling

**`scripts/run_training.py`** — launcher. Refuses to start if a trainer is already running,
moves the previous checkpoint dir aside (never deletes), pairs `norm`+`state` from an explicit
`--resume` directory, snapshots every outcome-relevant constant to `logs/config_<tag>.json`,
and confirms **exactly one** startup banner appears before declaring success. Each guard exists
because the corresponding failure happened: a dual-process launch invalidated ~20h of v16, a
bare `--resume` loaded an unrelated checkpoint at v14, and a `grep -c` exit code once
short-circuited a launch that never ran.

**`scripts/finish_run.py`** — scores a run's best deterministic checkpoint against all three
tiers, prints the BC reference alongside, and writes the result into `runs_registry.csv`.
Exists because comparing runs meant grepping five multi-MB logs by hand, which is how v21's
`od 0.0094` was briefly mistaken for a reproducible level rather than the top of a
0.0054–0.0094 spread.

**`scripts/validate.py`** — held-out sweeps at D1 and D2 plus action traces, in one command.
**No mastery claim is final without this**: v14 and v17 both passed the in-training gates and
then failed held-out validation. It also prints the harvest-fraction profile, because the
*shape* has diagnosed every failure mode here (never-harvest, decay-to-zero, over-harvest-early)
where aggregate scores did not.

**`diagnostics/tdmpc2_cost_probe.py`** — TD-MPC2's equivalent of a pre-flight check: a
correctness smoke test (TwoHotEncoder round-trip, one live `plan()`+`update()` call checked for
NaN/shape) followed by a full four-component wall-clock cost measurement (`plan()`, `update()`,
`env.step()`, LMU compressor) projected against the configured step budget. Exists because the
first two cost estimates for this file were wrong by ~20x and ~1.8x respectively — see the
TD-MPC2 section below. Run before trusting any new TD-MPC2 configuration's projected runtime.

**`diagnostics/tdmpc2_held_out_sweep.py`** — TD-MPC2's equivalent of `held_out_sweep.py`
(`agent.plan()` has no SB3 `model.predict()`-compatible interface, so it's a parallel
implementation, not a shared one). Same rule: no mastery claim is final without this, run on
fresh seeds disjoint from anything used to gate training.

## PPO: gate-mode experiment (v25 aborted, v26 confirmed false positive)

`GATE_MODE` (env var, `recurrent_ppo.py`) selects `dual` (default — stochastic AND
deterministic must both pass, the long-standing gate) or `stochastic` (stochastic-only, for a
self-consistent experiment: `GATE_MODE=stochastic python training/recurrent_ppo.py` paired with
`held_out_sweep.py --stochastic` to validate on the same policy mode it was gated on).

v26 (`GATE_MODE=stochastic`) declared D2 mastery on its stochastic rollouts at step 6.9M. Held-
out validation — 40 seeds, both deterministic **and** `--stochastic` modes, matching the mode it
trained under — gave harvest ~44mg / od ~0.0025 against a gate requiring harvest≥90mg /
od≥0.011, in **both** modes: roughly half and a quarter of the required thresholds. Same failure
class as v14 and v17 (in-training pass, held-out fail), now demonstrated to persist even when
the held-out check uses the gate's own mode. **This does not reopen the dual-gate question** —
it confirms gating on stochastic rollouts alone, even self-consistently validated, is not
sufficient; see `finalresults.md`'s "THE KEY FINDING" section for the underlying mechanism.

## TD-MPC2 upgrade (Fix #27) — genuine spec, ran to completion, D0 held-out miss

`legacy/TD_MPC2.py` was rewritten this session from a broken, pre-redesign-env implementation
(4D action space against a 3D env, 24-raw-step / 0.48h planning horizon that could not see a
600-step harvest event, its own disconnected curriculum logic, twin-Q, MSE reward/value) to a
genuine TD-MPC2 spec wired to the same project curriculum gate PPO uses:

- **3D action space** (`stir, light, harvest`), matching `genetic_env.py`.
- **Macro-timestep world model**: `MACRO_STEPS=50` — dynamics/reward trained on 50-raw-step
  blocks (action held constant, discounted reward summed) instead of single raw steps.
  `PLANNING_HORIZON=12` macro-steps × 50 = 600 raw steps of lookahead = exactly one
  `HARVEST_INTERVAL_STEPS`, the event the old horizon was structurally blind to.
- **5-critic ensemble** (`nn.ModuleList` of `ValueNetwork`s, random subset-of-2 drawn per call)
  replacing the twin-Q pair.
- **Two-hot regression** (101 bins, symlog-scaled) for reward/value in place of MSE.
- **Project dual gate**: imports `ADVANCE_TARGETS`/`MASTERY_MIN_EPISODES` from
  `curriculum_schedule.py` directly (no more local, disconnected copy); a new
  `run_tdmpc2_eval_episode()` provides the deterministic side (`agent.plan()` has no SB3
  `model.predict()`-compatible interface, so `deterministic_eval.py` could not be reused
  directly, only mirrored).
- `finetune_td_mpc2()` was **not** updated to the new architecture and raises
  `NotImplementedError` at entry rather than failing confusingly on a removed `q1`/`q2`.

**Cost measurement, corrected twice.** This project has a standing rule against trusting an
estimate over a direct measurement, applied here after two misses in a row: an initial estimate
of ~1017h (not having read `ACTION_REPEAT` or the file's own step budget) corrected to ~46.65h
after reading the full file; then, after launch, the observed throughput implied ~30h against a
claimed 12.98h — `tdmpc2_cost_probe.py`'s first version had only measured `plan()`/`update()`
and missed `env.step()` and the per-step LMU compressor entirely, **and** the file's
`max_cells=300_000` (vs `7_500` used everywhere else in the project) cost 13.7ms/step vs
4.6ms/step for an *identical* actual population (~2,990 cells either way — the cap was never
approached). Fixed both; final measured cost is **23.38h for 8,000,000 steps**, ~1.4x a PPO run.

**Two diagnostic findings from this session's monitoring, now fixed in code:**
1. `run_tdmpc2_eval_episode` hardcoded `initial_cells=3000` for every deterministic eval
   episode, vs the stochastic side's curriculum-sampled 100-1400 range at D0. A **no-op
   policy** (never harvest, neutral stir/light) at `initial_cells=3000` alone produces
   `time_avg_od=0.217` against D0's `0.004` gate — the det side's comfortable pass was
   substantially an artifact of its fixed head start, not evidence of policy competence. Fixed:
   now samples `initial_cells` via `_sample_init_cells`, matched to training.
2. Checkpointing fired every 2,000 raw steps, each save pickling the **full 25,000-transition
   replay buffer** on top of network weights (~7MB/save; ~15GB and 2,000+ files by ~55% through
   an 8M-step run) — a real, disproportionate I/O tax unique to this training regime (PPO's
   `CheckpointCallback` saves weights only, every 10,000 steps). Widened to every 50,000 steps.

Both fixes were applied on disk after v27 had already launched (editing the file does not
affect an already-running process), so v27 itself ran to completion on the **pre-fix** code —
they take effect on the next launch, not this one.

**v27's result**: D0's stochastic `time_avg_od` plateaued at 0.0013-0.0019 for ~18 chunks
before crossing its 0.004 gate, advancing to D1 at step 4.8M. D1 held for the rest of the
8,000,000-step budget — harvest/p25/crash cleared immediately, `time_avg_od` oscillated
0.0051-0.0071 against a 0.008 target, closest on the very last chunk (0.0077) before the budget
ran out. **Held-out validation of the D0 claim** (`diagnostics/tdmpc2_held_out_sweep.py`, 40
fresh seeds, using the fixed `initial_cells` sampling so this check isn't itself subject to
finding #1): harvest and crash clear comfortably, but `time_avg_od` median is **0.0036 against
the 0.004 gate in both deterministic and stochastic mode** — consistent, ~10% short. **The D0
claim does not independently replicate.** Same failure class as v14/v17/v26, with the tightest
margin of any RL run in this project — see `finalresults.md` for full detail.

## Reward fix (Fix #28) and D0 capability-abort (Fix #29) — in progress

A constant-harvest-fraction sweep of the raw env found a structural reward-landscape problem
common to *both* algorithm families: `harvested_mg` saturates almost immediately past the
physically-optimal fraction (~0.12-0.15) while `time_avg_od` collapses steeply over that same
range, and `reward_harvest` gives no signal distinguishing the two — a low-gradient plateau on
one side, a steep unpunished cliff on the other. **Fix #28** adds an immediate, harvest-event-
local penalty when post-harvest OD falls below a floor fraction of `OD_TARGET`, replacing what
was previously only a delayed, hard-to-credit-assign signal via `reward_od`'s per-step decay. A
companion fix (a rolling-window OD-average reward term, meant to close the gap between
`reward_od`'s instantaneous-peak reward and the gate's time-*averaged* metric) was tried and
**reverted** after direct verification showed it reintroduced the "never harvest" exploit
Fix #10 had already closed — a non-oscillating never-harvest trajectory's rolling average
trivially equals its own instantaneous value, so it rewarded the unharvested baseline more than
a real harvesting policy. Verified via the same sweep before any training: Fix #28 alone
correctly restores "harvesting beats never-harvesting" at D1/D2 (unaffected either way) without
touching D0's separate, pre-existing weak bias toward never-harvesting (present in the
*original* reward, not introduced by this fix).

Live monitoring of the first full-budget test run (v29) surfaced a second, independent gap:
PPO's deterministic policy actively collapsed at D0 (crash rate 0%→80% over 5 chunks) with
**no safety mechanism watching for it** — the existing capability-demotion logic (Fix #15) is
gated on `current_difficulty > 0`, since demotion has nowhere to go from the floor tier, but
that guard also meant D0 had zero active response to an in-place collapse. **Fix #29** extends
the failure-streak counter to track at D0 too; since there's no tier to demote to, it stops the
run cleanly with a diagnostic instead. The first version of this fix (v30) then hit its own
false-positive: it aborted a run that was healthy the entire time (0% crash, growing harvest)
purely because `time_avg_od` hadn't cleared its threshold in 12 chunks — `capability_failing`
measures "hasn't passed yet," not "is getting worse," and D0 runs commonly take many chunks to
first clear the OD bar even when nothing is wrong. **Corrected**: D0's abort now additionally
requires `crash_rate >= DEMOTION_CRASH_RATE`, so it only fires on genuine collapse, matching the
signal (elevated crash rate) that actually distinguished the two cases. The D1/D2 demotion path
deliberately keeps no crash floor — Fix #15 exists specifically to catch 0%-crash quality
regression there (v17's documented collapse), which has a proven prior-good baseline to regress
from; a D0 run stuck below the bar from the start has no such baseline, so the same floor-free
logic doesn't transfer.

`v31_d0_abort_crash_floor` (both fixes together) is running as of this writing — not yet
concluded. Update this section with the result once it reaches a verdict.

## Known issue: observation-space versioning

The observation went 6 → 8 channels (Fix #18), then was **reverted to 6 as the default**
(`OBS_EXTENDED = False` class attribute on `GeneticPhotobioreactorEnv`) — the 8-channel
extension remains available via `OBS_EXTENDED = True` but is opt-in, not default. Any change to
this flag orphans checkpoints saved under the other setting, in either direction, including
`model_data/BEST_bc_clone_D2_validated/` if it was produced under a different setting than the
one currently active. `validate.py` detects and reports the mismatch explicitly rather than
crashing confusingly. To use an orphaned checkpoint, either regenerate it under the current env
(`python bc/bc_pretrain.py` takes ~15 min) or check out the matching env revision. Any future
observation change orphans checkpoints the same way, so bump a version marker and note it here.

## Env configuration flags

`GeneticPhotobioreactorEnv` class attributes, so a configuration can be isolated rather than
bundled (v22 changed three things at once and its regression could not be attributed):

| flag | default | effect |
|---|---|---|
| `LIGHT_FOULING_COEF` | `0.0002` | light-path biofouling. **Inert at this value** — calibrated for lab OD600 (~1–10) while this sim's `od` is ~0.018, so it accumulates ~0.0003 against a 0.5 cap. A realistic value is ~0.075, but a probe showed the culture is light-saturated so it changes no behaviour. |
| `TURB_FOULING_COEF` | `0.0` | nephelometer window fouling (biases the reading high). Realistic, but works **against** Fix #18, which improved that same signal. |
| `HARVEST_PUMP_ERROR` | `0.0` | ±fraction harvest delivery error. Forces closed-loop harvest control. |
| `USE_EPISODE_PHASE` | `True` | `True`: obs channel 7 = `step/max_steps`. Not sim-to-real transferable, and reveals when the `time_avg_od` scoring window opens at step 3600 — a gaming hazard `held_out_sweep.py` cannot detect (an action trace of v21 showed no gaming). `False`: periodic harvest-cycle phase, transferable and non-gameable. Set `False` for anything intended for deployment. |
