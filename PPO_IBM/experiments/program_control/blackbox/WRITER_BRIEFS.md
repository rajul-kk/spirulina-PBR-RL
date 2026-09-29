# Writer briefs for the repeated head-to-head (physics v3)

Both tracks get the same model (Opus), the same 300-batch pilot budget through `plant_trial.py`,
their own seed block and run directory, and the same judging: the project owner runs the final
`controller.py` on the held-out test split. The only difference is access to the simulator
source. The briefs give no design hints (the first black-box run's prompt leaked two: the finite
batch and turbidity-vs-assay calibration).

Root: `E:\SEGP\.claude\worktrees\agent-a87c1f2edcbd1556b\PPO_IBM\experiments\program_control\blackbox\`
(Git Bash: `/e/SEGP/.claude/worktrees/agent-a87c1f2edcbd1556b/PPO_IBM/experiments/program_control/blackbox/`).

## Black-box (runs bb1, bb2, bb3)
- May read: `PLANT_MANUAL.md`, and everything under `runs/<own run>/`.
- Must not open, search, import or list anything else in the repository.
- Experiments: `python plant_trial.py --run <own run> runs/<own run>/work/<controller>.py ...`

## White-box (runs wb1, wb2, wb3)
- May read: `PLANT_MANUAL.md`, everything under `runs/<own run>/`, and the simulator source
  `PPO_IBM/environments/genetic_env.py` (plus anything it imports from `PPO_IBM/training/`).
- Must not read other writers' runs, `experiments/program_control/results/`, `blackbox/work/`,
  `blackbox/trials/`, `controllers/`, or any evaluation/split code (`harness.py`, `evolve.py`,
  `cmaes_tune.py`).
- Experiments only through `python plant_trial.py --run <own run> --privileged ...` (the logs
  then also carry the true OD, temperature and cell count). Must not instantiate or import the
  simulator in its own scripts, which would bypass the budget.

## Both
- Keep `runs/<own run>/work/LAB_NOTEBOOK.md`: every experiment (command, why, what was learned)
  and every file read.
- Deliver `runs/<own run>/work/controller.py` (numpy, math, collections only; no file/OS/network).
