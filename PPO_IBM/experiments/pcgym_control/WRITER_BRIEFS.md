# Writer briefs for the PC-Gym CSTR comparison

Both tracks get the same model (Opus), the same 300-batch pilot budget through `plant_trial.py`,
their own scenario block and run directory, and the same judging: the project owner runs the
final `controller.py` on the held-out `final` split. The only difference is access to the plant
model. The briefs give no design hints.

Root: `E:\SEGP\.claude\worktrees\agent-a87c1f2edcbd1556b\PPO_IBM\experiments\pcgym_control\`

## Black-box (runs bb1-bb5)
- May read: `PLANT_MANUAL.md`, and everything under `runs/<own run>/`.
- Must not open, search, import or list anything else in the repository or the Python
  environment (including `plant_trial.py`, `tasks.py`, `harness.py` and the `pcgym` package).
- Experiments: `python plant_trial.py --run <own run> runs/<own run>/work/<controller>.py ...`

## White-box (runs wb1-wb5)
- May read: `PLANT_MANUAL.md`, everything under `runs/<own run>/`, the task
  definition `tasks.py`, and PC-Gym's model source
  `E:\SEGP\.venv\Lib\site-packages\pcgym\model_classes.py` (and other files of that package).
- Must not read other writers' runs, `plant_trial.py`, `harness.py`, `cmaes_tune.py`, `rl_sac.py`, `controllers/`,
  `results/`, or `docs/`.
- Experiments only through `python plant_trial.py --run <own run> --privileged ...` (the logs
  then also carry the true state and the unmeasured feed disturbances). Must not import or run
  `tasks.py` or `pcgym` in its own scripts, which would bypass the budget.

## Both
- Keep `runs/<own run>/work/LAB_NOTEBOOK.md`: every experiment (command, why, what was learned)
  and every file read.
- Deliver `runs/<own run>/work/controller.py` (numpy, math, collections only).
- Other jobs share the machine: never kill processes you did not start.
