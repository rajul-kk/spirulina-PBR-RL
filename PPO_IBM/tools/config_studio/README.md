# config_studio

Visual editor for the curriculum gate thresholds (`training/curriculum_schedule.py`)
and scripted-expert control law (`bc/bc_pretrain.py`), with a live-preview panel that
runs real episodes against `genetic_env.py`, and git-backed saves.

## Run it

```
cd PPO_IBM
python tools/config_studio/server.py --port 8765
```

Open http://127.0.0.1:8765. No new dependencies — stdlib `http.server`.

## What it edits

- **Curriculum gates** — `ADVANCE_TARGETS[0/1/2]` in `curriculum_schedule.py`.
- **Demotion tuning** — `CAPABILITY_DEMOTION_CHUNKS`, `DEMOTION_CRASH_RATE`,
  `MASTERY_WINDOW`, `MASTERY_REQUIRED_STREAK`.
- **Expert control law** — `EXPERT_OD_SETPOINT`, `EXPERT_GAIN`, `EXPERT_FRAC_CAP`,
  `EXPERT_STIR_RANGE`, `EXPERT_LIGHT_RANGE` in `bc/bc_pretrain.py`.

Each field maps to one entry in `schema.py` — add a tunable constant by adding one
entry there.

## Live preview

Runs `n_episodes` (default 8) of the scripted expert against
`GeneticPhotobioreactorEnv` using whatever's currently in the fields (saved or not),
scored against the selected tier's gate. For a statistically-backed verdict, use
`experiments/bc_scaffold/scripts/expert_sweep.py --n 40` instead.

## Saving

Each save is a targeted regex edit to the one line/dict-entry that field owns,
verified by reading the value back, then `git add` + `git commit` scoped to that file.
Nothing is pushed — commits stay local.

## Files

- `schema.py` — editable-field declarations.
- `config_io.py` — regex read/write + git commit.
- `preview_runner.py` — scripted expert for the live-preview endpoint.
- `server.py` — stdlib `http.server` backend.
- `static/index.html`, `static/app.js` — vanilla-JS frontend.
