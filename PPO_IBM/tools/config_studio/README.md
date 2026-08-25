# config_studio

A Tina-CMS-style visual editor pointed at this project's environment instead of a
website's layout: the curriculum gate thresholds (`training/curriculum_schedule.py`)
and the scripted-expert control law (`bc/bc_pretrain.py`), with a live-preview panel
that runs real episodes against `genetic_env.py` instead of rendering a page, and
git-backed saves in place of a CMS's content commits.

## Why this exists

`experiments/bc_scaffold/` found that a fixed, non-learned proportional harvest law
clears the D0/D1/D2 `time_avg_od` gates by a wide margin, purely by choice of
`OD_SETPOINT`/`GAIN`/`FRAC_CAP`. Tuning those constants (and the gate thresholds
themselves) has been a manual edit-run-reread-log cycle throughout this project.
This tool collapses that into: drag a value, hit "Run preview," see PASS/FAIL and
the actual numbers in seconds, with every accepted change committed to git.

## Run it

```
cd PPO_IBM
python tools/config_studio/server.py --port 8765
```

Open http://127.0.0.1:8765. No new dependencies — built on the standard library's
`http.server`, since this is a small local dev tool, not a service.

## What it edits

- **Curriculum gates** — `ADVANCE_TARGETS[0/1/2]` (harvest / p25 / crash / time_avg_od
  thresholds per difficulty tier) in `curriculum_schedule.py`.
- **Demotion tuning** — `CAPABILITY_DEMOTION_CHUNKS`, `DEMOTION_CRASH_RATE`,
  `MASTERY_WINDOW`, `MASTERY_REQUIRED_STREAK`.
- **Expert control law** — `EXPERT_OD_SETPOINT`, `EXPERT_GAIN`, `EXPERT_FRAC_CAP`,
  `EXPERT_STIR_RANGE`, `EXPERT_LIGHT_RANGE` in `bc/bc_pretrain.py`.

Each field maps to one line (or one dict entry) in the schema declared in
`schema.py` — adding a new tunable constant means adding one entry there, nothing
else needs to change.

## Live preview

Runs `n_episodes` (default 8, keep it small — this is a live UI, not a full
held-out sweep) of the scripted expert directly against `GeneticPhotobioreactorEnv`,
using whatever values are currently typed into the fields (whether saved or not),
and scores the result against the selected tier's gate. For a statistically-backed
verdict rather than a quick check, use `experiments/bc_scaffold/scripts/expert_sweep.py
--n 40` instead — this preview trades sample size for speed.

## Saving

Every successful save performs a targeted regex edit to the one line/dict-entry
that field owns (not a full-file rewrite, so comments and formatting elsewhere in
the file are untouched), verifies the file now reads back the new value, then runs
`git add` + `git commit` scoped to that single file. Every change is therefore one
independently revertable commit (`git log -- PPO_IBM/training/curriculum_schedule.py`
or `bc/bc_pretrain.py` shows the history; the page's "Recent config commits" panel
shows the same thing). Nothing is pushed anywhere — commits stay local until you
push them yourself.

## Files

- `schema.py` — the editable-field declarations (the "content model").
- `config_io.py` — regex read/write against the real source files + git commit.
- `preview_runner.py` — runs the scripted expert against `genetic_env.py` for the
  live-preview endpoint.
- `server.py` — stdlib `http.server` backend, no framework dependency.
- `static/index.html`, `static/app.js` — vanilla-JS frontend, no build step.
