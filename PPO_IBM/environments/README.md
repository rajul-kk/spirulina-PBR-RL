# environments/

| file | what it is |
|---|---|
| `genetic_env.py` | `GeneticPhotobioreactorEnv`, the simulator every trainer uses: a super-individual model of a 20 L Spirulina flat-panel reactor (physics v2, 2026-09-26: real densities, thermostat, alkalinity/DIC carbonate system, stoichiometric nutrients) plus the reward (harvest + PBRS shaping). |

Regression checks for the env: `python experiments/env_diagnosis/core_audit_check.py`.
Model decisions and their history: `docs/decision_history.md`; accepted simplifications:
`docs/known_limitations.md`.
