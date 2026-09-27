# experiments/program_control/: controllers as programs

Head-to-head of controllers written as code against the TD3 actors, on identical episodes.
Every contender uses only the 6 sensor channels (the TD3 demo expert reads the simulator's true
OD; that version is kept only as the `oracle_expert` reference).

| file | what it does |
|---|---|
| `harness.py` | Runs a controller program on a split and scores it. Splits: `search` (12 D2 episodes, seeds 3,000,000+; the only split any search sees), `test` (the `td3_held_out_sweep.py` protocol: 40 main + 12 high-pop), `det` (the curriculum's `DET_EVAL_SET`). |
| `process_state.py` | Written process state: turbidity, temperature and growth estimates as named fields, the alternative to a recurrent hidden state. |
| `controllers/` | `oracle_expert.py` (privileged reference), `sensor_expert.py` (hand-tuned seed; CMA-ES tunes its knobs), `td3_actor.py` (runs a TD3 checkpoint in the harness; reproduces `run_td3_eval_episode` exactly). |
| `cmaes_tune.py` | Classical baseline: CMA-ES over the sensor expert's 6 knobs, on the search split. |
| `evolve.py` | LLM program evolution. Same prompt packet, seed program, candidate budget and scoring for every writer. `auto` drives a local model through Ollama; `prompt`/`score` let an outside writer (e.g. Claude in an agent session) take part. Generated code is statically checked (whitelisted imports, no file/OS/exec access) before it runs. |
| `results/` | Archives (`evolve/<writer>/archive.jsonl`, candidate files, traces), CMA-ES log and best params. |

```
python experiments/program_control/harness.py experiments/program_control/controllers/sensor_expert.py --split test
python experiments/program_control/cmaes_tune.py --gens 12 --popsize 8
python experiments/program_control/evolve.py auto --run qwen3_8b --model qwen3:8b --budget 16
python experiments/program_control/evolve.py table
```

The Claude track is a white-box, agentic writer: it had read the simulator source and ran its
own probe experiments on calibration seeds (7,000,000+), which the packet-only local models
could not. Compare writers with that in mind.
