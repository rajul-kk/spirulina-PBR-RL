#!/bin/sh
# physics v3 baselines on the held-out test split, then CMA-ES re-tuned on v3
export PYTHONIOENCODING=utf-8
python harness.py controllers/sensor_expert.py --split test --out results/v3/test_sensor_expert.json
python harness.py controllers/oracle_expert.py --privileged --split test --out results/v3/test_oracle.json
python harness.py controllers/sensor_expert.py --params '{"stir": 50.0, "light": 1714.0477402789018, "setpoint": 0.5442789325760237, "gain": 2.0432451037144785, "cap": 0.3130494971268078, "turb_per_od": 211.58875501824093}' --split test --out results/v3/test_cmaes_v2params.json
python harness.py results/evolve/claude/cand_004.py --split test --out results/v3/test_claude_whitebox_v2.json
python harness.py blackbox/work/controller.py --split test --out results/v3/test_claude_blackbox_v2.json
python cmaes_tune.py --gens 12 --popsize 8 --workers 2 --out results/v3/cmaes
