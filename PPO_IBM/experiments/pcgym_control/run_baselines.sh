#!/bin/sh
# Baselines for the PC-Gym comparison (protocol section 2): reference PID, CMA-ES seeds 0-4
# (each scored on final when done), then SAC seeds 1-5 (rl_sac.py scores final itself).
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
F=results/final
[ -f $F/ref-pid__only.json ] || python harness.py controllers/pid.py --split final --out $F/ref-pid__only.json
for s in 0 1 2 3 4; do
  [ -f results/cmaes/s$s/DONE ] || python cmaes_tune.py --seed $s --out results/cmaes/s$s
  [ -f $F/cmaes-pid__s$s.json ] || python harness.py controllers/pid.py --split final \
    --params "$(python -c "import json;print(json.dumps(json.load(open('results/cmaes/s$s/best.json'))['params']))")" \
    --out $F/cmaes-pid__s$s.json
done
for s in 1 2 3 4 5; do
  [ -f results/sac/s$s/final.json ] || python rl_sac.py --seed $s --steps 100000 --out results/sac/s$s
  [ -f results/sac/s$s/final.json ] && cp results/sac/s$s/final.json $F/sac__s$s.json
done
