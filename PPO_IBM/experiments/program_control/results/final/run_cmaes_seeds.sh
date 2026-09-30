#!/bin/sh
# CMA-ES seeds 1-4 on physics v3, same settings as seed 0 (results/v3/cmaes); protocol section 3.
export PYTHONIOENCODING=utf-8
for s in 1 2 3 4; do
  python cmaes_tune.py --gens 12 --popsize 8 --workers 2 --seed $s --out results/final/cmaes/s$s
done
