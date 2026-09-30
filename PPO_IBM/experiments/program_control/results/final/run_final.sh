#!/bin/sh
# Score every frozen controller once on the 'final' split (protocol section 4). Idempotent:
# an output that already exists is never re-run. Run from experiments/program_control.
export PYTHONIOENCODING=utf-8
F=results/final
run() {  # run <out-name> <controller> [extra harness args...]
  out=$F/$1.json; shift
  [ -f "$out" ] && return
  python harness.py "$@" --split final --workers 2 --out "$out.tmp" && mv "$out.tmp" "$out"
}
cma() {  # cma <out-name> <best.json>; best.json is rewritten every generation, so wait for gen 12
  grep -q '"gen": 12' "$(dirname "$2")/log.jsonl" 2>/dev/null || return
  run "$1" controllers/sensor_expert.py --params "$(python -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['params']))" "$2")"
}
# runs/<run>/FROZEN is written when the writer has handed in its final report.
for i in 1 2 3 4 5; do
  [ -f blackbox/runs/wb$i/FROZEN ] && run whitebox__wb$i blackbox/runs/wb$i/work/controller.py
  [ -f blackbox/runs/bb$i/FROZEN ] && run blackbox__bb$i blackbox/runs/bb$i/work/controller.py
done
run ref-expert__only controllers/sensor_expert.py
run ref-oracle__only controllers/oracle_expert.py --privileged
cma cmaes__s0 results/v3/cmaes/best.json
for s in 1 2 3 4; do cma cmaes__s$s $F/cmaes/s$s/best.json; done
