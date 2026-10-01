#!/bin/sh
# Score a finished TD3 seed on the final split: final checkpoint (primary) and best-det
# checkpoint (secondary). usage: score_rl.sh <core lru|lstm> <seed> <session dir>
export PYTHONIOENCODING=utf-8
# Scoring shares the laptop with other jobs: lift the per-episode wall-clock guard (protocol 4).
export PC_EPISODE_WALL_S=7200
core=$1; seed=$2; d=$(cd "$3" && pwd -W)
ck=td3_checkpoints; [ "$core" = lru ] && ck=td3_lru_checkpoints
run() {
  [ -f "$2" ] && return
  python harness.py controllers/td3_actor.py --params "{\"actor_path\": \"$1\", \"reset\": 600}" \
    --split final --workers 2 --out "$2.tmp" && mv "$2.tmp" "$2"
}
run "$d/model_data/$ck/actor.pth" results/final/td3_${core}__s$seed.json
run "$d/model_data/${ck}_best/actor.pth" results/final/secondary/td3_${core}_best__s$seed.json
