#!/bin/sh
# Score a 4M follow-up TD3 seed (comparison_protocol.md section 7) on the final split: final
# checkpoint (primary) and best-det checkpoint (secondary), evaluated with the reset it trained with.
# usage (from experiments/program_control):
#   sh results/final/followup/score_followup.sh <core lru|lstm> <arm> <steps 2m|4m> <seed> <session dir> <reset>
# e.g. score_followup.sh lstm lstm60 2m 3 results/rl_4m/lstm60-s3/session2 60
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
core=$1; arm=$2; steps=$3; seed=$4; d=$(cd "$5" && pwd -W); reset=$6
ck=td3_${core}_checkpoints; [ "$core" = lstm ] && ck=td3_checkpoints
out=results/final/followup
mkdir -p "$out/secondary"
run() {
  [ -f "$2" ] && return
  python harness.py controllers/td3_actor.py --params "{\"actor_path\": \"$1\", \"reset\": $reset}" \
    --split final --workers 2 --out "$2.tmp" && mv "$2.tmp" "$2"
}
run "$d/model_data/$ck/actor.pth" "$out/td3_${arm}_${steps}__s$seed.json"
run "$d/model_data/${ck}_best/actor.pth" "$out/secondary/td3_${arm}_${steps}_best__s$seed.json"
