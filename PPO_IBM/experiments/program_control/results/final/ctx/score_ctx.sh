#!/bin/sh
# Score one TD3 checkpoint for the training-context ablation (comparison_protocol.md section 9).
# usage (from experiments/program_control):
#   sh results/final/ctx/score_ctx.sh <core lstm|lru> <arm label> <ckpt final|best> <seed> <session dir> <split val|final>
# e.g. score_ctx.sh lstm lstmctl final 1 results/rl_v3/rl-v3-lstm-s1/session2 val
# Output: results/final/ctx/td3_<arm>_<ckpt>__s<seed>.<split>.json ; hidden reset 600.
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
core=$1; arm=$2; ckpt=$3; seed=$4; d=$(cd "$5" && pwd -W); split=$6
ck=td3_${core}_checkpoints; [ "$core" = lstm ] && ck=td3_checkpoints
[ "$ckpt" = best ] && ck=${ck}_best
out=results/final/ctx/td3_${arm}_${ckpt}__s${seed}.${split}.json
[ -f "$out" ] && exit 0
[ -f "$d/model_data/$ck/actor.pth" ] || { echo "missing $d/model_data/$ck/actor.pth"; exit 1; }
python harness.py controllers/td3_actor.py --params "{\"actor_path\": \"$d/model_data/$ck/actor.pth\", \"reset\": 600}" \
  --split "$split" --workers 2 --out "$out.tmp" && mv "$out.tmp" "$out"
