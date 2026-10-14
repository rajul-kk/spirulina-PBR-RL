#!/bin/sh
# Second scoring queue for the training-context ablation (comparison_protocol.md section 9).
# Waits (polling every 60 s) until no score_bi_val.sh process is running, then runs strictly
# sequentially: (1) a diagnostic scoring of bi-s1's best checkpoint on --split det (the 9-instance
# det-eval set used during training, reset 600) to check the training det-eval figure reproduces
# through the harness; (2) w150 final and best on --split val for s1 and s2.
# Skips outputs that already exist. Run from experiments/program_control; log: results/final/ctx/score_val.log
# usage: nohup sh results/final/ctx/score_queue2.sh >> results/final/ctx/score_val.log 2>&1 &
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200

while ps -ef 2>/dev/null | grep -v grep | grep -q "score_bi_val.sh"; do
  sleep 60
done
echo "$(date +%FT%T) queue2: score_bi_val.sh no longer running, starting"

# (1) diagnostic: bi-s1 best checkpoint on the det-eval set
d=$(cd results/rl_ctx/bi-s1/session3 && pwd -W)
out=results/final/ctx/diag/td3_bi_best__s1.det.json
mkdir -p results/final/ctx/diag
if [ ! -f "$out" ]; then
  echo "$(date +%FT%T) start diag bi best s1 det"
  python harness.py controllers/td3_actor.py \
    --params "{\"actor_path\": \"$d/model_data/td3_checkpoints_best/actor.pth\", \"reset\": 600}" \
    --split det --workers 2 --out "$out.tmp" && mv "$out.tmp" "$out"
  echo "$(date +%FT%T) done  diag bi best s1 det rc=$?"
fi

# (2) w150 val scoring
for seed in 1 2; do
  for ck in final best; do
    echo "$(date +%FT%T) start w150 $ck s$seed val"
    sh results/final/ctx/score_ctx.sh lstm w150 "$ck" "$seed" "results/rl_ctx/w150-s$seed/session3" val
    echo "$(date +%FT%T) done  w150 $ck s$seed rc=$?"
  done
done
echo "$(date +%FT%T) queue2 finished"
