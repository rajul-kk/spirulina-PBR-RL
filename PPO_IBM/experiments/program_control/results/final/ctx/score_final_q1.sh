#!/bin/sh
# Final-split scoring queue 1 for the training-context ablation (comparison_protocol.md section 9).
# For bi s1-3 and w150 s1-2: the val-selected checkpoint (endpoint; all selected "best" in selection.json)
# and the raw final 2M checkpoint (pre-registered secondary), each scored once on --split final.
# Strictly sequential; score_ctx.sh skips outputs that already exist. Run from experiments/program_control.
# usage: nohup sh results/final/ctx/score_final_q1.sh >> results/final/ctx/score_final.log 2>&1 &
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
echo "$(date +%FT%T) final_q1 start"
for job in bi:1 bi:2 bi:3 w150:1 w150:2; do
  arm=${job%%:*}; seed=${job##*:}
  for ck in best final; do
    echo "$(date +%FT%T) start $arm $ck s$seed final"
    sh results/final/ctx/score_ctx.sh lstm "$arm" "$ck" "$seed" "results/rl_ctx/$arm-s$seed/session3" final
    echo "$(date +%FT%T) done  $arm $ck s$seed rc=$?"
  done
done
echo "$(date +%FT%T) final_q1 finished"
