#!/bin/sh
# Score the bi-arm (training-context ablation, comparison_protocol.md section 9) runs on the val split,
# strictly sequentially: final and best checkpoints per finished run. Idempotent (score_ctx.sh skips
# existing outputs). Run from experiments/program_control. Extend LIST as further bi seeds finish.
while read core arm seed dir; do
  for ck in final best; do
    echo "$(date +%FT%T) start $arm $ck s$seed"
    sh results/final/ctx/score_ctx.sh "$core" "$arm" "$ck" "$seed" "$dir" val
    echo "$(date +%FT%T) done  $arm $ck s$seed rc=$?"
  done
done <<'LIST'
lstm bi 1 results/rl_ctx/bi-s1/session3
lstm bi 2 results/rl_ctx/bi-s2/session3
lstm bi 3 results/rl_ctx/bi-s3/session3
LIST
