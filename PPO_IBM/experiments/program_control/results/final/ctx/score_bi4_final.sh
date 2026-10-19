#!/bin/sh
# bi-s4 final-split scoring for the training-context ablation (comparison_protocol.md section 9).
# Follow-up to score_bi4.sh (kept as the record of what ran): that driver scored both bi s4 checkpoints on
# val, then its selection step FAILED (logged "bi s4 selection FAILED"). Root cause: the python -c body sat
# inside a double-quoted shell string and was written with ["4"] (copied from the single-quoted score_bi5.sh
# form ['5']); the shell stripped the inner double quotes, so Python saw e[4] -> KeyError: 4.
# The selection was then re-run by hand (selection.select_arm('bi', seeds=[4]), rule unchanged; selected=best)
# and merged into selection.json. This driver only does the final-split scoring: selected ckpt, then raw final.
# usage (from experiments/program_control): nohup sh results/final/ctx/score_bi4_final.sh > /dev/null 2>&1 &
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
D=results/rl_ctx/bi-s4/session3
FLOG=results/final/ctx/score_final.log

for ck in best final; do
  echo "$(date +%FT%T) start bi $ck s4 final" >> $FLOG
  sh results/final/ctx/score_ctx.sh lstm bi "$ck" 4 "$D" final >> $FLOG 2>&1
  echo "$(date +%FT%T) done  bi $ck s4 final rc=$?" >> $FLOG
done
echo "$(date +%FT%T) bi s4 scoring finished" >> $FLOG
