#!/bin/sh
# bi-s5 (finished by D0 capability abort, session 2) scoring for the training-context ablation
# (comparison_protocol.md section 9). Strictly sequential, idempotent. Run from experiments/program_control.
#  1. final and best checkpoints on --split val   (log: results/final/ctx/score_val.log)
#  2. section 9 selection via selection.select_arm('bi', seeds=[5]) (rule unchanged), merged into
#     selection.json (other bi seeds untouched)
#  3. selected checkpoint and raw final checkpoint on --split final (log: results/final/ctx/score_final.log)
# usage: nohup sh results/final/ctx/score_bi5.sh > /dev/null 2>&1 &
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
D=results/rl_ctx/bi-s5/session2
VLOG=results/final/ctx/score_val.log
FLOG=results/final/ctx/score_final.log

for ck in final best; do
  echo "$(date +%FT%T) start bi $ck s5 val" >> $VLOG
  sh results/final/ctx/score_ctx.sh lstm bi "$ck" 5 "$D" val >> $VLOG 2>&1
  echo "$(date +%FT%T) done  bi $ck s5 val rc=$?" >> $VLOG
done

sel=$(python -I -c "
import sys, json
sys.path.insert(0, 'results/final/ctx')
import selection
e = selection.select_arm('bi', seeds=[5])['5']
p = selection.OUT
d = json.load(open(p))
d.setdefault('bi', {})['5'] = e
json.dump(d, open(p, 'w'), indent=1)
print(e['selected'])
") || { echo "$(date +%FT%T) bi s5 selection FAILED" >> $FLOG; exit 1; }
echo "$(date +%FT%T) bi s5 selected=$sel (selection.json updated)" >> $FLOG

# selected checkpoint first (endpoint), then the raw final (pre-registered secondary); skip duplicate
for ck in $sel final; do
  [ "$ck" = "$done_ck" ] && continue
  echo "$(date +%FT%T) start bi $ck s5 final" >> $FLOG
  sh results/final/ctx/score_ctx.sh lstm bi "$ck" 5 "$D" final >> $FLOG 2>&1
  echo "$(date +%FT%T) done  bi $ck s5 final rc=$?" >> $FLOG
  done_ck=$ck
done
echo "$(date +%FT%T) bi s5 scoring finished" >> $FLOG
