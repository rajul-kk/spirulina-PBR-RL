#!/bin/sh
# w150-s3 (D0 capability abort, session 4) and w150-s5 (2M, D1, session 4) scoring for the training-context
# ablation (comparison_protocol.md section 9). Strictly sequential, idempotent: seed 3, then seed 5.
# Per seed (arm w150, core lstm, reset 600, session dir results/rl_ctx/w150-s<seed>/session4):
#  1. final and best checkpoints on --split val   (log: results/final/ctx/score_val.log)
#  2. section 9 selection via selection.select_arm('w150', seeds=[<seed>]) (rule unchanged), merged into
#     selection.json (other w150 seeds untouched)
#  3. selected checkpoint and raw final checkpoint on --split final (log: results/final/ctx/score_final.log);
#     scored once if the selected checkpoint is the final one
# usage: nohup sh results/final/ctx/score_w150_35.sh > /dev/null 2>&1 &
# quoting test (no scoring, writes nothing): SELTEST=<seed with val files> sh results/final/ctx/score_w150_35.sh
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
VLOG=results/final/ctx/score_val.log
FLOG=results/final/ctx/score_final.log

# prints the selected checkpoint; merges into selection.json unless SELTEST is set
select_seed() {
  python -I -c "
import sys, json, os
sys.path.insert(0, 'results/final/ctx')
import selection
e = selection.select_arm('w150', seeds=[$1])['$1']
p = selection.OUT
d = json.load(open(p))
d.setdefault('w150', {})['$1'] = e
if not os.environ.get('SELTEST'):
    json.dump(d, open(p, 'w'), indent=1)
print(e['selected'])
"
}

if [ -n "$SELTEST" ]; then
  select_seed "$SELTEST"
  exit $?
fi

for s in 3 5; do
  D=results/rl_ctx/w150-s$s/session4
  for ck in final best; do
    echo "$(date +%FT%T) start w150 $ck s$s val" >> $VLOG
    sh results/final/ctx/score_ctx.sh lstm w150 "$ck" $s "$D" val >> $VLOG 2>&1
    echo "$(date +%FT%T) done  w150 $ck s$s val rc=$?" >> $VLOG
  done

  sel=$(select_seed $s) || { echo "$(date +%FT%T) w150 s$s selection FAILED" >> $FLOG; exit 1; }
  echo "$(date +%FT%T) w150 s$s selected=$sel (selection.json updated)" >> $FLOG

  # selected checkpoint first (endpoint), then the raw final (pre-registered secondary); skip duplicate
  done_ck=""
  for ck in $sel final; do
    [ "$ck" = "$done_ck" ] && continue
    echo "$(date +%FT%T) start w150 $ck s$s final" >> $FLOG
    sh results/final/ctx/score_ctx.sh lstm w150 "$ck" $s "$D" final >> $FLOG 2>&1
    echo "$(date +%FT%T) done  w150 $ck s$s final rc=$?" >> $FLOG
    done_ck=$ck
  done
done
echo "$(date +%FT%T) w150 s3 s5 scoring finished" >> $FLOG
