#!/bin/sh
# Evaluation-time reset sweep on --split test (never final). Run from experiments/program_control:
#   nohup sh diagnostics/reset_sweep.sh >> diagnostics/reset_sweep.log 2>&1 &
# "never" = reset 1000000 (episodes are 7200 steps, so the state is never zeroed).
# Resumable: outputs that already exist are skipped. Strictly sequential.
export PYTHONIOENCODING=utf-8
export PC_EPISODE_WALL_S=7200
out=diagnostics/reset_sweep
mkdir -p "$out"

# newest session folder (by number) that holds the checkpoint
find_actor() {   # $1 run dir, $2 ckpt dir name
  best=""; bn=-1
  for s in "$1"/session*; do
    n=${s##*session}
    [ -f "$s/model_data/$2/actor.pth" ] && [ "$n" -gt "$bn" ] && { bn=$n; best=$s; }
  done
  if [ -n "$best" ]; then d=$(cd "$best/model_data/$2" && pwd -W); echo "$d/actor.pth"; fi
}

run() {   # $1 arm  $2 seed  $3 actor path (windows)
  for r in 30 60 150 600 never; do
    rv=$r; [ "$r" = never ] && rv=1000000
    o="$out/$1_s$2_r$r.json"
    [ -f "$o" ] && continue
    echo "$(date +%T) start $1 s$2 r$r"
    python harness.py controllers/td3_actor.py --params "{\"actor_path\": \"$3\", \"reset\": $rv}" \
      --split test --workers 2 --out "$o.tmp" && mv "$o.tmp" "$o"
    echo "$(date +%T) done  $1 s$2 r$r"
  done
}

for s in 1 2 3 4 5; do
  a=$(find_actor results/rl_4m/lru-s$s td3_lru_checkpoints);  [ -n "$a" ] && run lru600 $s "$a" || echo "MISSING lru600 s$s"
done
for s in 1 2 3 4 5; do
  a=$(find_actor results/rl_4m/lstm-s$s td3_checkpoints)
  [ -z "$a" ] && a=$(find_actor results/rl_v3/rl-v3-lstm-s$s td3_checkpoints)
  [ -n "$a" ] && run lstm600 $s "$a" || echo "MISSING lstm600 s$s"
done
for s in 1 2 3 4 5; do
  a=$(find_actor results/rl_4m/lstm60-s$s td3_checkpoints); [ -n "$a" ] && run lstm60 $s "$a" || echo "MISSING lstm60 s$s"
done
for s in 1 2 3 4 5; do
  a=$(find_actor results/rl_v3/rl-v3-gru-s$s td3_gru_checkpoints); [ -n "$a" ] && run gru $s "$a" || echo "MISSING gru s$s"
done
echo "$(date +%T) sweep complete"
