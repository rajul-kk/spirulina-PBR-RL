#!/bin/sh
# Score the section 4 controls (LSTM-600, LRU-600; 2M; results/rl_v3/*/session2) on the val split,
# strictly sequentially. Both checkpoints per run (final, best-det); all 10 control runs have a best
# checkpoint, so 20 jobs. Idempotent (score_ctx.sh skips existing outputs). Run from experiments/program_control.
while read core arm seed dir; do
  for ck in final best; do
    echo "$(date +%FT%T) start $arm $ck s$seed"
    sh results/final/ctx/score_ctx.sh "$core" "$arm" "$ck" "$seed" "$dir" val
    echo "$(date +%FT%T) done  $arm $ck s$seed rc=$?"
  done
done <<'LIST'
lstm lstmctl 1 results/rl_v3/rl-v3-lstm-s1/session2
lstm lstmctl 2 results/rl_v3/rl-v3-lstm-s2/session2
lstm lstmctl 3 results/rl_v3/rl-v3-lstm-s3/session2
lstm lstmctl 4 results/rl_v3/rl-v3-lstm-s4/session2
lstm lstmctl 5 results/rl_v3/rl-v3-lstm-s5/session2
lru lructl 1 results/rl_v3/rl-v3-lru-s1/session2
lru lructl 2 results/rl_v3/rl-v3-lru-s2/session2
lru lructl 3 results/rl_v3/rl-v3-lru-s3/session2
lru lructl 4 results/rl_v3/rl-v3-lru-s4/session2
lru lructl 5 results/rl_v3/rl-v3-lru-s5/session2
LIST
