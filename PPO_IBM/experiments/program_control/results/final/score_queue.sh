#!/bin/sh
# Score finished seeds one after another (each line: core seed session-dir). Idempotent.
while read core seed dir; do
  sh results/final/score_rl.sh "$core" "$seed" "$dir"
done <<'LIST'
lru 5 results/rl_v3/rl-v3-lru-s5/session2
lru 4 results/rl_v3/rl-v3-lru-s4/session2
lstm 3 results/rl_v3/rl-v3-lstm-s3/session2
lstm 4 results/rl_v3/rl-v3-lstm-s4/session2
LIST
