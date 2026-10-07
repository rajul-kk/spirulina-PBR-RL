- `td3_gru_best__s3.nocheckpoint.json`: gru-s3 has no best-det checkpoint. TD3 saves one only after a
  det-eval with 0% crash (td3/TD3.py), and every det-eval of this seed crashed (it ended in the D0
  capability abort). The scoring script ran on a missing file (200 load errors). Not a result; the
  seed's secondary score is "no checkpoint".
- `td3_rtu_best__s5.nocheckpoint.json`: rtu-s5 has no best-det checkpoint, for the same reason as
  gru-s3 (every det-eval crashed; the run ended in the D0 capability abort). 200 load errors. Not
  a result; the seed's secondary score is "no checkpoint".
- `td3_rtu__s4.walltimeout.json`: 3 of 200 episodes (seeds 20000009, 20000033, 20000034) hit the
  7200 s per-episode wall-clock guard because the laptop slept during scoring (the run took
  23,498 s). This is the same harness invalidation as lru-s3 (protocol section 4), so the run is
  repeated. Episodes are deterministic given the seed.
