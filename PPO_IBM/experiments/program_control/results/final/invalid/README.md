- `td3_gru_best__s3.nocheckpoint.json`: gru-s3 has no best-det checkpoint. TD3 saves one only after a
  det-eval with 0% crash (td3/TD3.py), and every det-eval of this seed crashed (it ended in the D0
  capability abort). The scoring script ran on a missing file (200 load errors). Not a result; the
  seed's secondary score is "no checkpoint".
