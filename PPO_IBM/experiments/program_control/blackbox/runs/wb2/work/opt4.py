import evalm
from evalm import evaluate
from opt3 import P3
evalm.DP = 0.08
cfgs = {"ref sp1.7 b24x2": dict(od_sp=1.7, k_final=3, burst_P=24, burst_B=2)}
for P, B in [(12, 1), (18, 1.5), (24, 1.5), (24, 3), (36, 2), (36, 3), (48, 3)]:
    cfgs[f"b{P}x{B}"] = dict(od_sp=1.7, k_final=3, burst_P=P, burst_B=B)
for L in [1700, 1900]:
    cfgs[f"L{L}"] = dict(od_sp=1.7, k_final=3, burst_P=24, burst_B=2, Lmax=L)
for r in [60, 90]:
    cfgs[f"rpm{r}"] = dict(od_sp=1.7, k_final=3, burst_P=24, burst_B=2, rpm=r)
base = None
for name, kw in cfgs.items():
    r = evaluate(lambda: P3(**kw))
    if base is None: base = r
    print(f"{name:18s}", " ".join(f"n0={k}: {v.mean():7.0f} ({(v/base[k]).mean()*100-100:+.1f}%)" for k, v in r.items()), flush=True)
