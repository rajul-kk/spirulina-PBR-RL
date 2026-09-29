# Per-interval production (mg/L per 12 h) = DW_{k+1} - DW_k*(1-f_k), by light level, with interval temp.
import sys
from analyze import *
pat = sys.argv[1]
for r in load():
    if pat not in r['batch']: continue
    c = csvrows(r['batch']); H = r['harvests']; out = []
    for k in range(len(H) - 1):
        d0, d1 = H[k]['lab_dry_weight_mg_per_L'], H[k + 1]['lab_dry_weight_mg_per_L']
        f = H[k]['harvested_mg'] / (20 * d0)
        m = (c['hour'] >= H[k]['hour']) & (c['hour'] < H[k + 1]['hour'])
        out.append(f"L{c['light_umol'][m].mean()/100:.0f} T{c['temp_c'][m].mean():.1f} X{d0*(1-f):.0f} P{d1-d0*(1-f):+.0f}")
    print(r['batch'], ' | '.join(out))
