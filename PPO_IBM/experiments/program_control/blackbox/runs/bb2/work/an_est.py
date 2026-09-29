# Replay a controller on logged turbidity (hourly) and compare its X estimate to lab DW before each harvest.
import sys, importlib
from analyze import *
pat = sys.argv[1]; mod = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else 'ctl_v1')
for r in load():
    if pat not in r['batch']: continue
    c = csvrows(r['batch']); ctl = mod.Controller()
    out = []
    for h in r['harvests']:
        i = int(h['hour']) - 1
        est = ctl._xest(c['turbidity_ntu'][i], c['hour'][i])
        out.append(f"{h['lab_dry_weight_mg_per_L']:.0f}/{est:.0f}")
    print(r['batch'], 'DW/est:', ' '.join(out))
