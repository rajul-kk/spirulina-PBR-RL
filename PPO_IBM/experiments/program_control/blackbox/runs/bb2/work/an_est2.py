# Replay a stateful controller (step-level, turbidity held per hour) and compare its X to lab DW.
import sys, importlib
from analyze import *
pat = sys.argv[1]; mod = importlib.import_module(sys.argv[2])
allerr = []
for r in load():
    if pat not in r['batch']: continue
    c = csvrows(r['batch']); ctl = mod.Controller(); est = {}
    for step in range(7200):
        i = min(int(step * 0.02), len(c['hour']) - 1)
        ctl.act({'turbidity_ntu': c['turbidity_ntu'][i], 'temp_c': c['temp_c'][i]})
        if step % 50 == 49: est[int((step + 1) * 0.02) - 1] = ctl.X
    out = []
    for h in r['harvests']:
        e = est.get(int(h['hour']) - 1); d = h['lab_dry_weight_mg_per_L']
        out.append(f'{d:.0f}/{e:.0f}'); allerr.append(np.log(e / d))
    if '-q' not in sys.argv: print(r['batch'], ' '.join(out))
a = np.array(allerr); print('rms log err %.3f bias %+.3f' % (np.sqrt(np.mean(a ** 2)), a.mean()))
