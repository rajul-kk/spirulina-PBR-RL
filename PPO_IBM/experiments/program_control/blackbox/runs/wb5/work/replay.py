# Replay a trial log through a controller's estimator (hourly rows held for 50 steps) and compare to true OD.
import sys, importlib.util
from ana import load
def est(ctrl_path, b, **params):
    spec = importlib.util.spec_from_file_location('c', ctrl_path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    c = m.Controller(params or None); rows = load(b); out = []
    for i, r in enumerate(rows):
        for k in range(50):
            t = int(round(r['hour'] / 0.02)) + k
            c.p['stir'] = r['stir_rpm']
            c.act({'t': t, 'turbidity_ntu': r['turbidity_ntu'], 'temp_c': r['temp_c']})
        out.append((r['hour'], r['true_od'], c.od, c.clump))
    return out
if __name__ == '__main__':
    for b in sys.argv[2:]:
        o = est(sys.argv[1], b)
        print(b, ' '.join(f"{h:.0f}:{tr:.2f}/{e:.2f}" for h, tr, e, cl in o if int(h) % 12 == 0))
