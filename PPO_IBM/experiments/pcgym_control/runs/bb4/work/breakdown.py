"""Where does the cost come from? Uses measured Ca (adds ~0.04 of noise floor to each mean)."""
import sys, glob, json, numpy as np
res = {}
for line in open('runs/bb4/trials/results.jsonl'):
    r = json.loads(line); res[r['batch']] = r
tot = dict(start=0.0, spchg=0.0, rest=0.0); n = 0; costs = []
for pat in sys.argv[1:]:
    for f in sorted(glob.glob('runs/bb4/trials/' + pat)):
        d = np.genfromtxt(f, delimiter=',', names=True); sp = d['Ca_sp']
        e2 = ((d['Ca'] - sp) / 0.01) ** 2 - 0.04
        ch = [i for i in range(1, 120) if sp[i] != sp[i - 1]]
        m = np.zeros(120, int); m[:10] = 1
        for c in ch: m[c:c + 10] = 2
        b = f.replace('\\', '/').split('/')[-1][:-4]
        parts = [e2[m == 1].sum() / 120, e2[m == 2].sum() / 120, e2[m == 0].sum() / 120]
        tot['start'] += parts[0]; tot['spchg'] += parts[1]; tot['rest'] += parts[2]; n += 1; costs.append(res[b]['cost'])
        print('%s cost %.3f | start %.3f spchg %.3f rest %.3f | x0 err %+.3f sp steps %s at %s | Tmax %.1f sat%% %.0f' % (
            b, res[b]['cost'], parts[0], parts[1], parts[2], d['Ca'][0] - sp[0],
            [round(float(sp[c] - sp[c - 1]), 3) for c in ch], ch, d['T'].max(),
            100 * np.mean((d['Tc'] <= 295.001) | (d['Tc'] >= 301.999))))
print('n %d mean %.3f median %.3f max %.3f | start %.3f spchg %.3f rest %.3f' % (
    n, np.mean(costs), np.median(costs), np.max(costs), tot['start'] / n, tot['spchg'] / n, tot['rest'] / n))
