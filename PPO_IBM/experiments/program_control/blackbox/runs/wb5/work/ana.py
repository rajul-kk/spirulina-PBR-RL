import csv, sys, glob, os, json, math
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
def load(b):
    rows = list(csv.DictReader(open(os.path.join(T, b + '.csv'))))
    return [{k: float(v) for k, v in r.items()} for r in rows]
def summary(b):
    r = load(b)
    by = {int(x['hour']): x for x in r}
    out = []
    for h in (0, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108):
        if h in by and h + 6 in by:
            a, c = by[h], by[h + 6]
            if c['pump_L'] == a['pump_L']:
                g = (c['true_od'] - a['true_od']) / 6
                mu = math.log(c['true_od'] / a['true_od']) / 6
                ratio = a['turbidity_ntu'] / (250 * a['true_od'] / (1 + 0.05 * a['true_od']))
                out.append(f"h{h}:od{a['true_od']:.2f} g{g*1000:.0f} mu{mu*1000:.0f} r{ratio:.2f} T{a['true_temp_c']:.1f}")
    return ' '.join(out)
if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else '*'
    for f in sorted(glob.glob(os.path.join(T, pat + '.csv'))):
        b = os.path.basename(f)[:-4]; print(b, summary(b))
