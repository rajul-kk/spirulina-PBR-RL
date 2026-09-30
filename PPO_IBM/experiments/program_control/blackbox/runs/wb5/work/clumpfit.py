import math, sys
from ana import load
def sim(b, kdil=1.0):
    r = load(b); c = 1.0; out = []
    d0 = r[0]['turbidity_ntu'] / (250 * r[0]['true_od'] / (1 + 0.05 * r[0]['true_od']))
    for a, n in zip(r, r[1:]):
        dt = n['hour'] - a['hour']; rpm = a['stir_rpm']; od = a['true_od']
        mu = max(0.0, math.log(n['true_cells'] / a['true_cells']) / dt) if n['pump_L'] == a['pump_L'] and a['true_cells'] > 0 else 0.0
        mu_m = max(0.0, math.log(n['true_od'] / a['true_od']) / dt) if n['pump_L'] == a['pump_L'] else 0.0
        for _ in range(50):
            stick = od * 0.05 * max(0.1, 1 - rpm / 250)
            br = 0.5 * max(0, (rpm - 80) / 120) ** 2 * math.sqrt(c) + 0.005 * math.sqrt(max(c - 1, 0))
            c += (stick - br - kdil * mu_m * (c - 1)) * 0.02
            c = max(c, 1.0)
        ratio = n['turbidity_ntu'] / (250 * n['true_od'] / (1 + 0.05 * n['true_od']))
        out.append((n['hour'], ratio, d0 * c ** (-1 / 3), c))
    return out
for b in sys.argv[1:]:
    o = sim(b)
    print(b, ' '.join(f"h{h:.0f}:{r:.2f}/{p:.2f}" for h, r, p, c in o if int(h) % 12 == 0))
