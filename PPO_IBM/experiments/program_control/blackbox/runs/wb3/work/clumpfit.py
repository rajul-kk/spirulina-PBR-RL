import csv, sys, math
for b in sys.argv[1:]:
    rows = list(csv.DictReader(open(f"../trials/{b}.csv")))
    r0 = rows[0]; drift = float(r0["turbidity_ntu"]) / (250 * float(r0["true_od"]) / (1 + 0.05 * float(r0["true_od"])))
    c = 1.0; prev = None
    print(b, "drift~", round(drift, 3))
    for i, r in enumerate(rows):  # rows every 1 h (50 steps)
        od = float(r["true_od"]); n = float(r["true_cells"]); rpm = float(r["stir_rpm"])
        if prev is not None:
            # 50 steps
            grow = max(math.log(max(n, 1) / max(prev, 1)), 0)
            for _ in range(50):
                stick = od * 0.05 * max(0.1, 1 - rpm / 250); shear = max(0, (rpm - 80) / 120) ** 2
                brk = 0.5 * shear * math.sqrt(c) + 0.005 * math.sqrt(max(c - 1, 0))
                c = max(1, c + (stick - brk) * 0.02)
            c = 1 + (c - 1) * math.exp(-grow)  # divisions: children at 1
        prev = n
        if i % 12 == 0:
            ratio = float(r["turbidity_ntu"]) / (250 * od / (1 + 0.05 * od) * drift)
            print(f"  {float(r['hour']):5.0f} od {od:.2f} implied clump {ratio**-3:.2f} model clump {c:.2f}")
