import csv, sys
for b in sys.argv[1:]:
    rows = list(csv.DictReader(open(f"../trials/{b}.csv")))
    print(b)
    for r in rows:
        h = float(r["hour"])
        if abs(h / 6 - round(h / 6)) < 1e-6:
            od = float(r["true_od"]); tb = float(r["turbidity_ntu"])
            print(f"  {h:5.0f} od {od:6.3f} turb {tb:6.1f} ratio {tb/max(od,1e-6):5.0f} L {float(r['light_umol']):6.0f} T {r['true_temp_c']} stir {r['stir_rpm']} f {r['harvest_frac']} cells {r['true_cells']} pH {r['ph']}")
