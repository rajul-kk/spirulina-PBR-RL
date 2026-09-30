import csv, sys, json
def load(b):
    rows = list(csv.DictReader(open(f"../trials/{b}.csv")))
    return rows
if __name__ == "__main__":
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rows = load(sys.argv[1])
    print("hour turb  od_true ratio temp  Ttrue stir light frac cells")
    for i, r in enumerate(rows):
        if i % step: continue
        od = float(r["true_od"]); tu = float(r["turbidity_ntu"])
        print("%5.1f %6.1f %6.3f %5.2f %5.2f %5.2f %4.0f %5.0f %5.3f %6s" % (
            float(r["hour"]), tu, od, tu / 250 / max(od, 1e-6), float(r["temp_c"]), float(r["true_temp_c"]),
            float(r["stir_rpm"]), float(r["light_umol"]), float(r["harvest_frac"]), r["true_cells"]))
