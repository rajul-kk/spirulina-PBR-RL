import csv, sys, importlib.util
from mfsim import run
spec = importlib.util.spec_from_file_location("c", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = run(m.Controller, inoc=int(sys.argv[2]), seed=1, log_every=300, strain={"mumax": 0.04, "topt": 36, "tau": 2.5})
print("model total", round(res["total"]))
rows = list(csv.DictReader(open(sys.argv[3]))) if len(sys.argv) > 3 else []
tr = {float(r["hour"]): r for r in rows}
for (h, od, turb, T, light, stir, clump, shock, fI, do2) in res["logs"]:
    r = tr.get(round(h, 1)) or tr.get(h)
    extra = f" | trial od {r['true_od']} turb {r['turbidity_ntu']} light {r['light_umol']} T {r['true_temp_c']}" if r else ""
    print(f"{h:5.0f} od {od:.3f} turb {turb:6.1f} T {T:.1f} L {light:6.0f} clump {clump:.2f} shock {shock:.2f} fI {fI:.2f} DO {do2:.1f}{extra}")
