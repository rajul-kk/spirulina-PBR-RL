# replay a controller's density estimator on logged hourly data (interpolated to 72 s steps); compare X_hat to lab DW
import json, sys, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('c', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pat = sys.argv[2]
R = [json.loads(l) for l in open('../trials/results.jsonl')]
errs = []
for r in R:
    if pat not in r['batch']: continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    stir = df.stir_rpm.median()
    c = m.Controller(dict(stir=stir))
    hrs = np.arange(0, len(df) * 50) * 0.02
    ntu = np.interp(hrs, df.hour, df.turbidity_ntu); T = np.interp(hrs, df.hour, df.temp_c)
    # emulate harvest drops: step NTU right after each harvest hour
    Xh = {}
    for t in range(len(hrs)):
        c.act(dict(t=t, turbidity_ntu=ntu[t], temp_c=T[t], ph=10, pump_L=0, conductivity=3e4, lux=0))
        if (t + 1) % 600 == 0: Xh[round((t + 1) * 0.02)] = c.X
    row = []
    for h in r['harvests']:
        x = Xh.get(round(h['hour'])); row.append(x / h['lab_dry_weight_mg_per_L'])
    errs += row
    print(f"{r['batch']:22s} stir {stir:4.0f}  Xhat/DW: " + ' '.join(f"{v:.2f}" for v in row))
print('mean', np.mean(errs), 'sd', np.std(errs))
