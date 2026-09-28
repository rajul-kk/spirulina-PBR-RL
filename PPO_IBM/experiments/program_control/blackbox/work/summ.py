import json, sys, pandas as pd, numpy as np
R = [json.loads(l) for l in open('../trials/results.jsonl')]
pat = sys.argv[1] if len(sys.argv) > 1 else ''
for r in R:
    if pat not in r['batch']: continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    dw = [h['lab_dry_weight_mg_per_L'] for h in r['harvests']]
    print(f"{r['batch']:32s} inoc {r['inoculum']:5d} lost {int(r['culture_lost'])} hrs {r['hours_run']:5.1f} tot {r['total_harvested_mg']:7.0f}")
    print("   DW:", ' '.join(f"{x:.0f}" for x in dw))
    idx = df.hour.isin(range(0, 145, 12))
    print("   NTU:", ' '.join(f"{x:.0f}" for x in df.turbidity_ntu[idx]))
    print("   T  :", ' '.join(f"{x:.1f}" for x in df.temp_c[idx]))
    print("   pH :", ' '.join(f"{x:.2f}" for x in df.ph[idx]), " cond", ' '.join(f"{x/1000:.1f}" for x in df.conductivity[idx]))
