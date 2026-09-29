import json, sys, csv
res = [json.loads(l) for l in open("../trials/results.jsonl")]
sel = sys.argv[1] if len(sys.argv) > 1 else ""
for r in res:
    if sel not in r["batch"]: continue
    rows = list(csv.DictReader(open(f"../trials/{r['batch']}.csv")))
    tod = {float(x["hour"]): float(x["true_od"]) for x in rows}
    turb = {float(x["hour"]): float(x["turbidity_ntu"]) for x in rows}
    s = []
    for e in r["harvests"]:
        h = e["hour"]; od_before = e["lab_dry_weight_mg_per_L"]/300
        f = e["harvested_mg"] / max(od_before*300*20, 1e-9)
        s.append(f"{h:.0f}h od{od_before:.2f} f{f:.2f} ->{od_before*(1-f):.2f} T/od{turb[h]/250/od_before:.2f}")
    print(r["batch"], r["inoculum"], r["total_harvested_mg"])
    print("   " + " | ".join(s))
