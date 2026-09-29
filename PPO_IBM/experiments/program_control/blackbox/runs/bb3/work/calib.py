import json,csv
import numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
tab={}
for r in R:
    rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
    rat=[]
    for h in r['harvests']:
        t=int(h['hour']); dw=h['lab_dry_weight_mg_per_L']
        ntu=np.mean([float(rows[i]['turbidity_ntu']) for i in (t-1,)])  # hour before harvest
        rat.append(ntu/(dw*250/300))
    print('%-18s'%r['batch'],' '.join('%.2f'%x for x in rat))
