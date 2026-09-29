import json,sys,csv
R=[json.loads(l) for l in open('../trials/results.jsonl')]
pat=sys.argv[1] if len(sys.argv)>1 else ''
for r in R:
    if pat not in r['batch']: continue
    print(r['batch'],r['inoculum'],r['total_harvested_mg'],'LOST' if r['culture_lost'] else '', r.get('error'))
    print('  DW :',' '.join('%.0f'%h['lab_dry_weight_mg_per_L'] for h in r['harvests']))
    print('  H  :',' '.join('%.0f'%h['harvested_mg'] for h in r['harvests']))
    if len(sys.argv)>2:
        rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
        step=int(sys.argv[2])
        for row in rows[::step]:
            print('   h%5s ntu %7s ph %6s pump %6s cond %8s T %6s st %s L %s hf %s'%(row['hour'],row['turbidity_ntu'],row['ph'],row['pump_L'],row['conductivity'],row['temp_c'],row['stir_rpm'],row['light_umol'],row['harvest_frac']))
