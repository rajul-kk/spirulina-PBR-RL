import json, csv, sys
T='../trials/'
pat = sys.argv[1] if len(sys.argv)>1 else ''
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    if pat not in r['batch']: continue
    rows=list(csv.DictReader(open(T+r['batch']+'.csv')))
    Tm=sum(float(x['temp_c']) for x in rows)/len(rows)
    print('%-18s inoc %5d tot %8.0f lost %d Tmean %.2f | ' % (r['batch'], r['inoculum'], r['total_harvested_mg'], r['culture_lost'], Tm) +
          ' '.join('%4.0f' % h['lab_dry_weight_mg_per_L'] for h in r['harvests']) + ' | ' + ' '.join('%4.0f'%h['harvested_mg'] for h in r['harvests']))
