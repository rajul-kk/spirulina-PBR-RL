import json, sys, csv, glob, os, math
import numpy as np
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
def load():
    out = []
    for line in open(os.path.join(T, 'results.jsonl')):
        out.append(json.loads(line))
    return out
def csvrows(b):
    with open(os.path.join(T, b + '.csv')) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}
if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else ''
    detail = '-d' in sys.argv
    for r in load():
        if pat not in r['batch']: continue
        dw = [h['lab_dry_weight_mg_per_L'] for h in r['harvests']]
        hv = [h['harvested_mg'] for h in r['harvests']]
        print(f"{r['batch']:40s} inoc {r['inoculum']:5d} tot {r['total_harvested_mg']:9.0f} lost {r['culture_lost']} hrs {r['hours_run']} err {r['error']}")
        print('   DW:', ' '.join(f'{x:.0f}' for x in dw))
        if any(hv): print('   HV:', ' '.join(f'{x:.0f}' for x in hv))
        if detail:
            c = csvrows(r['batch'])
            for i in range(0, len(c['hour']), 6):
                print('   ' + ' '.join(f'{k[:4]}={c[k][i]:.1f}' for k in c if k not in ('lux',)))
