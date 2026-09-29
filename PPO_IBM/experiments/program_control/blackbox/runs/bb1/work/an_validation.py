"""Validation summary for the final controller (labels val_* and c14_final_i30)."""
import json, numpy as np
R = [json.loads(l) for l in open('../trials/results.jsonl')]
def summ(name, rs):
    y = np.array([r['total_harvested_mg'] for r in rs]); lost = sum(r['culture_lost'] for r in rs)
    if len(y) == 0: return
    print(f'{name:<28} n {len(y):3d} median {np.median(y):8.0f} p25 {np.percentile(y,25):8.0f} min {y.min():8.0f} max {y.max():8.0f} lost {lost}')
nat = [r for r in R if '_val_nat_' in r['batch']]
summ('natural (val_nat a+b)', nat)
summ('  val_nat_a', [r for r in nat if 'nat_a' in r['batch']])
summ('  val_nat_b', [r for r in nat if 'nat_b' in r['batch']])
for lo, hi in [(0, 100), (100, 200), (200, 400), (400, 1000), (1000, 99999)]:
    summ(f'  natural inoc {lo}-{hi}', [r for r in nat if lo <= r['inoculum'] < hi])
summ('fixed inoc 30 (final)', [r for r in R if 'c14_final_i30' in r['batch']])
summ('fixed inoc 50', [r for r in R if '_val_i50' in r['batch']])
summ('fixed inoc 100', [r for r in R if '_val_i100' in r['batch']])
allf = [r for r in R if ('_val_' in r['batch'] or 'c14_final' in r['batch'])]
summ('all final-controller batches', allf)
print('errors:', [r['batch'] for r in allf if r.get('error')])
