# Validation summary of the final controller: median / p25 harvest and losses by inoculum group.
from analyze import *
R = [r for r in load() if r['batch'].split('_', 1)[1].startswith('val_')]
def grp(i):
    for lo, hi, name in [(0, 50, '30-49'), (50, 90, '50-89'), (90, 130, '90-129'), (130, 200, '130-199'), (200, 330, '200-329'),
                         (330, 600, '330-599'), (600, 1500, '600-1499'), (1500, 3500, '1500-3499'), (3500, 1e9, '>=3500')]:
        if lo <= i < hi: return (lo, name)
G = {}
for r in R: G.setdefault(grp(r['inoculum']), []).append(r)
print(f"{'inoculum':12s} {'n':>3s} {'median':>8s} {'p25':>8s} {'min':>8s} {'lost':>4s}")
for k in sorted(G):
    v = np.array([r['total_harvested_mg'] for r in G[k]]); lost = sum(r['culture_lost'] for r in G[k])
    print(f'{k[1]:12s} {len(v):3d} {np.median(v):8.0f} {np.percentile(v,25):8.0f} {v.min():8.0f} {lost:4d}')
for name, sel in [('ALL val', R), ('random draws', [r for r in R if 'val_rand' in r['batch']]),
                  ('inoc 100-400', [r for r in R if 100 <= r['inoculum'] <= 400])]:
    v = np.array([r['total_harvested_mg'] for r in sel])
    if len(v): print(f'{name:14s} n={len(v)} median {np.median(v):.0f} p25 {np.percentile(v,25):.0f} lost {sum(r["culture_lost"] for r in sel)} errors {sum(1 for r in sel if r["error"])}')
