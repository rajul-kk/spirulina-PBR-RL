import json, math, sys
T='../trials/'
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    hs=r['harvests']
    X=[h['lab_dry_weight_mg_per_L'] for h in hs]
    # regress ln X on t for first half and second half
    def slope(a,b):
        t=[hs[i]['hour'] for i in range(a,b)]; y=[math.log(X[i]) for i in range(a,b)]
        n=len(t); mt=sum(t)/n; my=sum(y)/n
        return sum((t[i]-mt)*(y[i]-my) for i in range(n))/sum((ti-mt)**2 for ti in t)
    if any(h['harvested_mg']>0 for h in hs): continue
    print('%-18s mu(12-60) %.4f mu(60-132) %.4f  linslope %.2f mg/L/h' % (r['batch'], slope(0,5), slope(4,11), (X[-1]-X[0])/120))
