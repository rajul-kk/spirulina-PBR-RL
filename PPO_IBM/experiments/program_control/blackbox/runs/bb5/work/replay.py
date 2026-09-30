import sys, csv, json, importlib.util, math
T='../trials/'
spec=importlib.util.spec_from_file_location('c', sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res={}
for line in open(T+'results.jsonl'):
    r=json.loads(line); res[r['batch']]=r
pats=sys.argv[2].split(',')
allerr=[]
for b,r in res.items():
    if not any(p in b for p in pats): continue
    rows=list(csv.DictReader(open(T+b+'.csv')))
    c=m.Controller()
    est={}
    for i,row in enumerate(rows):
        obs={k:float(v) for k,v in row.items()}
        for s in range(50):
            t=i*50+s
            if t>=7200: break
            o=dict(obs); o['t']=t
            c.act(o)
            if (t+1)%600==0: est[(t+1)//50]=c.xm
    lab={int(h['hour']):h['lab_dry_weight_mg_per_L'] for h in r['harvests']}
    s=' '.join('%4.0f/%4.0f'%(est.get(hh,0),lab[hh]) for hh in sorted(lab))
    allerr+= [math.log(est[hh]/lab[hh]) for hh in lab if hh in est]
    print('%-20s %s'%(b,s))
import statistics
print('rms log err', math.sqrt(sum(e*e for e in allerr)/len(allerr)))
