"""Per-batch physics-limited baseline: replay setpoints + EKF-estimated feed disturbances
from a plant log in my model with an oracle (full-state) MPC, no noise."""
import sys, glob, json, numpy as np
from simfit import load, step as fstep, P as P0
from controller import Controller
class Oracle(Controller):
    def _ekf(self, y):
        self.z = np.array(self._true, float)

def baseline(d):
    c=Controller({'q':[1e-6,1e-3,2e-5,0.02]}); zs=[]
    for i in range(len(d['Tc'])):
        c._ekf((d['Ca'][i],d['T'][i])); c.u_prev=d['Tc'][i]; zs.append(c.z.copy())
    zs=np.array(zs)
    # disturbance seen at k acts over k->k+1; use next filtered estimate (better informed)
    caf=np.r_[zs[1:,2],zs[-1,2]]; tf=np.r_[zs[1:,3],zs[-1,3]]
    x=np.array([zs[min(3,len(zs)-1),0]*0+d['Ca'][0], d['T'][0]])
    # smooth initial state with first filtered estimate
    x=zs[0,:2].copy()
    o=Oracle(); p=dict(P0); e=0
    for k in range(len(d['Tc'])):
        e+=((x[0]-d['Ca_sp'][k])/0.01)**2
        o._true=(x[0],x[1],caf[k],tf[k]); u=o.act(dict(Ca=x[0],T=x[1],Ca_sp=d['Ca_sp'][k],_z=(x[0],x[1],caf[k],tf[k])))
        p['Caf']=caf[k]; p['Tf']=tf[k]; x=fstep(x,u,p,n=10)
    return e/len(d['Tc'])
res=[json.loads(l) for l in open('../trials/results.jsonl')]
cost={r['batch']:r['cost'] for r in res}
groups={}
for fn in sorted(glob.glob('../trials/'+sys.argv[1])):
    import os; b=os.path.basename(fn)[:-4]; d=load(fn); bl=baseline(d)
    lab=b.split('_',1)[1]; groups.setdefault(lab,[]).append((cost[b],bl))
    print(b,'actual %.3f oracle-replay %.3f regret %+.3f'%(cost[b],bl,cost[b]-bl),flush=True)
for lab,v in groups.items():
    v=np.array(v); print(lab,'n',len(v),'actual mean %.3f  oracle %.3f  regret mean %.3f (se %.3f)'%(v[:,0].mean(),v[:,1].mean(),(v[:,0]-v[:,1]).mean(),(v[:,0]-v[:,1]).std()/np.sqrt(len(v))))
