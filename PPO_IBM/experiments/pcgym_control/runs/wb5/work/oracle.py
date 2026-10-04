# compare EKF controller vs oracle-state controller on my sim, per-seed cost split
import numpy as np, warnings, sys, json; warnings.filterwarnings('ignore')
import mysim
from controller import Controller
params=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
n=int(sys.argv[2]) if len(sys.argv)>2 else 10
def oracle_run(seed):
    sc=mysim.scenario(seed); c=Controller(params); st={'x':sc['x0'].copy(),'k':0}
    def ekf(y):
        k=c.k; x=st['x']; c.z=np.array([x[0],x[1],sc['Ti'][k],sc['Caf'][k]])
    c._ekf=ekf
    # need true state: re-run sim manually
    x=sc['x0'].copy(); cost=0; nr=np.random.RandomState(sc['noise_seed'])
    for k in range(120):
        st['x']=x
        obs={"t_min":k*mysim.DT,"Ca":x[0]+nr.normal(0,.002),"T":x[1]+nr.normal(0,.2),"Ca_sp":sc['sp'][k]}
        u=float(np.clip(c.act(obs),295,302)); x=mysim.step(x,u,sc['Ti'][k],sc['Caf'][k]); cost+=((x[0]-sc['sp'][k])/.01)**2
    return cost/120
o=[oracle_run(s) for s in range(n)]
e=[mysim.run(Controller(params),s)[0] for s in range(n)]
print("oracle mean %.3f  ekf mean %.3f"%(np.mean(o),np.mean(e)))
print(np.round(o,3)); print(np.round(e,3))
