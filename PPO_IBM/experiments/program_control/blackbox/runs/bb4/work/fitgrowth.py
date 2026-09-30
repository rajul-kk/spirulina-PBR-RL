import sys, csv, json, os, numpy as np
from scipy.optimize import least_squares
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
V=20.0
def load(stir=50, pat=''):
    I=[]
    for l in open(os.path.join(T, 'results.jsonl')):
        r = json.loads(l)
        if pat not in r['batch']: continue
        rows = {float(x['hour']): x for x in csv.DictReader(open(os.path.join(T, r['batch'] + '.csv')))}
        if abs(float(rows[0.0]['stir_rpm'])-stir)>1: continue
        H=r['harvests']
        for a,b in zip(H[:-1],H[1:]):
            F=a['harvested_mg']/(V*a['lab_dry_weight_mg_per_L'])
            x0=a['lab_dry_weight_mg_per_L']*(1-F); x1=b['lab_dry_weight_mg_per_L']
            Ls=np.array([float(rows[k]['light_umol']) for k in np.arange(a['hour'],b['hour'])])
            Ts=np.array([float(rows[k]['temp_c']) for k in np.arange(a['hour'],b['hour'])])
            I.append((r['batch'],a['hour'],x0,x1,Ls,Ts))
    return I
def mu(X,L,th):
    mumax,Ks,Ki,k,m=th
    kx=max(k*X,1e-9)
    # depth-averaged Haldane response, Beer-Lambert, numeric over 8 layers
    z=(np.arange(8)+0.5)/8
    I=L*np.exp(-kx*z)
    f=np.mean(I/(Ks+I+I*I/Ki))
    return mumax*f-m
def sim(x0,Ls,th):
    X=x0
    for L in Ls:
        for _ in range(4):
            X=X*np.exp(mu(X,L,th)*0.25)
    return X
def resid(p,I):
    th=np.exp(p)
    return np.array([np.log(sim(x0,Ls,th)/x1) for (_,_,x0,x1,Ls,Ts) in I])
if __name__=='__main__':
    I=load(50)
    print('intervals',len(I))
    p0=np.log([0.06,300,1500,0.002,0.003])
    r=least_squares(resid,p0,args=(I,))
    th=np.exp(r.x); print('mumax %.4f Ks %.1f Ki %.1f k %.5f m %.4f'%tuple(th), 'rms', np.sqrt(np.mean(r.fun**2)))
    for X in [20,50,100,200,400,700,1000,1500]:
        print(X, ' '.join('%4d:%.4f/%.1f'%(L,mu(X,L,th),mu(X,L,th)*X) for L in [400,800,1200,1600,2000]))
    np.save(os.path.join(os.path.dirname(__file__),'th.npy'),th)
    # residual by batch (strain effect)
    res=r.fun; bs=sorted(set(i[0] for i in I))
    for b in bs:
        m=[j for j,i in enumerate(I) if i[0]==b]
        print(b, '%.3f'%np.mean(res[m]), 'T %.1f'%np.mean([I[j][5].mean() for j in m]))
