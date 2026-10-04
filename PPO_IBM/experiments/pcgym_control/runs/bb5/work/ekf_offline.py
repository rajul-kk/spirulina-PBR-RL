from simfit import *
def fx(z,Tc,nsub=8):
    p=dict(P); p['Caf']=z[2]; p['Tf']=z[3]
    x=step(z[:2],Tc,p,n=nsub); return np.r_[x,z[2:]]
def run(d,q=(1e-6,1e-3,2e-4,0.05),r=(0.002**2,0.2**2)):
    z=np.array([d['Ca'][0],d['T'][0],1.0,350.]); Pm=np.diag([1e-5,0.05,0.01,4.0])
    Q=np.diag(q); R=np.diag(r); H=np.array([[1,0,0,0],[0,1,0,0.]]); out=[]
    for i in range(len(d['Tc'])):
        if i>0:
            Tc=d['Tc'][i-1]; z0=fx(z,Tc); F=np.zeros((4,4))
            for j,e in enumerate([1e-5,1e-3,1e-5,1e-3]):
                dz=np.zeros(4);dz[j]=e;F[:,j]=(fx(z+dz,Tc)-z0)/e
            z=z0;Pm=F@Pm@F.T+Q
        y=np.array([d['Ca'][i],d['T'][i]]); S=H@Pm@H.T+R; K=Pm@H.T@np.linalg.inv(S)
        z=z+K@(y-H@z); Pm=(np.eye(4)-K@H)@Pm; out.append(z.copy())
    return np.array(out)
if __name__=='__main__':
  for fn in sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else '../trials/b00*.csv')):
    d=load(fn); o=run(d)
    print(fn[-20:],'Caf:',' '.join(f'{v:.3f}' for v in o[::8,2]))
    print(' '*20,'Tf :',' '.join(f'{v:.1f}' for v in o[::8,3]))
