import controller as C, numpy as np
from innov import score
base=dict(U=C.U,J=C.J,K0=C.K0,QV=C.QV)
p={"q":[1e-6,1e-3,1e-5,0.01]}
for name in ['U','J','K0','QV']:
    for s in [0.9,0.97,1.03,1.1]:
        setattr(C,name,base[name]*s); r=score(p); setattr(C,name,base[name])
        print(name,s,'1-step %.5f 3-step %.5f'%(r[0],r[2]),flush=True)
print('base','%.5f %.5f'%(score(p)[0],score(p)[2]))
