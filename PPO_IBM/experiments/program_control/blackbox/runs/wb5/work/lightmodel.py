# Reduced light-response model transcribed from genetic_env._update_biology (reading source only).
import numpy as np
L=20e-3/0.3
def fI(X, I, rpm, clump=1.0, Ks=100., Ki=2500.):
    ksc=rpm*0.004
    kr=0.5+0.2*X+ksc; kb=0.2+0.25*X+ksc; kg=0.05+0.06*X+ksc
    z=(np.arange(8)+0.5)/8*L
    qr=I*0.4*np.exp(-kr*z); qt=qr+I*0.4*np.exp(-kb*z)+I*0.2*np.exp(-kg*z)
    s=clump**(-1/3); gr=s*qr; gq=s*qt
    raw=gr/(Ks+gr+gq**2/Ki); fresp=raw.mean()
    rm,tm=gr.mean(),gq.mean(); fint=rm/(Ks+rm+tm**2/Ki)
    w=0.5+0.5*np.clip((rpm-50)/150,0,1)
    Ip=np.sqrt(Ks*Ki); fmax=0.4*Ip/(2*Ks+0.4*Ip)
    return min(1,(w*fint+(1-w)*fresp)/fmax), tm
def tax(rpm):
    return 1-0.35/(1+np.exp(-0.12*(rpm-100)))
if __name__=="__main__":
    for rpm in (50,80,100,130,200):
        print("rpm",rpm,"repair tax",round(tax(rpm),3))
        for I in (300,600,1000,1400,1600,2000):
            row=[]
            for od in (0.1,0.25,0.5,0.75,1.0,1.5,2,3,4,6):
                X=od*300; f,tm=fI(X,I,rpm)
                row.append(f"{od*300*f*tax(rpm):6.0f}")
            print(f"  I={I:5d} X*f*tax:", " ".join(row))
