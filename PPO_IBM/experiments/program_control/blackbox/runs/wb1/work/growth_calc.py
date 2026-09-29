"""My own steady-state (mean-field) growth-rate calculator, written from the equations in
genetic_env.py. Not the simulator: no cells, no stochasticity, no gas/carbonate dynamics."""
import numpy as np, math
L = 0.02/0.3
def temp_factor(T, topt=36.0, tmin=10.0, tmax=44.5):
    if T <= tmin or T >= tmax: return 0.0
    num = (T-tmax)*(T-tmin)**2
    den = (topt-tmin)*((topt-tmin)*(T-topt)-(topt-tmax)*(topt+tmin-2*T))
    return min(max(num/den,0),1)
def steady_T(I, rpm):
    h = 0.001*I + (rpm/200)**3*0.1
    return max(35.0, 25 + (h-0.6)/0.1)
def f_light(I, od, rpm, clump=1.0, Ks=100., Ki=2500.):
    X = od*300
    ks = rpm*0.004
    z = (np.arange(8)+0.5)/8*L
    qr = I*0.4*np.exp(-(0.5+0.2*X+ks)*z)
    qt = qr + I*0.4*np.exp(-(0.2+0.25*X+ks)*z) + I*0.2*np.exp(-(0.05+0.06*X+ks)*z)
    cs = clump**(-1/3)
    g, q = cs*qr, cs*qt
    fresp = np.mean(g/(Ks+g+q**2/Ki))
    rm, tm = g.mean(), q.mean()
    fint = rm/(Ks+rm+tm**2/Ki)
    w = 0.5+0.5*min(max((rpm-50)/150,0),1)
    Ip = math.sqrt(Ks*Ki); fmax = 0.4*Ip/(2*Ks+0.4*Ip)
    return min(max((w*fint+(1-w)*fresp)/fmax,0),1), tm
def mu(I, od, rpm, clump=1.0, topt=36.0, mumax=0.04, fatigue=True):
    fI, pm = f_light(I, od, rpm, clump)
    rt = 1-0.35/(1+math.exp(-0.12*(rpm-100)))
    ss = min(max((rpm-80)/100,0),1)
    mi = 1/(1+ss*0.05/0.1) if fatigue else 1.0   # steady membrane integrity
    ft = 1-0.15*(1-mi)
    fph = math.exp(-0.5*(0.2/0.7)**2)
    T = steady_T(I, rpm)
    return mumax*fI*rt*ft*fph*temp_factor(T, topt), pm, T
if __name__ == "__main__":
    import sys
    print("rpm effect at OD 1, I=1500:")
    for r in [50,60,70,80,90,100,120,150,200]:
        print(r, round(mu(1500,1.0,r)[0],5))
    print("\nbest I and productivity vs OD (rpm 50/70):")
    for od in [0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0,9.0]:
        for r in [50,70]:
            best = max((mu(I,od,r)[0],I) for I in range(100,2001,50))
            m,pm,T = mu(best[1],od,r)
            print(f"od {od:5.2f} rpm {r}: I*={best[1]:5d} mu={m:.4f} P={m*od*300:.2f} mg/L/h pathmean={pm:.0f} T={T:.1f}  mu@2000={mu(2000,od,r)[0]:.4f}")
