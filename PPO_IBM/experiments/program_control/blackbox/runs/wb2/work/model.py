"""My own mean-field re-derivation of the growth equations (not the simulator)."""
import numpy as np
D = 20e-3/0.3
zq = (np.arange(8)+0.5)/8*D
def temp_factor(T, topt=36.0, tmin=10.0, tmax=44.5):
    if T <= tmin or T >= tmax: return 0.0
    num = (T-tmax)*(T-tmin)**2
    den = (topt-tmin)*((topt-tmin)*(T-topt)-(topt-tmax)*(topt+tmin-2*T))
    return min(max(num/den, 0), 1)
def f_light(X, L, rpm, clump=1.0, Ks=100., Ki=2500.):
    s = clump**(-1/3)
    ks = 0.004*rpm
    r = L*0.4*np.exp(-(0.5+0.2*X+ks)*zq)
    tot = r + L*0.4*np.exp(-(0.2+0.25*X+ks)*zq) + L*0.2*np.exp(-(0.05+0.06*X+ks)*zq)
    r, tot = r*s, tot*s
    fr = (r/(Ks+r+tot**2/Ki)).mean()
    rm, tm = r.mean(), tot.mean()
    fi = rm/(Ks+rm+tm**2/Ki)
    w = 0.5+0.5*np.clip((rpm-50)/150, 0, 1)
    Ip = np.sqrt(Ks*Ki); fmax = 0.4*Ip/(2*Ks+0.4*Ip)
    return min(max((w*fi+(1-w)*fr)/fmax, 0), 1), tm
def stir_tax(rpm, integ=None):
    rep = 1-0.35/(1+np.exp(-0.12*(rpm-100)))
    ss = np.clip((rpm-80)/100, 0, 1)
    # steady-state membrane integrity: dm = -ss*0.05 + (1-m)*0.1 = 0 -> m = 1 - ss/2
    m = max(0.0, 1-ss*0.5) if integ is None else integ
    return rep*(1-0.15*(1-m))
def T_ss(L, rpm):
    # steady temperature with capacity-limited chiller (0.6 C/h), ambient loss 0.1*(T-25)
    heat = L*0.001 + (rpm/200)**3*0.1
    if heat - 0.1*(35-25) <= 0.6: return 35.0
    return 25 + (heat-0.6)/0.1
if __name__ == "__main__":
    print("temp factor topt36:", [(T, round(temp_factor(T),3)) for T in [33,34,35,36,37,38,39,40,41]])
    print("temp factor topt35:", [(T, round(temp_factor(T,35),3)) for T in [34,35,36,37,38,39,40]])
    for rpm in [50, 60, 70, 80, 90, 100, 120, 150, 200]:
        print("rpm", rpm, "stir tax", round(stir_tax(rpm),3))
    print()
    for X in [20, 50, 100, 150, 225, 300, 400, 600, 1000]:
        row = []
        for rpm in [50, 70, 85, 100, 130]:
            best = max(((f_light(X, L, rpm)[0]*stir_tax(rpm)*temp_factor(T_ss(L, rpm)), L) for L in range(100, 2001, 50)))
            row.append(f"rpm{rpm}: g={best[0]:.3f}@L{best[1]} (gX={best[0]*X:.0f})")
        print(f"X={X:5}", " | ".join(row))
