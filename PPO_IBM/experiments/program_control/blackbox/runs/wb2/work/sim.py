"""Mean-field time simulation built from my reading of the equations (my own model)."""
import numpy as np
from model import temp_factor, D, zq

def light_terms(X, L, rpm, s, Ks, Ki):
    ks = 0.004*rpm
    r = L*s*0.4*np.exp(-(0.5+0.2*X+ks)*zq)
    tot = r + L*s*0.4*np.exp(-(0.2+0.25*X+ks)*zq) + L*s*0.2*np.exp(-(0.05+0.06*X+ks)*zq)
    fr = (r/(Ks+r+tot**2/Ki)).mean()
    rm, tm = r.mean(), tot.mean()
    fi = rm/(Ks+rm+tm**2/Ki)
    w = 0.5+0.5*min(max((rpm-50)/150, 0), 1)
    Ip = np.sqrt(Ks*Ki); fmax = 0.4*Ip/(2*Ks+0.4*Ip)
    return min(max((w*fi+(1-w)*fr)/fmax, 0), 1), tm

def inoc_X(n):
    m = 1.25e8 - min(n/15000, 1)*0.45e8
    return n*m*1e-7/20.0

def simulate(policy, n0=200, mu_max=0.04, topt=36.0, Ks=100., Ki=2500., tau=2.5, T0=35.0,
             fconst=0.88*0.92*0.96, verbose=False, dens_pen=0.0):
    X = inoc_X(n0); C = 1.0; T = T0; TI = 0.0; M = 1.0; do2 = 7.0; rpm_a = 50.0
    A = 200.0; dt = 0.02; total = 0.0; hsum = 0.0; hcnt = 0; log = []
    ncell = n0
    Lc = rc = Xc = -1e9
    policy.reset() if hasattr(policy, "reset") else None
    for t in range(7200):
        od = X/300
        rpm_t, L, fr = policy(t, X, C, T)
        hsum += fr; hcnt += 1
        rpm_a = 0.9*rpm_a + 0.1*rpm_t
        rpm = rpm_a
        # temperature
        T += L*0.001*dt - 0.1*(T-25)*dt + (rpm/200)**3*0.1*dt
        err = 35-T; u_raw = 2*err+0.5*TI; u = min(max(u_raw, -0.6), 2.0)
        if u == u_raw: TI += err*dt
        T += u*dt
        # clumps
        stick = od*0.05*dt*max(0.1, 1-rpm/250)
        shear = max(0, (rpm-80)/120)**2
        brk = (0.5*shear*np.sqrt(C) + 0.005*np.sqrt(max(C-1, 0)))*dt
        s = C**(-1/3)
        if t % 4 == 0 or abs(L - Lc) > 20 or abs(rpm - rc) > 5 or X < 0.97*Xc:
            fI, tm = light_terms(X, L, rpm, s, Ks, Ki); Lc, rc, Xc = L, rpm, X
        A += dt/tau*(tm-A)
        shock = np.exp(-3e-6*max(tm-A, 0)**2)
        rep = 1-0.35/(1+np.exp(-0.12*(rpm-100)))
        ss = min(max((rpm-80)/100, 0), 1)
        M = min(max(M - ss*0.05*dt + (1-M)*0.1*dt, 0), 1)
        fat = 1-0.15*(1-M)
        fO2 = 1/(1+(do2/35)**4)
        g = mu_max*fconst*max(0.3, 1-dens_pen*od)*fI*temp_factor(T, topt)*shock*fO2*rep*fat
        net = g - 0.0004*(2 if L <= 1 else 1)
        X *= np.exp(net*dt)
        C = max(1.0, C + stick - brk + max(net, 0)*dt*(1-C))
        mix = min(max(rpm/200, 0.25), 1)
        kla = (0.6+5*mix**1.3)*1.03/(1+(od/10)**2)
        do2 += (1.5*g*X - kla*(do2-7.6))*dt
        if t > 0 and t % 600 == 0:
            f = hsum/hcnt; hsum = 0; hcnt = 0
            f = min(max(f, 0), 0.5)
            total += f*X*20; X *= (1-f)
        if verbose and t % 300 == 0:
            log.append((t*dt, X/300, C, T, L, rpm, g, do2, total))
    return total, log

class Policy:
    """Setpoint policy: light = model-optimal for (X, C), capped; stir fixed; harvest to OD setpoint;
    final k events harvest max."""
    def __init__(self, od_sp=1.0, rpm=70, Lmax=1800, k_final=2, ramp=True, Ks=100., Ki=2500.):
        self.od_sp, self.rpm, self.Lmax, self.k_final, self.Ks, self.Ki = od_sp, rpm, Lmax, k_final, Ks, Ki
        self.Lgrid = np.arange(50, Lmax+1, 50.0)
    def reset(self):
        self.L = 300.0; self.f = 0.0
    def __call__(self, t, X, C, T):
        if t % 25 == 0:
            s = C**(-1/3)
            vals = [light_terms(X, L, self.rpm, s, self.Ks, self.Ki)[0] for L in self.Lgrid]
            self.L = self.Lgrid[int(np.argmax(vals))]
        if t % 600 == 1 or t == 0:
            ev = (t//600+1)
            if ev > 11 - self.k_final:
                self.f = 0.5
            else:
                # harvest so that post-harvest OD ~ setpoint, predicting 12h growth crudely as +0 (setpoint on current)
                self.f = min(max(1 - self.od_sp*300/X, 0), 0.5)
        return self.rpm, self.L, self.f

if __name__ == "__main__":
    import sys
    for n0 in [40, 200, 1500, 5000]:
        for od_sp in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            for rpm in [60, 80, 100]:
                tot, _ = simulate(Policy(od_sp=od_sp, rpm=rpm), n0=n0)
                print(f"n0={n0} od_sp={od_sp} rpm={rpm}: {tot:.0f} mg")
