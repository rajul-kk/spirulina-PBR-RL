# Closed-loop digital twin: runs a controller file against the mean-field model (mfsim physics), step dt=0.02 h.
import math, importlib.util, numpy as np, sys
from lightmodel import fI, tax
from mfsim import tfac
def loadc(path):
    spec = importlib.util.spec_from_file_location('c' + str(abs(hash(path))), path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.Controller
def run(C, od0, mu_eff=0.031, topt=36.0, drift=1.0, params=None, seed=0, trace=False):
    rng = np.random.RandomState(seed)
    ctrl = C(params) if params else C()
    od, c, T, mi, rpm_s = od0, 1.0, 35.0, 1.0, 50.0; tot = 0.0; hsum = 0.0; hn = 0; dt = 0.02; tr = []
    fcache = {}; A = 200.0
    for t in range(7200):
        turb = 250*od*c**(-1/3)/(1+0.05*od)*drift*(1+0.02*rng.randn())
        turb = min(1000.0, max(0.0, turb))
        rpm, I, h = ctrl.act({'t': t, 'turbidity_ntu': turb, 'temp_c': T*drift, 'ph': 10.0, 'pump_L': 0.0, 'conductivity': 30000.0, 'lux': I*30 if t else 0})
        h = min(0.5, max(0.0, h)); hsum += h; hn += 1
        rpm_s = 0.9*rpm_s + 0.1*rpm
        Teq = max(35.0, 25 + 10*(0.001*I + (rpm_s/200)**3*0.1 - 0.6)); T += (Teq - T)*0.1*dt*3
        if t % 5 == 0: f, tm = fI(od*300, I, rpm_s, c); fcache['f'] = f; fcache['tm'] = tm
        f, tm = fcache['f'], fcache['tm']
        A += (tm - A) * dt / 2.5; shock = math.exp(-3e-6*max(tm - A, 0)**2)
        s = min(max((rpm_s-80)/100,0),1); mi = min(max(mi + (-s*0.05 + (1-mi)*0.1)*dt,0),1)
        mu = mu_eff*f*tax(rpm_s)*(1-0.15*(1-mi))*tfac(T, topt)*shock
        kla = (0.6+5*min(max(rpm_s/200,0.25),1)**1.3)*1.05/(1+(od/10)**2)
        do2 = 8 + mu*od*300*1.5/kla; mu *= 1/(1+(do2/35)**4)
        od *= math.exp(mu*dt)
        stick = od*0.05*max(0.1,1-rpm_s/250)
        br = 0.5*max(0,(rpm_s-80)/120)**2*math.sqrt(c) + 0.005*math.sqrt(max(c-1,0))
        c = max(1.0, c + (stick - br - mu*(c-1))*dt)
        if t > 0 and t % 600 == 0:
            hf = hsum/hn; hsum = 0; hn = 0; tot += od*hf*6000; od *= (1-hf)
        if trace and t % 300 == 0: tr.append((t*dt, round(od,2), round(c,2), round(I), round(hf if t%600==0 and t else 0,2)))
    return (tot, tr) if trace else tot
SCEN = [(0.07,.10),(0.25,.25),(0.45,.25),(0.7,.20),(2.0,.1),(6.0,.1)]
def score(C, params=None, mus=(0.022,0.03,0.036), detail=False):
    tot = 0; d = []
    for od0, w in SCEN:
        vals = [run(C, od0, mu, params=params, drift=dr, seed=i) for i,(mu,dr) in enumerate(zip(mus,(1.03,0.97,1.0)))]
        tot += w*np.mean(vals); d.append((od0, [round(v) for v in vals]))
    return (tot, d) if detail else tot
if __name__ == '__main__':
    C = loadc(sys.argv[1]); print(score(C, detail=True))
