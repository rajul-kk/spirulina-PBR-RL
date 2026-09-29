"""My reduced mean-field model of the reactor, written from the equations in genetic_env.py
(not the simulator). State: X mg/L, mean clump c, acclimation A, membrane M, temp T, DO layers,
thermostat integral. Carbonate/pH/nutrients are replaced by a constant factor K_CONST
(f_carbon~0.92 * f_pH~0.96 * f_Q~0.89), to be calibrated against pilot trials."""
import math, numpy as np
L = 0.02/0.3; DT = 0.02
Z = (np.arange(8)+0.5)/8*L
K_CONST = 0.786

def temp_factor(T, topt):
    tmin, tmax = 10.0, 44.5
    if T <= tmin or T >= tmax: return 0.0
    num = (T-tmax)*(T-tmin)**2
    den = (topt-tmin)*((topt-tmin)*(T-topt)-(topt-tmax)*(topt+tmin-2*T))
    return min(max(num/den,0),1)

def light(I, X, rpm, c, Ks, Ki):
    ks = rpm*0.004
    qr = I*0.4*np.exp(-(0.5+0.2*X+ks)*Z)
    qt = qr + I*0.4*np.exp(-(0.2+0.25*X+ks)*Z) + I*0.2*np.exp(-(0.05+0.06*X+ks)*Z)
    cs = c**(-1/3); g, q = cs*qr, cs*qt
    fresp = float(np.mean(g/(Ks+g+q*q/Ki)))
    rm, tm = g.mean(), q.mean()
    fint = rm/(Ks+rm+tm*tm/Ki)
    w = 0.5+0.5*min(max((rpm-50)/150,0),1)
    Ip = math.sqrt(Ks*Ki); fmax = 0.4*Ip/(2*Ks+0.4*Ip)
    return min(max((w*fint+(1-w)*fresp)/fmax,0),1), float(tm)

class Model:
    def __init__(self, cells=300, mumax=0.04, topt=36.0, Ks=100., Ki=2500., tau=2.5, T0=35.0, k_const=K_CONST, seed=0):
        m0 = 1.25e8 - min(cells/15000,1)*0.45e8
        self.X = cells*m0*1e-7/20.0
        self.cells = cells; self.c = 1.0; self.A = 200.0; self.M = 1.0; self.T = T0; self.integ = 0.0
        self.dos, self.dob = 7.0, 7.0; self.rpm = 50.0
        self.mumax, self.topt, self.Ks, self.Ki, self.tau, self.k = mumax, topt, Ks, Ki, tau, k_const
        self.harvested = 0.0; self.hsum = 0.0; self.hn = 0; self.step_count = 0
        self.rng = np.random.RandomState(seed); self.drift = self.rng.uniform(0.95,1.05)
        self.od_prev = self.X/300
    @property
    def od(self): return self.X/300
    def obs(self):
        turb = 250*self.od*self.c**(-1/3)/(1+0.05*self.od)*(1+0.03*self.rpm/200*self.rng.randn())
        turb = min(max(turb,0),1000)*self.drift*self.rng.uniform(0.98,1.02)
        return {"turbidity_ntu": turb, "ph": 10.0, "pump_L": 0.0, "conductivity": 30000.0,
                "temp_c": self.T, "lux": 0.0, "t": self.step_count}
    def step(self, stir, I, frac):
        self.rpm = 0.9*self.rpm + 0.1*stir; rpm = self.rpm
        self.hsum += frac; self.hn += 1
        # temperature
        self.T += I*0.001*DT - 0.1*(self.T-25)*DT + (rpm/200)**3*0.1*DT
        err = 35.0-self.T; u_raw = 2*err+0.5*self.integ; u = min(max(u_raw,-0.6),2.0)
        if u == u_raw: self.integ += err*DT
        self.T = min(max(self.T+u*DT,15),45)
        # flocculation
        od = self.od_prev
        stick = od*0.05*max(0.1,1-rpm/250)
        shear = max(0.0,(rpm-80)/120)**2
        br = 0.5*shear*math.sqrt(self.c) + 0.005*math.sqrt(max(self.c-1,0))
        self.c = max(1.0, self.c + (stick - br)*DT)
        # biology
        fI, pm = light(I, self.X, rpm, self.c, self.Ks, self.Ki)
        self.A += DT/self.tau*(pm-self.A)
        shock = math.exp(-3e-6*max(pm-self.A,0)**2)
        fO2 = (1/3)/(1+(self.dos/35)**4) + (2/3)/(1+(self.dob/35)**4)
        rt = 1-0.35/(1+math.exp(-0.12*(rpm-100)))
        ss = min(max((rpm-80)/100,0),1)
        self.M += -ss*0.05*DT + (1-self.M)*0.1*DT; self.M = min(max(self.M,0),1)
        ft = 1-0.15*(1-self.M)
        mu = self.mumax*self.k*fI*temp_factor(self.T,self.topt)*shock*fO2*rt*ft
        resp = 0.01*self.mumax*(2 if I <= 1 else 1)
        net = mu-resp
        stress = min(max((resp-mu)/resp,0),1)
        lysis = 5e-4 + 2e-3*stress**2
        dX = self.X*(math.exp(net*DT)-1)
        metab = dX*20
        self.X = self.X*math.exp(net*DT)*(1-lysis*DT)
        self.c = max(1.0, self.c - max(net,0)*DT*(self.c-1))   # newborn cells enter at clump 1
        self.cells *= math.exp(net*DT)*(1-lysis*DT)
        # gas transfer (pre-harvest od)
        fr = 1+(self.od_prev/10)**2
        cof = 0.05*min(max(0.05,0),1)*0.1   # small CO2 feed
        tot = 0.3+cof
        mix = min(max(rpm/200,0.25),1.0); gas = min(max(tot/0.3,0.5),6)
        kLa = min(max((0.6+5*mix**1.3)*gas**0.35/fr,0.05),12)
        # harvest
        h = 0.0
        if self.step_count > 0 and self.step_count % 600 == 0:
            f = self.hsum/max(self.hn,1); self.hsum = 0; self.hn = 0
            h = self.X*20*f; self.X *= (1-f); self.cells *= (1-f)
            self.dos = self.dos*(1-f)+7*f; self.dob = self.dob*(1-f)+7*f
        self.harvested += h
        # O2 layers
        vs, vb = 20/3, 40/3
        o2sat = 8.0*(0.209*0.3/tot)/0.209
        mixk = kLa*(rpm/200)*0.5
        moved = mixk*(self.dob-self.dos)*DT*vs
        ds = self.dos + moved/vs; db = self.dob - moved/vb
        self.dos = min(max(ds + metab*1.5*(1/3)/vs + kLa*1.2*(o2sat-self.dos)*DT,0),60)
        self.dob = min(max(db + metab*1.5*(2/3)/vb + kLa*0.9*(o2sat-self.dob)*DT,0),60)
        self.od_prev = self.od
        self.step_count += 1
        self.last = dict(mu=mu, fI=fI, pm=pm, shock=shock, fO2=fO2, T=self.T, c=self.c)
        return h

def run(ctrl, n=7200, **kw):
    m = Model(**kw); hs = []
    for t in range(n):
        o = m.obs(); o["t"] = t
        s, I, f = ctrl.act(o)
        h = m.step(s, I, f)
        if h: hs.append(h)
        if m.cells < 10: return m.harvested, hs, True, m
    return m.harvested, hs, False, m
