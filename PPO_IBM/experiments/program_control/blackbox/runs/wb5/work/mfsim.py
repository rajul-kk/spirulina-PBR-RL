# Mean-field planning model of the reactor, transcribed from genetic_env.py (own code, no simulator import).
import math, numpy as np
from lightmodel import fI, tax
def tfac(T, topt=36.0, tmin=10., tmax=44.5):
    num=(T-tmax)*(T-tmin)**2; den=(topt-tmin)*((topt-tmin)*(T-topt)-(topt-tmax)*(topt+tmin-2*T)); return max(0,min(1,num/den))
def run(policy, od0, mu_eff=0.031, topt=36.0, dt=0.1, trace=False):
    """policy(t_h, od, c, k_event_next) -> (rpm, I, h_for_event). h used at each 12h event (last value in interval)."""
    od, c, T = od0, 1.0, 35.0; tot = 0.0; tr = []; rpm_s = 50.0; mi = 1.0
    hsum = 0.0; hn = 0
    n = int(round(144/dt)); ev = int(round(12/dt))
    for i in range(n):
        t = i*dt
        rpm, I, h = policy(t, od, c)
        hsum += h; hn += 1
        rpm_s += (rpm - rpm_s) * (1 - 0.9 ** (dt/0.02))
        # temperature: thermostat holds 35 unless light heat exceeds chiller
        Teq = 25 + 10*(0.001*I + (rpm_s/200)**3*0.1 - 0.6)
        Tt = max(35.0, Teq); T += (Tt - T) * min(1, dt*0.3)
        f, _ = fI(od*300, I, rpm_s, c)
        s = min(max((rpm_s-80)/100,0),1); mi += (-s*0.05 + (1-mi)*0.1)*dt; mi = min(max(mi,0),1)
        mu = mu_eff * f * tax(rpm_s) * (1-0.15*(1-mi)) * tfac(T, topt)
        # O2 inhibition: DO ~ 8 + prod*1.5/kLa
        kla = (0.6+5*min(max(rpm_s/200,0.25),1)**1.3)*1.05/(1+(od/10)**2)
        do2 = 8 + mu*od*300*1.5/kla
        mu *= 1/(1+(do2/35)**4)
        od *= math.exp(mu*dt)
        stick = od*0.05*max(0.1,1-rpm_s/250)
        br = 0.5*max(0,(rpm_s-80)/120)**2*math.sqrt(c) + 0.005*math.sqrt(max(c-1,0))
        c = max(1.0, c + (stick - br - mu*(c-1))*dt)
        if (i+1) % ev == 0 and i+1 < n:
            hf = hsum/hn; hsum = 0; hn = 0
            tot += od*hf*6000; od *= (1-hf)
        if trace and i % 60 == 0: tr.append((round(t,1), round(od,3), round(c,2), round(mu,4), round(T,1)))
    return (tot, od, tr) if trace else tot
