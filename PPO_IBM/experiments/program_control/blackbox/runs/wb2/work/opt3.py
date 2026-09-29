import itertools, sys
from opt1 import P2, simulate
class P3(P2):
    def __init__(self, burst_P=0, burst_B=0, burst_rpm=200, od_min=0.8, **kw):
        super().__init__(**kw); self.bP, self.bB, self.brpm, self.odmin = burst_P, burst_B, burst_rpm, od_min
    def __call__(self, t, X, C, T):
        rpm, L, f = super().__call__(t, X, C, T)
        h = t*0.02
        if self.bP and X/300 > self.odmin and (h % self.bP) < self.bB:
            rpm = self.brpm
        return rpm, L, f
if __name__ == '__main__':
  for n0 in [200, 1500]:
    for bP, bB, brpm in [(0,0,200), (12,0.5,200), (12,1,200), (24,1,200), (24,2,200), (12,1,150), (6,0.5,200)]:
        tot, _ = simulate(P3(burst_P=bP, burst_B=bB, burst_rpm=brpm, od_sp=2.0, k_final=3), n0=n0, mu_max=0.036)
        print(n0, bP, bB, brpm, f"{tot:.0f}", flush=True)
