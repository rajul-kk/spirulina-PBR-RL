import math

class Controller:
    # tunables
    STIR = 60.0
    L_LOW = 600.0      # light at very low density
    X_LOW = 40.0       # mg/L below which L_LOW is used
    X_FULL = 200.0     # mg/L at/above which full light is used
    L_MAX = 2000.0
    X_HOLD = 1300.0    # pre-harvest density to hold during the growth phase
    TH96 = 700.0       # start end-game at the 96 h harvest if X above this
    TH108 = 350.0
    TH120 = 0.0

    def __init__(self, params=None):
        if params:
            for k, v in params.items():
                setattr(self, k, v)
        self.xf = None
        self.tf = None

    def ratio(self, h):
        # turbidity NTU per mg/L dry weight, falls as filaments clump
        return max(0.58, 0.78 - 0.0017 * h)

    def act(self, obs):
        h = obs['t'] * 0.02
        turb = obs['turbidity_ntu']
        x = turb / self.ratio(h)
        if self.xf is None:
            self.xf = x
            self.tf = obs['temp_c']
        else:
            self.xf += 0.05 * (x - self.xf)
            self.tf += 0.02 * (obs['temp_c'] - self.tf)
        X = self.xf
        # light schedule by density
        if X <= self.X_LOW:
            light = self.L_LOW
        elif X >= self.X_FULL:
            light = self.L_MAX
        else:
            a = (X - self.X_LOW) / (self.X_FULL - self.X_LOW)
            light = self.L_LOW + a * (self.L_MAX - self.L_LOW)
        # harvest: which harvest is next (at 12k h)
        k = int(h // 12) + 1
        hn = 12 * k
        if hn >= 120:
            f = 0.5
        elif hn == 108:
            f = 0.5 if X >= self.TH108 else 0.0
        elif hn == 96:
            f = 0.5 if X >= self.TH96 else 0.0
        else:
            f = 0.0
        if f < 0.5 and X > self.X_HOLD:
            f = min(0.5, 1.0 - self.X_HOLD / X)
        return (self.STIR, light, f)
