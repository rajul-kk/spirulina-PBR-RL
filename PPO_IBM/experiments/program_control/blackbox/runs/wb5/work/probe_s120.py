import math
P = dict(stir=120.0, imax=1700.0, target=3.0, hfrac=0.15, endgame_t=5400)
class Controller:
    def __init__(self, params=None):
        self.p = dict(P); self.p.update(params or {})
        self.od = None; self.light = 400.0
    def act(self, obs):
        p = self.p; t = obs['t']
        turb = obs['turbidity_ntu']
        raw = turb / 250.0
        raw = raw / max(1e-3, 1 - 0.05 * raw) if raw < 15 else 20
        self.od = raw if self.od is None else self.od + 0.02 * (raw - self.od)
        od = self.od
        pts = [(0.1, 600), (0.25, 800), (0.5, 1100), (0.75, 1400), (1.0, p['imax'])]
        if od <= pts[0][0]: tgt = pts[0][1]
        elif od >= pts[-1][0]: tgt = pts[-1][1]
        else:
            for (a, x), (b, y) in zip(pts, pts[1:]):
                if a <= od <= b: tgt = x + (y - x) * (od - a) / (b - a); break
        step = 100.0 * 0.02
        self.light = min(tgt, self.light + step) if tgt > self.light else tgt
        k = t % 600
        if t >= p['endgame_t']:
            h = 0.5
        else:
            h = p['hfrac'] if od > p['target'] else 0.0
        return p['stir'], self.light, h
