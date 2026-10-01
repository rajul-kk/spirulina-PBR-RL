"""Innovation statistics of the controller's own EKF in my simulator (for comparison with innov.py on plant data)."""
import numpy as np, simlab
from controller import Controller
class Spy(Controller):
    def _update(self, ca_m, t_m):
        self.log.append((ca_m - self.x[0], t_m - self.x[1])); Controller._update(self, ca_m, t_m)
I = []
for s in range(700, 730):
    rng = np.random.default_rng(s); sp, caf, tf, x = simlab.scenario(rng); c = Spy(dict(r_ca=0.0021, r_t=0.21)); c.log = []
    for k in range(simlab.N):
        u = float(np.clip(c.act(dict(t_min=0, Ca=x[0]+rng.normal(0, simlab.SIG_CA), T=x[1]+rng.normal(0, simlab.SIG_T), Ca_sp=sp[k])), 295, 302))
        x = simlab.plant_step(x, u, caf[k], tf[k], simlab.TEXTBOOK)
    I += c.log[10:]
I = np.array(I)
print("sim: Ca innov std %.5f T innov std %.3f; lag-1 autocorr Ca %.3f T %.3f" % (I[:, 0].std(), I[:, 1].std(), np.corrcoef(I[:-1, 0], I[1:, 0])[0, 1], np.corrcoef(I[:-1, 1], I[1:, 1])[0, 1]))
