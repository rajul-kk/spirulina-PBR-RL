"""PC-Gym tasks for the controller-writing comparison (docs/reports/pcgym_protocol.md).

Each task wraps a stock PC-Gym model (Bloor et al., "PC-Gym: Benchmark Environments for Process
Control Problems") with a seeded scenario: setpoint schedule, unmeasured feed disturbances,
initial state and sensor noise. A controller sees only the noisy measurements and the current
setpoint; the dynamics are PC-Gym's, unmodified.

cstr: exothermic CSTR. Measured Ca (mol/L) and T (K); manipulated jacket temperature Tc
(295-302 K); setpoint on Ca. Unmeasured step disturbances in feed temperature Ti and feed
concentration Caf. Hot feed with a hot jacket can ignite the reactor (thermal runaway).
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from pcgym import make_env  # noqa: E402

TASKS = {}


class CSTR:
    name = "cstr"
    N, TSIM = 120, 26.0                 # steps, minutes (PC-Gym's standard CSTR horizon)
    A_LOW, A_HIGH = np.array([295.0]), np.array([302.0])
    SP_RANGE = (0.86, 0.90)
    TI_RANGE, CAF_RANGE = (348.5, 351.5), (0.98, 1.02)
    NOISE = {"Ca": 0.002, "T": 0.2}     # sensor standard deviations
    T_RUNAWAY = 335.0                   # K; above this the batch counts as a runaway
    ERR_SCALE = 0.01                    # mol/L; cost is the mean of (error / ERR_SCALE)^2

    @staticmethod
    def _steps(rng, n, lo, hi, n_changes, min_len=20):
        """Piecewise-constant signal with n_changes step changes at least min_len apart."""
        while True:
            cuts = np.sort(rng.randint(min_len, n - min_len, n_changes))
            if np.all(np.diff(np.concatenate([[0], cuts, [n]])) >= min_len):
                break
        levels = rng.uniform(lo, hi, n_changes + 1)
        return np.repeat(levels, np.diff(np.concatenate([[0], cuts, [n]])))

    def scenario(self, seed):
        rng = np.random.RandomState(seed)
        return {"sp": self._steps(rng, self.N, *self.SP_RANGE, 2),
                "Ti": self._steps(rng, self.N, *self.TI_RANGE, rng.randint(1, 3)),
                "Caf": self._steps(rng, self.N, *self.CAF_RANGE, rng.randint(1, 3)),
                "x0": np.array([rng.uniform(0.85, 0.91), rng.uniform(320.0, 326.0)]),
                "noise_seed": int(rng.randint(1 << 30))}

    def start(self, seed):
        """Begin an episode; returns an Episode with .obs(), .step(u) and .summary()."""
        return _CSTREpisode(self, seed)

    def run(self, controller, seed, privileged=False):
        """Run one episode with a controller object. Returns (summary dict, trace rows)."""
        ep = self.start(seed)
        trace, err = [], None
        while not ep.done:
            obs = ep.obs()
            try:
                u = float(np.asarray(controller.act(dict(obs)), dtype=float).reshape(-1)[0])
                if not np.isfinite(u):
                    raise ValueError("non-finite action")
            except Exception as e:  # a controller error forfeits control: jacket held mid-range
                err = err or f"{type(e).__name__}: {e}"
                u = 298.5
            u = ep.step(u)
            row = {"step": ep.k - 1, **obs, "Tc": u}
            if privileged:
                row.update(ep.truth())
            trace.append(row)
        return {**ep.summary(), "error": err}, trace


class _CSTREpisode:
    def __init__(self, task, seed):
        self.task, self.seed = task, seed
        sc = self.sc = task.scenario(seed)
        self.env = make_env({
            "N": task.N, "tsim": task.TSIM, "SP": {"Ca": list(sc["sp"])}, "model": "cstr_ode",
            "o_space": {"low": np.array([0.0, 250.0, 0.0, 300.0, 0.5]),
                        "high": np.array([2.0, 450.0, 2.0, 400.0, 1.5])},
            "a_space": {"low": task.A_LOW, "high": task.A_HIGH},
            "x0": np.array([sc["x0"][0], sc["x0"][1], sc["sp"][0]]),
            "normalise_a": False, "normalise_o": False, "noise": False,
            "integration_method": "casadi",
            "disturbances": {"Ti": sc["Ti"], "Caf": sc["Caf"]},
            "disturbance_bounds": {"low": np.array([300.0, 0.5]), "high": np.array([400.0, 1.5])}})
        self.state, _ = self.env.reset()
        self.nrng = np.random.RandomState(sc["noise_seed"])
        self.k, self.cost, self.t_max, self.done = 0, 0.0, float(self.state[1]), False
        self.last_sq = 0.0

    def obs(self):
        t = self.task
        return {"t_min": self.k * t.TSIM / t.N,
                "Ca": float(self.state[0] + self.nrng.normal(0, t.NOISE["Ca"])),
                "T": float(self.state[1] + self.nrng.normal(0, t.NOISE["T"])),
                "Ca_sp": float(self.sc["sp"][self.k])}

    def step(self, u):
        """Apply a jacket temperature (clipped to the actuator range); returns the applied value."""
        t = self.task
        u = float(np.clip(u, t.A_LOW[0], t.A_HIGH[0]))
        self.state, _, _, _, _ = self.env.step(np.array([u]))
        self.last_sq = ((self.state[0] - self.sc["sp"][self.k]) / t.ERR_SCALE) ** 2
        self.cost += self.last_sq
        self.t_max = max(self.t_max, float(self.state[1]))
        self.k += 1
        self.done = self.k >= t.N
        return u

    def truth(self):
        k = self.k - 1
        return {"Ca_true": float(self.state[0]), "T_true": float(self.state[1]),
                "Ti": float(self.sc["Ti"][k]), "Caf": float(self.sc["Caf"][k])}

    def summary(self):
        return {"seed": self.seed, "cost": float(self.cost / self.task.N),
                "runaway": bool(self.t_max > self.task.T_RUNAWAY), "t_max": self.t_max}


TASKS["cstr"] = CSTR()
