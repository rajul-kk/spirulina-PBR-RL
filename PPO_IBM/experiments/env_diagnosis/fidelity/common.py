"""Shared helpers for the fidelity audit scripts (2026-09-29): env construction, a ledger subclass
that books biomass and nitrogen through every process, and an episode runner that records extremes."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (ROOT, os.path.join(ROOT, "environments"), os.path.join(ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from genetic_env import GeneticPhotobioreactorEnv

MAX_CELLS = 7500   # TD3.MAX_CELLS / harness MAX_CELLS


def act(stir, light, frac, f_max=0.5):
    """Physical (rpm, umol, fraction) -> env action in [-1, 1]."""
    return np.array([np.interp(stir, [50, 200], [-1, 1]),
                     np.interp(light, [0, 2000], [-1, 1]),
                     np.interp(frac, [0, f_max], [-1, 1])], dtype=np.float32)


def expert(stir=65.0, light=1400.0, setpoint=0.6, cap=0.30):
    """The TD3 scripted expert law (td3/TD3.py expert_harvest_frac), reading the true od."""
    def pol(env, t):
        frac = float(np.clip(env.od / setpoint - 1.0, 0.0, cap))
        return act(stir, light, frac)
    return pol


def const(stir, light, frac=0.0):
    a = act(stir, light, frac)
    return lambda env, t: a


class LedgerEnv(GeneticPhotobioreactorEnv):
    """Books biomass (mg) and nitrogen (mg) through every process. Physics unchanged."""

    def reset(self, *a, **k):
        out = super().reset(*a, **k)
        self._ctx = None
        self.led = dict(bio_net=0.0, lysed=0.0, harvested=0.0, n_dosed=0.0, n_drawn=0.0,
                        n_harv_liquid=0.0, n_refill=0.0, cap_agent_steps=0, B0=self._B(),
                        N0_pool=self.n_pool * self.volume_L)
        return out

    def _B(self):
        return float(np.sum(self.cells_mass[self.active_mask])) * self.MG_PER_MASS_UNIT

    def _remove_cells(self, idx):
        if self._ctx == "bio":
            self.led["lysed"] += float(np.sum(self.cells_mass[idx])) * self.MG_PER_MASS_UNIT
        super()._remove_cells(idx)

    def _update_biology(self, I, rpm, mix, nut):
        B0, n0 = self._B(), self.n_pool
        self._ctx = "bio"
        dosed, shock = super()._update_biology(I, rpm, mix, nut)
        self._ctx = None
        self.led["bio_net"] += self._B() - B0          # growth - respiration - removals - cap
        dN = dosed * self.DOSE_N_FRAC
        self.led["n_dosed"] += dN
        self.led["n_drawn"] += (n0 - self.n_pool) * self.volume_L + dN
        self.led["cap_agent_steps"] += int(np.sum(self.cells_mass[self.active_mask] >= 5e8 * 0.9999))
        return dosed, shock

    def _apply_harvest(self):
        n_before = self.n_pool
        h, ev = super()._apply_harvest()
        self.led["harvested"] += h
        fresh = self.FRESH_MEDIUM["n_pool"]
        if ev and self.n_pool != n_before and n_before != fresh:
            f = (n_before - self.n_pool) / (n_before - fresh)
            self.led["n_harv_liquid"] += n_before * f * self.volume_L
            self.led["n_refill"] += fresh * f * self.volume_L
        return h, ev


TRACKED = ("od", "temp", "ph", "do2", "dissolved_co2", "conductivity", "salt", "n_pool", "p_pool",
           "alkalinity", "dic", "pigment", "fouling_factor", "kLa", "membrane_integrity")
EXTRA = ("clump", "cells", "obs_turb", "obs_lux", "obs_cond", "max_agent_mass", "mu")


def run(policy, init_cells=700, difficulty=2, seed=1, steps=7200, env_cls=LedgerEnv, env_kwargs=None,
        setup=None, record_every=0, max_cells=MAX_CELLS):
    """policy(env, t) -> env action. Returns extremes, NaN/negative flags, ledger and trace."""
    np.random.seed(seed)
    env = env_cls(max_cells=max_cells, initial_cells=init_cells, difficulty=difficulty, **(env_kwargs or {}))
    obs, _ = env.reset(seed=seed)
    if setup:
        setup(env)
        if hasattr(env, "led"):
            env.led["N0_pool"] = env.n_pool * env.volume_L   # setup may change the starting pools
    ext = {k: [np.inf, -np.inf] for k in TRACKED + EXTRA}
    bad, trace, crash_t, turb_sat, vals = [], [], None, 0, {}
    t = 0
    for t in range(steps):
        obs, r, term, trunc, info = env.step(policy(env, t))
        vals = {k: float(getattr(env, k)) for k in TRACKED}
        am = env.active_mask
        vals["clump"] = float(np.mean(env.clump_mass[am])) if env.num_active else 1.0
        vals["cells"] = float(env.num_active)
        vals["obs_turb"], vals["obs_lux"], vals["obs_cond"] = float(obs[0]), float(obs[5]), float(obs[3])
        vals["max_agent_mass"] = float(np.max(env.cells_mass[am])) if env.num_active else 0.0
        vals["mu"] = float(env.debug_mu)
        turb_sat += int(obs[0] >= 999.0)
        for k, v in vals.items():
            ext[k][0] = min(ext[k][0], v)
            ext[k][1] = max(ext[k][1], v)
            if not np.isfinite(v) or (v < 0 and k != "mu"):
                bad.append((t, k, v))
        if not (np.all(np.isfinite(obs)) and np.isfinite(r)):
            bad.append((t, "obs/reward", float(r)))
        if record_every and t % record_every == 0:
            trace.append(dict(t_h=round(t * env.dt, 2), **{k: round(v, 4) for k, v in vals.items()}))
        if term or trunc:
            crash_t = t if term else None
            break
    return dict(env=env, ext=ext, bad=bad[:10], n_bad=len(bad), crash_t=crash_t,
                harvest_mg=float(env.cumulative_harvested_mg), trace=trace,
                turb_sat_frac=turb_sat / (t + 1), steps=t + 1, final=vals,
                ledger=getattr(env, "led", None))


def fmt_ext(ext, keys):
    return "  ".join(f"{k}[{ext[k][0]:.4g},{ext[k][1]:.4g}]" for k in keys)


def biomass_balance(res):
    """Closing error of B_end = B0 + bio_net(already net of lysis/cap) - harvested."""
    env, L = res["env"], res["ledger"]
    B_end = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    return B_end, L["B0"] + L["bio_net"] - L["harvested"]
