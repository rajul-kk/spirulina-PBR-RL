"""Cross-core recurrent state dynamics: does state stay bounded, and does the actor freeze?

The mechanism claim behind this project's reset-cadence finding is:

  * LSTM's cell state `c` is UNBOUNDED (no squashing on the state path), so over a long
    episode |c| grows, `tanh(c)` saturates, and the actor's output becomes
    state-independent -- the v33-v44 collapse family.
  * The diagonal LRU's state is BOUNDED by construction (h_t = lam*h_{t-1} + gam*Wx with
    lam in (0,1)), so it has no equivalent failure mode and does not need a tight reset.

`lru_memory_check.py` measures the LRU side only (it hard-errors on an LSTM checkpoint).
This script measures BOTH cores under the SAME protocol so the two are directly comparable,
at several reset cadences, and adds the functional consequence (actor output variance vs
steps-since-reset) that links "state grew" to "policy froze".

Pure inference on existing checkpoints -- no training, CPU-cheap.

Usage:
  python experiments/env_diagnosis/state_dynamics_check.py \
      --actor-path model_data/archive_v49_od_tail_logfix/td3_checkpoints/actor.pth \
      --reset-intervals 60,600,0 --init-cells 4000
  (reset interval 0 == free-running, no periodic reset)
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments"), os.path.join(ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from genetic_env import GeneticPhotobioreactorEnv
from TD3 import OBS_DIM, ACTION_DIM, MAX_CELLS, DEVICE
from actor_io import load_actor


def state_norms(hidden, core):
    """Return (dict of named state magnitudes, saturation fraction).

    LSTM hidden is an (h, c) tuple; `c` is the unbounded path and the one that matters.
    Saturation is measured as the fraction of units whose tanh(c) is within 1e-2 of +-1,
    i.e. units whose contribution to h can no longer respond to input.
    LRU hidden is a single bounded state tensor; tanh-saturation is not the mechanism
    there, so its saturation fraction is reported as the fraction of |h| above a high
    absolute threshold purely for comparability, and flagged as not-the-same-quantity.
    """
    if core == "lstm":
        h, c = hidden
        c_flat = c.detach().reshape(-1)
        sat = float((torch.tanh(c_flat).abs() > 0.99).float().mean())
        return {"|c|": float(c_flat.abs().max()), "|h|": float(h.detach().abs().max())}, sat
    hid = hidden.detach().reshape(-1)
    sat = float((hid.abs() > 10.0).float().mean())  # NOT tanh saturation; see docstring
    return {"|h|": float(hid.abs().max())}, sat


def rollout(actor, core, difficulty, init_cells, seed, reset_interval):
    """One deterministic episode. reset_interval<=0 means free-running (never reset)."""
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells,
                                    difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    hid = actor.initial_hidden(1)
    rec = {"since": [], "cnorm": [], "hnorm": [], "sat": [], "act": [], "obs": [], "hid": []}
    since = 0

    def snap(h):
        """Detached copy of the recurrent state as it was BEFORE this step's forward,
        so the sensitivity probe can replay that exact state."""
        if isinstance(h, tuple):
            return tuple(x.detach().clone() for x in h)
        return h.detach().clone()

    with torch.no_grad():
        while True:
            if reset_interval > 0 and since >= reset_interval:
                hid = actor.initial_hidden(1)
                since = 0
            rec["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            rec["hid"].append(snap(hid))
            ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
            at, hid = actor(ot, hid)
            norms, sat = state_norms(hid, core)
            rec["since"].append(since)
            rec["cnorm"].append(norms.get("|c|", np.nan))
            rec["hnorm"].append(norms["|h|"])
            rec["sat"].append(sat)
            rec["act"].append(at.view(-1).cpu().numpy().copy())
            obs, _, term, trunc, info = env.step(at.view(-1).cpu().numpy())
            since += 1
            if term or trunc:
                break
    for k in ("since", "cnorm", "hnorm", "sat"):
        rec[k] = np.asarray(rec[k])
    rec["act"] = np.asarray(rec["act"])
    rec["harvested"] = float(info.get("cumulative_harvested_mg", 0.0))
    rec["crashed"] = env.step_count < env.max_steps
    return rec


MIN_AGE_RANGE_FOR_VERDICT = 600  # below this, there isn't enough dynamic range to tell
                                  # a slowly-saturating curve from a linear one -- say so
                                  # explicitly rather than fit noise (this is what produced
                                  # a verdict that flipped across reset intervals before).


def boundedness_verdict(smax, ages, vals):
    """Bounded vs. unbounded, via LOCAL GROWTH RATE rather than a global curve fit.

    The earlier version fit a 2-parameter linear model and a 2-parameter saturating model
    to as few as 4 binned means and picked whichever had lower SSE. With that few points,
    and with bin edges (hence effective sample density) differing across configs, the
    winner is not a stable function of the underlying process -- it flipped between
    "linear" and "saturating" for the SAME network at different reset intervals, which is
    incoherent (boundedness is a property of the architecture, not the reset schedule).

    This version instead asks the more directly interpretable question the math actually
    turns on: is the per-step growth RATE roughly constant with age (consistent with
    unbounded/linear growth), or does it decay toward zero as the state ages (consistent
    with a bounded state approaching an asymptote)? It requires enough age range to make
    that comparison meaningful at all, and refuses a verdict otherwise.
    """
    if smax < MIN_AGE_RANGE_FOR_VERDICT:
        print(f"      boundedness: NOT ASSESSED -- max state age in this rollout ({smax}) is "
              f"below the {MIN_AGE_RANGE_FOR_VERDICT}-step minimum needed to distinguish "
              "unbounded growth from a bounded state still approaching its asymptote. "
              "(Use --reset-intervals 600,0 for this question.)")
        return
    if len(vals) < 6:
        print("      boundedness: NOT ASSESSED -- too few age bins in this rollout")
        return

    a = np.asarray(ages, dtype=float)
    y = np.asarray(vals, dtype=float)
    # Local slope between consecutive bins, normalized per 100 steps of age, so early
    # (narrow, closely-spaced) and late (wide, sparse) bins are compared on the same footing.
    local_rate = np.diff(y) / np.diff(a) * 100.0
    mid_age = 0.5 * (a[:-1] + a[1:])

    n_each = max(1, len(local_rate) // 3)
    early_rate = float(np.mean(local_rate[:n_each]))
    late_rate = float(np.mean(local_rate[-n_each:]))
    print(f"      local growth rate (per 100 steps of age): "
          f"early(age~{mid_age[:n_each].mean():.0f})={early_rate:+.3f}  "
          f"late(age~{mid_age[-n_each:].mean():.0f})={late_rate:+.3f}")

    if abs(early_rate) < 1e-6:
        print("      boundedness: NOT ASSESSED -- no measurable early growth to compare against")
        return
    decay = 1.0 - (late_rate / early_rate)
    if late_rate <= 0.05 * early_rate:
        print(f"      -> BOUNDED: growth rate has decayed {100*decay:.0f}% from early to late "
              "age, consistent with approaching an asymptote")
    elif late_rate >= 0.6 * early_rate:
        print(f"      -> UNBOUNDED (over this window): growth rate is still "
              f"{100*late_rate/early_rate:.0f}% of its early value at late age -- "
              "no sign of leveling off within the range measured")
    else:
        print(f"      -> PARTIAL DECAY ({100*late_rate/early_rate:.0f}% of early rate remains at "
              f"late age) -- slowing down but not clearly asymptoted within this window")


def by_since(rec, key, edges):
    """Mean of `key` grouped by steps-SINCE-RESET, which is the variable the mechanism
    claim is about. Indexing by absolute step is wrong under a periodic reset: successive
    absolute steps sit at arbitrary phases of the sawtooth, so both the point probes and
    any slope/drift computed across resets are meaningless."""
    s, y = rec["since"], rec[key]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        out.append((lo, hi, float(np.nanmean(y[m])) if m.any() else float("nan"), int(m.sum())))
    return out


def report(tag, core, rec, reset_interval):
    label = "free-running (never reset)" if reset_interval <= 0 else f"reset every {reset_interval}"
    n = len(rec["since"])
    smax = int(rec["since"].max())
    print(f"\n  {tag} | core={core} | {label} | {n} steps | max steps-since-reset={smax}")

    key = "cnorm" if core == "lstm" else "hnorm"
    name = "|c| (cell state, UNBOUNDED path)" if core == "lstm" else "|h| (state, bounded by constr.)"

    # Bin by steps-since-reset. Within a window the state should grow with age if the
    # architecture is unbounded, and plateau if it is bounded.
    edges = [e for e in (0, 5, 15, 30, 60, 150, 300, 600, 1200, 3000, 7201) if e <= max(smax + 1, 2)]
    if edges[-1] <= smax:
        edges.append(smax + 1)
    print(f"    {name}  BY STEPS-SINCE-RESET (mean within bin)")
    rows = by_since(rec, key, edges)
    print("      " + "  ".join(f"[{lo}-{hi-1}]={v:.2f}" for lo, hi, v, c in rows if c > 0))

    vals = [v for lo, hi, v, c in rows if c > 0 and np.isfinite(v)]
    ages = [0.5 * (lo + hi - 1) for lo, hi, v, c in rows if c > 0 and np.isfinite(v)]
    if len(vals) >= 2:
        print(f"      youngest-bin={vals[0]:.2f}  oldest-bin={vals[-1]:.2f}  "
              f"ratio={vals[-1]/max(vals[0],1e-9):.2f}")

    boundedness_verdict(smax, ages, vals)

    if core == "lstm":
        print("    tanh(c) saturation fraction BY STEPS-SINCE-RESET")
        srows = by_since(rec, "sat", edges)
        print("      " + "  ".join(f"[{lo}-{hi-1}]={v:.3f}" for lo, hi, v, c in srows if c > 0))
        svals = [v for lo, hi, v, c in srows if c > 0 and np.isfinite(v)]
        if len(svals) >= 2:
            print(f"      youngest={svals[0]:.3f} -> oldest={svals[-1]:.3f}  "
                  f"({'SATURATING WITH AGE' if svals[-1] > svals[0] + 0.05 else 'stable'})")

    # DELIBERATELY NOT REPORTED: raw action-std young-vs-old. It was tried and discarded --
    # it fires "freezing" for every core at every reset interval, including configurations
    # that harvest >1000mg perfectly well, because most of the variation it measures is the
    # ENVIRONMENT's own trajectory calming down later in an episode, not the policy losing
    # responsiveness. See the sensitivity probe below, which controls for that by holding the
    # observation fixed and perturbing only the recurrent state.
    print(f"    episode: harvested={rec['harvested']:.1f}mg  crashed={rec['crashed']}")


def sensitivity_probe(actor, core, rec, reset_interval, n_probe=24, eps=0.05):
    """Does the actor still RESPOND to its input once the state is old?

    Confound-free version of the freeze test: replay recorded observations, but at matched
    young vs old state ages, perturb the OBSERVATION by a fixed relative epsilon and measure
    how much the action moves. A frozen/saturated actor responds ~0 regardless of input; a
    responsive one moves. Because the observation perturbation is identical in both age
    groups, environment trajectory differences cannot explain a gap.
    """
    s = rec["since"]
    smax = int(s.max())
    if reset_interval > 0:
        young_idx = np.where(s < max(2, int(reset_interval * 0.15)))[0]
        old_idx = np.where(s >= max(1, int(reset_interval * 0.80)))[0]
    else:
        young_idx = np.where(s < 200)[0]
        old_idx = np.where(s >= max(1, smax - 200))[0]
    if len(young_idx) < 5 or len(old_idx) < 5:
        return
    rng = np.random.default_rng(0)
    out = {}
    for label, idxs in (("young", young_idx), ("old", old_idx)):
        pick = rng.choice(idxs, size=min(n_probe, len(idxs)), replace=False)
        deltas = []
        with torch.no_grad():
            for i in pick:
                obs = rec["obs"][i]
                hid = rec["hid"][i]
                ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
                a0, _ = actor(ot, hid)
                pert = obs * (1.0 + eps * rng.choice([-1.0, 1.0], size=obs.shape))
                pt = torch.tensor(pert, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
                a1, _ = actor(pt, hid)
                deltas.append(float((a1 - a0).abs().mean()))
        out[label] = float(np.mean(deltas))
    ratio = out["old"] / max(out["young"], 1e-12)
    print(f"    input-sensitivity (|dAction| to a +-{100*eps:.0f}% obs perturbation, state held)")
    print(f"      young state={out['young']:.6f}   old state={out['old']:.6f}   ratio={ratio:.3f}")
    print("      -> " + ("RESPONSIVENESS LOST with state age (frozen)" if ratio < 0.25
                         else "still responsive at old state" if ratio > 0.6
                         else "partially reduced responsiveness"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-path", required=True)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--init-cells", type=int, default=4000,
                    help="high population by default -- where the collapses actually happened")
    ap.add_argument("--seed", type=int, default=900_001)
    ap.add_argument("--reset-intervals", default="60,600,0",
                    help="comma-separated; 0 means free-running")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    actor, core = load_actor(args.actor_path, OBS_DIM, ACTION_DIM, DEVICE)
    tag = args.tag or os.path.basename(os.path.dirname(os.path.dirname(args.actor_path)))

    print("=" * 92)
    print(f"  STATE DYNAMICS CHECK   {args.actor_path}")
    print(f"  core={core}  D{args.difficulty}  init_cells={args.init_cells}  seed={args.seed}")
    print("=" * 92)

    for ri in [int(x) for x in args.reset_intervals.split(",")]:
        rec = rollout(actor, core, args.difficulty, args.init_cells, args.seed, ri)
        report(tag, core, rec, ri)
        sensitivity_probe(actor, core, rec, ri)

    print("\n  Interpretation")
    print("    LSTM: |c| slope clearly >0 and rising saturation fraction over a window is the")
    print("          unbounded-growth mechanism; a short reset caps it, a long one does not.")
    print("    LRU : |h| slope ~0 and flat drift is boundedness by construction; a short reset")
    print("          then buys nothing and only injects a recurring cold-start transient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
