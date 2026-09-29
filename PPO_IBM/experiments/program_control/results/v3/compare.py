"""Paired comparison of the v3 writer tracks and baselines on the held-out test split.

Every controller ran the same 52 held-out episodes (seed + inoculum), so each comparison is
paired per episode (yield-scored episodes only, init > 80 cells). The bootstrap CI is over
episodes; run-to-run spread within a track is reported separately.

  python compare.py
"""
import json

import numpy as np


def eps(f):
    d = json.load(open(f))
    return {(x["seed"], x["init_cells"]): x["harvested_mg"] for x in d["episodes"] if x["init_cells"] > 80}


def boot_ci(d, rng, n=20000):
    b = [np.mean(rng.choice(d, len(d))) for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tracks = {"black-box": [eps(f"test_bb{i}.json") for i in (1, 2, 3)],
              "white-box": [eps(f"test_wb{i}.json") for i in (1, 2, 3)]}
    base = {"hand-tuned expert": eps("test_sensor_expert.json"), "CMA-ES (v3)": eps("test_cmaes_v3.json"),
            "oracle expert": eps("test_oracle.json")}
    keys = sorted(tracks["black-box"][0])
    T = {k: np.array([[r[e] for e in keys] for r in v]) for k, v in tracks.items()}
    for k, v in T.items():
        print(f"{k}: per-run medians {[round(float(np.median(r)) / 1000, 2) for r in v]} g")
    pairs = [("white-box", T["white-box"].mean(0), "black-box", T["black-box"].mean(0))]
    for bn, b in base.items():
        x = np.array([b[e] for e in keys])
        pairs += [(k, T[k].mean(0), bn, x) for k in T]
    for a, xa, b, xb in pairs:
        d = xa - xb
        lo, hi = boot_ci(d, rng)
        print(f"{a} minus {b}: {d.mean() / 1000:+.2f} g per episode [95% CI {lo / 1000:+.2f}, {hi / 1000:+.2f}], "
              f"better in {np.mean(d > 0):.0%} of {len(d)} episodes")
