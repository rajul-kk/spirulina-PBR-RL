"""Where does the cost come from? Split per-sample squared error by phase (own simulator)."""
import sys
import numpy as np
import sim
if __name__ == "__main__":
    path, n, seed0, oracle = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
    Ctrl = sim.load_controller(path)
    tot = {"startup": 0.0, "after_sp_step": 0.0, "after_feed_step": 0.0, "rest": 0.0}
    W = 12
    for s in range(seed0, seed0 + n):
        sc = sim.scenario(s)
        r = sim.run_batch(Ctrl, s, None, oracle, trace=True)
        lab = np.array(["rest"] * 120, dtype=object)
        for name, key in (("after_feed_step", "Ti"), ("after_feed_step", "Caf"), ("after_sp_step", "sp")):
            for c in np.where(np.diff(sc[key]) != 0)[0] + 1:
                lab[c:c + W] = name
        lab[:W] = "startup"
        for k in tot:
            tot[k] += r["sq"][lab == k].sum() / 120
    for k, v in tot.items():
        print("%-16s %.4f" % (k, v / n))
    print("total            %.4f" % (sum(tot.values()) / n))
