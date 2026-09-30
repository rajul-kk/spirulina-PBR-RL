from model import *
import itertools, sys

def make_policy(Xt, n_end, rpm_lo, rpm_hi, x_sw, Imax=1800, ramp=150.0):
    st = {"I": 300.0, "last": -1}
    def I_of(X):
        # light schedule from steady optimum table, capped
        return min(Imax, 500 + 6.0 * X) if X < 225 else Imax
    def pol(t, X, c):
        k = int(round(t / 0.02))
        target = I_of(X)
        dI = target - st["I"]
        st["I"] += max(-ramp * 0.02 * 5, min(ramp * 0.02, dI))
        rpm = rpm_lo if X < x_sw else rpm_hi
        ev = k // 600 + 1          # upcoming event index (1..11)
        if ev > 11: return rpm, st["I"], 0.0
        if ev > 11 - n_end: return rpm, st["I"], 0.5
        # predict X at the event from time remaining (approx growth 0.3/12h low X)
        rem = (ev * 600 - k) * 0.02
        Xp = X + rem * 9.0 * min(1.0, X / 500)
        f = min(0.5, max(0.0, (Xp - Xt) / Xp))
        return rpm, st["I"], f
    return pol

if __name__ == "__main__":
    for X0 in [19, 62, 125, 250, 1000, 2750]:
        res = []
        for Xt, n_end, (rl, rh, xs) in itertools.product(
                [300, 450, 600, 800, 1000, 1300], [1, 2, 3, 4],
                [(80, 80, 1e9), (70, 130, 700), (80, 150, 800), (100, 100, 1e9), (60, 130, 500)]):
            tot, Xf = batch(make_policy(Xt, n_end, rl, rh, xs), X0)
            res.append((tot, Xt, n_end, rl, rh, xs))
        res.sort(reverse=True)
        print("X0", X0, ["%.0f Xt%d end%d rpm%d/%d@%g" % r for r in res[:4]])
        # fixed reference
        print("   ref Xt600 end2 80/150@800:", "%.0f" % batch(make_policy(600, 2, 80, 150, 800), X0)[0],
              " Xt225 end1 80:", "%.0f" % batch(make_policy(225, 1, 80, 80, 1e9), X0)[0])
