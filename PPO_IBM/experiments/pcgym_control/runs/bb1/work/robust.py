"""(1) Open-loop steady states of my model at the actuator limits under extreme feed conditions (safety margin, reachable Ca range).
(2) Closed-loop sim with the 'plant' parameters perturbed away from the controller's model."""
import numpy as np, simlab
from controller import Controller
for caf, tf in [(1.0, 350.0), (1.02, 351.5), (0.98, 348.5), (1.02, 348.5), (0.98, 351.5)]:
    out = []
    for tc in (295.0, 298.5, 302.0):
        x = np.array([0.88, 324.0])
        for _ in range(600): x = simlab.plant_step(x, tc, caf, tf, simlab.TEXTBOOK, 2)
        out.append("Tc %.1f -> Ca %.3f T %.1f" % (tc, x[0], x[1]))
    print("Caf %.2f Tf %.1f: " % (caf, tf) + " | ".join(out), flush=True)
base = np.array(simlab.TEXTBOOK)
for name, f in [("nominal", [1, 1, 1, 1, 1]), ("k0 +10%", [1, 1.1, 1, 1, 1]), ("k0 -10%", [1, .9, 1, 1, 1]), ("alpha -10%", [1, 1, 1, 1, .9]), ("alpha +10%", [1, 1, 1, 1, 1.1]),
                ("beta +10%", [1, 1, 1, 1.1, 1]), ("a +5%", [1.05, 1, 1, 1, 1]), ("E/R +1%", [1, 1, 1.01, 1, 1])]:
    r = np.array([simlab.run_batch(Controller, None, s, tuple(base*np.array(f))) for s in range(200, 240)])
    print("%-10s mean %.4f median %.4f max %.3f Tmax %.1f" % (name, r[:, 0].mean(), np.median(r[:, 0]), r[:, 0].max(), r[:, 1].max()), flush=True)
