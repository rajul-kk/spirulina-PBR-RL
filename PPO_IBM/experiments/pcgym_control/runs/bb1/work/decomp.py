import simlab, sys
mod = sys.argv[1] if len(sys.argv) > 1 else "controller"
for name, o in [("all", {}), ("no noise", dict(noise=0.0)), ("no dist steps", dict(dist=False)), ("no sp change", dict(spchange=False)),
                ("only noise", dict(dist=False, spchange=False, init=False)), ("nothing", dict(noise=0.0, dist=False, spchange=False, init=False))]:
    simlab.OPTS.update(dict(noise=1.0, dist=True, spchange=True, init=True)); simlab.OPTS.update(o)
    print("%-14s mean %.3f median %.3f max %.3f Tmax %.1f (%.2fs)" % ((name,) + simlab.evaluate(mod, None, range(40))), flush=True)
