"""Measurement-noise check across batches: robust sigma from second differences, per batch; tail weight."""
import glob, sys, numpy as np
sc, st, allc, allt = [], [], [], []
for f in sorted(glob.glob(f"../trials/b*_{sys.argv[1]}.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True)
    dc = np.diff(d["Ca"], 2)/6**.5; dt = np.diff(d["T"], 2)/6**.5
    sc.append(dc.std()); st.append(dt.std()); allc += list(dc); allt += list(dt)
sc = np.array(sc); allc = np.array(allc)
print("per-batch Ca sigma (2nd diff): min %.4f median %.4f max %.4f" % (sc.min(), np.median(sc), sc.max()))
print("sorted:", np.sort(sc).round(4))
print("pooled: std %.4f, MAD-sigma %.4f, kurtosis %.2f, frac >3sigma %.4f" % (allc.std(), 1.4826*np.median(np.abs(allc-np.median(allc))), ((allc-allc.mean())**4).mean()/allc.var()**2, np.mean(np.abs(allc) > 3*allc.std())))
