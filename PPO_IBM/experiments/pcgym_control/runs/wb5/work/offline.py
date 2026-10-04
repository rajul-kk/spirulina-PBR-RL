import sys, time, json, importlib.util, numpy as np
from mysim import evaluate, run
spec=importlib.util.spec_from_file_location("c", sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
params=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
n=int(sys.argv[3]) if len(sys.argv)>3 else 20
t=time.time()
c,ra=evaluate(lambda: m.Controller(params), range(n))
print("mean %.3f median %.3f max %.3f runaways %d  (%.1fs)"%(c.mean(),np.median(c),c.max(),ra,time.time()-t))
print(np.round(c,2))
