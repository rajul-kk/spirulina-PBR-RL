import numpy as np, glob, json, os
TR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trials')
def load(pattern='*'):
    out = {}
    for f in sorted(glob.glob(os.path.join(TR, 'b*_%s.csv' % pattern))):
        d = np.genfromtxt(f, delimiter=',', names=True)
        out[os.path.basename(f)[:-4]] = d
    return out
def results():
    return [json.loads(l) for l in open(os.path.join(TR, 'results.jsonl'))]
