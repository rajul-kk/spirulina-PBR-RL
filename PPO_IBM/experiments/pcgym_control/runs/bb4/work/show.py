import sys, glob, numpy as np
def load(f):
    return np.genfromtxt(f, delimiter=',', names=True)
if __name__ == '__main__':
    for pat in sys.argv[1:]:
        for f in sorted(glob.glob('runs/bb4/trials/' + pat)):
            d = load(f)
            print(f)
            sp = d['Ca_sp']; ch = [0] + [i for i in range(1, len(sp)) if sp[i] != sp[i-1]]
            print('  sp changes at steps', ch, 'values', [round(float(sp[i]), 4) for i in ch])
            for i in range(0, len(sp), 6):
                print('  %3d Ca %.4f T %.2f sp %.4f Tc %.2f' % (i, d['Ca'][i], d['T'][i], sp[i], d['Tc'][i]))
