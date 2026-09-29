# python mksched.py name blockhours "s1,l1,h1;s2,l2,h2;..."  -> piecewise-constant schedule controller
import sys
name, bh, spec = sys.argv[1], float(sys.argv[2]), sys.argv[3]
blocks = [tuple(float(x) for x in b.split(',')) for b in spec.split(';')]
open(name + '.py', 'w').write(f'''class Controller:
    B = {blocks!r}
    def __init__(self, params=None): pass
    def act(self, obs):
        k = int(obs['t'] * 0.02 // {bh})
        return self.B[min(k, len(self.B) - 1)]
''')
