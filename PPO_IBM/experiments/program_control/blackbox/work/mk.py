# generate constant-actuator controller files: python mk.py name stir light h
import sys
name, s, l, h = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
open(name + '.py', 'w').write(f'''class Controller:
    def __init__(self, params=None): pass
    def act(self, obs):
        return ({s}, {l}, {h})
''')
