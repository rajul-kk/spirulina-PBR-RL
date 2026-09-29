PHASE = 1
HARV = 0.0
class Controller:
    def __init__(self, params=None):
        pass
    def act(self, obs):
        t = int(obs.get('t', 0))
        blk = (t // 600 + PHASE) % 2
        return (50.0 if blk == 0 else 200.0, 1800.0, HARV)
