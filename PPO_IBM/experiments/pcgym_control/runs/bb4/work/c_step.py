class Controller:
    """Open-loop identification: pseudo-random multi-level jacket steps (deterministic sequence)."""
    SEQ = [298.5, 302.0, 295.0, 300.0, 296.5, 302.0, 298.5, 295.0, 301.0, 297.0, 299.5, 295.0]
    HOLD = [8, 14, 12, 6, 12, 10, 8, 14, 6, 10, 12, 8]
    def __init__(self, params=None):
        self.k = 0
        self.sched = []
        for v, h in zip(self.SEQ, self.HOLD):
            self.sched += [v] * h
    def act(self, obs):
        u = self.sched[min(self.k, len(self.sched) - 1)]
        self.k += 1
        return u
