class Controller:
    """Open-loop identification: jacket steps of mixed size and duration."""
    LEVELS = [298.5, 302.0, 295.0, 300.0, 297.0, 301.0, 296.0, 302.0, 299.0, 295.0, 298.0, 300.5]
    DUR    = [10,    12,    12,    8,     8,     5,     5,     10,    14,    12,    12,    12]
    def __init__(self, params=None):
        self.k = 0
        self.sched = []
        for l, d in zip(self.LEVELS, self.DUR):
            self.sched += [l] * d
    def act(self, obs):
        u = self.sched[min(self.k, len(self.sched) - 1)]
        self.k += 1
        return u
