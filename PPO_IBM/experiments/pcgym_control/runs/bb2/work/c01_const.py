class Controller:
    """Open-loop: constant jacket temperature (mid-range)."""
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return 298.5
