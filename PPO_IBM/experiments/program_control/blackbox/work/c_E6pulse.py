class Controller:
    """stir 50, pulsed to 150 for the last 1.5 h of each 12 h interval; light 1200; no harvest"""
    def __init__(self, params=None): pass
    def act(self, obs):
        hr = (obs['t'] * 0.02) % 12.0
        return (150.0 if hr >= 10.5 else 50.0, 1200.0, 0.0)
