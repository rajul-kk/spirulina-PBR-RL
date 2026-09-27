```python
class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.g = 0.5  # growth rate gain

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        stir = np.clip(p["stir"] + p["g"] * (card["lux"] / 1400), 50, 200)  # added stir gain
        light = np.clip(p["light"] + p["g"] * (card["lux"] / 1400), 0, 2000)  # added light gain
        harvest_frac = np.clip(frac + p["g"] * (card["lux"] / 1400), 0.0, 0.5)  # added harvest gain
        return stir, light, harvest_frac
```

Changes and explanations:

1.  Added stir gain: The stir rate is now influenced by the light intensity, which can help maintain a healthy environment for growth. This gain is added to the original stir rate, ensuring it remains within the allowed range.
2.  Added light gain: Similar to the stir gain, the light intensity is now used to adjust the light rate. This gain is also added to the original light rate, allowing for more precise control of the light environment.
3.  Added harvest gain: The harvest fraction is now influenced by the light intensity, which can help optimize the harvesting process. This gain is added to the original harvest fraction, ensuring it remains within the allowed range.

These changes improve the controller's ability to adapt to changing environmental conditions and optimize the growth and harvesting processes.