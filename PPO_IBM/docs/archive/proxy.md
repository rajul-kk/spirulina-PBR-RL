# Removed Proxy Reward Terms

Removed from `genetic_env.py` reward assembly. Kept for reference — each caused a mode collapse exploit.

## reward_do2
```python
reward_do2 = o2_production * 0.1
```
O2 production proxy for photosynthesis. Redundant with reward_biomass; no known exploit but adds noise.

## reward_nut_consume / reward_nut_waste
```python
prev_nut = getattr(self, '_prev_n_pool', self.n_pool)
reward_nut_consume = max(0.0, (prev_nut - self.n_pool)) * 0.02
self._prev_n_pool = self.n_pool

n_addition = nut_flow * 0.75 * self.dt
n_wasted = max(0.0, n_addition - total_uptake_mg)
pop_fraction = self.num_active / self.max_cells
waste_scale = 0.1 + 0.9 * pop_fraction
reward_nut_waste = -0.005 * n_wasted * waste_scale
```
Consume rewarded N disappearing from pool. Waste penalised excess input over uptake.
- `reward_nut_consume` caused N flooding (old 4.9M collapse).
- `reward_nut_waste` caused N=0 starvation (5M collapse) — waste penalty > starvation penalty at low pop.

## reward_n_deficit / reward_p_deficit
```python
reward_n_deficit = -0.01 * max(0.0, (200.0 - self.n_pool) / 200.0)
reward_p_deficit = -0.01 * max(0.0, (10.0 - self.p_pool) / 10.0)
```
Threshold penalties for N < 200 mg/L, P < 10 mg/L. Threshold below initial Zarrouk N (323 mg/L) so deficit never fired during early starvation episodes.

## reward_ph
```python
delta_ph_bio = (delta_mass_mg / max(self.volume_L, 1e-9)) * 0.02
co2_scale = np.clip(1.0 - (co2_flow / (self.max_co2_flow_lpm * 1000.0 + 1e-9)), 0.0, 1.0)
reward_ph = delta_ph_bio * 50.0 * co2_scale
```
Rewarded pH rise attributed to biological growth, gated by `co2_scale`. At CO2=max, `co2_scale=0` so reward_ph=0 regardless of pH — provided no signal against acid conditions.

## reward_dic
```python
dic_target = float(np.clip(2.0 + 0.25 * co2_sat, 1.5, 12.0))
dic_err = abs(self.dissolved_co2 - dic_target)
prev_dic_err = self._prev_dic_err if self._prev_dic_err is not None else dic_err
dic_progress = np.clip(prev_dic_err - dic_err, -0.2, 0.2)
self._prev_dic_err = dic_err
dic_scale = 0.12  # D0; 0.06 D1/D2
reward_dic = dic_progress * dic_scale
```
Progress toward dissolved CO2 target. Caused CO2=max collapse (5M): during the initial CO2 ramp-up from 2→33 mg/L the dissolved CO2 passed through dic_target (~10.5), generating a burst of positive dic_progress that locked the policy onto CO2=max.

## reward_carbon_eff
```python
_mean_fc = mean_f_carbon if self.num_active > 0 else 0.7
reward_carbon_eff = max(0.0, _mean_fc - 0.7) * 0.05
```
Bonus when mean f_carbon > 0.7. With HCO3- CCM, f_carbon ≥ 0.7 almost always (bicarbonate term alone covers it). Fires spuriously.

## mean_f_Q metabolic momentum
```python
mean_f_Q * 0.005
```
Tiny bonus for high intracellular quota. Redundant with reward_biomass signal.
