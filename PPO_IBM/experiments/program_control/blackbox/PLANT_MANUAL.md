# Spirulina photobioreactor: operator's manual (black-box controller brief)

You are the process engineer for a Spirulina (Arthrospira platensis) photobioreactor. Your job
is to write its supervisory controller: a Python program that reads the plant's sensors every
control step and sets stirring, LED intensity and the harvest pump. You do **not** have the
plant's design model or source code. You learn how it behaves only by running pilot batches.

## The batch
- A batch runs 144 h, controlled every 72 s (7200 steps). Every 12 h the harvest pump removes a
  fraction of the broth and refills it with fresh Zarrouk medium. Nutrient dosing is automatic.
- The objective is to **maximise total biomass harvested over the batch**, and above all never
  lose a culture (extinction). Starting culture sizes vary a lot between batches (30 to about
  5000 cells-equivalent; most are 100-400), and strains vary batch to batch (growth rate,
  temperature optimum). Sensors are noisy, drift and lag.

## Actuators
| actuator | range | notes |
|---|---|---|
| stir_rpm | 50-200 | mixing and gas exchange; more stir means more shear and some heat |
| light_umol | 0-2000 umol/m2/s | drives growth, but saturates, photo-inhibits and heats the tank; the thermostat (35 C setpoint) can only remove limited heat |
| harvest_frac | 0-0.5 | the pump removes the MEAN of the fractions you requested over the preceding 12 h interval, then refills with fresh medium |

## Sensors (the controller sees only these)
| key | meaning |
|---|---|
| turbidity_ntu | nephelometer; ~250 NTU per OD unit on a fresh batch; reads low as filaments clump over days (up to ~30%); saturates at 1000 |
| ph | pH probe (Zarrouk medium; a pH-stat CO2 feed holds it near 10) |
| pump_L | cumulative harvest pump volume, litres |
| conductivity | uS/cm |
| temp_c | broth temperature, C |
| lux | LED panel light meter, ~30 lux per umol/m2/s (it reads back your own light setting) |
| t | step index (1 step = 0.02 h); dt_h = 0.02 |

1 OD unit = 300 mg/L dry weight. The plant team's working target is OD ~0.75.

## Your controller program
One Python file:

```python
class Controller:
    def __init__(self, params=None): ...
    def act(self, obs: dict) -> tuple:   # (stir_rpm, light_umol, harvest_frac)
```
Allowed imports: numpy, math, collections. No file, network or OS access. act() is called 7200
times per batch, so keep it well under 1 ms. Keep any state you need on `self`.

## Pilot plant
```
python plant_trial.py <your_controller.py> [--n 4] [--inoculum CELLS] [--label name]
```
Each call runs NEW batches (never the same batch twice) under your controller and prints, per
batch: the inoculum, total biomass harvested, whether the culture was lost, and the weighed
biomass at each harvest. `trials/<batch>.csv` has the hourly sensor log and your actions, and
`trials/results.jsonl` also holds the **lab dry-weight assay** taken just before each harvest
(mg/L, +-5% assay error): the only ground truth on biomass you get, as in a real plant.
Budget: **300 batches in total**. Pilot batches are expensive, so plan experiments.

## How you will be judged
The final program is run by the project owner on held-out batches: median and 25th-percentile
harvest, and the fraction of cultures lost.
