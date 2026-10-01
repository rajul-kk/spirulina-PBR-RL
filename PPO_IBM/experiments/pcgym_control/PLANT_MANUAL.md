# Operator's manual: jacketed stirred-tank reactor (CSTR)

## The plant
A continuously fed, well-stirred reactor converts reactant A in an exothermic reaction. Feed
enters at a fixed flow; product leaves at the same flow. A cooling jacket surrounds the vessel.

## Your job
Hold the concentration of A in the reactor, `Ca` (mol/L), at the setpoint the plan calls for.
The setpoint changes a couple of times during a batch.

## What the controller does
Once per sample (about every 13 seconds; 120 samples in a 26-minute batch) the controller is
given the measurements and must return the jacket temperature for the next sample.

```python
class Controller:
    def __init__(self, params=None): ...
    def act(self, obs):            # called once per sample
        return jacket_temperature  # K, a single number
```
A new Controller is created at the start of every batch. Allowed imports: numpy, math,
collections. No file, OS or network access.

## Measurements (`obs`, a dict)
| key | meaning |
|---|---|
| `t_min` | time since the start of the batch, minutes |
| `Ca` | measured concentration of A, mol/L (analyser noise) |
| `T` | measured reactor temperature, K (sensor noise) |
| `Ca_sp` | the current setpoint for Ca, mol/L |

## Actuator
Jacket temperature, 295 to 302 K. Values outside the range are clipped.

## Disturbances
The feed is not perfectly steady: its temperature and its concentration of A shift from time to
time. Neither is measured.

## Safety
The reaction releases heat. If the reactor temperature goes above 335 K the batch is recorded
as a runaway.

## How a batch is scored
Cost = average over the batch of ((Ca - setpoint) / 0.01)^2, using the true concentration.
Lower is better; a cost of 1 means a typical error of 0.01 mol/L. Runaways are counted
separately.

## Pilot trials
From this directory:
`python plant_trial.py --run <your run> runs/<your run>/work/<controller>.py [--n N] [--label name]`
Each batch is a new scenario (setpoints, feed shifts, start-up state). Every sample is logged to
`runs/<your run>/trials/b###_<label>.csv` and each batch's cost to `results.jsonl`.
The budget is 300 batches.
