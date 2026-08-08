# PPO-IBM Project — Usage Guide

---

## 1. Clearing `__pycache__` (Bytecode Cache)

Python caches compiled bytecode in `__pycache__` folders.  
After any significant code change, stale `.pyc` files can cause old behaviour to persist.  
Always clear the cache before re-running if you have edited source files.

**Delete all `__pycache__` folders (PowerShell):**

```powershell
Get-ChildItem -Path e:\SEGP\PPO_IBM -Filter __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
```

This removes the top-level `__pycache__` **and** the ones inside `environments/`, `model_data/`, etc., in a single command.

**Delete just the top-level one:**

```powershell
Remove-Item -Recurse -Force e:\SEGP\PPO_IBM\__pycache__
```

**When to do this:**
- After pulling new commits
- After editing any `.py` file and seeing unexpected old behaviour
- When a log line looks different from the current source (e.g., missing `chunk_eps=` prefix)

---

## 2. `visualize_env.py` — Interactive Reactor Visualizer

Launches a real-time Pygame window showing the reactor simulation.  
You drive the actuators manually with keyboard keys.

**Requirements:** `pygame`, `matplotlib`, `numpy`

### Basic Usage

```powershell
# Genetic environment (default)
python visualize_env.py

# Heavy environment
python visualize_env.py --env heavy

# Total environment at difficulty 0 (Easy)
python visualize_env.py --env total --difficulty 0

# Total environment at difficulty 1 (Medium)
python visualize_env.py --env total --difficulty 1

# Total environment at difficulty 2 (Hard, default for total)
python visualize_env.py --env total --difficulty 2
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--env` | `genetic` \| `heavy` \| `total` | `genetic` | Which environment to run |
| `--difficulty` | `0` \| `1` \| `2` | `2` | Difficulty for `total` env only |

### Keyboard Controls

| Key | Action | Range |
|---|---|---|
| `↑` / `↓` | Stirring speed | 0 – 500 RPM |
| `→` / `←` | Light intensity | 0 – 2000 µE |
| `W` / `S` | Nutrient flow | 0 – 2000 mg/hr |
| `D` / `A` | CO₂ sparging | 0 – 440 mL/min |

At episode end the OD-growth curve is saved as `latest_od_plot.png` in the project root.

---

## 3. `visualize_growth.py` — Rule-Based Growth Benchmark

Runs two deterministic rule-based policies (*Fixed* and *Optimised Rule*) for up to 50,000 steps each and plots Biomass (OD) and Nutrient curves side-by-side.  
No arguments — just run it:

```powershell
python visualize_growth.py
```

Stop early at any time with `Ctrl+C`; it will plot whatever data was collected.

---

## 4. `recurrent_ppo.py` — Recurrent PPO Trainer

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--finetune` | flag | off | Load saved model and continue training |
| `--steps` | `int` | `500000` | Extra steps when using `--finetune` |

### Train from Scratch

```powershell
python recurrent_ppo.py
```

Trains the RecurrentPPO agent through the adaptive curriculum (D0 → D1 → D2).  
Checkpoints are saved in `model_data/recurrent_checkpoints/`.  
TensorBoard logs land in `ppo_recurrent_tensorboard/`.

### Fine-tune a Saved Model

```powershell
# Fine-tune with default 500,000 extra steps
python recurrent_ppo.py --finetune

# Fine-tune with a custom step count
python recurrent_ppo.py --finetune --steps 1000000
```

### View TensorBoard

```powershell
tensorboard --logdir e:\SEGP\PPO_IBM\ppo_recurrent_tensorboard
```

---

## 5. `Var_MPC.py` — Variational MPC Trainer

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--resume` | flag | off | Resume curriculum from the latest checkpoint |
| `--finetune [N]` | optional `int` | `500000` | Fine-tune saved model; optionally pass step count |

### Train from Scratch

```powershell
python Var_MPC.py
```

### Resume Interrupted Training

```powershell
python Var_MPC.py --resume
```

Picks up from the highest-numbered checkpoint in `model_data/varmpc_checkpoints/`.

### Fine-tune a Saved Model

```powershell
# Fine-tune with default 500,000 extra steps
python Var_MPC.py --finetune

# Fine-tune with a custom step count
python Var_MPC.py --finetune 1000000
```

Checkpoints are saved in `model_data/varmpc_checkpoints/`.

---

## 6. `TD_MPC2.py` — TD-MPC2 Trainer

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--resume` | flag | off | Resume curriculum from the latest checkpoint |
| `--finetune [N]` | optional `int` | `500000` | Fine-tune saved model; optionally pass step count |
| `--priv-distill` | flag | off | Enable privileged-information distillation during training |

### Train from Scratch

```powershell
python TD_MPC2.py
```

### Resume Interrupted Training

```powershell
python TD_MPC2.py --resume
```

### Fine-tune a Saved Model

```powershell
# Fine-tune with default 500,000 extra steps
python TD_MPC2.py --finetune

# Fine-tune with a custom step count
python TD_MPC2.py --finetune 1000000
```

### Train with Privileged Distillation

Passes ground-truth environment state (OD, nutrients, pH, temperature) as a teacher signal during the latent representation update. Useful when you want the world model to be guided by privileged information only available at training time.

```powershell
# Fresh training with privileged distillation
python TD_MPC2.py --priv-distill

# Resume + privileged distillation
python TD_MPC2.py --resume --priv-distill

# Fine-tune + privileged distillation
python TD_MPC2.py --finetune --priv-distill
```

Checkpoints are saved in `model_data/tdmpc2_checkpoints/`.

---

## 7. Curriculum Overview

All three trainers share the same adaptive mastery curriculum:

| Level | Label | Description |
|---|---|---|
| D0 | Easy | Stable conditions, low noise |
| D1 | Medium | Moderate perturbations |
| D2 | Hard | Full sensor noise, events, disturbances |

**Promotion criteria (must be met for 2 consecutive chunks):**

| Gate | D0 → D1 | D1 → D2 |
|---|---|---|
| `median_OD ≥` | 0.02 | 0.05 |
| `crash_rate ≤` | 5 % | 5 % |
| `reward_std ≤` | 250 (PPO/VarMPC) / 300 (TD) | 350 (PPO/VarMPC) / 400 (TD) |

**Demotion** occurs on any chunk where `crash_rate ≥ 20 %` or `median_OD < 50 %` of the previous level's baseline.

---

## 8. Evaluate / Benchmark

```powershell
python evaluate_agent.py
```

Results are appended to `benchmark_results.csv`.
