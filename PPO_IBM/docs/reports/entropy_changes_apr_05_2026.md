# Entropy Control Changes (April 5, 2026)

This document explains all entropy-related updates made in recurrent_ppo.py to stop exploration from overpowering learning and to recover from runaway policy standard deviation.

## Goal

The main issue was entropy pressure dominating policy updates, shown by very high train/std and collapsing reward. The fixes below were applied without changing reward scaling.

## What Was Removed Earlier

- Removed the older adaptive EntropyTuning callback logic that directly changed entropy coefficient from std bands each rollout.
- Replaced that approach with a chunk-based schedule plus guarded feedback.

## What Was Added

### 1) Explicit entropy metric logging

- Added EntropyLoggingCallback.
- Logs train/entropy_coef at rollout end so entropy coefficient appears in train metrics and TensorBoard.

Why:
- Makes entropy pressure visible in the same place as std, losses, and KL.

### 2) Baseline entropy schedule

- Added exponential decay baseline:
  - ENTROPY_INIT = 0.02
  - ENTROPY_DECAY = 0.99
  - ENTROPY_MIN = 0.005
  - ENTROPY_MAX = 0.20
- Added helper functions:
  - entropy_decay_value(chunk_number)
  - entropy_hybrid_value(chunk_number, entropy_multiplier)

Why:
- Provides predictable entropy reduction over time.
- Lower init and faster decay reduce early overpowering.

### 3) Std-band multiplier controller (two-sided)

- Added multiplier-based feedback around std band:
  - Floor: ENTROPY_STD_FLOOR = 0.35
  - Ceil: ENTROPY_STD_CEIL = 0.50
  - Step: ENTROPY_ADJUST_STEP = 0.05
  - Multiplier bounds: ENTROPY_MULT_MIN = 0.30, ENTROPY_MULT_MAX = 1.40
- Behavior per chunk:
  - If std < floor: increase multiplier by +0.05.
  - If std > ceil: decrease multiplier by -0.05.
  - If std is in-band: relax multiplier toward 1.0 by 0.025.

Why:
- Prevents one-way drift.
- Keeps exploration responsive but bounded.

### 4) Panic mode for runaway std

- Added panic threshold and emergency damping:
  - ENTROPY_STD_PANIC = 1.00
  - ENTROPY_PANIC_DAMP_FACTOR = 4.0
  - ENTROPY_PANIC_COEF_CAP = 0.010
- If std > panic threshold:
  - Multiplier is damped quickly (4x normal step).
  - Next chunk entropy coefficient is hard-capped (PANIC_CAP path).

Why:
- Standard damping was too slow for runaway states like std above 2.0.

### 5) Entropy budget guard (anti-overpower)

- Added pressure metrics and trigger logic:
  - ENTROPY_PRESSURE_BUDGET = 0.05
  - ENTROPY_PRESSURE_RATIO_MAX = 3.0
- Derived metrics each chunk:
  - entropy_pressure = abs(ent_coef * entropy_loss)
  - entropy_pressure_ratio = entropy_pressure / abs(policy_gradient_loss)
- If either threshold is exceeded:
  - Apply panic-style damping immediately.
  - Force panic cap next chunk.
- Logged metrics:
  - train/entropy_pressure
  - train/entropy_pressure_ratio

Why:
- Std alone can lag or miss entropy dominance.
- Budgeting the weighted entropy term directly controls objective balance.

## Runtime Flow Now (Per Chunk)

1. Compute scheduled entropy from decay and multiplier.
2. Apply PANIC_CAP if scheduled for this chunk.
3. Train for the chunk.
4. Read train/std and update multiplier (low, high, panic, or relax).
5. Compute entropy pressure and ratio.
6. If budget exceeded, trigger panic damping for the next chunk.

## What Was Not Changed

- Reward scale was not increased.
- Reward normalization behavior on resume was not changed by this patch set.

## Expected Outcome

- Faster correction when exploration runs away.
- Lower risk of entropy overpowering policy gradient updates.
- Clearer observability of entropy contribution in logs.

## How To Apply In Practice

- Stop any run started before these code changes.
- Resume from a checkpoint using the updated script.
- Monitor these together:
  - train/std
  - train/entropy_coef
  - train/entropy_pressure
  - train/entropy_pressure_ratio

If std remains above panic threshold for multiple chunks, lower ENTROPY_PANIC_COEF_CAP further (for example to 0.008) or reduce ENTROPY_MULT_MIN slightly.
