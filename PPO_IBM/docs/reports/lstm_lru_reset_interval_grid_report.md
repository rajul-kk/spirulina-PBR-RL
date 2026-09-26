# Reset-interval grid: LSTM vs LRU x reset 60 vs 600

Consolidated 2026-09-22 from v49, v50, v54, v55, v56. Held-out numbers use n=100 main + n=30
high-population seeds with bootstrapped 95% CIs (`experiments/env_diagnosis/bootstrap_sweep_ci.py`),
except v50, which only has the original n=40 sweep.

## Question

`HIDDEN_RESET_INTERVAL` zeroes the actor's recurrent state every N rollout steps. It was added in
v45 because the LSTM cell state grows without bound over a 7200-step episode, saturates, and
freezes the policy; resetting every 60 steps fixed that. Is 60 an LSTM property or a universal
requirement? A bounded-by-construction core (diagonal LRU) run at both intervals answers it.

## Mechanism

- **LSTM:** `c_t = σ(f_t)·c_{t-1} + σ(i_t)·tanh(g_t)`. The forget gate is learned, with nothing
  keeping it away from 1, so the recurrence isn't provably contractive and `‖c_t‖` can keep
  growing.
- **LRU:** `h_t = Λh_{t-1} + Bx_t` with `|λ_i| < 1` guaranteed by parameterization (Orvieto et al.
  2023). That makes it a contraction, so `‖h_t‖` is bounded regardless of rollout length.
- Measured by `state_dynamics_check.py`: late-rollout growth-rate decay is ~23% for the LSTM
  (still climbing) and ~96% for the LRU (asymptoting).

The stability property itself is standard; it's why LRU/S4/S5 exist. The contribution here is
linking it to a concrete operational requirement (reset cadence) and failure signature
(high-population retention) in a control task.

## Results

| Cell | Run | Harvest median [95% CI] | p25 [95% CI] | Gate | High-pop median | High-pop retained (median / p25) |
|---|---|---|---|---|---|---|
| LSTM, 60 | v49 | 139.8 [119.6, 157.6] | 95.4 [82.0, 108.9] | PASS | 290.8 | **11% / 5%** |
| LRU, 60 | v50 (n=40) | 134.4 | 85.0 | PASS (point) | 316.3 (n=12) | 146% |
| LRU, 600 | v54 | 137.0 [110.8, 153.0] | 83.5 [65.5, 101.7] | PASS | 449.8 | 193% / 153% |
| LRU, 600 | v55 | 131.1 [106.1, 147.1] | 83.7 [64.1, 95.1] | PASS | 439.9 | 219% / 173% |
| LSTM, 600 | v56 best | 135.4 [116.6, 151.8] | 84.3 [76.8, 103.9] | PASS | 352.0 | 66% / 43% |
| LSTM, 600 | v56 final | 135.1 [115.9, 154.4] | 85.4 [77.2, 98.7] | PASS | 235.0 | 37% / 19% |

v50 is the weakest row: no CI, and its training run itself collapsed late (the best checkpoint is
pre-collapse).

> **Caveat added 2026-09-26.** Every row here is on the pre-physics-v2 simulator, and none of
> these runs had the actor tanh-saturation penalty. On physics v2, v60 (LRU, reset 600) locked
> into a lights-off corner from every start while the same weights harvested normally at reset
> 60, so "a bounded core doesn't need resets" did not hold there: bounded is not the same as
> within the training distribution (60-step windows build the slow LRU channels to ~2.6x their
> input level; by step 600 they approach ~45x). v61's LSTM actor also runs saturated (|pre-tanh|
> up to 21), so part of the LSTM's high-population loss below may be saturation rather than
> cell-state growth. Treat Finding 1 as unconfirmed until the matched rerun (v62 LRU/600 vs v63
> LSTM/600, both with the penalty) is in. See `docs/decision_history.md#--tanh-saturation-penalty-2026-09-26`.

## Findings

1. **Cross-core at matched reset=600 (the robust result):** the LRU retains 153-219% at high
   population (net-growing the culture, consistent across two runs); the LSTM retains 37-66%.
   Holding the reset interval fixed, the bounded core keeps far more high-population capacity,
   as the contraction argument predicts.
2. **Same core, varying interval (not a finding):** the LSTM retained *more* at 600 (37-66%) than
   at 60 (11%), the opposite of the naive prediction. Each side is a single unreplicated run, so
   this could be seed noise. Don't claim "more frequent resets always help the LSTM". The solid
   same-core result is the older one: free-running (no reset) vs reset=60 (v44->v45), which fixed
   a catastrophic freeze.
3. **The standard gate doesn't separate the cells.** All properly swept cells pass with
   overlapping CIs (131-140mg). The effect lives entirely in the high-population tail, which the
   gate doesn't sample.
4. **Open:** v56 self-recovered from a mid-training D1 collapse matching v50/v52's
   (non-recovering) signature. Single seed, no mechanism identified.

## Confidence

Well supported: the v44->v45 reset fix; LRU/600 success replicated (n=2); the cross-core
retention gap, backed by a directly measured mechanism.

Uncertain: whether the LSTM specifically needs 60; v50's numbers; why v56 recovered. Everything is
one simulator with 1-2 seeds per cell, and a PPO-era seed-variance check (v21 vs v23) found ~30%
spread from seed alone.

## Artifacts

- Checkpoints: `model_data/archive_v{49,50,54,55,56}_*/`.
- Sweep logs: `/tmp/bigsweeps/v{49,54,55}_final_n100.log`, `/tmp/v56_sweep_{best,final}_n100.log`.
  At n=40 (`/tmp/v54_sweep_*.log`, `/tmp/v55_sweep_final.log`) the median-gate CI widens to
  ambiguous, which is why n=100 was adopted.
- Per-run detail: `model_data/runs_registry.csv`, and `docs/decision_history.md` sections
  `#--legacy-TD3-py-hidden-reset-decouple-v54-result`, `-v55-replication`, `-v56-result`.
