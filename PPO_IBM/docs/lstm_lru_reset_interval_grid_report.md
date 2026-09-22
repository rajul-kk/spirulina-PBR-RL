# Final report: recurrent-core reset-interval grid (LSTM vs LRU x reset=60 vs reset=600)

> Consolidated 2026-09-22 from v49, v50, v54, v55, v56. Supersedes the scattered per-run
> narratives in `runs_registry.csv` and `decision_history.md` as the single reference for this
> finding. All held-out numbers below are freshly recomputed with a consistent n=100 main +
> n=30 high-population sample size and bootstrapped 95% CIs
> (`experiments/env_diagnosis/bootstrap_sweep_ci.py`), except v50 where only the original n=40
> sweep exists (flagged explicitly below).

## The question

`HIDDEN_RESET_INTERVAL` zeroes the actor's recurrent hidden/cell state every N rollout steps.
It was introduced (v45) to fix a real, previously-unexplained collapse: an LSTM actor's cell
state grows without bound over a long (7200-step) episode, saturates, and freezes the policy.
Resetting every 60 steps fixed it completely. The open question this grid answers: **is 60 a
property of the LSTM specifically, or a universal requirement any recurrent core would need?**
Swapping in a bounded-by-construction core (a diagonal Linear Recurrent Unit, LRU) and testing
both reset cadences on both cores answers this directly.

## Mechanism, stated formally

- **LSTM**: `c_t = sigmoid(f_t)*c_{t-1} + sigmoid(i_t)*tanh(g_t)`. The forget gate is a
  *learned*, input-dependent scalar in (0,1) with no architectural floor forcing it away from 1.
  When it sits near 1 (common once trained, to preserve long-range memory), the recurrence is
  not provably contractive -- `c_t`'s norm can climb indefinitely over a long rollout.
- **LRU**: `h_t = Λh_{t-1} + Bx_t`, `Λ = diag(λ_i)` with `|λ_i| < 1` **guaranteed by
  parameterization** (Orvieto et al. 2023's stable exponential form), not learned freely. This
  makes the recurrence a provable contraction: `||h_t|| <= |λ_max|·||h_{t-1}|| + ||Bx||`, so
  steady-state norm is bounded independent of rollout length.
- Directly measured (`experiments/env_diagnosis/state_dynamics_check.py`, cross-core state
  dynamics diagnostic): LSTM cell-state magnitude shows only ~23% late-age growth-rate decay
  over a long rollout (still climbing); LRU shows ~96% (clearly asymptoting).
- This is not a novel theoretical result on its own -- it is the standard justification for why
  stable-parameterization linear recurrent architectures (LRU, S4, S5) exist. What this project
  adds empirically is connecting that known stability property to a specific operational
  requirement (reset cadence) and a specific failure signature (high-population retention) in a
  continuous-control task, an angle the literature search run alongside this project did not
  find addressed elsewhere.

## The grid, with consistent n=100+30 held-out numbers and bootstrap CIs

| Cell | Run(s) | Main sweep: harvest median [95% CI] | p25 [95% CI] | Gate | High-pop (n=30) median harvest | High-pop pop retained (median / p25) |
|---|---|---|---|---|---|---|
| **LSTM, reset=60** | v49 | 139.8mg [119.6, 157.6] | 95.4 [82.0, 108.9] | PASS | 290.8mg | **11% / 5%** |
| **LRU, reset=60** | v50 | 134.4mg (n=40, no CI) | 85.0 (n=40) | PASS (point est.) | 316.3mg (n=12) | 146% (n=12, no p25 recorded) |
| **LRU, reset=600** | v54 | 137.0mg [110.8, 153.0] | 83.5 [65.5, 101.7] | PASS | 449.8mg | 193% / 153% |
| **LRU, reset=600** | v55 (replicate) | 131.1mg [106.1, 147.1] | 83.7 [64.1, 95.1] | PASS | 439.9mg | 219% / 173% |
| **LSTM, reset=600** | v56 | 135.4mg [116.6, 151.8] | 84.3 [76.8, 103.9] | PASS | 352.0mg | 66% / 43% (best ckpt) |
| **LSTM, reset=600** | v56 (final ckpt) | 135.1mg [115.9, 154.4] | 85.4 [77.2, 98.7] | PASS | 235.0mg | 37% / 19% |

All four cells that have a proper n=100 sweep **pass the D2 held-out gate with the CI fully
clear of every threshold** -- no cell fails outright on the standard gate. v50's LRU/60 sweep
predates the n=100/bootstrap-CI protocol (only n=40 exists) and its own registry entry notes a
methodological caveat (its "best" checkpoint directory was contaminated by a later run) -- treat
its numbers as a point estimate, not CI-backed, and lower-confidence than the other four rows.

**v50 is also not really the same experiment as the others**: v50's *training run itself*
collapsed at high population late in its 2,000,000-step budget (best checkpoint banked from
before the collapse, at step 1.3M) -- it is the one cell in this grid where a genuine
non-recovering training-time failure occurred, distinct from the held-out generalization
question the other four rows answer cleanly.

## What the grid actually shows

**1. Cross-core, matched reset interval (the cleanest comparison): LRU has 3-9x the high-population retention headroom of LSTM.**

At reset=600, held everything else fixed:

- LRU (v54/v55): 153-219% retained (both p25 and median stay above 100% -- these policies are
  net-growing the culture at high starting population, not just failing to shrink it).
  Consistent across two independent runs.
- LSTM (v56): 37-66% retained -- positive but well short of LRU, and well short of 100%.

This is the finding the grid was built to test, and it comes through cleanly and consistently:
**holding the reset interval fixed, the bounded core retains far more capacity at high
population than the unbounded one**, exactly as the contraction-mapping argument predicts.

**2. Same-core, varying reset interval: the picture is more complicated than "shorter is always safer for LSTM," and this needs to be stated honestly.**

- LSTM/60 (v49): 11% retained -- the *worst* number in the entire grid.
- LSTM/600 (v56): 37-66% retained -- better than LSTM/60, not worse.

Naive extrapolation from the boundedness mechanism would predict LSTM/600 should retain *less*
than LSTM/60 (more steps for the unbounded state to grow before a reset). That is not what
happened. Two important caveats keep this from overturning the main finding:

- n=1 per cell here (v49 and v56 are each a single training seed) -- this specific comparison
  has no replication, unlike the LRU/600 cell (v54+v55, n=2).
- The mechanism the reset interval was originally introduced to fix is a **catastrophic**
  failure mode (full freeze, near-zero output) measured against a **free-running / no-reset**
  baseline (v44 and earlier, ~7200-step unbroken rollout) -- that comparison remains solid and
  unaffected by this nuance. What v49-vs-v56 tests is a narrower question one level down: *given
  that some periodic reset is already happening, does making it less frequent (60->600) cost the
  LSTM anything on high-population retention specifically?* On this specific run pair, the
  answer was "no, or even a mild net positive" -- but with n=1 each, this could easily be
  seed/trajectory noise rather than a real effect, and should not be reported as a finding on
  its own.

**Bottom line on this point: report the cross-core (LSTM vs LRU) comparison at matched reset
interval as the robust result. Do not claim a monotonic "more frequent reset = better for LSTM"
relationship -- the one data pair testing that directly (v49 vs v56) points the other way, and
neither run is replicated.**

**3. The core D2 gate itself is not the differentiator.** All four properly-swept cells pass
with statistically indistinguishable main-sweep medians (131-140mg, overlapping CIs). If this
project only reported the standard 4-criterion gate, the grid would look like "no effect of core
or reset interval" -- the effect is real but lives entirely in the high-population tail, which
the standard curriculum gate does not sample (by design; the gate is a survival/competence
check on the training distribution, not a stress test).

**4. A genuinely new, unexplained data point: v56's self-recovering mid-training collapse.**
During training (not the held-out sweep), v56 had a real high-population collapse at population
step D1, chunk 10 (harvest_mg=88.0, p25=6.1, capability-check failures rising 1/12->4/12) --
matching the signature of v50/v52's collapses. Unlike either of those (both non-recovering,
ending in demotion or permanent decline), **v56 self-recovered by chunk 14** with no
intervention, then proceeded to a clean D2 advance and hold. This has no precedent elsewhere in
the project. It is flagged here as open and unexplained -- single-seed, no mechanism identified,
not something the state-dynamics or Q-magnitude diagnostics were run against at the time it
happened. A plausible but *untested* speculation is that the 600-step reset boundary happened to
fall favorably relative to the collapse window; this has not been checked against the actual
step-level reset schedule and should not be treated as established.

## What's well-supported vs. still uncertain

**Well-supported:**
- The free-run (no reset) vs reset=60 LSTM fix (v44 -> v45) -- this eliminated a catastrophic,
  reproducible failure mode and is the project's strongest single result.
- The LRU/600 high-population success is genuinely replicated (v54, v55 -- n=2, consistent to
  within normal run-to-run variance).
- The cross-core retention gap at matched reset=600 (LRU 153-219% vs LSTM 37-66%) is consistent
  with a mechanism directly measured via `state_dynamics_check.py`, not just inferred from
  outcomes.

**Still uncertain / explicitly not claimed:**
- Whether LSTM's optimal reset interval is actually 60 specifically, vs. some looser requirement
  -- the one direct test of 60-vs-600 on the same core (v49 vs v56) did not show the expected
  direction, and neither run is replicated.
- v50 (LRU/60) lacks a CI-backed comparable sweep and its training run itself collapsed --
  weakest data point in the grid, kept for completeness rather than as load-bearing evidence.
- Why v56 recovered from a collapse that looked identical in signature to v50/v52's
  non-recovering ones -- open question, single seed, no diagnosis run at the time.
- Everything here is one simulator, 1-2 seeds per configuration (LRU/600 is the only n=2 cell).
  No cross-architecture-family comparison beyond LSTM/LRU (no transformer, no S5/Mamba). A prior
  training-seed variance measurement in this project (PPO era, v21 vs v23, nominally identical
  config) found ~30% spread from seed alone on a different metric -- a standing reminder that
  every single-run number in this grid should be read with that magnitude of uncertainty in
  mind.

## Artifacts

- Checkpoints: `model_data/archive_v49_od_tail_logfix/`, `archive_v50_lru_od_tail/`,
  `archive_v54_lru_long_reset_interval/`, `archive_v55_lru_long_reset_interval_replicate/`,
  `archive_v56_lstm_long_reset_interval/`.
- Raw sweep logs (n=100+30, bootstrap-CI'd): `/tmp/bigsweeps/v49_final_n100.log`,
  `v54_final_n100.log`, `v55_final_n100.log`; `/tmp/v56_sweep_best_n100.log`,
  `v56_sweep_final_n100.log`. v54/v55's original n=40 sweeps also on disk
  (`/tmp/v54_sweep_*.log`, `/tmp/v55_sweep_final.log`) showing the CI widens to AMBIGUOUS on the
  main-sweep median gate at that smaller sample size -- a concrete illustration of why the n=100
  protocol was adopted.
- Diagnostics: `experiments/env_diagnosis/state_dynamics_check.py` (boundedness/sensitivity),
  `experiments/env_diagnosis/bootstrap_sweep_ci.py` (this report's CIs).
- Per-run narrative detail: `model_data/runs_registry.csv` (v49/v50/v54/v55/v56 rows),
  `docs/decision_history.md` (`#--legacy-TD3-py-hidden-reset-decouple-v54-result`,
  `-v55-replication`, `-v56-result`).
