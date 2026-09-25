# Statistical Validation — Held-Out Sweep Results

> Companion to `finalresults.md` and `novelty_report.md`. Addresses the "no statistical
> treatment across seeds" gap flagged in `novelty_report.md`, for the two held-out sweeps that
> have saved raw per-seed logs (`logs/validation_v27_D0_*.log`, `logs/validation_v26_D2_*.log`).
> Method: 10,000-resample bootstrap, 95% percentile CI, computed directly from the raw per-seed
> `harvest_mg` / `time_avg_od` values in those logs (40 held-out seeds each, disjoint from
> training). Script: not checked in (ad hoc); trivially reproducible from the log format.

## Scope — what this does and doesn't cover

This puts a confidence interval on **evaluation-time variance**: given a single trained policy,
how much does its held-out score vary across 40 different held-out seeds, and is that variance
enough to make a pass/fail verdict uncertain? It does **not** address **training-time variance**
(would a different training seed have produced a meaningfully different policy at all) — that
requires multiple full training runs per configuration, each 15–25h, and wasn't run as part of
this pass. The one existing training-seed data point in this project (v21 vs. v23, nominally
identical configuration) already shows that source of variance can be large: median `time_avg_od`
0.0094 vs. 0.0066, a ~30% spread from seed alone. Any claim resting on a single training run
should be read with that in mind regardless of the held-out CI on top of it.

## Results

| run / mode | n | harvest_mg median [95% CI] | time_avg_od median [95% CI] | crash | gate (harvest / p25 / od / crash) | verdict |
|---|---|---|---|---|---|---|
| TD-MPC2 v27 D0, deterministic | 40 | 59.0 [51.7, 70.1] | 0.0036 [0.00295, 0.00405] | 0.0% | 30 / 15 / 0.004 / ≤15% | **od: CI straddles gate** |
| TD-MPC2 v27 D0, stochastic | 40 | 56.7 [50.7, 66.3] | 0.0036 [0.00310, 0.00405] | 0.0% | 30 / 15 / 0.004 / ≤15% | **od: CI straddles gate** |
| PPO v26 D2, deterministic | 40 | 44.2 [36.3, 62.7] | 0.0026 [0.00205, 0.00345] | 0.0% | 90 / 50 / 0.011 / ≤8% | od: CI fully below gate |
| PPO v26 D2, stochastic | 40 | 43.4 [35.6, 62.2] | 0.0025 [0.00195, 0.00335] | 0.0% | 90 / 50 / 0.011 / ≤8% | od: CI fully below gate |

Harvest and p25 fail decisively for both runs regardless of CI treatment (point estimates are
far outside the gate); the interesting statistical question in both cases is `time_avg_od`, the
criterion each run came closest to.

## What changes vs. the point-estimate framing in `finalresults.md`

**TD-MPC2 v27's D0 claim needs a softer characterization.** `finalresults.md` currently
describes the D0 held-out result as "a consistent, narrow miss (~10% short), not noise." The
bootstrap 95% CI for median `time_avg_od` is **[0.00295, 0.00405] deterministic** and
**[0.00310, 0.00405] stochastic** — both intervals *include* the 0.004 gate value. With this
sample size (n=40), the data cannot statistically distinguish "the true median is just under the
gate" from "the true median is at or slightly above the gate and this sample happened to land
low." **"Not noise" overstates the certainty the data supports** — it should read as a genuine
near-miss whose statistical significance is itself unresolved, not a confirmed sub-gate result.
This doesn't change the practical verdict (`HOLDS ON HELD-OUT SAMPLE: NO` is still correct, since
the gate requires the *observed* median to clear the threshold, and it doesn't) — it changes how
confidently the *margin* can be described, which matters if this result is cited as "TD-MPC2 came
close."

**PPO v26 D2's failure is robust and doesn't need softening.** Both modes' `time_avg_od` 95% CIs
sit entirely below the 0.011 gate (deterministic: [0.00205, 0.00345], nowhere close to 0.011).
This failure is not a borderline call — the existing "roughly half and a quarter of target"
framing in `finalresults.md` is, if anything, statistically conservative.

## Recommendation if pursuing publication

1. Update `finalresults.md`'s TD-MPC2 v27 section to state the CI explicitly rather than "not
   noise" — reviewers computing basic seed statistics from any released per-seed data would catch
   this immediately, and it's a more defensible claim regardless of venue.
2. If the D0 near-miss becomes central to a paper's narrative (as `novelty_report.md`
   recommends for the C1/C2-anchored writeup), either (a) run more held-out seeds to narrow the
   CI, or (b) run TD-MPC2 v27 a second time with a different training seed to see whether the
   near-miss is a stable property of the approach or a property of that one trained policy — the
   v21/v23 precedent (30% spread from seed alone) means this can't be assumed away.
