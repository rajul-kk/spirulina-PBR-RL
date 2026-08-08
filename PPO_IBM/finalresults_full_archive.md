# Final Results — Reward/Curriculum Validation and Fix Cycle

## Summary

Following the successful periodic-harvest redesign (D2 "mastery" declared at step 2,700,000), a deeper validation pass found that result did not hold up on a broader held-out sample. Root-causing that failure led to four targeted fixes, and three subsequent training attempts (v2-v4). **None of the four post-original-run attempts reached a genuinely validated D2-mastery checkpoint.** v4 came the furthest by a wide margin — cleanly advanced D0→D1→D2 and declared mastery at step 2,300,000 — but failed held-out validation for a newly identified and precisely diagnosed reason: **the trained policy's deterministic (mean) behavior differs fundamentally from the stochastic behavior used to evaluate it during training.**

That diagnosis led to a fifth change — a deterministic-evaluation gate built directly into the training loop — and a fifth training attempt (v5). **v5 is the first attempt this session to produce a genuinely trustworthy result**: it validated D0→D1 advancement with real agreement between stochastic and deterministic evaluation, and — just as importantly — the new gate was directly observed blocking a would-be false-positive D1→D2 advancement three separate times, the exact failure mode that fooled every prior attempt. It did not reach D2 before its budget ran out. This document records what was found, what was changed, what happened across all five attempts, and what to try next.

## Background: the original "successful" run didn't actually generalize

The periodic-harvest-redesigned environment was trained once before this validation cycle and declared D2 mastery at step 2,700,000 (2 consecutive fully-passing curriculum chunks). Three diagnostic scripts built this session (`held_out_sweep.py`, `curriculum_gate_sweep.py`, `reward_breakdown.py`) found that this result was not trustworthy:

- **A 40-seed held-out sweep against the "mastered" checkpoint failed the D2 gate it supposedly passed**: median harvested mg 53.6 (gate ≥90.0), median time_avg_od 0.0037 (gate ≥0.0110) — both well under half the requirement, despite 0% crash rate.
- **A completely static, non-adaptive baseline** (constant stir=80rpm, light=1000µmol, harvest frac=0.15) beat the trained policy on these exact metrics at every difficulty tier — the "smart" dynamic policy underperformed a fixed setpoint.
- **`reward_harvest` contributed only ~4% of realized episode reward** at its original weight (0.25) — structurally dwarfed by the dense `reward_od`/`reward_biomass` terms, which fire every step vs. harvest's 12/episode.
- **A "zombie" failure mode was invisible to the crash-rate metric**: episodes that ran the full 7,200 steps parked near-zero OD for 4,000+ consecutive steps without ever hitting the hard extinction threshold.
- **The curriculum gate itself was evaluated on too small a sample**: `EpisodeMetricsCallback` was recreated fresh every 100,000-step chunk, so advancement decisions only ever saw ~14 episodes — never a rolling history. `MASTERY_WINDOW=10` was imported but never actually used anywhere (dead code). Given harvested-mg swings ~18x and time_avg_od swings ~55x across seeds depending on initial population size, a 14-episode chunk could pass a gate by chance on a lucky draw of larger cold starts — which is exactly what had happened.

## The four fixes implemented

| # | Change | File | Detail |
|---|---|---|---|
| 1 | Reweight `reward_harvest` | `environments/genetic_env.py` | 0.25 → 2.0 (first attempt, later found to overcorrect) → **0.5 (final)** |
| 2 | Escalating near-washout penalty | `environments/genetic_env.py` | New `low_od_streak` counter; milestone penalties at 500/1500/3000/5000 consecutive low-OD steps; whole-term capped at −20/episode (first version was an unbounded per-step tax that reached −360+ and had to be fixed before any training could proceed) |
| 3 | Persistent rolling curriculum window | `callbacks.py`, `recurrent_ppo.py`, `curriculum_schedule.py` | `EpisodeMetricsCallback` now instantiated once for the whole run, keeps a per-difficulty `deque(maxlen=40)` that survives chunk boundaries, instead of a flat list reset every chunk. `MASTERY_WINDOW` 10→40 (now actually wired in). `MASTERY_MIN_EPISODES` 5→20. |
| 4 | Re-derive `ADVANCE_TARGETS` | *(deferred)* | Not hand-picked; intended to be re-anchored empirically once a stable post-fix policy exists. Never reached this step — see Outcome below. |

Change 2's first implementation had its own bug (caught via direct testing before any training, not live): the escalating penalty was a per-step recurring tax that could reach −1,080 for a chronically-stuck culture, dwarfing every other reward term. Fixed by capping the whole term's cumulative per-episode contribution at −20, verified via both a synthetic worst-case test and a stress test with a deliberately bad (over-harvesting) policy before any training was attempted.

## Training attempts

### v1 — aborted (bug, not a real result)
Killed within the first chunk after `ep_rew_mean` was found to be −493 to −494 — caused by Change 2's initial per-step-tax bug described above. Fixed and relaunched as v2 without spending further budget.

### v2 — completed 4,000,000 steps, failed
`reward_harvest` at weight 2.0. Ran the full budget. **D0 never advanced.** Final chunk: `time_avg_od=0.0010` (regressed from a mid-run peak of 0.0030), `ep_rew_mean` settled at 14.2-14.3 (down from a mid-run peak of ~52).

**Root cause, confirmed by direct calculation (not just correlation):** at weight 2.0, `reward_harvest`'s realized per-episode total (~9.24, roughly constant regardless of culture health) **exceeded** `reward_od`'s realized total whenever OD dropped below ~0.0015 — exactly the regime a struggling D0 policy sits in. The reweighting meant to fix harvest's under-weighting (Change 1) had overcorrected, creating pressure to chase harvest yield at the expense of the OD-sustaining behavior the D0 gate actually needs.

### v3 — completed 4,000,000 steps, failed (but much closer)
`reward_harvest` reduced to weight 0.5 (verified via calculation to never dominate `reward_od` across the observed low-OD range 0.0005–0.003+ before relaunching). Ran the full budget. **D0 never advanced**, but the trajectory was qualitatively different and healthier than v2:

- Peaked at `ep_rew_mean` ~57 around chunk 23-24 (vs. v2's peak of ~52).
- **Cleared the D0 gate outright twice** — chunk 23 (`time_avg_od=0.0047`) and chunk 26 (`time_avg_od=0.0042`) — but both times the very next chunk missed (0.0037, 0.0037), resetting the 2-consecutive-pass streak requirement.
- Underwent a genuine multi-chunk decline later (peak ~57 → trough ~31-33) that did not fully reverse before the budget ran out, ending with `time_avg_od=0.0024`, `harvest=76.0mg` (2.5x gate), `crash=0.0%`, `adv=0/2`.
- **A likely root cause surfaced only in the final chunk's log line**: `Std hard-cap applied: 0.714 -> 0.484 (cap=0.65)`. The policy's action-distribution std had drifted above its intended cap and was forcibly clamped. This is a plausible explanation for the "clears the gate once, then misses" pattern seen twice — elevated exploration noise destabilizing the consistency needed to sustain OD across an evaluation window, even though the policy had clearly already learned the underlying capability (it hit the gate, decisively, on two separate occasions).

All three runs were monitored live via `training_run_4M_reward_fix{,_v2,_v3}.log`, with chunk-by-chunk `ep_rew_mean`/`time_avg_od`/curriculum-gate tracking throughout. v2 and v3 model checkpoints, norm stats, and full logs are archived under `model_data/archive_v2_reward_harvest_2.0_failed_D0_stuck/` and `model_data/archive_v3_reward_harvest_0.5_close_but_failed_D0/` respectively, alongside the original (invalidated) run-1 archive.

## Reward simplification and entropy fix (post-v3)

Two further changes were made after v3, motivated by a direct question — "how do we fix this, and should we simplify the reward structure?" — and by the specific evidence v3 left behind.

| # | Change | File | Detail |
|---|---|---|---|
| 5 | Simplify reward from 5 terms to 4 | `environments/genetic_env.py` | `reward_stagnation` folded into `reward_biomass` (same flat -0.010 penalty below 0.01 growth, just accumulated into one bucket instead of two — verified numerically identical via direct calculation before training, not a behavior change). `reward_washout`'s escalating milestone/cap machinery (added mid-session, needed two live bug fixes) reverted to the original simple flat -0.05 form, trusting the already-fixed rolling curriculum window to select against "zombie" policies at the population level instead. |
| 6 | Lower `ENTROPY_PLATEAU_CAP` | `entropy_schedule.py` | 2.0 → 1.3. v3's final chunk showed `Std hard-cap applied: 0.714 -> 0.484` right after a plateau kick pushed the multiplier to 2.0x — the same failure class the file's own comments already documented once before (3.6x → 2.0x, for an even worse regression). The plateau mechanism's aggressive upward kick was fighting the std-band feedback loop that already has its own escalation path. |

Both were verified with a static-baseline `curriculum_gate_sweep.py` pass (identical physics numbers, confirming reward/entropy changes don't touch env mechanics) and a standalone numerical check of the folded biomass/stagnation curve before any training was launched.

## v4 — the furthest by far, still not genuinely validated

Trained fresh with both fixes. Progress was dramatically better than any prior attempt:

- **D0 → D1 advanced cleanly at chunk 10** (`time_avg_od` 0.0055 then 0.0043, both 4/4, first-try 2-consecutive streak — no near-miss oscillation like v3).
- **D1 → D2 advanced cleanly at chunk 14** (harvest 202mg vs 60mg gate, `time_avg_od=0.0233` vs 0.0080 gate).
- The one plateau kick that fired (chunk 6, capped correctly at the new 1.0x-1.3x range) produced **no subsequent std hard-cap correction** — the entropy fix appears to have worked as intended.
- `ep_rew_mean` climbed to **265** by the end — an order of magnitude beyond any previous attempt (v2 peaked ~52, v3 peaked ~57).
- **D2 mastery was declared at step 2,300,000** — `[EARLY STOP] D2 mastery confirmed (2 consecutive passing chunks at full difficulty)` — using barely over half the 4M-step budget.

(Note: the run was interrupted once mid-training by an external process kill unrelated to the code — likely the machine sleeping — after ~460k steps with no error in the log; it was relaunched fresh from step 0 and the second attempt is the one that reached mastery.)

### The held-out sweep failed it — worse than the original run

Per standing practice, `held_out_sweep.py` (40 seeds, D2, including adversarial cold starts) was run against the checkpoint immediately, before accepting the mastery claim:

```
crash_rate           : 0.0%
harvested_mg  median : 15.8   p25: 2.9    (gate: median>=90, p25>=50)
time_avg_od   median : 0.0129 (gate: >=0.011 — barely clears alone)
holds on held-out sample: NO
```

Median harvested mass was only 17.6% of the gate — a *larger* gap than the original invalidated run showed (which had reached 53.6mg median). Despite the best live-training trajectory of the whole session, this is the worst-performing checkpoint under held-out validation.

### Root cause, confirmed directly (not inferred): the deterministic policy never harvests

Running the 4-seed `test_actions.py` action-trace check (deterministic policy, same tool used earlier this session to characterize the original run) revealed the precise mechanism, consistent across all four seeds:

- **`Stir` pinned at exactly 50 RPM** (the action-space minimum) for essentially the entire episode.
- **`Harvest` fraction collapsed to 0.00-0.05** throughout — i.e., the policy's mean/deterministic action is to almost never harvest, regardless of standing biomass.
- With nothing ever harvested, population and OD grow **unboundedly**: seed 0 ends at Pop=1,076 (OD=0.024); seed 1 at Pop=2,258 (OD=0.047); seed 2 at Pop=4,017 (OD=0.071); seed 3 at Pop=2,781 (OD=0.061) — 4-13x the population any prior checkpoint's test episodes reached.
- Cumulative reward still climbs to 79-154 across these episodes purely from `reward_od`/`reward_biomass` compounding on ever-growing biomass — a degenerate strategy that is genuinely reward-maximizing under the *deterministic* policy, but produces almost no actual harvest.

**Why this passed the live curriculum gate but fails held-out validation**: SB3's rollout collection during `model.learn()` uses **stochastic** action sampling (the whole point of the entropy term), not the deterministic mean. The policy's mean harvest action sits at essentially zero, but sampling noise around that mean (raw action std of 0.02-0.2 observed in the traces) occasionally produces small-but-nonzero harvest fractions purely by chance — enough of them, accumulated across an episode, to inflate the *stochastic* rollout's `harvested_mg` into gate-passing territory (177-202mg during live training). `held_out_sweep.py` and `test_actions.py` both call `model.predict(..., deterministic=True)` — the same setting that would matter for actual deployment — which strips out that noise and reveals the real, degenerate policy underneath.

This is a distinct and more precise finding than a superficial read of "harvest reward too low" would suggest: the issue isn't that `reward_harvest`'s weight (0.5) is miscalibrated in isolation, it's that **the curriculum's live evaluation methodology (stochastic rollouts) and the actual deployed/tested policy (deterministic) can diverge arbitrarily**, and nothing in the training loop currently checks for or penalizes that divergence. A policy can look like it mastered harvesting while its actual learned behavior is "never harvest, let biomass grow forever."

v4's checkpoint, norm stats, and full log are archived under `model_data/archive_v4_reward_simplified_D2mastery_2.3M/`.

## Fix #5: deterministic-evaluation gate (post-v4)

Acting directly on the "recommended next step" from the v4 analysis: `EpisodeMetricsCallback` only ever records episodes generated during `model.learn()`, which is inherently stochastic (SB3 has no built-in mechanism for deterministic rollout collection). Rather than trying to make degenerate strategies unprofitable through more reward tuning (a narrower, more speculative fix), a **structural** fix was implemented: a small number of genuinely deterministic episodes are now run inside the training loop itself, on the same schedule as curriculum evaluation, and gated the same way.

| # | Change | File | Detail |
|---|---|---|---|
| 7 | Deterministic-eval gate | `deterministic_eval.py` (new), `curriculum_schedule.py`, `recurrent_ppo.py` | New `run_deterministic_eval_episode()` — modeled directly on `held_out_sweep.py`'s proven `run_episode`, but takes a live `obs_rms` snapshot (`copy.deepcopy`) instead of loading one from disk, so it can run against the still-training model. Wired into the chunk loop: `DET_EVAL_EPISODES_PER_CHUNK=3` deterministic episodes per 100k-step chunk, tracked in a persistent per-difficulty `deque(maxlen=DET_EVAL_WINDOW=15)`, gated against the same `ADVANCE_TARGETS` thresholds (`DET_MASTERY_MIN_EPISODES=9` before eligible). Curriculum advancement now requires **both** the existing stochastic gate **and** this new deterministic gate to independently pass. |

Verified before any training: a unit test built two stub "models" — one that always outputs `harvest=-1` (never harvest, the exact degenerate v4 behavior) and one that always outputs `harvest=+1` (max fraction every event) — and confirmed `run_deterministic_eval_episode` correctly reports `harvested_mg=0.0` for the first and a crash (over-harvest washout) for the second, before trusting the mechanism inside a multi-hour run.

## v5 — first genuinely trustworthy result this session

Trained with the deterministic-eval gate active. Two runs were needed (the first was interrupted after ~460k steps by an external process kill unrelated to the code, likely the machine sleeping — same class of interruption v4 also hit once; relaunched fresh from step 0).

**The gate worked exactly as designed, twice over:**

- **D0 → D1 advanced at chunk 30 with genuine agreement between both gates** — stochastic `time_avg_od=0.0044`, deterministic `time_avg_od=0.0054`, both independently above the 0.0040 threshold for 2 consecutive chunks. Unlike v4, this is not a stochastic-only artifact.
- **D1 → D2 never advanced — and the log shows exactly why the gate mattered.** Across chunks 34-36, the *stochastic* gate hit a clean 4/4 pass three separate times (`time_avg_od` 0.0108, 0.0117, 0.0101 — all comfortably above the 0.0080 D1→D2 threshold). Under the old (pre-fix) logic, this would have advanced to D2, quite possibly declaring mastery shortly after, exactly as v4 did. But the *deterministic* gate's harvest metric hovered at 52-58mg the entire time, short of its own 60mg threshold — so `mastery_streak` never built, and the run correctly stayed at D1 for the remainder of the budget. It finished at step 4,000,000 with the deterministic gate's harvest sitting at **59.9mg — one-tenth of a milligram short** of the gate, having never crossed it.
- `ep_rew_mean` reached 97-112 in the final stretch — lower than v4's inflated 265 peak, which is itself informative: v4's number was partly an artifact of the same exploration-noise effect the deterministic gate now prevents from being rewarded.

A follow-up 1-seed `test_actions.py` action-trace check at D1 (the genuinely validated tier) confirmed the qualitative difference from v4 directly: **harvest fraction sits at a real, consistent ~0.33-0.36 throughout the episode** — nothing like v4's collapse to ~0.00-0.05. This is a genuinely harvesting policy, not a degenerate one. The same trace also showed this specific seed's population declining over the episode and ending with negative reward (though completing the full 144h without crashing) — an honest sign that D1-level harvesting, while real, isn't yet robust across all cold starts, consistent with the curriculum correctly not advancing it further.

v5's checkpoint, norm stats, and full log are archived under `model_data/archive_v5_det_eval_gate_D1_validated_no_D2/`.

## Current state

- **No D2-level checkpoint has been produced by any attempt this session that also survives independent validation.** However, v5's D1-level checkpoint is the first checkpoint this session whose curriculum-reported capability is corroborated by a genuinely deterministic evaluation, at every tier it claims to have reached.
- The deterministic-eval gate (`deterministic_eval.py`) is now a permanent part of the training loop, not an optional check run after the fact — this closes the exact class of false positive that affected the original run and v4, structurally, rather than relying on remembering to run `held_out_sweep.py` afterward.
- `TOTAL_TRAINING_STEPS=4_000_000`, `ENTROPY_PLATEAU_CAP=1.3`, `reward_harvest` weight `0.5`, reward function simplified to 4 terms — all unchanged from the post-v4 state, since v5's shortfall was not diagnosed as a reward-tuning problem.

## v5b — resumed from v5, tests the "budget-limited" hypothesis, finds a real ceiling instead

Option 1 from the prior recommendation was acted on directly: `TOTAL_TRAINING_STEPS` raised 4,000,000 → 7,000,000, and training resumed from v5's exact checkpoint (chunk 41, step 4,000,000, D1, `mastery_streak=0`).

**Immediate post-resume instability (chunks 41-42), self-corrected:** the deterministic gate's crash rate spiked to 66.7% then 50.0% (vs. 0% where v5 left off), and det harvest fell to ~30mg. This settled on its own over the next several chunks as the resumed model re-equilibrated — not treated as a hard failure, and it wasn't one: crash rate declined monotonically (67% → 50% → 33% → 25% → 27% → 13% → 6.7% → 0%) through chunk 50, exactly the kind of transient a fresh checkpoint-reload can cause.

**A genuine capability ceiling, not just budget exhaustion, is what actually blocked D2:**

- Chunks 51-53: the *stochastic* gate cleared a clean 4/4 for three consecutive chunks — further and more consistently than v5 ever achieved. But `mastery_streak` never moved off 0/2, because the deterministic gate kept failing on exactly the two hardest criteria: harvest (51.1 → 54.1 → 48.1mg, need ≥60) and `time_avg_od` (0.0030 → 0.0033 → 0.0033, need ≥0.008 — never even reaching half the threshold).
- `plateau_counter` reached 6 at chunk 54, firing a third entropy-multiplier kick this session (→1.30x, the hard cap). Unlike v5's post-resume dip, **this kick made things measurably worse, not better**: the stochastic gate regressed from 4/4 back to 3/4, det harvest kept falling (43.6mg, then 41.1mg — a fourth consecutive decline), and det crash rate ticked back up to 6.7% off its 0% floor.
- At that point — det harvest declining for four straight chunks, det `time_avg_od` stuck under half its gate for eight straight chunks, and the standard plateau remedy (entropy kick) actively regressing both gates rather than helping — this was judged a genuine plateau/regression rather than a transient, and the run was stopped at step ~5,590,000 (chunk 55, `steps_done=5,500,000` in the last saved state), well short of the 7,000,000 budget, rather than continuing to spend budget against a wall.

This resolves the ambiguity from the prior "budget-limited vs. capability-limited" framing: **it was not simply budget-limited.** Given 1.5M additional steps and a fresh entropy kick, the deterministic harvest/`time_avg_od` metrics did not trend toward their gates — they oscillated and then declined. v5's 59.9mg case now reads as this checkpoint's high-water mark under the current reward structure and physics calibration, not a near-miss it was about to clear with more time.

v5b's final checkpoint (chunk 55, step 5,500,000, D1, never advanced to D2), training state, norm stats, and full log are archived under `model_data/archive_v5b_resume_D1_plateau_5.5M/`. `model_data/recurrent_ppo_genetic_ibm.zip` itself was **not** updated by this run (SB3 only writes it on `Training Complete`, which this run never reached) — it still holds v5's original save; the actually-furthest checkpoint is the archived one above.

## Current state (updated)

- **D1 remains the highest tier validated by both gates this session.** v5b pushed further within D1 (stochastic 4/4 sustained across more chunks than v5 managed) but did not move the deterministic harvest/`time_avg_od` metrics past their D1→D2 thresholds even with 1.5M extra steps and an entropy kick, and both regressed rather than improved once the ceiling was pushed on.
- `TOTAL_TRAINING_STEPS` is now `7_000_000` in `curriculum_schedule.py` (raised from 4M for this run) — worth reverting or reconsidering before the next attempt, since the extra budget did not translate into progress here.

## Recommended next step (updated)

The "just add more budget" hypothesis is now tested and did not hold. What v5b's failure pattern actually points at:

1. **`time_avg_od` is the metric that never moved** (stuck at 0.001-0.005 across the entire resumed run, need 0.008) — this is a *steady-state* biomass-density measure over the back half of the episode, not a one-off event like harvest. A policy that harvests adequately but doesn't sustain OD in the target band between harvests will fail this specific criterion indefinitely regardless of training time. Worth checking directly (e.g. via a `test_actions.py` trace at the current archived checkpoint) whether the policy is cycling OD too aggressively around harvest events rather than holding a stable higher baseline.
2. **The entropy plateau-kick mechanism regressed this run** rather than helping, for the second time this session (also implicated in v3). It may be worth capping `ENTROPY_PLATEAU_CAP` further, increasing `PLATEAU_CHUNKS` so kicks fire less eagerly, or removing the mechanism in favor of just letting a genuine plateau stand and be diagnosed rather than perturbed.
3. **Consider whether the D1→D2 `time_avg_od` threshold (0.008) is well-calibrated** against what the current reward structure actually incentivizes — `reward_od` saturates via `tanh(od/0.20)`, and if the policy's harvest-driven OD sawtooth rarely spends enough time above a level that integrates to 0.008 over the back half of the episode, this may need a physics/reward-level fix rather than more training time.

As always: no advancement or mastery claim should be trusted without checking that both gates genuinely agree, and a `test_actions.py` action-trace spot-check remains a useful independent sanity check on top of that.

## Zombie-episode diagnosis (post-v5b) — why more training wasn't the fix

Before acting on recommendation #1 above, a targeted diagnostic (`zombie_diagnosis.py`, new) ran 24 deterministic episodes against v5b's archived checkpoint at D1, tracking per-step OD and actions to find what actually distinguishes the ~1/3 of cold-start episodes that fall into the "zombie" pattern (extended OD<0.001 stretches that never hard-crash but score heavily negative via the flat washout term) from the ones that don't.

**Findings:**
- **Strong cold-start-size effect**: episodes starting under 150 cells zombied 86% of the time (6/7); episodes starting at 150+ cells zombied only 12% of the time (2/17). Median init_cells: 115 (zombie) vs. 217 (healthy).
- **Severe cost, rare detection**: median zombie duration was 1,098 steps (~15% of the full 7,200-step episode), up to 1,524 steps. All 8/8 zombie episodes eventually recovered before episode end (none hard-crashed), so this entire failure mode is **invisible to the curriculum's `crash_rate` gate** — it only shows up as reward damage (mean total deterministic reward: zombie −125.83 vs. healthy +35.13) and as suppressed `time_avg_od`.
- **No differentiated recovery behavior**: mean action (stir/light/harvest) was statistically indistinguishable before a zombie stretch, during it, and in healthy episodes overall — the policy wasn't reacting differently to being in trouble, because `reward_od` (`tanh(od/0.20)`) and `reward_biomass` are both nearly flat/silent near od≈0, giving almost no gradient to learn a recovery response from.
- **Conclusion this pointed to**: not a training-budget problem (v5b had already spent 1.5M extra steps and an entropy kick chasing this exact metric with no improvement, even regressing after the kick) but a **reward-signal gap** — nothing in the function rewarded the direction of change near the low-OD region where it mattered most.

## Fix #6 (v6): dense OD-movement term, replacing the flat washout floor

Per explicit direction to avoid further hardcoded thresholds/floors/ceilings, the flat `reward_washout = -0.05 if od < 0.001 else 0.0` term was **removed** and replaced with a dense, threshold-free term based on the *sign and size of the step-to-step OD change*, present at every OD level:

```python
delta_od = self.od - self._prev_od_for_rate
reward_od_delta = 0.0 if is_harvest_event else 0.01 * float(np.tanh(delta_od / 1e-6))
self._prev_od_for_rate = self.od
```

- No `if od < X` anywhere — od_delta rewards OD going up and penalizes it going down proportionally, regardless of the absolute level, so it stays informative during exactly the low-OD stretches where `reward_od`/`reward_biomass` go quiet.
- Skipped on harvest-event steps: the OD drop there is the intended, already-rewarded effect of harvesting (`reward_harvest`), not decline — including it there would directly fight that incentive.
- The `1e-6` divisor and `0.01` weight are squashing/scaling constants (the same role `od/0.20` and `growth/5.0` already play for the other terms), not floors or ceilings — they were calibrated empirically before training by measuring real per-step OD deltas under several scripted policies (healthy growth, starving, random, small-cold-start) and choosing values that put `od_delta`'s per-episode total in the same range as `reward_od`'s (~30-70 under a healthy fixed policy) rather than letting it dominate. An initial naive scale choice (`5e-4`) was verified too weak (contributed a negligible ~0.0002/step); a subsequent one (`1e-6` with the original `0.10` weight) was verified far too strong (~688/episode, ~10x `reward_od`) — both caught by direct calculation against measured deltas before committing to a training run, same discipline used for every prior reward change this session.
- `reward_term_sums` tracking updated: `washout` key replaced with `od_delta` (`genetic_env.py`, `reward_breakdown.py`).

Reward is now `reward_od + reward_biomass + reward_od_delta + reward_harvest` (still 4 terms).

**v6 training launched fresh** (not resumed from v5b — the old checkpoint's policy/value function were calibrated against the old reward scale, including the removed washout term, so resuming would carry stale value estimates into a changed reward landscape). `TOTAL_TRAINING_STEPS` reset to `4,000,000`. Log: `training_run_4M_od_delta_v6.log`.

**D0 → D1 advanced at chunk 6 (step 600,000)** — both gates genuinely agreeing, det `time_avg_od` reaching 0.0067-0.0071 against a 0.0040 gate. v5 took 30 chunks (3,000,000 steps) to reach the same milestone — **5x faster** under the new reward.

**D1 → D2 advanced at chunk 11 (step ~1,100,000)** — again both gates genuinely agreeing (det: harvest=81.4mg/60.0, p25=51.7/30.0, crash=0.00%/10.00%, time_avg_od=0.0093/0.0080). **This is the first genuine D2 advancement of the entire session** — the original run's and v4's D2 claims were both stochastic-only artifacts that failed 40-seed held-out validation; this one cleared both the stochastic rollout gate and the independent deterministic-eval gate with real margin, not a last-chunk squeak. Det crash rate stayed at a flat 0% through the entirety of D0 and D1 — no recurrence of the zombie pattern that the diagnosis identified.

D0→D1→D2 took 11 chunks (1.1M steps) total — dramatically faster than any prior attempt (v5 needed 30+ chunks just to reach D1 and never reached D2 in 4M steps; v5b spent another 1.5M steps on top of that without reaching D2).

**But D2 exposed a new failure mode, caught and fixed live.** Over chunks 14-19, both deterministic and stochastic harvest totals collapsed steadily and consistently (det: 34.9→29.2→24.7→16.5→12.8→8.0mg; stochastic: 142→137.8→137.8→113.2→98.4→78.7, falling below its own 90mg gate) while crash rate stayed a flat 0% and `time_avg_od` stayed healthy — i.e. the population wasn't dying, it just wasn't being harvested. Direct calculation confirmed the mechanism: `reward_od_delta` as first implemented used the *absolute* per-step OD change, which scales with population size — a larger standing culture produces larger absolute OD deltas at the same per-cell growth rate, making "never harvest, just keep the population large and growing" directly reward-maximizing under that term (verified: a healthy episode totaled 552.79 in `od_delta` at a 3,000-cell init vs. 43.90 at a 300-cell init, off the same underlying growth *rate*). This is the same structural mistake `reward_biomass` had already avoided by using per-cell growth rather than raw mass delta — `od_delta` had reintroduced it.

**Fix**: switched `od_delta` to *relative* OD change (`delta_od / (prev_od + 1e-6)`, rescaled), the same population-size-invariant pattern `reward_biomass` uses. Verified directly before relaunching: a healthy episode now totals ~44-49 in `od_delta` regardless of whether the population starts at 300 or 3,000 cells, and a real harvesting policy (frac=0.15) isn't penalized (harvest term unaffected, `od_delta` stays in the same healthy range). Training stopped at step 1,900,000 (chunk 19, mid-D2-collapse) and archived to `model_data/archive_v6_od_delta_D2_harvest_collapse_1.9M/`. Relaunched fresh as **v7** (`training_run_4M_od_delta_relative_v7.log`) — a resumed run would have carried the collapsing-harvest policy and value function forward, so starting over was necessary here too, same reasoning as the v5b→v6 transition.

**v7 completed its full 4,000,000-step budget without reaching D2** — final state: D1, `mastery_streak=0/2`, det harvest=53.9mg (below the 60mg gate), det `time_avg_od`=0.0034 (below 0.008), det crash 0%, `ep_rew_mean`=158. Clean completion (`Training Complete`, model and norm stats both saved). Checkpoint archived to `model_data/archive_v7_relative_od_delta_D1_no_D2_4M/`.

**The run's shape was qualitatively different from every prior D1-stall (v5b included) — this was a threshold-oscillation problem, not a collapse or a crash-rate regression.** D0→D1 advanced cleanly at chunk 10. Through much of D1, det harvest repeatedly touched or cleared 60-83mg and det `time_avg_od` repeatedly touched or cleared 0.008-0.012 — `mastery_streak` reached 1/2 three separate times (chunks 20, 22) but never consolidated to 2/2, always losing the streak to one metric dipping just under its gate the very next chunk while the other stayed fine. In the run's final third (chunks 27 onward), det harvest settled into a persistent 37-47mg band — still nowhere near v6's collapse-to-8mg, but a real, unrecovered dip that combined with the oscillating `time_avg_od` to exhaust the budget before two clean passes ever lined up. Det crash rate stayed a flat 0% for the entire 4,000,000-step run — no zombie/washout regression recurred at any point.

**Independent validation (`reward_breakdown.py`, `zombie_diagnosis.py`) confirms the harvest-collapse fix genuinely worked, and quantifies a new, smaller finding:**

- `reward_breakdown.py` (8 deterministic episodes, D1): mean total reward **+93.96**, every single episode positive (range +67.8 to +102.7). This is a complete reversal from v5b's mean of **-6.88** (6 of 8 episodes wrecked by the washout term). Per-term breakdown: `od_delta` 71.3% of total, `od` 16.3%, `biomass` 10.8%, `harvest` only 1.6%.
- `zombie_diagnosis.py` (20 deterministic episodes, D1): **3/20 (15%) zombie rate**, down from 8/24 (33%) pre-fix. Every episode — zombie or healthy — scored positive reward (range +81.8 to +118.8); the old catastrophic split (healthy +35 vs. zombie -126) is gone. Zombie episodes also recovered faster (median duration 205 steps vs. the old 1,098) and the small-cold-start correlation weakened substantially (zombie median init 203 vs. healthy median 225, nearly overlapping, vs. the old 115-vs-217 split).

**Verdict**: the relative-OD-delta fix resolved the catastrophic failure mode it was built to fix — zombie episodes are rarer, shorter, and no longer reward-destroying. v7's D1 checkpoint is meaningfully higher-quality and more reliable than v5's (which had bad seeds ending in outright negative reward). But it exposed a second-order issue: `od_delta` now dominates the reward (71% of total) while `harvest` is nearly negligible (1.6%), which plausibly explains why harvest itself proved to be the metric that couldn't consistently clear its gate — the policy has far more reward available from sustaining OD growth than from actually harvesting it. D2 was never reached this run.

## Physics scan + root-cause investigation (post-v7)

Before touching the reward again, two physics-only (no RL) diagnostics isolated exactly why D1→D2 stalled:

1. **`curriculum_gate_sweep.py` re-run (physics unchanged since session start)**: a simple constant action (stir=80rpm, light=1000µmol, frac=0.15) clears every tier's gate with real headroom — D0 4.5x/4.3x, D1 (the D1→D2 gate) 2.3x/2.2x, D2 (own terminal gate) 1.6x/1.6x. **Physics was never the blocker**, at any tier, including D2's harsher settings (`genetic_env.py`: full oxygen-toxicity growth inhibition and full thermal dynamics at D2 vs. 75%-scaled at D1, 50%-scaled/off at D0 — `phys_scale`/`f_O2` at lines 604, 849).
2. **`dynamic_profile_sweep.py` re-run at D2 with the (pre-this-fix) reward function, sweeping harvest_frac 0.0-0.50**: mean reward was *highest at frac=0.0* (never harvest, +300.6) and fell monotonically as harvest increased, bottoming at the actual best-yield fraction (+161.4 at frac=0.15, the physically optimal point). A direct per-term decomposition (same fixed trajectories, algebraically rescaled) isolated the cause precisely: `od_delta` (this session's earlier fix) stayed flat (~69) across every fraction exactly as its population-invariant design intends — **it was not the problem**. `reward_od` (the original standing-OD level term, untouched since before this session) swung from +221 (frac=0) to +29 (frac=0.30), because it rewards absolute OD level and harvesting directly lowers OD. Its raw per-episode ceiling (0.15/step × 7200 steps = 1080) dwarfed `reward_harvest`'s (0.5 × 12 events = 6) by ~180x — a structural imbalance present since the reward function's original design, only now decisively measured.
3. **v7's D1-trained checkpoint tested directly against D2 physics** (never trained there): 12/15 healthy, 3/15 zombie (20%, comparable to its 15% D1 rate), all 15 episodes still positive reward (+84.3 to +144.4). Generalization to D2 physics was reasonable, not a hard wall — reinforcing that the blocker was the reward landscape, not an unfamiliar-physics cliff.

## Fix #7 (v8): rebalancing reward_od against reward_harvest

Per explicit direction to fix this and keep changes non-engineered/non-overfit to one run: `reward_od`'s weight reduced 0.15→0.05, `reward_harvest`'s weight raised 0.5→4.0 (`genetic_env.py`). This closes most (not all) of the measured ~180x ceiling gap — new ceilings: `reward_od` 360, `reward_harvest` 48, ~7.5x gap — a direct, measured correction rather than an arbitrarily-chosen pair of constants.

**Verified before training** (same discipline as every reward change this session):
- Re-ran the harvest-fraction sweep with the new weights: frac=0 total dropped to +154.2, now *beaten* by frac=0.05 (+161.4) — never-harvesting is no longer the reward-maximizing choice. (Not perfectly monotonic — frac=0.10/0.15 land slightly below frac=0.05 — but the old ~140-point/1.9x gap favoring frac=0 over the best physical yield is now a ~7-point edge in harvesting's favor, and even frac=0.15 sits only ~10% below frac=0 rather than ~46% below.)
- Checked a synthetic struggling low-OD D0 scenario (weaker light/stir, od dropping to ~0.0002-0.011) specifically to rule out recreating **v2's exact historical failure** (harvest weight 2.0 caused harvest's near-constant per-episode total to exceed reward_od's in exactly this low-OD regime, stalling D0 for a full 4M-step budget). Result: all four fractions tested landed within a narrow 71.8-75.3 band — no single fraction dominates, `od_delta`+`biomass` (unaffected by this change, ~55+~6 in this scenario) remain the largest stable contributors regardless of harvest amount. v2's dominance pattern was not reproduced.

`TOTAL_TRAINING_STEPS` reset to `4,000,000`, launched fresh (not resumed — v7's policy/value function were calibrated against the old od/harvest balance). Log: `training_run_4M_reward_rebalanced_v8.log`. Results to follow.

**v8 stalled and was corrected mid-session.** Early D0 progress looked normal (harvest consistently 43-63mg, well above the 30mg gate — confirming the rebalance fix itself worked), but det `time_avg_od` froze at **exactly 0.0014 for eight consecutive 100k-step chunks (9-16)**, unmoved even by an entropy-multiplier kick at chunk 12, while `ep_rew_mean` was simultaneously flat (100-101) for 12 straight readings — over 800,000 steps (20% of budget) with no forward progress on either metric. Diagnosis: reducing `reward_od`'s weight to 0.05 fixed the harvest bias but left too little dense OD-sustaining signal for the policy to learn to balance both objectives at once (the physics sweep had already shown both harvest≥60mg and time_avg_od≥0.008 are simultaneously achievable at a moderate ~0.15 fraction — this was a learning-signal problem, not a physical impossibility).

**Fix**: stopped the run (archived to `model_data/archive_v8_stalled_time_od_1.8M/`), nudged `reward_od`'s weight 0.05→**0.06**. The exact value was chosen by re-running the same fixed-action decomposition against three candidates (0.06, 0.07, 0.08): 0.07 was already a dead heat between frac=0 and the best-harvest fraction, 0.08 clearly re-flipped back to favoring never-harvesting outright — 0.06 was the largest strengthening (+20% over 0.05) that still keeps harvesting the clear reward-maximizing choice in the same test. Relaunched fresh (not resumed, same reasoning as every reward change this session) as **v9** (`training_run_4M_reward_rebalanced_v9.log`), `TOTAL_TRAINING_STEPS=4,000,000`.

**v9 repeated the stall, slightly worse.** Same pattern as v8: det `time_avg_od` stuck at 0.0013-0.0014 for five straight chunks (7-11), but this time `ep_rew_mean` was actively *declining* (84.6→78.2), not just flat. Stopped and archived to `model_data/archive_v9_stalled_declining_1.2M/`.

**Root-cause investigation via `test_actions.py` action trace (not another blind weight nudge) found the real mechanism**: the trained policy had converged to `stir` pinned near minimum (50-55rpm, negligible variance) and `harvest` fraction 0.30-0.36 — well past the physically-sustainable range (~0.10-0.15 found in every physics sweep this session). A direct fixed-action test isolating harvest fraction (stir=50, light=1300, matching the trace) confirmed this precisely:

| frac | time_avg_od (need ≥0.0040) |
|---|---|
| 0.10 | 0.0246 (6x gate) |
| 0.15 | 0.0138 (3.4x gate) |
| 0.20 | 0.0089 (2.2x gate) |
| **0.30** | **0.0025 (below gate)** |
| **0.36** | **0.0013 (below gate)** |

Harvesting that aggressively and frequently never lets the population build enough standing biomass for OD to grow, even though `harvested_mg` still looks superficially healthy (65-107mg, above D0's 30mg gate) — explaining exactly why harvest kept clearing its own gate while `time_avg_od` stayed stuck. **Puzzlingly, the current reward weights don't actually favor this**: re-running the term decomposition at the policy's own stir/light setting showed total reward is monotonically *highest* at frac=0.10 (+142.97) and lowest at frac=0.36 (+108.25) — the static, fixed-action reward landscape already prefers restraint. The trained policy nonetheless converged to the worse end of that range, which points to a **PPO training-dynamics/local-optimum issue** (harvest events give immediate, easily-attributed reward; the payoff of restraint is delayed and diffuse, and once population stays chronically small from over-harvesting, hitting the fixed 12.32mg-per-event target requires progressively larger fractions of a smaller pool — a self-reinforcing bad equilibrium) rather than a term-weight imbalance fixable by further reweighting.

Given two consecutive weight nudges (v8→v9) hadn't resolved this and the diagnosis pointed to training dynamics rather than magnitude, further blind reweighting was judged likely to just relocate the problem rather than fix it. Asked the user how to proceed; chose to **retrain fresh with a new random seed, no further reward changes** — testing whether this was simply an unlucky early-training trajectory rather than something inherent to the current weights. Launched as **v10** (`training_run_4M_reward_rebalanced_v10.log`), same weights as v9 (`reward_od=0.06`, `reward_harvest=4.0`, `od_delta` unchanged).

**v10 ran its full 4,000,000-step budget to completion and never advanced past D0 — but its trajectory tells a more nuanced story than a simple stall.** An early action trace (step 800,000, chunk 8) found a genuinely healthy, non-collapsing policy — harvest fraction ~0.25-0.27 (vs. v9's 0.30-0.36), stable/growing population (282→223→374), OD oscillating healthily 0.0035-0.0049 — directly confirming the "seed-specific bad luck" hypothesis for v9's collapse. `time_avg_od` climbed accordingly through chunk 13, peaking at 0.0031 (0% crash the entire time). But from chunk 13 onward it reversed: six straight chunks of decline (0.0031→0.0013) coinciding with a completely flat `ep_rew_mean` (101, unchanged for 10+ readings), then — after budget exhaustion was flagged to the user, who chose to keep monitoring — a **genuine late-run collapse** in the back third: det crash rate climbed steadily from 0% to 40% (chunks ~30-40), det `time_avg_od` fell to near-zero (0.0004-0.0011), and the final raw episode trace showed the culture nearly extinct (Active~19-28 cells, OD~0.0003-0.0006) at episode end.

So v10 demonstrates a third distinct failure shape this session: healthy early progress → mid-run stall → late-run collapse into rising crash rate, different from both v9's over-harvest-fraction plateau and the earlier zombie/washout pattern. Checkpoint archived to `model_data/archive_v10_D0_late_collapse_4M/` (final state) — the healthier chunk-13 peak was not separately archived and is not recoverable from the checkpoint directory post-completion.

## Recommended next step (updated again)

Across v8, v9, and v10, the common thread is that the *current reward shape* (relative `od_delta`, rebalanced `reward_od`=0.06/`reward_harvest`=4.0) produces policies that can reach a healthy-looking intermediate state but do not reliably hold or build on it — v8/v9 stalled or regressed early, v10 progressed further before collapsing late. This now looks less like a single fixable weight and more like a genuine training-stability question with this reward configuration. Untried directions worth considering before another blind attempt:
1. **Longer `n_steps`/rollout horizon** so PPO's advantage estimates span more of the delayed OD-building payoff relative to the immediate per-harvest-event reward (the credit-assignment gap identified during v9's investigation).
2. **Checkpoint-interval evaluation with early stopping/selection** — since the healthiest point in a run (v10's chunk 13) can come well before the run's end and isn't captured by the current "final checkpoint only" save discipline, consider tracking the best-seen chunk and defaulting to that rather than whatever's live when the budget runs out or a run is stopped.
3. Given three consecutive runs (v8, v9, v10) under this exact reward configuration failed to cleanly clear even D0→D1, further training attempts without a structural change are unlikely to be a good use of budget — this is the point to pause and reassess rather than keep iterating on seeds.

Standing validation discipline unchanged: no advancement or mastery claim should be trusted without both gates agreeing, and `held_out_sweep.py`/`test_actions.py`/`zombie_diagnosis.py`-style independent checks remain essential — demonstrated repeatedly this session (v6's collapse, v7's harvest/reward_od imbalance, v8/v9/v10's stalls and collapse) as catching real problems chunk-level logging alone doesn't fully surface. The `test_actions.py` trace specifically was what cracked open the v9 diagnosis after two rounds of weight-nudging alone failed to.

## Consolidation (v11): what to keep, what to revert, and why

After v8/v9/v10 all failed under the rebalanced reward, the whole reward-fix arc was reviewed and consolidated into a single confidence-ranked decision rather than a fourth weight guess.

**Kept (high-confidence structural wins):**
- **Deterministic-eval gate** (Fix #5) — caught every false-positive "mastery" this session; the single most important structural change.
- **Relative `od_delta`** — population-size-invariant OD-movement term; fixed the harvest-collapse, made every deterministic validation episode positive (vs. the old −6.88 mean) and cut zombie rate 33%→15%.
- **4-term simplification** and the flat-washout removal.

**Reverted (the session's clearest mistake):**
- `reward_od` **0.06 → 0.15** and `reward_harvest` **4.0 → 0.5** — back to v7's exact values. The rebalance was motivated by a real *static* measurement (reward_od's ~180× per-episode ceiling advantage in a fixed-action sweep), but empirically it made everything worse: v8 froze at D0 for 8 chunks, v9 over-harvested (0.30–0.36 fraction) and stalled, v10 (fresh seed) couldn't clear D0 and collapsed late. By contrast v7 — relative `od_delta` + the original 0.15/0.5 — produced the session's best, independently-validated result (genuine D1, hovering at the D2 edge, harvesting healthily throughout).

**The methodological lesson that drove the revert:** the fixed-action static reward sweep proved to be an *anti-predictor* of trained behavior. v7's static landscape favored never-harvest yet v7 harvested fine; v8/v9/v10's static landscape favored restraint yet v9 over-harvested. Reward-weight surgery driven off that sweep was therefore unsound — the revert restores the empirically-proven config and deliberately does **not** re-derive weights from the sweep.

**v11 result: it did NOT reproduce v7 — the run finished its full 4,000,000-step budget never advancing past D0.** `Curriculum ADVANCED` fired zero times across the whole run. The trajectory was genuinely eventful, not a flat failure: `time_avg_od` (the persistent bottleneck) repeatedly approached and even briefly touched/crossed the 0.0040 gate on both the stochastic and deterministic sides individually (e.g. det spiked to 0.0045-0.0059 at several points; stochastic hit 0.0041-0.0046 at chunks 14-15) but the two gates were never simultaneously above threshold on the same evaluation for two consecutive chunks (`MASTERY_REQUIRED_STREAK=2`), so `mastery_streak` never reached 2. Three entropy-plateau kicks fired (chunks 6, 13/18, 24/30, the last hitting the 1.30x hard cap with no further escalation available) without producing a sustained breakout — after the last kick, `time_avg_od` and harvest actually drifted down further before a late partial recovery that still fell short. Final state: harvest_mg=80.0 (comfortably clearing its own gate the whole run), `time_avg_od`=0.0035 (just under 0.0040). Checkpoint archived to `model_data/archive_v11_reverted_D0_stuck_4M/`.

One methodological note surfaced mid-run and worth remembering for future log-reading: the printed `(N/4 met)` on the `[D0->D1]` gate line reflects **only the stochastic side's four sub-criteria** — the deterministic gate is computed and required separately (`recurrent_ppo.py` line 350: `criteria_passed = criteria_passed and det_criteria_passed`) but isn't broken out in that printed line, so a chunk can show "(4/4 met)" and still not advance if the deterministic side quietly failed. This was directly confirmed by inspecting chunks 15-16 in v11's log, where "(4/4 met)" printed twice with `adv` staying at 0/2 both times because det `time_avg_od` (0.0040 rounding-boundary, then 0.0025 outright) hadn't independently cleared.

**Verdict on the consolidation itself**: reverting to v7's weights did not reproduce v7's success on a new seed — meaning v7's original result likely had a meaningful seed-luck component, not purely a deterministic consequence of that weight configuration. This doesn't undo the case for keeping the revert (the *rebalanced* config's 3-for-3 failure record — v8/v9/v10 — is still worse and better-understood as broken, specifically vs. v11's more marginal, closer-to-threshold stall), but it does mean the reverted config alone isn't a reliable fix either.

## PPO algorithm research + Fix #8 (v12): target_kl + manual LR decay

Given reward-weight guessing had run its course, researched (not just guessed) training-side PPO levers actually available in `sb3-contrib.RecurrentPPO` without added compute cost, targeted specifically at this session's demonstrated failure modes (huge single-seed variance, v10's late-training crash-rate blowup, a hand-engineered entropy-kick system that hasn't reliably helped). Confirmed via docs that `RecurrentPPO` natively supports `target_kl`, `use_sde`, and a callable `learning_rate` — all zero-cost (no new dependencies, no added rollout/compute). Ranked options: `target_kl` + LR decay (pure hyperparameters, lowest risk) → `use_sde` (same cost, bigger behavioral change, worth isolating in its own test) → `n_envs` parallelization (highest expected impact on the variance problem, but requires reworking `CurriculumStartController` for per-worker state — the only non-trivial item) → running N parallel seeds (directly treats the variance symptom but is a literal N× compute cost, not free) → off-policy algorithms like TQC/CrossQ (ruled out: real engineering cost adapting to recurrent+partial-obs+curriculum, uncertain payoff).

Implemented the first, lowest-risk pair:
- **`target_kl=0.02`** — early-stops a PPO epoch's gradient updates once the policy's mean approx-KL from its pre-update state exceeds this, guarding against the large destructive updates suspected in v10's late-run crash-rate blowup (0%→40% in that run's final third). SB3's commonly-cited starting point for continuous control; not yet tuned to this env specifically.
- **Linear LR decay, 5e-4 → 5e-5** — but implemented manually rather than via SB3's native schedule mechanism, after checking SB3's source directly: this codebase calls `model.learn(total_timesteps=100_000, reset_num_timesteps=False)` once per chunk, and SB3 computes `progress_remaining` relative to the `total_timesteps` argument of *that specific call*, re-deriving its own total each time as `num_timesteps_so_far + this_chunk_size`. Traced through the arithmetic: this makes `progress_remaining` restart near 1.0 at the start of every chunk and hit exactly 0 by the end of that same chunk — a 40-chunk sawtooth, not a smooth decay across the true 4M-step budget. A naive schedule would have silently driven the LR to its floor inside nearly every chunk rather than only near the actual end of training. Sidestepped using the same pattern this file already uses for `ent_coef`: an external mutable value (`_lr_state`), a schedule function that ignores SB3's own (here-meaningless) `progress_remaining` argument and returns whatever the training loop last wrote, refreshed once per chunk from the real `steps_done / TOTAL_TRAINING_STEPS` ratio. Verified before trusting it in a live run: constructed a throwaway small model with a callable LR schedule, updated the external state, called `learn()`, and confirmed `model.policy.optimizer.param_groups[0]['lr']` actually changed to the new value.

Launched fresh as **v12** (`training_run_4M_targetkl_lrdecay_v12.log`), same reward config as v11 (the reverted v7 weights), 4M-step budget. Isolating this change from `use_sde`/`n_envs` deliberately, so if it helps or hurts it's attributable to just these two levers.

**v12 result: a clear, sustained regression — stopped early at step 700,000 (17.5% of budget).** Both target_kl and LR decay were confirmed mechanically correct in isolation before this run (LR decay verified smoothly stepping 0.000500→0.000421 across 8 chunks with no jumps/sawtooth; target_kl confirmed actually firing). But the training signal itself was worse than v11 at every comparable point: deterministic-eval crash rate climbed steadily from ~50% (chunks 2-5) to **73.3%** (chunks 6-7) — v11's comparable early chunks ran a clean 0% crash under the identical reward config with no other change. `time_avg_od` declined the whole time (0.0015→0.0004, moving *away* from its 0.0040 gate) and det harvest also declined (43.4→31.6mg). `ep_rew_mean` sat deeply negative and flat (~-50 to -53) instead of the healthy early climb v11 showed. An entropy plateau-kick fired at chunk 6 (to the 1.30x cap) and was immediately followed by crash rate getting *worse*, not better (53.3%→73.3%) — the opposite of its intended effect. Stopped and archived to `model_data/archive_v12_targetkl_lrdecay_regression_800k/` rather than continue burning budget on a run trending the wrong way.

**Working hypothesis for the regression, not yet confirmed**: `target_kl=0.02` is the more likely culprit of the two changes. The log showed frequent per-minibatch early-stopping ("Early stopping ... max kl: 0.03–0.09") on most iterations, meaning PPO's nominal 4 epochs were regularly cut short — plausibly starving the policy of the gradient steps needed to correct crash-prone behavior during early, fast-changing training, exactly when full updates matter most. The LR decay is less likely to be the cause this early since by chunk 8 it had only drifted to 0.000421 (84% of its 5e-4 start) — not yet meaningfully different from v11's constant 5e-4 at a comparable point.

**Fix #9 (v13): isolating the two v12 changes.** Reverted `target_kl` to `None` (disabled) while **keeping** the LR decay, to test the hypothesis directly — if crash rate returns to v11's clean 0% pattern with LR decay alone, that confirms `target_kl=0.02` was the specific problem (and a looser value, e.g. 0.05-0.1, could be tried later as a separate, single-variable test rather than being bundled with LR decay again). If crash rate stays elevated even with `target_kl` removed, the LR decay itself (or some other factor) would need reconsideration instead. Launched fresh as **v13** (`training_run_4M_lrdecay_only_v13.log`), same reward config, 4M-step budget.

Same lesson as the reward-weight guessing earlier this session, reapplied here: change one variable at a time, and don't bundle a new fix on top of an unconfirmed one.

**v13 result: hypothesis confirmed — `target_kl=0.02` was the specific cause of v12's regression.** Through 6 completed chunks (~612,000 steps), v13 (LR decay only, `target_kl=None`) stayed completely healthy: deterministic-eval crash rate **0.0% across all 6 evaluations**, `ep_rew_mean` climbing steadily and monotonically (55→87.7) — a stark contrast to v12's 73.3% crash rate and flat/negative reward (~-50) at the identical point, with every other variable (reward config, LR schedule, seed-launch conditions) held the same. The LR decay itself is cleared as a cause; the frequent per-minibatch early-stopping from `target_kl=0.02` is confirmed as what actually broke v12. `time_avg_od` remains the persistent D0→D1 blocker (3/4 gate criteria met every chunk, same pattern as v7/v11), which is expected and unrelated to this fix — it's the known, separate bottleneck this whole reward-config lineage has shown, not a new problem.

Standing takeaway for any future `target_kl` attempt: don't reintroduce it bundled with another change — if revisited, test a substantially looser value (e.g. 0.05-0.1, since SB3's cited 0.01-0.05 range evidently runs too tight for this env's per-minibatch KL spikes) in complete isolation, with the same "one variable at a time" discipline that resolved this.

**v13 milestone: D0 → D1 genuinely advanced at chunk 18 (step ~1,800,000)** — the first curriculum advancement across v11, v12, and v13. Both gates cleared cleanly and simultaneously: `Curriculum ADVANCED: D0 -> D1 | harvest_mg=53.7 p25=34.2 time_avg_od=0.0090 crash=0.00%`. The path there: det `time_avg_od` finally broke its long D0 stall over chunks 16-18 (0.0038→0.0048→0.0071→0.0090, a real climb, not noise) while det harvest — which had been declining for several chunks (68.6mg down to a low of 34.0mg) as `time_avg_od` rose — flattened out just above its 30mg floor rather than continuing to crash, and det crash rate stayed a clean 0% throughout the entire climb. `ep_rew_mean` continued rising through the transition (105→148). LR decay continued its smooth schedule throughout with no anomalies.

**v13 outcome: peaked at D1, then a genuine late-training collapse — stopped at step 3,200,000 (80% of budget).** At D1, det `time_avg_od` climbed toward the 0.008 D1→D2 threshold, peaking at 0.0067 (chunk 25, harvest healthy, crash still 0%) — close, similar to how D0's stall eventually broke. But rather than crossing the gate, it reversed: 0.0067→0.0044→0.0028→0.0026→0.0015→0.0013, its lowest values of the entire run, over chunks 25-32. Critically, this coincided with det crash rate — a clean 0% for every single chunk through chunk 28 — appearing at chunks 29-30 (6.7%) and then **doubling to 13.3%** for chunks 31-32, with no reversion back toward 0%. `ep_rew_mean` drifted down in step (161→153). The entropy-plateau-kick mechanism fired to its 1.30x hard cap at chunk 30, and crash rate got *worse* immediately after (6.7%→13.3%), not better — the same "kick backfires" pattern seen with v10 and v12. This is the same late-training collapse shape as v10 (which climbed 0%→40% crash in its final third) — stopped this run at the first confirmed multi-chunk escalation (13.3% held for 2 chunks, no automatic corrective lever left) rather than let it run further, per the cost-aware, catch-it-early discipline established after v10.

Two checkpoints archived for reference:
- `model_data/archive_v13_D1_final_late_collapse_3.2M/` — the final state at stop time (step 3,200,000, chunk 32), reflecting the degraded, mid-collapse policy. Not recommended for use.
- `model_data/archive_v13_D1_peak_health_2.5M/` — the checkpoint from step 2,500,000 (chunk 25), at det `time_avg_od`'s peak (0.0067) and still 0% crash. Caveat: only the model weights are from that exact step — the paired `recurrent_vec_normalize.pkl` is the run's *final* (post-collapse) normalization statistics, not a snapshot from step 2.5M, since this codebase only checkpoints the model on a rolling 10k-step cadence and doesn't separately snapshot `VecNormalize` stats mid-run. VecNormalize stats usually stabilize well before this point in training, so the mismatch is likely small, but this hasn't been verified — treat this checkpoint as informative, not deployment-ready, without first re-validating it (e.g. `test_actions.py`) using its actual paired norm stats if those can be recovered, or by recalibrating fresh.

**Net read on v13**: same overall shape as v11 — real progress (a genuine, validated D0→D1 advance, better than v11's D0-only result) followed by a ceiling it couldn't sustainably push past, this time manifesting as a late-training collapse rather than a persistent stall. Three attempts now (v11, v12, v13) under this same reward-config lineage (relative `od_delta`, `reward_od=0.15`, `reward_harvest=0.5`) have each found a different way to fall short of D2, none reaching it cleanly. Given the training-side hyperparameter research this session (`target_kl`, LR decay) has now been tested and the LR-decay-only combination still hit a late collapse, the remaining higher-leverage unexplored lever is likely `n_envs` parallelization (reducing the single-trajectory variance that's plausibly at the root of these late-training divergences) — still not attempted this session, and flagged earlier as the highest-effort item of the four researched options.

## Action-trace diagnosis of v12 and v13 (`test_actions.py`, deterministic, seed=0)

Ran `test_actions.py` directly against the archived checkpoints to see what the deterministic policy actually *does*, not just what the aggregate metrics say, and cross-referenced against each log's entropy-multiplier trajectory.

**v12 (crash-regressed, `target_kl=0.02`, D0 checkpoint, step 700k)**: the deterministic policy is a rigid, near-zero-variance bang-bang controller — `Stir` pinned at raw=**exactly** -1.000 (50 RPM, the minimum) and `Light` pinned at raw=**exactly** +1.000 (2000 µmol, the max) with **0.000 std** for the *entire 7,200-step episode*, while `Harvest` sits at a roughly-constant 0.36-0.37 fraction with almost no modulation (std 0.001-0.007). This constant over-harvest against a population that's given no chance to recover (stir/light never adapt) drives a monotonic die-off: population 381→334→137→49→21→15 over the episode — this specific seed survives to the 7,200-step limit without technically hard-crashing, but is most of the way to zero. This is a qualitatively different failure from v13's: not erratic or unstable, but a *collapsed, unresponsive* policy — consistent with the target_kl hypothesis that aggressive per-minibatch early-stopping starved the network of the gradient signal needed to learn adaptive (population-responsive) control at all, leaving it stuck outputting saturated boundary actions. The entropy log confirms the timing: a `[PLATEAU]` kick pushed the multiplier to 1.00x right at chunk 6 (line 3596 in the log), and det crash rate is what spiked immediately after (53.3%→73.3%) — same "kick makes it worse" pattern documented earlier.

**v13 peak-health (2.5M) vs. final (3.2M, post-collapse), same D1 seed**: unlike v12, both v13 checkpoints show a genuinely *adaptive*, population-responsive harvest policy — `Harvest` starts high (~0.40-0.43 frac) while population is large, then throttles down as it depletes (peak: 0.43→0.39→0.26→0.14→0.07→0.03; final: 0.41→0.41→0.40→0.30→0.11→0.07) — this reactive shape survived the later collapse, it isn't what broke. What differs is the *outcome* of that shared strategy: peak's population recovers strongly once harvest backs off (102→155→275→**532** by episode end, OD climbing to 0.0086, reward 105.4) while final's recovery is much weaker from a similar trough (31→54→**78**, OD stuck at 0.0012, reward 87.9) — the same reactive policy, but materially worse at judging *when/how much* to back off and less able to convert the pause into real regrowth. `Light` also diverges: peak keeps light near the 1000µmol midpoint throughout, final dips it to ~600µmol during the early over-harvest window before drifting back — a subtler miscalibration, not a collapse to a fixed boundary like v12.

Cross-referencing entropy: chunk 25 (the peak-health snapshot) sits right after a plateau-triggered reset to **0.70x** multiplier (down from a prior high-band correction to 0.25x at chunk 23) — i.e., peak health coincides with a *recently-lowered*, not elevated, exploration level. From there the multiplier climbed steadily through chunks 26-31 (0.75→0.80→0.85→0.90→0.95→**1.30x, hard cap** at chunk 30) with no further high-band correction, and det crash rate (0% through chunk 28) appeared and doubled almost exactly as the multiplier pinned at its cap. **The pattern is consistent across both v12 and v13: sustained entropy escalation to (or near) the 1.3x hard cap directly precedes the crash-rate emergence, and in both cases a fresh, lower entropy state coincides with the run's healthiest observed point.** This doesn't yet prove causation, but it's the second independent case this session of the entropy-plateau mechanism's hard-cap corner being where things go wrong rather than right, and it's a concrete, falsifiable thing worth watching in v14 (does another cap-out precede another decline?) rather than a new hypothesis to test in isolation.

## v14 (resume of v13, budget extended 4M → 8M)

Rationale: rather than starting a 4th fresh attempt at the same D0-plateau problem, extend the one run that actually reached D1 and briefly threatened D2 (v13) with more budget, per the "see if it can reach D2 with time" framing — this also gives the entropy/crash pattern above more runway to either resolve on its own or repeat, which is itself informative.

`TOTAL_TRAINING_STEPS` raised 4,000,000 → 8,000,000 (`curriculum_schedule.py`). Resumed directly from v13's final archived checkpoint (`model_data/archive_v13_D1_final_late_collapse_3.2M/`, step 3,200,000, chunk 32, D1, mastery_streak=0) via `--resume <path>`.

**Operational note, worth remembering**: the plain `--resume` (no path) auto-detects "latest checkpoint" by scanning `model_data/recurrent_checkpoints/` for the highest step-numbered `*_steps.zip` file — but that directory is shared across this project's *entire* history, not just the current run, and isn't cleaned between differently-launched attempts. On the first v14 launch attempt, auto-detect picked up a stale `recurrent_ppo_ibm_5590000_steps.zip` (leftover from an older, unrelated experiment — likely the pre-session `archive_continuous_D_5M_stuck_D0` run) while `recurrent_training_state.pkl` (a separate, fixed-path file) still correctly held v13's bookkeeping (steps_done=3,200,000, chunk 32, D1) — i.e. it silently paired mismatched model weights with mismatched training state. Caught before any training happened by checking the printed `[CONTINUE] Loading checkpoint:` line against the printed `steps=` value and noticing the checkpoint's step number didn't match. Fixed by passing `--resume <explicit path, no .zip>` to bypass the auto-detect entirely. **Any future resume should pass an explicit checkpoint path rather than relying on bare `--resume`** unless `model_data/recurrent_checkpoints/` is known to hold only the current run's history.

A second, unrelated snag on the first launch attempt: an immediate `UnicodeEncodeError` on the file's box-drawing banner characters (`─`) when stdout was redirected to a log file under the default Windows `cp1252` console encoding. Fixed by launching with `PYTHONIOENCODING=utf-8`.

v14 launched successfully: `[CONTINUE] steps=3,200,000 | D1 | streak=0 | completed_eps=448`, chunk 33 started with LR correctly recomputed against the new 8M budget (`progress_remaining=0.600`, LR=0.000320 — consistent with 3.2M/8M done). Log: `training_run_8M_v14_resumed_from_v13.log`. Monitoring in progress.

### v14 milestone: D1 → D2 advanced cleanly at chunk 74 (step ~7,400,000) — the first clean D2 advance of the whole reward-fix arc

The run's D1 stretch (chunks 33-73) was eventful, not a flat hold. A summary of the path there:

- **Chunks 33-38**: crash rate briefly spiked to 16.7% (chunk 34) — likely residual instability carried over from resuming v13's already-degraded, mid-collapse policy — then fully recovered to 0.0% by chunk 39-40, even as the entropy multiplier passed through its 1.30x hard cap multiple times (chunks 37, 42, 49, 55, 60, 66, 72) without a single one triggering the "entropy cap → crash climbs" collapse pattern seen in v10/v12/v13. This is a meaningful divergence from that pattern worth remembering: hitting the entropy cap is not inherently destabilizing — v13's collapse specifically was not simply "entropy hit the cap," since v14 hit that same cap at least 7 separate times with no recurrence.
- **Chunks 42-44**: the stochastic D1→D2 gate hit 4/4 for the first time, with `time_avg_od` running 2-3.5x over its own gate — but the deterministic-eval gate's own `time_avg_od` stayed below its 0.008 threshold (0.0067), correctly blocking what would have been a stochastic-only false-positive advance (same mechanism that protected v11 and v5 earlier this session).
- **Chunks 44-49**: det `time_avg_od` then drifted *down* to a new low (0.0023) even as the stochastic side stayed comfortably 4/4 — a real divergence between the two gates, not noise.
- **Chunks 50-58**: det `time_avg_od` recovered and then broke cleanly above 0.008 for the first time (chunks 58-59: 0.0093, 0.0105) — but det `harvest_mg` then became the new blocker, first oscillating, then declining sharply to a low of 11.6mg (chunk 66) before partially recovering.
- **Chunks 68-73**: det `harvest_mg` and `p25` both climbed steadily (16.9→60.9, 6.9→38.3), finally clearing their 60.0/30.0 gates simultaneously with `time_avg_od` and crash rate at chunk 73 — all 4 deterministic criteria passing together for the first time all run, building `adv=1/2`.
- **Chunk 74**: a second consecutive fully-passing chunk (det harvest_mg=60.9, p25=38.3, time_avg_od=0.0114, crash=0.0%) completed the streak: `Curriculum ADVANCED: D1 -> D2 | eps=14 harvest_mg=106.4 p25=35.1 time_avg_od=0.0139 crash=0.00%`.

Training continued into D2 (chunk 75 onward); the first D2-tier deterministic eval already looked healthy (harvest_mg=73.7, p25=65.5, time_avg_od=0.0113, crash=0.0%).

**This is the first D1→D2 advance in the entire session where both the stochastic and deterministic gates were required to (and did) genuinely agree** — unlike the original pre-session run and v4, which both advanced/declared mastery on stochastic-only evidence and failed `held_out_sweep.py` badly. It is also the first clean advance in the post-v7 reward-config lineage (v11, v12, v13 all stalled or collapsed before reaching it). **Standing caveat unchanged: this is not yet a trustworthy mastery claim.** `held_out_sweep.py` (40 seeds) and `test_actions.py` must be run independently before this checkpoint is treated as validated — this session's two prior "D2 mastery" claims (original run, v4) looked just as convincing at this stage of their own logs and both failed held-out validation. That validation is planned once the run either reaches D2 mastery or exhausts its 8M-step budget.

### v14 final outcome: completed its full 8,064,000-step budget, D2 held crash-free throughout — but held-out validation reveals the advance was a false positive

The run completed normally (`Training Complete`, model and norm stats saved) at step 8,064,000 (slightly past the 8M nominal budget, due to PPO's rollout granularity — the chunk loop only checks the budget between chunks, and the last chunk's `n_steps=7200` rollouts overran the exact target by 64,000 steps). D2 held crash-free for its entire ~1.2M-step duration (chunks 75-80, every deterministic eval showed crash=0.0%), and the live curriculum's own deterministic-eval gate reported all 4 D2 criteria passing repeatedly and strongly by the end (final in-training det eval: harvest_mg=202-338mg vs a 90mg gate, p25=88-194mg vs a 50mg gate, `time_avg_od`=0.0277-0.0281 vs a 0.011 gate). Archived to `model_data/archive_v14_D2_ADVANCED_8M_final/`.

**`held_out_sweep.py` (40 seeds, D2, deterministic) fails this checkpoint decisively:**
```
crash_rate: 0.0%
harvested_mg  median: 0.4   p25: 0.0   min: 0.0   max: 25.7   (gate: median>=90.0, p25>=50.0)
time_avg_od   median: 0.0150   p25: 0.0105                     (gate: >=0.011 — this one clears)
holds on held-out sample: NO
```

`test_actions.py` (deterministic, seed=0) confirms the mechanism directly: `Harvest` sits at raw≈-1.00 (decoded to 0.00 frac) with near-zero std for the **entire 7,200-step episode** — the policy never harvests at all. OD instead climbs unboundedly the whole episode (0.0087→0.0308 by the end, population growing to 2,249 cells), and cumulative reward climbs smoothly to 149 purely from `reward_od`/`reward_biomass` compounding on unconstrained biomass growth. **This is the exact same degenerate "never harvest, let biomass grow forever" failure documented for the original pre-session run and v4** — the reward function still doesn't make sustained harvesting strictly better than never harvesting under every condition, and the deterministic policy has once again found the exploit.

**This is a significant new finding: the deterministic-eval gate — the session's single most important structural fix, and the mechanism that correctly blocked false-positive advances in v11 and v5 earlier this session — was itself fooled here.** The in-training det eval reported strongly passing harvest numbers (200-338mg) right up to and after the D1→D2 advance, yet a broader 40-seed held-out sample run immediately afterward against the identical saved checkpoint shows a median of 0.4mg. Both use the same cold-start distribution (`_sample_init_cells("random", difficulty)`: lognormal(100,400) init cells, 10% adversarial 30-80 at D2) so that isn't the explanation. The most likely candidate, not yet confirmed: `deterministic_eval.py` evaluates against a live, continuously-updated `obs_rms` snapshot (`copy.deepcopy(obs_rms)` from the still-training `vec_env`), while `held_out_sweep.py`/`test_actions.py` load the last-saved `recurrent_vec_normalize.pkl` from disk — if the two ever diverge even slightly, observation normalization mismatches could shift the policy's effective inputs enough to flip its behavior. This has not been root-caused with certainty and is flagged here as an open problem, not a solved one: **the deterministic-eval gate needs its own validation against `held_out_sweep.py`-style broader sampling before it can be fully trusted as a stand-alone advancement gate** — it caught real problems earlier this session but is not fully bulletproof against this specific exploit.

**Net read on the whole v11→v14 arc**: no version has yet produced a D2-tier checkpoint that both advances cleanly through training-time gating *and* survives independent held-out validation — v6 advanced fast but on a reward bug and collapsed inside D2; v14 advanced slower, on genuinely fixed reward terms and a working entropy schedule, held D2 crash-free for its full duration, and *still* turned out to be the same degenerate never-harvest policy underneath. The structural fixes this session made (relative `od_delta`, the deterministic-eval gate, `target_kl`/LR-decay tuning) are all real and each closed a genuine problem — but the fundamental reward-shape issue flagged back in the physics/reward audit (`reward_od`/`reward_biomass` rewarding unconstrained growth with no ceiling, while `reward_harvest`'s per-episode ceiling is structurally tiny by comparison) remains unresolved and appears to be the actual root cause behind three separate degenerate-policy incidents now (original run, v4, v14).

## Fix #10 + #11 (v15): reshaping `reward_od` and fixing `reward_od_delta`'s near-zero noise — verified before training, not guessed

Before touching any code, checked directly whether the near-zero action variance seen in v12's/v14's deterministic policies could actually be optimal under the current reward — i.e. whether this was a training problem at all. A fresh `dynamic_profile_sweep.py` run (physics-only, no RL, fixed stir=80/light=1000, sweeping harvest fraction) confirmed it decisively: `frac=0.00` scored **300.6**, the highest of any fraction, falling monotonically as harvest increased, bottoming near the physically-best `frac=0.15` (161.4). **Three independent trained policies (the original pre-session run, v4, v14) converging on the identical "never harvest" behavior is not a training bug — it is the actual reward-maximizing policy.** No hyperparameter or entropy fix could have changed this.

Measured the OD range a genuinely healthy periodic-harvest trajectory operates in before designing the fix (rather than guessing a target): a fixed-action `frac=0.15/stir=80/light=1000` episode oscillates OD with median 0.0095, p10-p90 0.007-0.012 — nowhere near the 0.03-0.07+ range the degenerate never-harvest policy runs to.

**Fix #10 — `reward_od` reshaped, not just reweighted.** `0.15*tanh(od/0.20)` is monotonically increasing with no ceiling short of saturation (~od=0.6+); its raw per-episode ceiling (0.15×7200=1080, dense every step) dwarfs `reward_harvest`'s (0.5×12=6, only 12 discrete event-steps/episode) enough that growing OD forever always beats harvesting it back down, independent of any hyperparameter. Replaced with a peaked target-band shape: `0.15 * x * exp(1-x)` where `x = od/OD_TARGET`, `OD_TARGET=0.012` (the top of the measured healthy-trajectory band). This rises toward the full weight at the target, then **decays** for OD beyond it (~74% of peak at 2× target, ~9% at 5× target) — standing OD above the healthy range now actively stops paying rather than merely plateauing.

Note on methodology: this differs from the v8-era reward_od reweighting that was "the session's clearest mistake" — that was a single static-sweep-driven weight guess that turned out to be an anti-predictor of real trained behavior (v7 harvested fine despite the sweep favoring never-harvest; v9 over-harvested despite the sweep favoring restraint). This fix instead (a) is motivated by a pattern independently confirmed by three separate real training runs, not a single static sweep, and (b) was verified by re-running the same sweep against the new reward shape *before* committing to a training run — confirming the fix actually flips the landscape rather than assuming it would.

**Verification, post-fix**: `frac=0.20` (1115.3) and `frac=0.15` (1071.9) now both substantially beat `frac=0.0` (630.1) — the optimum of the reward landscape sits right at the physically best-yield fraction, not at zero. `frac=0.50` still crashes (100%), confirming the over-harvest sanity check is undisturbed. Static-sweep verification is necessary but not sufficient (per the v7 lesson above) — the real test is the upcoming training run.

**Fix #11 — `reward_od_delta`'s near-zero-OD noise, flagged in the physics/reward audit, now fixed.** The term divided by `(prev_od + 1e-6)`; in the near-zero-OD "zombie" regime (od~1e-6-1e-5 — exactly the regime this term exists to guide recovery from), physically-insignificant noise in `delta_od` produced relative changes of order 1+, saturating `tanh(rel_delta_od/2e-4)` to a near-random ±1 sign every step — pure noise exactly where the signal should matter most. Fixed by flooring the denominator at `OD_RATE_FLOOR=1e-4`: identical behavior for od well above that floor (the vast majority of normal operation), graduated and non-random near zero.

**Deliberately not touched this round** (both previously flagged, both explained in-place in `genetic_env.py`'s comments as needing their own dedicated work, not a quick patch bundled into this run):
- **Bicarbonate ceiling** (clipped to 5.0mM vs. the 200mM Zarrouk baseline) — widening it requires re-deriving several interlinked carbonate-system constants together (`Kc_HCO3`, conductivity formula, Henderson-Hasselbalch terms), already confirmed via direct testing to break pH equilibrium if done naively.
- **Deterministic-eval gate's blind spot** (fooled by v14's exploit) — real, but not root-caused with confidence yet; the standing `held_out_sweep.py`/`test_actions.py` requirement remains the actual safety net regardless of what the training-time gate itself reports.

## v15: fresh run under the reshaped reward (not resumed)

Launched fresh rather than resumed from v14 — the reward function's shape fundamentally changed (not just reweighted), so a policy and value function trained entirely under the old landscape would need to unlearn the old optimum rather than build on it; same reasoning applied at every previous reward-shape change this session (v6→v7, v10→v11).

### v15 result: the reward fix worked — genuine, sustained harvesting behavior, clean D0→D1 advance, and by far the healthiest sustained run of the session — but a late-training crash-rate escalation forced stopping it in D1

**The core fix is validated.** Det harvest_mg climbed steadily from the very first evals (37.6→51.6→83.6mg through D0) and stayed in a healthy, non-degenerate 50-240mg range for the rest of the run — never once collapsing toward zero the way v4/v14/the original run did. `test_actions.py`-style behavior (inferred from the sustained, non-trivial harvest across hundreds of episodes) confirms the policy is actually harvesting, not exploiting unconstrained growth. D0→D1 advanced cleanly at chunk 13 with genuine dual-gate agreement (harvest 56.9mg, `time_avg_od`=0.0046, crash 0.00%) — the healthiest D0→D1 transition of the whole session.

**D1 lasted from chunk 14 to chunk 69 (~5.5M steps) and was remarkably stable for almost that entire span**: crash rate held at a clean, unbroken 0.0% across roughly 50 consecutive deterministic evaluations — by far the longest clean streak this session produced, surviving at least 9 separate entropy-plateau cap-touches (hitting the 1.30x hard cap repeatedly) without the "cap triggers crash" pattern that afflicted v10/v12/v13 recurring in any lasting way. `time_avg_od` oscillated for a very long stretch just below its 0.0080 D1→D2 gate (typically 0.001-0.008), with harvest/p25 comfortably clearing their own gates throughout — a genuine capability plateau on the OD-sustaining criterion specifically, not a health problem. It briefly cleared the gate on both sides at chunks 65-66 (`time_avg_od`=0.0089, 0.0087, full 4/4 stochastic gate met) — the closest any run got to a real D1→D2 push — but the deterministic side's own `time_avg_od` (0.0013 at the same point) never independently cleared 0.0080, so `mastery_streak` correctly never built and D1→D2 never advanced.

**Then, starting at chunk 66, crash rate appeared and escalated for four consecutive evaluations — 6.7%→13.3%→20.0%→26.7% (chunks 66-69) — coinciding with a fresh entropy-plateau boost back to the 1.30x hard cap.** This is the same shape of late-training collapse documented for v10 and v13 earlier this session, though it took roughly 10x longer to develop and never approached their severity (v10 reached 40%, v13 13.3%) before being caught. Per the standing "stop once a multi-chunk escalation is confirmed, don't wait for budget exhaustion" precedent set with those runs, **the run was stopped at step 6,900,000 (86% of its 8M budget)** rather than let it continue degrading.

Two checkpoints archived:
- `model_data/archive_v15_D1_peak_health_6.5M/` — end of chunk 65 (step 6,500,000), the last chunk with both a clean 0% crash rate and the gate's best-ever `time_avg_od` reading (0.0089, full stochastic 4/4). Same caveat as prior peak-health archives: paired `VecNormalize` stats are the run's final (post-escalation) ones, not an exact snapshot from step 6.5M.
- `model_data/archive_v15_D1_final_crash_escalation_6.96M/` — final state at stop time (step 6,960,000, chunk ~69), reflecting the mid-escalation, degraded policy (26.7% crash). Not recommended for use standalone; kept for the complete record, alongside the full training log.

**Net read on v15**: this is the strongest evidence yet that the reward-shape hypothesis was correct — genuine harvesting behavior sustained across an entire ~6.9M-step run, the longest unbroken crash-free stretch of the session, and a real (if ultimately unconsummated) D1→D2 push. The remaining problems are narrower than before: (1) the D1→D2 `time_avg_od` gate remains a stubborn, long-standing plateau independent of the reward-shape fix, suggesting it may be a genuine capability ceiling rather than a reward-landscape artifact — worth revisiting whether the 0.0080 threshold itself is well-calibrated (a question flagged earlier this session and still unresolved); and (2) the late-training crash-escalation pattern first seen in v10/v13 has now recurred a third time, always in apparent proximity to an entropy-plateau cap-touch, though this run also survived roughly 8 prior cap-touches unaffected — the relationship between the plateau mechanism and eventual instability is real but not deterministic, and remains only partially understood.

### Post-mortem: is the D1 `time_avg_od` plateau a real physical ceiling? No — `dynamic_profile_sweep_od.py` shows a wide, comfortable joint-optimum region

Before deciding whether to resume v15 or attempt a v16, checked whether the D1 gate's two thresholds (`median_harvested_mg>=60`, `median_time_avg_od>=0.008`) are even jointly achievable, since v15's own deterministic-eval trace showed them moving in opposite directions over the last ~30 D1 chunks (harvest recovering into the 60-240mg range while `time_avg_od` decayed steadily from a session-peak 0.0135 down to 0.0010, bottoming out right as the crash escalation began).

Extended `dynamic_profile_sweep.py` into a new read-only physics probe, `dynamic_profile_sweep_od.py` (D1 physics, stir=80rpm/light=1000umol fixed, 4 seeds/frac, reporting median `harvested_mg` and `time_avg_od` per fixed harvest fraction):

| frac | harvest_mg | time_avg_od | crash% | gate (h>=60, od>=0.008) |
|---|---|---|---|---|
| 0.00 | 0.0 | 0.0735 | 0.0% | fail (no harvest) |
| 0.05 | 98.9 | 0.0495 | 0.0% | **PASS** |
| 0.08 | 135.6 | 0.0384 | 0.0% | **PASS** |
| 0.10 | 146.7 | 0.0316 | 0.0% | **PASS** |
| 0.12 | 152.6 | 0.0255 | 0.0% | **PASS** |
| 0.15 | 153.3 | 0.0202 | 0.0% | **PASS** |
| 0.18 | 152.5 | 0.0144 | 0.0% | **PASS** |
| 0.20 | 147.5 | 0.0118 | 0.0% | **PASS** |
| 0.25 | 133.5 | 0.0073 | 0.0% | fail (od just under) |
| 0.30 | 115.6 | 0.0040 | 0.0% | fail |

Under a **static, held-fixed** harvest fraction, both D1 gate criteria pass simultaneously and comfortably across a wide 0.05-0.20 frac range — harvest 99-153mg (65-155% over the 60mg gate) and `time_avg_od` 0.0118-0.0495 (48-518% over the 0.008 gate). There is no physical tradeoff cliff here: it's a broad, easy joint-optimum plateau, not a narrow knife-edge. This rules out "genuine capability ceiling" as the explanation for v15's failure to sustain both criteria together.

It also happens to line up with the reward function's own optimum: `reward_od`'s peaked target-band shape (`OD_TARGET=0.012`) is maximized when `time_avg_od` sits near 0.012 — closest to frac=0.20 (od=0.0118, essentially at-target) — which *also* clears the D1 gate with margin on both axes. Reward-optimal and gate-passing are not in tension under a static policy.

**Conclusion: the D1 plateau is a training-dynamics problem, not a reward-shape or physical-capability problem.** v15's deterministic policy never settled into this broad safe window — instead its frac (inferred from the anti-correlated harvest/OD trace) appears to have drifted across chunks rather than holding steady, spending time at points off this plateau. This points toward the same entropy-plateau-cap-touch mechanism already flagged as correlating with instability: repeated forced increases in action-sampling variance may be what's preventing the policy from converging onto (and holding) the wide, otherwise perfectly achievable frac≈0.10-0.20 operating point. Next step, if pursued: investigate whether a lower entropy hard cap (or slower cap-driven escalation) lets a resumed/fresh run settle into and hold this window rather than oscillating through it.

## Fix #12 + v16/v16b: bounding the entropy plateau kick, and a decisive new diagnosis

**Fix #12 had two halves. One was based on an analytical error and did nothing; the other worked and produced the session's first fully-clean 8M-step run.**

### The error, stated plainly

`STD_BAND_LOW` was lowered 0.20 -> 0.08 on the reasoning that v15's policy was "pinned at the controller's floor": `test_actions.py` reported a decoded harvest std of 0.05, and since the raw->frac map has slope 0.25 (raw [-1,1] -> frac [0,0.5]), 0.05 decoded "==" 0.20 raw == exactly `STD_BAND_LOW`.

**That inference was wrong.** `test_actions.py` runs with `deterministic=True`, so its 0.05 measured how much the *mean* action varied across timesteps within an episode — not the width of the sampling distribution. `STD_BAND_LOW` gates `train/std`, the policy distribution's std parameter, which is a different quantity entirely. The numerical agreement was coincidence.

Confirmed empirically by v16b: `train/std` descended to ~0.49 early and then settled at **0.528** for the rest of the run, never approaching 0.20 let alone 0.08. The lowered floor never bound once. **This half of Fix #12 was inert.**

### The half that worked: `MAX_PLATEAU_KICKS_PER_DIFFICULTY = 2`

Fired and exhausted exactly as designed:

    [PLATEAU] 6 chunks with no streak - kick budget exhausted (2/2 used at D1), allowing convergence (mult held at 1.00x)

**v16b completed its full 8,064,000-step budget (80 chunks) with stochastic crash rate 0.00% throughout — the first run of the session to survive its entire budget cleanly.** Compare v10 (escalated to 40%), v13 (13.3%), v15 (26.7%, forced an early stop at 86% of budget). Deterministic crash rate peaked at only 6.7%. Bounding the *number* of kicks (rather than only their ceiling, which had already been lowered 3.6x->2.0x->1.3x across prior runs) appears to be what finally stopped the late-training collapse pattern. This is a real result and should be kept.

### Operational incident: two concurrent training processes invalidated the first v16 attempt

The original v16 launch used `nohup python ... &`; the command reported exit 1 with a missing-log error, which was misread as "it didn't start". It had started. A second launch 96 seconds later meant **two independent processes ran for ~20 hours, both writing to the same log file and — worse — the same `checkpoint_dir`, `state_path`, and `norm_path`.**

This produced a log in which chunk headers from one process interleaved with chunk summaries from the other: chunk 11 reading `D0` in its header and `D1` in its summary, `plateau` jumping 4/6 -> 2/6, `[Det] eps=` dropping 15 -> 6, and no `Curriculum ADVANCED` line surviving anywhere. Considerable effort went into hunting a nonexistent state-machine bug before checking the process list. **The curriculum state machine was never at fault** — v16b's log shows `plateau` incrementing cleanly 1..5 and a proper `Curriculum ADVANCED: D0 -> D1` line. Earlier single-process runs (v14, v15) were never affected; a concern raised at the time that their advance narratives might be unreliable is withdrawn.

Artifacts quarantined to `contaminated_v16_dual_process/` (moved, not deleted). **Operational rule going forward: verify the process list after launching, never trust the launcher's exit code alone, and assert a single startup banner in the log on every monitoring cycle.** Note `recurrent_ppo.py:74-76` hardcodes the checkpoint/state/norm paths, so two concurrent runs will always corrupt each other silently.

### v16b outcome: full budget, stayed in D1, and the deterministic gate correctly vetoed 22 stochastic passes

The **stochastic** gate hit 4/4 met on ~22 separate chunks, including a run of **9 consecutive** — far more than the 2-chunk streak required to advance. It never advanced, because the dual gate also requires the deterministic side, and the deterministic side never came close:

| | harvest_mg (D1 gate >=60) | time_avg_od (D1 gate >=0.008) |
|---|---|---|
| stochastic rollouts | 85-212 | 0.0080-0.0126 |
| deterministic eval | **20-48** | **0.0018-0.0058** |

A 3-5x divergence in harvest yield. **The deterministic-eval gate did exactly the job it was added for** — it blocked 22 chunks' worth of stochastic passes that were riding on sampling noise. Without it, v16b would have advanced to D2 and then failed held-out validation, precisely as v4 and v14 did.

Independent validation of the archived final model (`model_data/archive_v16b_D1_fullbudget_8M/`, norm stats matched to the model) confirms the deterministic read:
- `held_out_sweep.py` (n=40, D1): median harvested_mg **21.8** (p25 17.4, range 6.6-70.8), median time_avg_od **0.0029**, crash 2.5% — fails the D2 gate decisively, and fails D1 too.
- `test_actions.py` (4 seeds, D1): agrees with the 20-48mg training-time figure.

### The new exploit — created by Fix #10 itself

`test_actions.py` decoded harvest fraction per 600-step block reveals a failure mode that is *not* the old "never harvest from step 0":

| seed | harvest frac, block 1 -> block 12 | final OD | reward |
|---|---|---|---|
| 0 | 0.41, 0.29, 0.19, 0.15, 0.12, 0.10, 0.06, 0.03, 0.01, 0.00, 0.00, 0.00 | 0.0095 | 957.4 |
| 1 | 0.46, 0.31, 0.15, 0.04, 0.04, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00 | 0.0118 | 986.8 |
| 2 | 0.46, 0.31, 0.13, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.04, 0.14 | 0.0237 | 1027.6 |
| 3 | 0.47, 0.46, 0.36, 0.26, 0.22, 0.12, 0.13, 0.11, 0.11, 0.08, 0.19, 0.10 | 0.0033 | 672.6 |

It **starts above the optimum (~0.45) and decays monotonically to ~0.00-0.04 by mid-episode** — a within-episode drift, not a constant degenerate setpoint. And critically, the other two action dimensions converged to: **stir ~51 RPM (the 50 RPM floor) and light ~440-490 umol (roughly half the 900-1000 used in every sweep so far).**

That combination is the exploit. Fix #10 replaced the monotonic `tanh(od/0.20)` with a peaked target-band reward maximized at `od = OD_TARGET = 0.012`. But there are **two** ways to park OD at 0.012:

1. Grow fast (high light) and harvest at frac ~0.18 to hold it there — the intended behaviour.
2. **Throttle light down so the culture grows slowly, and never harvest at all** — OD coasts up to ~0.012 and sits there.

Route 2 collects nearly the same `reward_od` while doing no harvesting work, and `reward_harvest` cannot punish it: at weight 0.5 firing on only 12 event-steps, its entire per-episode ceiling is **6**, against `reward_od`'s ~1080 — a 180:1 imbalance driven by a 600:1 frequency asymmetry (7200 dense steps vs 12 event steps). The reward totals confirm the mechanism: seeds land at 957-1028 reward with near-zero harvest, and seed 3 (OD 0.0033, furthest from target) is the only one that scores poorly at 673.

So Fix #10 closed the "grow unboundedly" exploit and opened a "coast at target OD on low light" exploit. Both share the same root cause, which has been visible in this file's own comments since the earliest reweighting attempts: **`reward_harvest` is structurally incapable of competing with a dense per-step term, at any weight that doesn't cause over-harvesting.** Weights of 2.0 and 4.0 were tried and reverted for exactly that reason.

### RETRACTED: "a constant-action controller beats every learned policy"

**This claim was made, then disproved within the hour. It is recorded here rather than deleted because the error is instructive.**

The claim was: a constant-action controller (stir=60, light=900, frac=0.18) scores **1116.8** reward at D1 with harvest 139.9mg, time_avg_od 0.0131, 0% crash — clearing even the D2 gate — and therefore every learned policy in the project is worse than a constant.

**The measurement came from the wrong environment.** `dynamic_profile_sweep_od.py` instantiates the BARE `GeneticPhotobioreactorEnv` at a FIXED `initial_cells=300`. The training stack wraps it in `ActionSmoothnessWrapper` -> `CurriculumStartWrapper` -> `Monitor` -> `VecNormalize`, and `CurriculumStartWrapper` draws `initial_cells` log-uniformly per episode (100-400 "low", 600-1500 "mid", 2000-5000 "high", plus a 10% 30-80 cell adversarial draw at D2).

Running the *identical* constant action through the *real* training stack (11 episodes, D0/D1/D2 mix):

    30.7, 41.3, 46.3, 57.3, 63.4, 93.3, 95.8, 186.0, 248.3, 347.8, 368.2  mg
    median ~93mg, a 12x spread, time_avg_od 0.0025-0.0326

So the constant controller does **not** robustly clear D2, and only marginally clears D1. The 139.9mg figure was one favourable point on a distribution whose spread swamps the effect being measured.

**The substantive lesson is bigger than the retraction.** The episode outcome is dominated by the cold-start draw, not by the action: the same fraction strips a small culture before it establishes (46mg at od 0.0054) and under-harvests a large one (368mg at od 0.0326). **A constant harvest fraction is therefore not the right policy at all** — the task genuinely requires conditioning on current biomass, harvesting little while the culture builds and steadily once established. Any comparison of a learned policy against a constant baseline must be run through the same wrapper stack and the same start distribution, or it measures the initial-condition lottery instead of the policy.

This also partially rehabilitates the learned policies: they face a start distribution the sweep never exposed them to. It does not excuse v16b's harvest-decaying-to-zero behaviour, which remains a genuine failure, but the "worse than a constant" framing was wrong.

### Recommended next step: behaviour-cloning warm start, not another reward tweak

Literature search surfaced a directly applicable precedent: an industrial photobioreactor RL deployment (arXiv 2509.06853) bootstraps its agent from PID-controller trajectories before any online RL, specifically because bioprocess RL from scratch needs either massive data or an accurate simulator and risks catastrophic early actions. Also relevant: "Evaluation-Aware Reinforcement Learning" (arXiv 2509.19464) documents exactly the stochastic-train/deterministic-evaluate gap seen here, and notes **the gap widens on long-horizon tasks** — these episodes are 7200 steps, which plausibly explains why this failure has recurred in four separate runs.

The case for BC survives the retraction above, but the EXPERT had to change. Cloning a constant fraction is off the table: it would teach the policy to ignore its observation on roughly half of all episodes, given that the same constant yields anywhere from 31mg to 368mg depending purely on the cold-start draw.

The expert is instead a **proportional surplus-harvest feedback law** (`bc_pretrain.py`):

    frac = clip(GAIN * (od / OD_SETPOINT - 1), 0, FRAC_CAP)     GAIN=1.0, OD_SETPOINT=0.015, FRAC_CAP=0.30

Below the setpoint it harvests nothing and lets the culture establish; above it, it removes roughly the excess. This drives `time_avg_od` toward the setpoint *regardless of where the episode started* — which is precisely what the curriculum's `time_avg_od` criterion asks for, and what a constant fraction structurally cannot deliver. `OD_SETPOINT=0.015` sits above the D2 gate's `time_avg_od>=0.011` with margin and above `genetic_env`'s `OD_TARGET=0.012` (the peak of `reward_od`), so the controller holds the culture in the band the reward function pays most for.

`bc_pretrain.py --eval-only` scores this expert against the real training start distribution and **refuses to clone it unless it clears the D1 gate** — cloning a sub-D1 expert guarantees a sub-D1 policy, and an 8M-step run could only confirm that at great cost. Tuning the three controller constants this way costs minutes and no GPU time.

Deliberately **not** recommended: another `reward_harvest` reweighting (tried at 0.25/0.5/2.0/4.0, reverted twice), and the entropy-collapse methods from the LLM-RL literature (EPO/DAPO/Clip-Cov) which target the opposite problem — entropy collapsing too fast, whereas here `train/std` sat flat at 0.528 all run.

**Still open:** whether the dense-vs-sparse reward frequency asymmetry should be addressed structurally (e.g. a dense term tracking cumulative harvest progress, or making `reward_od` conditional on recent harvest activity) rather than worked around by BC initialisation. BC fixes where the policy starts; it does not change the fact that route 2 above remains reward-competitive, so a BC-initialised policy could still drift back toward it.

## v17: behaviour-cloning warm start — best run of the session, and still fails held-out validation

### The expert redesign (see the retraction above for why the constant expert was abandoned)

`bc_pretrain.py` clones a proportional surplus-harvest feedback law:

    frac = clip(1.0 * (od / 0.015 - 1), 0, 0.30)

Scored against the REAL training start distribution (n=20, D0/D1/D2 mix, through the full wrapper stack):

    median harvest 153.3mg  p25 135.5  min 105.5  max 331.4
    median time_avg_od 0.0162   crash 0.0%
    vs D1 gate: PASS      vs D2 gate: PASS

Every one of the 20 episodes landed in time_avg_od 0.0159-0.0166 despite starting populations spanning 100-5000 cells — the feedback law removes the initial-condition lottery on precisely the criterion that blocked every prior run. `mean_frac` ranged 0.039-0.147, i.e. it genuinely adapts.

BC converged cleanly (MSE 0.0676 -> 0.0038 over 10 epochs). The clone's own deterministic rollouts: harvest 147-181mg, time_avg_od 0.0171-0.0185 — clearing D2 thresholds before any RL.

### v17 training outcome: the best result this project has produced

- **First legitimate D0->D1->D2 progression.** D0->D1 at chunk 4 (vs chunk 13 for v15 and v16b). D1->D2 with harvest 211.7mg, time_avg_od 0.0219, crash 0.00%.
- **Completed the full 8,064,000-step budget with 0.00% stochastic crash rate throughout** — including 48 consecutive D2 chunks. Combined with v16b, this confirms `MAX_PLATEAU_KICKS_PER_DIFFICULTY` fixed the late-collapse pattern for good.
- **Both previously-diagnosed exploits eliminated.** The v4/v14 "never harvest" mode: gone (frac settles ~0.18, not 0). The v16b "throttle light and coast" mode: gone (light ~1000umol across all seeds, vs v16b's ~445).
- In D2 the stochastic gate showed harvest 113-262mg (vs 90 needed) and p25 53-163 (vs 50) — but `time_avg_od` 0.0037-0.0080 against the 0.011 threshold. **3/4 met on all 48 D2 chunks, always the same criterion failing.**

### Held-out validation: FAILS, at both tiers

`held_out_sweep.py`, n=40, against the archived final model (norm stats matched):

| | median harvest | p25 | median time_avg_od | crash | verdict |
|---|---|---|---|---|---|
| D2 (gate 90 / 50 / 0.011 / 8%) | 47.6 | 31.5 | 0.0030 | 0.0% | **FAIL** |
| D1 (gate 60 / 30 / 0.008 / 10%) | 49.2 | 32.5 | 0.0029 | 0.0% | **FAIL** |

### Why: the policy inverted the expert's phase structure

`test_actions.py` at D2, decoded harvest fraction per 600-step block:

    seed 0:  0.28 0.28 0.24 0.22 0.20 0.20 0.20 0.19 0.19 0.19 0.18 0.19
    seed 1:  0.30 0.27 0.27 0.29 0.24 0.21 0.20 0.19 0.19 0.19 0.18 0.18
    seed 2:  0.25 0.26 0.24 0.24 0.20 0.21 0.20 0.19 0.18 0.17 0.17 0.17
    seed 3:  0.27 0.24 0.25 0.21 0.22 0.20 0.21 0.19 0.19 0.19 0.19 0.19
    (stir ~53-60rpm, light ~1000umol)

The expert harvested ~0 EARLY and Rose as OD climbed. v17 does the reverse: **starts at 0.25-0.30 and declines to a ~0.18 plateau.** It kept the "harvest a lot" lesson and discarded the "wait for the culture to establish first" lesson — and the second half is the one that holds OD up.

This is exactly why held-out fails: the held-out sample's cold starts are small (init 108, 179, 210 cells in the tail episodes), and harvesting 25-30% of a 108-cell culture destroys it before it can build biomass. In-training stochastic rollouts reported 113-262mg because the training start distribution includes many "mid"/"high"/"stitched" starts (600-5000 cells) that tolerate early over-harvesting.

### The BC benefit decayed during fine-tuning

Deterministic eval, first D2-capable chunks vs end of run:

    chunk 1-4:  harvest 113.0, 103.0, 113.0, 98.9mg   time_avg_od 0.0215, 0.0204, 0.0215, 0.0199
    end of run: harvest  72.0,  80.0mg                time_avg_od 0.0023, 0.0022

So the D1->D2 advance was **earned** by a genuinely good policy (both gates passed while the clone was still largely intact), and the policy then degraded across the following 48 D2 chunks. This is NOT a v14-style gate deception — the gate was honest at the moment it fired. The answer to the experiment's question is therefore:

**BC's benefit does not persist under PPO fine-tuning with the current reward. The policy drifts away from the demonstrated behaviour.** That is the "drift" branch of the v17 experiment, which means the dense/sparse reward imbalance is now implicated **by evidence rather than inference**, and a structural reward change is justified.

### Structural gap found: demotion is crash-rate-only

`recurrent_ppo.py:428` — demotion triggers solely on `stats["crash_rate"] >= DEMOTION_CRASH_RATE`. Crash rate was 0.00% for the entire run, so nothing ever demoted v17 out of D2 even as `time_avg_od` collapsed from 0.0215 to 0.0022. A policy that degrades on a capability criterion while remaining perfectly stable can sit at the top tier indefinitely, burning budget. Worth adding a capability-based demotion (e.g. sustained failure of the SAME criterion for N chunks) so the curriculum can walk itself back down.

### Unverified hypothesis, flagged not acted on

`reward_biomass = 0.20 * tanh(per_cell_growth / 5.0)` has a per-episode ceiling of **1440**, larger than `reward_od`'s 1080. Aggressive early harvesting leaves fewer cells competing, raising per-cell growth. If that outweighs the OD loss it would make early over-harvesting reward-positive through the one dense term never scrutinised in this project. This is plausible arithmetic, **not a measured finding** — it should be tested directly (e.g. `reward_breakdown.py` on a v17 rollout vs an expert rollout) before any change is made to that term. The same discipline that caught the v8 reweighting mistake applies.

### Recommended next step

Not another BC run — the clone was demonstrably good and PPO walked away from it, so re-cloning changes nothing. The evidence now points at the reward. In order:

1. **Measure, don't guess**: run `reward_breakdown.py` on a v17 deterministic rollout and on a scripted-expert rollout, and compare per-term totals. That directly tests the `reward_biomass` hypothesis and shows which term actually pays for early over-harvesting.
2. Only then make ONE structural change to the implicated term, verify the expert's advantage widens on the physics sweep, and re-run from the existing BC warm start (which is archived and reusable — no need to regenerate).
3. Add capability-based demotion independently of the above.

## v18: gamma + critic pretraining + capability demotion — FAILED, and ended worse than v17

### Two hypotheses measured and REFUTED before this run

Before changing anything, both candidate explanations for v17's drift were tested directly.

**`reward_ab.py`** (new) — runs the trained policy and the scripted expert through IDENTICAL episodes (same seed, same initial_cells, same difficulty) and compares per-term reward sums:

    term                  v17       expert   expert-v17
    od                  683.1        996.7       +313.6
    biomass              11.8         11.3         -0.4
    od_delta             68.8         68.1         -0.7
    harvest               2.4          3.0         +0.5
    TOTAL               766.1       1079.1       +313.0

**The reward-exploit hypothesis is dead.** The reward ranks the expert +313 ABOVE v17, entirely via `reward_od`. Critically, `reward_biomass` contributes **11.8**, not the ~1440 its theoretical ceiling (0.20 x 7200) suggested — `tanh(per_cell_growth/5)` is tiny at realistic growth rates and the flat -0.010 penalty offsets most of the rest. The two policies differ by 0.4 on it. Had the "fix reward_biomass" plan gone ahead unmeasured, it would have edited a term contributing ~1% of episode reward. This is the same class of error as the v8 reweighting mistake, avoided only by measuring first.

**`noise_sensitivity.py`** (new) — adds Gaussian noise of varying sigma to each policy's raw action:

    sigma |  EXPERT reward   harv      od |  v17 reward   harv      od
     0.00 |        1072.1  176.9  0.0163 |      781.9   64.2  0.0049
     0.50 |        1016.4   50.8  0.0136 |      627.6   52.0  0.0025
     0.70 |         927.6   31.2  0.0081 |      561.5   43.7  0.0020

**The exploration-noise hypothesis is dead too.** The expert keeps 94.8% of its noise-free reward at sigma=0.50 — exactly the `train/std` v15/v16b/v17 all sat at — and dominates v17 at every sigma tested. No crossover. Entropy was therefore left untouched.

### What the evidence pointed at instead

v17 learned **stir and light correctly** (light settled at ~1000umol, the sweep optimum) and **only harvest incorrectly**. The distinguishing feature is credit frequency: stir/light act on all 7200 steps, while harvest is applied only on the 12 event steps (`HARVEST_INTERVAL_STEPS=600`) — on 7188 of 7200 steps the policy emits a harvest value the env discards while PPO still assigns it advantage. `gamma=0.995` compounds this: effective horizon 1/(1-gamma) = 200 steps, and `0.995^600 = 0.049`, so a harvest decision's immediate gain is undiscounted while its OD cost is ~95% invisible.

### The three fixes

- **Fix #13** — `gamma` 0.995 -> 0.9995 in BOTH the fresh-construction and resume paths. Horizon 200 -> 2000 steps; discount at 600 steps 0.049 -> 0.741. Not pushed to 0.9999 (horizon 10000 > the 7200-step episode) to avoid value-variance blowup.
- **Fix #14** — critic pretraining in `bc_pretrain.py`. v17 cloned only the actor, leaving a random value head. Now regresses the value head onto discounted returns-to-go at matching gamma, with a joint phase (`VALUE_LOSS_COEF=0.001`, protecting the actor from the O(100s) value targets) followed by critic-only refinement with the actor's parameters frozen. Value RMSE 14.4 on returns spanning 0.1-133.5 (~11% error), vs 24% after the joint phase alone.
- **Fix #15** — capability-based demotion (`CAPABILITY_DEMOTION_CHUNKS=12`). Demotes a tier after 12 consecutive DETERMINISTIC-gate failures, closing the gap where v17 held D2 for 48 chunks at 0.00% crash while degrading. `capfail=n/12` added to every chunk summary.

A real bug was caught pre-launch by reading the sb3-contrib source: `predict_values` returns a **bare tensor**, not a `(value, states)` tuple. The original `pred_v, _ = policy.predict_values(...)` would have silently unpacked along the batch dimension.

### v18 result: FAILED — three advance/demote cycles, ended in D0

The clone was validated on a 40-seed held-out sweep BEFORE launch (median harvest 79.1, p25 28.4, time_avg_od 0.0240, crash 0% — 3/4 on the D1 gate, p25 marginally short), so the starting point was known-good and a v18 failure would be attributable to the fixes rather than a bad warm start.

Completed all 80 chunks. The curriculum oscillated:

    ADVANCED  D0 -> D1   (harvest 52.8, time_avg_od 0.0086)
    [CAPABILITY DEMOTION] det gate failed 12 consecutive chunks at D1 (crash 0.00%)
    DEMOTED   D1 -> D0
    ADVANCED  D0 -> D1   (harvest 65.8, time_avg_od 0.0046)
    [CAPABILITY DEMOTION] ... 12 consecutive chunks ...
    DEMOTED   D1 -> D0
    ADVANCED  D0 -> D1   (harvest 76.1, time_avg_od 0.0041)
    [CAPABILITY DEMOTION] ... 12 consecutive chunks ...
    DEMOTED   D1 -> D0

Deterministic `time_avg_od` across the run: **0.0254 -> 0.0042 -> 0.0023 -> 0.0078 -> 0.0029 -> 0.0034 -> 0.0031 -> 0.0029 -> 0.0017 -> 0.0010**, ending at 0.0009-0.0012 with 6.7% det crash. Final state: D0, det harvest 35.9mg.

**This is worse than v17**, which at least reached D2 and held it with stochastic harvest 113-262mg. v18 never got past D1.

### Honest verdict on each fix

- **Fix #15 WORKS.** It fired three times, correctly identified that the policy could not sustain D1 at 0.00% crash, and demoted — exactly as designed, with an accurate diagnostic message. The mechanism is sound and should be kept. It cannot fix the underlying problem, only stop the budget being wasted on an unattainable tier.
- **Fix #13 (gamma) did NOT fix the drift.** The over-harvest / low-OD collapse recurred. Note the post-demotion D0 chunks: harvest 174.6, 161.3, 188.0mg with time_avg_od 0.0061, 0.0061, 0.0083 — the identical over-harvest signature v17 showed. Lengthening the horizon was not sufficient.
- **Fix #14 (critic pretraining) did NOT prevent the decay either.** det od still fell from 0.0254 to 0.0009 over the run.

Since #13 and #14 shipped together, this run cannot attribute the failure between them — only establish that the two combined are insufficient. That was a deliberate trade (the user asked for one best-shot run) but it does cost attribution.

### What remains unexplained, and the most likely next target

The reward demonstrably prefers the expert's behaviour (+313, measured). Noise does not explain the drift (measured). A longer horizon and a calibrated critic do not prevent it (this run). Yet PPO reliably walks from a ~1079-reward policy to a ~766-reward one, and always via the harvest dimension.

The strongest remaining candidate is the one structural fact none of these fixes touched: **on 7188 of 7200 steps the harvest action has no effect, yet still receives advantage.** 599 of every 600 gradient samples on that dimension are spurious credit — pure noise injected into exactly the dimension that keeps failing, while the two dimensions with clean per-step credit (stir, light) are learned correctly every time.

Recommended next step: make the harvest action's credit honest rather than tuning around it. Options, cheapest first:
1. Have the env HOLD the harvest action constant between events (latch the value emitted at the last event step), so the policy is not graded on values the simulator discards.
2. Failing that, restructure so harvest is decided only on event steps (e.g. a 2-dim continuous action plus an event-step-only harvest head).

Both are env/action-space changes rather than hyperparameter changes, which is a larger commitment — but every hyperparameter-level explanation has now been tested and eliminated.

## Fix #16 + v19: the credit fix WORKED on the policy, and PPO destroyed it anyway

### Fix #16 — interval-averaged harvest action

The harvest action was previously read ONLY on event steps (every `HARVEST_INTERVAL_STEPS=600`). On the other 7188 of 7200 steps the env discarded the emitted value while PPO still assigned those timesteps advantage and updated the harvest dimension toward whatever was sampled — **599 of every 600 gradient samples on that dimension were spurious**.

That is the one structural difference between the dimensions that work and the one that never has. Stir and light act every step, receive honest per-step credit, and were learned correctly in EVERY run (light reliably settles near the 1000umol sweep optimum). Harvest, the only 1-in-600 dimension, failed in every run and in a DIFFERENT direction each time (never-harvest / drift-up / coast-on-low-light / decay-to-zero / over-harvest-early) — the signature of a dimension driven by noise rather than gradient.

This was reached only after every hyperparameter-level explanation was measured and eliminated: reward exploit (`reward_ab.py`: reward prefers the expert +313), exploration noise (`noise_sensitivity.py`: expert dominates at every sigma), credit horizon (Fix #13's gamma 0.9995: drift unchanged).

**Change**: the harvest event applies the MEAN of the harvest action over the preceding interval. Physics unchanged — still one discrete dilution per 12h, same `F_MAX`, same washout cliff; only WHICH number the event reads changes. Verified: constant frac=0.15 gives 146.9mg/od 0.0194, matching the old sweep's ~143.5mg. An alternating 0.0/0.30 policy — which under the old code would have harvested NOTHING, since every event step landed on the 0.0 phase — now correctly yields 152.6mg.

### The fix demonstrably improved the policy, before any RL

Same expert, same BC procedure, same hyperparameters. Only the env changed:

| | v18 clone (old env) | v19 clone (interval-averaged) | D2 gate |
|---|---|---|---|
| median harvest | 79.1 | **109.4** | >=90 |
| p25 harvest | 28.4 | **63.8** | >=50 |
| median time_avg_od | 0.0240 | **0.0191** | >=0.011 |
| crash | 0% | **0%** | <=8% |
| held-out verdict | 3/4 FAIL | **PASS** | |

**This was the first held-out D2 pass in the entire project** — achieved by behaviour cloning alone, with no reinforcement learning.

Mechanism, in two parts (only the first was anticipated): (a) PPO's per-step credit on the harvest dimension becomes honest; (b) any imperfect policy's EXECUTED behaviour moves closer to its INTENDED behaviour, because the applied value averages ~600 samples instead of depending on whichever single noisy sample landed on the event step. Part (b) is why a cloned policy improved without any training. The expert itself also improved (median 175.1mg vs 153.6).

### v19 result: PPO destroyed the validated policy

Started from that D2-passing clone. Completed all 80 chunks.

    ADVANCED  D0 -> D1  (harvest 56.9, time_avg_od 0.0076)
    [CAPABILITY DEMOTION] det gate failed 12 consecutive chunks at D1 (crash 0.00%)
    DEMOTED   D1 -> D0

Deterministic trajectory across the run:

    harvest 149.1 -> 62.1 -> 56.4 -> 30.7 -> 50.2 -> 50.5 -> 47.3 -> 27.5 -> 28.3 mg
    od      0.0203 -> 0.0063 -> 0.0043 -> 0.0011 -> 0.0025 -> 0.0015 -> 0.0011 -> 0.0003 -> 0.0002
    crash      0% ->     0% ->     0% ->     0% ->     0% ->     0% ->  13.3% ->  60.0% -> 80-93%

Final state: D0, det harvest 28.3mg, time_avg_od 0.0002, **det crash rate 80-93%** — while the STOCHASTIC crash rate stayed at 0.0% throughout.

**PPO took a policy that passed held-out D2 validation and, over 8M steps, turned it into one that crashes 80-93% of deterministic episodes.** This is not a failure to improve; it is active, severe destruction.

### The conclusion this forces

Across v17, v18 and v19 the pattern is now unambiguous:

- v17: good clone -> reached D2 -> degraded to det 72-80mg / od 0.0022 -> held-out FAIL
- v18: good clone -> three D1 advance/demote cycles -> ended D0, od 0.0009
- v19: **D2-VALIDATED clone** -> one advance/demote cycle -> ended D0, od 0.0002, det crash 80-93%

Every fix that made the STARTING policy better produced a WORSE final policy. The best artifact this project has produced is `model_data/BEST_bc_clone_D2_validated/` — a behaviour-cloned policy with no RL applied, which passes the held-out D2 gate (median 109.4mg, p25 63.8, time_avg_od 0.0191, 0% crash over 40 seeds including adversarial cold starts).

**The RL fine-tuning stage is the problem, and it is not explained by any of: reward shape (measured), exploration noise (measured), credit horizon (tested), critic initialisation (tested), or harvest credit frequency (tested).** All five were addressed and the destruction continued.

### Note on the deterministic/stochastic divergence

The most striking unexplained signal: at the end of v19 the deterministic policy crashed 80-93% of episodes while the stochastic policy crashed 0.0%. PPO optimises the stochastic objective, which remained healthy the entire time — the stochastic gate never demoted, and stochastic harvest stayed in a normal range. The deterministic (mean) policy meanwhile collapsed completely. This is the training/evaluation mismatch of arXiv 2509.19464 in an extreme form, and it is consistent across every run in this project.

It also means the curriculum's stochastic gate is measuring something that can be fully decoupled from deployed behaviour, and only the deterministic gate ever detected the damage.

### Recommended next steps

1. **Ship the BC clone.** It is validated, it is the best policy produced, and it required no RL. If the goal is a working controller, this is it.
2. If RL fine-tuning is still wanted, the next thing to try is constraining the policy from moving away from the clone at all — KL-anchoring to the BC policy (a fixed reference-policy penalty), or a much lower learning rate with early stopping keyed on the DETERMINISTIC eval rather than the stochastic one. Every attempt so far has let PPO wander freely from a good initialisation.
3. Investigate why deterministic and stochastic behaviour decouple so completely. Until that is understood, any stochastic-objective optimiser will keep destroying deterministic performance while its own metrics look fine.
