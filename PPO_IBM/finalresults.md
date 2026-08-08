# Final Results — Reward/Curriculum Validation and Fix Cycle

> Condensed. Full detail, including every intermediate measurement, is in `finalresults_full_archive.md`.

## Bottom line

**The best policy this project produced is a behaviour-cloned controller with no reinforcement learning applied.** It is preserved at `model_data/BEST_bc_clone_D2_validated/` and passes the held-out D2 gate: median harvest 109.4mg, p25 63.8, `time_avg_od` 0.0191, 0% crash over 40 seeds including adversarial cold starts.

**No RL run ever produced a held-out-validated D2 policy.** Two runs (v14, v17) advanced to D2 in training and failed independent validation. Across v17/v18/v19 a clear pattern emerged: every change that made the *starting* policy better produced a *worse* final policy. The RL fine-tuning stage is destructive here, and five candidate explanations were measured or tested and eliminated.

## Run history

| run | change | outcome |
|---|---|---|
| v11 | reverted reward weights | D0 stuck, 4M |
| v12 | `target_kl=0.02` + LR decay | regression (det crash 73.3% by chunk 7); target_kl reverted |
| v13 | LR decay on true `steps_done` | D1, late collapse at 3.2M |
| v14 | resume of v13, budget 4M→8M | **D2 advanced** — held-out FAIL (median 0.4mg vs 90 gate) |
| v15 | Fix #10, #11 (reward reshape) | D1, genuine harvesting restored; crash escalation forced stop at 6.9M |
| v16b | Fix #12 (kick budget) | D1, **first full 8M budget at 0.00% crash** |
| v17 | BC warm start | **D2 reached**, full budget, 0% crash — held-out FAIL both tiers |
| v18 | Fix #13, #14, #15 | 3× D1 advance/demote cycles, ended D0 |
| v19 | Fix #16 (harvest credit) | clone passed held-out D2; PPO destroyed it → D0, 80–93% det crash |

## Fixes and verdicts

| # | change | verdict |
|---|---|---|
| **#10** | `reward_od`: monotonic `tanh(od/0.20)` → peaked band `x·e^(1−x)`, `x=od/0.012` | **Worked.** Killed the "never harvest, grow forever" exploit. Later *measured* correct. |
| **#11** | `reward_od_delta` denominator floor `1e-6` → `1e-4` | Sound; removed near-random sign flips at near-zero OD. |
| **#12a** | `STD_BAND_LOW` 0.20 → 0.08 | **Inert.** Analytical error — a deterministic action's temporal variation was misread as sampling std. `train/std` never fell below ~0.49. |
| **#12b** | `MAX_PLATEAU_KICKS_PER_DIFFICULTY = 2` | **Worked.** Ended the late-training collapse that killed v10/v13/v15. Three consecutive full budgets at 0% crash followed. |
| **#13** | `gamma` 0.995 → 0.9995 (horizon 200 → 2000 steps) | Did not stop the drift. |
| **#14** | Critic pretraining in BC (value RMSE 14.2) | Did not prevent the decay. |
| **#15** | Capability-based demotion after 12 det-gate failures | **Works mechanically.** Fired correctly in v18 (3×) and v19. Stops budget being burned at an unreachable tier; cannot fix the cause. |
| **#16** | Harvest event applies the interval **mean**, not the instantaneous sample | **Worked on the policy.** Produced the project's first held-out D2 pass — see below. |

## Three hypotheses measured and refuted

Each was tested before changing code, which is what stopped three wrong fixes.

**Reward exploit — refuted.** `reward_ab.py` runs the trained policy and the scripted expert through *identical* episodes:

```
term        v17      expert   expert-v17
od        683.1      996.7      +313.6
biomass    11.8       11.3        -0.4
od_delta   68.8       68.1        -0.7
harvest     2.4        3.0        +0.5
TOTAL     766.1     1079.1      +313.0
```

The reward ranks the expert **+313 above** the drifted policy. `reward_biomass` contributes **11.8**, not the ~1440 its theoretical ceiling (`0.20 × 7200`) suggested — `tanh(per_cell_growth/5)` is tiny at realistic growth rates. A planned "fix" to that term would have edited ~1% of episode reward.

**Exploration noise — refuted.** `noise_sensitivity.py`: the expert keeps **94.8%** of its noise-free reward at σ=0.50 (exactly the observed `train/std`) and dominates the trained policy at every σ from 0.0 to 0.70. No crossover. Entropy left untouched.

**Constant-action baseline — retracted.** An earlier claim that a constant controller beat every learned policy was measured on the *bare* env at fixed `initial_cells=300`. Through the real wrapper stack, where `CurriculumStartWrapper` draws initial cells log-uniformly (100–400 / 600–1500 / 2000–5000), the same action gives 31–368mg — a 12× spread. The outcome is dominated by the cold-start draw, not the action.

## Fix #16 in detail — the credit-assignment defect

The harvest action was read **only** on event steps (every `HARVEST_INTERVAL_STEPS=600`). On the other 7188 of 7200 steps the env discarded the emitted value while PPO still assigned those timesteps advantage — **599 of every 600 gradient samples on that dimension were spurious**.

This is the one structural difference between the dimensions that work and the one that never did. Stir and light act every step, get honest credit, and were learned correctly in *every* run (light reliably settles near the 1000 µmol optimum). Harvest failed in every run, and in a *different direction* each time — never-harvest, drift-up, coast-on-low-light, decay-to-zero, over-harvest-early. Different outcomes from the same setup is the signature of noise, not gradient.

**Change:** the event applies the mean of the harvest action over the interval. Physics unchanged — one discrete dilution per 12h, same `F_MAX`, same washout cliff. Verified: constant frac=0.15 → 146.9mg/od 0.0194, matching the old sweep (~143.5mg). An alternating 0.0/0.30 policy — which under the old code harvested *nothing*, since every event step landed on the 0.0 phase — now correctly yields 152.6mg.

**Effect on the policy, before any RL** (same expert, same BC procedure, only the env changed):

| | v18 clone | v19 clone | D2 gate |
|---|---|---|---|
| median harvest | 79.1 | **109.4** | ≥90 |
| p25 | 28.4 | **63.8** | ≥50 |
| median `time_avg_od` | 0.0240 | **0.0191** | ≥0.011 |
| held-out verdict | 3/4 FAIL | **PASS** | |

Two mechanisms, only the first anticipated: (a) PPO's per-step harvest credit becomes honest; (b) any imperfect policy's *executed* behaviour moves closer to its *intended* behaviour, since the applied value averages ~600 samples rather than depending on one. (b) is why a cloned policy improved with no training.

## v19: PPO destroyed a validated policy

Starting from the D2-passing clone, over 8M steps:

```
harvest 149.1 → 62.1 → 56.4 → 30.7 → 50.2 → 47.3 → 27.5 → 28.3 mg
od      0.0203 → 0.0063 → 0.0043 → 0.0011 → 0.0025 → 0.0011 → 0.0002
crash      0% →    0% →    0% →    0% →    0% →  13.3% →  80–93%
```

Final: D0, det harvest 28.3mg, `time_avg_od` 0.0002, **80–93% deterministic crash rate** — while the **stochastic crash rate stayed at 0.0% throughout** and the stochastic gate never flagged a problem.

## What was learned about the agent's action control

1. **Action dimensions are not equally learnable, and the split is structural.** Per-step dimensions (stir, light) were learned correctly in every run. The 1-in-600 dimension (harvest) failed in every run. Fix #16 removed that asymmetry and immediately improved the policy — but did not make RL safe.

2. **The correct policy is a feedback law, not a setpoint.** Because outcome is dominated by the cold-start draw, harvest must condition on current biomass. A constant fraction strips a 108-cell culture and under-harvests a 5000-cell one. The scripted expert `frac = clip(1.0·(od/0.015 − 1), 0, 0.30)` holds `time_avg_od` at 0.0159–0.0166 across starting populations spanning 100–5000 cells.

3. **The agent cannot observe what it needs.** The expert reads true `od`; the policy observes only *turbidity* (`od × pigment × clump × saturation`). Any clone is an approximation by construction — this caps achievable fidelity and is a strong candidate for future work (add OD or a better proxy to the observation).

4. **Deterministic and stochastic behaviour decouple completely, in both directions.** Det was 3–5× worse than stochastic in v16b, better in v17, and catastrophically worse in v19 (80–93% vs 0.0% crash). PPO optimises the stochastic objective, which stayed healthy while deployed behaviour collapsed. Only the deterministic gate ever detected damage — the dual gate is essential, and in-training stochastic numbers alone are never trustworthy.

5. **Late-training collapse is solved** (Fix #12b) — three consecutive full 8M budgets at 0% stochastic crash, versus earlier runs dying at 40%. This is the one unambiguous, durable win.

## Recommended next steps

1. **Ship the BC clone** (`model_data/BEST_bc_clone_D2_validated/`, with its validation logs). It is validated, it is the best policy produced, and it needs no RL.
2. If fine-tuning is still wanted, stop letting PPO wander from a good initialisation: **KL-anchor to the BC policy**, or use a much lower LR with early stopping keyed on the **deterministic** eval rather than the stochastic one.
3. **Root-cause the deterministic/stochastic decoupling.** Until it is understood, any stochastic-objective optimiser will keep destroying deployed performance while its own metrics look clean.
4. Consider adding OD (or a cleaner proxy) to the observation, so the policy can represent the feedback law it is being asked to learn.

## Tooling added

`bc_pretrain.py` (BC from the OD-feedback expert, actor + critic, with `--eval-only` gating), `reward_ab.py` (per-term reward A/B on identical episodes), `noise_sensitivity.py` (action-noise sweep), `dynamic_profile_sweep_od.py` (harvest-fraction sweep reporting both gate metrics and reward).

## Operational notes

- `recurrent_ppo.py` hardcodes `checkpoint_dir`/`state_path`/`norm_path`, so **two concurrent runs silently corrupt each other**. A dual-process launch invalidated ~20h of v16. Always verify the process list after launching and assert a single startup banner; never trust the launcher's exit code alone.
- Always pass an explicit `--resume <path>`; bare `--resume` scans a shared, never-cleared checkpoint dir and picked up an unrelated checkpoint at v14's launch.
- Prefix `PYTHONIOENCODING=utf-8` (Windows cp1252 crashes on box-drawing characters).
- No mastery claim is final without `held_out_sweep.py` and `test_actions.py`. v14 and v17 both passed both in-training gates and failed held-out validation.

## The 5-run series (v20-v24): PPO from scratch, and why it cannot reach D2

Five 8M-step runs, no behaviour cloning, with the BC controller's held-out scores printed
inline as a reference line. Verdict: **no run reached D2, and the reason is now understood
mechanistically rather than guessed at.**

| run | configuration | best-det harvest | p25 | **time_avg_od** | D1 |
|---|---|---|---|---|---|
| v20 | 6-channel obs | 89.0mg | 60.7 | 0.0054 | fail |
| v21 | 8-channel obs (Fix #18) | 101.9mg | 77.3 | **0.0094** | **PASS** |
| v22 | + fouling, pump error, periodic phase | 82.5mg | 65.8 | 0.0056 | fail |
| v23 | replicate v21 | **121.0mg** | **98.7** | 0.0066 | fail |
| v24 | + std annealing (Fix #22) | 67.8mg | 44.7 | 0.0035 | fail |
| — | BC clone (no RL) | 109.4mg | 63.8 | **0.0191** | PASS + **D2** |

Harvest yield is solved: v23's 121.0mg / p25 98.7 clears even the D2 thresholds (90 / 50).
`time_avg_od` is the entire obstacle — five runs span 0.0035-0.0094, mean ~0.006, against
D2's requirement of 0.011.

### Fixes introduced in this series

- **#17 (v20)** Best-deterministic-checkpoint tracking + BC reference line in the `[Det]`
  output. Immediately justified: v20's peak policy was **4.2x its final weights** (89.0mg vs
  21.0mg). v17/v18/v19 had all ended at or near their worst policy and silently discarded
  their best. Every run since has yielded its peak instead of its last.
- **#18 (v21)** Observation 6 -> 8 channels: long-window turbidity EMA + phase. The agent had
  **no OD, no biomass and no clock** in its observations; its only biomass signal was a
  turbidity reading that is nonlinear in OD (`saturation_factor = 1/(1+0.05*od)`),
  multiplicatively noised by the stir it sets itself (`flow_noise ∝ rpm`), and at D1+ lagged
  by an amount that also tracks stir. It was graded on time-averaged OD while barely able to
  observe OD. Best single change of the series: od +74% vs v20, and the first from-scratch
  policy to clear all four D1 criteria. Neither channel assumes new hardware — a filter over
  an existing sensor and the controller's own clock. True `od` was deliberately NOT exposed:
  it is simulator-internal and would not transfer.
- **#19/#20 (v22)** Nephelometer window fouling; harvest pump ±5% delivery error. Both
  realistic; #20 closed a real gap (the sim applied actuator noise to stir and nutrient flow
  but skipped the harvest pump, the one actuator the agent never learned to command).
- **#21 (v22)** Episode phase -> periodic harvest-cycle phase, removing a latent hazard:
  `time_avg_od` is scored over `step_count>=3600`, so episode phase told the policy exactly
  when the scoring window opened, and `held_out_sweep.py` could not have detected the
  resulting gaming (same episode length, same metric). An action trace of v21's best
  checkpoint showed **no** gaming, so v21's result stands.
- **#22 (v24)** Annealed hard cap on actor std, 0.65 -> 0.12 over the back half. See below.

### Methodological error worth recording

v21 -> v22 changed **three things at once plus the seed**, and v22 regressed. That confounded
attribution — the exact mistake avoided elsewhere in this project. Two of those changes also
worked against each other: **#18 improved the biomass signal (+74% od) and #19 degrades that
same signal (up to +10% bias)**, introduced in consecutive runs. The realism options were
subsequently made switchable class flags (`TURB_FOULING_COEF`, `HARVEST_PUMP_ERROR`,
`USE_EPISODE_PHASE`) so configurations can be isolated rather than bundled.

### v23 established that v21 was not reproducible

Replicating v21's exact configuration gave od **0.0066**, not 0.0094 (with *better* harvest,
121.0mg vs 101.9mg). Across four comparable runs od spans 0.0054-0.0094 with mean ~0.0065.
So v21's 0.0094 was the top of the distribution, not a stable ceiling 15% short of D2 —
the real gap to D2 is ~70%, not ~15%. Without this replication the series would have
concluded from a lucky draw.

### THE KEY FINDING: the exploration noise was the policy

Every run showed deterministic performance far below stochastic (v22: det 39.7mg/od 0.0086 vs
stoch 211mg/od 0.0209). The hypothesis was boundary-clipping: the harvest action is clipped at
0, so with the mean near that floor samples can only deviate UPWARD — the sampled policy
harvests while the mean does not (`E[f(x)] != f(E[x])`, asymmetrically). Supporting evidence:
`train/std` sat at 0.50-0.54 in **every** run, the reactive std-band controller never lowered
it in 13 runs, and v21 — whose mean harvest fraction was 0.16-0.18, off the floor — had the
smallest gap.

Fix #22 tested it directly by annealing a hard cap on std to 0.12. The annealer worked exactly
as scheduled (std tracked the cap: 0.526 -> 0.442 -> 0.361 -> 0.278 -> 0.179 -> 0.120).

**The gap closed — from 2.4-5x down to 1.3x. But it closed downward:**

    earlier runs (std ~0.54):  stoch od 0.0209  |  det od 0.0086
    v24 (std 0.12):            stoch od 0.0008  |  det od 0.0006

The deterministic policy did not rise to meet the stochastic one; the stochastic performance
**fell to meet the deterministic one**. The mechanism was correct and the conclusion drawn
from it was backwards.

**The exploration noise was not obscuring a good policy — the noise WAS the policy.** The agent
learned to sit near the harvest floor and let sampling variance perform the harvesting.
Remove the variance and the competence disappears: v24 is the worst run of the series
(best od 0.0035, ending at od 0.0006 with 47% deterministic crash).

This is the same degeneracy first seen in v4, but now established mechanistically by
intervention rather than inferred from correlation.

### Consequences

1. **The deterministic gate is emphatically necessary**, and for a deeper reason than it was
   built for. It was added to stop exploration noise inflating metrics. What it has actually
   been doing is refusing to certify a policy whose competence is *borrowed from its own
   exploration noise* and therefore cannot be deployed at all.
2. **Stochastic curriculum metrics are not merely optimistic — they can be entirely noise.**
   v22's stochastic side cleared D2 comfortably while the same policy, run deterministically,
   was below D1. No amount of stochastic-side tuning addresses this.
3. **PPO under this reward and observation plateaus at deterministic od ~0.006**, roughly 40%
   short of D2 and about a third of what the behaviour-cloned controller achieves with no RL
   at all (0.0191). Five runs, four configurations, one replication.

### Recommendation

**Ship the behaviour-cloned controller** (`model_data/BEST_bc_clone_D2_validated/`): median
harvest 109.4mg, p25 63.8, time_avg_od 0.0191, 0% crash over 40 held-out seeds including
adversarial cold starts — the only artifact in this project that passes held-out D2.

If RL is pursued further, the target is no longer a hyperparameter. The agent must be made to
learn a policy whose **mean** is competent, which means the training objective has to reward
the deterministic policy rather than the sampled one. Options, in order of directness:
1. Include deterministic-rollout return in the training objective (evaluation-aware RL,
   arXiv 2509.19464 — which documents exactly this train/eval mismatch and notes it widens on
   long-horizon tasks; these episodes are 7200 steps).
2. KL-anchor to the BC policy, so PPO refines a mean-competent controller instead of
   discovering a noise-dependent one.
3. Reconsider whether an LSTM policy trained by on-policy RL is the right tool at all: a
   proportional feedback law on OD, tuned in minutes, already outperforms every learned policy
   here and carries far less sim-to-real risk.
