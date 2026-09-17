# Novelty and Publishability Report

> Rewritten 2026-09-18, superseding the 2026-08-17 audit below in full. That audit predated
> every TD3 run in this project (v33 onward) and its central claim is now false: it asserted
> "no RL run ever produced a held-out-validated D2 policy" and recommended shipping a pure
> behaviour-cloned controller instead. TD3+BC has passed the D2 held-out gate on every criterion
> since v45 (2026-09-06), and the project's actual strongest contribution — an architecture-
> dependent recurrent-state-reset-cadence finding — did not exist as a candidate when the first
> version of this report was written. This is a search-based audit, not an exhaustive prior-art
> clearance — treat "not found" as "not found by this search," not as proof of absence.

## Bottom line

This project's best-supported claim is no longer the PPO-era negative result. It is now
**C1 below: a mechanistic diagnosis of recurrent-network collapse in long-horizon RL, a fix
that produced the project's first held-out-validated policy of any kind, and a follow-up
finding that the fix's correct cadence is architecture-dependent** — replicated at n=2 for the
bounded-state case. That chain is real, causal, and — per the literature search run alongside
this rewrite — currently unoccupied. The realistic target remains a **workshop paper or an
applied-domain journal**, not a flagship ML venue, because the empirical base is still one
simulator and 1-2 training seeds per configuration. Two secondary contributions (C2, C3) are
also citable, narrower boundary conditions. The PPO-era BC-vs-RL result (C4) is retained but
reframed: it no longer means "RL doesn't work here" (RL does, via TD3+BC); it means "RL
*fine-tuning of a fixed feedforward PPO policy* was destructive under this project's specific
credit-assignment defect," a materially smaller and more defensible claim.

## The contributions, individually assessed

### C1 — Recurrent-core collapse in long-horizon RL: mechanism, fix, and an architecture-dependent boundary condition

**What was done, across the whole TD3 line (v33 through v56):**

1. **Diagnosis (v44).** A recurring, previously-unexplained pattern of sudden mid-training
   det-eval collapses across six earlier runs (v33/v34/v40/v42/v43/v44) — each blamed on that
   run's own specific change at the time — was traced to a single shared mechanism: the LSTM
   actor's cell state grows **unbounded** over a 7200-step episode, saturating ~40% of hidden
   units by step 10-20 and freezing the actor (identical, corner-valued action regardless of
   input) for the remainder of the episode.
2. **Fix and first held-out pass (v45).** Resetting the recurrent hidden/cell state to zero
   every `HIDDEN_RESET_INTERVAL=60` steps during rollout and evaluation eliminated the failure
   mode entirely — v45 exhausted its full 2,000,000-step budget with zero collapse and became
   **the first policy of any kind in this project (PPO, TD-MPC2, or TD3) to pass the D2
   held-out validation gate on every criterion.**
3. **A second, independent collapse mode in a bounded-state variant (v48/v50/v52/v53).**
   Swapping the LSTM for a diagonal Linear Recurrent Unit (LRU) — bounded by construction, so
   it cannot suffer unbounded cell-state growth — nonetheless collapsed at high starting
   population across four separate attempts, under three different reward formulations and on
   both recurrent cores. This ruled out the reward shape as the sole cause (a directional
   reward fix, tested on both cores, made the collapse *worse* on both) and pointed at the
   architecture/protocol interaction instead.
4. **The architecture-dependent finding (v54, replicated v55).** `HIDDEN_RESET_INTERVAL=60` is
   a correctness requirement specific to the LSTM's unbounded state — it has no equivalent
   justification for a bounded core. Decoupling the *rollout* reset cadence from the *training*
   sequence window (`TD3_HIDDEN_RESET_INTERVAL=600` vs `SEQ_LEN=60`, avoiding the `O(T²)`
   training-cost blowup a matching increase in `SEQ_LEN` would cause) turned the LRU from
   reliably collapsing into reliably completing: v54 held D2 for 16 straight chunks through
   full budget exhaustion with zero collapse, and **v55, an exact replicate, reproduced this
   cleanly** (14 clean D2 chunks, held-out numbers consistent with v54 to within normal
   run-to-run variance). This is the project's only n=2 result and its most novel finding.
5. **In progress (v56):** a matched-reset LSTM run (`TD3_HIDDEN_RESET_INTERVAL=600` on the
   LSTM core) is running to complete the {LSTM, LRU} × {reset 60, reset 600} grid as a more
   controlled test of architecture-dependence — see `docs/decision_history.md` for the live
   result once it concludes.

**What the literature says.** A targeted search (2026-09-18) for hidden-state reset cadence,
context length, and bounded-vs-unbounded recurrent state in RL found no work addressing this
specific question. [RLBenchNet](https://arxiv.org/html/2505.15040v1) (2025) benchmarks
LSTM/GRU/Mamba/Mamba-2 for RL but runs no reset-cadence ablations, and explicitly flags
unresolved state *leakage* across episode boundaries for Mamba as an open limitation rather
than a studied one. [Yang & Nguyen's recurrent off-policy baselines](https://arxiv.org/abs/2110.12628)
(recurrent DDPG/TD3/SAC) never reset hidden state mid-episode at all, run no context-length
ablations, and name "inability to deal with too long episodes" as an inherited RNN weakness
they do not solve. General LRU-for-RL work found (e.g.
[real-time recurrent learning with trace units](https://arxiv.org/pdf/2409.01449)) is about
gradient-computation efficiency, not state-reset policy. R2D2-style stored-state/burn-in
targets a related but distinct problem — stale hidden states sampled from a replay buffer —
not rollout reset cadence for an on-policy-collected trajectory. **This appears to be a
genuinely open question**, though this is a search-based finding, not exhaustive clearance.

**Novelty tier: real and, on current search evidence, unoccupied — the strongest claim in this
project.** It is a complete unit: a mechanistic diagnosis that retroactively explains six prior
"unrelated" failures, a fix validated by the project's first-ever held-out pass, a second
independent failure mode in a different architecture that the same fix does not address, and a
targeted follow-up experiment (with a replication) that isolates *why* — the fix's necessary
cadence is a property of the architecture's state dynamics, not a universal constant. The
grid-completion run (v56) in progress will either strengthen this (if the LSTM degrades at
reset=600, as predicted) or produce an equally interesting revision (if it doesn't, suggesting
longer context helps regardless of core) — either outcome is reportable.

**Caveats, stated plainly:** n=1 for every cell of the grid except LRU/600 (n=2); one
simulator; no cross-architecture-family comparison beyond LSTM/LRU (no transformer, no S5/
Mamba); the free-running (no periodic reset at all) condition is only spot-checked, not run to
completion at either core. A prior training-seed variance measurement in this project (v21 vs.
v23, PPO era, nominally identical config) found ~30% spread from seed alone on a different
metric — a reminder that single-run claims anywhere in this project should be read with that
in mind.

### C2 — TD3+BC's behaviour-cloning anchor is load-bearing, not merely helpful, once the RL policy has already surpassed the expert

**What was done:** With `TD3_BC_COEF=0` (removing the BC term from the actor loss while
keeping demo transitions in the replay buffer as ordinary off-policy data) and a real
confound removed first — the `alpha/|Q|` actor-loss normalization is specific to TD3+BC and
was still silently damping the Q-gradient with no BC term to balance against; fixing that
was itself necessary before the ablation was clean — a formula-correct vanilla TD3 actor
reliably diverged to a degenerate, state-independent policy within 1-3 D0 chunks. This
matters specifically *because* the expert prior is no longer a performance ceiling at this
point in the project (TD3+BC policies out-harvest the scripted expert by 15-25% at high
population), so the ablation asks a live question rather than a rhetorical one.

**What the literature says:** that a stochastic-policy-gradient actor-critic method can diverge
without a behavioural anchor is textbook extrapolation-error theory, and TD3+BC's own paper
(Fujimoto & Gu 2021) motivates the BC term exactly this way. What is not generic is the
specific, controlled demonstration that removing *only* the loss term (not the data) causes
collapse in this domain, after ruling out a real implementation confound that could otherwise
have been mistaken for the effect.

**Novelty tier: a clean ablation, not a new mechanism.** Citable as a case study; does not
generalize beyond the demonstrated setting.

### C3 — Reward dead zones from bounded shaping terms

**What was done:** A bounded above-target OD penalty (`max(floor, decay(x))`) was shown to
reach *exactly zero gradient* past a calculable threshold (7.67× target under one formulation),
diagnosed by directly measuring that high-population episodes spent 53-90% of their steps
inside that zone. A subsequent fix (log-tail decay, never reaching zero) resolved it; a later,
separate attempt to fix a *different* reward term (`reward_od_delta`'s sign) was shown, via a
controlled test on both recurrent cores, to be independently harmful regardless of core —
isolating the true remaining mechanism as a 5-7× scale mismatch between an unavoidable
per-step state penalty and a capped per-event corrective reward, making the needed recovery
action effectively undiscoverable by gradient ascent alone.

**Novelty tier: a generalizable diagnostic pattern** (bounded shaping terms can silently zero
their own gradient exactly where the policy most needs signal), demonstrated with an unusually
complete before/after/root-cause trail, but not a new theoretical result.

### C4 — BC-then-RL fine-tuning was destructive for a specific, diagnosed reason (PPO era, reframed)

**What was done:** Across 24+ PPO runs and one full TD-MPC2 run, the only PPO-track artifact to
pass held-out D2 validation was a behaviour-cloned controller with **no RL fine-tuning
applied**, and every attempt to fine-tune it with PPO made it strictly worse (v19: 149.1mg →
28.3mg harvest, 0% → 80-93% crash, over 8M steps). The root cause was independently diagnosed
(Fix #16): the harvest action is only read on 1-in-600 timesteps, so PPO's per-step advantage
assignment was overwhelmingly spurious on that dimension specifically.

**This claim MUST now be stated narrowly.** It is not "RL fine-tuning doesn't work on this
task" — TD3+BC, a different RL method with an explicit anchor term and off-policy replay, both
avoids the collapse (C2) and beats the BC clone outright on held-out harvest (v45: 95.7mg;
v49: 135.8mg; both exceed the clone's 109.4mg). The defensible claim is narrower and more
interesting: **on-policy fine-tuning (PPO) with a severe, undiagnosed sparse-credit defect
destroyed a good BC initialization; off-policy fine-tuning with an explicit BC anchor and the
credit-assignment defect fixed did not.** Framed this way it is a real, useful boundary
condition on when RL fine-tuning helps vs. hurts a BC baseline, consistent with the contrast
noted against Gil et al.'s pH-control paper below, where RL fine-tuning (SAC+BC) *did* improve
on a PID/BC baseline by 8%.

### C5 — Harvest/dilution-fraction control specifically, for Spirulina (narrowed since the original audit)

**What was done:** The action space controls periodic biomass harvest fraction via a
semi-continuous harvest-and-dilute mechanism every `HARVEST_INTERVAL_STEPS`, rather than only
growth-condition setpoints (light/temperature/nutrients).

**What changed since the original audit:** a supervised-learning harvest-fraction paper
(**[Machine learning-based decision support for harvest optimization in commercial microalgae
photobioreactors](https://www.sciencedirect.com/science/article/abs/pii/S0168169926006356)**,
*Computers and Electronics in Agriculture*, 4 July 2026) was found in a follow-up search and
was not caught by the original audit (published after its literature cutoff, and this
project's own August search). It uses a Random Forest (R²=0.705) to predict next-day
post-harvest concentration and an inference module that scores candidate harvest fractions —
field-validated across 12 commercial *Nannochloropsis* reactors over 4 weeks, achieving
productivity comparable to (not exceeding) expert human operators.

**Novelty tier, revised: narrower than previously stated, but still real.** "RL applied to
microalgae harvest fraction" is no longer an unclaimed combination in the *broadest* sense —
ML-guided harvest-fraction *decision support* is now published and field-validated. What
remains open, on current search evidence, is **closed-loop, multi-step-credit-assignment RL
control** of harvest fraction specifically (as distinct from single-step supervised
recommendation, RL control of bacterial chemostat dilution generally — see Treloar et al.,
PLOS Comp Bio, already cited in the original audit — or RL control of microalgae
growth-condition setpoints without harvest, as in the UNIST Spirulina MARL work). Any writeup
must state the July 2026 paper explicitly to avoid an easy, embarrassing "this already exists"
rejection.

## Closest related work, examined in detail: Gil et al. 2025

**[Reinforcement learning meets bioprocess control through behaviour cloning: real-world
deployment in an industrial photobioreactor](https://arxiv.org/abs/2509.06853)** (Gil, Del Rio
Chanona, Guzmán, Berenguel; arXiv:2509.06853, Sept 2025) is the closest methodological sibling
to this project found in either search pass, and is worth detailing precisely rather than
citing loosely.

**What they built.** Soft Actor-Critic (SAC) — off-policy, entropy-regularized, **feedforward
MLP actor and critic, no recurrent component**. It regulates pH in an open photobioreactor by
controlling CO₂ injection, observing dissolved oxygen, pH, optical density, and temperature.
Behaviour cloning is used as an **offline pretraining phase**: the SAC policy is first trained
to imitate trajectories generated by a nominal PID controller, entirely offline, before ever
touching the real system; a daily online fine-tuning phase then adapts it to the real plant's
drift and disturbances.

**What they achieved.** Deployed for an 8-day real-world validation on an industrial-scale
photobioreactor under varying environmental conditions — a genuine hardware deployment, not a
simulation-only result. Against a PID baseline: **8% lower integral absolute error, 54% lower
control effort.** Against a standard off-policy RL baseline (SAC without the BC pretraining,
by implication): **5% lower IAE, 7% lower control effort.** The paper positions itself
explicitly as "the first application of an RL-based control strategy to such a nonlinear and
disturbance-prone bioprocess" for open-PBR pH regulation specifically.

**How close this is to the current project — assessed honestly on both axes:**

| axis | Gil et al. | this project |
|---|---|---|
| algorithm | SAC | TD3(+BC) |
| network | feedforward MLP | recurrent (LSTM or diagonal LRU) |
| controlled variable | pH (single loop, CO₂ injection) | harvest fraction (periodic, 1-in-600-step event) |
| observability | fully observed at every step (pH is read continuously) | partially observed / long-horizon credit assignment (harvest signal is 600 steps sparse) |
| BC's role | offline pretraining phase, then discarded | persistent anchor term in the actor loss throughout training (shown load-bearing by ablation, C2) |
| validation | **real hardware, 8-day field deployment** | simulation only, held-out sweep across 40 seeds + adversarial cold starts |
| central finding | RL+BC beats PID and beats plain RL on a real system | architecture-dependent recurrent-state-reset cadence (an orthogonal question their feedforward design cannot raise, since it has no persistent hidden state to reset) |

**The honest verdict:** these are sibling papers under "RL+BC bioprocess control," not
competing claims on the same result, and each is stronger where the other is weaker. Gil et
al. have something this project does not — a real deployment, which is the single biggest gap
this project's own `README.md` and prior audit both name. This project has something Gil et
al.'s design cannot produce — because their network is feedforward with no persistent hidden
state, the entire recurrent-state-reset-cadence question (C1) simply does not arise for them;
it is a genuinely different sub-problem, unlocked specifically by long-horizon partial
observability, which their fully-observed single-loop pH task does not have. Any writeup
citing Gil et al. should state plainly: they establish that RL+BC *works* on a real PBR for a
different (fully-observed, feedforward-adequate) control problem; this project's contribution
is a mechanism-level finding about *how* to make a recurrent RL controller work reliably on a
long-horizon, partially-observed one, which their architecture was never exposed to needing.

## What's missing for publication

Unchanged in kind from the original audit, restated against the current best contribution:

1. **No statistical treatment across training seeds for C1.** v54/v55 give n=2 for the LRU/600
   cell; every other cell in the grid is n=1. `statistical_validation.md`'s existing bootstrap
   analysis (10,000-resample, 95% CI) was run against PPO/TD-MPC2-era held-out logs and would
   need to be re-run against the TD3 held-out sweep logs to put a confidence interval on any of
   C1's numbers.
2. **No real-world or cross-simulator validation** — still the single largest gap, thrown into
   sharper relief by Gil et al.'s real deployment above. Every number in this report comes from
   one custom simulator.
3. **The grid is incomplete.** v56 (LSTM/600) is running as of this rewrite; the free-running
   (no periodic reset at all) condition has only ever been spot-checked, never run to
   completion, at either core.
4. **Related-work depth remains a handful of targeted searches**, not a systematic review with
   defined inclusion criteria, across both the original audit and this rewrite's follow-up pass.

## Recommended framing and venue tier, if pursued

- **Best single paper to write:** C1 as the spine — mechanism, fix, first held-out pass,
  independent second failure mode, and the architecture-dependent resolution — with C3 (the
  reward-dead-zone diagnosis) as a supporting methodological finding from the same
  investigation, and C2 (the BC ablation) as a discussion point on why the fix worked via
  TD3+BC specifically. This is a more coherent and more novel spine than the original audit's
  C2-anchored recommendation (the deterministic/stochastic decoupling finding), which remains
  true but is a narrower, more purely confirmatory result by comparison.
- **Tier:** workshop paper (RL or ML-for-science workshop track) or a bioprocess-engineering
  journal's applied-ML case-study format. Still not a fit for a flagship ML conference main
  track without the seed-count and cross-validation work in the gaps above.
- **C4 and C5** are supporting/discussion material — C4 as a "when does RL fine-tuning help vs.
  hurt a BC baseline" boundary condition (strengthened by contrast with Gil et al.'s opposite
  finding), C5 as a narrowly-scoped claim about closed-loop harvest control specifically (must
  cite the July 2026 decision-support paper to pre-empt an obvious reviewer objection).
- **Gil et al.** should be cited as the closest sibling work, explicitly distinguishing the two
  papers' architectures (feedforward vs. recurrent) and control problems (fully-observed
  single-loop vs. long-horizon sparse-credit) rather than treated as prior art that weakens
  this project's claim — it does not address C1 at all, since a feedforward network has no
  persistent state whose reset cadence could matter.

---

## Appendix: original 2026-08-17 audit (superseded above, retained for history)

> Audit date: 2026-08-17. Compares this project's contributions against literature located via
> targeted web search through 2026-08-16 (see `docs/literature.md` for the underlying
> citations). This is a search-based audit, not an exhaustive prior-art clearance — treat
> "not found" as "not found by this search," not as proof of absence.
>
> **This section is retained for historical record only. Its bottom-line claim ("no RL run
> ever produced a held-out-validated D2 policy... the best policy this project produced is a
> behaviour-cloned controller with no reinforcement learning applied") was true when written
> and is FALSE as of TD3+BC's v45 (2026-09-06). Do not cite this section's conclusions; see
> the rewrite above.**

### The contributions, individually assessed (original)

**C1 — TD-MPC2 applied to photobioreactor/harvest control:** combination, not invention — an
existing SOTA algorithm applied to a structurally unremarkable new domain, publishable only in
combination with the near-miss finding below.

**C2 — Mechanistic, intervention-based demonstration that exploration noise was standing in
for competence (Fix #22):** real but narrow — confirms known deterministic/stochastic-gap
theory empirically via a clean causal intervention in a new applied domain; does not
generalize the theory or propose a new fix beyond what it already suggests.

**C3 — Dual-mode curriculum gate with held-out validation:** good engineering, not yet a
citable technique — the apparatus that caught v14/v17/v26/TD-MPC2 v27 all passing in-training
and failing held-out, but currently project-internal tooling rather than a generalized,
separately-evaluated method.

**C4 — Harvest/dilution-fraction control for Spirulina:** novel scope within an established
sub-field (RL-controlled bioreactor dilution is established via Treloar et al., PLOS Comp
Bio; RL-controlled microalgae harvest fraction specifically was, at the time, absent from a
2024 systematic review) — not a novel control paradigm.

**C5 — The BC-clone-beats-every-RL-run negative result:** solid at the time, modest
generalizability even then — framed as a documented boundary condition (the harvest
dimension's 1-in-600-step credit-assignment sparsity, Fix #16) rather than a general claim
that RL fine-tuning is broken.

### What was missing for publication (original)

1. No statistical treatment across seeds (partially addressed by `statistical_validation.md`
   for the PPO/TD-MPC2-era held-out sweeps only).
2. No real-world or cross-simulator validation.
3. Related-work depth: a handful of targeted searches, not a systematic review.
