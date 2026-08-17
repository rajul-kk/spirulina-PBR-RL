# Novelty and Publishability Report

> Audit date: 2026-08-17. Compares this project's contributions against literature located via
> targeted web search through 2026-08-16 (see `docs/literature.md` for the underlying
> citations). This is a search-based audit, not an exhaustive prior-art clearance — treat
> "not found" as "not found by this search," not as proof of absence.

## Bottom line

This project has **one contribution with a real, defensible novelty claim** (C2 below), **two
contributions that are novel in combination but not in kind** (C1, C4), and **one methodological
practice that is good engineering but not yet a citable technique** (C3). Nothing here clears the
bar for a top-tier ML or controls venue as currently written, and none of it should be submitted
without first addressing the gaps in [What's missing](#whats-missing-for-publication) below. The
realistic target is a **workshop paper or a domain-application journal**, not a flagship RL
conference.

## The contributions, individually assessed

### C1 — TD-MPC2 applied to photobioreactor/harvest control

**What was done:** `legacy/TD_MPC2.py` was rewritten to a genuine TD-MPC2 spec (Fix #27) — 5-critic
ensemble, two-hot reward/value regression, macro-timestep world model sized to exactly one harvest
interval — and run against this project's curriculum gate. No prior instance of TD-MPC2 applied to
any bioprocess or photobioreactor task turned up in this search; the algorithm's own benchmark
suite (DMControl, Meta-World, ManiSkill2, MyoSuite) doesn't include this domain.

**Novelty tier: combination, not invention.** Applying an existing SOTA algorithm to a new but
structurally unremarkable domain (continuous control, moderate-dimensional observation, no
special structure TD-MPC2 wasn't designed for) is not on its own a publishable contribution at a
research venue — it's an application note. It becomes a contribution only when paired with C2
below: TD-MPC2 got closer to a held-out pass than any other method here (0.0036 vs 0.004, ~10%
short), and that near-miss plus the *reason* for it is the actual finding.

### C2 — Mechanistic, intervention-based demonstration that exploration noise was standing in for competence

**What was done:** Fix #22 annealed PPO's action-std cap from ~0.54 to 0.12 and *directly measured*
the deterministic/stochastic gap close — not by the mean improving, but by the sampled policy's
performance collapsing to meet it (`stoch od 0.0209 → 0.0008`, `det od 0.0086 → 0.0006`). This
converts a plausible-sounding theoretical concern into a demonstrated, causal finding in an applied
RL setting.

**What the literature says:** the *general* phenomenon — that a stochastic policy's return doesn't
transfer to its own deterministic (mean) version — is established RL theory
([Learning Optimal Deterministic Policies with Stochastic Policy Gradients](https://arxiv.org/pdf/2405.02235),
ICML 2024), with convergence guarantees and exploration-strategy analysis. What that paper does
*not* provide is a live, single-environment intervention showing the mechanism actively operating
and being falsified as an assumption in real time, in a domain where the failure mode was initially
invisible (train-time stochastic metrics looked healthy throughout).

**Novelty tier: real, but narrow.** This is a case study, not a new theorem or a new algorithm — it
confirms known theory empirically in a new domain, with a clean causal intervention rather than a
correlational observation. That is a legitimate, citable contribution (case studies validating
theory in applied domains are a normal, accepted class of RL-application paper), but it is bounded:
it does not generalize the theory, does not propose a new fix beyond what the theory paper already
suggests (anneal exploration, or train against the deterministic objective), and rests on a single
environment and a modest number of runs.

### C3 — Dual-mode (stochastic + deterministic) curriculum gate with held-out validation

**What was done:** Curriculum advancement requires both a stochastic-rollout gate and a
noise-free deterministic-eval gate to pass; a separate 40-seed held-out sweep (`held_out_sweep.py`,
`tdmpc2_held_out_sweep.py`) independently re-checks any in-training mastery claim before it's
trusted. This is the apparatus that caught v14, v17, v26, and TD-MPC2's v27 all passing
in-training and failing held-out — the single most load-bearing piece of methodology in the whole
project.

**What the literature comparison shows:** none of the comparator papers found (the BC+RL PBR pH
paper, the UNIST multi-agent Spirulina paper, the 2021 LSTM-RL Spirulina paper) report a
dual-mode gate or a held-out re-validation step distinct from their training/field metrics. Each
reports single-mode in-training or field-trial results.

**Novelty tier: not yet a citable technique.** This is good, disciplined engineering practice —
genuinely rarer in the surveyed literature than it should be — but as currently written it's
project-internal tooling embedded in one codebase, not a described, generalized method with its
own evaluation. To become a contribution in its own right (rather than a footnote inside a paper
about C2), it would need to be extracted, named, and demonstrated on more than one environment, with
an explicit argument for when a stochastic-only gate is/isn't sufficient — the "stochastic-only
gate" ablation Fix #23 already ran (v25/v26) is a start, not a full case for the method.

### C4 — Harvest/dilution-fraction control specifically, for Spirulina

**What was done:** The action space controls periodic biomass harvest fraction (not just
growth-condition setpoints like light/temperature/nutrients), with a semi-continuous
harvest-and-dilute mechanism every 12h.

**Follow-up bioprocess-engineering search (closes the gap flagged in the first draft of this
report).** RL control of chemostat/bioreactor dilution *is* an established line of work — most
directly, [Deep reinforcement learning for the control of microbial co-cultures in bioreactors](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007783)
(Treloar et al., PLOS Comp Bio) uses Neural Fitted Q-learning (bang-bang nutrient-inflow control,
not continuous harvest fraction) to hold a two-strain *E. coli* co-culture at a target ratio in a
chemostat, and other work in this space ([arXiv:2503.22409](https://arxiv.org/pdf/2503.22409))
covers multi-setpoint bioprocess trajectory tracking more generally. **So "RL controls a
bioreactor's dilution/removal dynamics" is not novel** — that claim should not be made.

What remains genuinely underexplored, and is now backed by an independent secondary source rather
than just this search's own gaps: a 2024 systematic review of AI/ML in microalgae bioprocesses
([Bioengineering, PMC11592280](https://pmc.ncbi.nlm.nih.gov/articles/PMC11592280/)) surveys
supervised/deep-learning applications throughout microalgae cultivation and harvesting and
**cites no RL-based harvest-control work at all** — its only mention of reinforcement learning is
a one-line general definition, with nothing tying it to any specific microalgae study. Combined
with the UNIST MARL paper (growth-condition control, not harvest) and the 2021 LSTM-RL Spirulina
paper (yield via environmental setpoints, not harvest) both missing harvest control specifically,
the narrower claim — **RL applied to microalgae *harvest/biomass-removal fraction* specifically
(as opposed to bioreactor dilution rate in bacterial chemostats generally, or microalgae
growth-condition control)** — is better supported than the first draft of this report gave it
credit for, precisely because an independent review's *absence* of citations is stronger evidence
than one more search finding nothing.

**Novelty tier: novel scope within an established sub-field, not a novel control paradigm.** Any
writeup must state the RL-bioreactor-dilution precedent explicitly (the PLOS paper above) so the
claim reads as "the harvest-fraction/microalgae combination is underexplored" rather than
"RL-controlled bioreactor dilution is new" — the latter would be an easy, embarrassing rejection
at review. The former is a defensible, appropriately narrow scope claim.

### C5 — The BC-clone-beats-every-RL-run negative result

**What was done:** Across 24+ PPO runs and one full TD-MPC2 run, the only artifact that passes
held-out D2 validation is a behaviour-cloned controller trained on a scripted expert with **no RL
fine-tuning applied at all** — and every attempt to fine-tune that clone with PPO made it worse
(v19: 149.1mg → 28.3mg harvest, 0% → 80-93% crash, over 8M steps).

**Novelty tier: solid negative result, modest generalizability.** "RL fine-tuning destroyed a good
BC-initialized policy" is a known failure mode in the RL literature generally, and the BC+RL PBR
paper found in this search (pH control) reports the *opposite* — RL fine-tuning improved on the
BC/PID baseline by 8%. That contrast is itself worth stating explicitly in any writeup: this
project's finding is domain- and reward-structure-specific (the harvest dimension's 1-in-600-step
credit-assignment sparsity is the root cause identified, Fix #16), not a claim that BC-then-RL is
broken in general. Framed that way — as a *documented boundary condition* on when RL fine-tuning
helps vs. hurts a BC baseline — it's a real, useful negative result. Framed as a general claim
("RL fine-tuning doesn't work here"), it overreaches.

## What's missing for publication

Three gaps would be raised by any competent reviewer, regardless of venue:

1. **No statistical treatment across seeds — partially addressed.** See
   `statistical_validation.md`: a 10,000-resample bootstrap now puts 95% CIs on the two held-out
   sweeps with saved raw per-seed logs (TD-MPC2 v27 D0, PPO v26 D2, 40 seeds each). Result: PPO
   v26's D2 failure is statistically robust (CI fully below gate); **TD-MPC2 v27's D0 near-miss is
   not** — its CI straddles the gate, so "consistent, narrow miss, not noise" (the original
   `finalresults.md` phrasing) overstated the certainty the data supports and has been corrected
   there. This only covers **evaluation-time** variance (same trained policy, 40 held-out seeds).
   It does **not** cover **training-time** variance (would a different training seed produce a
   meaningfully different policy) — that requires multiple full training runs per configuration
   (15–25h each) and remains unaddressed; the existing v21/v23 replication (same config, 0.0094 vs
   0.0066 od) shows this source of variance can be large enough to matter on its own.
2. **No real-world or cross-simulator validation.** Every number in this report comes from one
   custom simulator. That's normal for a first paper in an applied-RL line of work, but it caps
   the ceiling: reviewers at a bioprocess-engineering venue will ask whether any of this
   transfers to real Zarrouk-medium cultivation, and reviewers at an RL venue will ask whether the
   findings are simulator-specific artifacts (e.g. is the exploitability of `reward_od`'s early
   shape, or the harvest-clip-induced upward bias, an artifact of this particular reward
   engineering rather than a general phenomenon?).
3. **Related-work depth — bioprocess-engineering pass now done for C4, general coverage still
   thin.** A follow-up search specifically targeting chemostat/bioreactor RL control literature
   found the PLOS Comp Bio co-culture paper and related work (see C4 above), and cross-checked
   the harvest-specific claim against an independent 2024 systematic review rather than relying
   on this project's own search gaps. That closes the most urgent hole. Still open: this remains
   a handful of targeted searches, not a systematic review with defined inclusion criteria — a
   real submission would need a proper related-work section built the normal way (backward/forward
   citation chasing from the papers found here, not just search-engine queries).

## Recommended framing and venue tier, if pursued

- **Best single paper to write:** C2 (the mechanistic det/stochastic decoupling finding) as the
  spine, with C1 (TD-MPC2's near-miss) as a second data point showing the same failure class
  generalizes across algorithm families, and C5 (the BC-vs-RL contrast with the pH-control paper)
  as a discussion point about when RL fine-tuning helps vs. hurts. This is a coherent, focused
  applied-RL paper, not a grab-bag of every finding in this repo.
- **Tier:** workshop paper at an RL or ML-for-science workshop (e.g. NeurIPS/ICML "RL for Real
  Life"-style workshops), or a short/technical-note-length submission to a bioprocess-engineering
  journal that publishes applied-ML case studies. Not a fit for a flagship ML conference main track
  (ICML/NeurIPS/ICLR) without the statistical and cross-validation work in the gaps above — those
  venues expect either a methodological advance (this project doesn't propose a new algorithm or
  training method beyond what ICML 2024's theory paper already covers) or much more rigorous
  empirical breadth than exists here today.
- **C3 (the dual-gate methodology)** is better positioned as a *tooling/methods section* inside
  that paper than as its own contribution, unless it's separately generalized to another
  environment first.
- **C4 (harvest-specific RL)** can now be stated, narrowly: RL-controlled bioreactor dilution is
  established (cite the PLOS co-culture paper to pre-empt an obvious reviewer objection), but
  RL-controlled microalgae *harvest fraction* specifically is absent from a 2024 systematic
  review of AI/ML in microalgae bioprocesses. Useful as a supporting scope claim in the C2-anchored
  paper, not strong enough alone to anchor a paper by itself.
