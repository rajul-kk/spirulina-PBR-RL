# Novelty and Publishability Report

> **2026-10-06. Supersedes the 2026-09-18 version (in git history).** That version predates the
> LLM-writer comparison and the pre-registered final results. As a result it:
> - led with the wrong contribution (the LSTM/LRU reset-cadence finding, not the writer comparison);
> - called the reset-cadence finding "unoccupied" (overstated: the mechanism is classic and only
>   a narrow part is open, see R1);
> - cited n=2 where the protocol now has n=5 per arm;
> - made optimistic TD3 claims that do not survive the final-split results (TD3 loses to CMA-ES
>   on the photobioreactor, see R2).
>
> **Scope of this check.** It is search-based: about 15 searches on 2026-10-06. It is not
> exhaustive prior-art clearance; "not found" means "not found by this search". The author has
> **not yet read the 2026 arXiv papers listed below in full**; the descriptions of them come from
> abstracts and search snippets and must be checked before anything is written against them.
> This area moves monthly; repeat the search before submission.

## 1. Bottom line

- The strongest contribution is now the **LLM controller-writer comparison (L1)**. On two
  simulators, both Claude-writer arms (white-box and black-box) beat CMA-ES, TD3 (photobioreactor)
  and SAC (PC-Gym), and on the photobioreactor also the hand-tuned expert and oracle references.
  Statistics were pre-registered (frozen controllers, disjoint final split, Holm family, n=5 per
  arm).
- Second: the **white-box vs black-box source-access ablation (L2)** on the unpublished
  photobioreactor (+1.94 g, seed-level p=0.0079).
- L1 is the crowded part. Several 2025-2026 papers already show coding agents or LLMs writing
  controllers that match or beat RL. What is open is the specific combination: bioprocess plus
  PC-Gym, head-to-head against CMA-ES and RL with pre-registered paired statistics, and the
  source-access ablation.
- The recurrent-RL material (R1-R3) is now secondary. It is mostly a careful application of known
  mechanisms, and the pre-registered LRU vs LSTM test came out null on the primary endpoint.
- Realistic venue: an agents-for-science or ML4Science workshop, or an applied journal
  (Computers & Chemical Engineering, Journal of Process Control). Not a flagship-ML main track
  as-is: one custom simulator, one easy public benchmark, no hardware.

## 2. Contributions, with status

Status key: SCOOPED = published already; PARTIALLY COVERED = core idea published, a narrower
part open; APPEARS OPEN = not found by this search.

### L1. LLM-written controllers beat tuned and learned baselines. PARTIALLY COVERED, crowded

**Our result (final split, n=5 runs per arm).**

| Photobioreactor, harvest g (higher better) | median | p25 | crash |
|---|---|---|---|
| White-box writer | 19.00 | 16.68 | 0% |
| Black-box writer | 17.10 | 15.14 | 0% |
| CMA-ES (6 knobs of the expert law) | 13.23 | 11.41 | 0% |
| Oracle expert (reference) | 13.12 | 11.17 | 0% |
| Hand-tuned expert (reference) | 13.08 | 11.19 | 0% |
| TD3, LRU core | 6.87 | 5.47 | 0% |
| TD3, LSTM core | 3.00 | 2.11 | 40% |

Paired differences vs CMA-ES: black-box +3.60 g [+3.19, +4.04], white-box +5.53 g
[+5.22, +5.84]; Holm p<5e-05 (episode-level), seed-level exact permutation p=0.0079 (the
minimum possible at 5 v 5). TD3 LRU is -8.36 g [-13.24, -3.99] and TD3 LSTM -10.00 g
[-11.92, -8.17] vs CMA-ES.

| PC-Gym CSTR, mean cost (lower better) | mean | median | runaway |
|---|---|---|---|
| White-box writer | 0.180 | 0.134 | 0% |
| Black-box writer | 0.192 | 0.144 | 0% |
| SAC (SB3 defaults, 100k steps) | 0.242 | 0.203 | 0% |
| CMA-ES over PID gains | 0.293 | 0.238 | 0% |
| Untuned PID (reference) | 0.540 | 0.460 | 0% |

Black-box vs SAC -0.050 [-0.060, -0.040]; vs CMA-ES-PID -0.101 [-0.113, -0.089]. White-box:
-0.062 vs SAC, -0.113 vs CMA-ES-PID. All Holm p<5e-05 at episode level; seed-level p=0.0079.

**Closest prior work (all 2025-2026, not read in full by the author).**
- "Heuristic Learning for Active Flow Control Using Coding Agents", arXiv 2607.11565 (Jul 2026):
  coding agents match or beat deep RL on 10 of 13 benchmarks. This is the nearest threat: same
  headline claim, different domain.
- "Code Evolution for Control", arXiv 2601.06845.
- ControlAgent, arXiv 2410.19811; AgenticControl, arXiv 2506.19160; GenControl, arXiv 2506.12554.
- "AI Control Scientist", arXiv 2608.26780 (Aug 2026).
- An LLM workflow for process control, arXiv 2607.21292.

**How we differ.** Domain (bioprocess and PC-Gym), a head-to-head against both CMA-ES and RL,
pre-registered paired statistics with n=5 writer runs per arm, a frozen-controller protocol, and
the source-access ablation (L2). We do not claim a new method; the claim is an evaluation. If a
reviewer knows 2607.11565, the "LLMs can beat RL at control" framing alone will not read as new.

### L2. Source access helps. APPEARS OPEN

- Photobioreactor (unpublished plant): white-box over black-box +1.94 g [+1.46, +2.39],
  seed-level p=0.0079, white-box better in 100% of episodes. Clean test: the plant is ours and
  not in any training corpus.
- PC-Gym: white-box cost lower by 0.012 [-0.021, -0.005] (about 6%), Holm p=0.0001, seed-level
  p=0.0159. Small, and **not a clean test** (next paragraph).
- Nearest prior work found: "Software Engineering Agents for Embodied Controller Generation",
  arXiv 2510.21902 (Minigrid). Not a source-access ablation on a process plant.

**PC-Gym contamination caveat** (`pcgym_protocol.md` sections 5-6). The black-box writers are
not blind: all five assumed the textbook exothermic CSTR structure, and two (bb1, bb5) shipped
the exact published constants (k0 7.2e10, E/R 8750) from memory. The access audit confirms they
read only the manual and their run directory. Recalled constants gave no visible edge (mean cost
0.187 and 0.204 recalled vs 0.185, 0.198, 0.186 fitted). Treat this as a small contamination data
point, not a finding. It does mean the PC-Gym white/black-box gap understates or muddles the
effect of source access; the photobioreactor is the real test.

### R1. LSTM cell-state saturation and the periodic-reset fix. PARTIALLY COVERED, mechanism is classic

- The mechanism (unbounded LSTM cell-state growth on long sequences) is known: Gers et al. 2000
  ("Learning to Forget", Neural Computation). Periodic hidden-state reset at a fixed interval is
  known practice (e.g. arXiv 1905.13469). Recurrent off-policy baselines exist (arXiv 2110.12628).
- What this project found: saturation of about 40% of hidden units explained a series of
  mid-training collapses (v33-v44); reset 60 fixed it (v45).
- What may be open, and it is narrow: the architecture-dependent cadence ablation (LRU works at
  reset 600, LSTM needs about 60). Only n=1-2 per cell in the earlier grid, and it has not been
  tested at n=5. Do not call this "unoccupied" or lead with it.

### R2. Pre-registered LRU vs LSTM for TD3. NULL on the primary endpoint

- Primary (final checkpoint, 5 v 5): +1.64 g [-3.56, +6.31], Holm p=0.51, seed-level p=0.1349.
  No detectable difference.
- Secondary (unadjusted, `td3_secondary.txt`): seeds reaching D1 5/5 (LRU) vs 1/5 (LSTM); D2 4/5
  vs 0/5; Fisher p=0.0476 for both. Final-checkpoint crash rate 0.0% vs 40.3% (seed-level
  p=0.1667). This is a **hypothesis, not a finding**: unadjusted, outside the Holm family, and a
  p of 0.0476 with 5 v 5 is fragile.
- **Confound.** Both cores ran at reset 600, which was chosen for the LRU (R1 says the LSTM
  wants about 60). The LSTM may simply have been handicapped. The `comparison_protocol.md`
  section 7 follow-up (4M steps, plus an LSTM reset-60 arm; family F1-F3, Holm over three) is
  designed to settle this. **Status: pending.**
- Both TD3 cores lose to CMA-ES on the photobioreactor by 8-10 g, so TD3 is not a competitive
  controller in this study. Earlier optimistic claims about TD3 are withdrawn.
- Exploratory GRU and RTU cores (protocol section 6): in progress; so far no better than LSTM.
  Pending; do not cite as a result.

### R3. Final vs best checkpoint gap in TD3. KNOWN PHENOMENON, practical note only

LRU best-det checkpoints scored 9.8-12.6 g on 4 of 5 seeds against finals of 0.9-12.0 g. The gap
is large only on seeds 1 (2.97 vs 11.86) and 3 (0.90 vs 12.34); seed 2 has none, seed 5 is
near-equal, and seed 4 is reversed (best 1.16, final 8.40). Policy degradation and checkpoint
selection are well-known in RL. Worth one sentence on why the primary endpoint used the final
checkpoint (no selection leakage); not a contribution.

### Smaller items

- **C2 (BC anchor is load-bearing in TD3+BC) and C3 (bounded-shaping reward dead zones):**
  textbook-adjacent. Divergence without a behavioural anchor is the motivation of Fujimoto & Gu
  2021 (arXiv 2106.06860). Case-study material only.
- **C4 (PPO fine-tuning destroyed the BC policy):** narrow. The harvest action is read on 1 of
  600 steps, so per-step advantages were spurious on that dimension. Keep as discussion; state
  narrowly (it is about on-policy fine-tuning under a credit-assignment defect, not about RL in
  general; TD3+BC later beat the clone).
- **C5 (closed-loop RL for the harvest fraction): APPEARS OPEN but low-value.** Must cite the
  4 July 2026 paper "Machine learning-based decision support for harvest optimization in
  commercial microalgae photobioreactors" (*Computers and Electronics in Agriculture*; Random
  Forest, R2=0.705, 12 commercial *Nannochloropsis* reactors over 4 weeks, productivity
  comparable to human operators, not above). Open part: closed-loop multi-step RL control of the
  harvest fraction. Also cite Treloar et al. (PLOS Comp Bio) for RL chemostat dilution control.

### Closest sibling: Gil et al. 2025 (arXiv 2509.06853)

SAC with behaviour-cloning pretraining from a PID controller, feedforward MLP, pH control by CO2
injection in an industrial open photobioreactor, 8-day real deployment: 8% lower IAE and 54%
lower control effort than PID, 5% and 7% better than plain SAC. They have hardware validation,
which this project lacks. We have a different problem (partially observed, harvest every 600
steps), recurrent policies, and a comparison against LLM-written controllers, which they do not
have. Cite as a sibling, not as prior art on L1 or L2. Note their finding that RL fine-tuning
helped, against our C4.

## 3. What a reviewer will hit

1. **Scope.** One custom simulator, one easy public benchmark (a single-loop CSTR that PID suits
   well), no hardware. Gil et al. have a real deployment.
2. **Budget comparability.** Writers: 300 pilot batches each, plus LLM pretraining and inference,
   which is not counted. CMA-ES: 1,164 episodes. TD3: 2M steps. SAC: 100k steps (833 batches).
   **LLM cost (tokens, dollars, wall-clock) is not yet reported**; a cost-per-controller
   comparison is the first thing an applied reviewer will ask for.
3. **Baseline strength.** TD3 is untuned by design (the tuning grid was withdrawn so it could not
   favour one core); SAC is SB3 defaults with no tuning, stated in the protocol; CMA-ES tunes only
   the expert law's 6 knobs (or PID gains) and cannot change controller structure. The writers
   can change structure, so "writers beat CMA-ES" partly measures structure, not search. The
   "oracle" reference (13.12 g) is not an upper bound (CMA-ES and the writers exceed it), so
   there is no known ceiling.
4. **Contamination.** Language models know textbook control and the textbook CSTR (see L2). The
   protocol records this honestly; the photobioreactor is the clean test, and it is one plant.
5. **n=5.** Headline the seed-level p (0.008, the minimum for 5 v 5), not the episode-level
   p<5e-05, which treats 175-200 episodes per run as if they were independent of the run. Five
   writer runs per arm is also a small sample of "what the writer does".
6. **Post-hoc elements.** GRU/RTU arms and the section 7 follow-up were chosen after seeing
   results; both are labelled exploratory or a separate family.

## 4. Evidence quality

Above typical for the target venues: protocols written before the final runs (the PC-Gym and
comparison protocols are dated before scoring), frozen controllers, a final split disjoint from
every seed used in search, pilot and training, a Holm-corrected pre-registered family,
hierarchical bootstrap plus seed-level exact permutation tests, a documented access audit, and
one invalid-run repeat disclosed (TD3 LRU seed 3, two of 200 episodes hit the 300 s wall-clock
guard; moved to `results/final/invalid/` and repeated). Weak spot: test coverage of the
comparison and RL code is thin (being addressed); the held-out tests for CMA-ES and the writer
controller exist, but the TD3 path is less covered.

## 5. What would most strengthen it, in order

1. **A third plant**, ideally another public benchmark (another PC-Gym model or a standard
   process-control benchmark), so the result is not "one custom simulator plus one easy CSTR".
2. **Report LLM cost per writer run** (tokens, dollars, wall-clock) and compare with CMA-ES and
   RL compute.
3. **Run the section 7 follow-up** (4M steps, LSTM reset 60) so R2 is settled either way.
4. **A stronger RL baseline** with a tuning budget matched to the writers (tuned SAC or TD3,
   not defaults), so that "beats RL" cannot be dismissed as a weak baseline.

## 6. Sources

Agents writing or evolving controllers:
- [Heuristic Learning for Active Flow Control Using Coding Agents](https://arxiv.org/abs/2607.11565) (2607.11565)
- [Code Evolution for Control](https://arxiv.org/abs/2601.06845) (2601.06845)
- [ControlAgent](https://arxiv.org/abs/2410.19811) (2410.19811)
- [AgenticControl](https://arxiv.org/abs/2506.19160) (2506.19160)
- [GenControl](https://arxiv.org/abs/2506.12554) (2506.12554)
- [AI Control Scientist](https://arxiv.org/abs/2608.26780) (2608.26780)
- [LLM workflow for process control](https://arxiv.org/abs/2607.21292) (2607.21292)
- [Software Engineering Agents for Embodied Controller Generation](https://arxiv.org/abs/2510.21902) (2510.21902)

Recurrent state and RL:
- Gers, Schmidhuber, Cummins (2000), "Learning to Forget: Continual Prediction with LSTM", Neural Computation 12(10). No arXiv version.
- [Interval Timing in Deep Reinforcement Learning Agents](https://arxiv.org/abs/1905.13469) (1905.13469; found by search, mentions a fixed reset interval; verify before citing)
- [Recurrent Off-policy Baselines for Memory-based Continuous Control](https://arxiv.org/abs/2110.12628) (2110.12628)
- [RLBenchNet](https://arxiv.org/abs/2505.15040) (2505.15040)
- [Recurrent Trace Units, Elelimy et al.](https://arxiv.org/abs/2409.01449) (2409.01449)
- [A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)](https://arxiv.org/abs/2106.06860) (2106.06860)

Bioprocess:
- [Gil et al. 2025, RL meets bioprocess control through behaviour cloning](https://arxiv.org/abs/2509.06853) (2509.06853)
- [ML-based decision support for harvest optimization in commercial microalgae photobioreactors](https://www.sciencedirect.com/science/article/abs/pii/S0168169926006356), *Computers and Electronics in Agriculture*, 4 July 2026
- Treloar et al., RL control of bacterial chemostat dilution, PLOS Computational Biology (cited in the earlier audit; citation to be re-checked).

Project documents: `docs/reports/comparison_protocol.md`, `docs/reports/pcgym_protocol.md`,
`experiments/program_control/results/final/compare.txt`, `.../td3_secondary.txt`,
`experiments/pcgym_control/results/final/compare.txt`.
