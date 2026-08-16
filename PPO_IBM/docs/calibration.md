# PBR Environment — Parameter Calibration Notes

> **⚠ STALE — describes a discontinued organism/medium configuration.** This entire document
> was written for a **Chlorella vulgaris on BG-11 medium** configuration (5D observation space,
> 4-action stir/CO₂/nutrient/light control, no harvest mechanism). The current environment
> models **Spirulina (Arthrospira) platensis on Zarrouk medium** — 6D observations, 3-action
> stir/light/harvest control with a periodic semi-continuous harvest mechanism central to the
> entire curriculum/RL design (see `docs/known_limitations.md`'s O13 correction and
> `finalresults.md`). Essentially every organism-specific parameter below (mu_max, T_opt, pH
> optimum, shear tolerance, osmotic threshold, stoichiometry) is for the wrong organism and
> does not describe the current `genetic_env.py`. Kept as a historical record of the prior
> configuration and the calibration-correction discipline used at the time — **for current
> parameter values and their citations, see `docs/literature.md` (Spirulina-specific,
> up to date) and `genetic_env.py` directly, not this file.**

Tracks which parameters are physically grounded vs deliberate training compromises, and why.

---

## Strain / Medium: Chlorella vulgaris on BG-11

The environment models a **semi-continuous fed-batch Chlorella vulgaris PBR on depleted BG-11 medium** (30L flat-panel, dt=0.02h, 7200-step episode = 144h).

**Why Chlorella / BG-11 (vs Spirulina / Zarrouk):**
- Hardware sensors (turbidity SEN0189, pH SEN0161, conductivity DFR0300, DS18B20) better match Chlorella's freshwater range.
- BG-11's low bicarbonate (0.38 mM) creates a tight CO₂-pH coupling — both gas actions are immediately consequential.
- All four control actions (stir, CO₂, nutrients, light) are simultaneously limiting at low-to-mid OD.

---

## Observation Space (5D — real hardware sensors only)

| Dim | Sensor | Range | Hardware |
|-----|--------|-------|----------|
| 0 | Turbidity (NTU) | 0–5000 | SEN0189 |
| 1 | pH | 0–14 | SEN0161 |
| 2 | Dosing integral (mg N cumulative) | 0–5000 | Pump counter |
| 3 | Conductivity (µS/cm) | 0–5000 | DFR0300 |
| 4 | Temperature (°C) | 0–50 | DS18B20 |

`n_pool` (privileged, no inline sensor) is replaced by the dosing integral — the running sum of mg N added via pump, accumulating at `nut_flow × 0.87 × dt` per step.

---

## Parameters: Physically Grounded (Chlorella vulgaris / BG-11)

| Parameter | Value | Basis |
|-----------|-------|-------|
| `bicarbonate` (reset) | 0.38 mM | BG-11: 0.02 g/L Na₂CO₃ → ~0.38 mM HCO₃⁻. Extremely low buffer capacity. |
| `dissolved_co2` (reset) | 2.0 mg/L | Pre-sparged. At 0.38 mM HCO₃⁻, gives pH 7.27 via H-H (correct for BG-11 equilibrium). |
| `buffer_equilibrium_ph` | 7.3 | H-H: pKa₁=6.35, pH = 6.35 + log₁₀(0.38/0.04545) ≈ 7.27 ✓ |
| pH lag rate | 2.0/h | kLa ~1.5–5/h; 2.0/h gives ~30-min response. Correct for diffusion-limited 30L PBR. |
| pKa₁ correction | −0.002/°C from 25°C | Stumm & Morgan (1996); symmetric across full 22–28°C range. |
| pH clamp | [5.0, 11.0] | Allows full CO₂ injection to drop pH below 7.3 (realistic for BG-11 unbuffered system). |
| `n_pool` (reset) | 100 mg N/L | BG-11 depleted-medium start (full BG-11 = 247 mg N/L as NaNO₃). |
| `p_pool` (reset) | 7.0 mg P/L | BG-11 K₂HPO₄ 0.04 g/L → 7.1 mg P. |
| `mu_max` | N(0.045, 0.007) /h | Chlorella vulgaris: 0.03–0.06/h (Sorokin & Krauss 1958; Semenenko 1981). |
| `Ks` (N half-sat) | N(1.0, 0.2) mg N/L, floor 0.3 | Chlorella: 1–5 mg NO₃-N/L (Eppley & Rogers 1970). Previously 0.05 (cyanobacteria literature — wrong organism). |
| `Ks_P` | U(0.3, 1.5) mg P/L | Chlorella P half-saturation (Droop 1974). |
| `T_opt` | N(25, 1)°C | Chlorella vulgaris optimum ~25°C (Semenenko 1981). |
| `Kii` | N(2500, 250) µmol/m²/s | Chlorella photoinhibition threshold (Vonshak 1997 range). |
| f_pH optimum | 7.2 | Chlorella vulgaris pH optimum 6.5–8.0 (Semenenko & Abdullaev 1981). Acid σ=1.0, alkaline σ=1.2. Previously 9.5 (Spirulina — wrong organism). |
| pH penalty | >9.0 or <5.5 | Hard toxic bounds for Chlorella (optimum 6.5–8.0). |
| CO₂ stoichiometry | 3.67 mg CO₂/mg net DW | Photosynthesis 6CO₂+6H₂O→C₆H₁₂O₆+6O₂: 6×44/(6×12)=3.67 mg CO₂/mg C; at 50% C biomass and accounting for respiratory CO₂ return, net value is ~3.67. Previously 1.8 (4× too low — CO₂ never depleted, pH never rose). |
| O₂ stoichiometry | 1.5 mg O₂/mg net DW | Net O₂ yield from photosynthesis (gross 2.67 mg O₂/mg C, minus ~40% growth respiration overhead). Previously 1.2. |
| Bicarbonate uptake fraction | 0.3 | Consistent with f_carbon = 0.3 × f_hco3 (Chlorella CCM, 30% from HCO₃⁻). Previously 0.5 (inconsistent). |
| f_to_hco3 | pH-weighted | Fraction of dissolved CO₂ converting to HCO₃⁻ = 10^(pH−6.35)/(1+10^(pH−6.35)). At pH 7.3: ~90%. |
| Osmotic stress threshold | 5000 µS/cm, σ=3000 | Chlorella freshwater alga: inhibition starts ~5000 µS/cm (Rai & Gaur 2001). BG-11 starts at ~3200 µS/cm. Previously 25000 µS/cm (Zarrouk-adapted Spirulina — never fired). |
| Shear onset (membrane) | 150 RPM | Chlorella unicells (5–10 µm, rigid wall) far more shear-tolerant than Spirulina filaments. Previously 80 RPM. Max fatigue penalty 5% (was 15%). |
| Shear repair tax | max 10% | Chlorella max 10% growth penalty at 200 RPM (was 35% for Spirulina). |
| Droop Q initialization | U(3.5, 5.0) | Start near Q_max in replete BG-11 medium. Previously U(2.0, 4.0) → cells started 17% N-limited despite fresh medium. |
| Droop quota dilution | µ × Q × dt | Added: growing cells dilute intracellular quota proportionally. Prevents unbounded Q accumulation. |
| Q_max enforced | 5.0 | Clamp added after uptake. Previously Q_max was defined but never applied. |
| N:P dosing ratio | 87%N / 8%P / 5%salt | N:P = 10.9:1 (between Redfield 7:1 and BG-11 28:1). Previously 75%/15%/10% → N:P=5:1 (biologically wrong). |
| P uptake factor | 0.0014 | 0.01 / 7.2 = Redfield N:P ratio by mass. Previously 0.005 → P drained at same rate as N (biologically wrong; algae use N:P ~7:1). |
| Dosing integral accumulation | 0.87 × nut_flow × dt | Matches 87% N fraction in dosing solution. |
| `kLa` correlation | stir-dependent, 0.05–12/h | 30L sparged flat-panel. Base kLa at 50 RPM ≈ 0.6–1.4/h; at 200 RPM ≈ 5–8/h. |
| `m_respiration` | 0.010 × mu_max | Endogenous respiration at ~10% of max growth (Cornet et al. 1992). Night 2× (dark respiration). |

---

## Henderson-Hasselbalch Unit Analysis

pH = pKa₁ + log₁₀([HCO₃⁻] / [CO₂(aq)])

Unit identity (code uses `co2_b / 44.0`):
- `co2_b` is in mg/L; MW(CO₂) = 44 g/mol
- mg/L ÷ g/mol = (mg/L) × (mol/g) = 10⁻³ g/L × mol/g = 10⁻³ mol/L = **mM** ✓
- `bicarbonate` is in mM ✓

Numerical check at BG-11 reset state:
- co2_b = 2.0 mg/L → co2_aq_mM = 2.0/44 = 0.04545 mM
- bicarbonate = 0.38 mM
- pH = 6.35 + log₁₀(0.38/0.04545) = 6.35 + 0.922 = **7.27** ✓ (matches `buffer_equilibrium_ph = 7.3`)

---

## Parameters: Deliberate Training Compromises

| Parameter | Value | Why compromised | Physical reality | Risk |
|-----------|-------|-----------------|-----------------|------|
| N drain factor | 0.01 | Max drain ~37.5 mg N/h at 7500 cells. At N uptake Ks=1 mg/L, drain is effectively this even at n_pool=100. | Real Chlorella: ~5–15 mg N/g DW/h. At 30L scale, consistent order of magnitude. | Moderate: recalibrate if max_cells or super-agent mass changes. |
| `n_pool` (reset) | 100 mg N/L | Full BG-11 = 247 mg N/L. At 247, N is non-limiting for all 144h — agent learns nothing. 100 mg/L creates N-limitation in mid-episode without immediate starvation. | Semi-continuous operation drains ~60% then refills — 100 mg/L is physically realistic. | Low. |
| `n_pool` (bleaching threshold) | 20 mg N/L | Chlorosis in Chlorella begins at ~5–20 mg N/L. 20 mg/L is a conservative (slightly early) trigger. | Slightly generous but prevents late-episode N crash without warning. | Low. |
| phi_cur N peak | 200 mg/L | Agent gets gradient to maintain N around 200 mg/L (between reset 100 and full BG-11 247). | Physiological optimum is 100–247 mg/L (non-limiting), so 200 is reasonable. | Low. |
| CO₂ toxicity Ki | 30 mg/L | Mild onset at 30 mg/L dissolved CO₂. At our operating range (0–30 mg/L) toxicity is near-zero. | Chlorella tolerates 5% CO₂ in gas (~60 mg/L dissolved) with some inhibition above 50 mg/L. | Low. |

---

## Nutrient Dynamics (Chlorella / BG-11)

**N dynamics at max density (7500 cells, n_pool=100 mg/L, Ks=1 mg/L):**
- Uptake rate = 0.5 × 100/(1+100) ≈ 0.495 ≈ 0.5 (Monod saturation)
- Max N drain = 0.5 × 0.02h × 7500 × 0.01 = 0.75 mg/step = 37.5 mg/h → 1.25 mg/L/h in 30L
- Without dosing: n_pool depletes from 100 in ~80h (much slower than early Spirulina model)

**P dynamics at max density (Redfield P uptake, p_pool=7 mg/L):**
- Max P drain = 0.93 × 0.02 × 7500 × 0.0014 = 0.195 mg/step = 9.75 mg/h → 0.325 mg/L/h
- Without dosing: p_pool depletes from 7 in ~21h ← agent must dose nutrients within 21h at max density

**N:P depletion ratio:** 37.5/9.75 = 3.85:1 by mass drain rate (Redfield 7:1 by content; drain faster for P because cells allocate N to quota storage)

**CO₂ dynamics (corrected stoichiometry, 3.67×):**
- At 300 cells growing at 0.03/h: CO₂ consumption ~4 mg/h; in 30L (60 mg total), depletes in ~15h without injection
- At 7500 cells growing at 0.04/h: CO₂ consumption ~138 mg/h; depletes in ~26 min → agent must inject CO₂ constantly at high OD

---

## High-Init Curriculum Bucket

`bucket == "high"` → 2000–5000 cells (log-uniform). Equivalent OD: 0.022–0.055.

Purpose: exposes policy to the density regime where CO₂ injection, O₂ stripping, and light attenuation are physically significant. At <500 cells all three dynamics are below noise floor.

---

## pH Clamp and CO₂ Control (Key Invariant — BG-11)

```
At reset: bicarbonate=0.38 mM, co2_b=2.0 mg/L → pH 7.27 (buffer_equilibrium_ph=7.3)

Without CO₂ injection:
  - Atmospheric CO₂(aq) saturation: 1276 × 420e-6 = 0.536 mg/L
  - CO₂ degasses to 0.536 mg/L within ~1h (kLa × (2.0-0.536) × dt)
  - pH equilibrates to: 6.35 + log10(0.38/0.01218) = 6.35 + 1.494 = 7.84
  - Then photosynthesis draws down bicarbonate → pH rises further toward 9+

With CO₂ injection (at 5% CO₂ in gas):
  - co2_sat = 1276 × 0.05 = 63.8 mg/L
  - Agent can hold pH below 7.5 throughout the episode

Agent must learn: inject CO₂ proportional to culture density (more cells = more CO₂ needed).
```

At pH 8.7 (old training plateau), f_pH = exp(-0.5×(1.5/1.2)²) = exp(-0.78) = 0.46 → 46% growth.
At pH 7.2 (target with CO₂ control), f_pH = 1.0 → full growth rate.

This 2.2× growth rate difference is the primary incentive for CO₂ management.
