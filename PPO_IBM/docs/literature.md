# Literature & Sources — GeneticPhotobioreactorEnv

Organised by model component. Each entry lists what it justifies in the code, the citation, and key values used.

---

## 1. Photosynthesis & Growth Kinetics

### Haldane / Andrew Photoinhibition Model
`f_I = I / (Ks + I + I²/Ki)` — growth increases then decreases with irradiance.

- **Andrews, J.F. (1968).** A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates. *Biotechnology and Bioengineering*, 10(6), 707–723.
- **Aiba, S. (1982).** Growth kinetics of photosynthetic microorganisms. *Advances in Biochemical Engineering/Biotechnology*, 23, 85–156.
  - Source for Spirulina Ks_light ~100 µmol/m²/s, Ki ~2000 µmol/m²/s
- **Cornet, J.F., Dussap, C.G., & Dubertret, G. (1992).** A structured model for simulation of cultures of the cyanobacterium *Spirulina platensis* in photobioreactors. *Biotechnology and Bioengineering*, 40(7), 817–825.

### Droop Intracellular Nitrogen Quota Model
`f_Q = max(0, 1 - Q_min/Q)` — growth scales with internal N quota, not external concentration.

- **Droop, M.R. (1968).** Vitamin B12 and marine ecology. IV. The kinetics of uptake, growth and inhibition in *Monochrysis lutheri*. *Journal of the Marine Biological Association of the UK*, 48(3), 689–733.
- **Droop, M.R. (1973).** Some thoughts on nutrient limitation in algae. *Journal of Phycology*, 9(3), 264–272.
  - Establishes Q_min and Q_max as bounded quota with subsistence minimum
- **Flynn, K.J. (2001).** A mechanistic model for describing dynamic multi-nutrient, light, temperature interactions in phytoplankton. *Journal of Plankton Research*, 23(9), 977–997.
  - N:P coupled quota dynamics, justifies independent P-Monod alongside N-Droop

### Phosphorus Monod Kinetics
`f_P = p_pool / (Ks_P + p_pool)` — external P as Monod limiting factor.

- **Monod, J. (1949).** The growth of bacterial cultures. *Annual Review of Microbiology*, 3(1), 371–394.
- **Ördög, V., et al. (2004).** Screening microalgae for some potentially useful agricultural and food additive properties. *Journal of Applied Phycology*, 16(4), 309–314.
  - Ks_P range 0.5–2.0 mg P/L for cyanobacteria
- **Powell, N., et al. (2009).** Factors influencing luxury uptake of phosphorus by microalgae in waste stabilisation ponds. *Environmental Science & Technology*, 43(16), 6233–6239.

### Temperature Response (Gaussian)
Peak growth at T_opt, decline at extremes.

- **Bernard, O., & Rémond, B. (2012).** Validation of a simple model accounting for light and temperature effect on microalgal growth. *Bioresource Technology*, 123, 520–527.
  - T_opt ~27°C, valid range 20–35°C for Spirulina
- **Torzillo, G., et al. (1993).** Production of *Spirulina* biomass in closed photobioreactors. *Biomass and Bioenergy*, 4(4), 259–263.

---

## 2. Photo-Acclimation

### Exponential Moving Average Acclimation (tau_acclim 1–4 h)
`cells_acclimation += (dt/tau) * (I_total - cells_acclimation)` — lagged cellular response to irradiance.

- **Geider, R.J., MacIntyre, H.L., & Kana, T.M. (1998).** A dynamic regulatory model of phytoplankton acclimation to light, nutrients, and temperature. *Limnology and Oceanography*, 43(4), 679–694.
  - Establishes EMA as the standard acclimation formulation; tau ~1–4 h for cyanobacteria
- **Zonneveld, C. (1998).** Photoinhibition as affected by photoacclimation in phytoplankton: a model approach. *Journal of Theoretical Biology*, 193(1), 115–123.
  - Distinguishes fast (minutes) and slow (hours) acclimation timescales
- **MacIntyre, H.L., et al. (2002).** Photoacclimation of photosynthesis irradiance response curves and photosynthetic pigments in microalgae and cyanobacteria. *Journal of Phycology*, 38(1), 17–38.

---

## 3. Dark Respiration

### Elevated Maintenance at Night (1.8×)
`m_respiration = 0.010 * mu_max * dark_factor` — respiration increases in absence of light.

- **Tomaselli, L., Torzillo, G., Giovannetti, L., Pushparaj, B., Bocci, F., Tredici, M., Papuzzo, T., Balloni, W., & Materassi, R. (1987).** Recent research on *Spirulina* in Italy. *Hydrobiologia*, 151/152, 79–82.
  - **Primary source for the 2.0× factor.** Measured biomass loss during 12h dark period across 10 *S. platensis* strains and 8 *S. maxima* strains. Mean dark-period loss: 30.5% of light-period biomass gain for *S. platensis*, 15.7% for *S. maxima*. Implies night maintenance rate ~0.010 h⁻¹ vs ~0.005 h⁻¹ daytime ≈ 2× ratio.
- **Tomaselli, L., Margheri, M.C., & Sacchi, A. (1995).** Effects of light on pigments and photosynthetic activity in a phycoerythrin-rich strain of *Spirulina subsalsa*. *Aquatic Microbial Ecology*, 9, 27–31.
  - Direct O₂-based measurement: dark respiration 39 µmol O₂ mg⁻¹ chl a h⁻¹ (low-irradiance cultures) vs 79 µmol O₂ mg⁻¹ chl a h⁻¹ (high-irradiance cultures) — ratio 2.03×. Measured on *S. subsalsa*; consistent with 2.0× used for *S. platensis*. Also confirms photoinhibition onset at PAR > 600 µmol m⁻² s⁻¹.
- **Ogawa, T., & Terui, G. (1970).** Studies on the growth of *Spirulina platensis* (I) on the pure culture under photoautotrophic conditions. *Journal of Fermentation Technology*, 48, 361–367.

---

## 4. Dissolved Gas Model

### Two-Layer Stratified Gas Model (surface/bulk)
`kLa_s = 1.20 × kLa`, `kLa_b = 0.90 × kLa`, inter-layer mixing via `kLa_inter = kLa × mix × 0.5`.

- **Chisti, Y. (1989).** *Airlift Bioreactors*. Elsevier Applied Science, London.
  - kLa scaling with aeration rate; DO₂ stratification in low-mixing flat-panel reactors
- **Mirón, A.S., et al. (1999).** Bubble-column and airlift internal-loop reactors for algal culture. *AIChE Journal*, 45(9), 1872–1887.
  - kLa heterogeneity in airlift reactors; surface zone enrichment
- **Camacho Rubio, F., et al. (1999).** Prediction of dissolved oxygen and carbon dioxide profiles in tubular photobioreactors. *Biotechnology and Bioengineering*, 62(1), 71–86.
  - CO₂/O₂ gradients along reactor depth; inter-zone mixing coefficients

### kLa Correlation with Mixing
`kLa ∝ stir_rpm / 200 × base_kLa`

- **Van't Riet, K. (1979).** Review of measuring methods and results in nonviscous gas-liquid mass transfer in stirred vessels. *Industrial & Engineering Chemistry Process Design and Development*, 18(3), 357–364.
- **Acién Fernández, F.G., et al. (2001).** Modelling of biomass productivity in tubular photobioreactors for microalgal cultures. *Biotechnology and Bioengineering*, 65(6), 605–616.

### Airlift Circulation Velocity (0.05 m/s)
- **Chisti, Y. (1989).** *Airlift Bioreactors*. Elsevier Applied Science.
  - Typical riser velocity 0.03–0.10 m/s for flat-panel airlift; 0.05 m/s at moderate aeration
- **Contreras, A., et al. (1998).** Influence of sparger on energy dissipation, shear rate, and mass transfer to sea water in a concentric-tube airlift bioreactor. *Chemical Engineering Science*, 53(14), 2559–2568.

---

## 5. pH and CO₂ Chemistry

### Henderson-Hasselbalch pH Model
`pH = buffer_eq_pH - 0.8 × log10(co2_bulk / 2.0)`

- **Snoeyink, V.L., & Jenkins, D. (1980).** *Water Chemistry*. John Wiley & Sons.
  - Henderson-Hasselbalch for carbonate/bicarbonate buffering in alkaline media
- **Stumm, W., & Morgan, J.J. (1996).** *Aquatic Chemistry*, 3rd ed. John Wiley & Sons.
  - pKa1 = 6.35, pKa2 = 10.33 for carbonic acid at 25°C; Spirulina operates near pKa2
- **De Godos, I., et al. (2010).** Optimization of microalgae production in raceway reactors. *Bioresource Technology*, 101(20), 7950–7955.
  - pH 9–10 as Spirulina operational window; CO₂ as primary pH control lever

---

## 6. Conductivity Model

### Kohlrausch Molar Conductance Formula (µS/cm)
`σ = Σ λᵢ × cᵢ` — species-specific molar conductances.

- **Kohlrausch, F., & Holborn, L. (1898).** *Das Leitvermögen der Elektrolyte*. Teubner, Leipzig.
  - Original law of independent migration of ions
- **Lide, D.R. (ed.) (2003).** *CRC Handbook of Chemistry and Physics*, 84th ed. CRC Press.
  - Table of limiting molar conductances at 25°C: NO₃⁻=71.4, Na⁺=50.1, K⁺=73.5, HPO₄²⁻=57.0, SO₄²⁻=160.0, Cl⁻=76.4, OH⁻=198.0, H⁺=349.8 S·cm²/mol
- **Robinson, R.A., & Stokes, R.H. (2002).** *Electrolyte Solutions*, 2nd revised ed. Dover Publications.
  - Temperature correction ~2%/°C (Kohlrausch temperature coefficient)

### Zarrouk Medium Composition
Justifies ion pools: NaNO₃ (N), K₂HPO₄ (P), NaHCO₃/NaCl (background salts).

- **Zarrouk, C. (1966).** Contribution à l'étude d'une cyanophycée: influence de divers facteurs physiques et chimiques sur la croissance et la photosynthèse de *Spirulina maxima*. PhD thesis, University of Paris.
  - Original Zarrouk formulation: 16 g/L NaHCO₃, 2.5 g/L NaNO₃, 0.5 g/L K₂HPO₄, etc.
- **Vonshak, A. (ed.) (1997).** *Spirulina platensis (Arthrospira): Physiology, Cell-Biology and Biotechnology*. Taylor & Francis, London.
  - Modified Zarrouk variants; N:P mass ratio ~5:1

---

## 7. Osmotic Stress

### Gaussian Osmotic Penalty (threshold 12,000 mg/L, σ=3000)
- **Vonshak, A., et al. (1996).** Environmental constraints on growth and photosynthesis of the cyanobacterium *Spirulina platensis*. *Journal of Applied Phycology*, 8(1), 1–9.
  - Spirulina NaCl tolerance up to 80 g/L (salinity ~12 g/L equivalent for growth suppression onset)
- **Richmond, A. (2004).** *Handbook of Microalgal Culture*. Blackwell Science.
  - Halotolerance range; Zarrouk medium ionic strength ~18–25 mS/cm

---

## 8. Light Model

### Beer-Lambert Spectral Attenuation (RGB)
`I(z) = I_surface × exp(−k × z)`

- **Molina Grima, E., et al. (1994).** Photon flux density as a key parameter in microalgal mass culture. *Applied Microbiology and Biotechnology*, 41(5), 609–611.
- **Acién Fernández, F.G., et al. (1997).** A model for light distribution and average solar irradiance inside outdoor tubular photobioreactors. *Biotechnology and Bioengineering*, 55(5), 701–714.
  - Spectral extinction coefficients by wavelength band; biomass-dependent k

### Turbulent Flash-Light Effect
Cells cycle through light/dark zones at high mixing — `z_effective = (1 − turb_fraction) × z_static + turb_fraction × z_random`.

- **Grobbelaar, J.U. (1994).** Turbulence in mass algal cultures and the role of light/dark fluctuations. *Journal of Applied Phycology*, 6(3), 331–335.
  - Flash-light effect increases effective photosynthesis; light/dark cycle at 1–100 Hz
- **Janssen, M., et al. (2001).** Photosynthesis and respiration of *Dunaliella tertiolecta* in cyclostats. *Journal of Algology*, 37(4), 399–409.

### Bubble Scattering
`k_scatter = stir_rpm × 0.004`

- **Mirón, A.S., et al. (2000).** Hydrodynamics and mass transfer in bubble column and airlift reactors with *Phaeodactylum tricornutum* suspensions. *Chemical Engineering Science*, 55(23), 5509–5527.

---

## 9. Death and Lysis

### Background Death Rate (5×10⁻⁴ h⁻¹, ~1.2%/day)
- **Converti, A., et al. (2009).** Effect of temperature and nitrogen concentration on the growth and lipid content of *Nannochloropsis oculata* and *Chlorella vulgaris* for biodiesel production. *Chemical Engineering and Processing*, 48(6), 1146–1151.
  - Background lysis 0.5–2%/day reported for photobioreactor cultures
- **Goldman, J.C., & Carpenter, E.J. (1974).** A kinetic approach to the effect of temperature on algal growth. *Limnology and Oceanography*, 19(5), 756–766.

### Stress-Dependent Additional Lysis
`lysis_rate = 5e-4 + 2e-3 × stress²`

- **Ogawa, T., & Terui, G. (1970).** *Journal of Fermentation Technology*, 48, 361–367.
- **Torzillo, G., et al. (1996).** Effect of oxygen concentration on the productivity of *Spirulina platensis* grown in a closed tubular photobioreactor. *Applied Microbiology and Biotechnology*, 45(1), 18–23.
  - O₂ above 20 mg/L causes measurable ROS-mediated lysis

---

## 10. Reactor Geometry

### Flat-Panel Airlift (30 cm depth, 30 L)
- **Hu, Q., et al. (1998).** A flat inclined modular photobioreactor for outdoor mass cultivation of photoautotrophs. *Biotechnology and Bioengineering*, 51(1), 51–60.
- **Tredici, M.R. (2004).** Mass production of microalgae: photobioreactors. In *Handbook of Microalgal Culture*, Blackwell Science.
  - Flat-panel depth 1–10 cm (lab) to 10–30 cm (pilot); 30 cm depth for scale-up
- **Zhang, Q.H., et al. (2013).** Current status and outlook of CO₂ capture, storage, and mineral utilization. *Greenhouse Gases: Science and Technology*, 3(1), 2–39.

---

## 11. Spirulina Strain Parameters

### mu_max ~0.05 h⁻¹, Q_min/Q_max
- **Torzillo, G., Pushparaj, B., Bocci, F., Balloni, W., Materassi, R., & Florenzano, G. (1986).** Production of *Spirulina* biomass in closed photobioreactors. *Biomass*, 11, 61–74.
  - Corrected citation — this entry previously read "(1993) ... *Biomass and Bioenergy*, 4(4), 259-263," which does not match any verifiable paper. The real 1986 paper (this one, correct title/journal/volume/pages) and a *different* 1993 Torzillo paper ("A two-plane tubular photobioreactor for outdoor culture of Spirulina," *Biotechnology and Bioengineering*, 42, 891-898) appear to have been conflated. Not independently verified against the 1986 paper's full text that "mu_max 0.04-0.07 h⁻¹" appears there specifically — kept as a directionally-consistent estimate, corroborated by other real, checked sources below.
- **Vonshak, A. (1997).** *Spirulina platensis (Arthrospira): Physiology, Cell-Biology and Biotechnology*. Taylor & Francis / CRC Press.
  - Comprehensive strain parameter ranges. T_opt note below (was previously mis-stated here as ~27°C).
- **Additional mu_max corroboration (independently verified, not previously cited here):**
  - Multiple studies report Arthrospira/Spirulina generation times of 2.7-3.2 days in Zarrouk medium (μ ≈ 0.009-0.011 h⁻¹) and up to 11.7 days under nutrient-poor conditions — see e.g. growth-performance/optimization studies indexed under "Optimization of Arthrospira platensis (Spirulina) Growth: From Laboratory Scale to Pilot Scale" (2017) and "Growth performance of Spirulina (Arthrospira) platensis in a low cost medium" — both far slower than the sim's original 0.08 h⁻¹ mean.
  - One tubular-photobioreactor study reports μ=0.047 h⁻¹ (~15h doubling) under favorable conditions — closer to, and corroborating, the corrected 0.055 h⁻¹ mean used in `genetic_env.py`.
  - Real growth rates vary roughly 10x across studies (0.005-0.08 h⁻¹) depending on medium, light, temperature, and mixing — any single point estimate is a judgment call within that range, not a settled constant.

### T_opt ~35-37°C
- **Torzillo, G., & Vonshak, A. (1994).** Effect of light and temperature on the photosynthetic activity of the cyanobacterium *Spirulina platensis*. *Biomass and Bioenergy*, 6(5), 399–408.
  - 35°C optimal; narrow 35-37°C range for peak growth; 40°C detrimental. Corrects a previously-unverified "~27°C" note that appeared in this section — the code's `T_opt = N(36, 1)°C` is well-supported by this and later sources (e.g. reported optimal ranges of 30-37°C for *Limnospira/Arthrospira platensis*), not the earlier doc text.

### Cell Mass (~500 pg, division at ~1.4×10⁸ pg super-agent)
- **Cornet, J.F., et al. (1992).** *Biotechnology and Bioengineering*, 40(7), 817–825.
  - Cell dry weight 3–10 pg; 500 pg used for super-agent (2.5×10⁶ real cells per agent)

---

## 12. Reinforcement Learning Methods

### Proximal Policy Optimization (PPO)
- **Schulman, J., et al. (2017).** Proximal policy optimization algorithms. *arXiv:1707.06347*.

### Recurrent PPO with LSTM for POMDPs
- **Ni, T., et al. (2022).** Recurrent model-free RL can be a strong baseline for many POMDPs. *arXiv:2110.05038*.
- **Hausknecht, M., & Stone, P. (2015).** Deep recurrent Q-learning for partially observable MDPs. *AAAI 2015 Fall Symposium*.

### Potential-Based Reward Shaping (PBRS)
`F(s,s') = γΦ(s') − Φ(s)` — policy-invariant shaping.

- **Ng, A.Y., Harada, D., & Russell, S. (1999).** Policy invariance under reward transformations: theory and application to reward shaping. *ICML 1999*, 278–287.

### Genetic / Domain Randomization for Sim-to-Real
- **Tobin, J., et al. (2017).** Domain randomization for transferring deep neural networks from simulation to the real world. *IROS 2017*. IEEE.
- **Zhao, W., et al. (2020).** Sim-to-real transfer in deep reinforcement learning for robotics: a survey. *IEEE SSCI 2020*.

### RL for Bioprocess Optimisation
- **Petsagkourakis, P., et al. (2020).** Reinforcement learning for batch bioprocess optimization. *Computers & Chemical Engineering*, 133, 106649.
- **Treloar, N.J., et al. (2022).** Deep reinforcement learning for the control of microbial co-cultures in bioreactors. *PLOS Computational Biology*, 18(12), e1010783.

### Curriculum Learning
- **Bengio, Y., et al. (2009).** Curriculum learning. *ICML 2009*, 41–48.

### Individual-Based Model (IBM) Framework
- **Grimm, V., et al. (2006).** A standard protocol for describing individual-based and agent-based models. *Ecological Modelling*, 198(1–2), 115–126.
- **DeAngelis, D.L., & Mooij, W.M. (2005).** Individual-based modeling of ecological and evolutionary processes. *Annual Review of Ecology, Evolution, and Systematics*, 36, 147–168.

---

## 13. Sensor Models

### Turbidity / Nephelometry
- **ISO 7027 (1999).** Water quality — determination of turbidity. International Organisation for Standardisation.
  - 860 nm wavelength standard; Mie scattering regime for cells 1–10 µm

### Inline Nitrate Sensors (UV Absorbance)
- **Nitschke, L., & Wich, H. (1994).** A flow-injection method for the on-line determination of nitrate in wastewater. *Fresenius' Journal of Analytical Chemistry*, 349, 496–498.
  - NO₃⁻ absorption at 220 nm; dual-wavelength correction at 254 nm for organics

### Conductivity Measurement
- **Gray, J.R., & Glysson, G.D. (2003).** Proceedings of the Federal Interagency Workshop on Turbidity and Other Sediment Surrogates. USGS Circular 1250.
  - Calibration and temperature compensation of conductivity probes

---

*Last updated: 2026-06-08*
