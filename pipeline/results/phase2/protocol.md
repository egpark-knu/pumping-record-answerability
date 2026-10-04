# D03 Phase 2 protocol — which operational pumping record answers which pumping what-if, in which aquifer

Version 1.0 (Role B, executor). This protocol is frozen before any official case, envelope, forecast, reference fit or score. The SHA-256 is in `protocol.sha256`; the code, source and decision hashes are in `protocol_freeze.json`.

**Authority**
- Author instruction D03 (`direction/D03_record_features_and_engine_response.md` = `results/phase2/author instruction`).
- The author's matching definition (`results/phase2/LATEST_STEERING.md`).
- The root design decision after debate `debate_20260930_191313` (`results/phase2/ROOT_DESIGN_DECISION.json`, QUESTION_OK, scope Same).
- README rules 1–7.

The approved D02 pilot is the hash-verified base. Its protocol (`cf0cf607…`), its v1.2 W addendum and its code are imported unchanged; nothing in D02 is edited.

---

## 1. Question and hypotheses

A well operator keeps a pumping record: when the pump was on, when it was stopped, and for how long. Two what-if questions are asked of that record. What happens in ten days if the pump keeps running rather than stopping (a question about the early response), and what happens in thirty days (a question about the eventual effect)?

D03 proposes that different features of the record answer these two questions:
- **Short on/off transitions** tell the record about the early shape of the response.
- **A long rest** tells it about the eventual gain.

Aquifers are distinguished by storage. A smaller storage coefficient makes the same pumping produce a faster and larger head response, which changes what a record of a given length can reveal. A scenario tool is then asked whether it uses this record information.

The quantities are the pilot's, with physical pumping:
- **E_true(k)** = h*(Q_a; k) − h*(Q_b; k), the noise-free twin-truth difference between two future schedules at lead k (days).
- **W(C, k)** = sup − inf of E_g(k) over the declared family members g whose fit to the 1,024-day record C passes the frozen tolerance. W is the width of what the record leaves undetermined inside the declared family. It is a sampled inner envelope of direct-certified curves (v1.2), not a probability interval and not a universal information limit. The generator lies inside the family, so W is the ambiguity of a correctly specified family: the most favourable structural case.
- **E_tool(k)** = q50_a(k) − q50_b(k) of the frozen scenario tool, i.e. a difference of marginal medians.

Hypotheses, stated before scoring. Reversed or absent responses are results. Monotonicity is not a completion criterion, and no hypothesis outcome is a success gate.
- **R1.** W at lead 10 d decreases with the number of short transitions N and responds weakly to the rest ratio ρ.
- **R2.** W at lead 30 d decreases with ρ and responds weakly to N.
- **R3.** The tool's point-contrast error decreases with neither N nor ρ.

## 2. Grid

ρ ∈ {0.25, 1, 4} × N ∈ {2, 6, 18} × 6 physical layers × 10 realizations gives **540 cases**. The two schedule pairs give **1,080 contrasts**. Leads 10 d and 30 d are both primary. The maximum of W over days 1–10 is supplementary.

Realizations are the pilot's:
- `results/pilot/phase0_checks/realization_calendar.json`: Andong r0–3, Sancheong r4–6, Namhae r7–9, with the same windows and the same seeds for schedule, natural component, noise and reference sampling.
- The real KMA rain is SHA-checked at load.

No level is dropped or merged.

## 3. Physical layers: controlled regimes, source-qualified

### 3.1 Map
Hantush–Jacob leaky well, in the exact form already verified in D02:
- λ = √(T c)
- a = c·S
- b = r²/(4 T c)
- t_r = a b = r² S/(4T), and π_r = t_r / (1 d)
- gain A = K0(r/λ)/(2πT), in m of head per m³/d
- s(r, t) = Q·A·S_unit(t) = Q/(4πT)·W(u, r/λ)

The unit step is the frozen `p3_kernels.hantush_unit_step`. t50 and t95 come from exact root finding. Units are m, d, m²/d and m³/d throughout.

Unit fixtures (`results/phase2/B_preflight/unit_fixtures.json`):
- The kernel agrees with an independent quadrature of W(u, r/λ) to 1.8e−7 of the steady drawdown.
- W(0, ρ) = 2K0(ρ).
- The Theis limit holds.
- The pilot's t95 = 60 d is reproduced.

In the production generator, E_P1(k) equals −q0·A·S(k) to 4.4e−16 m (`B_merge/merge_tests.json`).

### 3.2 Values (D1)

| layer | S or Sy | T (m²/d) | c = a*/S (d) | λ (m) | a (d) | b | π_r | t50 (d) | t95 (d) | gain (m per m³/d) | Q̄·gain (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| confined_T50 | 1e−4 | 50 | 2e5 | 3162 | 20 | 0.001 | 0.02 | 0.63 | 16.83 | 9.17e−3 | 0.917 |
| confined_T500 | 1e−4 | 500 | 2e5 | 10000 | 20 | 1.0e−4 | 0.002 | 0.20 | 13.22 | 1.28e−3 | 0.128 |
| leaky_T50 | 1e−3 | 50 | 2e4 | 1000 | 20 | 0.01 | 0.2 | 2.00 | 22.66 | 5.58e−3 | 0.558 |
| leaky_T500 | 1e−3 | 500 | 2e4 | 3162 | 20 | 0.001 | 0.02 | 0.63 | 16.83 | 9.17e−4 | 0.092 |
| unconfined_T50 | 0.1 | 50 | 200 | 100 | 20 | 1 | 20 | 20.0 | 57.75 | 3.63e−4 | 0.036 |
| unconfined_T500 | 0.1 | 500 | 200 | 316 | 20 | 0.1 | 2 | 6.32 | 33.57 | 2.35e−4 | 0.023 |

The common values are r = 200 m, nominal Q̄ = 100 m³/d, and a* = c·S = 20 d. All six (a, b) lie inside the frozen family box (log10 a ∈ [0, 3.5], log10 b ∈ [−4, log10 25]). No family range is changed.

### 3.3 Why these values, and what they are not
These are **controlled physical regimes** for a twin-truth experiment. They are not a field-measured Korean well. No opened source gives a 200 m observation offset, a metered calendar-day volume of 100 m³/d, or a fitted resistance for these layers (C: `sources/C_physical_followup.md`, `C_physical_followup_manifest.json`).

**Finite memory requires a resistance.** A purely confined (Theis) response has no finite steady gain. So the D03 storage labels acquire a finite response time only through a declared vertical resistance c: aquitard leakage for the confined and leaky labels, and drainage resistance for the linearised unconfined label.
- One shared c is not feasible on the daily grid. With c = 10³ d, t95 spans 0.08–147 d across the storage values, so a ρ = 0.25 rest would round to zero days in four layers (`B_preflight/stratum_scan.json`).
- The layers therefore share a **leakage time** a* = c·S = 20 d, which makes c inversely proportional to S.
- Storage then acts through the leakage length λ, the onset time t_r and the gain. Smaller S gives a faster and larger response, which is the D03 premise. t95 stays within 13–58 d.
- The storage label and the resistance vary together. Layer contrasts are contrasts between these coupled regimes, **not effects of the storage coefficient alone**.
- Handbook ranges: SATEM (Boonstra & Kselik 2001, p. 22) treats c of about 2,000–2,500 d or more as aquiclude-like; Kruseman & de Ridder (2nd ed., p. 24) give some hundreds to several ten-thousand days. So 200 d lies in the aquitard range. 2e4 d and 2e5 d are finite resistances of nearly confining layers. They are not measured leaky aquitards.

**r = 200 m** is the adjustment that D03 asks for when π_r does not cover 1.
- At the measured Korean offsets of 5–26 m (Mok et al. 2018, Table 4; Lee et al. 2014, Table 1), all six π_r are below 1.
- At 200 m, both unconfined layers are above 1 and the other four are below.
- 200 m lies in SATEM's confined placement band (100–250 m or more, p. 24) and outside its unconfined band. One shared r cannot sit in both bands.

**Q̄ = 100 m³/d** is a greenhouse-scale stress scenario. It is below the step-test rates of 198–373 m³/d reported by Mok et al. (2018, Table 3), and it is the level those authors say water-curtain wells exceed. It is not a metered daily duty, a permit capacity or an annual mean. Q̄ is common to all six layers.

**Sy linearisation** (Tier A) is conditional: the steady drawdown at r must be small relative to the saturated thickness of the layer represented. Under D1 that drawdown is 0.023–0.036 m. The opened sources give alluvium thickness (18–21 m), not the saturated thickness of a represented layer, so the condition is stated rather than verified. Large-drawdown nonlinearity belongs to Tier B.

### 3.4 Derived coordinates (stored per case; never used to rescale pumping)
- SR = A · q̄_out / σ_bg. Here q̄_out is the actual mean pumping over all context days except the designated rest (m³/d). σ_bg is the SD over the context of h_nat + ε, the pilot definition, which is bit-equal to the pilot record (0.140–0.642 m).
- SR_context_mean uses the calendar mean instead of q̄_out and is stored alongside.
- π_r, t95, realized ρ = R/t95, k/t95, rest/t95.

Before scoring, the design check (`B_merge/schedule_check.json`) gives:
- SR stratum medians of 2.89 (confined_T50) and 1.76 (leaky_T50) above 1, and 0.07–0.40 for the other four.
- 171 cases above 1 and 369 below.
- π_r above 1 in two layers and below in four.
- **Both quantities cover both sides of 1.**

Across layers SR and π_r are anti-correlated (confined: high SR, low π_r). This is stated for the collapse analysis (§9.4).

## 4. Pumping record (author matching, root decision)

For each layer l, realization i, ρ and N (day indices 0…1023):

1. t95_l is exact. The **layer rest end** is e_l = 1024 − (⌊t95_l⌋ + 1) − 18. It is fixed for all ρ, N and realizations of the layer. Values: 989, 992, 983, 989, 948 and 972 (layer order of §3.2).
2. R = round(ρ·t95_l), the pilot rule unchanged. The **designated rest** is [e_l − R, e_l). Realized ρ ranges: 0.227–0.265 (nominal 0.25), 0.984–1.015 (1) and 3.980–4.016 (4).
3. The **short-transition prefix** follows the rest. ON on day e_l, then OFF on days e_l+1, e_l+3, …, e_l+N−1. Each interval lasts one day, which is shorter than t95 in every layer. After day e_l+N−1 the pump is ON through day 1023.
   - The prefix starts with ON so that its first OFF cannot join the designated rest.
   - The design has N/2 short ON and N/2 short OFF intervals.
4. The base is base0 = Q̄·m_i, with Q̄ = 100 m³/d and m_i the realization's pilot multipliers (0.8/1.0/1.2, lognormal segments). The actual daily rate is not 100 on every day.
5. **Compensation** uses the window C = [0, e_l − R_max), with R_max = round(4·t95_l).
   - κ_N = 1 + Σ base0[OFF] / Σ base0[C].
   - base_N = base0, with base_N[C] multiplied by κ_N and base_N[OFF] set to 0.
6. Q = base_N, with only the designated rest window set to 0.

**Consequences, verified on all 540 production schedules with the actual seeds (`B_merge/schedule_check.json`, 540/540):**
- **Within** (layer, ρ, realization), the context volume V and the mean over all days except the designated rest are **identical across N**. The maximum relative spread is 6.8e−16 for V and 7.0e−16 for the mean, which is floating-point summation order. This is the author's matching.
- **Across ρ** the pilot convention holds. base_N is bit-identical across ρ (hash equal), and Q differs **only** inside the rest window. A longer rest therefore removes volume. The volume lost between ρ = 0.25 and ρ = 4 is 3.9–21.5% of the ρ = 0.25 volume, depending on layer and realization (largest in unconfined_T50). Actual totals and means are stored per case.
- **q0** = Q[1023] = 100·m_i[1023] is unchanged across ρ and N. It is 80–120 m³/d across realizations.
- κ ranges 1.0008–1.0131. Calendar means are 74.8–105.4 m³/d, and means outside the rest are 95.6–106.8 m³/d.
- **Census of actual maximal ON/OFF runs:**
  - The designated rest is identified by exact run identity and excluded from N.
  - Every other interior run shorter than t95 is counted.
  - Result in every case: N_ON = N_OFF = N/2, all short runs last 1 day, there are no other OFF runs, the first run is ON, and the tail is ON with length ≥ t95.
  - At ρ = 0.25 the designated rest is itself shorter than t95. It is reported as the rest, not as a transition.
- **Timing:**
  - After N = 18 the ON tail lasts ⌊t95⌋ + 1 days, which is 1.004–1.059·t95. For N = 2 it is 16 days longer (1.28–2.27·t95).
  - The fewer-transition records use the leading part of the same alternating chain.
  - The number of transitions and the recency of the last transition therefore move together (16 days between N = 2 and N = 18). The N effect is the effect of this matched record intervention, not a pure count effect valid for every placement.
- **Negative fixtures, all detected:**
  - an OFF-first prefix (it merges with the rest);
  - κ = 1 (volume mismatch across N);
  - compensation over the ρ-dependent outside-rest window (ρ bit identity breaks);
  - a burst ending at the origin (no ON tail, N not certifiable).

The pre-context operation is a hidden constant Q̄ = 100 m³/d. It is generator-only, and the methods see only the observed boundary Q_start plus the nuisance η_pump, as in D02.

## 5. Generator, family, W and δ (D02 v1.2, unchanged)

- The natural component (gamma → linear reservoir with real KMA rain; f, n, θ, τ, S_eff draws), AR(1) noise (φ = 0.8, σ_ε = 0.02 m) and the calendar are frozen. The natural component and noise are shared by all 54 cells of a realization, so layer, ρ and N contrasts are matched in weather and noise.
- Family F:
  - the gamma–reservoir plus Hantush structure;
  - the box above, with finite memory t95 ≤ 1,000 d;
  - linear bounds with A_max = 10·range(h_obs)/median(positive Q) and |η| ≤ 10·range;
  - β_nat ≥ 0 and c0 free;
  - p = 10.
- τ_C = SSE_min·[1 + p/(n_eff − p)·F_0.95(p, n_eff − p)], locked per case.
- The v1.2 bookkeeping applies:
  - best-fit direct seed;
  - full-curve merging of refinement terminals;
  - coherent witnesses;
  - a direct-X certified published envelope;
  - the witness reconstruction check.
  Code: `lib/p3_wenvelope.py` + `lib/p3_wenvelope_repaired_v1_2.py`, unchanged.
- δ(k) = max(0.25|E_true(k)|, 0.02 m), used for evaluation only.
- Pair identity: P2 is exactly −0.5 × P1 in both truth and family. It is checked (≤ 1e−12 m) and reported.
- Truth is used only in the separate evaluation step. It covers compatibility (truth_nonlinear_in_G, truth_full_vector_in_G, A_true_in_direct_profile_interval) and E_true inclusion. None of these is a success criterion.

## 6. Sampling budget (fixed before any score; uniform)

Every one of the 540 cases uses:
- W seed 20260930;
- global Sobol levels 128/256/512/1024;
- local rounds of 512², at most 12;
- the inherited stopping rule: change ≤ 5% and ≥ 200 accepted;
- refinement at leads {10, k*}, with at most 4 restarts.

The budget is the same for every layer and cell. W is a sampled inner envelope, and the toy comparison showed budget-dependent changes of −4.1% to +3.6% that are not monotone (`B_preflight/toy_budget.json`). A budget that varied by stratum would therefore confound stratum contrasts. No budget, resampling or rerun is triggered by any W, tool or truth value.

Per layer and cell, the following are reported:
- the flags `sparse_G`, `not_converged`, `seed_interval_unresolved`, `witness_uncertified` and `direct_audit_material`;
- local rounds, refinement increments, accepted counts and runtime.

Flagged cases stay in all tables. W_status ≠ ok is shown as a flag.

## 7. Scenario tool (frozen TimesFM-3, pilot configuration)

- Checkpoint snapshot `43046b85…`. API exactly as in the pilot: return_quantiles, no symmetric averaging, make_positive = False, sort_quantiles, use_znorm = False, padding none, multivariate.
- No fine-tuning, adapters or LoRA.
- **Input:** the 1,024-day head context, with covariates [pumping feature, raw rainfall] over 1,024 + H days.
- **Four separate tracks,** each on the full grid (540 cases, 1,080 contrasts and 2,160 queries per horizon):
  - raw Q (main);
  - exp10, exp30 and exp90 (supplementary). These use the causal exponential filter z[0] = Q[0], z[t] = e^(−1/τ) z[t−1] + (1 − e^(−1/τ)) Q[t], applied continuously through history and the known future, with τ ∈ {10, 30, 90} d (`lib/p3_phase2_timesfm.py`, A).
  - No scale is selected, and no hydraulic constant enters any feature.
- **Horizons:** H10 and H30 are separate requests.
- **Future schedules,** enforced by the producer (`lib/p3_phase2_export.py`), with q_last = observed Q[1023]:
  - P1: continue = q_last vs stop = 0;
  - P2: current = q_last vs 1.5·q_last.
  - Exact equality is required. A case with observed 80 and future 100 is rejected (`B_merge/merge_tests.json`).
  - Head, rain and pumping history are bit-identical within a case.
- **IDs:** `r{rr}_{storage}_T{T}_rho{ρ}_N{N}`, `…__{pair}`, `…__{a|b}__H{H}`.
- Metadata carries labels only: storage type and value, T, r, Q̄, N, t95, ρ, SR, π_r, units and daily alignment. It is never a covariate. The A validator runs with `--require-full-grid`.

## 8. Structurally matched reference (Pastas)

`lib/p3_pastas.fit_and_propagate`, unchanged:
- the same 1,024-day window;
- the same boundary and η terms;
- family bounds equal to the W box;
- 12-start multistart;
- 1,000 joint parameter draws with seed `seed_pastas_param_sample`;
- E for both pairs at leads 1–30, with H10 and H30 both read.

It is presented as the reference with the declared structure, not as a competitor.

## 9. Metrics and analysis (fixed before scoring)

### 9.1 Row metrics (`lib/p3_phase2_metrics.py`)
For every case × pair × track × lead:
- **Truth class:** nonfinite, exact zero, or finite nonzero.
- **Flags** `small_001` and `small_02` (|E_true| ≤ 0.001 m and ≤ 0.02 m) apply to finite nonzero truth only. They are flags; the rows stay in every denominator.
- **Signed effect ratio** E_tool/E_true is defined when E_true is finite and nonzero and E_tool is finite. It is never clipped. E_tool = 0 gives ratio 0.
- **Sign agreement** uses the same mask. E_tool = 0 counts as disagreement.
- **Absolute error** |E_tool − E_true|.
- **error/W** is defined only when W is finite and W > 1e−6 m. Negative or nonfinite W is flagged as invalid.
- **Envelope inclusion:** inf − TOL_IN ≤ E ≤ sup + TOL_IN, with TOL_IN = 1e−12 m imported from the v1.2 collector.

Every summary reports:
- the total rows;
- the exact-zero, nonfinite and nonzero truth counts;
- both flag counts;
- the ratio, sign, error/W and inclusion denominators;
- the invalid-width count.

Marginal widths (w_a, w_b, w_a + w_b) are supplementary only. They contain natural and noise variance that cancels in E, so they are not compared with W in the main text.

### 9.2 Primary quantities
- Raw W(10) and W(30) in metres, for pair P1. P2 is reported and checked through the identity.
- The same leads for E_tool/E_true, sign agreement, error/W and inclusion, raw track (main); the three filter tracks as supplements.
- Supplementary: max_{1–10} W for the raw track, plus the normalised forms W/q0, W/|E_true| and log W (never substituted in Figure A).

### 9.3 R1–R3 comparisons
A block is (realization × layer): 60 blocks of 9 cells each.
- **N contrast:** ΔW_N(k; ρ) = W(k; N = 18) − W(k; N = 2), in metres, for each ρ.
- **ρ contrast:** ΔW_ρ(k; N) = W(k; ρ = 4) − W(k; ρ = 0.25), in metres, for each N.
- N = 6 and ρ = 1 are shown as intermediate levels.

Each contrast is summarised over blocks by:
- the median and IQR (m);
- the fraction of blocks with Δ < 0;
- the median relative change ΔW/W_reference (supplementary);

given per layer, per storage label (pooling both T values) and pooled.

- **R1 read-out (k = 10):** the direction of the median ΔW_N(10), the fraction of blocks that are negative, and the ratio of median |ΔW_N(10)| to median |ΔW_ρ(10)|.
- **R2 read-out (k = 30):** the same, with ρ and N exchanged.
- **R3 read-out:**
  - the same N and ρ contrasts on the tool's absolute error |E_tool − E_true| (m) and on the signed ratio E_tool/E_true, raw track;
  - the filter tracks reported alongside, each separately.
- These are descriptions of direction and magnitude. **No threshold, significance test, or positive-completion gate is attached.** Whatever direction and size are observed are reported, including reversed and null.
- "The answer depends on storage" is stated only by comparing these per-layer read-outs.

### 9.4 Collapse (Figure A2, supplementary to Figure A)
- Cell medians of W(10) and W(30) (raw, P1) are plotted against the derived SR and π_r, with colour for the storage label and marker for T.
- Descriptive measure: the share of the variance of cell-median log W across the 6 layers × 9 cells that is explained by a linear model in log SR, log π_r and log(ρ_realized), compared with the same model plus layer indicators.
- The layer pair confined_T50 / leaky_T500 has an identical unit response (same a, b) and a 10× gain difference. It is reported as an SR-only comparison.
- Limits printed with the figure: six layers; SR and π_r anti-correlated across layers; π_r constant within a layer. No universality claim is made.

### 9.5 Figures and report
- **Figure A:** W(10) and W(30) on the N × ρ plane, one row per storage label, T = 50 and 500 as separate columns.
- **Figure A2:** the collapse plot.
- **Figure B:** E_tool/E_true on the same plane, centred at 1; cells with partial denominators are marked defined/n.
- **Supplementary figures:** exp10, exp30 and exp90 ratio planes, the day-1–10 maximum, and P1 vs P2 widths.
- Figures are in English, in Times New Roman, with no model name in titles, and are checked visually. They are drawn by `lib/p3_phase2_figures.py` (C) from `cell_metrics.csv`, which follows `FIGURE_INPUT_CONTRACT.json` and is written by B with `data_status = official_analysis`.
- The report's first paragraph gives:
  - R1–R3 direction and size;
  - whether the two leads responded to different record features;
  - how the answer changed across the storage labels;
  - whether the results gathered on the derived coordinates;
  - whether the tool's effect ratio changed with record information.
- The subject of every result sentence is the record condition. There are no model ranking tables. Requested items and self-imposed items are listed separately (README rule 7).

## 10. Execution (after this freeze; not part of the freeze turn)

1. `lib/p3_phase2_make_cases.py --out results/phase2/cases`. It checks this hash and writes the truth-free tf_inputs, the separate truth files, `derived_manifest.json` and the validated `tool_inputs_H10.npz` / `tool_inputs_H30.npz`.
2. A: the four tracks × two horizons, run through `lib/p3_phase2_timesfm.py` against these archives.
3. B: v1.2 W for all 540 cases under the §6 budget, then the separate truth evaluation, then the reference.
4. B: `cell_metrics.csv`. It holds 8,640 lead rows (P1/P2 × 4 tracks × 2 leads) plus 1,080 raw max_{1–10} rows, together with the full denominators.
5. Figures (C), report (B), and one independent review.

Execution runners must import the frozen modules unchanged and record their own hashes.

## 11. Scope

This is a Tier A controlled twin-truth study. It asks which features of an operational pumping record constrain which pumping what-if, in six coupled storage–resistance regimes, within a correctly specified declared family. Its conclusions are conditional on those regimes, the declared family, the matched schedule template and the frozen tolerance.
