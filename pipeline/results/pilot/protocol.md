# D02 Tier A pilot — Phase 0 protocol DRAFT (Role B, executor)

Status: **FROZEN (v1.0), 2026-09-30.** Merged from draft v0.2, A peer check, root join and C literature check (`phase0_merge.md`). Frozen before any pilot case was generated and before any W, tool forecast or score. SHA-256 in `protocol.sha256`; code, schema and source hashes in `protocol_freeze.json`.

**Change log, v0.1 → v0.2 (B, after root concerns and A `phase0_peer_check.md`):**
(1) All methods now see the same 1,024-day record. The 365-day real pre-window rain and the declared Q0 are removed from the W and reference inputs. The initial state is the observed boundary (P̄, Q_start) plus fitted nuisance η_nat and η_pump. The generator starts from equilibrium under the context-mean rain, so p = 8 becomes p = 10.
(2) The v0.1 claim that "w_a + w_b < W proves overconfidence under any dependence" was wrong and is withdrawn. `l_comon` and `u_indep` are removed, and the proxy is renamed as descriptive.
(3) The transfer function is now the structurally matched reference: composite gamma→reservoir and exact Hantush, with joint draws.
(4) A_max uses the median of the observed positive Q.
(5) The W algorithm is implemented and measured on fixtures: global, bounded local stage, refinement restarts.
(6) Results are presented condition-first, following LATEST_STEERING.
(7) The δ wording is corrected.

Authority: `direction/D02_decisions_and_pilot.md` (identical to the copy in `results/pilot/SOURCE_PACKET.md`, checked by `diff`), approved `debate/REPORT.md`, `inputs/research_plan_FINAL.md` §5.1–5.7 and §13, and README rules 1–7. D02 supersedes the D01 no-experiment rule.

Every number below is a **proposed design value**, not a result. The only computed numbers are kernel-design and API checks in `results/pilot/phase0_checks/`. They involve no pilot cell, no KMA-driven series and no scoring.

---

## 1. Questions and estimands (fixed wording from D02 §2)

Main question (D01 §1 recommended sentence): *How much drawdown and recovery, relative to the aquifer response time, and how large a pumping-induced head variation, relative to background variability, must an operational pumping–head record contain to constrain the uncertainty in the head difference between two future pumping schedules to a predeclared width within a specified response family?* Sub-question: *and does a frozen pretrained forecaster, used as a scenario engine, respect that limit?*

Definitions follow D01 §1 and are not changed:

- C: the operational record of one case: heads, pumping and rainfall on the same 1,024-day context window, identical for every method (§6.4), plus the known future rainfall and the two future schedules.
- G(C): the set of responses g in the declared family F (§6) whose past fit to C passes the past-fit tolerance (§7).
- E_g(k) = h_g(Q_a; k) − h_g(Q_b; k), for leads k = 1…30 (days after the origin).
- W(C, k) = sup_{g∈G(C)} E_g(k) − inf_{g∈G(C)} E_g(k). W is the envelope of declared alternatives. It is not a probability interval.
- E_true(k): the same difference computed from the noise-free generator state h* (plan §5.2). Noise never enters the truth.

**Positioning (from `debate/literature_check.md`, C):** axis 1 overlaps Brakenhoff et al. (2022): criteria 3–4 and a §5 two-regime head difference with a fitted-model interval. It also overlaps Kruseman and de Ridder's recovery-duration conditions and the data-worth literature (Dausman et al. 2010; Vilhelmsen and Ferré 2018). No "first" claim is made. W as a rest-dependent envelope against twin truth, and a frozen scenario engine judged against W, were not found in the texts read.

**Scope statement (must appear in every report):** W is conditional on the declared family F and its parameter box. It is not a universal information limit. In this pilot the generator lies inside F, so the family is correctly specified. W therefore measures record ambiguity in the most favourable structural case and is not a bound for misspecified real aquifers.

## 2. Leads and summaries (D02 §2)

- Main lead window: k = 1…10. Diagnostic: k = 1…30.
- Per-lead curves W(k), E_true(k) and the tool outputs are stored for k = 1…30.
- Reported summaries: **W(10)** and **max_{k=1…10} W(k)**, both always reported. Diagnostic: W(30) and max_{k=1…30} W(k).
- The tools are run as separate horizon-10 and horizon-30 requests, never truncated from one run (A preflight: TimesFM-3 masks covariates beyond the requested horizon, so the two requests are different queries). The main summaries use horizon-10 runs.

## 3. Real rainfall: sites and read-only mapping

Source files are read-only and are never copied into Paper 1 or Paper 2 directories. Paths:

- Rain: `0_zero_shot/data/raw/resubmit/data/wt_cl_pairs/climate/<stem>_CL.txt`, column `RAIN` (tab-separated; `Date` YYYYMMDD). Provenance: `0_zero_shot/inventory.md` and `MANIFEST.txt` (SHA-256 of copies equal to the originals).
- Station identity: `1_zero_shot/results/KMA_mapping/stage_b/station_mapping.csv`, rows with `variable == RAIN`. Identity was recovered by day-by-day matching to the KMA archive.

**Eligibility rule, applied before selection** (computed by a read-only scan of all 50 CL files, see §13):

1. Full record length, 7,305 days (2005-01-01 to 2024-12-31). This is the maximum in the dataset, held by 33 of 50 series.
2. Zero missing `RAIN` values. 15 series meet rules 1 and 2.
3. Identity `verified` in the Paper 2 mapping, with network ASOS and an exact match on every compared day (`frac = 1.0`, MAE 0). ASOS was preferred because its identity is exact. Several AWS identities are exact only on a subset of days. Four series pass: Andong (ASOS 136, stem `안동태화_충적`), Mungyeong (ASOS 273, `문경영순_암반`), Sancheong (ASOS 289, `산청산청_암반`) and Namhae (ASOS 295, `남해남해_암반`).
4. Climatic spread: take the driest and the wettest of the survivors by mean annual rain, then the survivor closest to their midpoint.

| site | KMA station | stem | mean annual rain 2005–2024 (mm, computed) | wet days ≥1 mm per yr | RAIN missing |
|---|---|---|---:|---:|---:|
| Andong | ASOS 136 | 안동태화_충적 | 1011 | 72 | 0 |
| Sancheong | ASOS 289 | 산청산청_암반 | 1528 | 79 | 0 |
| Namhae | ASOS 295 | 남해남해_암반 | 1934 | 80 | 0 |

(Mungyeong, 1315 mm, is 157 mm from the midpoint of 1472 mm. Sancheong is 56 mm from it.)

Rainfall is never synthesised or edited. It is read as mm/day and used unchanged as (i) the generator forcing and (ii) the rainfall covariate given to both tools. Temperature is not used: the Tier A generator has no ET term (plan §5.1), so no tool receives temperature. A's adapter passes pumping and rainfall only.

## 4. Grid: 5 × 2 × 2 × 10 = 200 cases, 2 schedule pairs each (400 E contrasts)

| factor | levels | definition |
|---|---|---|
| rest ratio ρ = R / t95 | 0.25, 0.5, 1, 2, 4 | R: length of the one full stop (Q = 0) in the context. t95: exact time for the true step response to reach 95% of gain (§5) |
| signal ratio SR | 0.5, 2 | A_true·Q_ref / σ_bg, where σ_bg is the SD over the context window of the no-pumping observed series h_nat + ε |
| π_r stratum | 0.5, 5 | π_r = t_r/Δt = r²S/(4TΔt), Δt = 1 day |
| realization | 0–9 | site, calendar, natural-component parameters, noise series and base-schedule seed (§8) |

No grid level is dropped or merged. The D02 example levels are adopted as given.

**Realization-to-site allocation (explicit):** realizations 0–3 use Andong, 4–6 Sancheong and 7–9 Namhae. Sancheong and Namhae use the same three calendar windows, which gives a matched site contrast on the same dates. The Andong windows are shifted by one quarter-year each so that their origins span the seasons. Origins are listed in `case_schema.json` → `realizations` and in `phase0_checks/realization_calendar.json`.

**Matching:** all 20 cells of a realization share the site, the dates, the rainfall, the natural-component parameters, the noise series ε (same seed, bit-identical) and the base-schedule multipliers outside the rest segment. Cells differ only in (i) the rest segment, (ii) the pumping kernel (π_r stratum) and (iii) Q_ref (set by SR and the stratum's physical gain). Every rest-ratio contrast is therefore matched in calendar, weather and noise, as are every SR contrast and every π_r contrast.

## 5. Pumping response: finite-memory Hantush leaky-well kernel

### 5.1 Formula and its verification against the installed code

Hantush–Jacob (1955) leaky confined aquifer drawdown: s(r,t) = Q/(4πT) · W(u, r/λ), with u = r²S/(4Tt), λ = √(T c) and W(u, ρ) = ∫_u^∞ y⁻¹ exp(−y − ρ²/(4y)) dy.

The installed Pastas 2.0.0 (`env/.venv_pilot/.../pastas/rfunc.py`, class `Hantush`, `impulse`) uses θ(t) = A / (2 t K0(2√b)) · exp(−t/a − a b / t). Substituting y = t_r/τ into W shows that this is the same kernel with

- a = c S (days), b = r²/(4λ²) = r²/(4Tc), a·b = t_r = r²S/(4T), ρ = r/λ = 2√b,
- gain A = K0(r/λ)/(2πT) (head per unit pumping rate), and s(t)/A = W(t_r/t, 2√b) / (2 K0(2√b)).

This mapping is my own derivation. It was checked numerically in `phase0_checks/hantush_kernel_check.py` (JSON output alongside):

- the exact integral of s/A tends to 1.000000 for three (a, b) pairs;
- the Pastas `numpy_step` (Veling & Maas 2010 approximation) deviates from the exact integral by at most 0.15–0.49% of the gain on four test kernels;
- **Pastas `get_tmax(cutoff=0.95)` (Lambert-W approximation) returns 97.3 and 84.8 days where the exact t95 is 60.0 days.** It is therefore not used for any design quantity. All t50, t95 and t99 values come from exact root finding on W(u, ρ).

The generator, the W family and the transfer-function reference all use the exact kernel. The reference uses `HantushExactP3`, which equals Pastas `quad_step` to 5.5e−7 of gain (§11.2). The Veling–Maas approximation is therefore not used.

### 5.2 Response time, π_r and the rest axis are separate quantities

- Response time t_resp := **t95**, the time at which the exact step response reaches 95% of the gain. This is the same quantity as the t_95 in Brakenhoff et al. (2022) reliability criterion 3, so the result can be read against that rule.
- π_r = t_r/Δt is the early-time (onset) coordinate. It stratifies the grid and is never used as the x-axis (D01 §3).
- Rest ratio ρ = R/t95 is the x-axis. R is the observed full stop in days.

The design fixes **t95 = 60 d in both strata**, so that R in days is identical across strata for a given ρ and the strata differ only in the shape of the early response. The unique solutions (exact) are:

| stratum | a (d) | b | t_r (d) | t50 (d) | t95 (d) | t99 (d) | s/A at 1 d | at 10 d | at 30 d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| π_r = 0.5 | 53.485 | 0.0093485 | 0.5 | 5.17 | 60.0 | 119.2 | 0.155 | 0.650 | 0.864 |
| π_r = 5 | 32.776 | 0.15255 | 5.0 | 12.80 | 60.0 | 100.2 | 0.001 | 0.404 | 0.805 |

Rests: R = 15, 30, 60, 120, 240 days. The 1,024-day context is 17× t95, so every case satisfies Brakenhoff's t95 ≤ ½ calibration-period rule. The pilot tests what the rule does not state: whether the observed rest still changes the width W.

### 5.3 Physical parameter mapping (declared, for reporting)

A reference aquifer: T = 50 m²/d and S = 1×10⁻³ in both strata. π_r = 0.5 → r = 316.2 m, c = 5.35×10⁴ d, λ = 1,635 m. π_r = 5 → r = 1,000 m, c = 3.28×10⁴ d, λ = 1,280 m. These give b = 0.0093 and 0.153 respectively (checked against the table). The physical gain is A_true = K0(2√b)/(2πT) = 5.68×10⁻³ and 1.85×10⁻³ m per m³/d. No inversion for T, S or c is reported (D01 §3).

## 6. Generator (Tier A only) and the declared W family

### 6.1 Natural component (plan §5.1–5.2)

- Recharge R = f · (g_Γ(n, θ) ∗ P), with daily gamma blocks g[m] = F(m+1) − F(m), where F is the gamma CDF. Linear reservoir with exact daily recursion h[n] = e^(−1/τ) h[n−1] + (1 − e^(−1/τ)) R[n] (unit gain). Physical gain f·τ/S_eff (m per mm/d). f = 0.15, an assumed regional value; annual recharge is then 152–290 mm across the three sites.
- **Initial state (v0.2, peer correction 1):** before the context the filter and reservoir are at steady state under the **context-only mean rain P̄**, a declared boundary computed from the observed window. No real pre-context rainfall enters the truth or any tool. The real KMA rain inside the window and over the 30 future days is used unchanged.
- Per-realization draws (seed `seed_natural`, order n, θ, τ, S_eff): n ~ U(1.5, 3.0), θ ~ U(5, 20) d, τ_nat ~ U(30, 120) d, S_eff ~ U(0.03, 0.10).

### 6.2 Pumping component and schedule

- h_pump = A_true · [Q ∗ B](n) with the exact daily block response (B[m] = S(m+1) − S(m), head at the end of the day). h* = h_nat − h_pump; h_obs = h* + ε.
- Pre-context pumping in the generator is a constant Q_ref, kept hidden from all methods. Inverse methods use the observed boundary Q_start = Q[context day 0] plus a free nuisance amplitude (§6.5). The implied true nuisance coefficient A(Q_ref − Q_start) is stored in the truth record for diagnostics only.
- Base operation: irregular exogenous multipliers m ∈ {0.8, 1.0, 1.2}, lognormal segments (median 20 d, σ_ln 0.6, truncated to [3, 90] d), seed `seed_schedule`, independent of weather and head. There is one designed rest, Q = 0 on context days [874 − R, 874). Q_ref = SR·σ_bg/A_true, and q0 = Q[1023] > 0.

### 6.3 Observation noise

Stationary AR(1), φ = 0.8, σ_ε = 0.02 m, seed `seed_noise`, bit-identical across the 20 cells of a realization. Truth uses h* only.

### 6.4 Record C seen by each method (v0.2: identical for all methods)

| method | heads | rainfall | pumping | initial state |
|---|---|---|---|---|
| W family | context 1,024 d | context + future (known) | context + two future schedules | declared boundary (P̄, Q_start) + fitted nuisance η_nat, η_pump |
| reference (Pastas) | context 1,024 d | same | same | same boundary via Pastas `fill_before` ("mean" for rain, "bfill" for the well) + the same η terms |
| TimesFM-3 | context 1,024 d | context + H | context + H | model-internal |

No method sees generator truth, Q_ref, A_true, hydraulic constants or real pre-context stresses.

### 6.5 Declared family F (conditional; not a universal information limit)

h_g[n] = c0 + β_nat·[(P∗K_nat)[n] + P̄(1 − Kc_nat[n+1])] + η_nat·e^(−(n+1)/τ) − A·[(Q∗B)[n] + Q_start(1 − S[n+1])] − η_pump·(1 − S[n+1]), for n = 0…1023.

- Nonlinear: n ∈ [0.5, 5], log10 θ ∈ [0, 2], log10 τ ∈ [log10 5, log10 500], log10 a ∈ [0, 3.5], log10 b ∈ [−4, log10 25]. Finite memory is imposed as t95(a, b) ≤ 1,000 d (exact root). Candidates that violate it are excluded and counted.
- Linear: A ∈ [0, A_max] with A_max = 10·range(h_obs)/median(positive observed Q), which uses no Q_ref. β_nat ≥ 0. |η_nat|, |η_pump| ≤ 10·range(h_obs). c0 is free. Endpoints that exceed a bound are counted and reported.
- p = 10 (5 nonlinear + 5 linear).
- Kernel convention shared by the generator, W and the reference: exact kernels, no 0.999 truncation inside the window, and an analytic pre-window tail (1 − S) and (1 − Kc). The reference's Pastas warm-up (3,650 d, filled stresses) truncates only the tail beyond 3,650 d. That residual is absorbed by η_pump; the reference's consistency check gives 1.4e-7 m.
- Truth lies exactly in F. The fixture check reproduces noise-free h* with RMSE 1.4e−14 m and recovers the coefficients to 4e−13 relative (`prep_checks.json`). W is therefore the ambiguity of a correctly specified family: a best case, not a bound for misspecified aquifers.

## 7. Past-fit tolerance and G(C) (truth-free)

SSE on the 1,024 context days; E_g depends only on (A, a, b), and every other term cancels between schedules.

1. SSE_min is the minimum of (i) the feasible SSE over the global 512² product design and (ii) a bounded variable-projection Powell search (bounded linear least squares, `lsq_linear` BVLS) started from the 20 best design points.
2. n_eff = n(1 − r1)/(1 + r1), where r1 is the lag-1 autocorrelation of the best-fit residual, clipped to [p + 10, n].
3. τ_C = SSE_min·[1 + p/(n_eff − p)·F_0.95(p, n_eff − p)], p = 10. **This is a heuristic fit-discrepancy rule, not an exact 95% coverage statement for nonlinear AR(1) fitting.** τ_C is locked once per case after the best-fit stage and reused at every sampling level, in the local rounds and in refinement.
4. Diagnostics only: W at excess × 0.5 and × 2 is computed inline for the same designs.
5. Nuisance and bounds (exact): the linear parameters are projected with batched 5×5 Gram solves (column-scaled, ridge 1e−12 relative). Where the profile points at both ellipsoid endpoints Â ± √((τ_C − SSE_loc)·Σ_AA) ∩ [0, A_max] satisfy all bounds (β_nat ≥ 0, |η| ≤ 10·range), those endpoints are exact, because the constrained set is contained in the unconstrained one. Otherwise the feasible A-interval is computed exactly: the convex profile f(A) = min over the other linear parameters within bounds of SSE, by BVLS on the Gram Cholesky factor, with bisection for f = τ_C, or rejection if min f > τ_C. The count of bound-active combinations is reported. The 20 best design combinations are re-evaluated with the exact bounded SSE for SSE_min. Refinement uses the same exact routine. Verified against a brute-force bounded grid with bounds inactive and forced active (`prep_checks.json`, `exact_bounded_interval_vs_bruteforce`).
6. **Empty G / failures:** G contains the best fit whenever the fit succeeds, so empty G means a computational failure. Such a case is labelled `W_failed`, counted, and never described as an information limit. There is no "≥200 accepted" gate. Accepted counts are reported, and `sparse_G` (fewer than 200) is only a flag.
7. Truth compatibility (validation only): SSE at the true nonlinear parameters, truth ∈ G, and E_true inside the envelope per lead.

## 8. W computation, sampling and convergence (implemented: `lib/p3_wenvelope.py`)

- **Global stage:** nested scrambled-Sobol prefixes of 128, 256 and 512 natural points (3-D) × the same numbers of pumping points (2-D, the first N that satisfy the t95 filter). Every product combination is solved in closed form from cached columns and cross-Gram BLAS blocks. There is no [N, N, T] prediction array; memory is streamed per chunk of 64 natural points. W per level is recorded.
- **Local stage:** product Sobol of 256 × 256 inside the box spanned by all accepted points (or the 20 best fits if fewer than 5 are accepted), padded by 25% of the span and at least 2% of the global range. There are at most 8 rounds, each merged into the envelope. The loop stops when the change in W(10) and max_{1–10} W is ≤ 5% and cumulative accepted ≥ 200. The bound of 8 rounds replaces the unbounded rule of v0.1. I kept convergence-driven stopping rather than the peer's fixed two rounds, because two rounds can stop before convergence while this rule is still bounded.
- **Refinement:** at k = 10 and k* = argmax_{1–10} W, Powell maximises and minimises E(k) in the continuous 5-D space. A continuous penalty applies outside G, and a value is accepted only at a strictly feasible end point. The search restarts from its own optimum until the increment is ≤ 5%, at most 4 restarts. The first-pass gain over the sampled envelope is reported as a sampling-adequacy diagnostic.
- **Converged** := the local-stage change is ≤ 5% and the last refinement increment is ≤ 5%. Otherwise the case is flagged `not_converged` and its values are reported.
- Pair identity (implementation check): sup_P2 = −0.5·inf_P1, inf_P2 = −0.5·sup_P1 and W_P2 = 0.5·W_P1. The fixture maximum deviation is 2.8e−17 m.
- **Measured feasibility on FIXTURE cells (synthetic rain; these are not pilot numbers):** 4–10 s per case for W (with exact bounds), 15–28 s for the reference (12 multistarts plus 1,000 joint draws). This projects to about 84 min for 200 cases. All 8 fixture cells converged; in two of them refinement first added 6–10%, and the restarts then settled.

## 9. Future schedule pairs (both, reproducible)

At the origin (context index 1024) the pump runs at q0 = Q[1023] > 0.

| pair | Q_a (days 1…H) | Q_b | E = h(Q_a) − h(Q_b) |
|---|---|---|---|
| P1 `continue_vs_stop` | q0 | 0 | negative |
| P2 `current_vs_1p5x` | q0 | 1.5·q0 | positive |

Only the future pumping differs. Future rainfall is the actual record and identical within a pair; histories are bit-identical (checked by A's validator and the generator assertions).

## 10. δ (frozen before scoring; evaluation only)

δ(k) = max(0.25·|E_true(k)|, σ_ε = 0.02 m). `determined_lead10` := W(10) ≤ δ(10); `determined_1to10` := max_{k≤10}[W(k) − δ(k)] ≤ 0. The label means only that the declared width threshold is passed. Truth-in-envelope and sign resolution are reported separately, because a narrow envelope need not contain truth or exclude zero. The noise floor is a declared evaluation scale, not a claim that no record can resolve effects below one measurement's SD. W/|E_true| and W/σ_ε are always stored. δ uses truth, which is allowed because δ never enters G(C).

## 11. Scenario tools on the same record

### 11.1 Frozen TimesFM-3 (A's adapter, unchanged schema)

- Point contrast E_TFM(k) = q50_a(k) − q50_b(k), a contrast of marginal medians.
- Widths: w_a and w_b (q90 − q10 of each schedule), and **width_proxy = w_a + w_b** with rectangle endpoints (q10_a − q90_b, q90_a − q10_b). These form a **nonprobabilistic marginal-range difference proxy**.
- **What does not follow (v0.1 error withdrawn):** two marginal quantile pairs bound neither E's 10–90 range nor its coverage under any dependence. A's non-pilot counterexample and the Makarov-type argument (80% of a difference is bounded only by 90% marginals) show this. Marginal widths also contain natural and noise variance that cancels in E, and W is a set envelope rather than a probability width. `l_comon` and `u_indep` are removed.

### 11.2 Structurally matched transfer-function reference (`lib/p3_pastas.py`)

- It uses Pastas 2.0.0 machinery (Model, ArNoiseModel, LeastSquares, covariance sampling) with custom response classes matching the declared structure. `GammaReservoirP3` is the same discrete kernel as the generator (difference 0.0). `HantushExactP3` is the exact step, equal to Pastas' own `Hantush.quad_step` to 5.5e−7 of gain; it replaces the default approximation and the slow quad loop.
- The same observed boundary and η nuisance (`StressModelIS`) apply, with the (a, b) bounds of family F. It has a declared 12-start multistart over (a₀ ∈ {10, 100, 1000}, b₀ ∈ {1e−3, 1e−2, 1e−1, 1}), selected by Pastas' noise-model objective, and the per-start log is stored.
- Paired E comes from **joint** draws `get_parameter_sample(name=None, n=1000, max_iter=50, seed)`. The well parameters are extracted by name, and the same draw is applied to both schedules. w_TF = q90 − q10 of E_s. This is a Gaussian/local-covariance approximation, and the accepted count is stored.
- Consistency: convolution E against Pastas simulation of the schedule-extended stress, tolerance 1e−6 m (implementation). The observed maximum is 1.4e−7 m, caused by Pastas truncating at the simulation length.
- It is presented as the reference with the declared structure. It is not a competitor, and no winner or loser table is made.

### 11.3 Axis-2 quantities (computed only after the hash freeze)

- (a) Envelope inclusion of each tool's E(k) at k = 10 and for all k in 1–10. Secondary: the signed distance outside the envelope divided by δ(k).
- (b) Width response: tool width (w_a, w_b, width_proxy; w_TF) against W at k = 10 and max_{1–10}. `proxy_below_family_width` is a descriptive discrepancy rate, **not** a guaranteed overconfidence rate. Width-response tracking is the Spearman correlation across the 5 rest levels within each matched block (realization × SR × π_r). A block of constant width gives an undefined correlation, which is reported with its count rather than coerced. When W = 0, ratios are stored as undefined with a zero-W indicator.
- Also stored: E_tool − E_true.
- Reporting per plan §5.7: cell medians and IQRs across realizations. There are no significance tests.

## 12. Outputs and presentation (LATEST_STEERING 2026-09-30)

- Artifacts after the freeze: `cases/` (tool-input NPZ per A's schema; the tf_input and truth NPZ kept apart), `W/`, `tools/`, `tables/W_table.csv`, `tables/tool_table.csv`, and two figures (matplotlib-quantitative-figures skill: Times New Roman, no Korean, visual overlap check).
- Figure 2: x = ρ, y = W (m), SR distinguished, π_r panels. Axis-2 figure: tool width against W with inclusion marked, grouped by record condition.
- **Results are organised by record condition** (ρ, SR, π_r). The subject of each conclusion is "the record" or "a scenario tool", never a model name. There are no ranking tables or cell win/loss tables in the body or the report's first paragraph. Every result sentence is checked against the "model X works, Y does not" reading and rewritten with the record condition as the subject. The transfer function is labelled the structurally matched reference.
- Completion means the requested items are produced; findings may be negative. No monotonic or numeric success criterion is set.

## 13. Provenance and environment

- Rain: read-only scan of 50 CL files. Selected SHA-256 values are verified at load by `p3_generator.load_rain`; they are recorded in `prep_checks.json`.
- Environment: `2_zero_shot/env/.venv_pilot` (uv, Python 3.11.14). pastas 2.0.0, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, numba 0.67.0, matplotlib 3.11.2, and tqdm 4.70.1, which pastas imports without declaring. No Paper 1 or Paper 2 environment was modified.
- Code: `lib/p3_kernels.py`, `lib/p3_generator.py`, `lib/p3_wenvelope.py`, `lib/p3_pastas.py`, `lib/tests/p3_fixture.py` (FIXTURE only) and `lib/tests/p3_prep_checks.py` → `results/pilot/prep_checks/prep_checks.json` (15/15 pass).
- Phase 0 kernel checks: `phase0_checks/`. The superseded v0.1 is at `phase0_checks/protocol_draft_v0.1_superseded.md`.

## 14. Decisions resolved at freeze (root join, 2026-09-30)

1. Root adopts the peer's ordinary defaults as implementation choices inside D02 (§§6–11 as written).
2. Root accepts two documented deviations. (i) The fast exact Hantush integral replaces Pastas `quad=True`: it is the same integral, verified against `quad_step`, and shared by the generator, W and the reference. (ii) The local stage stops on convergence with a hard cap of 8 rounds and at most 4 refinement restarts; actual counts, flags and times are recorded.
3. Exact bounded linear feasibility (§7.5) replaces the v0.2 inner approximation and the counted-but-unenforced η bounds (see `phase0_merge.md`).
4. Implementation tolerances (1e−6 m reference consistency, ≤5% convergence, restart cap) are numerical checks, not scientific success criteria. There is no positivity, monotonicity or accepted-count criterion.

## 15. Execution contract (after freeze authority)

1. The prep checks (`prep_checks.json`, 16/16) were run on the frozen W, generator, kernel and reference code; their hash is in `protocol_freeze.json`.
2. `lib/p3_make_cases.py --out results/pilot/cases` (checks `require_frozen`) writes the 200 cases: tool NPZ for H10 and H30 (800 rows each), tf_input and truth NPZ kept apart, manifests, `generation_checks.json`, then `cases_ready.json`.
3. A: `timesfm_pilot_adapter.py --validate-only`, then H10 and H30.
4. B: `WEngine.run` and `truth_compatibility` per case, and `fit_and_propagate` per case (about 78 min projected). Convergence and flags are logged per case.
5. Scoring tables and figures organised by record condition; the report per D02 §4, with requested items and self-imposed items listed separately.
