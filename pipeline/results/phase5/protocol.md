# D06 final scientific protocol: genuine AR increment transfer

Status: **complete pre-score protocol frozen** by executor (HIGH), turn run. A/B implementation and the genuine-AR C delta are terminal and their actual source/fixture receipts are bound below. This freeze author turn performs no official generation, W, model query or scientific scoring. Inherited A/B scientific author is executor; C scientific author is executor. Full D06 remains open until A–C results, manuscript, independent reader note and root delivery exist.

Inputs:
- Authority: `author instruction` and README rules 1–7.
- Selected decisions: `PROJECT_DESIGN_DECISION_v2.json` for A/B corrections and `C_METRIC_DECISION.json` for the superseding genuine-AR C selection.
- Proposals: `DESIGN_SHARD_B.md` and `DESIGN_SHARD_A.md`.
- Frozen inheritance: D03, D04 and D05 protocols. D05 SHA is 49942fd4…3831.

Machine-readable interfaces are in `IMPLEMENTATION_INTERFACES_v2.json`. Source checks and flags are in `C_AR_ADDENDUM_AUDIT.json` and `C_AR_DESIGN_ADDENDUM.json`.

Two rules apply throughout:
- Everything inherited from D03–D05 stays exactly as frozen unless this file names a change. Old files, old W, old reference and old tool outputs are never rewritten or rerun.
- An undefined, aliased, unbounded, separated or empty result is a result. It always keeps an explicit status. No positive outcome, agreement level or R² is a completion criterion.

## 0. Global identities and constants

| symbol | value | source |
|---|---|---|
| context n, horizon | 1024 d, 30 d | p3_generator |
| noise | AR(1), σ = 0.02 m marginal, φ = 0.8, Σ_ij = σ²φ^\|i−j\| | p3_generator.Design |
| family parameters (p = 10) | v = (n, log10θ, log10τ, log10a, log10b, A, η_pump, c0, β_nat, η_nat) | p3_wenvelope |
| original family box | n∈[0.5,5], log10θ∈[0,2], log10τ∈[log10 5, log10 500], log10a∈[0,3.5], log10b∈[−4, log10 25], t95≤1000 d | p3_wenvelope, p3_pastas.FAMILY |
| linear bounds | A∈[0, 10·ptp(h)/median(Q>0)], β_nat≥0, \|η_pump\|,\|η_nat\|≤10·ptp(h), c0 free | p3_wenvelope |
| tolerance | T_C = SSE_min(1+γ), γ = p/(n_eff−p)·F_0.95(p, n_eff−p), n_eff = clip(n(1−r1)/(1+r1), 20, n) | p3_wenvelope.tolerance |
| pairs | P1 = (q_o, 0), P2 = (q_o, 1.5q_o), E = h_a − h_b. Identities: E_P2 = −0.5E_P1, W_P2 = 0.5W_P1 | D05 §2 |
| bootstrap | 999×10 multiplicity matrix, PCG64 seed 20261001 | statistics.json `bootstrap.draw_matrix` |
| bootstrap logical SHA | 7520bcd1cd416b25f22e1e3c7ce6f3905209b837eac4d5a703a038b79f7aa30e (int64 little-endian bytes) | statistics.json |
| bootstrap physical SHA | d2c3ea3a3359fadf0a6d9d58420781f560a92c5c42a3c3c10d774c0e210ed5a7 (`bootstrap_draw_matrix.npy` file bytes) | file |

The .npy-loaded array and statistics.json multiplicity matrix are element-equal, including shape/dtype. The .npy container bytes and logical little-endian int64 bytes are different hash objects. Both digests are pinned and stored separately. Clusters are realizations 0–9. One draw multiplies every A, B and C row in the same way.

## 1. A: operating calendars at r = 20 m

### 1.1 Inventory

- Cases: 3 forms × 12 origins × 6 layers × 10 realizations = **2,160**.
- Identity: `d06A_{form}_m{MM}_r{RR}_{sid}`, with `source_phase=phase5` and `experiment_group=A_distance20`.
- Parent link: `parent_source_phase=phase4`, `parent_case_id=d05B_{form}_m{MM}_r{RR}_{sid}`.
- Zero-origin cases: 918. Water-curtain variant cases: 216.
- Model queries: 2 pairs × 2 schedules × independent H10/H30 = **17,280** (8,640 per horizon).

### 1.2 Generation (after the final freeze, once)

The new case is built from the hash-verified parent D05 B `tf_input` and `truth`.

**Copied byte-for-byte from the parent:**
- Q_context, all four future_Q arrays, rain, dates, h_nat and ε.
- Seeds, station, origin date, Q_pre = 100, q_origin and q_last.

**Recomputed with r = 20 m** (frozen `p3_phase2_physmap` formulas; Hantush block from frozen `p3_kernels`):
- λ = √(Tc), a = cS, b = r²/(4Tc), π_r = ab, A = K0(r/λ)/(2πT), and exact t50/t95.
- Pumping head: h_pump = A[Q_pre(1−S(t)) + (Q*B)].
- Heads: h* = h_nat − h_pump, h_obs = h* + ε.
- Effects and tolerances: E, δ = max(0.25|E|, 0.02).
- Observed features: L/t95, D, SR_origin = A·q_o/σ_bg, SR_context.

**Fixed per-layer values** (`DESIGN_SHARD_B` pf1 and `DESIGN_SHARD_A` agree):

| layer | b at r=20 | t95 at r=20 (d) |
|---|---|---|
| confined_T50 | 1e-5 | 10.7275 |
| confined_T500 | 1e-6 | 8.8919 |
| leaky_T50 | 1e-4 | 13.2180 |
| leaky_T500 | 1e-5 | 10.7275 |
| unconfined_T50 | 1e-2 | 22.6575 |
| unconfined_T500 | 1e-3 | 16.8322 |

**Forbidden code paths:**
- `p3_phase2_cases.layer` and `p4_cases.physical` for r = 20. Their cache is keyed by sid only.
- `p4_cases.build_B`. It asserts the r = 200 t95.
- Passing A_distance20 through the unchanged `p4_export.validate`. It accepts groups A/B only and treats A as the artificial last-context-origin design.

**Generation gate (`generation_checks.all_ok`). All must hold:**
- 2,160 unique ids.
- 918 cases with zero q_origin, and E ≡ 0 for each.
- 216 water-curtain variant cases.
- Parent byte equality for the copied fields.
- head_context differs from the parent.
- Per-case b and gain equal the r = 20 formula, which catches stale-cache reuse.
- Pair identity holds to < 1e-12.
- Truth lies inside the new declared family (§1.3).
- No truth key appears in tf_inputs or the tool archives.
- The protocol stamp equals the final phase5 digest.

### 1.3 Declared family for new cases only

For new cases the family is the original family **with the lower bound log10b = −6** (b ≥ 1e-6). Every other bound, budget, algorithm, starts set and seed is unchanged.

Rationale, to be recorded in the protocol:
- Three r = 20 shapes lie outside the original box: confined_T50 and leaky_T500 at b = 1e-5, confined_T500 at b = 1e-6.
- leaky_T50 sits on the old face.
- The best in-box unit response misfits the true shape by 1.5–2% at the face.
- 1e-6 is the original λ_max = 10⁴ m scaled by (r/200)². It is the smallest closed widening that contains all six shapes.
- Widening only, so the old box is a subset of the new one.

The rule applies to every W stage (global Sobol, local box, best_fit, refinement and direct-certification clipping) and to the structural reference `well_b` pmin = 1e-6.

Mechanism (any implementation is allowed; acceptance is behavioral):
1. With the bound set to −4, a fixture run equals the frozen engine and frozen reference **byte-for-byte**, runtime fields excluded.
2. With the bound set to −6, every box reader reports the new bound.
3. No process that handles an old-family case can see the new bound. Prefer isolated phase5 module copies with an AST/source diff allowlist over shared-global patching.
4. Every W and reference JSON stores `family_version=D06_new_r20_b1e-6` and its bounds.

Old cases keep `family_version=D03_D05_original_b1e-4`.

**Not adopted:** a W rerun of r = 20 cases under the old box.

### 1.4 Execution

**Shards.** `actor = p4_partition.actor_of(parent_case_id)`.
- shard_b and shard_a get 1,080 cases each, disjoint and exhaustive.
- Each actor has 5 of the 10 realizations in every (form, month, layer) cell.
- Each actor has 459 zero-origin cases and 108 variant cases.

**W, reference and truth evaluation.** All D05 settings are inherited:
- W: seed 20260930, Sobol 128/256/512/1024, local 512 × 12 rounds, 5% / ≥200 accepted, refinement at leads {10, k\*}, 4 restarts.
- Reference: 12 starts, 1000 draws, max_iter 50, calendar `seed_pastas_param_sample`.
- Truth is opened only after the W and reference records are written.
- All 918 zero-origin cases are run.
- Resume uses the D05 marker rules.

**CPU.** 8 single-thread processes per actor, BLAS/OMP = 1. Check resources before launch. Contention changes scheduling, never budgets.

**Model.** The frozen checkpoint is used offline:
- revision 43046b85…129c2, weights a7592b0a…7f98f17da, config ff17bbc0…bc6;
- the D05 API flags and raw Q/rain covariates; H10 and H30 are independent; batch 8, chunk 128;
- one lifetime lock, `results/phase5/tools/MODEL_WAVE.lock`, with sequential actor waves and no concurrent legacy MPS wave.

### 1.5 A analyses (no D05 refit)

**Q1: which layers and months become answerable.**
- Twins are matched by `parent_case_id`.
- For each form × month × layer and each storage cell, at leads 10 and 30 and for both pairs, count:
  - became_determined, lost_determination, both_determined, neither;
  - missing;
  - the matching magnitude (W/|E| < 1) transitions.
- The main denominator is **active origins only**, following RAW §3.4. All-scheduled counts and no-question (zero-origin) counts are kept in companion tables.

**Q2: does the D05 threshold map predict the new cases?**
- Predictor: η = b0 + bS·log SR_origin + bR·log(L/t95 + 0.05), using the frozen `statistics.json` `cohort_A.models` coefficients.
- Sign: use `no_layer` only. The layer sign fits are `quasi_separation` at leads 10 and 30 and have no coefficients.
- Magnitude: use both `no_layer` and `layer`, whose status is ok.
- Report against actual outcomes:
  - p̂ ≥ 0.5 and p̂ ≥ 0.9 classification tables;
  - Brier score;
  - mean(observed − p̂);
  - the magnitude residual and the agreement on W/|E| < 1.
- Flag every case against the reconstructed 1,260-case support hull from D05. Out-of-hull cases are flagged, never clipped or refitted.
- Paired reference: the same projection on the 2,160 D05 B r = 200 cases.
- Label: operating-calendar projection, because the SR and ρ definitions differ from the map's.

**Contradictions (D05 C rules).**
- Among sign-determined rows, the primary count is strictly opposite nonzero sign.
- Tool zero, tool missing and undetermined-truth errors are counted separately.
- The structural reference is shown on the same rows.
- Use the frozen D05 five log-SR bin edges, plus an explicit `above_D05_range` stratum.
- Also report by month and form.

## 2. B: timescale reanalysis (existing results, no new execution)

**Universe.** `B_CASES.json`:
- D03: 540.
- D04: A 240 + B 360 + C 90.
- D05: A 720.
- Total **1,950** (source_phase, case_id) records and **7,800** rows (raw, leads 10/30, P1/P2), from `results/phase4/primary_rows.csv`.
- Joined to `derived_manifest.json` fields SR, pi_r, R_days and t95_d.
- `D04C_SUBSET.json` holds 90 cases and 360 rows.
- Calendars, filter tracks and max-day summaries are excluded.

**Covariates.** k is the lead. Define:
- z0 = 1[R = 0].
- Lρ = log(R/t95) if R > 0, else 0.
- L_Rk = log(R/k) if R > 0, else 0.
- Lk = log(k/t95).

R and t95 are the actual `R_days` and `t95_d` values; ρ_realized = R/t95. The z0 column is a separate intercept, so the protocol does not assert log 0 = 0. No 0.05 offset is used in B.

**Bundles.**
- (i) [1, log SR, log π_r, z0, Lρ]
- (ii) (i) + Lk
- (iii) [1, log SR, log π_r, z0, L_Rk]

Drop z0 only where it is identically zero (D04 C, positive-rest sensitivity), and record a schema note when it is dropped.

**Outcomes.**
- y_mag = log(W/|E_true|), using finite W > 0 and finite E ≠ 0.
- y_sign = 1[inf > 0 or sup < 0]. An endpoint at 0 counts as undetermined.
- For a given outcome, every bundle uses the same row set. Log exclusions and missing data have separate disposition codes.

**Fits.**
- Per pair, both lead-pooled (10 + 30, primary for the timescale question) and per lead (10, 30).
- OLS: weighted rank-revealing SVD on the standardized design with relative cutoff 1e-10. Under aliasing, report fitted values, SSE and R² with `coefficient_nonunique` plus the null basis.
- Logistic: the frozen `p4_statistics` separation LP and unpenalized IRLS (max 50 iterations, tol 1e-8, clip 1e-12). Separation or numerical failure produces a null MLE with its status. No ridge or Firth fallback.

**Reported quantities.**
- R², SSE, log-likelihood, deviance and McFadden pseudo-R².
- Paired deltas (ii−i), (iii−i), (ii−iii) on all 999 draws, with 2.5/97.5 percentile intervals over identified replicates and n_unidentified with reasons.
- Positive-rest sensitivity: 1,770 cases with plain logs and the same draws.

**D04 C, settled before any fit.**
- With b = 0.01 fixed, S(t; a, b) = F(t/a; b), so log t95 = log π_r + const. The input check gives slope 1.000000 and residual 1.2e-14.
- At a fixed lead, (ii) is rank-deficient (rank 4 of 5). Its fitted values equal those of (i), so **ΔR² = 0 by construction** (status `fixed_lead_alias`).
- (iii) spans the same space at a fixed lead.
- Lead-pooled (ii) adds only a lead contrast. Report it under that label.
- This negative answer to "does (ii) recover the 30-day collapse?" is a result.

**Baseline gate.** Before any B number is published, refit D04 C at a fixed lead with the D04 convention (log SR, log ρ_nominal, log π_r, plain OLS). It must reproduce `results/phase3/READOUT.json` C `descriptive_collapse`:
- R² = 0.597678432661343 at 10 d;
- R² = 0.08218368104823925 at 30 d;
- n = 90 per pair, to ≤ 1e-9.

That historical fit used **nominal ρ** (the cell_metrics `rho` column, built from `rho_nominal`). The main bundles use realized R/t95, which at D04 C S = 1e-4 is 0.44/0.88/3.97 against nominal 0.25/1/4. The report therefore shows the D04 C fixed-lead (i) R² under both definitions, so that a definitional change is never read as recovery.

## 3. C: linearized information width

### 3.1 Universe

**`C_CASES.json`** contains **6,270** cases:
- D03: 540.
- D04: 690.
- D05: A 720 and B 2,160.
- D06 A: 2,160.

**Rows.**
- 25,080 primary rows per required mode (2 pairs × leads 10/30).
- 376,200 full-curve rows per required mode (2 pairs × leads 1–30).
- 1,836 zero-contrast cases (q_o = 0, D05 B 918 + D06 A 918). They are kept with E ≡ 0, interval [0, 0], W_lin = 0, status `no_question`, and ratio and log undefined.

**Per-case join.** Each case is joined by (source_phase, case_id) to its own tf_input, truth, W JSON tolerance and family version:
- old cases use b ≥ 1e-4;
- D06 cases use b ≥ 1e-6.

**Truth use.** Truth is read for this oracle diagnostic as requested. It never enters W, reference or model runs.

### 3.2 Model and Jacobians at truth v0

The model is

μ_j = c0 + β_nat[(P*K)_j + P̄(1−Kc(j+1))] + η_nat·e^{−(j+1)/τ} − A[(Q*B)_j + Q_start(1−S(j+1))] − η_pump(1−S(j+1)),

with E_p(k) = −A(dQ_p*B)_{k−1}.

The truth point v0:
- (n, θ, τ, a, b) from `theta_true_json`;
- A = gain, β_nat = nat_gain;
- η_pump = A(Q_pre − Q_start);
- η_nat = 0. In raw-head coordinates c0 = 0; in the existing runtime y=h_obs−mean(h_obs), use c0 = −mean(h_obs). These conventions give the identical residual and J because c0 is free. Save the centering convention, and apply it consistently to e, μ and v0.

S_eff, T, S, c and r are not family parameters.

J_h is 1024×10 and J_E,p is 30×10. Only the A, log10a and log10b columns of J_E are nonzero, and J_E,P2 = −0.5 J_E,P1.

**Derivatives.**
- Linear columns: exact source columns.
- Existing runtime J uses central source-family differences for the five nonlinear columns, with half-step checks, and exact linear columns. Keep that source-discretized route. Independent continuous Hantush checks use analytic derivatives in ln a and ln b (×ln10), with S(t;a,b) = F(t/a;b):
  - ∂S/∂ln a = −e^{−t/a−ab/t}/(2K0(2√b))
  - ∂S/∂ln b = −(2K0)⁻¹∫_{ab/t}^∞e^{−v−b/v}dv + √b·K1/K0(2√b)·S(t)
  - Equivalently, impulse factors (t/a − ab/t) and (−ab/t + √b·K1/K0).
- Natural columns: central difference with step 1e-4 in each parameter's native coordinate, plus a half-step check.
- Kernels may be evaluated outside a box only as derivative probes.

**Validation** (`DESIGN_SHARD_B` pf3b reaches ≤ 1.6e-5; `DESIGN_SHARD_A` reaches 7.5e-5):
- Source-discretized Hantush columns: selected ≤ 1e-3 relative, with continuous-exact FD checks separately reported. The stricter prior draft ≤1e-4 is not a D06 completion condition; see §3.7.
- Source-family central/half-step and analytic-source diagnostic errors are recorded separately; the current source-relative criterion is ≤1e-3. The old draft ≤2e-2 diagnostic does not relax this criterion.
- Directional h derivative: ≤ 1e-3 relative.

### 3.3 Genuine AR metric and selected deterministic increment transfer

The authoritative selection is `C_METRIC_DECISION.json`, after the six actual HIGH turns in `D06_C_metric_debate.json`. It supersedes the old C-only exact-SSE Fisher-coordinate selection. A and B retain their contracts above, with the source corrections in `PROJECT_DESIGN_DECISION_v2.json`. The actual implemented modules and terminal component receipts are pinned by this pre-score freeze; no official outcomes are asserted.

Let J = J_h(v0), g = J_E(k), E0 = E(v0), e = h_obs − μ(v0), G = JᵀJ, H = JᵀΣ⁻¹J. Keep all ten family columns, including the natural component and both initial-condition terms. A zero entry of g does not justify deleting that parameter's column from J. β_nat at truth is `nat_gain`, without another storage/gain multiplier. μ, its rainfall mean, pre-context pumping and natural state follow the frozen source family.

**Exact AR whitening.** For values X (vector or column matrix) on original sorted times t:

    (L X)_0 = X_0 / σ
    (L X)_i = (X_i − a_i X_{i−1}) / [σ sqrt(1−a_i²)]
    a_i = φ^(t_i−t_{i−1}),  σ = 0.02 m, φ = 0.8.

Thus LᵀL = Σ_keep⁻¹. In the full daily record a_i=φ. Retain the stationary first-row variance, not innovation variance. Masking uses the covariance marginal on kept times, never the corresponding principal block of full-record precision.

**Scaled estimable rank.** Compute Jw=LJ. Let s_j = 1/||Jw[:,j]||₂ for norms >1e−14, otherwise 1, and S=diag(s_j). Use SVD Jw S = U_w diag(ζ) Vᵀ, keeping ζ_i>1e−10 ζ_max. For rank r:

    C = S V_r diag(1/ζ_r),       Cᵀ H C = I_r
    N = S V_null,                J N = 0
    D = (J C)ᵀ(J C),            λ = λ_max((D+Dᵀ)/2).

No ridge, raw unscaled Gram eigencut, or nuisance-column pruning. Record singular values, scales, retained rank, null basis and whitening orthogonality error. Construct the Euclidean tangent basis U by thin QR of J C, using the same retained rank, so P=UUᵀ is the Euclidean projector onto the retained column space. Algebraically P=JJ⁺; it is not the GLS projector. Report rank disagreements with independently scaled SSE diagnostics rather than silently changing the transfer basis. Rank r is diagnostic; the declared tolerance parameter count remains p=10.

For λ>0, define the **saved SSE increment** Δ_L = τ_saved − SSE_min,saved and the dimensionless AR increment c_L = Δ_L/λ. These are the original case's W tolerance fields, not a new fitted or sampled width. Negative or nonfinite Δ_L, nonpositive λ, rank zero, invalid source bounds or undefined moments retain explicit failure/status receipts. Do not silently clip a negative budget, add ridge or substitute a trace radius. Δ_L=0 is a degenerate support problem that still needs its own certificate.

**One transfer convention, three required full-cohort modes.** All use the same complete bound set B on δ=v−v0: original/new shape box, A≥0, A≤10·ptp(h_obs)/median(Q_context>0), β_nat≥0, |η_pump| and |η_nat|≤10·ptp(h_obs), c0 free, and ∇t95(v0)ᵀδ≤1000−t95(v0). Translate absolute bounds into δ bounds without changing them. Head-derived bounds are retained for every mode and frozen for masks.

| canonical mode | region intersected with B | purpose |
|---|---|---|
| `sse_locked` | ||e−Jδ||² ≤ τ_saved | faithful legacy SSE benchmark |
| `ar_locked` | ||L(e−Jδ)||² ≤ Q_min,B + c_L | main genuine AR width, residual recentered |
| `ar_prospective` | ||LJδ||² ≤ c_P | true-centered conditional prediction without W sampling |

Here Q_min,B = min_{δ∈B} ||L(e−Jδ)||² is a **bounded affine GLS** minimum certified on the same bounds and memory face. The locked SSE benchmark continues to use τ_saved directly; do not replace it with SSE_min,affine+Δ_L. Retain the affine OLS and GLS minima and their fitted δ vectors to expose recentering. These local affine optima are numerical feasibility centers, not replacement scientific family fits.

**Containment and scope of the proof.** For a common center δ_c, write δ−δ_c=Cz+Nu. Then (δ−δ_c)ᵀG(δ−δ_c)=zᵀDz≤λ||z||². Consequently {quadratic AR increment≤Δ/λ} is contained in {quadratic SSE increment≤Δ}, even after intersecting both with identical bounds. The top eigenvector attains equality, so this is the largest universal scalar AR ball for that common-center quadratic containment. It does not reproduce the directional original SSE region. It also does not prove containment of the residual-recentered `ar_locked` region in the legacy `sse_locked` region. OLS and GLS centers differ; saved nonlinear and affine minima differ. If the bounded GLS optimum has an active face, Q(δ)−Q_min includes a nonzero linear gradient term, so do not identify it with a pure centered H quadratic. A narrower common-center AR width is a consequence of the transfer convention, not evidence of extra information.

**Strongest retained dissent.** executor's mean-SSE trace transfer c_trace=r Δ/tr(D) is scientifically defensible: it matches mean SSE over an isotropic Fisher ball. It can be larger than c_L because it preserves an average rather than the worst direction. Neither scalar exactly equals the old directional SSE region. Retain this dissent in the interpretation; extra full-cohort trace-width fits, chi-square comparisons and `prospective_local` SSE production rows are optional and not completion requirements.

### 3.4 Expected source tolerance without saved W

The prospective radius uses only Q, rainfall, oracle v0, Σ and the actual B. B already depends on observed-head range. The conditional no-W-sampling calculation does not claim unconditional prediction from a pumping log alone. The functional must not read saved W τ, fit, r1, n_eff, endpoints or widths, or observed residual e. Separate later joins with sampled W from radius construction.

For the retained Euclidean projector P=UUᵀ:

    R = (I−P) Σ (I−P)
    r_pred = sum_{i=0}^{n−2} R[i,i+1]
             / sqrt[(tr(R)−R[n−1,n−1]) (tr(R)−R[0,0])]
    n_eff,pred = clip[n(1−r_pred)/(1+r_pred), 20, n]
    γ_pred = 10 F_0.95(10, n_eff,pred−10) / (n_eff,pred−10)
    Δ_P = γ_pred tr(R),           c_P = Δ_P / λ.

This is a ratio of expected residual covariance terms. It is not E[sample Pearson correlation] and does not assert that a bounded nonlinear fit has this exact residual law. r_pred comes from the declared covariance and tangent space, not fitted residuals. The p=10 convention and inherited clip are unchanged. Zero residual trace or denominator is `undefined_residual_moments`, not an invented epsilon radius. Validate |r_pred|≤1 before allowing a ≤1e−10 rounding adjustment; nonfinite/out-of-range values are a numerical failure. At r_pred=−1 the limiting n_eff is n.

**Efficient O(nr²) time and O(nr) storage, without a dense n×n matrix.** For each row u_i of U, evaluate the forward and backward recurrences:

    f_i = u_i + φ f_{i−1},       f_{−1}=0
    b_i = u_i + φ b_{i+1},       b_n=0
    s_i = σ²(f_i+b_i−u_i),      S_cov = ΣU
    M = UᵀS_cov,                M ← (M+Mᵀ)/2.

Then

    R_ii = σ² − 2 u_i·s_i + u_iᵀ M u_i
    R_i,i+1 = σ²φ − u_i·s_{i+1} − s_i·u_{i+1} + u_iᵀ M u_{i+1}
    tr(R) = nσ² − tr(M).

Use diagonal sums as a consistency check against the trace formula. Tiny dense comparisons test rank 0/4/9/10, φ=0/0.8/−0.4, and an aliased, differently scaled ten-column J of rank 8. The design-author fixtures are embedded with executable reference code in `C_AR_ADDENDUM_AUDIT.json`. The port and dense-fixture comparison have been validated in the pinned science environment by the terminal AR delta receipt. No dense 1024×1024 matrix is needed per case.

### 3.5 Certified support, diagnostics and comparisons

For each required mode compute the lower and upper support of E0+gδ over its complete region. Reuse the existing generic affine support solver with `(J,e,T)` for SSE, `(Jw,L e,Q_min,B+c_L)` for locked AR, and `(Jw,zero,c_P)` for prospective AR. A prepared minimum receipt must belong to the identical design matrix, residual, bounds and memory face; never pass the SSE minimum into the AR problem.

Use column-scaled rank-revealing least squares and convex constrained support. The current exact-gradient SLSQP plus numerical quadratic dual is a suitable implementation route. A solver Boolean is not certification. Save primal violation, nonnegative multipliers, active bound labels, stationarity/KKT residual, complementarity and lower dual bound/gap for both ends and the affine minimum. Current numerical tolerances are relative rank 1e−10, primal/dual 2e−7 scaled by 1+|T| or 1+|endpoint objective| as appropriate, and stationarity/complementarity 2e−6 in solver-normalized coordinates. These are numerical certificates, not interval arithmetic proofs. Record actual units and scaling in receipts. The GLS minimum receipt encloses Q_min between its valid dual lower bound and feasible primal value. Save this minimum gap and the resulting threshold interval [Q_dual+c_L,Q_primal+c_L]; if it affects endpoint/sign accuracy beyond the reported certificate tolerance, retain numerical uncertainty. A support certificate at an approximate primal threshold alone does not prove an exact-minimum threshold. Flag a negative dual gap beyond rounding as invalid rather than forcing it to zero. Support can still succeed with active bounds.

When g annihilates the nullspace and both unconstrained ends satisfy every bound, the interior AR interval is

    E0+g δ_hat ± sqrt(c_L g H_est^− gᵀ),      H_est^− = C Cᵀ,

with δ_hat the unconstrained GLS minimum. This inverse is the inverse on the scaled estimable subspace. Prospective uses center zero and c_P. SSE uses its own affine OLS center and remaining radius τ_saved−SSE_min,affine. Apply the formulas only after feasibility and all bound checks. If g does not annihilate the nullspace, retain actual finite faces; use a recession LP for an unbounded null witness, or report `bounded_by_family_only` with certified finite supports. No automatic infinity from rank deficiency alone.

Keep both scientific and numerical status fields: `no_question`, `zero_contrast`, `certified_interior`, `certified_constrained`, `bounded_by_family_only`, `unbounded`, `linearization_incompatible`, `feasibility_uncertified`, `support_uncertified`, `undefined_residual_moments`, `invalid_transfer_budget`, `zero_information`, `invalid_source_bounds`, `derivative_invalid`, `missing_input` as applicable. A certified minimum above a locked threshold gives an empty linearized region (`linearization_incompatible`); an uncertified minimum cannot establish emptiness. An A≥0 face alone does not establish linearization failure. Keep `linearization_failure` as a separate flag for a source-inconsistent affine sign cone (P1 positive or P2 negative) or independently demonstrated approximation failure, without clipping the endpoints.

For q_origin=0, all 1,836 cases retain the scientific identity E=0, interval [0,0], W=0 and `no_question`; any feasibility diagnostic is kept separately. No-question is distinct from a numerical failure. Other exactly zero contrasts on a nonempty region have zero width. Negative budgets and zero-information transfers keep their input/status diagnostics even when the scientific question is absent.

Strict sign is positive only if the certified lower envelope is >0, negative only if the certified upper envelope is <0, otherwise undetermined. Expand nominal endpoints by dual gaps and a reported floating-point allowance before claiming determination. Store roundoff-sensitive nominal signs as `endpoint_numerical_uncertainty`, not a positive decision. Endpoint zero is undetermined. Magnitude answerability uses W/|E_true|<1 on defined rows.

**Matched-white-noise fixture.** At φ=0, H=G/σ² and λ=σ². Equality is between AR threshold `Q_min,B+Δ/σ²` and SSE threshold `SSE_min,affine,B+Δ`, with identical B, residual e and the matched affine minima. Equality with legacy τ_saved requires its saved minimum to equal that affine minimum. This author tested the identity with A=0 active: endpoint discrepancy 5.72e−14. Do not demand equality against unmatched saved nonlinear minima.

**Full-cohort reporting.** For each of `ar_locked`, `ar_prospective`, `sse_locked`, use x=log sampled W and y=log W_lin: slope, intercept, R², layer residuals, and strata by lead, pair, source, distance and family. Report all 6,270 cases, 376,200 curve rows per mode and 25,080 primary rows per mode (totals 1,128,600 and 75,240 for three modes). Retain complete status counts, nonfinite/log exclusions, active-origin denominators and sign confusion/direction agreement; include magnitude agreement and the shared realization bootstrap. Paired comparison modes use a common eligible intersection with its count, alongside each mode's available-row fit. Zero/empty/unbounded/uncertified rows remain in the inventory. Sampled W is an inner envelope. Metric-convention differences and recentering differences are reported separately; no positive R² or agreement threshold is a success gate. Poor regions are an answer to RAW C.

### 3.6 Prespecified mask information experiment

The 18 representatives are the six designed cases already selected, plus r00 confined_T50/unconfined_T50 calendar cases for water_curtain m01, paddy_irrigation m07 and domestic_continuous m12 at both 200 m (D05) and 20 m (D06). The task packet authorizes the input-only water m12→m01 and paddy m06→m07 replacements in `PROJECT_DESIGN_DECISION_v2.json`; no additional root-confirmation gate is pending. The original water m12 has q_origin=0; original paddy m06 has an unobserved season-window end for the slow 200 m layer. Selection uses inputs, not outcomes.

Define and save half-open head-row masks and dates before width analysis:
- Designed pause: declared rest [end−R,end).
- Calendar pause: latest fully observed maximal zero run ≥ceil(t95), with observed positive resumption.
- Designed transition: first controlled event through final ON boundary plus one day.
- Domestic transition: latest maintenance stop and observed resumption.
- Seasonal transition: latest interior one-day edge within the active season.
- Seasonal start: latest Nov 1 (water) or May 1 (paddy) in the context, ±ceil(t95). If that latest window is not wholly observed, retain `window_absent`; do not silently substitute an older year. Domestic season start is `not_applicable`.

Delete only the selected head rows. Preserve the complete Q/rain forcings, rainfall mean, Q_pre/Q_start, natural and pumping states, full-record derivative convention, oracle center δ=0 and complete B. Compute J_keep by row deletion; compute H_keep from exact gap-marginal whitening. Freeze the **full-record** Δ_L, λ and c_L for every mask of that case. No masked recentering, re-estimated minimum, r1, expected moments, tolerance, scale, radius or head bounds.

Compare two separate true-centered information baselines:

    AR full:    B ∩ {δᵀH_fullδ≤c_L}
    AR masked:  B ∩ {δᵀH_keepδ≤c_L}
    SSE full:   B ∩ {δᵀG_fullδ≤Δ_L}
    SSE masked: B ∩ {δᵀG_keepδ≤Δ_L}.

These are distinct from residual-centered main widths and from prospective c_P. Exact marginal information satisfies H_keep≼H_full and G_keep≼G_full, so each masked set contains its corresponding full set. Certified W_mask/W_full≥1 and width difference≥0 are numerical-consistency checks for identical center/B/c, with certificate tolerances; violations are audit flags, not clipped values. If W_full=0 the ratio is undefined. A prospective-mask sensitivity is optional only with its full c_P held fixed; it is not required. If δ=0 is outside B, retain the baseline feasibility diagnostic rather than resetting the family. Missing/absent/nonapplicable windows keep explicit rows.

Outputs carry `mask_metric` (`ar_information`/`sse_information`), `center_rule=true`, `budget_source=full_record_saved_increment`, fixed λ/Δ_L/c_L, full/keep time hashes, same bound-set digest, indices/dates/status, per-pair per-lead full and masked supports, widths, ratio, difference and certificate gap. Figure 1 uses these mask results, not an unmatched comparison against main residual widths.

### 3.7 Derivative validation and bounded implementation delta

The source-family derivative criterion is relative error ≤1e−3, selected in DESIGN_SHARD_A and this task, plus the existing directional/half-step checks. `implementation_C/QUADRATURE_FIXTURE.json` reports a continuous-analytic versus frozen daily-quadrature source discrepancy 0.0003554718199581628, which meets that criterion. It does not meet the earlier draft's optional 1e−4 criterion. Preserve both facts; do not change the frozen daily family to make the analytic derivative look exact, or declare a D06 failure on the stricter unadopted threshold. Revalidate the source derivative receipt in the pinned runtime if the Jacobian implementation changes. Report continuous-exact quadrature tests separately from derivatives of the discretized declared family, including near-bound one-sided/source-probe treatment and natural half-step error.

The MEDIUM developer delta is terminal. Actual source hashes and receipts are in `implementation_C/AR_DELTA_COMPONENT_RECEIPT.json` and `implementation_A_B/COMPONENT_RECEIPT.json`. The production science environment is CPython 3.11.14, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6 and Pastas 2.0.0. Existing source/behavior evidence is reused because exact current component/evidence hashes match. A/B has 13 passing behavior fixtures; C has 16 original plus 15 new AR tests, 180 certified default synthetic rows, 360 dual-mask rows, and an explicit 120-row legacy endpoint comparison with maximum difference 0.0. These are implementation fixtures, not cohort outcomes. Derivative source error 0.0003554718199581628 passes the selected 1e-3. No optional trace/chi-square/SSE-local full-cohort run becomes compulsory.

The final interface authority is the actual pinned implementation with the mathematical contracts above, the terminal component interfaces, and `FROZEN_EXECUTION_CONTRACT.json`. Draft interfaces are retained as scientific lineage; operational aliases below override only stale paths/labels/flags, never equations or bounds. The only C production manifest is `results/phase5/C_CASES.json`, written by `p5_collect --mode production_A` after A generation and complete W/model collection. No `C/C_CASES.json` copy is created. A/B drivers use `--mode production`; collector uses `--mode production_A`; C uses `--production` or `--merge`.


## 4. Frozen source, preservation and operational binding

The complete pre-score bundle is `protocol.md`, `protocol.sha256`, `protocol_freeze.json` (status exactly `complete_protocol_frozen`), `OLD_HASHES.json`, `FROZEN_EXECUTION_CONTRACT.json` and `PRODUCTION_COMMAND_RECIPE.json/.md`. `FINAL_FREEZE_AUDIT.json` records actual dry-gate evidence and the final bundle digests. Production starts only after both actual A/B and C gates accept the bundle.

`OLD_HASHES.json` covers existing files under `results/{pilot,phase2,phase3,phase4}`, `direction/`, `lib/p3_*.py` and `lib/p4_*.py`, and preserves the broader inherited D05 old snapshot. Existing manifest/source-set/output hashes are reused. Only uncovered files are added from real bytes. Every OLD_HASHES entry is also a direct `protocol_freeze.hashes` pin because the actual A/B gate does not read OLD_HASHES itself. All stable 107 source pins, the real final C sources, complete AST local-import closure including dynamically loaded frozen family sources, actual climate/rain/calendar inputs, old W/truehead source files, bootstrap file and logical matrix, current editable model source/config/weights revision, environment configurations/interpreter and package metadata are bound.

The actual gates verify the protocol and sidecar, exact complete status, all SHA pins, import audit and the inherited D05 gate. C additionally requires the final execution contract and its implemented source pins. Package versions and runtime fixture provenance are recorded in the execution contract; package METADATA/RECORD and pyvenv.cfg are pinned. This is package provenance, not a claim of hashing every installed third-party binary. Model weights are read locally/offline and never redistributed.

Self-hash cycles are avoided with the existing gate convention: protocol digest binds the freeze; the freeze directly hashes the protocol, sidecar, OLD_HASHES, execution contract and recipes. The execution contract binds the protocol and actual code, while its own digest is enforced by the freeze. Neither the freeze nor audit embeds a fake self-digest. Audit is written after gate verification, carries the freeze digest, and is outside the freeze pin set. Post-freeze generated cases, model outputs, W and the authoritative C_CASES manifest receive source/protocol generation receipts and are not invented as pre-score pins.

Commands in `PRODUCTION_COMMAND_RECIPE.json/.md` use actual parser flags and interpreters. CPU cap is 8 single-thread W processes per actor, 16 total. Other CPU calculations share that total cap. Model actors run sequentially under the lifetime phase5 lock and existing read-only legacy lock, with batch 8/chunk 128 and unchanged D05 API flags. Model environment uses `HF_HOME=ROOT/env/cache/hf`, `HF_HUB_CACHE=ROOT/env/cache/hf/hub`, `TMPDIR=ROOT/results/phase5/runtime_tmp`, offline hub/transformers settings and thread variables 1. No activate_paths.sh is sourced. Model cache blobs and editable environment sources under Paper 1 remain read-only.

The exact expected actor identities/digests are in the execution contract: each actor 1080 cases, 459 no-question cases, 108 water variants, five realizations per form/month/layer cell. Generated SHARD_SHARD_B/SHARD_SHARD_A files must have those identities. Total A queries are 17280, 8640 per actor; horizons H10/H30 are independent. Full B has 1950 cases/7800 primary rows. Full C has 6270 cases/1836 no-question cases, three required modes and 18 approved mask representatives. Poor scientific agreement is an outcome; unresolved numerical computation remains an explicit result/status requiring reporting rather than being dropped.

## 5. Deliverable mapping (RAW §2–4)

| RAW item | protocol section | output path |
|---|---|---|
| A cases and execution | §1.2–1.4 | `results/phase5/cases/`, `wb/`, `tools/{shard_b,shard_a}/` |
| A answerability and D05 prediction | §1.5 | `results/phase5/A_rows.csv`, `tables/A_distance_summary.json`; later `A/{distance_calendar,map_prediction_check,contradiction*}.csv`, `DISTANCE_SOURCE_AUDIT.json` |
| B | §2 | `results/phase5/B/{B_CASES.json, D04C_SUBSET.json, B_timescale_rows.csv, B_fits.json, B_comparisons.csv}` |
| C | §3.1–3.7 | `results/phase5/C_CASES.json; C/{C_linearized_rows.csv, C_loglog.json, C_sign_agreement.csv, C_window_masks.json, C_masking_rows.csv}`, `ANALYSIS_SOURCE_AUDIT.json` |
| REPORT first paragraph | the four RAW §4 items | `results/phase5/REPORT.md`, `figures/` |
| draft v1 (after A–C), RAW §3 rules | later HIGH writer | `manuscript/{draft.md, supplementary.md, numbers_sources.md, figures/}` |
| reader note | later independent HIGH reader | `manuscript/READER_NOTE.md` |
| final completion audit | root | `results/phase5/FINAL_SOURCE_AUDIT.json` |

RAW sections 1 and 3 are passed verbatim to the analysis and writer roles.


## 6. Author source passed verbatim (RAW §1 and §3)

## 1. 저자 판단: 결과를 묶는 원리

D03–D05를 한 번에 놓고 보면, 결과를 차례로 늘어놓은 초고는 백화점식이 된다. 결과들은 하나의 원리로 묶인다.

> **운영 양수 기록은, 그 안의 양수 변화가 관측 지점의 배경 변동보다 크고(신호비), 질문이 묻는 시간척도만큼 길 때 what-if에 답한다.**

시스템 식별의 지속 가진(persistent excitation) 개념을 운영 기록과 what-if 질문에 옮긴 것이다. 각 결과는 이 원리의 한 면이다.

- **모호함의 크기.** 기록이 남기는 모호함 W(m)는 배경 변동과 응답 모양이 정하고, 양수 규모나 이득에는 거의 무관하다. D03에서 같은 응답 모양에 이득만 10배 다른 두 층의 W 비는 0.995였다. D04·D05에서 배율 0.3·1·3의 W 중앙은 0.090, 0.089, 0.089 m였다. 그래서 답할 수 있음 |E|/W는 신호비에 거의 비례한다.
- **시간척도.** 짧은 전환은 처음 며칠의 모양을 알려 주고, 이때 중요한 것은 횟수이며 최근성은 거의 상관이 없다(D04 A). 응답시간에 견줄 만한 휴지는 10–30일 효과를 알려 준다(D03). 저장계수가 크면 응답이 느려서 같은 10일이 응답의 앞부분에 불과하다. 그래서 W(m)는 거의 같아도 상대 모호함이 커진다(D04 C: 0.19 대 0.46).
- **교환 비율.** 방향 결정 확률 0.5의 신호비는 휴지가 없을 때 0.23, 응답시간 네 배의 휴지가 있을 때 0.12였다(D05 A). 신호비가 주 조절 변수이고, 긴 휴지는 필요한 신호를 약 절반으로 낮춘다. 휴지가 없는 기록도 신호가 크면 방향에 답한다. 따라서 "휴지가 핵심"이라는 이전 가제는 쓰지 않는다.
- **대수층 유형과 관측 거리는 신호비를 바꾸는 경로다.** D05 달력에서 비피압 층이 거의 답하지 못한 것은 대수층 유형 때문이라기보다 r = 200 m에서 신호비가 0.07–0.11이었기 때문이다. 저자가 Hantush 이득 K0(r/√(Tc))로 계산하면, r을 20 m로 줄일 때 이득이 피압·누수 층에서 1.6–2.3배, 비피압 T50에서 15배, T500에서 3.9배 커진다. r = 200 m는 π_r을 1 양쪽에 두려고 정한 값이었고, 국내 문헌의 관측 거리는 5–26 m였다. 아래 A로 확인한다.
- **운영 달력은 자연 실험이다.** 실제 운영에는 계획하지 않은 가진이 있다. 수막재배의 시즌 시작, 비 오는 날의 관개 정지, 생활용 관정의 점검 정지가 그렇다. 달력 결과는 신호비를 다시 말하는 것이 아니라, 이런 가진이 어느 달에 what-if에 답할 정보를 만드는지를 보여야 한다.
- **시나리오 엔진은 기록에 있는 정보를 꺼내 쓰지 못한다.** 기록이 방향을 정한 8,314행에서 엔진은 758행(9.1%)에서 반대 부호를 냈고, 같은 기록에 맞춘 구조 정합 전달함수는 0행이었다. 엔진의 효과 비율은 휴지가 길어져도 나아지지 않았다(D03). 휴지는 수리지질학자가 가장 믿는 가진이다. 증량 질문에서는 신호가 큰 피압·누수 기록에서도 7–9%가 반대 부호였다. 도구 절은 실패 목록이 아니라 이 한 메시지로 쓴다. 전환 횟수에 대한 엔진 반응은 D03(작은 개선)과 D04(혼합)가 어긋나므로 "일관된 반응 없음"으로 쓴다. 구조 정합 기준은 경쟁자가 아니며, 기록에 정보가 있었음을 보이는 대조로만 쓴다. 정규화 기제는 두 쌍에서 부호가 반대였으므로 토론의 한 문장 가설로만 둔다.

**지표의 사다리**를 초고 방법 절에 이 순서로 정의한다. W(m)는 기록이 남긴 모호함이다. |E_true|는 질문의 신호다. |E_true|/W와 부호 결정 여부가 답할 수 있음이다. 신호비는 그것을 미리 짐작하는 값이다. D03·D04의 W(m) 결과는 이 사다리의 첫 단으로만 쓴다.

## 3. 초고 v1 (A–C가 끝난 뒤 작성)

`manuscript/draft.md`에 WRR 형식으로 쓴다. 결과 문장의 주어는 기록, 대수층, 운영 달력이다. 모든 수치는 `results/` 파일에서 오고, `manuscript/numbers_sources.md`에 수치마다 출처 파일과 행·열을 적는다.

**글쓰기 규칙**
- 영어로만 쓴다. 표와 그림에도 한글을 쓰지 않는다. 그림은 Times New Roman으로 그리고 겹침을 육안으로 확인한다.
- 세미콜론과 em-dash는 최소로 쓴다. 자기 고백조나 과장("state-of-the-art", "novel", "first")은 쓰지 않는다.
- 모형 이름은 제목, Key Points, Figure 1에 넣지 않는다.
- AGU 규칙을 지킨다. Key Points는 3개, 각 140자 이하다. 초록은 250단어 미만이다. Plain Language Summary는 200단어 이하다. Open Research 절을 둔다.
- AI 사용 고지는 Paper 2 원고(`1_zero_shot/manuscript/draft.md` 2.7절)와 같은 형식으로 방법 절에 둔다. 그 파일은 읽기만 한다.
- 저자 목록은 `[AUTHORS TBD]`로 둔다.

**가제:** "When can a pumping record answer 'what if we stop?'"

**중심 문장(초록과 결론에 같은 뜻으로):** A pumping record answers a what-if question when the pumping changes it contains are large against background variability at the observation point and long against the time scale the question asks about.

**절 구성.** 각 결과 절은 1절 원리의 한 면을 맡는다. 원리와 관계없는 결과는 보충으로 보낸다.
1. **Introduction.** 운영 관정의 양수 기록은 흔하다. 그러나 그 기록만으로 "양수를 멈추거나 늘리면 수위가 어떻게 되는가"를 물을 수 있는지는 정해져 있지 않다. 회복시험, 자료가치 연구(Brakenhoff et al. 2022의 t95 규칙, Dausman et al. 2010, Vilhelmsen and Ferré 2018), 시스템 식별의 지속 가진 개념이 이 질문에 가깝다. 하지만 운영 기록의 휴지와 전환, what-if의 방향과 크기를 직접 다루지 않는다. 사전학습 예측 모형을 시나리오 엔진으로 쓰려는 흐름이 있으므로, 기록이 정한 답을 그런 엔진이 존중하는지도 묻는다. 문헌은 `debate/literature_check.md`에서 확인된 것과, 새로 인용하는 경우 본문까지 확인한 것만 쓴다.
2. **Methods.** 쌍둥이 진실(선형 저류 자연 성분, Hantush 누수 응답, AR(1) 잡음, 실측 강수). 선언한 응답족. 지표의 사다리(1절). 기록 설계(휴지 비, 전환 횟수와 최근성, 양수량 배율, 저장계수, 관측 거리). 운영 달력 세 형태와 문헌 근거. 선형화 정보량. 시나리오 엔진(동결 사전학습 모형, 원시 양수 공변량)과 구조 정합 기준. 실현 단위 부트스트랩.
3. **Results.**
   - 3.1 **모호함은 잡음과 응답 모양이 정한다.** W(m)가 양수 규모와 이득에 거의 무관하므로, 답할 수 있음은 신호비에 비례한다. 선형화 정보량이 표집 W를 얼마나 재현하는지(C)도 여기에 둔다.
   - 3.2 **가진의 시간척도가 질문의 시간척도와 맞아야 한다.** 전환 횟수는 처음 며칠, 휴지는 10–30일을 알려 준다. 저장계수는 응답시간을 통해 같은 리드의 의미를 바꾼다. 시간척도 붕괴(B)와 구간 가리기(C)가 근거다.
   - 3.3 **답할 수 있음 지도.** 방향과 크기의 신호비 문턱, 그리고 휴지와 신호의 교환 비율이다. 관측 거리와 대수층 유형은 사례를 신호비 축 위에서 옮기는 경로로 쓴다(A).
   - 3.4 **운영 달력은 자연 실험이다.** 어떤 운영의 어떤 가진이 어느 달에 답할 정보를 만드는가. r = 200 m와 20 m의 차이를 함께 보인다. 활성 원점만 분모로 삼고, 양수가 없는 원점은 "질문이 없음"으로 짧게 밝힌다.
   - 3.5 **시나리오 엔진은 기록의 정보를 꺼내 쓰지 못한다.** 기록이 방향을 정한 곳에서의 모순 비율과 신호비에 따른 변화, 증량 질문의 고신호 모순, 휴지에 대한 무반응을 다룬다. 구조 정합 대조를 함께 둔다.
4. **Discussion.**
   - 실무 규칙: 운영자는 한 번의 정지로 관측 지점의 신호비를 대략 추정할 수 있다. 그 값이 약 0.4를 넘으면 기록이 방향에 답한다. 모자라면 응답시간 몇 배의 휴지가 부족분의 일부를 메운다. 선형화 정보량이 맞는 범위에서는 양수 일지만으로 이를 미리 계산할 수 있다.
   - 무엇을 기록해야 하는가: 유량과 수위를 같은 시간 해상도로, 휴지의 실제 길이와 함께 기록한다. 수막재배의 야간 정지처럼 하루보다 짧은 가진은 일 단위 기록에 남지 않는다.
   - 한계: Tier A 선형 쌍둥이, 선언한 응답족 안의 결론, 비피압 선형 근사, 단일 엔진과 단일 설정, 현장 검증 없음.
   - 정규화 기제는 한 문장의 가설로만 쓴다.
5. **Conclusions.** 3–4문장. 중심 문장, 교환 비율, 엔진에 대한 한 문장.

**그림 (본문 5개 안팎)**
- **Figure 1:** 원리 도식. 양수 기록 위에 전환·휴지·시즌 시작을 표시하고, 각각이 정보를 주는 리드 범위를 보인다. C의 구간 가리기 결과를 실제 수치로 쓴다.
- **Figure 2:** 모호함과 시간척도. 배율에 따른 W(m) 불변성, W_lin 대 표집 W, 리드/응답시간 붕괴를 한 그림의 패널로 둔다.
- **Figure 3:** 답할 수 있음 지도(D05 Figure 1 P1 기반). 거리 20 m 사례를 덧그린다.
- **Figure 4:** 운영 달력. r = 200 m와 20 m를 나란히 두고, 활성 원점만 색으로 칠한다.
- **Figure 5:** 엔진과 기록(D05 Figure 4의 P1과 P2를 한 그림에).
- **보충(`manuscript/supplementary.md`):** 층 지시변수 적합, 양의 ρ 민감도, 휴지와 전환의 원 폭 그림(D03 Figure A, D04 Figure 2), 최근성 대비, 정규화 진단, 필터 트랙(D03·D04), 구조 정합 기준의 수치 품질, D05 r = 200 m 달력 세부.

