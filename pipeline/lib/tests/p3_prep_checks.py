"""Implementation-preparation checks (D02, B). FIXTURE ONLY for anything generated:
synthetic rain, fake realization. No pilot cell is generated, forecast or scored.
Writes results/pilot/prep_checks/prep_checks.json."""
import json, sys, time, hashlib, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "lib/tests"))
import p3_kernels as K, p3_generator as g, p3_wenvelope as w, p3_pastas as pt
from p3_fixture import fixture_rain, fixture_realization
from scipy.integrate import quad
from scipy.special import k0

out = {"label": "FIXTURE/unit checks — not pilot results", "checks": {}}
C = out["checks"]


def record(name, passed, **info):
    C[name] = dict(passed=bool(passed), **info)
    print(("PASS " if passed else "FAIL ") + name, {k: v for k, v in info.items() if not isinstance(v, (list, dict))})


# 1. exact kernel vs independent quadrature of W(u, rho)
def Wq(u, rho):
    f = lambda z: np.exp(-np.exp(z) - rho ** 2 / (4 * np.exp(z)))
    return quad(f, np.log(u), 8.0, limit=500)[0]
ks = [(53.48475127, 0.009348458918), (32.77609031, 0.1525502265), (1.0, 25.0), (3000.0, 1e-4), (10.0, 1e-3), (500.0, 2.0)]
t = np.array([0.5, 1, 2, 5, 10, 30, 60, 100, 300, 1000, 1054.0])
S = K.hantush_unit_step([k[0] for k in ks], [k[1] for k in ks], t)
err = max(abs(S[i, j] - Wq(ks[i][0] * ks[i][1] / t[j], 2 * np.sqrt(ks[i][1])) / (2 * k0(2 * np.sqrt(ks[i][1])))) for i in range(len(ks)) for j in range(len(t)))
record("kernel_exact_vs_quad", err < 1e-5, max_abs_err_frac_gain=float(err))
t95 = K.hantush_t_frac([ks[0][0], ks[1][0]], [ks[0][1], ks[1][1]])
record("design_t95_60d", np.all(np.abs(t95 - 60) < 0.01), t95=t95.tolist())
Kn, Kc = K.natural_kernel([2.0], [10.0], [60.0], 3000)
q = np.exp(-1 / 60.0); P = np.random.default_rng(1).gamma(0.5, 10, 500)
rec = np.zeros(500); R = np.convolve(P, np.diff(__import__("scipy.special", fromlist=["gammainc"]).gammainc(2.0, np.arange(501) / 10.0)))[:500]
for n in range(500):
    rec[n] = q * (rec[n - 1] if n else 0.0) + (1 - q) * R[n]
record("natural_kernel_equals_reservoir_recursion", np.max(np.abs(np.convolve(P, Kn[0])[:500] - rec)) < 1e-10,
       max_abs=float(np.max(np.abs(np.convolve(P, Kn[0])[:500] - rec))), unit_gain_sum=float(Kc[0, -1]))

# 2. generator invariants over the 20 cells of one FIXTURE realization
d = g.Design(); rain = fixture_rain(); real = fixture_realization()
bg = g.realization_background(real, rain, d)
cases = [g.build_case(bg, rho, sr, pir, d) for pir in d.pi_layers for sr in d.signal_ratios for rho in d.rest_ratios]
ids = [c["case_id"] for c in cases]
eps0 = cases[0]["truth"]["eps"]; rain0 = cases[0]["tf_input"]["rain"]
same_noise = all(np.array_equal(c["truth"]["eps"], eps0) for c in cases)
same_rain = all(np.array_equal(c["tf_input"]["rain"], rain0) for c in cases)
mult_ok = True; zero_ok = True
for c in cases:
    tt = c["truth"]["theta_true"]; Qw = c["tf_input"]["pumping_context"]; R_ = tt["R_days"]
    base = tt["Q_ref"] * bg["mult"]
    outside = np.ones(g.CTX, bool); outside[g.REST_END - R_:g.REST_END] = False
    mult_ok &= np.array_equal(Qw[outside], base[outside])
    zero_ok &= (Qw == 0).sum() == R_ and np.all(Qw[~outside] == 0) and tt["q0"] > 0
pid = max(float(np.max(np.abs(c["truth"]["E_true"]["P2_current_vs_1p5x"] + 0.5 * c["truth"]["E_true"]["P1_continue_vs_stop"]))) for c in cases)
record("generator_matched_cells", same_noise and same_rain and mult_ok and zero_ok, n_cells=len(cases), unique_ids=len(set(ids)) == 20,
       noise_identical=same_noise, rain_identical=same_rain, base_identical_outside_rest=bool(mult_ok), rest_only_zero_days=bool(zero_ok))
record("truth_pair2_identity", pid < 1e-12, max_abs_m=pid)
sr_ok = all(abs(c["truth"]["theta_true"]["A"] * c["truth"]["theta_true"]["Q_ref"] / bg["sigma_bg"] - c["meta"]["signal_ratio"]) < 1e-12 for c in cases)
record("signal_ratio_definition", sr_ok)
leak = [k for k in cases[0]["tf_input"] if k in ("E_true", "theta_true", "h_star_a", "eps", "h_nat")]
record("no_truth_in_tf_input", not leak, keys=sorted(cases[0]["tf_input"].keys()))


# 2b. exact well-specification: W family at the true nonlinear parameters reproduces noise-free h* (SSE ~ 0)
import pastas as ps
worst = 0.0; coef_err = 0.0
for c in cases[:: 5]:
    tf0 = dict(c["tf_input"]); tf0["head_context"] = c["truth"]["h_star_a"][:g.CTX]
    eng0 = w.WEngine(tf0); tt = c["truth"]["theta_true"]
    x = np.array([tt["n"], np.log10(tt["theta"]), np.log10(tt["tau"]), np.log10(tt["a"]), np.log10(tt["b"])])
    sse, th, _, _, _ = eng0.point(x)
    worst = max(worst, np.sqrt(sse / g.CTX))
    coef_err = max(coef_err, abs(th[w.IA] - tt["A"]) / tt["A"], abs(th[w.IPP] - tt["eta_pump_true"]) / max(abs(tt["eta_pump_true"]), 1e-12),
                   abs(th[w.INAT] - tt["nat_gain"]) / tt["nat_gain"])
record("family_contains_truth_exactly", worst < 1e-9 and coef_err < 1e-6, max_rmse_m=float(worst), max_rel_coef_err=float(coef_err))
# 2c. matched reference kernels equal the shared kernels
Hx = pt.HantushExactP3(); tq = np.arange(1, 300.0)
e1 = max(float(np.max(np.abs(Hx.step([1.0, a_, b_], dt=tq) - ps.Hantush.quad_step(1.0, a_, b_, tq)))) for a_, b_ in ks[:2] + [(10.0, 1e-3)])
Gr = pt.GammaReservoirP3(); Kn2, Kc2 = K.natural_kernel([2.0], [10.0], [60.0], 400)
e2 = float(np.max(np.abs(Gr.step([1.0, 2.0, 10.0, 60.0], dt=np.arange(1, 401.0)) - Kc2[0, 1:])))
record("reference_kernels_match_shared", e1 < 1e-5 and e2 < 1e-12, hantush_exact_vs_pastas_quad_step=e1, gamma_reservoir_vs_kernel=e2)


# 2d. exact bounded linear feasibility vs brute force (bounds inactive and forced active)
from scipy.optimize import lsq_linear
cfx = cases[7]; engx = w.WEngine(cfx["tf_input"]); ttx = cfx["truth"]["theta_true"]
xx = np.array([ttx["n"], np.log10(ttx["theta"]), np.log10(ttx["tau"]), np.log10(ttx["a"]), np.log10(ttx["b"])])
maxdev = {}
for label, eb in (("bounds_inactive", engx.eta_bound), ("eta_bounds_forced_active", 1e-4)):
    engx.eta_bound = eb
    sse, th, X, u, t95 = engx.point(xx)
    tau = sse * 1.2
    iv = engx.exact_interval(X.T @ X, X.T @ engx.yc, tau)
    lb, ub = engx._lin_bounds()
    grid = np.linspace(0, min(engx.Amax, 3 * ttx["A"]), 3001)
    feas = []
    for A in grid:
        r = lsq_linear(X[:, 1:], engx.yc - X[:, 0] * A, bounds=(lb[1:], ub[1:]), method="bvls")
        feas.append(np.sum(r.fun ** 2) <= tau)
    feas = np.array(feas); step = grid[1] - grid[0]
    bl, bh = grid[feas].min(), grid[feas].max()
    maxdev[label] = dict(exact=list(map(float, iv)), brute=[float(bl), float(bh)], grid_step=float(step))
    maxdev[label]["ok"] = bool(abs(iv[0] - bl) <= step and abs(iv[1] - bh) <= step)
engx.eta_bound = 10.0 * np.ptp(engx.y)
record("exact_bounded_interval_vs_bruteforce", all(v["ok"] for v in maxdev.values()), detail=maxdev)

# 3. A adapter schema compatibility (validator only; no model import)
spec = importlib.util.spec_from_file_location("adapter", ROOT / "lib/timesfm_pilot_adapter.py")
ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)
for H in (10, 30):
    rows = [r for c in cases[:3] for r in g.tool_rows(c, H)]
    arr = g.tool_npz_arrays(rows, "0" * 64)
    order, _ = ad.validate(arr, H)
    tampered = dict(arr); tampered["pumping"] = arr["pumping"].copy(); tampered["pumping"][1, 5] += 1.0
    try:
        ad.validate(tampered, H); neg = False
    except ValueError:
        neg = True
    record(f"adapter_validate_H{H}", len(order) == len(rows) and neg, n_rows=len(rows), negative_history_tamper_rejected=neg)

# 4. real KMA files: hash + calendar fit (read only; no series generated)
cal = json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))
pref = {"안동태화_충적": "20dddf7f15db", "산청산청_암반": "e95466d5ea3b", "남해남해_암반": "7733e1352ddc"}
hash_ok = {}
for stem, p in pref.items():
    s = g.load_rain(stem, p)
    hash_ok[stem] = dict(sha256=s.attrs["sha256"], start=str(s.index[0].date()), end=str(s.index[-1].date()), n=len(s))
fit_ok = all(pd.Timestamp(r["context_start"]) - pd.Timedelta(days=g.GEN_WARMUP) >= pd.Timestamp("2005-01-01")
             and pd.Timestamp(r["context_start"]) + pd.Timedelta(days=g.CTX + g.HMAX - 1) <= pd.Timestamp("2024-12-31") for r in cal)
record("kma_hash_and_calendar", fit_ok, files=hash_ok, generator_warmup_days=g.GEN_WARMUP)

# 5. W engine + Pastas on a spread of FIXTURE cells: feasibility, convergence, truth compatibility, consistency
sel = [c for c in cases if c["meta"]["signal_ratio"] in (0.5, 2.0) and c["meta"]["rest_ratio"] in (0.25, 4.0)]
wlog = []
for c in sel:
    eng = w.WEngine(c["tf_input"]); rec = eng.run()
    tc = w.truth_compatibility(eng, rec, c["truth"]["theta_true"], c["truth"]["E_true"]) if "envelope" in rec else {}
    pr = rec.get("envelope", {})
    ident = max(max(abs(pr["P2_current_vs_1p5x"]["W"][k] - 0.5 * pr["P1_continue_vs_stop"]["W"][k]),
                    abs(pr["P2_current_vs_1p5x"]["sup"][k] + 0.5 * pr["P1_continue_vs_stop"]["inf"][k]),
                    abs(pr["P2_current_vs_1p5x"]["inf"][k] + 0.5 * pr["P1_continue_vs_stop"]["sup"][k])) for k in range(30)) if pr else None
    t0 = time.time(); fit = pt.fit_and_propagate(c["tf_input"], seed=real["seed_pastas_param_sample"]); tp = time.time() - t0
    cons = pt.simulate_consistency(c["tf_input"], fit)
    wlog.append(dict(case_id=c["case_id"], W_runtime_s=rec.get("runtime_s"), flags=rec["flags"], converged=rec.get("converged"),
                     n_accepted=rec.get("n_accepted_total"), local_rounds=len(rec.get("local", [])), refinement_first_gain=rec.get("refinement_first_gain_over_sampled"), refinement_increments=rec.get("refinement_increments"),
                     n_eff=rec.get("tolerance", {}).get("n_eff"), rmse_tol_m=rec.get("tolerance", {}).get("rmse_equiv_m"),
                     truth_nonlinear_in_G=tc.get("truth_nonlinear_in_G"), E_true_inside_1to10=all(tc["E_true_inside"]["P1_continue_vs_stop"][:10]) if tc else None,
                     pair2_W_identity_maxabs=ident, n_bound_active_exact={N: lv.get("x1_n_bound_active_exact") for N, lv in rec.get("global", {}).items()}, pastas_runtime_s=tp, pastas_draws=fit["n_param_draws_accepted"], pastas_simulate_consistency_m=cons))
    print(wlog[-1])
out["fixture_engine_runs"] = wlog
record("W_engine_runs_fixture", all(x["W_runtime_s"] is not None and "W_failed" not in x["flags"] for x in wlog),
       n=len(wlog), n_converged=sum(bool(x["converged"]) for x in wlog), n_truth_in_G=sum(bool(x["truth_nonlinear_in_G"]) for x in wlog),
       max_W_runtime_s=max(x["W_runtime_s"] for x in wlog), pair2_identity_max=max(x["pair2_W_identity_maxabs"] for x in wlog))
record("pastas_consistency", all(x["pastas_simulate_consistency_m"] < 1e-6 for x in wlog), max_m=max(x["pastas_simulate_consistency_m"] for x in wlog))
est = 200 * (np.mean([x["W_runtime_s"] for x in wlog]) + np.mean([x["pastas_runtime_s"] for x in wlog]))
out["budget_estimate_200_cases_s"] = float(est)
(ROOT / "results/pilot/prep_checks").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "results/pilot/prep_checks/prep_checks.json", "w"), ensure_ascii=False, indent=1, default=str)
print("ALL PASS" if all(v["passed"] for v in C.values()) else "SOME FAIL", "budget_s", est)
