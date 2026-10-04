"""Correctness suite for lib/p3_wenvelope_repaired_v1_2.py (D02 W repair, B). Imports the NEW module.
Analytic/synthetic designs and FIXTURE cells only (synthetic rain, fake realization); no pilot case is scored.
Writes results/pilot/repair_v1_2_tests.json. Expected legacy failures are recorded as such, not as passes."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "lib/tests"))
import p3_generator as g  # noqa: E402
import p3_wenvelope as w  # noqa: E402
import p3_wenvelope_repaired_v1_2 as v  # noqa: E402
from p3_fixture import fixture_rain, fixture_realization  # noqa: E402

OUT = ROOT / "results/pilot/repair_v1_2_tests.json"
res = {"label": "v1.2 correctness suite: analytic/synthetic + FIXTURE cells; not pilot results", "module": "lib/p3_wenvelope_repaired_v1_2.py", "checks": {}}
C = res["checks"]


def record(name, passed, expected_failure=False, **info):
    C[name] = dict(passed=bool(passed), expected_failure=bool(expected_failure), **info)
    tag = ("XFAIL-OK " if passed else "XFAIL-UNEXPECTED ") if expected_failure else ("PASS " if passed else "FAIL ")
    print(tag + name, {k: val for k, val in info.items() if not isinstance(val, (list, dict))})


def trf_f(X, y, lb, ub, A, ia=0):
    """Independent oracle: trust-region reflective bounded LS (different algorithm from BVLS), unscaled."""
    rest = [c for c in range(X.shape[1]) if c != ia]
    r = lsq_linear(X[:, rest], y - X[:, ia] * A, bounds=(lb[rest], ub[rest]), method="trf", tol=1e-14, max_iter=10000)
    return float(np.sum((y - X[:, ia] * A - X[:, rest] @ r.x) ** 2))


def probes(Xp, yp, lbp, ubp, ivp, tp, rel=1e-6):
    """Independent TRF check at +-rel*width around each endpoint: inside feasible, outside infeasible (bounds excepted)."""
    wdt = ivp["hi"] - ivp["lo"]; ok = True
    for end, sgn, clip in (("lo", -1, ivp["clip_lo"]), ("hi", 1, ivp["clip_hi"])):
        ok &= trf_f(Xp, yp, lbp, ubp, ivp[end] - sgn * rel * wdt) <= tp
        if not clip:
            ok &= trf_f(Xp, yp, lbp, ubp, ivp[end] + sgn * rel * wdt) > tp
    return bool(ok)


rng = np.random.default_rng(12)
n = 300
X = np.column_stack([rng.normal(size=n) + 2.0, rng.normal(size=n), np.ones(n), rng.normal(size=n), np.exp(-np.arange(n) / 50)])
th0 = np.array([1.5, 0.3, 0.1, 0.8, -0.2])
y = X @ th0 + 0.3 * rng.normal(size=n)
BIG = 1e6
lb_in = np.array([0.0, -BIG, -np.inf, 0.0, -BIG]); ub_in = np.array([10.0, BIG, np.inf, np.inf, BIG])

# 1. inactive bounds: analytic ellipsoid
thh, *_ = np.linalg.lstsq(X, y, rcond=None); smin = float(np.sum((y - X @ thh) ** 2))
Sig = np.linalg.inv(X.T @ X); tau = smin * 1.2
r_ = np.sqrt((tau - smin) * Sig[0, 0])
iv = v.direct_interval(X, y, lb_in, ub_in, tau)
err = max(abs(iv["lo"] - (thh[0] - r_)), abs(iv["hi"] - (thh[0] + r_)))
record("inactive_bounds_vs_analytic_ellipsoid", iv["ok"] and err < 1e-9 * abs(thh[0]) and not iv["clip_lo"] and not iv["clip_hi"],
       max_abs_err=err, lo=iv["lo"], hi=iv["hi"], n_eval=iv["n_eval"])

# 2. forced active nuisance bounds (beta >= 0 and |eta| small) vs independent TRF oracle + grid
lb_f = np.array([0.0, -0.05, -np.inf, 0.9, -0.01]); ub_f = np.array([10.0, 0.05, np.inf, np.inf, 0.01])
tau_f = None
f_bv, Astar = v.direct_profile(X, y, lb_f, ub_f)
fmin_f = f_bv(Astar)[0]; tau_f = fmin_f * 1.15
ivf = v.direct_interval(X, y, lb_f, ub_f, tau_f)
thlo = ivf["th_lo"]
act = bool(np.any(np.isclose(thlo[1:], lb_f[1:])) or np.any(np.isclose(thlo[1:], ub_f[1:])))
grid = np.linspace(ivf["lo"] - 0.05, ivf["hi"] + 0.05, 2001); step = grid[1] - grid[0]
feas = np.array([trf_f(X, y, lb_f, ub_f, A) <= tau_f for A in grid])
bl, bh = grid[feas].min(), grid[feas].max()
end_ok = trf_f(X, y, lb_f, ub_f, ivf["lo"]) <= tau_f * (1 + 1e-9) and trf_f(X, y, lb_f, ub_f, ivf["hi"]) <= tau_f * (1 + 1e-9)
out_ok = trf_f(X, y, lb_f, ub_f, ivf["out_lo"]["A"]) > tau_f * (1 - 1e-9) and trf_f(X, y, lb_f, ub_f, ivf["out_hi"]["A"]) > tau_f * (1 - 1e-9)
bnd_ok = bool(np.all(thlo >= lb_f) and np.all(thlo <= ub_f) and np.all(ivf["th_hi"] >= lb_f) and np.all(ivf["th_hi"] <= ub_f))
res_ok = abs(float(np.sum((y - X @ thlo) ** 2)) - ivf["sse_lo"]) < 1e-12 * tau_f and ivf["sse_lo"] <= tau_f and ivf["sse_hi"] <= tau_f
pr2 = probes(X, y, lb_f, ub_f, ivf, tau_f)
record("forced_active_bounds_vs_trf_oracle_and_grid", pr2 and act and abs(bl - ivf["lo"]) <= step and abs(bh - ivf["hi"]) <= step and end_ok and out_ok and bnd_ok and res_ok,
       trf_probes_1e6width=pr2, nuisance_bound_active_at_lo=act, grid=[float(bl), float(bh)], direct=[ivf["lo"], ivf["hi"]], grid_step=float(step), endpoints_feasible_trf=end_ok,
       outside_probes_infeasible_trf=out_ok, endpoint_coefs_in_bounds=bnd_ok, endpoint_residual_consistent=res_ok,
       out_gap=[ivf["out_lo"]["gap"], ivf["out_hi"]["gap"]])

# 3. tau below / at / above the constrained minimum
below = v.direct_interval(X, y, lb_f, ub_f, fmin_f * (1 - 1e-6))
at = v.direct_interval(X, y, lb_f, ub_f, fmin_f)
above = v.direct_interval(X, y, lb_f, ub_f, fmin_f * (1 + 1e-6))
record("tau_below_at_above_min", (not below["ok"]) and at["ok"] and at["lo"] <= Astar + 1e-9 and at["hi"] >= Astar - 1e-9 and above["ok"] and above["lo"] < Astar < above["hi"],
       below_reason=below.get("reason"), at=[at.get("lo"), at.get("hi")], at_singleton_within_allow=at.get("singleton_within_allow"), Astar=Astar,
       above=[above.get("lo"), above.get("hi")])

# 4. A clipping at 0 and at A_max
y0 = X @ np.array([0.02, 0.3, 0.1, 0.8, -0.2]) + 0.3 * rng.normal(size=n)
ivc = v.direct_interval(X, y0, lb_in, ub_in, float(np.sum((y0 - X @ np.linalg.lstsq(X, y0, rcond=None)[0]) ** 2)) * 1.3)
ub_small = ub_in.copy(); ub_small[0] = 1.52
ivm = v.direct_interval(X, y, lb_in, ub_small, tau)
record("A_clipping_bounds", ivc["ok"] and ivc["clip_lo"] and ivc["lo"] == 0.0 and ivc["out_lo"] is None and ivm["ok"] and ivm["clip_hi"] and ivm["hi"] == 1.52,
       lo_clip=[ivc["lo"], ivc["hi"]], hi_clip=[ivm["lo"], ivm["hi"]])

# 5. endpoint curves and sign swap, pair identity (analytic A in [1,2], u=(2,-3))
env = v.new_env(["P1", "P2"])
u1 = np.zeros(v.HMAX); u1[:2] = (2.0, -3.0); u2 = -0.5 * u1
for A in (1.0, 2.0):
    v.merge_curve(env, {"P1": A * u1, "P2": A * u2}, dict(A=A, src="t"))
ok5 = (tuple(env["P1"]["inf"][:2]) == (2.0, -6.0) and tuple(env["P1"]["sup"][:2]) == (4.0, -3.0)
       and env["P1"]["wsup"][1]["A"] == 1.0 and env["P1"]["winf"][1]["A"] == 2.0
       and np.allclose(env["P2"]["sup"], -0.5 * env["P1"]["inf"]) and np.allclose(env["P2"]["inf"], -0.5 * env["P1"]["sup"]))
record("endpoint_sign_swap_and_pair_identity", ok5, P1_inf=env["P1"]["inf"][:2].tolist(), P1_sup=env["P1"]["sup"][:2].tolist(),
       negative_u_sup_attained_by_low_A=env["P1"]["wsup"][1]["A"] == 1.0)

# 6. rank deficiency / near-collinearity / column scaling
Xd = np.column_stack([X, X[:, 2]]); lbd = np.append(lb_in, -np.inf); ubd = np.append(ub_in, np.inf)
ivd = v.direct_interval(Xd, y, lbd, ubd, tau)
Xn = np.column_stack([X, X[:, 3] * (1 + 1e-9 * rng.normal(size=n))])
fn_, An_ = v.direct_profile(Xn, y, lbd, ubd); tau_n = fn_(An_)[0] * 1.2; ivn = v.direct_interval(Xn, y, lbd, ubd, tau_n)
Xs = X.copy(); Xs[:, 1] *= 1e6; lbs = lb_in.copy(); ubs = ub_in.copy(); lbs[1] /= 1e6; ubs[1] /= 1e6
Xs[:, 4] *= 1e-5; lbs[4] *= 1e5; ubs[4] *= 1e5
ivs = v.direct_interval(Xs, y, lbs, ubs, tau)
dd = max(abs(ivd["lo"] - iv["lo"]), abs(ivd["hi"] - iv["hi"]))

dn = probes(Xn, y, lbd, ubd, ivn, tau_n)
ds = max(abs(ivs["lo"] - iv["lo"]), abs(ivs["hi"] - iv["hi"]))
record("rank_deficient_collinear_and_scaling", dd < 1e-9 and dn and ds < 1e-9, duplicate_col_dev=dd, near_collinear_trf_probes_1e6width=dn, column_scaling_dev=ds)

# 7. legacy ridge path: expected failure on exact-fit boundary; new direct path keeps the direct optimum
leg = object.__new__(w.WEngine)
Xe = X.copy(); te = np.array([1.5, 0.3, 0.1, 0.8, -0.2]); ye = Xe @ te
leg.yy = float(ye @ ye); leg.Amax = 10.0; leg.eta_bound = BIG
dmin = float(np.sum((ye - Xe @ np.linalg.lstsq(Xe, ye, rcond=None)[0]) ** 2))
tau_e = max(dmin, 1e-20)
legacy_iv = leg.exact_interval(Xe.T @ Xe, Xe.T @ ye, tau_e)
new_iv = v.direct_interval(Xe, ye, lb_in, ub_in, tau_e)
legacy_fails = legacy_iv is None or not (legacy_iv[0] <= 1.5 <= legacy_iv[1])
record("legacy_gram_ridge_rejects_exact_fit_boundary", legacy_fails, expected_failure=True, legacy=None if legacy_iv is None else list(legacy_iv),
       tau=tau_e, note="unchanged frozen exact_interval (ridge 1e-12 on scaled Gram); known structural failure at tau = direct minimum")
record("direct_path_keeps_exact_fit_singleton", new_iv["ok"] and abs(new_iv["lo"] - 1.5) < 1e-6 and abs(new_iv["hi"] - 1.5) < 1e-6,
       direct=[new_iv.get("lo"), new_iv.get("hi")], reason=new_iv.get("reason"))
# magnified-lambda scalar: X=1, y=1, (1-A)^2 + lam A^2; tau between ridged min and ridged objective at A=1
lam = 1e-3; rmin = lam / (1 + lam); tau_l = 0.5 * (rmin + lam)
disc = 1 - (1 + lam) * (1 - tau_l)
rl = [(1 - np.sqrt(disc)) / (1 + lam), (1 + np.sqrt(disc)) / (1 + lam)]
record("magnified_lambda_ridge_interval_excludes_direct_A", not (rl[0] <= 1.0 <= rl[1]), expected_failure=True, ridge_interval=rl, lam=lam, tau=tau_l)
Xo = np.eye(2)
ivo = v.direct_interval(Xo, np.array([1.0, 0.0]), np.array([0.0, -1.0]), np.array([2.0, 1.0]), tau_l)
record("direct_scalar_contains_A1", ivo["ok"] and ivo["lo"] <= 1.0 <= ivo["hi"], direct=[ivo.get("lo"), ivo.get("hi")])

# 8. full-curve bookkeeping (analytic)
H = v.HMAX
def curve(a, b):
    c = np.ones(H); c[0], c[1] = a, b; return c
e = v.new_env(["P"]); v.merge_curve(e, {"P": curve(1, 1)}, dict(id="old"))
scalar = e["P"]["sup"].copy(); scalar[0] = 2.0  # legacy target-only write
v.merge_curve(e, {"P": curve(2, 3)}, dict(id="new"))
b1 = tuple(e["P"]["sup"][:2]) == (2.0, 3.0) and tuple(scalar[:2]) == (2.0, 1.0)
e2 = v.new_env(["P"]); v.merge_curve(e2, {"P": curve(1, 1)}, dict(id="old")); v.merge_curve(e2, {"P": curve(1, 3)}, dict(id="tie"))
b2 = e2["P"]["sup"][1] == 3.0 and e2["P"]["wsup"][0]["id"] == "old" and e2["P"]["wsup"][1]["id"] == "tie"
e3 = v.new_env(["P"]); v.merge_curve(e3, {"P": -curve(1, 1)}, dict(id="old")); v.merge_curve(e3, {"P": -curve(2, 3)}, dict(id="new"))
b3 = tuple(e3["P"]["inf"][:2]) == (-2.0, -3.0) and e3["P"]["winf"][1]["id"] == "new"
e4 = v.new_env(["P"]); v.merge_curve(e4, {"P": curve(1, 1)}, dict(id="old")); v.merge_curve(e4, {"P": curve(0.5, 4)}, dict(id="loss"))
b4 = e4["P"]["sup"][0] == 1.0 and e4["P"]["wsup"][0]["id"] == "old" and e4["P"]["sup"][1] == 4.0 and e4["P"]["inf"][0] == 0.5
record("full_curve_merge_vs_scalar_counterexample", b1 and b2 and b3 and b4, full=e["P"]["sup"][:2].tolist(), legacy_scalar=scalar[:2].tolist(),
       tie_keeps_old_witness=b2, inf_twin=b3, losing_target_terminal_expands_other_lead=b4)

# 9. engine on FIXTURE cells: terminals, infeasible terminal not merged, zero-accepted seed, >=5 hull branch, witnesses
d = g.Design(); rain = fixture_rain(); real = fixture_realization()
bg = g.realization_background(real, rain, d)
cells = {c["case_id"]: c for c in (g.build_case(bg, rho, sr, pir, d) for pir in d.pi_layers for sr in d.signal_ratios for rho in d.rest_ratios)}
cid = sorted(cells)[0]; cell = cells[cid]
eng = v.WEngineV12(cell["tf_input"])
# 9a. infeasible terminal is logged and not merged
xb = np.array([2.0, 1.0, 1.7, 1.5, -1.0])
sse_x = eng.point(xb)[0]
dp_bad = eng.direct_point_interval(xb, sse_x * 0.5)
env9 = v.new_env(eng.pairs); led = []; log = []
eng.merge_terminals(env9, [dict(side="sup", k=10, dp=dp_bad, feasible=bool(dp_bad["iv"].get("ok")))], led, log)
record("infeasible_terminal_not_merged", (not dp_bad["iv"]["ok"]) and not led and np.all(~np.isfinite(env9[eng.pairs[0]]["sup"])) and log[0]["feasible"] is False,
       reason=log[0].get("reason"))
dp_ok = eng.direct_point_interval(xb, sse_x * 1.2)
eng.merge_terminals(env9, [dict(side="sup", k=10, dp=dp_ok, feasible=True)], led, log)
P1 = eng.pairs[0]
full_ok = all(np.isclose(env9[p]["sup"], np.maximum(dp_ok["iv"]["lo"] * dp_ok["u"][p], dp_ok["iv"]["hi"] * dp_ok["u"][p]), rtol=0, atol=0).all() for p in eng.pairs)
record("feasible_terminal_full_30lead_both_pairs", full_ok and len(led) == 2, n_changed=log[-1]["n_ordinates_changed"])


def zero_hook(engine):
    def hook(st, taus=()):
        s, b, envs = w.WEngine.scan(engine, st, taus)
        for e_ in envs:
            e_["lo"][:] = np.inf; e_["hi"][:] = -np.inf; e_["nacc"] = 0; e_["acc_pts"] = np.empty((0, 5))
        return s, b, envs
    return hook


def contained(rec, dp_curves):
    return all(rec["envelope"][p]["inf"][k] - 1e-15 <= c[p][k] <= rec["envelope"][p]["sup"][k] + 1e-15 for c in dp_curves for p in c for k in range(v.HMAX))


small = dict(global_levels=(32, 64), n_local=48, max_local_rounds=2)
t0 = time.time()
engz = v.WEngineV12(cell["tf_input"]); recz = engz.run(_scan_hook=zero_hook(engz), **small)
xz = np.asarray(recz["tolerance"]["xbest"]); dpz = engz.direct_point_interval(xz, recz["tolerance"]["tau"])
cz = [{p: dpz["iv"]["lo"] * dpz["u"][p] for p in engz.pairs}, {p: dpz["iv"]["hi"] * dpz["u"][p] for p in engz.pairs},
      {p: dpz["th_point"][v.IA] * dpz["u"][p] for p in engz.pairs}]
record("zero_accepted_samples_bestfit_seed", recz["W_status"] == "ok" and recz["n_accepted_total"]["x1"] == 0 and "sparse_G" in recz["flags"]
       and contained(recz, cz) and all(l["xbest_in_box"] and l["box_branch"] == "best_starts_hull" for l in recz["local"])
       and recz["witness_check"]["pass"] and all(recz["bestfit_seed"][k]["direct_feasible"] and recz["bestfit_seed"][k]["interval_ok"] for k in ("x0.5", "x1", "x2")),
       W10=recz["summary"][P1]["W10"], flags=recz["flags"], witness_check=recz["witness_check"], runtime_s=time.time() - t0)
t0 = time.time()
engh = v.WEngineV12(cell["tf_input"]); rech = engh.run(**small)
xh = np.asarray(rech["tolerance"]["xbest"]); dph = engh.direct_point_interval(xh, rech["tolerance"]["tau"])
ch = [{p: dph["iv"]["lo"] * dph["u"][p] for p in engh.pairs}, {p: dph["iv"]["hi"] * dph["u"][p] for p in engh.pairs}]
branches = [l["box_branch"] for l in rech["local"]]
record("accepted_hull_branch_seed_available", "accepted_hull" in branches and contained(rech, ch) and rech["witness_check"]["pass"],
       branches=branches, xbest_in_box=[l["xbest_in_box"] for l in rech["local"]], n_acc=rech["n_accepted_total"]["x1"],
       direct_audit=rech["direct_audit"], runtime_s=time.time() - t0)
# every stored witness reproduces its ordinate and every ledger curve is inside the envelope
pid = max(max(abs(rech["envelope"][eng.pairs[1]]["sup"][k] + 0.5 * rech["envelope"][P1]["inf"][k]),
              abs(rech["envelope"][eng.pairs[1]]["inf"][k] + 0.5 * rech["envelope"][P1]["sup"][k])) for k in range(v.HMAX))
record("pair_linear_identity_engine", pid < 1e-15, max_abs_m=pid)
term_curves = []
for t in rech["refinement_terminals"]:
    if t["feasible"]:
        dpt = engh.direct_point_interval(np.asarray(t["x"]), rech["tolerance"]["tau"])
        term_curves += [{p: dpt["iv"]["lo"] * dpt["u"][p] for p in engh.pairs}, {p: dpt["iv"]["hi"] * dpt["u"][p] for p in engh.pairs}]
record("all_feasible_terminal_curves_contained", contained(rech, term_curves) and len(term_curves) > 0, n_terminal_curves=len(term_curves),
       n_terminals=len(rech["refinement_terminals"]))

res["summary"] = dict(n=len(C), passed=sum(c["passed"] for c in C.values()), expected_failures=[k for k, c in C.items() if c["expected_failure"]],
                      unexpected=[k for k, c in C.items() if not c["passed"]])
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res["summary"]))
