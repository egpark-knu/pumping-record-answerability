"""W(C,k) envelope, repaired bookkeeping v1.2 (D02 pilot, B). Addendum: results/pilot/protocol_W_repair_addendum_v1_2.md.

The frozen module `p3_wenvelope` (protocol cf0cf607...) is imported UNCHANGED. Physics, family F, window, bounds,
tau formula and the frozen per-case tau, the Sobol designs and box rules, the 8-round / 4-restart caps, the 5% rule
and the refinement leads {10, k*} are all inherited. Changes (bookkeeping and numerical path only):

 1. Best-fit seed. When the direct (unpenalised, original-X, bound-constrained) SSE at xbest is <= tau, the direct
    best-fit curve E = A_direct u(k) is merged into the envelope of every tolerance multiplier, separately from the
    two endpoint curves of the direct-X profile interval at xbest. A_direct must lie inside [lo, hi] up to the bracket
    resolution. The seed is not an accepted sample (counts, boxes and sparse_G are unaffected).
 2. Full-curve merging. Every feasible refinement or restart terminal (feasible at the original tau under the
    direct-X oracle), including terminals that tie or lose on the targeted lead, contributes BOTH endpoint curves at
    all 30 leads and both pairs. Infeasible terminals and the penalised surrogate `_soft` values are never merged.
 3. Coherent witnesses. Every envelope ordinate keeps the (x, A, linear coefficients, source) that attains it. A
    strictly better value replaces value and witness together; a tie keeps the previous witness.
 4. Direct-X oracle. Feasible A-intervals of seeds, terminals and all envelope witnesses are recomputed on the
    original design X (column-scaled only, no ridge) by BVLS profiling and bracketed root finding: the returned
    endpoint is feasible (f <= tau) and a probe one bracket step outside is infeasible unless the endpoint is a bound.
    The published envelope is the envelope of these direct-certified full curves (sampled Gram witnesses are a search
    device only). Gram-vs-direct discrepancies are recorded, never absorbed into tau.
No generator truth enters any function in this module.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import brentq, lsq_linear, minimize

import p3_wenvelope as base
from p3_wenvelope import (CONV_TOL, EXCESS_MULTS, HMAX, IA, LEVELS, MAX_RESTARTS, NAT_BOX, NLIN, PUMP_BOX, T95_MAX)

VERSION = "v1.2"
SSE_REL_ALLOW = 1e-10   # roundoff allowance on SSE, relative to tau; used only when min f lies in (tau, tau(1+allow)]
A_REL_TOL = 1e-12       # bracket resolution of A endpoints, relative to A_max (no SSE->A conversion is used)
E_AUDIT_REL = 1e-6      # Gram-search vs direct-certified ordinate difference above this x max|E| of the envelope is "material"
BOX = np.vstack([NAT_BOX, PUMP_BOX])


# ---------------------------------------------------------------- direct-X profile oracle (standalone, testable)
def _scaled(X, lb, ub):
    d = np.linalg.norm(X, axis=0)
    d = np.where(d > 0, d, 1.0)
    return X / d, lb * d, ub * d, d


def direct_profile(X, y, lb, ub, ia=IA):
    """f(A) = min over the other coefficients within [lb, ub] of ||y - X[:,ia] A - X_rest beta||^2 (no penalty).
    Returns f(A) -> (sse, full coefficient vector), and the bounded joint minimum (sse, coefficients)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xs, lbs, ubs, d = _scaled(X, np.asarray(lb, float), np.asarray(ub, float))
    rest = [c for c in range(X.shape[1]) if c != ia]

    def f(A):
        r = lsq_linear(Xs[:, rest], y - X[:, ia] * A, bounds=(lbs[rest], ubs[rest]), method="bvls")
        th = np.empty(X.shape[1]); th[ia] = A; th[rest] = r.x / d[rest]
        th[rest] = np.clip(th[rest], np.asarray(lb, float)[rest], np.asarray(ub, float)[rest])
        res = y - X @ th
        return float(res @ res), th
    full = lsq_linear(Xs, y, bounds=(lbs, ubs), method="bvls")
    Astar = float(np.clip(full.x[ia] / d[ia], lb[ia], ub[ia]))
    return f, Astar


def direct_interval(X, y, lb, ub, tau, ia=IA, a_tol=None):
    """Feasible A-interval {A in [lb_A, ub_A] : f(A) <= tau} of the convex direct profile.
    Returns None (with reason in a dict) or a dict with certified endpoints, their coefficients/SSE and outside probes."""
    lbA, ubA = float(lb[ia]), float(ub[ia])
    a_tol = a_tol if a_tol is not None else A_REL_TOL * max(ubA - lbA, 1e-300)
    f, Astar = direct_profile(X, y, lb, ub, ia)
    fmin, thmin = f(Astar)
    n_eval = [1]

    def g(A):
        n_eval[0] += 1
        return f(A)[0] - tau
    info = dict(Astar=Astar, fmin=fmin, tau=float(tau))
    if fmin > tau * (1 + SSE_REL_ALLOW):
        return dict(ok=False, reason="min_f_above_tau", **info)
    if fmin > tau:  # singleton at the minimum within the declared SSE roundoff allowance
        return dict(ok=True, lo=Astar, hi=Astar, th_lo=thmin, th_hi=thmin, sse_lo=fmin, sse_hi=fmin, clip_lo=False, clip_hi=False,
                    out_lo=None, out_hi=None, singleton_within_allow=True, n_eval=n_eval[0], **info)

    def side(bound, sign):
        fb, thb = f(bound); n_eval[0] += 1
        if fb <= tau:
            return bound, thb, fb, True, None
        r = brentq(g, *sorted((bound, Astar)), xtol=a_tol, rtol=1e-15, maxiter=200)
        a_in, step = r, a_tol  # move inward until certified feasible, outward until certified infeasible
        fin, thin = f(a_in); n_eval[0] += 1
        while fin > tau:
            a_in = a_in - sign * step
            if (a_in - Astar) * sign <= 0:
                a_in = Astar
            fin, thin = f(a_in); n_eval[0] += 1; step *= 2
            if a_in == Astar:
                break
        a_out, step = a_in, a_tol
        while True:
            a_out = a_out + sign * step
            if (a_out - bound) * sign >= 0:
                a_out = bound
            fo, _ = f(a_out); n_eval[0] += 1
            if fo > tau or a_out == bound:
                break
            step *= 2
        return a_in, thin, fin, False, dict(A=a_out, sse=fo, gap=abs(a_out - a_in))
    lo, th_lo, s_lo, c_lo, o_lo = side(lbA, -1.0)
    hi, th_hi, s_hi, c_hi, o_hi = side(ubA, +1.0)
    return dict(ok=True, lo=lo, hi=hi, th_lo=th_lo, th_hi=th_hi, sse_lo=s_lo, sse_hi=s_hi, clip_lo=c_lo, clip_hi=c_hi,
                out_lo=o_lo, out_hi=o_hi, singleton_within_allow=False, n_eval=n_eval[0], **info)


# ---------------------------------------------------------------- witness-carrying envelope
def new_env(pairs):
    return {p: dict(sup=np.full(HMAX, -np.inf), inf=np.full(HMAX, np.inf), wsup=[None] * HMAX, winf=[None] * HMAX) for p in pairs}


def merge_value(e, k, v, wit, side):
    """Strict improvement replaces value and witness together; ties keep the previous witness."""
    if side == "sup" and v > e["sup"][k]:
        e["sup"][k] = v; e["wsup"][k] = wit; return 1
    if side == "inf" and v < e["inf"][k]:
        e["inf"][k] = v; e["winf"][k] = wit; return 1
    return 0


def merge_curve(env, curves, wit):
    """Merge one full response curve (dict pair -> [HMAX]) at every lead, both sides."""
    n = 0
    for p, c in curves.items():
        for k in range(HMAX):
            n += merge_value(env[p], k, float(c[k]), wit, "sup")
            n += merge_value(env[p], k, float(c[k]), wit, "inf")
    return n


def env_ok(env, p):
    return bool(np.all(np.isfinite(env[p]["sup"])) and np.all(np.isfinite(env[p]["inf"])))


def summ(env, p):
    W = env[p]["sup"] - env[p]["inf"]
    return float(W[9]), float(W[:10].max())


class WEngineV12(base.WEngine):
    """Frozen engine + v1.2 bookkeeping. All frozen methods (scan, sobol_stage, envelope, best_fit, tolerance, _soft)
    are inherited unchanged."""

    # ------------------------------------------------------------ direct-X oracle at a nonlinear point
    def direct_point_interval(self, x, tau):
        sse, th, X, u, t95 = self.point(np.clip(np.asarray(x, float), BOX[:, 0], BOX[:, 1]))
        lb, ub = self._lin_bounds()
        out = dict(x=np.clip(np.asarray(x, float), BOX[:, 0], BOX[:, 1]), sse_point=sse, th_point=th, u=u, t95=t95, X=X)
        if t95 > T95_MAX:
            out["iv"] = dict(ok=False, reason="t95_above_max"); return out
        out["iv"] = direct_interval(X, self.yc, lb, ub, tau, a_tol=A_REL_TOL * self.Amax)
        return out

    def endpoint_curves(self, dp, src):
        """Two certified endpoint curves (A = lo, A = hi) at all leads/pairs with coherent witnesses."""
        iv = dp["iv"]; res = []
        for end in ("lo", "hi"):
            A = float(iv[end])
            wit = dict(x=dp["x"].tolist(), A=A, lin=iv["th_" + end].tolist(), sse=float(iv["sse_" + end]), src=f"{src}:{end}", certified=True)
            res.append(({p: A * dp["u"][p] for p in self.pairs}, wit))
        return res

    def _merge_sampled(self, env, st_env, src):
        """Frozen sampled Gram envelope (values + attaining nonlinear x) -> witness env (uncertified, search only)."""
        for p in self.pairs:
            if st_env[p] is None:
                continue
            for k in range(HMAX):
                merge_value(env[p], k, float(st_env[p]["sup"][k]), dict(x=np.asarray(st_env[p]["xsup"][k]).tolist(), src=src, certified=False), "sup")
                merge_value(env[p], k, float(st_env[p]["inf"][k]), dict(x=np.asarray(st_env[p]["xinf"][k]).tolist(), src=src, certified=False), "inf")

    # ------------------------------------------------------------ refinement with terminal capture
    def refine_v12(self, env_p1, tau, k, start_vals=None, starts=None):
        """Frozen refine() objective/Powell settings; the terminal of each side is admitted by the direct-X oracle and
        returned for full-curve merging whether or not it improves the targeted ordinate."""
        P1 = self.pairs[0]; out = {}; terms = []
        for side in ("sup", "inf"):
            sgn = -1.0 if side == "sup" else 1.0
            if starts is None:
                x0 = np.asarray((env_p1["wsup"] if side == "sup" else env_p1["winf"])[k - 1]["x"], float)
                bestv = float(env_p1[side][k - 1])
            else:
                x0 = np.asarray(starts[side], float); bestv = float(start_vals[side])
            bestx = x0
            scale = 100.0 * (abs(bestv) + 1e-6)

            def obj(x):
                sse, t95, e = self._soft(np.clip(x, BOX[:, 0], BOX[:, 1]), tau, P1, k)
                val = max(e) if side == "sup" else min(e)
                pen = scale * max(0.0, sse / tau - 1.0) + scale * max(0.0, t95 / T95_MAX - 1.0)
                return sgn * val + pen
            r = minimize(obj, x0, method="Powell", bounds=BOX, options=dict(maxfev=600, xtol=1e-5, ftol=1e-12))
            xr = np.clip(r.x, BOX[:, 0], BOX[:, 1])
            dp = self.direct_point_interval(xr, tau)
            feas = bool(dp["iv"].get("ok"))
            terms.append(dict(side=side, k=k, dp=dp, feasible=feas))
            if feas:
                uk = dp["u"][P1][k - 1]
                e = (dp["iv"]["lo"] * uk, dp["iv"]["hi"] * uk)
                v = max(e) if side == "sup" else min(e)
                if (side == "sup" and v > bestv) or (side == "inf" and v < bestv):
                    bestv, bestx = v, xr
            out[side] = bestv; out["x" + side] = bestx
        return out, terms

    def merge_terminals(self, env, terms, ledger, log):
        """Full-curve merge of every feasible terminal (ties/losses on the target included); infeasible ones are only logged."""
        for t in terms:
            dp = t["dp"]
            ent = dict(k=t["k"], side=t["side"], x=dp["x"].tolist(), feasible=t["feasible"], sse_point=float(dp["sse_point"]))
            if t["feasible"]:
                nchg = 0
                for cd, wit in self.endpoint_curves(dp, f"refine_k{t['k']}_{t['side']}"):
                    nchg += merge_curve(env, cd, wit); ledger.append((cd, wit))
                ent.update(lo=float(dp["iv"]["lo"]), hi=float(dp["iv"]["hi"]), n_ordinates_changed=nchg)
            else:
                ent["reason"] = dp["iv"].get("reason")
            log.append(ent)

    # ------------------------------------------------------------ main
    def run(self, global_levels=LEVELS, n_local=256, max_local_rounds=8, min_accepted=200, refine=True, cached=None, _scan_hook=None):
        t0 = time.time()
        rec = {"version": VERSION, "flags": [], "global": {}, "local": []}
        scan = _scan_hook or self.scan
        top = max(global_levels)
        full = self.sobol_stage(top, top, NAT_BOX, PUMP_BOX, self.seed)
        sse_s, best, _ = self.scan(full)
        starts = [np.concatenate([full.xn[i], full.xp[j]]) for _, i, j in best]
        sse_o, xbest = self.best_fit(starts[:20])
        sse_min = min(sse_s, sse_o)
        if not np.isfinite(sse_min):
            rec["flags"].append("W_failed"); rec["W_status"] = "W_failed"; rec["runtime_s"] = time.time() - t0; return rec
        xbest = starts[0] if (xbest is None or sse_s < sse_o) else xbest
        taus = {}
        for m in EXCESS_MULTS:
            taus[m], info = self.tolerance(sse_min, xbest, m)
            if m == 1.0:
                rec["tolerance"] = dict(sse_min=sse_min, sse_min_sampled=sse_s, sse_min_optimizer=sse_o, tau=float(taus[m]),
                                        rmse_equiv_m=float(np.sqrt(taus[m] / base.CTX)), rmse_min_m=float(np.sqrt(sse_min / base.CTX)),
                                        xbest=list(map(float, xbest)), **info)
        # cached frozen tau/xbest (original run): reproduce, then lock the cached values
        if cached is not None:
            rep = dict(tau_recomputed=float(taus[1.0]), tau_cached=float(cached["tau"]), sse_min_recomputed=float(sse_min),
                       sse_min_cached=float(cached["sse_min"]), xbest_maxabs_diff=float(np.max(np.abs(np.asarray(cached["xbest"]) - xbest))))
            rep["identical"] = bool(rep["tau_recomputed"] == rep["tau_cached"] and rep["sse_min_recomputed"] == rep["sse_min_cached"] and rep["xbest_maxabs_diff"] == 0.0)
            rec["cached_fit_check"] = rep
            xbest = np.asarray(cached["xbest"], float); sse_min = float(cached["sse_min"])
            for m in EXCESS_MULTS:
                taus[m], info = self.tolerance(sse_min, xbest, m)
            taus[1.0] = float(cached["tau"])
            rec["tolerance"] = dict(cached["tolerance_record"], tau=float(cached["tau"]), locked_from="original_run_W_record")
        P1 = self.pairs[0]
        ledger = []  # certified full curves (main tau)
        # ---------------- best-fit seed (every multiplier), direct point separate from profile endpoints
        cum = {m: new_env(self.pairs) for m in EXCESS_MULTS}
        seed = {}
        for m in EXCESS_MULTS:
            dp = self.direct_point_interval(xbest, taus[m])
            s = dict(tau=float(taus[m]), sse_direct=float(dp["sse_point"]), A_direct=float(dp["th_point"][IA]), t95=float(dp["t95"]),
                     direct_feasible=bool(dp["sse_point"] <= taus[m] and dp["t95"] <= T95_MAX))
            if s["direct_feasible"]:
                wit = dict(x=dp["x"].tolist(), A=s["A_direct"], lin=dp["th_point"].tolist(), sse=s["sse_direct"], src="bestfit_direct", certified=True)
                cd = {p: s["A_direct"] * dp["u"][p] for p in self.pairs}
                merge_curve(cum[m], cd, wit)
                if m == 1.0:
                    ledger.append((cd, wit))
                iv = dp["iv"]
                s["interval_ok"] = bool(iv.get("ok"))
                if iv.get("ok"):
                    s.update(lo=float(iv["lo"]), hi=float(iv["hi"]), clip_lo=iv["clip_lo"], clip_hi=iv["clip_hi"],
                             out_lo=iv["out_lo"], out_hi=iv["out_hi"], sse_lo=float(iv["sse_lo"]), sse_hi=float(iv["sse_hi"]),
                             A_direct_contained=bool(iv["lo"] - 2 * A_REL_TOL * self.Amax <= s["A_direct"] <= iv["hi"] + 2 * A_REL_TOL * self.Amax))
                    for cd2, w2 in self.endpoint_curves(dp, "bestfit"):
                        merge_curve(cum[m], cd2, w2)
                        if m == 1.0:
                            ledger.append((cd2, w2))
                else:
                    s["interval_reason"] = iv.get("reason"); s["fmin"] = iv.get("fmin")
                # legacy Gram interval at the same tau (audit only)
                g = base.WEngine.point_interval(self, xbest, taus[m])
                s["legacy_gram_interval"] = None if g is None else [float(g[0]), float(g[1])]
                if g is not None and iv.get("ok"):
                    s["legacy_vs_direct_A_maxabs"] = float(max(abs(g[0] - iv["lo"]), abs(g[1] - iv["hi"])))
                    s["legacy_excludes_A_direct"] = bool(not (g[0] <= s["A_direct"] <= g[1]))
            seed[f"x{m:g}"] = s
        rec["bestfit_seed"] = seed
        s1 = seed["x1"]
        if not s1["direct_feasible"]:
            rec["flags"].append("bestfit_direct_infeasible"); rec["W_status"] = "cached_center_invalid" if cached else "W_failed"
            rec["runtime_s"] = time.time() - t0; return rec
        if not s1.get("interval_ok") or not s1.get("A_direct_contained"):
            rec["flags"].append("seed_interval_unresolved")
        # ---------------- global stage (frozen designs); top level merged
        acc_pts = np.empty((0, 5)); nacc = {m: 0 for m in EXCESS_MULTS}
        for N in sorted(global_levels):
            st = full.subset(N, N)
            _, _, envs = scan(st, [taus[m] for m in EXCESS_MULTS])
            lv = {"n_nat": N, "n_pump": st.Pp.shape[0], "pump_drawn_top": full.pump_drawn}
            for m, e in zip(EXCESS_MULTS, envs):
                env = self.envelope(st, e)
                lv[f"x{m:g}_n_accepted"] = e["nacc"]; lv[f"x{m:g}_n_bound_active_exact"] = e["n_bn"]
                lv[f"x{m:g}_W10_Wmax"] = None if env[P1] is None else self._summ(env, P1)
                if N == top:
                    if env[P1] is not None:
                        self._merge_sampled(cum[m], env, f"global{N}")
                    nacc[m] = e["nacc"]
                    if m == 1.0:
                        acc_pts = e["acc_pts"]
            rec["global"][N] = lv
        # ---------------- local stage: frozen box rule (seed is not an accepted point)
        prev = summ(cum[1.0], P1)
        box_all = BOX
        conv = False
        for r in range(max_local_rounds):
            branch = "accepted_hull" if acc_pts.shape[0] >= 5 else "best_starts_hull"
            pts = acc_pts if acc_pts.shape[0] >= 5 else np.vstack([acc_pts] + [s[None] for s in starts[:20]] + [np.asarray(xbest)[None]])
            lo, hi = pts.min(0), pts.max(0)
            pad = np.maximum(0.25 * (hi - lo), 0.02 * (box_all[:, 1] - box_all[:, 0]))
            bx = np.column_stack([np.maximum(lo - pad, box_all[:, 0]), np.minimum(hi + pad, box_all[:, 1])])
            st = self.sobol_stage(n_local, n_local, bx[:3], bx[3:], self.seed + 1000 * (r + 1))
            _, _, envs = scan(st, [taus[m] for m in EXCESS_MULTS])
            lr = {"round": r + 1, "box": bx.tolist(), "n_pump": st.Pp.shape[0], "box_branch": branch,
                  "xbest_in_box": bool(np.all((bx[:, 0] <= xbest) & (xbest <= bx[:, 1])))}
            for m, e in zip(EXCESS_MULTS, envs):
                env = self.envelope(st, e)
                if env[P1] is not None:
                    self._merge_sampled(cum[m], env, f"local{r + 1}")
                nacc[m] += e["nacc"]
                lr[f"x{m:g}_n_accepted"] = e["nacc"]
                if m == 1.0 and e["acc_pts"].shape[0]:
                    acc_pts = np.vstack([acc_pts, e["acc_pts"]])
            cur = summ(cum[1.0], P1)
            lr["cum_W10_Wmax"] = cur; lr["cum_n_accepted"] = nacc[1.0]
            rel = [abs(cur[i] - prev[i]) / max(abs(cur[i]), 1e-12) for i in range(2)]
            lr["rel_change"] = rel
            if max(rel) <= CONV_TOL and nacc[1.0] >= min_accepted:
                conv = True
            rec["local"].append(lr); prev = cur
            if conv:
                break
        env = cum[1.0]
        if nacc[1.0] < min_accepted:
            rec["flags"].append("sparse_G")
        # ---------------- refinement at {10, k*} (k* from the pre-refinement envelope); full-curve terminal merging
        gains, restarts, term_log = {}, {}, []

        def take_terms(terms):
            self.merge_terminals(env, terms, ledger, term_log)
        if refine:
            W = env[P1]["sup"] - env[P1]["inf"]
            kstar = int(np.argmax(W[:10])) + 1
            rec["kstar"] = kstar
            for k in sorted({10, kstar}):
                W_old = float(env[P1]["sup"][k - 1] - env[P1]["inf"][k - 1])
                ref, terms = self.refine_v12(env[P1], taus[1.0], k)
                take_terms(terms)
                seq = [float((ref["sup"] - ref["inf"] - W_old) / max(W_old, 1e-12))]
                for _ in range(MAX_RESTARTS):
                    prevW = ref["sup"] - ref["inf"]
                    ref2, terms = self.refine_v12(env[P1], taus[1.0], k, start_vals=dict(sup=ref["sup"], inf=ref["inf"]),
                                                  starts=dict(sup=ref["xsup"], inf=ref["xinf"]))
                    take_terms(terms)
                    # coherent chain: value and coordinate move together (ties keep the earlier coordinate)
                    nsup = ref2["sup"] > ref["sup"]; ninf = ref2["inf"] < ref["inf"]
                    ref = dict(sup=ref2["sup"] if nsup else ref["sup"], xsup=ref2["xsup"] if nsup else ref["xsup"],
                               inf=ref2["inf"] if ninf else ref["inf"], xinf=ref2["xinf"] if ninf else ref["xinf"])
                    inc = (ref["sup"] - ref["inf"] - prevW) / max(prevW, 1e-12)
                    seq.append(float(inc))
                    if inc <= CONV_TOL:
                        break
                gains[k] = seq[0]; restarts[k] = seq
            rec["refinement_first_gain_over_sampled"] = gains
            rec["refinement_increments"] = restarts
        rec["refinement_terminals"] = term_log
        last_ok = all(v[-1] <= CONV_TOL for v in restarts.values()) if restarts else True
        rec["refined_leads"] = sorted(restarts)
        rec["converged"] = bool(conv and last_ok)
        if not rec["converged"]:
            rec["flags"].append("not_converged")
        # ---------------- direct-X certification of every search witness; published envelope = certified curves
        search = {p: dict(sup=env[p]["sup"].copy(), inf=env[p]["inf"].copy()) for p in self.pairs}
        pend = {}
        for p in self.pairs:
            for side in ("wsup", "winf"):
                for wt in env[p][side]:
                    if wt is not None and not wt.get("certified"):
                        pend.setdefault(tuple(np.round(wt["x"], 15)), wt["x"])
        n_uncert = 0; uncert_reasons = {}
        for key, x in pend.items():
            dp = self.direct_point_interval(x, taus[1.0])
            if dp["iv"].get("ok"):
                for cd, wit in self.endpoint_curves(dp, "sampled_witness"):
                    ledger.append((cd, wit))
            else:
                n_uncert += 1; rs = dp["iv"].get("reason"); uncert_reasons[rs] = uncert_reasons.get(rs, 0) + 1
        cert = new_env(self.pairs)
        for cd, wit in ledger:
            merge_curve(cert, cd, wit)
        audit = dict(n_search_witnesses=len(pend), n_uncertified=n_uncert, uncertified_reasons=uncert_reasons, n_ledger_curves=len(ledger))
        dmax_over, dmax_under = 0.0, 0.0
        escale = max(float(np.max(np.abs(np.concatenate([cert[p]["sup"], cert[p]["inf"]])))) for p in self.pairs)
        for p in self.pairs:
            over = np.concatenate([search[p]["sup"] - cert[p]["sup"], cert[p]["inf"] - search[p]["inf"]])
            over = over[np.isfinite(over)]
            if over.size:
                dmax_over = max(dmax_over, float(over.max())); dmax_under = max(dmax_under, float((-over).max()))
        audit.update(search_minus_certified_max_m=dmax_over, certified_minus_search_max_m=dmax_under,
                     scale_m=escale, allow_m=E_AUDIT_REL * escale,
                     material=bool(max(dmax_over, dmax_under) > E_AUDIT_REL * escale))
        rec["direct_audit"] = audit
        if n_uncert:
            rec["flags"].append("witness_uncertified")
        if audit["material"]:
            rec["flags"].append("direct_audit_material")
        rec["search_envelope"] = {p: dict(sup=search[p]["sup"].tolist(), inf=search[p]["inf"].tolist()) for p in self.pairs}
        env = cert
        rec["W_status"] = "ok" if all(env_ok(env, p) for p in self.pairs) and "seed_interval_unresolved" not in rec["flags"] else "unresolved"
        rec["witness_check"] = self.check_witnesses(env, taus[1.0])
        rec["n_accepted_total"] = {f"x{m:g}": nacc[m] for m in EXCESS_MULTS}
        rec["envelope"] = {p: dict(sup=env[p]["sup"].tolist(), inf=env[p]["inf"].tolist(), W=(env[p]["sup"] - env[p]["inf"]).tolist()) for p in self.pairs}
        rec["witnesses"] = {p: dict(sup=env[p]["wsup"], inf=env[p]["winf"]) for p in self.pairs}
        rec["summary"] = {p: dict(zip(("W10", "Wmax1to10"), summ(env, p)), W30=float(env[p]["sup"][29] - env[p]["inf"][29]),
                                  Wmax1to30=float((env[p]["sup"] - env[p]["inf"]).max())) for p in self.pairs}
        rec["diagnostic_tolerance"] = {f"x{m:g}": dict(zip(("W10", "Wmax1to10"), summ(cum[m], P1)), note="search envelope incl. seed; not refined or certified")
                                       for m in EXCESS_MULTS if m != 1.0}
        rec["runtime_s"] = time.time() - t0
        return rec

    def check_witnesses(self, env, tau):
        """Reconstruct every ordinate from its stored witness: direct residual, bounds, finite memory, value."""
        lb, ub = self._lin_bounds()
        worst = dict(sse_over_tau_max=-np.inf, bound_violation_max=0.0, value_mismatch_max=0.0, t95_max=0.0, n_checked=0, n_missing=0)
        cache = {}
        for p in self.pairs:
            for side, ws in (("sup", env[p]["wsup"]), ("inf", env[p]["winf"])):
                for k, wt in enumerate(ws):
                    if wt is None:
                        worst["n_missing"] += 1; continue
                    key = tuple(wt["x"])
                    if key not in cache:
                        _, _, X, u, t95 = self.point(np.asarray(wt["x"], float)); cache[key] = (X, u, t95)
                    X, u, t95 = cache[key]
                    th = np.asarray(wt["lin"], float)
                    res = self.yc - X @ th
                    worst["sse_over_tau_max"] = max(worst["sse_over_tau_max"], float(res @ res) / tau - 1.0)
                    worst["bound_violation_max"] = max(worst["bound_violation_max"], float(np.max(np.maximum(lb - th, 0) + np.maximum(th - ub, 0))))
                    worst["value_mismatch_max"] = max(worst["value_mismatch_max"], abs(float(th[IA] * u[p][k]) - float(env[p][side][k])))
                    worst["t95_max"] = max(worst["t95_max"], float(t95))
                    worst["n_checked"] += 1
        worst["pass"] = bool(worst["n_missing"] == 0 and worst["sse_over_tau_max"] <= SSE_REL_ALLOW and worst["bound_violation_max"] == 0.0
                             and worst["value_mismatch_max"] <= 1e-15 and worst["t95_max"] <= T95_MAX)
        return worst
