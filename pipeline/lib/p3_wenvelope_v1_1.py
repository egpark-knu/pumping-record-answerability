"""CANDIDATE v1.1 (NOT FROZEN; for root decision). Changes vs frozen p3_wenvelope.py (cf0cf607 protocol):
(a) the best fit (always in G when the fit succeeds) seeds the envelope with its exact bounded A-interval;
(b) with <5 accepted design points the local box is centred on the best fit (pad 2% of the global range,
    doubled after each round with no new acceptance, cap 8 rounds) instead of the hull of the 20 best
    sampled points. Everything else identical.

W(C,k) envelope for the D02 pilot: declared finite-memory Hantush family plus
gamma-reservoir natural component, fitted ONLY on the 1024-day observed window.

Family F (protocol_draft §6.5, revised for the same-window rule):
  h_g[n] = c0 + beta_nat * [(P * K_nat)[n] + P_bar (1 - Kc_nat[n+1])] + eta_nat * exp(-(n+1)/tau)
           - A * [(Q * B_pump)[n] + Q_start (1 - S_pump[n+1])] - eta_pump * (1 - S_pump[n+1]),   n = 0..1023
  declared boundary from the observed window only: P_bar = context-mean rain, Q_start = first observed rate.
  nonlinear x = (n_shape, log10 theta, log10 tau, log10 a, log10 b); linear = (A, eta_pump, c0, beta_nat, eta_nat)
  eta_*: unknown initial state (pre-context pumping departure, reservoir offset), |eta| <= 10 range(h_obs);
  bounds enforced exactly: ellipsoid endpoints where the profile point is inside the bounds, otherwise the exact
  convex profile f(A) = min_{other linear in bounds} SSE (BVLS on the Gram Cholesky factor) with bisection;
  A in [0, A_max], A_max = 10 range(h_obs) / median(positive Q); beta_nat >= 0.
  finite memory: t95(a, b) <= 1000 d.
E_g(k) = -A * (dQ_future * B_pump)[k-1]  (natural part, noise and history cancel).
G(C): feasible SSE <= tau_C = SSE_min [1 + p/(n_eff - p) F_0.95(p, n_eff - p)], p = 10 (heuristic fit-discrepancy rule,
  not an exact coverage statement for nonlinear AR(1) fitting).
No generator truth enters any function in this module except `truth_compatibility`.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import brentq, lsq_linear, minimize, minimize_scalar
from scipy.stats import f as f_dist, qmc

from p3_kernels import hantush_block, natural_kernel, causal_conv

CTX, HMAX = 1024, 30
NAT_BOX = np.array([[0.5, 5.0], [0.0, 2.0], [np.log10(5.0), np.log10(500.0)]])
PUMP_BOX = np.array([[0.0, 3.5], [-4.0, np.log10(25.0)]])
T95_MAX = 1000.0
P_FREE = 10
NLIN = 5
IA, IPP, IC0, INAT, IN1 = range(NLIN)
ETA = (IPP, IN1)
LEVELS = (128, 256, 512)
FALLBACK_LEVEL = 1024
CONV_TOL = 0.05
MAX_RESTARTS = 4
EXCESS_MULTS = (0.5, 1.0, 2.0)  # 1.0 primary; 0.5 and 2.0 diagnostic only


# ---------------------------------------------------------------- columns
def nat_columns(xn, P, Pbar):
    """xn [M,3] -> columns [M, CTX, 2]: response with steady P_bar prehistory, reservoir-offset nuisance."""
    n_shape, theta, tau = xn[:, 0], 10 ** xn[:, 1], 10 ** xn[:, 2]
    K, Kc = natural_kernel(n_shape, theta, tau, CTX)
    resp = causal_conv(P[:CTX], K) + Pbar * (1.0 - Kc[:, 1:CTX + 1])
    lvl = np.exp(-np.arange(1, CTX + 1)[None, :] / tau[:, None])
    return np.stack([resp, lvl], axis=2)


def t95_from_step(S):
    """S [M, L+1] unit step at t=0..L -> t95 (inf if not reached)."""
    out = np.full(S.shape[0], np.inf)
    hit = S >= 0.95
    ok = hit.any(axis=1)
    k = np.argmax(hit, axis=1)
    for i in np.where(ok)[0]:
        kk = k[i]
        out[i] = kk - 1 + (0.95 - S[i, kk - 1]) / (S[i, kk] - S[i, kk - 1]) if kk > 0 else 0.0
    return out


def pump_columns(xp, Qw, future_dQ):
    """xp [M,2] -> columns [M, CTX, 2] (-(Q*B + Q_start tail), -tail), unit E responses u[pair] [M, HMAX], t95 [M]."""
    a, b = 10 ** xp[:, 0], 10 ** xp[:, 1]
    B, S = hantush_block(a, b, CTX + HMAX)
    t95 = t95_from_step(S)
    tail = 1.0 - S[:, 1:CTX + 1]
    resp = -(causal_conv(Qw, B) + Qw[0] * tail)
    u = {p: -causal_conv(dq, B)[:, :HMAX] for p, dq in future_dQ.items()}
    return np.stack([resp, -tail], axis=2), u, t95


# ---------------------------------------------------------------- batched linear algebra
def _solve_batch(G, bvec, yy, drop=None):
    """Least squares on Gram blocks. G [N,6,6], b [N,6]. drop: column index fixed to 0.
    Returns theta [N,NLIN], Sigma_AA [N], Sigma[:,:,IA] [N,NLIN], SSE [N]."""
    idx = [c for c in range(NLIN) if c != drop]
    Gs = G[:, idx][:, :, idx]
    bs = bvec[:, idx]
    d = np.sqrt(np.maximum(np.einsum("nii->ni", Gs), 1e-300))
    Gn = Gs / (d[:, :, None] * d[:, None, :]) + 1e-12 * np.eye(len(idx))[None]
    inv = np.linalg.inv(Gn) / (d[:, :, None] * d[:, None, :])
    th = np.einsum("nij,nj->ni", inv, bs)
    sse = yy - np.einsum("ni,ni->n", th, bs)
    theta = np.zeros((G.shape[0], NLIN))
    theta[:, idx] = th
    colA = np.zeros((G.shape[0], NLIN))
    colA[:, idx] = inv[:, :, idx.index(IA)]
    return theta, colA[:, IA], colA, np.maximum(sse, 0.0)


def _intervals(theta, sAA, colA, sse, tau, Amax):
    """A-interval for SSE <= tau with A in [0, Amax]; beta_nat evaluated at both ends."""
    r = np.sqrt(np.maximum(tau - sse, 0.0) * np.maximum(sAA, 0.0))
    lo = np.maximum(theta[:, IA] - r, 0.0)
    hi = np.minimum(theta[:, IA] + r, Amax)
    ok = (sse <= tau) & (lo <= hi)
    slope = colA[:, INAT] / np.where(sAA > 0, sAA, np.inf)
    bn_lo = theta[:, INAT] + (lo - theta[:, IA]) * slope
    bn_hi = theta[:, INAT] + (hi - theta[:, IA]) * slope
    return lo, hi, ok, bn_lo, bn_hi


def _feasible_sse(theta, sAA, colA, sse, Amax):
    Ac = np.clip(theta[:, IA], 0.0, Amax)
    s = sse + (Ac - theta[:, IA]) ** 2 / np.where(sAA > 0, sAA, np.inf)
    bn = theta[:, INAT] + (Ac - theta[:, IA]) * colA[:, INAT] / np.where(sAA > 0, sAA, np.inf)
    return s, bn


class Stage:
    """A product design: nat points xn [Nn,3] x pump points xp [Np,2] with cached columns."""

    def __init__(self, xn, xp, P, Qw, dQ, Pbar):
        self.xn = xn
        cols = nat_columns(xn, P, Pbar)
        self.Mn = np.concatenate([np.ones((xn.shape[0], CTX, 1)), cols], axis=2)
        cp, u, t95 = pump_columns(xp, Qw, dQ)
        keep = t95 <= T95_MAX
        self.n_pump_excluded_t95 = int((~keep).sum())
        self.xp, self.Pp, self.t95 = xp[keep], cp[keep], t95[keep]
        self.u = {p: v[keep] for p, v in u.items()}

    def subset(self, Nn, Np):
        s = object.__new__(Stage)
        s.xn, s.Mn = self.xn[:Nn], self.Mn[:Nn]
        s.xp, s.Pp, s.t95 = self.xp[:Np], self.Pp[:Np], self.t95[:Np]
        s.u = {p: v[:Np] for p, v in self.u.items()}
        s.n_pump_excluded_t95 = self.n_pump_excluded_t95
        return s


class WEngine:
    def __init__(self, tf_input, seed=20260930, amax_factor=10.0):
        self.y = np.asarray(tf_input["head_context"], float)
        self.yc = self.y - self.y.mean()
        self.yy = float(self.yc @ self.yc)
        self.P = np.asarray(tf_input["rain"], float)
        self.Qw = np.asarray(tf_input["pumping_context"], float)
        fq = tf_input["future_Q"]
        pairs = sorted({k.rsplit("_", 1)[0] for k in fq})
        self.dQ = {p: fq[f"{p}_a"] - fq[f"{p}_b"] for p in pairs}
        self.pairs = pairs
        self.Pbar = float(self.P[:CTX].mean())
        pos = self.Qw[self.Qw > 0]
        if pos.size == 0 or np.ptp(self.y) == 0:
            raise ValueError("input/design failure: no positive pumping or constant head in the window")
        self.Amax = amax_factor * np.ptp(self.y) / float(np.median(pos))  # observed-only family bound
        self.eta_bound = amax_factor * np.ptp(self.y)
        self.seed = seed

    # ------------------------------------------------------------ designs
    def sobol_stage(self, Nn, Np, nat_box, pump_box, seed):
        xn = qmc.scale(qmc.Sobol(3, scramble=True, seed=seed).random(Nn), nat_box[:, 0], nat_box[:, 1])
        # draw pump points until Np satisfy the finite-memory constraint (sequence order kept -> nested)
        sq = qmc.Sobol(2, scramble=True, seed=seed + 1)
        xs = []; n_valid = 0; drawn = 0
        while n_valid < Np:
            m = max(Np - n_valid, 64)
            c = qmc.scale(sq.random(m), pump_box[:, 0], pump_box[:, 1]); drawn += m
            _, _, t95 = pump_columns(c, self.Qw, {k: v[:1] for k, v in self.dQ.items()})
            c = c[t95 <= T95_MAX]
            xs.append(c); n_valid += c.shape[0]
            if drawn > 50 * Np:
                break
        xp = np.vstack(xs)[:Np]
        st = Stage(xn, xp, self.P, self.Qw, self.dQ, self.Pbar)
        st.pump_drawn = drawn
        return st

    def _chunks(self, st, chunk=64):
        Mn, Pp = st.Mn, st.Pp
        Nn, Np = Mn.shape[0], Pp.shape[0]
        Gnn = np.einsum("inp,inq->ipq", Mn, Mn); bn = np.einsum("inp,n->ip", Mn, self.yc)
        Gpp = np.einsum("jnp,jnq->jpq", Pp, Pp); bp = np.einsum("jnp,n->jp", Pp, self.yc)
        Pflat = Pp.transpose(1, 0, 2).reshape(CTX, Np * 2)
        for i0 in range(0, Nn, chunk):
            i1 = min(Nn, i0 + chunk); ni = i1 - i0
            C = (Mn[i0:i1].transpose(0, 2, 1).reshape(ni * 3, CTX) @ Pflat).reshape(ni, 3, Np, 2).transpose(0, 2, 1, 3)
            G = np.zeros((ni, Np, NLIN, NLIN))
            G[:, :, :2, :2] = Gpp[None]
            G[:, :, 2:, 2:] = Gnn[i0:i1, None]
            G[:, :, 2:, :2] = C
            G[:, :, :2, 2:] = C.transpose(0, 1, 3, 2)
            b = np.zeros((ni, Np, NLIN)); b[:, :, :2] = bp[None]; b[:, :, 2:] = bn[i0:i1, None]
            yield i0, ni, Np, G.reshape(-1, NLIN, NLIN), b.reshape(-1, NLIN)

    # ------------------------------------------------------------ exact bounded linear feasibility
    def _lin_bounds(self):
        eb = self.eta_bound
        return (np.array([0.0, -eb, -np.inf, 0.0, -eb]), np.array([self.Amax, eb, np.inf, np.inf, eb]))

    def _profile(self, G, b):
        """Returns f(A) = min over the other linear parameters within bounds of SSE(A, .) (exact, BVLS in
        Gram/Cholesky form) and the bounded minimum SSE with its A."""
        d = np.sqrt(np.maximum(np.diag(G), 1e-300))
        Gs = G / np.outer(d, d) + 1e-12 * np.eye(NLIN)
        R = np.linalg.cholesky(Gs).T                     # Gs = R^T R
        z = np.linalg.solve(R.T, b / d)                  # SSE = yy - |z|^2 + |R th' - z|^2, th' = d * th
        base = self.yy - float(z @ z)
        lb, ub = self._lin_bounds()
        lbs, ubs = lb * d, ub * d
        rest = [c for c in range(NLIN) if c != IA]

        def f(A):
            r = lsq_linear(R[:, rest], z - R[:, IA] * (A * d[IA]), bounds=(lbs[rest], ubs[rest]), method="bvls")
            return base + float(np.sum(r.fun ** 2))
        full = lsq_linear(R, z, bounds=(lbs, ubs), method="bvls")
        return f, base + float(np.sum(full.fun ** 2)), float(full.x[IA] / d[IA])

    def exact_interval(self, G, b, tau):
        """Exact feasible A-interval {A : f(A) <= tau} (f convex); None if empty."""
        f, smin, Astar = self._profile(G, b)
        if smin > tau:
            return None
        lo = 0.0 if f(0.0) <= tau else brentq(lambda A: f(A) - tau, 0.0, Astar, xtol=1e-14 * max(1.0, Astar), rtol=1e-12)
        hi = self.Amax if f(self.Amax) <= tau else brentq(lambda A: f(A) - tau, Astar, self.Amax, xtol=1e-14 * max(1.0, Astar), rtol=1e-12)
        return lo, hi

    def bounded_sse(self, G, b):
        return self._profile(G, b)[1]

    def _endpoint_violation(self, full, lo, hi, ok):
        """Unconstrained-ellipsoid endpoints are exact iff their profile points satisfy the parameter bounds."""
        th0, sAA0, colA0 = full[0], full[1], full[2]
        sl = colA0 / np.where(sAA0 > 0, sAA0, np.inf)[:, None]
        bad = np.zeros(th0.shape[0], bool)
        for Aend in (lo, hi):
            thE = th0 + np.where(ok, Aend - th0[:, IA], 0.0)[:, None] * sl
            bad |= (thE[:, INAT] < 0) | (np.abs(thE[:, list(ETA)]) > self.eta_bound).any(axis=1)
        return ok & bad

    def scan(self, st, taus=()):
        """Product scan. Returns feasible-SSE minimum, 20 best combos, and per tau the A-interval
        union per pump sample (with the nat index achieving each end) and accepted combo coordinates."""
        Np = st.Pp.shape[0]
        sse_min = np.inf; best = []
        envs = [dict(lo=np.full(Np, np.inf), hi=np.full(Np, -np.inf), ilo=np.zeros(Np, int), ihi=np.zeros(Np, int),
                     nacc=0, n_bn=0, acc_pts=[]) for _ in taus]
        for i0, ni, _, G, b in self._chunks(st):
            full = _solve_batch(G, b, self.yy)
            red = _solve_batch(G, b, self.yy, drop=INAT)
            sf, bnf = _feasible_sse(*full, self.Amax)
            sr, _ = _feasible_sse(*red, self.Amax)
            sfeas = np.where(bnf >= 0, sf, sr)  # ranking only; the 20 best are re-evaluated exactly
            for t in np.argsort(sfeas)[:20]:
                ex = self.bounded_sse(G[t], b[t])
                best.append((ex, i0 + t // Np, t % Np))
                sse_min = min(sse_min, ex)
            for e, tau in zip(envs, taus):
                lo, hi, ok, _, _ = _intervals(*full, tau, self.Amax)
                bad = self._endpoint_violation(full, lo, hi, ok)
                lo_f = np.where(ok, lo, np.inf); hi_f = np.where(ok, hi, -np.inf)
                for t in np.where(bad)[0]:  # bound-active: exact convex profile
                    iv = self.exact_interval(G[t], b[t], tau)
                    lo_f[t], hi_f[t] = (iv if iv is not None else (np.inf, -np.inf))
                e["n_bn"] += int(bad.sum())
                acc = np.isfinite(lo_f) & np.isfinite(hi_f) & (lo_f <= hi_f)
                lo_m = np.where(acc, lo_f, np.inf).reshape(ni, Np); hi_m = np.where(acc, hi_f, -np.inf).reshape(ni, Np)
                a_lo, a_hi = lo_m.argmin(0), hi_m.argmax(0)
                v_lo, v_hi = lo_m[a_lo, np.arange(Np)], hi_m[a_hi, np.arange(Np)]
                blo, bhi = v_lo < e["lo"], v_hi > e["hi"]
                e["lo"][blo] = v_lo[blo]; e["ilo"][blo] = i0 + a_lo[blo]
                e["hi"][bhi] = v_hi[bhi]; e["ihi"][bhi] = i0 + a_hi[bhi]
                e["nacc"] += int(acc.sum())
                ia = np.where(acc)[0]
                if ia.size:
                    e["acc_pts"].append(np.hstack([st.xn[i0 + ia // Np], st.xp[ia % Np]]))
        best.sort()
        for e in envs:
            e["acc_pts"] = np.vstack(e["acc_pts"]) if e["acc_pts"] else np.empty((0, 5))
        return sse_min, best[:20], envs

    @staticmethod
    def envelope(st, e):
        """E-envelope over one stage: E = A u(k), A in [lo_j, hi_j] per pump sample j."""
        valid = np.isfinite(e["lo"]) & np.isfinite(e["hi"])
        out = {}
        for p, u in st.u.items():
            if not valid.any():
                out[p] = None; continue
            j = np.where(valid)[0]
            e1 = e["lo"][j, None] * u[j]; e2 = e["hi"][j, None] * u[j]
            mx, mn = np.maximum(e1, e2), np.minimum(e1, e2)
            ks, ki = mx.argmax(0), mn.argmin(0)
            # coordinates of the extreme combos (nat index giving the relevant A end)
            def coord(jj, kk, which):
                A_is_lo = (e1[jj, kk] >= e2[jj, kk]) if which == "sup" else (e1[jj, kk] <= e2[jj, kk])
                i_nat = e["ilo"][j[jj]] if A_is_lo else e["ihi"][j[jj]]
                return np.concatenate([st.xn[i_nat], st.xp[j[jj]]])
            out[p] = dict(sup=mx.max(0), inf=mn.min(0),
                          xsup=[coord(ks[k], k, "sup") for k in range(HMAX)], xinf=[coord(ki[k], k, "inf") for k in range(HMAX)])
        return out

    @staticmethod
    def merge(a, b):
        if a is None:
            return b
        out = {}
        for p in a:
            if a[p] is None or b[p] is None:
                out[p] = a[p] or b[p]; continue
            s_b = b[p]["sup"] > a[p]["sup"]; i_b = b[p]["inf"] < a[p]["inf"]
            out[p] = dict(sup=np.where(s_b, b[p]["sup"], a[p]["sup"]), inf=np.where(i_b, b[p]["inf"], a[p]["inf"]),
                          xsup=[b[p]["xsup"][k] if s_b[k] else a[p]["xsup"][k] for k in range(HMAX)],
                          xinf=[b[p]["xinf"][k] if i_b[k] else a[p]["xinf"][k] for k in range(HMAX)])
        return out

    # ------------------------------------------------------------ continuous point
    def point(self, x):
        xn, xp = x[None, :3], x[None, 3:]
        cols_p, u, t95 = pump_columns(xp, self.Qw, self.dQ)
        X = np.concatenate([cols_p[0], np.ones((CTX, 1)), nat_columns(xn, self.P, self.Pbar)[0]], axis=1)
        eb = self.eta_bound
        lb = np.array([0, -eb, -np.inf, 0, -eb]); ub = np.array([self.Amax, eb, np.inf, np.inf, eb])
        res = lsq_linear(X, self.yc, bounds=(lb, ub), method="bvls")
        return float(np.sum(res.fun ** 2)), res.x, X, {p: v[0] for p, v in u.items()}, float(t95[0])

    def point_interval(self, x, tau):
        sse, th, X, u, t95 = self.point(x)
        if t95 > T95_MAX or sse > tau:
            return None
        iv = self.exact_interval(X.T @ X, X.T @ self.yc, tau)
        return None if iv is None else (iv[0], iv[1], u)

    def best_fit(self, starts):
        box = np.vstack([NAT_BOX, PUMP_BOX])
        best = (np.inf, None)
        for x0 in starts:
            def fun(x):
                sse, _, _, _, t95 = self.point(np.clip(x, box[:, 0], box[:, 1]))
                return sse if t95 <= T95_MAX else 1e12
            r = minimize(fun, x0, method="Powell", bounds=box, options=dict(maxfev=400, xtol=1e-3, ftol=1e-9))
            if r.fun < best[0]:
                best = (float(r.fun), np.clip(r.x, box[:, 0], box[:, 1]))
        return best

    def tolerance(self, sse_min, xbest, excess_mult=1.0):
        _, th, X, _, _ = self.point(xbest)
        res = self.yc - X @ th
        r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        neff = float(np.clip(CTX * (1 - r1) / (1 + r1), P_FREE + 10, CTX))
        excess = P_FREE / (neff - P_FREE) * f_dist.ppf(0.95, P_FREE, neff - P_FREE)
        return sse_min * (1 + excess_mult * excess), dict(r1=r1, n_eff=neff, excess=excess)

    def _soft(self, x, tau, pair, k):
        """Continuous surrogate for refinement: exact bounded A-interval at max(tau, bounded SSE) and the SSE."""
        sse, th, X, u, t95 = self.point(x)
        iv = self.exact_interval(X.T @ X, X.T @ self.yc, max(tau, sse * (1 + 1e-12)))
        L, H = iv if iv is not None else (th[IA], th[IA])
        uk = u[pair][k - 1]
        return sse, t95, (L * uk, H * uk)

    def refine(self, env_p, tau, pair, k, starts=None):
        """Powell on continuous x from the extreme design points: extend sup and inf of E(k).
        Continuous penalty outside G; a refined value is kept only if its end point is feasible."""
        box = np.vstack([NAT_BOX, PUMP_BOX])
        out = {}
        for side in ("sup", "inf"):
            sgn = -1.0 if side == "sup" else 1.0
            x0 = np.asarray(env_p["x" + side][k - 1]) if starts is None else np.asarray(starts[side])
            bestv = float(env_p[side][k - 1]); bestx = x0
            scale = 100.0 * (abs(bestv) + 1e-6)

            def obj(x):
                sse, t95, e = self._soft(np.clip(x, box[:, 0], box[:, 1]), tau, pair, k)
                val = max(e) if side == "sup" else min(e)
                pen = scale * max(0.0, sse / tau - 1.0) + scale * max(0.0, t95 / T95_MAX - 1.0)
                return sgn * val + pen
            r = minimize(obj, x0, method="Powell", bounds=box, options=dict(maxfev=600, xtol=1e-5, ftol=1e-12))
            xr = np.clip(r.x, box[:, 0], box[:, 1])
            iv = self.point_interval(xr, tau)
            if iv is not None:
                lo, hi, u = iv
                e = (lo * u[pair][k - 1], hi * u[pair][k - 1])
                v = max(e) if side == "sup" else min(e)
                if (side == "sup" and v > bestv) or (side == "inf" and v < bestv):
                    bestv, bestx = v, xr
            out[side] = bestv; out["x" + side] = bestx
        return out

    @staticmethod
    def _summ(env, pair):
        W = env[pair]["sup"] - env[pair]["inf"]
        return float(W[9]), float(W[:10].max())

    def run(self, global_levels=LEVELS, n_local=256, max_local_rounds=8, min_accepted=200, refine=True):
        t0 = time.time()
        rec = {"flags": [], "global": {}, "local": []}
        top = max(global_levels)
        full = self.sobol_stage(top, top, NAT_BOX, PUMP_BOX, self.seed)
        sse_s, best, _ = self.scan(full)
        starts = [np.concatenate([full.xn[i], full.xp[j]]) for _, i, j in best]
        sse_o, xbest = self.best_fit(starts[:20])
        sse_min = min(sse_s, sse_o)
        if not np.isfinite(sse_min):
            rec["flags"].append("W_failed"); rec["runtime_s"] = time.time() - t0; return rec
        xbest = starts[0] if (xbest is None or sse_s < sse_o) else xbest
        taus = {}
        for m in EXCESS_MULTS:
            taus[m], info = self.tolerance(sse_min, xbest, m)
            if m == 1.0:
                rec["tolerance"] = dict(sse_min=sse_min, sse_min_sampled=sse_s, sse_min_optimizer=sse_o, tau=float(taus[m]),
                                        rmse_equiv_m=float(np.sqrt(taus[m] / CTX)), rmse_min_m=float(np.sqrt(sse_min / CTX)), xbest=list(map(float, xbest)), **info)
        P1 = self.pairs[0]
        cum = {m: None for m in EXCESS_MULTS}; acc_pts = np.empty((0, 5)); nacc = {m: 0 for m in EXCESS_MULTS}
        # (a) best-fit seed: exact bounded interval at xbest for each tolerance
        seed_env = {}
        for m in EXCESS_MULTS:
            iv = self.point_interval(np.asarray(xbest), taus[m])
            if iv is not None:
                lo_, hi_, u_ = iv
                seed_env[m] = {p: dict(sup=np.maximum(lo_ * u_[p], hi_ * u_[p]), inf=np.minimum(lo_ * u_[p], hi_ * u_[p]),
                                       xsup=[np.asarray(xbest)] * HMAX, xinf=[np.asarray(xbest)] * HMAX) for p in self.pairs}
        rec["best_fit_seed_in_G"] = bool(1.0 in seed_env)
        for N in sorted(global_levels):
            st = full.subset(N, N)
            _, _, envs = self.scan(st, [taus[m] for m in EXCESS_MULTS])
            lv = {"n_nat": N, "n_pump": st.Pp.shape[0], "pump_drawn_top": full.pump_drawn}
            for m, e in zip(EXCESS_MULTS, envs):
                env = self.envelope(st, e)
                lv[f"x{m:g}_n_accepted"] = e["nacc"]; lv[f"x{m:g}_n_bound_active_exact"] = e["n_bn"]
                lv[f"x{m:g}_W10_Wmax"] = None if env[P1] is None else self._summ(env, P1)
                if N == top:
                    cum[m] = env if env[P1] is not None else None; nacc[m] = e["nacc"]
                    if m == 1.0:
                        acc_pts = e["acc_pts"]
            rec["global"][N] = lv
        for m in EXCESS_MULTS:
            if m in seed_env:
                cum[m] = self.merge(cum[m], seed_env[m])
        # local stage: product Sobol in the box around accepted points (or around the best fit), merged into the envelope
        prev = None if cum[1.0] is None else self._summ(cum[1.0], P1)
        grow = 1.0
        box_all = np.vstack([NAT_BOX, PUMP_BOX])
        conv = False
        for r in range(max_local_rounds):
            if acc_pts.shape[0] >= 5:
                pts = acc_pts
                lo, hi = pts.min(0), pts.max(0)
                pad = np.maximum(0.25 * (hi - lo), 0.02 * (box_all[:, 1] - box_all[:, 0]))
            else:  # (b) centre on the best fit; widen after empty rounds
                pts = np.vstack([acc_pts, np.asarray(xbest)[None]])
                lo, hi = pts.min(0), pts.max(0)
                pad = np.maximum(0.25 * (hi - lo), 0.02 * grow * (box_all[:, 1] - box_all[:, 0]))
            bx = np.column_stack([np.maximum(lo - pad, box_all[:, 0]), np.minimum(hi + pad, box_all[:, 1])])
            st = self.sobol_stage(n_local, n_local, bx[:3], bx[3:], self.seed + 1000 * (r + 1))
            _, _, envs = self.scan(st, [taus[m] for m in EXCESS_MULTS])
            lr = {"round": r + 1, "box": bx.tolist(), "n_pump": st.Pp.shape[0], "box_mode": "accepted_hull" if acc_pts.shape[0] >= 5 else "best_fit_centred", "grow": grow}
            for m, e in zip(EXCESS_MULTS, envs):
                env = self.envelope(st, e)
                if env[P1] is not None:
                    cum[m] = self.merge(cum[m], env)
                nacc[m] += e["nacc"]
                lr[f"x{m:g}_n_accepted"] = e["nacc"]
                if m == 1.0 and e["acc_pts"].shape[0]:
                    acc_pts = np.vstack([acc_pts, e["acc_pts"]])
                if m == 1.0 and e["nacc"] == 0:
                    grow *= 2.0
            if cum[1.0] is None:
                rec["local"].append(lr); continue
            cur = self._summ(cum[1.0], P1)
            lr["cum_W10_Wmax"] = cur; lr["cum_n_accepted"] = nacc[1.0]
            if prev is not None:
                rel = [abs(cur[i] - prev[i]) / max(abs(cur[i]), 1e-12) for i in range(2)]
                lr["rel_change"] = rel
                if max(rel) <= CONV_TOL and nacc[1.0] >= min_accepted:
                    conv = True
            rec["local"].append(lr); prev = cur
            if conv:
                break
        env = cum[1.0]
        if env is None:
            rec["flags"].append("W_failed"); rec["runtime_s"] = time.time() - t0; return rec
        if nacc[1.0] < min_accepted:
            rec["flags"].append("sparse_G")
        gains, restarts = {}, {}
        if refine:
            W = env[P1]["sup"] - env[P1]["inf"]
            kstar = int(np.argmax(W[:10])) + 1
            for k in sorted({10, kstar}):
                W_old = W[k - 1]
                ref = self.refine(env[P1], taus[1.0], P1, k)
                seq = [float((ref["sup"] - ref["inf"] - W_old) / max(W_old, 1e-12))]
                for _ in range(MAX_RESTARTS):  # restart from own optimum until the increment is <= CONV_TOL
                    prevW = ref["sup"] - ref["inf"]
                    ref2 = self.refine(dict(sup=[ref["sup"]] * HMAX, inf=[ref["inf"]] * HMAX), taus[1.0], P1, k,
                                       starts=dict(sup=ref["xsup"], inf=ref["xinf"]))
                    ref = dict(sup=max(ref["sup"], ref2["sup"]), inf=min(ref["inf"], ref2["inf"]), xsup=ref2["xsup"], xinf=ref2["xinf"])
                    inc = (ref["sup"] - ref["inf"] - prevW) / max(prevW, 1e-12)
                    seq.append(float(inc))
                    if inc <= CONV_TOL:
                        break
                gains[k] = seq[0]; restarts[k] = seq
                for p in self.pairs:  # linear family, constant future dQ: pairs scale by dQ ratio
                    ratio = self.dQ[p][0] / self.dQ[P1][0]
                    assert np.allclose(self.dQ[p], self.dQ[p][0]) and np.allclose(self.dQ[P1], self.dQ[P1][0])
                    s_, i_ = ref["sup"] * ratio, ref["inf"] * ratio
                    env[p]["sup"][k - 1] = max(env[p]["sup"][k - 1], max(s_, i_))
                    env[p]["inf"][k - 1] = min(env[p]["inf"][k - 1], min(s_, i_))
            rec["refinement_first_gain_over_sampled"] = gains
            rec["refinement_increments"] = restarts
        last_ok = all(v[-1] <= CONV_TOL for v in restarts.values()) if restarts else True
        rec["refined_leads"] = sorted(restarts)
        rec["converged"] = bool(conv and last_ok)
        if not rec["converged"]:
            rec["flags"].append("not_converged")
        rec["n_accepted_total"] = {f"x{m:g}": nacc[m] for m in EXCESS_MULTS}
        rec["envelope"] = {p: dict(sup=env[p]["sup"].tolist(), inf=env[p]["inf"].tolist(), W=(env[p]["sup"] - env[p]["inf"]).tolist()) for p in self.pairs}
        rec["summary"] = {p: dict(zip(("W10", "Wmax1to10"), self._summ(env, p)), W30=float(env[p]["sup"][29] - env[p]["inf"][29]),
                                  Wmax1to30=float((env[p]["sup"] - env[p]["inf"]).max())) for p in self.pairs}
        rec["diagnostic_tolerance"] = {f"x{m:g}": (None if cum[m] is None else dict(zip(("W10", "Wmax1to10"), self._summ(cum[m], P1)))) for m in EXCESS_MULTS if m != 1.0}
        rec["runtime_s"] = time.time() - t0
        return rec


def truth_compatibility(engine: WEngine, rec: dict, theta_true: dict, E_true: dict):
    """Validation only: never feeds back into G(C)."""
    x = np.array([theta_true["n"], np.log10(theta_true["theta"]), np.log10(theta_true["tau"]), np.log10(theta_true["a"]), np.log10(theta_true["b"])])
    sse, th, _, _, _ = engine.point(x)
    tau = rec["tolerance"]["tau"]
    inside = {p: [bool(rec["envelope"][p]["inf"][k] - 1e-12 <= E_true[p][k] <= rec["envelope"][p]["sup"][k] + 1e-12) for k in range(HMAX)]
              for p in rec["envelope"]}
    return dict(sse_family_at_true_nonlinear=sse, tau=tau, truth_nonlinear_in_G=bool(sse <= tau), A_profiled_at_truth=float(th[IA]),
                A_true=float(theta_true["A"]), E_true_inside=inside)
