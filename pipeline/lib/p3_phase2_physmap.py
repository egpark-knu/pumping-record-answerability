"""D03 Phase 2 (PREFLIGHT, UNFROZEN): physical aquifer coordinates -> exact Hantush (a, b, gain) of the frozen family.

Physical inputs (units: m, d, m3/d):
    T  transmissivity (m2/d), S storage coefficient (confined/leaky) or specific yield Sy (unconfined, linearised),
    c  vertical resistance (d): aquitard resistance for confined/leaky, drainage resistance for the linearised unconfined
       case (declared interpretation, pending author/root, see results/phase2/B_phase0.md section 2),
    r  observation distance (m), Q pumping rate (m3/d).
Map (Hantush & Jacob 1955; Pastas (a, b) form, derivation in results/pilot/protocol.md section 5.1):
    lambda = sqrt(T c), a = c S, b = r^2 / (4 T c), t_r = a b = r^2 S / (4 T), pi_r = t_r / (1 d),
    gain A = K0(r / lambda) / (2 pi T)  [m per m3/d], s(r, t) = Q A S_unit(t) = Q / (4 pi T) W(u, r/lambda), u = r^2 S / (4 T t).
The unit step comes from the FROZEN lib/p3_kernels.py (imported unchanged). Response times use exact root finding
(brentq on the frozen step), never the Pastas Lambert-W approximation.
No truth scoring, no case generation: design quantities only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import exp1, k0

from p3_kernels import hantush_unit_step

STORAGE = {"confined": 1e-4, "leaky": 1e-3, "unconfined": 0.1}  # D03 section 3 values (S, S, Sy)


@dataclass(frozen=True)
class Stratum:
    storage_type: str
    S: float
    T: float
    c: float
    r: float
    Q: float

    @property
    def sid(self) -> str:
        return f"{self.storage_type}_T{self.T:g}"


def hantush_params(T, S, c, r):
    lam = np.sqrt(T * c)
    a = c * S
    b = r * r / (4.0 * T * c)
    return dict(lam=float(lam), a=float(a), b=float(b), t_r=float(a * b), pi_r=float(a * b), r_over_lambda=float(r / lam),
                gain=float(k0(r / lam) / (2.0 * np.pi * T)))


def unit_step(a, b, t):
    return hantush_unit_step(a, b, np.atleast_1d(np.asarray(t, float)))[0]


def t_frac(a, b, frac, tmax=1e5):
    f = lambda t: unit_step(a, b, t)[0] - frac  # noqa: E731
    lo = 1e-6
    if f(tmax) < 0:
        return float("nan")
    return float(np.exp(brentq(lambda lt: f(np.exp(lt)), np.log(lo), np.log(tmax), xtol=1e-12)))


def describe(st: Stratum) -> dict:
    h = hantush_params(st.T, st.S, st.c, st.r)
    out = dict(asdict(st), sid=st.sid, **h)
    for q in (0.5, 0.95, 0.99):
        out[f"t{int(q * 100)}"] = t_frac(h["a"], h["b"], q)
    out["steady_drawdown_m"] = st.Q * h["gain"]
    out["unit_step_at_1_10_30_d"] = [float(v) for v in unit_step(h["a"], h["b"], [1.0, 10.0, 30.0])]
    return out


# ---------------------------------------------------------------- independent unit fixtures
def hantush_W(u, rho):
    """Leaky well function W(u, rho) = int_u^inf y^-1 exp(-y - rho^2/(4y)) dy by adaptive quadrature (independent path)."""
    f = lambda y: np.exp(-y - rho * rho / (4.0 * y)) / y  # noqa: E731
    v1, _ = quad(f, u, max(u, 1.0), limit=400, epsabs=0, epsrel=1e-12)
    v2, _ = quad(f, max(u, 1.0), np.inf, limit=400, epsabs=0, epsrel=1e-12)
    return v1 + v2


def fixture_checks(st: Stratum, times=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
    """Compare Q*gain*S_unit(t) (frozen kernel) with Q/(4 pi T) W(u, r/lambda) (direct quad), and the steady state."""
    h = hantush_params(st.T, st.S, st.c, st.r)
    rows = []
    for t in times:
        u = st.r ** 2 * st.S / (4.0 * st.T * t)
        s_direct = st.Q / (4.0 * np.pi * st.T) * hantush_W(u, h["r_over_lambda"])
        s_kernel = st.Q * h["gain"] * unit_step(h["a"], h["b"], t)[0]
        rows.append(dict(t_d=t, u=u, s_direct_m=s_direct, s_kernel_m=s_kernel, abs_err_m=abs(s_kernel - s_direct),
                         rel_err_of_steady=abs(s_kernel - s_direct) / (st.Q * h["gain"])))
    s_inf_direct = st.Q / (2.0 * np.pi * st.T) * k0(h["r_over_lambda"])  # W(0, rho) = 2 K0(rho)
    w0 = hantush_W(1e-14, h["r_over_lambda"])
    return dict(sid=st.sid, rows=rows, max_rel_err_of_steady=max(r["rel_err_of_steady"] for r in rows),
                steady_direct_m=s_inf_direct, steady_gain_m=st.Q * h["gain"], W0_over_2K0=w0 / (2 * k0(h["r_over_lambda"])))


def theis_limit_check(T=50.0, S=1e-4, r=100.0, Q=100.0, t=(0.01, 0.1, 1.0)):
    """Very large c: Hantush -> Theis (Q/(4 pi T) E1(u)) at early time (units sanity)."""
    out = []
    for tt in t:
        u = r * r * S / (4 * T * tt)
        out.append(dict(t_d=tt, theis_m=Q / (4 * np.pi * T) * exp1(u), hantush_c1e9_m=Q / (4 * np.pi * T) * hantush_W(u, r / np.sqrt(T * 1e9))))
    return out
