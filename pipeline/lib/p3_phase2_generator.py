"""D03 Phase 2 case generator adapter (B, PREFLIGHT / UNFROZEN).

Everything except the pumping component is the frozen D02 generator (lib/p3_generator.py, imported unchanged):
real KMA rain, natural gamma->reservoir component, AR(1) noise, pilot calendar and seeds, base multipliers, both
schedule pairs, delta. The pumping component uses PHYSICAL strata (lib/p3_phase2_physmap.py): kernel (a, b) and gain
from (T, S, c, r); Q is the matched mean rate in m3/d; short stops and the long rest are placed by a declared template.

Pending gates (official generation refuses to run until they are set and the Phase 2 protocol hash is frozen):
  matching   "M1" (equal outside-rest mean, equal volume within rho) or "M2" (equal volume, equal mean within rho)
             -> author answer pending (results/phase2/B_phase0.md section 4)
  count_rule "N_intervals" (N short stops) or "N_switches" (N/2 short stops) -> pending (section 5)
Fixture mode (synthetic rain from lib/tests/p3_fixture.py) is for toy sanity runs only.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import p3_generator as g  # frozen
from p3_kernels import hantush_block
from p3_phase2_physmap import Stratum, describe

CTX, HMAX, REST_END = g.CTX, g.HMAX, g.REST_END
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")


@dataclass(frozen=True)
class Phase2Design:
    strata: tuple
    rest_ratios: tuple = (0.25, 1.0, 4.0)
    n_levels: tuple = (2, 6, 18)
    count_rule: str | None = None  # "N_intervals" | "N_switches"  (pending)
    matching: str | None = None  # "M1" | "M2"  (pending author)
    ell_frac: float = 0.1  # short-stop length = max(1, round(ell_frac * t95)) days (candidate)
    Qbar: float = 100.0  # m3/d, matched rate (meaning depends on matching option)
    base: g.Design = field(default_factory=g.Design)


def stratum_kernel(st: Stratum) -> dict:
    d = describe(st)
    return dict(a=d["a"], b=d["b"], A=d["gain"], t95=d["t95"], t50=d["t50"], pi_r=d["pi_r"], lam=d["lam"], sid=st.sid)


def n_stops(N: int, rule: str) -> int:
    if rule == "N_intervals":
        return N
    if rule == "N_switches":
        if N % 2:
            raise ValueError("N_switches needs even N")
        return N // 2
    raise ValueError(f"count_rule pending/unknown: {rule}")


def stop_positions(t95: float, n: int, ell: int, rmax: int) -> list[int]:
    """Greedy isolated stops, all on intervals >= ceil(t95) (so only designed stops are short),
    avoiding [REST_END - rmax, REST_END) so positions are identical across rho. Raises if infeasible (no surrogate)."""
    g_on = math.ceil(t95)
    pos = []
    for lo, hi in ((0, REST_END - rmax), (REST_END, CTX)):
        s = lo + g_on
        while s + ell + g_on <= hi and len(pos) < n:
            pos.append(s)
            s += ell + g_on
    if len(pos) < n:
        raise ValueError(f"infeasible: {n} isolated stops of {ell} d with on-gaps >= {g_on} d (capacity {len(pos)})")
    return pos


def nested_positions(t95, ns, rule, ell, rmax):
    """Nested subsets: positions(N_small) subset positions(N_large), evenly spread over the N_max list."""
    nmax = n_stops(max(ns), rule)
    full = stop_positions(t95, nmax, ell, rmax)
    out = {}
    for N in ns:
        k = n_stops(N, rule)
        idx = sorted({int(round(x)) for x in np.linspace(0, nmax - 1, k)}) if k > 1 else [nmax // 2]
        out[N] = [full[i] for i in idx]
    order = sorted(ns)
    for s, l in zip(order, order[1:]):
        assert set(out[s]) <= set(out[l]), "nesting violated"
    return out


def interval_census(Q: np.ndarray, t95: float) -> dict:
    """Run lengths of on (Q>0) and off (Q==0) intervals inside the context; open-ended first/last runs flagged."""
    on = Q > 0
    runs, i = [], 0
    while i < on.size:
        j = i
        while j < on.size and on[j] == on[i]:
            j += 1
        runs.append((bool(on[i]), i, j - i))
        i = j
    short = [(s, st, L) for s, st, L in runs[1:-1] if L < t95]  # interior runs only (first/last are truncated by the window)
    rate_steps = int(np.sum((Q[1:] != Q[:-1]) & (Q[1:] > 0) & (Q[:-1] > 0)))
    return dict(n_short_off=sum(not s for s, _, _ in short), n_short_on=sum(s for s, _, _ in short), n_short_total=len(short),
                n_onoff_switches=int(np.sum(on[1:] != on[:-1])), n_positive_rate_steps=rate_steps, runs=len(runs))


def build_case(bg: dict, st: Stratum, rho: float, N: int, d: Phase2Design) -> dict:
    if d.matching not in ("M1", "M2") or d.count_rule not in ("N_intervals", "N_switches"):
        raise PermissionError("matching / count_rule pending: official cells cannot be built (fixture callers must set them explicitly)")
    kp = stratum_kernel(st)
    t95, A = kp["t95"], kp["A"]
    R = int(round(rho * t95))
    rmax = int(round(max(d.rest_ratios) * t95))
    ell = max(1, int(round(d.ell_frac * t95)))
    pos = nested_positions(t95, d.n_levels, d.count_rule, ell, rmax)[N]
    on = np.ones(CTX, bool)
    on[REST_END - R:REST_END] = False
    for p in pos:
        on[p:p + ell] = False
    shape = bg["mult"] * on
    outside = np.ones(CTX, bool); outside[REST_END - R:REST_END] = False
    kappa = d.Qbar * outside.sum() / shape[outside].sum() if d.matching == "M1" else d.Qbar * CTX / shape.sum()
    Qw = kappa * shape
    q0 = Qw[CTX - 1]
    assert q0 > 0 and np.all(Qw[REST_END - R:REST_END] == 0)
    B, S = hantush_block(kp["a"], kp["b"], CTX + HMAX)
    B, S = B[0], S[0]
    Qpre = d.Qbar  # hidden constant pre-context operation (generator only)
    fut = {PAIRS[0]: (np.full(HMAX, q0), np.zeros(HMAX)), PAIRS[1]: (np.full(HMAX, q0), np.full(HMAX, 1.5 * q0))}

    def h_pump(Qfut):
        Q = np.concatenate([Qw, Qfut])
        return A * (Qpre * (1.0 - S[1:CTX + HMAX + 1]) + np.convolve(Q, B)[:CTX + HMAX])

    w = slice(g.GEN_WARMUP, g.GEN_WARMUP + CTX + HMAX)
    h_nat, eps = bg["h_nat"][w], bg["eps"][w]
    hp_a = h_pump(fut[PAIRS[0]][0])
    h_star_a = h_nat - hp_a
    E_true, delta = {}, {}
    for p, (qa, qb) in fut.items():
        ha, hb = h_nat - h_pump(qa), h_nat - h_pump(qb)
        assert np.array_equal(ha[:CTX], h_star_a[:CTX]) and np.array_equal(hb[:CTX], h_star_a[:CTX])
        E_true[p] = (ha - hb)[CTX:]
        delta[p] = np.maximum(d.base.delta_rel * np.abs(E_true[p]), d.base.sigma_eps)
    real = bg["real"]
    cid = f"r{real['realization']:02d}_{st.sid}_rho{rho:g}_N{N}"
    cen = interval_census(Qw, t95)
    derived = dict(sid=st.sid, storage_type=st.storage_type, T=st.T, S=st.S, c=st.c, r=st.r, Qbar=d.Qbar, a=kp["a"], b=kp["b"], gain=A,
                   t50=kp["t50"], t95=t95, pi_r=kp["pi_r"], lam=kp["lam"], R_days=R, rho_nominal=rho, rho_realized=R / t95, N_nominal=N,
                   ell_days=ell, stop_starts=pos, kappa=float(kappa), V=float(Qw.sum()), qbar_out=float(Qw[outside].mean()), q0=float(q0),
                   SR_out=float(A * Qw[outside].mean() / bg["sigma_bg"]), SR_ctx=float(A * Qw.mean() / bg["sigma_bg"]), sigma_bg=bg["sigma_bg"],
                   rest_is_short_interval=bool(R < t95), lead_over_t95={k: k / t95 for k in (10, 30)}, census=cen)
    tf_input = dict(case_id=cid, head_context=(h_star_a + eps)[:CTX], rain=bg["P"][g.GEN_WARMUP:], pumping_context=Qw,
                    future_Q={f"{p}_{s}": q for p, (qa, qb) in fut.items() for s, q in (("a", qa), ("b", qb))},
                    dates=np.array([str(x.date()) for x in bg["dates"][g.GEN_WARMUP:]]), meta=dict(case_id=cid, sid=st.sid, rho=rho, N=N))
    truth = dict(case_id=cid, h_star_a=h_star_a, E_true=E_true, delta=delta,
                 theta_true=dict(a=kp["a"], b=kp["b"], A=A, Q_pre=Qpre, q0=q0, Q_start=float(Qw[0]), eta_pump_true=float(A * (Qpre - Qw[0])),
                                 eta_nat_true=0.0, nat_gain=bg["nat_gain"], **bg["nat"]))
    return dict(case_id=cid, tf_input=tf_input, truth=truth, derived=derived)


def require_frozen_phase2(protocol_path: Path, sha_path: Path) -> str:
    if not protocol_path.exists() or not sha_path.exists():
        raise PermissionError("Phase 2 protocol not frozen: official case generation is not authorised")
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if sha_path.read_text().split()[0] != digest:
        raise PermissionError("Phase 2 protocol hash mismatch")
    return digest
