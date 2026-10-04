"""D03 Phase 2 production case generator (B). Implements the root design decision (debate_20260930_191313,
results/phase2/ROOT_DESIGN_DECISION.json) and the author matching definition (results/phase2/LATEST_STEERING.md).

Unchanged from the frozen D02 generator (lib/p3_generator.py, imported, not edited): KMA rain loading with SHA check,
natural gamma->reservoir component, AR(1) noise, pilot calendar/seeds, base multipliers, both schedule pairs, delta.
Changed: the pumping component only.

Physical strata (D1): r = 200 m, nominal Q = 100 m3/d, a* = c S = 20 d (c = a*/S), T in {50, 500} m2/d,
S in {1e-4 confined, 1e-3 leaky, Sy = 0.1 unconfined (Tier A linear)}. Kernel (a, b) = (cS, r^2/(4Tc)), gain
K0(r/sqrt(Tc))/(2 pi T) [m per m3/d], exact step from lib/p3_kernels.py via lib/p3_phase2_physmap.py.

Schedule (per layer l, realization, rho, N), all days are context indices 0..1023:
  t95_l exact; e_l = 1024 - (floor(t95_l) + 1) - 18  (layer rest end, same for all rho, N, realizations)
  R = round(rho * t95_l)  (pilot rule unchanged); designated rest = [e_l - R, e_l)
  short prefix after the rest: ON at e_l, OFF at e_l+1, e_l+3, ..., e_l+N-1 (each 1 day), then ON to day 1023
  base0 = 100 * m (pilot multipliers of the realization, m3/d)
  compensation window C = [0, e_l - R_max), R_max = round(4 t95_l)
  kappa_N = 1 + sum(base0[OFF]) / sum(base0[C]); base_N = base0 with base_N[C] *= kappa_N, base_N[OFF] = 0
  Q = base_N with Q[e_l - R, e_l) = 0 (the ONLY rho-dependent change)
Consequences (checked per case): within (layer, rho, realization) the context volume and the mean over all days except
the designated rest are identical across N; across rho, Q differs only on the rest window (bit identity elsewhere);
q0 = Q[1023] = 100 m[1023] is unchanged across rho and N.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import p3_generator as g  # frozen
from p3_kernels import hantush_block
from p3_phase2_physmap import STORAGE, Stratum, describe

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
P2 = ROOT / "results/phase2"
CTX, HMAX = g.CTX, g.HMAX
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
R_M, Q_NOM, A_STAR = 200.0, 100.0, 20.0
T_LEVELS = (50.0, 500.0)
RHOS = (0.25, 1.0, 4.0)
NS = (2, 6, 18)
BURST_MAX = 18
SITE_SHA = {"안동태화_충적": "20dddf7f15db", "산청산청_암반": "e95466d5ea3b", "남해남해_암반": "7733e1352ddc"}


def strata() -> list[Stratum]:
    return [Stratum(t, S, T, A_STAR / S, R_M, Q_NOM) for t, S in STORAGE.items() for T in T_LEVELS]


_LAYER_CACHE: dict = {}


def layer(st: Stratum) -> dict:
    """Physical layer record: kernel, exact response times, layer rest end, rest lengths (design quantities)."""
    if st.sid not in _LAYER_CACHE:
        d = describe(st)
        t95 = d["t95"]
        e = CTX - (math.floor(t95) + 1) - BURST_MAX
        R = {rho: int(round(rho * t95)) for rho in RHOS}
        B, S = hantush_block(d["a"], d["b"], CTX + HMAX)
        _LAYER_CACHE[st.sid] = dict(st=st, d=d, t95=t95, e=e, R=R, Rmax=R[max(RHOS)], B=B[0], S=S[0])
    return _LAYER_CACHE[st.sid]


def schedule(mult: np.ndarray, L: dict, rho: float, N: int) -> dict:
    """Context pumping for one cell (m3/d) plus the pieces needed to verify the matching contract."""
    e, R, Rmax = L["e"], L["R"][rho], L["Rmax"]
    if N not in NS or N % 2:
        raise ValueError("N must be one of 2, 6, 18")
    if not (0 < e - Rmax and e + BURST_MAX < CTX):
        raise ValueError(f"{L['st'].sid}: layer end / rest do not fit the context")
    base0 = Q_NOM * np.asarray(mult, float)
    off = np.zeros(CTX, bool)
    off[e + 1:e + N:2] = True
    C = np.zeros(CTX, bool)
    C[:e - Rmax] = True
    kappa = 1.0 + base0[off].sum() / base0[C].sum()
    base = base0.copy()
    base[C] *= kappa
    base[off] = 0.0
    q = base.copy()
    q[e - R:e] = 0.0
    return dict(q=q, base_N=base, base0=base0, kappa=float(kappa), off=off, C=C, e=e, R=R, Rmax=Rmax)


def census(q: np.ndarray, t95: float, rest: tuple[int, int]) -> dict:
    """Maximal ON (Q>0) / OFF (Q==0) runs of the actual context series. The designated rest is identified by exact
    identity of its run [rest0, rest1); every other interior run shorter than t95 is a short interval."""
    on = q > 0
    runs, i = [], 0
    while i < q.size:
        j = i
        while j < q.size and on[j] == on[i]:
            j += 1
        runs.append((bool(on[i]), i, j))
        i = j
    rest_runs = [r for r in runs if (not r[0]) and (r[1], r[2]) == rest]
    other_off = [r for r in runs if (not r[0]) and (r[1], r[2]) != rest]
    interior = runs[1:-1]
    short = [r for r in interior if (r[1], r[2]) != rest and (r[2] - r[1]) < t95]
    first, last = runs[0], runs[-1]
    return dict(n_runs=len(runs), designated_rest_identity=len(rest_runs) == 1, n_off_runs_outside_rest=len(other_off),
                N_ON=sum(r[0] for r in short), N_OFF=sum(not r[0] for r in short), N_short=len(short),
                short_lengths=sorted({r[2] - r[1] for r in short}), all_short_1d_lt_t95=all(r[2] - r[1] == 1 < t95 for r in short),
                first_run_on=first[0], first_run_len=first[2] - first[1], tail_on=last[0], tail_len=last[2] - last[1],
                tail_over_t95=(last[2] - last[1]) / t95, rest_len=rest[1] - rest[0], rest_is_shorter_than_t95=(rest[1] - rest[0]) < t95)


def build_case(bg: dict, st: Stratum, rho: float, N: int) -> dict:
    """One matched cell. Returns tf_input/truth in the frozen pilot NPZ layout plus the derived record."""
    L = layer(st)
    sc = schedule(bg["mult"], L, rho, N)
    Qw, t95, A = sc["q"], L["t95"], L["d"]["gain"]
    q0 = float(Qw[CTX - 1])
    if not q0 > 0:
        raise AssertionError("q0 must be positive")
    B, S = L["B"], L["S"]
    Qpre = Q_NOM  # hidden constant pre-context operation (generator only; not seen by any method)
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
        delta[p] = np.maximum(g.Design().delta_rel * np.abs(E_true[p]), g.Design().sigma_eps)
    real = bg["real"]
    cid = f"r{real['realization']:02d}_{st.storage_type}_T{st.T:g}_rho{rho:g}_N{N}"
    rest = (sc["e"] - sc["R"], sc["e"])
    cen = census(Qw, t95, rest)
    outside = np.ones(CTX, bool)
    outside[rest[0]:rest[1]] = False
    V = float(Qw.sum())
    mean_out = float(Qw[outside].mean())
    derived = dict(
        case_id=cid, sid=st.sid, storage_type=st.storage_type, storage_value=st.S, T_m2_d=st.T, r_m=st.r, Q_nominal_m3_d=Q_NOM,
        a_star_d=A_STAR, c_d=st.c, a_d=L["d"]["a"], b=L["d"]["b"], lambda_m=L["d"]["lam"], gain_m_per_m3d=A, t50_d=L["d"]["t50"], t95_d=t95,
        pi_r=L["d"]["pi_r"], layer_rest_end_day=sc["e"], rest_start_day=rest[0], R_days=sc["R"], R_max_days=sc["Rmax"], rho_nominal=rho,
        rho_realized=sc["R"] / t95, N_nominal=N, kappa=sc["kappa"], V_m3=V, mean_outside_rest_m3d=mean_out, mean_context_m3d=V / CTX, q0_m3d=q0,
        calendar_mean_m3d=float(Qw.mean()), base_mult_mean=float(np.mean(bg["mult"])),
        base_N_sha256=hashlib.sha256(np.ascontiguousarray(sc["base_N"]).tobytes()).hexdigest(),
        SR=A * mean_out / bg["sigma_bg"], SR_context_mean=A * (V / CTX) / bg["sigma_bg"], sigma_bg_m=bg["sigma_bg"],
        lead_over_t95_10=10.0 / t95, lead_over_t95_30=30.0 / t95, rest_over_t95=sc["R"] / t95, last_short_event_day=sc["e"] + N - 1,
        tail_on_days=cen["tail_len"], census=cen, realization=int(real["realization"]), site=real["site_stem"],
        units=dict(head="m", pumping="m3/d", rain="mm/d", T="m2/d", r="m", c="d", gain="m per m3/d", time="d"))
    meta = dict(origin_date=str(bg["dates"][g.GEN_WARMUP + CTX].date()), rain_site=real["site_stem"], realization=int(real["realization"]),
                rest_ratio=float(rho), signal_ratio=float(derived["SR"]), pi_layer=float(derived["pi_r"]), head_unit="m", pumping_unit="m3/d",
                rainfall_unit="mm/d", daily_alignment=True, storage_type=st.storage_type, storage_value=float(st.S), T_m2_d=float(st.T),
                r_m=float(st.r), Q_m3_d=float(Q_NOM), transition_count=int(N), t95_days=float(t95))
    tf_input = dict(case_id=cid, head_context=(h_star_a + eps)[:CTX], rain=bg["P"][g.GEN_WARMUP:], pumping_context=Qw,
                    future_Q={f"{p}_{s}": q for p, (qa, qb) in fut.items() for s, q in (("a", qa), ("b", qb))},
                    dates=np.array([str(x.date()) for x in bg["dates"][g.GEN_WARMUP:]]), meta=meta)
    truth = dict(case_id=cid, h_star_a=h_star_a, h_nat=h_nat, h_pump_a=hp_a, eps=eps, E_true=E_true, delta=delta,
                 theta_true=dict(a=L["d"]["a"], b=L["d"]["b"], A=A, Q_pre=Qpre, q0=q0, R_days=float(sc["R"]), Q_start=float(Qw[0]),
                                 eta_pump_true=float(A * (Qpre - Qw[0])), eta_nat_true=0.0, sigma_bg=bg["sigma_bg"], nat_gain=bg["nat_gain"],
                                 **bg["nat"], T=st.T, S=st.S, r=st.r, c=st.c))
    return dict(case_id=cid, meta=meta, tf_input=tf_input, truth=truth, derived=derived, base_N=sc["base_N"])


def realizations(fixture: bool = False) -> tuple[list[dict], dict]:
    """Pilot calendar (official) or a FIXTURE variant that keeps the pilot seeds but uses synthetic rain and a fixed
    2003 start (so no KMA-driven head series exists before the freeze)."""
    reals = json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))
    if not fixture:
        return reals, {s: g.load_rain(s, p) for s, p in SITE_SHA.items()}
    import sys
    sys.path.insert(0, str(ROOT / "lib/tests"))
    from p3_fixture import fixture_rain
    fx = [dict(r, site_stem="FIXTURE_SYNTHETIC", context_start="2003-01-01") for r in reals]
    return fx, {"FIXTURE_SYNTHETIC": fixture_rain()}


def all_cases(fixture: bool = False):
    """Yield the 540 matched cells in a stable order (realization, storage, T, rho, N)."""
    reals, rains = realizations(fixture)
    for real in reals:
        bg = g.realization_background(real, rains[real["site_stem"]], g.Design())
        for st in strata():
            for rho in RHOS:
                for N in NS:
                    yield build_case(bg, st, rho, N)


def require_frozen_phase2() -> str:
    pp, sp = P2 / "protocol.md", P2 / "protocol.sha256"
    if not pp.exists() or not sp.exists():
        raise PermissionError("Phase 2 protocol not frozen: official case generation is not authorised")
    digest = hashlib.sha256(pp.read_bytes()).hexdigest()
    if sp.read_text().split()[0] != digest:
        raise PermissionError("Phase 2 protocol hash mismatch")
    return digest
