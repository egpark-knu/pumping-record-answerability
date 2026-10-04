"""D03 Phase 2 PREFLIGHT (B, UNFROZEN): design-quantity checks only. No official case, no W on KMA cases, no score.

Outputs (results/phase2/B_preflight/):
  unit_fixtures.json       physical map vs independent Hantush W(u, r/lambda) quadrature, steady state, Theis limit
  sigma_bg.json            natural-background SD per pilot realization (frozen generator, read-only rain), vs pilot truth
  stratum_scan.json        (r, a*) scan: a, b, t50/t95/t99, gain, pi_r, box membership, rho rounding, rho=4 fit
  coverage.json            derived SR / pi_r coverage of 1 for candidate (r, Q, a*) using sigma_bg of the 10 realizations
  schedule_feasibility.json  short-transition templates x N x rho: capacity, nested positions, slack, rest-as-transition issue
  matching_identity.json   volume / outside-rest-mean identity and numeric consequences of the two matching options
Usage: env/.venv_pilot/bin/python lib/p3_phase2_preflight.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_generator as g  # noqa: E402  (frozen, imported unchanged)
from p3_phase2_physmap import STORAGE, Stratum, describe, fixture_checks, theis_limit_check  # noqa: E402

OUT = ROOT / "results/phase2/B_preflight"
PILOT = ROOT / "results/pilot"
SITE_SHA = {"안동태화_충적": "20dddf7f15db", "산청산청_암반": "e95466d5ea3b", "남해남해_암반": "7733e1352ddc"}
PUMP_BOX_FROZEN = ((0.0, 3.5), (-4.0, math.log10(25.0)))
L, REST_END, RHOS, NS = g.CTX, g.REST_END, (0.25, 1.0, 4.0), (2, 6, 18)
T_LEVELS = (50.0, 500.0)
R_GRID = (30.0, 50.0, 100.0, 150.0, 200.0)
ASTAR_GRID = (10.0, 20.0, 30.0, 40.0, 60.0)
Q_GRID = (50.0, 100.0, 150.0, 200.0)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=1))


def strata(r, astar, Q=100.0):
    """Declared leakage rule L1 (candidate): common leakage time constant a* = c S across storage types -> c = a*/S."""
    return [Stratum(t, S, T, astar / S, r, Q) for t, S in STORAGE.items() for T in T_LEVELS]


# ---------------------------------------------------------------- 1. unit fixtures
def unit_fixtures():
    rows = [fixture_checks(s) for s in strata(100.0, 20.0)] + [fixture_checks(s) for s in strata(50.0, 40.0)]
    th = theis_limit_check()
    ok = all(r["max_rel_err_of_steady"] < 1e-5 and abs(r["W0_over_2K0"] - 1) < 1e-9 and abs(r["steady_direct_m"] - r["steady_gain_m"]) < 1e-12 * max(1, r["steady_gain_m"])
             for r in rows) and all(abs(x["theis_m"] - x["hantush_c1e9_m"]) / x["theis_m"] < 1e-4 for x in th)
    return dict(all_ok=bool(ok), tolerance="kernel vs direct quad <= 1e-5 of steady drawdown; W(0,rho)=2K0(rho) to 1e-9; Theis limit 1e-4 rel",
                pilot_regression_t95=dict(expected=60.0, note="pilot pi_r=0.5 and 5 kernels, exact t95 via brentq on frozen step"), strata=rows, theis_limit=th)


# ---------------------------------------------------------------- 2. natural background (design quantity)
def sigma_bg():
    reals = json.load(open(PILOT / "phase0_checks/realization_calendar.json"))
    rains = {s: g.load_rain(s, p) for s, p in SITE_SHA.items()}
    d = g.Design()
    out = []
    for real in reals:
        bg = g.realization_background(real, rains[real["site_stem"]], d)
        z = np.load(PILOT / f"cases/truth/{g.case_id(real['realization'], 0.5, 2.0, 1.0)}.npz")
        pilot = json.loads(str(z["theta_true_json"]))["sigma_bg"]
        out.append(dict(realization=real["realization"], site=real["site_stem"], sigma_bg_m=bg["sigma_bg"], pilot_truth_sigma_bg_m=pilot,
                        bit_equal=bool(bg["sigma_bg"] == pilot), nat=bg["nat"], nat_gain=bg["nat_gain"],
                        sd_h_nat_only_m=float(np.std(bg["h_nat"][:L])), sd_eps_m=float(np.std(bg["eps"][:L]))))
    return dict(definition="sigma_bg = SD over the 1,024-d context of h_nat + eps (no pumping), frozen p3_generator.realization_background, pilot calendar/seeds",
                rain_sha256={s: r.attrs["sha256"] for s, r in rains.items()}, rows=out)


# ---------------------------------------------------------------- 3. stratum scan
def stratum_scan():
    rows = []
    for r in R_GRID:
        for astar in ASTAR_GRID:
            for s in strata(r, astar):
                d = describe(s)
                la, lb = math.log10(d["a"]), math.log10(d["b"])
                t95 = d["t95"]
                R = {rho: int(round(rho * t95)) for rho in RHOS}
                d.update(astar=astar, log10_a=la, log10_b=lb,
                         in_frozen_pump_box=bool(PUMP_BOX_FROZEN[0][0] <= la <= PUMP_BOX_FROZEN[0][1] and PUMP_BOX_FROZEN[1][0] <= lb <= PUMP_BOX_FROZEN[1][1]),
                         R_days=R, R_round_relerr={rho: abs(R[rho] - rho * t95) / (rho * t95) for rho in RHOS},
                         rest4_start=REST_END - R[4.0], t95_le_1000=bool(t95 <= 1000.0), brakenhoff_half_window=bool(t95 <= L / 2))
                rows.append(d)
    return dict(leakage_rule="L1 candidate: c = a*/S (common leakage time constant a = cS across storage types); Q=100 m3/d here only scales drawdown",
                frozen_pump_box_log10=PUMP_BOX_FROZEN, rows=rows)


# ---------------------------------------------------------------- 4. SR / pi_r coverage
def coverage(sig):
    sb = np.array([x["sigma_bg_m"] for x in sig["rows"]])
    rows = []
    for r in R_GRID:
        for astar in ASTAR_GRID:
            ds = [describe(s) for s in strata(r, astar)]
            for Q in Q_GRID:
                SR = {d["sid"]: (Q * d["gain"] / sb).tolist() for d in ds}
                med = {k: float(np.median(v)) for k, v in SR.items()}
                allv = np.concatenate([np.array(v) for v in SR.values()])
                pis = {d["sid"]: d["pi_r"] for d in ds}
                rows.append(dict(r=r, astar=astar, Q=Q, SR_case=SR, SR_median_by_stratum=med,
                                 n_strata_SR_median_lt1=sum(v < 1 for v in med.values()), n_strata_SR_median_gt1=sum(v > 1 for v in med.values()),
                                 n_cases_SR_lt1=int((allv < 1).sum()), n_cases_SR_gt1=int((allv > 1).sum()),
                                 pi_r=pis, n_strata_pi_lt1=sum(v < 1 for v in pis.values()), n_strata_pi_gt1=sum(v > 1 for v in pis.values()),
                                 covers_both=bool(any(v < 1 for v in med.values()) and any(v > 1 for v in med.values())
                                                  and any(v < 1 for v in pis.values()) and any(v > 1 for v in pis.values()))))
    return dict(definition="SR_case = gain_stratum * Qbar / sigma_bg_realization (Qbar = matched mean outside-rest pumping rate; see matching gate); "
                           "pi_r = r^2 S / (4 T * 1 d), stratum constant. One case count per (stratum, realization); rho/N do not change SR under option M1.",
                sigma_bg_m=sb.tolist(), rows=rows)


# ---------------------------------------------------------------- 5. short-transition schedule templates
def place_stops(t95, n_stops, ell, rmax):
    """Greedy isolated stops of length ell (days, Q=0), all on-gaps >= ceil(t95) (so no on interval counts as short),
    avoiding the maximal rest span [REST_END - rmax, REST_END) so positions are identical across rho.
    Regions: pre-rest [0, REST_END - rmax), post-rest [REST_END, L). Returns positions (start days) or None."""
    g_on = math.ceil(t95)
    pos = []
    for lo, hi in ((0, REST_END - rmax), (REST_END, L)):
        s = lo + g_on
        while s + ell + g_on <= hi and len(pos) < n_stops:
            pos.append(s)
            s += ell + g_on
    return pos if len(pos) >= n_stops else None, len(pos)


def schedule_feasibility(scan):
    rows = []
    seen = set()
    for d in scan["rows"]:
        key = (round(d["t95"], 6))
        if key in seen:
            continue
        seen.add(key)
        t95 = d["t95"]
        rmax = d["R_days"][4.0]
        for count_rule, n_stops_of in (("N_intervals", lambda N: N), ("N_switches", lambda N: N // 2)):
            for len_rule, ell in (("f0.1", max(1, round(0.1 * t95))), ("f0.25", max(1, round(0.25 * t95))), ("fixed2d", 2)):
                if ell >= t95:
                    rows.append(dict(t95=t95, count_rule=count_rule, len_rule=len_rule, ell=ell, feasible=False, reason="ell >= t95")); continue
                cap_pos, cap = place_stops(t95, 10 ** 6, ell, rmax)
                need = n_stops_of(18)
                pos, _ = place_stops(t95, need, ell, rmax)
                rows.append(dict(t95=t95, R_days=d["R_days"], count_rule=count_rule, len_rule=len_rule, ell=ell, stops_needed_N18=need,
                                 capacity_isolated=cap, feasible=pos is not None, positions_N18=pos,
                                 rest_rho025_is_short_interval=bool(d["R_days"][0.25] < t95),
                                 short_off_days_N18=need * ell))
    return dict(assumptions=dict(context_days=L, rest_end_day=REST_END, rest_positions="[REST_END - R, REST_END) as pilot; stop positions fixed across rho by avoiding the rho=4 rest span",
                                 on_gap_rule="every on interval (incl. before first stop and between stop and rest) >= ceil(t95), so only designed stops count",
                                 origin_rule="context day 1023 is on (q0 > 0)"),
                rows=rows)


# ---------------------------------------------------------------- 6. matching identity
def matching_identity(scan):
    mult = g.base_schedule(json.load(open(PILOT / "phase0_checks/realization_calendar.json"))[0]["seed_schedule"], g.Design())
    rows = []
    for d in scan["rows"]:
        if not (d["r"] == 100.0 and d["astar"] == 20.0):
            continue
        t95 = d["t95"]; ell = max(1, round(0.1 * t95))
        for N in NS:
            for rho in RHOS:
                R = d["R_days"][rho]
                pos, _ = place_stops(t95, N, ell, d["R_days"][4.0])
                if pos is None:
                    rows.append(dict(sid=d["sid"], N=N, rho=rho, feasible=False)); continue
                on = np.ones(L, bool); on[REST_END - R:REST_END] = False
                for p in pos:
                    on[p:p + ell] = False
                base = mult * on
                out = np.ones(L, bool); out[REST_END - R:REST_END] = False
                # M1: mean outside rest = 100 m3/d  |  M2: total volume = 100 * L  (both reach equal V within (stratum, rho) across N)
                k1 = 100.0 * out.sum() / base[out].sum()
                k2 = 100.0 * L / base.sum()
                rows.append(dict(sid=d["sid"], t95=t95, N=N, rho=rho, R=R, ell=ell,
                                 M1=dict(kappa=k1, V=float((k1 * base).sum()), mean_out=float((k1 * base)[out].mean()), q0=float(k1 * base[-1])),
                                 M2=dict(kappa=k2, V=float((k2 * base).sum()), mean_out=float((k2 * base)[out].mean()), q0=float(k2 * base[-1]))))
    return dict(identity="V = sum_context Q = (L - R) * qbar_out because Q = 0 on the R rest days. For two cells with R1 != R2: V1 = V2 and qbar_out,1 = qbar_out,2 "
                         "imply (R2 - R1) qbar_out = 0, i.e. qbar_out = 0. Hence both matches hold jointly only among cells with the same R (same stratum and rho); "
                         "across rho one of them must give way. Within (stratum, rho) the two conditions are the same condition.",
                note="Short stops lie outside the long rest and are part of the outside-rest mean; the on-level is rescaled by kappa. "
                     "Illustration uses realization-0 base multipliers, r=100 m, a*=20 d, ell=round(0.1 t95); design quantities only.",
                rows=rows)


def main():
    uf = unit_fixtures(); dump("unit_fixtures.json", uf)
    sig = sigma_bg(); dump("sigma_bg.json", sig)
    sc = stratum_scan(); dump("stratum_scan.json", sc)
    cv = coverage(sig); dump("coverage.json", cv)
    sf = schedule_feasibility(sc); dump("schedule_feasibility.json", sf)
    mi = matching_identity(sc); dump("matching_identity.json", mi)
    print(json.dumps(dict(unit_fixtures_ok=uf["all_ok"], sigma_bg_bit_equal_to_pilot=all(r["bit_equal"] for r in sig["rows"]),
                          sigma_bg=[round(r["sigma_bg_m"], 4) for r in sig["rows"]])))


if __name__ == "__main__":
    main()
