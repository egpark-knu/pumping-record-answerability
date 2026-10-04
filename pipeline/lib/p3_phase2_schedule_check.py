"""D03 Phase 2 production schedule check (B), all 540 cells with the ACTUAL pilot seeds (realization_calendar.json).
Schedules and design quantities only: no head series, no W, no forecast, no score. sigma_bg per realization is the
natural-background design quantity (results/phase2/B_preflight/sigma_bg.json, bit-equal to the pilot truth record).

Writes results/phase2/B_merge/schedule_cases.csv (540 rows) and schedule_check.json (invariants, ranges, negatives).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_generator as g  # noqa: E402
from p3_phase2_cases import CTX, NS, Q_NOM, RHOS, census, layer, schedule, strata  # noqa: E402

OUT = ROOT / "results/phase2/B_merge"
REL_TOL = 1e-12  # floating-point tolerance for the matched sums (kappa is exact algebra; only summation order differs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reals = json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))
    sig = {r["realization"]: r["sigma_bg_m"] for r in json.load(open(ROOT / "results/phase2/B_preflight/sigma_bg.json"))["rows"]}
    rows, fails = [], []
    inv = dict(n_cells=0, census_N_ON_OFF=True, short_all_1d_lt_t95=True, designated_rest_identity=True, only_short_offs_outside_rest=True,
               tail_on_ge_t95=True, first_run_on=True, rest_zero=True, rho_bit_identity_outside_rest=True, base_hash_equal_across_rho=True,
               volume_equal_across_N=True, mean_outside_equal_across_N=True, q0_equal_across_rho_N=True, short_outside_rest=True,
               compensation_outside_rest_and_burst=True)
    max_rel_V, max_rel_mean = 0.0, 0.0
    cross_rho, layers = {}, {}
    for real in reals:
        mult = g.base_schedule(real["seed_schedule"], g.Design())
        for st in strata():
            L = layer(st)
            layers[st.sid] = dict(t95_d=L["t95"], e=L["e"], R=L["R"], R_max=L["Rmax"], tail_N18=CTX - L["e"] - 18, tail_N2=CTX - L["e"] - 2,
                                  tail_over_t95_N18=(CTX - L["e"] - 18) / L["t95"], tail_over_t95_N2=(CTX - L["e"] - 2) / L["t95"],
                                  compensation_window_days=L["e"] - L["Rmax"], gain=L["d"]["gain"], pi_r=L["d"]["pi_r"], c_d=st.c, a_d=L["d"]["a"], b=L["d"]["b"])
            cells = {}
            for rho in RHOS:
                for N in NS:
                    sc = schedule(mult, L, rho, N)
                    q = sc["q"]
                    rest = (sc["e"] - sc["R"], sc["e"])
                    cen = census(q, L["t95"], rest)
                    out = np.ones(CTX, bool); out[rest[0]:rest[1]] = False
                    V, mo = float(q.sum()), float(q[out].mean())
                    SR = L["d"]["gain"] * mo / sig[real["realization"]]
                    cells[(rho, N)] = (q, sc)
                    ok = dict(census_N_ON_OFF=cen["N_ON"] == N // 2 and cen["N_OFF"] == N // 2 and cen["N_short"] == N,
                              short_all_1d_lt_t95=cen["all_short_1d_lt_t95"], designated_rest_identity=cen["designated_rest_identity"],
                              only_short_offs_outside_rest=cen["n_off_runs_outside_rest"] == N // 2, tail_on_ge_t95=cen["tail_on"] and cen["tail_len"] >= L["t95"],
                              first_run_on=cen["first_run_on"], rest_zero=bool(np.all(q[rest[0]:rest[1]] == 0)),
                              short_outside_rest=bool(not np.any(sc["off"][rest[0]:rest[1]]) and sc["e"] >= rest[1]),
                              compensation_outside_rest_and_burst=bool(not np.any(sc["C"][sc["e"] - sc["Rmax"]:])))
                    for k, v in ok.items():
                        if not v:
                            inv[k] = False; fails.append((real["realization"], st.sid, rho, N, k))
                    inv["n_cells"] += 1
                    rows.append(dict(case_id=f"r{real['realization']:02d}_{st.storage_type}_T{st.T:g}_rho{rho:g}_N{N}", realization=real["realization"],
                                     storage_type=st.storage_type, S=st.S, T_m2_d=st.T, r_m=st.r, c_d=st.c, Q_nominal_m3_d=Q_NOM, t95_d=L["t95"],
                                     rho=rho, N=N, e=sc["e"], R_days=sc["R"], rest_start=rest[0], rho_realized=sc["R"] / L["t95"], kappa=sc["kappa"],
                                     V_m3=V, mean_outside_rest_m3d=mo, calendar_mean_m3d=V / CTX, q0_m3d=float(q[-1]), base_mult_mean=float(mult.mean()),
                                     N_ON=cen["N_ON"], N_OFF=cen["N_OFF"], tail_on_days=cen["tail_len"], tail_over_t95=cen["tail_over_t95"],
                                     last_short_event_day=sc["e"] + N - 1, SR=SR, pi_r=L["d"]["pi_r"], sigma_bg_m=sig[real["realization"]],
                                     base_N_sha256=hashlib.sha256(np.ascontiguousarray(sc["base_N"]).tobytes()).hexdigest()))
            # rho: base identical, only the rest window differs (bit identity)
            for N in NS:
                qs = {rho: cells[(rho, N)] for rho in RHOS}
                h = {hashlib.sha256(np.ascontiguousarray(qs[r][1]["base_N"]).tobytes()).hexdigest() for r in RHOS}
                if len(h) != 1:
                    inv["base_hash_equal_across_rho"] = False; fails.append((real["realization"], st.sid, N, "base_hash"))
                for rho in RHOS:
                    q, sc = qs[rho]
                    m = np.ones(CTX, bool); m[sc["e"] - sc["R"]:sc["e"]] = False
                    if not np.array_equal(q[m], sc["base_N"][m]):
                        inv["rho_bit_identity_outside_rest"] = False; fails.append((real["realization"], st.sid, rho, N, "rho_bit"))
                Vs = [float(qs[r][0].sum()) for r in RHOS]
                cross_rho.setdefault(st.sid, []).append(dict(realization=real["realization"], N=N, V_by_rho=dict(zip(map(str, RHOS), Vs)),
                                                             V_rel_range=(max(Vs) - min(Vs)) / max(Vs)))
            # N: equal volume and equal mean outside the designated rest within (rho)
            for rho in RHOS:
                qN = [cells[(rho, N)][0] for N in NS]
                sc0 = cells[(rho, NS[0])][1]
                out = np.ones(CTX, bool); out[sc0["e"] - sc0["R"]:sc0["e"]] = False
                V = [x.sum() for x in qN]; mo = [x[out].mean() for x in qN]
                rv = (max(V) - min(V)) / max(V); rm = (max(mo) - min(mo)) / max(mo)
                max_rel_V, max_rel_mean = max(max_rel_V, rv), max(max_rel_mean, rm)
                if rv > REL_TOL:
                    inv["volume_equal_across_N"] = False; fails.append((real["realization"], st.sid, rho, "V"))
                if rm > REL_TOL:
                    inv["mean_outside_equal_across_N"] = False; fails.append((real["realization"], st.sid, rho, "mean"))
            q0s = {cells[k][0][-1] for k in cells}
            if len(q0s) != 1:
                inv["q0_equal_across_rho_N"] = False; fails.append((real["realization"], st.sid, "q0"))
    with open(OUT / "schedule_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sr_med = {}
    for sid in layers:
        v = [r["SR"] for r in rows if f"{r['storage_type']}_T{r['T_m2_d']:g}" == sid]
        sr_med[sid] = dict(min=min(v), median=float(np.median(v)), max=max(v))
    cov = dict(SR_stratum_median_lt1=[k for k, v in sr_med.items() if v["median"] < 1], SR_stratum_median_gt1=[k for k, v in sr_med.items() if v["median"] > 1],
               n_cases_SR_lt1=sum(r["SR"] < 1 for r in rows), n_cases_SR_gt1=sum(r["SR"] > 1 for r in rows),
               pi_lt1=[k for k, v in layers.items() if v["pi_r"] < 1], pi_gt1=[k for k, v in layers.items() if v["pi_r"] > 1])
    cov["covers_both_sides"] = bool(cov["SR_stratum_median_lt1"] and cov["SR_stratum_median_gt1"] and cov["pi_lt1"] and cov["pi_gt1"])
    V_loss = [(1 - min(x["V_by_rho"].values()) / max(x["V_by_rho"].values())) for v in cross_rho.values() for x in v]
    rec = dict(status="production schedules (actual pilot seeds), design quantities only; no heads, no scores",
               invariants=inv, all_invariants_pass=all(v for k, v in inv.items() if k != "n_cells") and inv["n_cells"] == 540,
               failures=fails[:50], n_failures=len(fails), max_rel_V_spread_across_N=max_rel_V, max_rel_mean_spread_across_N=max_rel_mean,
               cross_rho_volume_loss_fraction_range=[min(V_loss), max(V_loss)],
               cross_rho_by_layer={k: [min(x["V_rel_range"] for x in v), max(x["V_rel_range"] for x in v)] for k, v in cross_rho.items()},
               kappa_range=[min(r["kappa"] for r in rows), max(r["kappa"] for r in rows)],
               q0_range_m3d=[min(r["q0_m3d"] for r in rows), max(r["q0_m3d"] for r in rows)],
               calendar_mean_range_m3d=[min(r["calendar_mean_m3d"] for r in rows), max(r["calendar_mean_m3d"] for r in rows)],
               mean_outside_rest_range_m3d=[min(r["mean_outside_rest_m3d"] for r in rows), max(r["mean_outside_rest_m3d"] for r in rows)],
               rho_realized_range={str(rho): [min(r["rho_realized"] for r in rows if r["rho"] == rho), max(r["rho_realized"] for r in rows if r["rho"] == rho)] for rho in RHOS},
               layers=layers, SR_by_layer=sr_med, coverage=cov, negatives=negatives(reals))
    (OUT / "schedule_check.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("all_invariants_pass", "n_failures", "max_rel_V_spread_across_N", "max_rel_mean_spread_across_N",
                                          "cross_rho_volume_loss_fraction_range", "kappa_range", "q0_range_m3d")}, default=float))
    print(json.dumps(rec["coverage"]), json.dumps({k: v["detected"] for k, v in rec["negatives"].items()}))


def negatives(reals) -> dict:
    """Meaningful negative fixtures: each wrong variant must be DETECTED by the same checks."""
    mult = g.base_schedule(reals[0]["seed_schedule"], g.Design())
    st = strata()[0]
    L = layer(st)
    out = {}
    # (1) OFF-first prefix: OFF at e joins the designated rest -> rest identity lost
    sc = schedule(mult, L, 1.0, 6)
    q = sc["q"].copy(); e = sc["e"]
    q2 = sc["base_N"].copy(); q2[e - sc["R"]:e] = 0; q2[e:e + 6:2] = 0; q2[e + 1:e + 6:2] = sc["base0"][e + 1:e + 6:2]
    c2 = census(q2, L["t95"], (e - sc["R"], e))
    out["off_first_prefix"] = dict(detected=bool(not c2["designated_rest_identity"]), census=c2)
    # (2) wrong kappa (no compensation): volume differs across N
    V = []
    for N in NS:
        s = schedule(mult, L, 1.0, N); b = s["base0"].copy(); b[s["off"]] = 0; b[s["e"] - s["R"]:s["e"]] = 0; V.append(b.sum())
    out["kappa_one"] = dict(detected=bool((max(V) - min(V)) / max(V) > REL_TOL), rel_spread=float((max(V) - min(V)) / max(V)))
    # (3) compensation applied on the whole outside-rest window (rho-dependent): rho bit identity broken
    bases = []
    for rho in RHOS:
        s = schedule(mult, L, rho, 6); b = s["base0"].copy(); b[s["off"]] = 0
        out_m = np.ones(CTX, bool); out_m[s["e"] - s["R"]:s["e"]] = False; out_m[s["off"]] = False
        b[out_m] *= 1 + s["base0"][s["off"]].sum() / s["base0"][out_m].sum(); bases.append(b)
    out["rho_dependent_compensation"] = dict(detected=bool(not all(np.array_equal(bases[0], x) for x in bases[1:])))
    # (4) burst ending at the origin (no tail): the last short OFF would be the final run -> N not certifiable
    e_bad = CTX - 18
    qb = 100 * mult.copy(); qb[e_bad - L["R"][1.0]:e_bad] = 0; qb[e_bad + 1:e_bad + 18:2] = 0
    cb = census(qb, L["t95"], (e_bad - L["R"][1.0], e_bad))
    out["no_tail_endpoint"] = dict(detected=bool(not (cb["tail_on"] and cb["tail_len"] >= L["t95"]) or cb["N_short"] != 18), census=cb)
    return out


if __name__ == "__main__":
    main()
