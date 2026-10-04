"""Collect W and reference outputs into tables/curves and verify the run (no plots, no report text).

Writes results/pilot/wb/tables/{W_table.csv, reference_table.csv, curves_long.csv} and
results/pilot/wb/wb_manifest.json (counts, statuses, flags, verification maxima, output hashes).
Rows are case x pair; record conditions (rest_ratio, signal_ratio, pi_layer) are columns so that
all later summaries are organised by record condition.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PILOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))) / "results/pilot"
WB = PILOT / "wb"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = json.load(open(PILOT / "cases/observed_manifest.json"))
    (WB / "tables").mkdir(exist_ok=True)
    wrows, rrows, crows = [], [], []
    status = Counter(); flags = Counter(); missing = []
    ver = dict(W_pair_identity_max=0.0, ref_pair_identity_max=0.0, ref_draws_reproduced_max=0.0, ref_simulate_consistency_max=0.0,
               W_nonfinite=0, ref_nonfinite=0, truth_in_G=0, truth_eval_n=0, E_true_inside_all_1to10=0, E_true_inside_lead10=0,
               draws_below_requested=0, local_round_cap_hit=0, restarts_cap_hit=0, global512_zero_accepted=0, runtime_W_s=[], runtime_ref_s=[])
    for m in man:
        cid = m["case_id"]
        mk = WB / "markers" / f"{cid}.done"
        if not mk.exists():
            missing.append(cid); continue
        mkd = json.loads(mk.read_text())
        status[(mkd.get("W_status"), mkd.get("reference_status"))] += 1
        ch = mkd.get("checks", {})
        for k, key in (("W_pair_identity_max", "W_pair_identity_maxabs_m"), ("ref_pair_identity_max", "reference_pair_identity_maxabs_m"),
                       ("ref_draws_reproduced_max", "reference_draws_reproduced_maxabs_m"), ("ref_simulate_consistency_max", "reference_simulate_consistency_m")):
            if ch.get(key) is not None:
                ver[k] = max(ver[k], float(ch[key]))
        ver["W_nonfinite"] += int(ch.get("W_finite") is False)
        ver["ref_nonfinite"] += int(ch.get("reference_finite") is False)
        Wr = json.loads((WB / "W" / f"{cid}.json").read_text())
        Rr = json.loads((WB / "reference" / f"{cid}.json").read_text())
        te_p = WB / "truth_eval" / f"{cid}.json"
        te = json.loads(te_p.read_text()) if te_p.exists() else {}
        for fl in Wr.get("flags", []):
            flags[fl] += 1
        tc = te.get("truth_compatibility")
        if tc:
            ver["truth_eval_n"] += 1
            ver["truth_in_G"] += int(tc["truth_nonlinear_in_G"])
        nloc = len(Wr.get("local", []))
        ver["global512_zero_accepted"] += int(((Wr.get("global", {}) or {}).get("512", {}) or {}).get("x1_n_accepted", 1) == 0)
        ver["local_round_cap_hit"] += int(nloc >= 8 and not Wr.get("converged", False))
        inc = Wr.get("refinement_increments", {})
        ver["restarts_cap_hit"] += int(any(len(v) >= 5 and v[-1] > 0.05 for v in inc.values()))
        if "wall_s" in Wr:
            ver["runtime_W_s"].append(Wr["wall_s"])
        if "wall_s" in Rr:
            ver["runtime_ref_s"].append(Rr["wall_s"])
        ver["draws_below_requested"] += int(bool(Rr.get("draws_below_requested")))
        base = dict(case_id=cid, realization=m["realization"], site=m["site"], origin_date=m["origin_date"],
                    rest_ratio=m["rest_ratio"], signal_ratio=m["signal_ratio"], pi_layer=m["pi_layer"])
        for p in PAIRS:
            env = Wr.get("envelope", {}).get(p)
            Et = np.array(te["E_true"][p]) if te else None
            dl = np.array(te["delta"][p]) if te else None
            inside = tc["E_true_inside"][p] if tc else None
            if tc:
                ver["E_true_inside_all_1to10"] += int(all(inside[:10]))
                ver["E_true_inside_lead10"] += int(inside[9])
            row = dict(base, pair=p, W_status=mkd.get("W_status"), converged=Wr.get("converged"), flags=";".join(Wr.get("flags", [])))
            if env:
                Wk = np.array(env["W"])
                row.update(W10=Wk[9], Wmax1to10=Wk[:10].max(), W30=Wk[29], Wmax1to30=Wk.max())
                if Et is not None:
                    row.update(E_true10=Et[9], E_true30=Et[29], delta10=dl[9], W10_over_absE10=(Wk[9] / abs(Et[9]) if Et[9] != 0 else np.nan),
                               W10_over_sigma=Wk[9] / 0.02, determined_lead10=bool(Wk[9] <= dl[9]),
                               determined_1to10=bool(np.max(Wk[:10] - dl[:10]) <= 0), E_true_inside_lead10=inside[9], E_true_inside_all_1to10=all(inside[:10]))
            tol = Wr.get("tolerance", {})
            row.update(n_eff=tol.get("n_eff"), rmse_min_m=tol.get("rmse_min_m"), rmse_tol_m=tol.get("rmse_equiv_m"),
                       n_accepted_x1=Wr.get("n_accepted_total", {}).get("x1"), n_local_rounds=nloc,
                       refine_first_gain_k10=(Wr.get("refinement_first_gain_over_sampled", {}) or {}).get("10"),
                       truth_in_G=(tc or {}).get("truth_nonlinear_in_G"), W_wall_s=Wr.get("wall_s"))
            dt = Wr.get("diagnostic_tolerance", {})
            if p == PAIRS[0]:  # diagnostic tolerances are computed on the P1 envelope only (P2 = 0.5 x P1 in the linear family)
                row.update(W10_diag_x0_5=(dt.get("x0.5") or {}).get("W10"), W10_diag_x2=(dt.get("x2") or {}).get("W10"))
            g512 = (Wr.get("global", {}) or {}).get("512", {})
            row.update(global512_n_accepted_x1=g512.get("x1_n_accepted"))
            wrows.append(row)
            rp = Rr.get("pairs", {}).get(p, {})
            rrow = dict(base, pair=p, reference_status=mkd.get("reference_status"))
            if rp:
                Ep = np.array(rp["E_point"]); wt = np.array(rp.get("w_TF", [np.nan] * 30))
                rrow.update(E_point10=Ep[9], E_q10_10=rp.get("E_q10", [np.nan] * 30)[9], E_q90_10=rp.get("E_q90", [np.nan] * 30)[9],
                            w_TF10=wt[9], w_TFmax1to10=np.nanmax(wt[:10]), w_TF30=wt[29])
                if env:
                    lo, hi = env["inf"][9], env["sup"][9]
                    rrow.update(E_point10_inside_envelope=bool(lo - 1e-12 <= Ep[9] <= hi + 1e-12),
                                E_point_inside_all_1to10=bool(all(env["inf"][k] - 1e-12 <= Ep[k] <= env["sup"][k] + 1e-12 for k in range(10))))
                if Et is not None:
                    rrow.update(E_point10_minus_E_true10=Ep[9] - Et[9])
            opt = Rr.get("optimal_params", {})
            rrow.update(well_a=opt.get("well_a"), well_b=opt.get("well_b"), well_A=opt.get("well_A"), fit_rmse_m=Rr.get("fit_rmse_m"),
                        n_draws=Rr.get("n_param_draws_accepted"), draws_below_requested=Rr.get("draws_below_requested"),
                        n_multistart_ok=sum(1 for s in Rr.get("multistart", []) if "obj" in s), ref_wall_s=Rr.get("wall_s"))
            rrows.append(rrow)
            for k in range(30):
                c = dict(case_id=cid, pair=p, lead=k + 1, rest_ratio=m["rest_ratio"], signal_ratio=m["signal_ratio"], pi_layer=m["pi_layer"])
                if env:
                    c.update(W=env["W"][k], sup=env["sup"][k], inf=env["inf"][k])
                if rp:
                    c.update(ref_E_point=rp["E_point"][k], ref_E_q10=rp.get("E_q10", [None] * 30)[k], ref_E_q90=rp.get("E_q90", [None] * 30)[k])
                if te:
                    c.update(E_true=te["E_true"][p][k], delta=te["delta"][p][k])
                crows.append(c)

    def write(path, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
            for r in rows:
                wr.writerow({k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()})
    write(WB / "tables/W_table.csv", wrows); write(WB / "tables/reference_table.csv", rrows); write(WB / "tables/curves_long.csv", crows)
    rtW, rtR = ver.pop("runtime_W_s"), ver.pop("runtime_ref_s")
    manifest = dict(protocol_sha256=(PILOT / "protocol.sha256").read_text().split()[0], n_cases_manifest=len(man), n_markers=len(man) - len(missing),
                    missing=missing, status_counts={f"W={a}|ref={b}": n for (a, b), n in status.items()}, W_flag_counts=dict(flags),
                    verification=ver, runtime=dict(W_median_s=float(np.median(rtW)) if rtW else None, W_max_s=float(np.max(rtW)) if rtW else None,
                                                   ref_median_s=float(np.median(rtR)) if rtR else None, ref_max_s=float(np.max(rtR)) if rtR else None,
                                                   W_total_s=float(np.sum(rtW)), ref_total_s=float(np.sum(rtR))),
                    tables={p.name: dict(path=str(p.relative_to(PILOT)), sha256=sha(p), rows=sum(1 for _ in open(p)) - 1) for p in (WB / "tables").glob("*.csv")},
                    file_counts={d: len(list((WB / d).glob("*"))) for d in ("W", "reference", "truth_eval", "markers")})
    json.dump(manifest, open(WB / "wb_manifest.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: manifest[k] for k in ("n_markers", "status_counts", "W_flag_counts", "verification", "runtime", "file_counts")}, default=float, indent=1))


if __name__ == "__main__":
    main()
