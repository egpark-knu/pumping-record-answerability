"""Collect the v1.2 repaired W run into tables/curves/manifest and compare with the original run (B). No plots, no report text.

Writes, under --dir (default results/pilot/wb_repaired_v1_2):
  tables/W_table.csv, tables/reference_table.csv, tables/curves_long.csv   (rows case x pair; record conditions as columns)
  wb_manifest.json                                                         (counts, flags, verification, hashes)
  repair_comparison.json (official dir: results/pilot/repair_comparison.json)
Reference fits/draws are READ from the original results/pilot/wb/reference (values unchanged); every envelope-inclusion
column is recomputed against the v1.2 envelope. Truth columns come from the v1.2 truth_eval (evaluation only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
PILOT = ROOT / "results/pilot"
OLD = PILOT / "wb"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
TOL_IN = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inside(env, E, k):
    return bool(env["inf"][k] - TOL_IN <= E[k] <= env["sup"][k] + TOL_IN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=PILOT / "wb_repaired_v1_2")
    a = ap.parse_args()
    D = a.dir if a.dir.is_absolute() else ROOT / a.dir
    official = D.resolve() == (PILOT / "wb_repaired_v1_2").resolve()
    man = json.load(open(PILOT / "cases/observed_manifest.json"))
    (D / "tables").mkdir(exist_ok=True)
    wrows, rrows, crows, comp = [], [], [], []
    status = Counter(); flags = Counter(); old_flags = Counter(); missing = []
    ver = dict(W_pair_identity_max=0.0, W_nonfinite=0, input_identity_valid=0, cached_fit_bit_identical=0, seed_direct_feasible=0,
               seed_interval_ok=0, seed_A_direct_contained=0, seed_legacy_gram_excludes_A_direct=0, seed_legacy_vs_direct_A_maxabs=0.0,
               seed_outside_probe_gap_max=0.0, witness_check_pass=0, witness_sse_over_tau_max=-np.inf, witness_bound_violation_max=0.0,
               witness_value_mismatch_max=0.0, direct_audit_material=0, direct_audit_search_minus_certified_max_m=0.0,
               direct_audit_certified_minus_search_max_m=0.0, witnesses_uncertified_total=0, refinement_terminals_total=0,
               refinement_terminals_feasible=0, global512_zero_accepted=0, local_round_cap_hit=0, restarts_cap_hit=0,
               truth_eval_n=0, truth_nonlinear_in_G=0, truth_full_vector_in_G=0, A_true_in_direct_profile_interval=0,
               E_true_inside_all_1to10=0, E_true_inside_lead10=0, E_true_inside_all_1to30=0, reference_reused_ok=0)
    rt_W = []
    for m in man:
        cid = m["case_id"]
        mk = D / "markers" / f"{cid}.done"
        if not mk.exists():
            if official:
                missing.append(cid)
            continue
        mkd = json.loads(mk.read_text())
        Wr = json.loads((D / "W" / f"{cid}.json").read_text())
        Wo = json.loads((OLD / "W" / f"{cid}.json").read_text())
        Rr = json.loads((OLD / "reference" / f"{cid}.json").read_text())
        oldmk = json.loads((OLD / "markers" / f"{cid}.done").read_text())
        ref_ok = sha(OLD / "reference" / f"{cid}.json") == oldmk["outputs"].get(f"wb/reference/{cid}.json") and oldmk.get("reference_status") == "ok"
        ver["reference_reused_ok"] += int(ref_ok)
        te_p = D / "truth_eval" / f"{cid}.json"
        te = json.loads(te_p.read_text()) if te_p.exists() else {}
        st = Wr.get("W_status", "error"); status[st] += 1
        for fl in Wr.get("flags", []):
            flags[fl] += 1
        for fl in Wo.get("flags", []):
            old_flags[fl] += 1
        ch = Wr.get("checks", {})
        ver["W_pair_identity_max"] = max(ver["W_pair_identity_max"], float(ch.get("W_pair_identity_maxabs_m", 0.0)))
        ver["W_nonfinite"] += int(ch.get("W_finite") is False)
        ver["input_identity_valid"] += int(Wr.get("input_identity", {}).get("valid", False))
        ver["cached_fit_bit_identical"] += int(Wr.get("cached_fit_check", {}).get("identical", False))
        s1 = (Wr.get("bestfit_seed") or {}).get("x1", {})
        ver["seed_direct_feasible"] += int(s1.get("direct_feasible", False)); ver["seed_interval_ok"] += int(s1.get("interval_ok", False))
        ver["seed_A_direct_contained"] += int(s1.get("A_direct_contained", False))
        ver["seed_legacy_gram_excludes_A_direct"] += int(s1.get("legacy_excludes_A_direct", False))
        ver["seed_legacy_vs_direct_A_maxabs"] = max(ver["seed_legacy_vs_direct_A_maxabs"], float(s1.get("legacy_vs_direct_A_maxabs") or 0.0))
        for o in (s1.get("out_lo"), s1.get("out_hi")):
            if o:
                ver["seed_outside_probe_gap_max"] = max(ver["seed_outside_probe_gap_max"], float(o["gap"]))
        wc = Wr.get("witness_check", {})
        ver["witness_check_pass"] += int(wc.get("pass", False))
        if wc:
            ver["witness_sse_over_tau_max"] = max(ver["witness_sse_over_tau_max"], wc["sse_over_tau_max"])
            ver["witness_bound_violation_max"] = max(ver["witness_bound_violation_max"], wc["bound_violation_max"])
            ver["witness_value_mismatch_max"] = max(ver["witness_value_mismatch_max"], wc["value_mismatch_max"])
        au = Wr.get("direct_audit", {})
        ver["direct_audit_material"] += int(au.get("material", False))
        ver["direct_audit_search_minus_certified_max_m"] = max(ver["direct_audit_search_minus_certified_max_m"], au.get("search_minus_certified_max_m", 0.0))
        ver["direct_audit_certified_minus_search_max_m"] = max(ver["direct_audit_certified_minus_search_max_m"], au.get("certified_minus_search_max_m", 0.0))
        ver["witnesses_uncertified_total"] += int(au.get("n_uncertified", 0))
        terms = Wr.get("refinement_terminals", [])
        ver["refinement_terminals_total"] += len(terms); ver["refinement_terminals_feasible"] += sum(t["feasible"] for t in terms)
        nloc = len(Wr.get("local", []))
        ver["global512_zero_accepted"] += int(((Wr.get("global", {}) or {}).get("512", {}) or {}).get("x1_n_accepted", 1) == 0)
        ver["local_round_cap_hit"] += int(nloc >= 8 and not any(max(l.get("rel_change", [1, 1])) <= 0.05 and l.get("cum_n_accepted", 0) >= 200 for l in Wr.get("local", [])))
        inc = Wr.get("refinement_increments", {})
        ver["restarts_cap_hit"] += int(any(len(v) >= 5 and v[-1] > 0.05 for v in inc.values()))
        if "wall_s" in Wr:
            rt_W.append(Wr["wall_s"])
        tc = te.get("truth_compatibility")
        if tc:
            ver["truth_eval_n"] += 1
            ver["truth_nonlinear_in_G"] += int(tc["truth_nonlinear_in_G"]); ver["truth_full_vector_in_G"] += int(tc["truth_full_vector_in_G"])
            ver["A_true_in_direct_profile_interval"] += int(tc["A_true_in_direct_profile_interval"])
        base = dict(case_id=cid, realization=m["realization"], site=m["site"], origin_date=m["origin_date"],
                    rest_ratio=m["rest_ratio"], signal_ratio=m["signal_ratio"], pi_layer=m["pi_layer"])
        cmp_case = dict(case_id=cid, rest_ratio=m["rest_ratio"], signal_ratio=m["signal_ratio"], pi_layer=m["pi_layer"],
                        old_flags=Wo.get("flags", []), new_flags=Wr.get("flags", []), new_status=st,
                        old_status="ok" if Wo.get("envelope") else "W_failed", pairs={})
        for p in PAIRS:
            env = (Wr.get("envelope") or {}).get(p) if st == "ok" else None
            envo = (Wo.get("envelope") or {}).get(p)
            Et = np.array(te["E_true"][p]) if te else None
            dl = np.array(te["delta"][p]) if te else None
            ins = tc["E_true_inside"][p] if tc else None
            if tc:
                ver["E_true_inside_all_1to10"] += int(all(ins[:10])); ver["E_true_inside_lead10"] += int(ins[9]); ver["E_true_inside_all_1to30"] += int(all(ins))
            row = dict(base, pair=p, version="v1.2", W_status=st, converged=Wr.get("converged"), flags=";".join(Wr.get("flags", [])))
            if env:
                Wk = np.array(env["W"])
                row.update(W10=Wk[9], Wmax1to10=Wk[:10].max(), W30=Wk[29], Wmax1to30=Wk.max(), kmax1to10=int(np.argmax(Wk[:10])) + 1)
                if Et is not None:
                    row.update(E_true10=Et[9], E_true30=Et[29], delta10=dl[9], delta30=dl[29], W10_over_absE10=(Wk[9] / abs(Et[9]) if Et[9] != 0 else np.nan),
                               W10_over_sigma=Wk[9] / 0.02, determined_lead10=bool(Wk[9] <= dl[9]), determined_1to10=bool(np.max(Wk[:10] - dl[:10]) <= 0),
                               determined_lead30=bool(Wk[29] <= dl[29]), E_true_inside_lead10=ins[9], E_true_inside_all_1to10=all(ins[:10]),
                               E_true_inside_lead30=ins[29], E_true_inside_all_1to30=all(ins))
            tol = Wr.get("tolerance", {})
            au = Wr.get("direct_audit", {})
            row.update(n_eff=tol.get("n_eff"), rmse_min_m=tol.get("rmse_min_m"), rmse_tol_m=tol.get("rmse_equiv_m"), tau=tol.get("tau"),
                       n_accepted_x1=Wr.get("n_accepted_total", {}).get("x1"), n_local_rounds=nloc,
                       global512_n_accepted_x1=((Wr.get("global", {}) or {}).get("512", {}) or {}).get("x1_n_accepted"),
                       refine_first_gain_k10=(Wr.get("refinement_first_gain_over_sampled", {}) or {}).get("10"), kstar=Wr.get("kstar"),
                       n_refine_terminals=len(terms), n_refine_terminals_feasible=sum(t["feasible"] for t in terms),
                       seed_direct_feasible=s1.get("direct_feasible"), seed_A_direct_contained=s1.get("A_direct_contained"),
                       direct_audit_material=au.get("material"), direct_audit_max_abs_m=max(au.get("search_minus_certified_max_m", 0.0), au.get("certified_minus_search_max_m", 0.0)),
                       witnesses_uncertified=au.get("n_uncertified"), witness_check_pass=wc.get("pass"), cached_fit_identical=Wr.get("cached_fit_check", {}).get("identical"),
                       truth_nonlinear_in_G=(tc or {}).get("truth_nonlinear_in_G"), truth_full_vector_in_G=(tc or {}).get("truth_full_vector_in_G"),
                       A_true_in_direct_profile_interval=(tc or {}).get("A_true_in_direct_profile_interval"), W_wall_s=Wr.get("wall_s"))
            dt = Wr.get("diagnostic_tolerance", {})
            if p == PAIRS[0]:
                row.update(W10_diag_x0_5=(dt.get("x0.5") or {}).get("W10"), W10_diag_x2=(dt.get("x2") or {}).get("W10"))
            wrows.append(row)
            # comparison to the original run
            cp = {}
            if envo:
                Wo_ = np.array(envo["W"]); cp.update(old_W10=Wo_[9], old_Wmax1to10=Wo_[:10].max(), old_W30=Wo_[29], old_Wmax1to30=Wo_.max())
            if env:
                cp.update(new_W10=Wk[9], new_Wmax1to10=Wk[:10].max(), new_W30=Wk[29], new_Wmax1to30=Wk.max())
            if envo and env:
                for q in ("W10", "Wmax1to10", "W30", "Wmax1to30"):
                    cp[f"rel_change_{q}"] = (cp[f"new_{q}"] - cp[f"old_{q}"]) / cp[f"old_{q}"] if cp[f"old_{q}"] else None
                sup_o, inf_o = np.array(envo["sup"]), np.array(envo["inf"]); sup_n, inf_n = np.array(env["sup"]), np.array(env["inf"])
                cp["new_contains_old_all_leads"] = bool(np.all(sup_n >= sup_o - 1e-12) and np.all(inf_n <= inf_o + 1e-12))
                cp["max_old_outside_new_m"] = float(max(np.max(sup_o - sup_n), np.max(inf_n - inf_o), 0.0))
            if Et is not None:
                cp["new_E_true_inside_1to10"] = bool(env is not None and all(ins[:10]))
                cp["old_E_true_inside_1to10"] = bool(envo is not None and all(envo["inf"][k] - TOL_IN <= Et[k] <= envo["sup"][k] + TOL_IN for k in range(10)))
            cmp_case["pairs"][p] = cp
            # reference: values reused, inclusion recomputed against the v1.2 envelope
            rp = Rr.get("pairs", {}).get(p, {})
            rrow = dict(base, pair=p, reference_status=oldmk.get("reference_status"), reference_reused_hash_ok=ref_ok,
                        W_version="v1.2", W_status=st)
            if rp:
                Ep = np.array(rp["E_point"]); wt = np.array(rp.get("w_TF", [np.nan] * 30))
                rrow.update(E_point10=Ep[9], E_q10_10=rp.get("E_q10", [np.nan] * 30)[9], E_q90_10=rp.get("E_q90", [np.nan] * 30)[9],
                            E_point30=Ep[29], E_q10_30=rp.get("E_q10", [np.nan] * 30)[29], E_q90_30=rp.get("E_q90", [np.nan] * 30)[29],
                            w_TF10=wt[9], w_TFmax1to10=np.nanmax(wt[:10]), w_TF30=wt[29])
                if env:
                    rrow.update(E_point10_inside_envelope=inside(env, Ep, 9), E_point_inside_all_1to10=all(inside(env, Ep, k) for k in range(10)),
                                E_point30_inside_envelope=inside(env, Ep, 29), E_point_inside_all_1to30=all(inside(env, Ep, k) for k in range(30)),
                                w_TF10_over_W10=(wt[9] / Wk[9] if Wk[9] > 0 else np.nan))
                if envo:
                    rrow.update(old_E_point10_inside_envelope=inside(envo, Ep, 9))
                if Et is not None:
                    rrow.update(E_point10_minus_E_true10=Ep[9] - Et[9], E_point30_minus_E_true30=Ep[29] - Et[29])
            opt = Rr.get("optimal_params", {})
            rrow.update(well_a=opt.get("well_a"), well_b=opt.get("well_b"), well_A=opt.get("well_A"), fit_rmse_m=Rr.get("fit_rmse_m"),
                        n_draws=Rr.get("n_param_draws_accepted"), draws_below_requested=Rr.get("draws_below_requested"),
                        n_multistart_ok=sum(1 for s in Rr.get("multistart", []) if "obj" in s), ref_wall_s=Rr.get("wall_s"))
            rrows.append(rrow)
            for k in range(30):
                c = dict(case_id=cid, pair=p, lead=k + 1, rest_ratio=m["rest_ratio"], signal_ratio=m["signal_ratio"], pi_layer=m["pi_layer"],
                         W_version="v1.2", W_status=st)
                if env:
                    c.update(W=env["W"][k], sup=env["sup"][k], inf=env["inf"][k])
                if rp:
                    c.update(ref_E_point=rp["E_point"][k], ref_E_q10=rp.get("E_q10", [None] * 30)[k], ref_E_q90=rp.get("E_q90", [None] * 30)[k])
                    if env:
                        c.update(ref_E_point_inside=inside(env, np.asarray(rp["E_point"]), k))
                if te:
                    c.update(E_true=te["E_true"][p][k], delta=te["delta"][p][k], E_true_inside=ins[k] if ins is not None else None)
                crows.append(c)
        comp.append(cmp_case)

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
    write(D / "tables/W_table.csv", wrows); write(D / "tables/reference_table.csv", rrows); write(D / "tables/curves_long.csv", crows)
    # ------------------------------------------------------------ comparison summary (differences both ways are reported)
    rel = {q: [c["pairs"][PAIRS[0]].get(f"rel_change_{q}") for c in comp if c["pairs"][PAIRS[0]].get(f"rel_change_{q}") is not None]
           for q in ("W10", "Wmax1to10", "W30", "Wmax1to30")}
    def dist(x):
        x = np.asarray(x, float)
        return None if not x.size else dict(n=int(x.size), min=float(x.min()), q10=float(np.quantile(x, 0.1)), median=float(np.median(x)),
                                             q90=float(np.quantile(x, 0.9)), max=float(x.max()), n_increase_gt_1pct=int((x > 0.01).sum()),
                                             n_decrease_gt_1pct=int((x < -0.01).sum()), n_within_1pct=int((np.abs(x) <= 0.01).sum()))
    cmpsum = dict(n_cases=len(comp), old_status=dict(Counter(c["old_status"] for c in comp)), new_status=dict(Counter(c["new_status"] for c in comp)),
                  old_flag_counts=dict(old_flags), new_flag_counts=dict(flags),
                  previously_failed_now_ok=[c["case_id"] for c in comp if c["old_status"] == "W_failed" and c["new_status"] == "ok"],
                  rel_change_P1=dict((q, dist(v)) for q, v in rel.items()),
                  n_new_contains_old_all_leads=sum(bool(c["pairs"][PAIRS[0]].get("new_contains_old_all_leads")) for c in comp),
                  max_old_outside_new_m=max([c["pairs"][PAIRS[0]].get("max_old_outside_new_m", 0.0) for c in comp] or [0.0]),
                  E_true_inside_1to10_P1=dict(old=sum(bool(c["pairs"][PAIRS[0]].get("old_E_true_inside_1to10")) for c in comp),
                                               new=sum(bool(c["pairs"][PAIRS[0]].get("new_E_true_inside_1to10")) for c in comp)),
                  reference_E_point10_inside=dict(old=sum(bool(r.get("old_E_point10_inside_envelope")) for r in rrows),
                                                  new=sum(bool(r.get("E_point10_inside_envelope")) for r in rrows), n_rows=len(rrows)),
                  note="Cross-version differences are reported, not treated as failures; the old run lacked the best-fit seed and full-curve "
                       "merging, and decreases can arise from different refinement starts/stopping. E_true inclusion is a diagnostic, not a criterion.")
    comp_out = dict(summary=cmpsum, per_case=comp)
    comp_path = (PILOT / "repair_comparison.json") if official else (D / "repair_comparison.json")
    comp_path.write_text(json.dumps(comp_out, indent=1, default=float))
    fz = PILOT / "repair_freeze.json"
    manifest = dict(version="v1.2", base_protocol_sha256=(PILOT / "protocol.sha256").read_text().split()[0],
                    repair_freeze=dict(path="results/pilot/repair_freeze.json", sha256=sha(fz)) if fz.exists() else None,
                    repair_freeze_hashes=json.loads(fz.read_text())["hashes"] if fz.exists() else None,
                    reuse_manifest_sha256=sha(PILOT / "wb_repaired_v1_2/reuse_manifest.json"),
                    official=official, n_cases_manifest=len(man), n_markers=len(list((D / "markers").glob("*.done"))), missing=missing,
                    n_contrasts_with_W=sum(1 for r in wrows if r.get("W10") is not None),
                    status_counts=dict(status), W_flag_counts=dict(flags), verification=ver,
                    runtime=dict(W_median_s=float(np.median(rt_W)) if rt_W else None, W_max_s=float(np.max(rt_W)) if rt_W else None, W_total_s=float(np.sum(rt_W))),
                    tables={p.name: dict(path=str(p.relative_to(PILOT)), sha256=sha(p), rows=sum(1 for _ in open(p)) - 1) for p in sorted((D / "tables").glob("*.csv"))},
                    repair_comparison=dict(path=str(comp_path.relative_to(PILOT)), sha256=sha(comp_path)),
                    file_counts={d: len(list((D / d).glob("*"))) for d in ("W", "truth_eval", "markers")},
                    notes=["W is a sampled inner envelope of direct-X-certified full curves under the declared family and frozen tau; it is not a "
                           "certified global extremum over G.", "truth_nonlinear_in_G is profile compatibility; truth_full_vector_in_G uses the true "
                           "linear coefficients (c0 profiled). Neither is a success criterion.", "Reference values are reused unchanged; "
                           "inclusion columns are recomputed against this envelope."])
    json.dump(manifest, open(D / "wb_manifest.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: manifest[k] for k in ("n_markers", "n_contrasts_with_W", "status_counts", "W_flag_counts", "verification", "runtime")}, default=float, indent=1))
    print(json.dumps(cmpsum, default=float, indent=1))


if __name__ == "__main__":
    main()
