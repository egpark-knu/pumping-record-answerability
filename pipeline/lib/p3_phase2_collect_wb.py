"""Collect the official Phase 2 W / truth-evaluation / reference records (B). No tool values, no figures, no report text.

Writes under results/phase2/wb/:
  tables/W_table.csv          one row per case x pair: physical/derived coordinates, W(10), W(30), max W days 1-10 and 1-30,
                              status/flags/convergence/accepted/rounds/runtime, E_true(10/30), delta, truth compatibility,
                              reference E(10/30), reference q10-q90 width, reference inclusion and row metrics vs truth
  tables/curves_long.csv      case x pair x lead (1..30): sup, inf, W, E_true, reference E_point
  wb_manifest.json            counts, flags by layer and cell, identities, runtime, driver/protocol/output hashes, completeness
The reference is the structurally matched reference, not a competitor; no ranking is formed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_phase2_metrics as mt  # noqa: E402
from p3_wb_collect_repaired_v1_2 import TOL_IN  # noqa: E402

P2 = ROOT / "results/phase2"
WB = P2 / "wb"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = json.load(open(P2 / "cases/derived_manifest.json"))
    (WB / "tables").mkdir(parents=True, exist_ok=True)
    rows, curves, missing, errors = [], [], [], []
    ident = dict(W_pair=0.0, ref_pair=0.0, truth_pair=0.0, ref_consistency=0.0, ref_draws_reproduced=0.0)
    flags_by = defaultdict(Counter)
    outs_hash = {}
    driver = set()
    for m in man:
        cid = m["case_id"]
        mk = WB / "markers" / f"{cid}.done"
        if not mk.exists():
            missing.append(cid); continue
        mr = json.loads(mk.read_text())
        driver.add(mr.get("driver_sha256"))
        for p, h in mr.get("outputs", {}).items():
            if sha(ROOT / p) != h:
                errors.append(f"{cid}: output hash changed {p}")
            outs_hash[p] = h
        W = json.loads((WB / "W" / f"{cid}.json").read_text())
        R = json.loads((WB / "reference" / f"{cid}.json").read_text())
        T = json.loads((WB / "truth_eval" / f"{cid}.json").read_text())
        if W.get("W_status") != "ok" or R.get("error") or "truth_compatibility" not in T:
            errors.append(f"{cid}: W={W.get('W_status')} ref_error={bool(R.get('error'))} truth={'truth_compatibility' in T}")
        env = W.get("envelope")
        tc = T.get("truth_compatibility", {})
        ident["W_pair"] = max(ident["W_pair"], W.get("checks", {}).get("W_pair_identity_maxabs_m", np.nan))
        rc = R.get("checks", {})
        ident["ref_pair"] = max(ident["ref_pair"], rc.get("reference_pair_identity_maxabs_m", np.nan))
        ident["ref_consistency"] = max(ident["ref_consistency"], rc.get("reference_simulate_consistency_m", np.nan))
        ident["ref_draws_reproduced"] = max(ident["ref_draws_reproduced"], rc.get("reference_draws_reproduced_maxabs_m") or 0.0)
        Et = {p: np.array(T["E_true"][p]) for p in PAIRS}
        ident["truth_pair"] = max(ident["truth_pair"], float(np.max(np.abs(Et[PAIRS[1]] + 0.5 * Et[PAIRS[0]]))))
        for f in W.get("flags", []):
            flags_by[m["sid"]][f] += 1
        for p in PAIRS:
            sup, inf = np.array(env[p]["sup"]), np.array(env[p]["inf"])
            Wc = sup - inf
            ref = R["pairs"][p]
            Er = np.array(ref["E_point"])
            wtf = np.array(ref["w_TF"]) if "w_TF" in ref else np.full(30, np.nan)
            row = dict(case_id=cid, pair_id=p, realization=m["realization"], site=m["site"], storage_type=m["storage_type"], S=m["storage_value"],
                       T_m2_d=m["T_m2_d"], c_d=m["c_d"], r_m=m["r_m"], rho=m["rho_nominal"], rho_realized=m["rho_realized"], N=m["N_nominal"],
                       t95_d=m["t95_d"], pi_r=m["pi_r"], SR=m["SR"], SR_context_mean=m["SR_context_mean"], sigma_bg_m=m["sigma_bg_m"],
                       q0_m3d=m["q0_m3d"], V_m3=m["V_m3"], mean_outside_rest_m3d=m["mean_outside_rest_m3d"], kappa=m["kappa"], R_days=m["R_days"],
                       W_status=W.get("W_status"), flags=";".join(W.get("flags", [])), converged=W.get("converged"),
                       n_accepted=(W.get("n_accepted_total") or {}).get("x1"), local_rounds=len(W.get("local", [])),
                       witness_pass=(W.get("witness_check") or {}).get("pass"), direct_audit_material=(W.get("direct_audit") or {}).get("material"),
                       W_wall_s=W.get("wall_s"),
                       W10_m=Wc[9], W30_m=Wc[29], Wmax1to10_m=float(Wc[:10].max()), Wmax1to30_m=float(Wc.max()),
                       E_true10_m=Et[p][9], E_true30_m=Et[p][29], delta10_m=T["delta"][p][9], delta30_m=T["delta"][p][29],
                       truth_nonlinear_in_G=tc.get("truth_nonlinear_in_G"), truth_full_vector_in_G=tc.get("truth_full_vector_in_G"),
                       A_true_in_direct_profile_interval=tc.get("A_true_in_direct_profile_interval"),
                       E_true_inside_10=tc.get("E_true_inside_TOL_IN", {}).get(p, [None] * 30)[9], E_true_inside_30=tc.get("E_true_inside_TOL_IN", {}).get(p, [None] * 30)[29],
                       E_true_inside_all_1to30=all(tc.get("E_true_inside_TOL_IN", {}).get(p, [False])),
                       ref_E10_m=Er[9], ref_E30_m=Er[29], ref_wTF10_m=wtf[9], ref_wTF30_m=wtf[29], ref_draws=R.get("n_param_draws_accepted"),
                       ref_fit_rmse_m=R.get("fit_rmse_m"), ref_wall_s=R.get("wall_s"))
            for k, lead in ((10, 9), (30, 29)):
                rm = mt.row_metrics(Et[p][lead], Er[lead], Wc[lead], inf[lead], sup[lead])
                row.update({f"ref_ratio{k}": rm["effect_ratio"], f"ref_ratio_defined{k}": rm["ratio_defined"], f"ref_sign_agree{k}": rm["sign_agreement"],
                            f"ref_abs_error{k}_m": rm["abs_error_m"], f"ref_error_over_W{k}": rm["error_over_W"], f"ref_in_envelope{k}": rm["envelope_includes"],
                            f"truth_class{k}": rm["truth_class"], f"small_001_{k}": rm["small_001"], f"small_02_{k}": rm["small_02"], f"W_invalid{k}": rm["W_invalid"]})
            rows.append(row)
            for k in range(30):
                curves.append(dict(case_id=cid, pair_id=p, lead_day=k + 1, sup_m=sup[k], inf_m=inf[k], W_m=Wc[k], E_true_m=Et[p][k], ref_E_m=Er[k]))
    for name, data in (("W_table.csv", rows), ("curves_long.csv", curves)):
        if data:
            with open(WB / "tables" / name, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(data[0])); wr.writeheader(); wr.writerows(data)
    n_cases = len(man) - len(missing)
    den = {}
    for k in (10, 30):
        den[k] = dict(n_rows=len(rows), W_invalid=sum(bool(r[f"W_invalid{k}"]) for r in rows),
                      truth_zero=sum(r[f"truth_class{k}"] == "zero" for r in rows), truth_nonfinite=sum(r[f"truth_class{k}"] == "nonfinite" for r in rows),
                      small_001=sum(bool(r[f"small_001_{k}"]) for r in rows), small_02=sum(bool(r[f"small_02_{k}"]) for r in rows),
                      ref_ratio_defined=sum(bool(r[f"ref_ratio_defined{k}"]) for r in rows), ref_in_envelope=sum(bool(r[f"ref_in_envelope{k}"]) for r in rows),
                      E_true_inside=sum(bool(r[f"E_true_inside_{k}"]) for r in rows))
    rec = dict(protocol_sha256=(P2 / "protocol.sha256").read_text().split()[0], driver_sha256=sorted(d for d in driver if d),
               n_cases_expected=len(man), n_cases_done=n_cases, n_missing=len(missing), missing=missing[:50], n_errors=len(errors), errors=errors[:50],
               complete=bool(n_cases == 540 and not errors), n_pair_rows=len(rows), n_curve_rows=len(curves),
               W_status=dict(Counter(r["W_status"] for r in rows if r["pair_id"] == PAIRS[0])),
               flags_total=dict(Counter(f for r in rows if r["pair_id"] == PAIRS[0] for f in (r["flags"].split(";") if r["flags"] else []))),
               flags_by_layer={k: dict(v) for k, v in flags_by.items()},
               converged=sum(bool(r["converged"]) for r in rows if r["pair_id"] == PAIRS[0]),
               witness_pass=sum(bool(r["witness_pass"]) for r in rows if r["pair_id"] == PAIRS[0]),
               direct_audit_material=sum(bool(r["direct_audit_material"]) for r in rows if r["pair_id"] == PAIRS[0]),
               truth_nonlinear_in_G=sum(bool(r["truth_nonlinear_in_G"]) for r in rows if r["pair_id"] == PAIRS[0]),
               truth_full_vector_in_G=sum(bool(r["truth_full_vector_in_G"]) for r in rows if r["pair_id"] == PAIRS[0]),
               E_true_inside_all_1to30_pairs=sum(bool(r["E_true_inside_all_1to30"]) for r in rows),
               ref_draws_below_1000=sum((r["ref_draws"] or 0) < 1000 for r in rows if r["pair_id"] == PAIRS[0]),
               identities_maxabs_m=ident, TOL_IN_m=TOL_IN, denominators=den,
               runtime_s=dict(W_sum=float(np.nansum([r["W_wall_s"] or np.nan for r in rows if r["pair_id"] == PAIRS[0]])),
                              ref_sum=float(np.nansum([r["ref_wall_s"] or np.nan for r in rows if r["pair_id"] == PAIRS[0]])),
                              W_median=float(np.nanmedian([r["W_wall_s"] or np.nan for r in rows if r["pair_id"] == PAIRS[0]])) if rows else None,
                              ref_median=float(np.nanmedian([r["ref_wall_s"] or np.nan for r in rows if r["pair_id"] == PAIRS[0]])) if rows else None),
               tables={n: sha(WB / "tables" / n) for n in ("W_table.csv", "curves_long.csv") if (WB / "tables" / n).exists()},
               n_output_files_hashed=len(outs_hash), outputs_sha256_digest=hashlib.sha256(json.dumps(sorted(outs_hash.items())).encode()).hexdigest())
    (WB / "wb_manifest.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("n_cases_done", "n_missing", "n_errors", "complete", "W_status", "flags_total", "converged", "identities_maxabs_m")}, default=float))


if __name__ == "__main__":
    main()
