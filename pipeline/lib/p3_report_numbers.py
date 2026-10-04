"""Aggregates quoted in results/pilot/REPORT.md (B). Reads ONLY official v1.2 tables, C's repaired aggregates and stored
manifests; no model, W or reference computation. Writes results/pilot/report_numbers.json (every entry names its source
file, column and grouping)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))) / "results/pilot"
F = P / "figures"
WB = P / "wb_repaired_v1_2"
P1, P2 = "P1_continue_vs_stop", "P2_current_vs_1p5x"
RHO = [0.25, 0.5, 1.0, 2.0, 4.0]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spearman(x, y):
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    cc = pd.read_csv(F / "cell_contrasts.csv")
    e30 = pd.read_csv(F / "event30_contrasts.csv")
    wt = pd.read_csv(WB / "tables/W_table.csv")
    man = json.load(open(WB / "wb_manifest.json"))
    comp = json.load(open(P / "repair_comparison.json"))
    out = {"sources": {str(p.relative_to(P)): sha(p) for p in [F / "cell_contrasts.csv", F / "event30_contrasts.csv", F / "condition_summary.csv",
                                                                 F / "event30_condition_summary.csv", F / "width_response_blocks.csv",
                                                                 WB / "tables/W_table.csv", WB / "wb_manifest.json", P / "repair_comparison.json"]}}
    assert len(cc) == 400 and len(e30) == 400 and len(wt) == 400 and (cc.W_status == "ok").all()
    wt1 = wt[wt.pair == P1].set_index("case_id")
    # ---------------- axis 1: W vs rest, stratified by SR and pi (P1; P2 = 0.5 x P1 exactly)
    ax1 = {}
    for col, src, lead in (("W10", cc, "lead10"), ("Wmax1to10", cc, "max1to10"), ("W30", e30, "lead30")):
        d = src[src.pair == P1]
        tab = d.groupby(["signal_ratio", "pi_layer", "rest_ratio"])[col].median().unstack("rest_ratio")
        ax1[f"median_{col}_P1_m"] = {f"SR{sr:g}_pi{pi:g}": [float(tab.loc[(sr, pi), r]) for r in RHO] for sr, pi in tab.index}
        # matched within-block response: realization x SR x pi, 5 rest levels
        blocks = d.groupby(["realization", "signal_ratio", "pi_layer"])
        rs, ratio = {}, {}
        for (re, sr, pi), g in blocks:
            g = g.sort_values("rest_ratio")
            rs.setdefault(f"SR{sr:g}_pi{pi:g}", []).append(spearman(g.rest_ratio, g[col]))
            ratio.setdefault(f"SR{sr:g}_pi{pi:g}", []).append(float(g[col].iloc[-1] / g[col].iloc[0]))
        ax1[f"block_spearman_rest_vs_{col}"] = {k: dict(n=len(v), negative=int(np.sum(np.array(v) < 0)), positive=int(np.sum(np.array(v) > 0)),
                                                         zero_or_undefined=int(np.sum(~(np.array(v) != 0))), median=float(np.nanmedian(v))) for k, v in rs.items()}
        ax1[f"block_ratio_rho4_over_rho0.25_{col}"] = {k: dict(n=len(v), median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)))
                                                        for k, v in ratio.items()}
    ax1["_source"] = "figures/cell_contrasts.csv (W10, Wmax1to10) and figures/event30_contrasts.csv (W30), pair P1; blocks = realization x signal_ratio x pi_layer over rest_ratio"
    # relative width and determination
    d = wt[wt.pair == P1]
    ax1["median_W10_over_absE10_P1"] = {f"SR{sr:g}_pi{pi:g}": [float(d[(d.signal_ratio == sr) & (d.pi_layer == pi) & (d.rest_ratio == r)].W10_over_absE10.median()) for r in RHO]
                                        for sr in (0.5, 2.0) for pi in (0.5, 5.0)}
    ax1["_source_rel"] = "wb_repaired_v1_2/tables/W_table.csv column W10_over_absE10, pair P1, median over 10 realizations"
    det = {}
    for lab, src, col in (("lead10", cc, "determined_lead10"), ("max1to10", cc, "determined_1to10"), ("lead30", e30, "determined_lead30")):
        det[lab] = dict(total=f"{int(src[col].sum())}/{len(src)}",
                        by_SR_pi={f"SR{sr:g}_pi{pi:g}": f"{int(g[col].sum())}/{len(g)}" for (sr, pi), g in src.groupby(["signal_ratio", "pi_layer"])},
                        by_rest={f"rho{r:g}": f"{int(g[col].sum())}/{len(g)}" for r, g in src.groupby("rest_ratio")})
    ax1["determined_counts"] = det
    ax1["_source_det"] = "figures/cell_contrasts.csv determined_lead10/determined_1to10; figures/event30_contrasts.csv determined_lead30 (W <= frozen delta)"
    # kmax
    ax1["kmax1to10_counts_P1"] = {f"SR{sr:g}_pi{pi:g}": {int(k): int(n) for k, n in g.kmax1to10.value_counts().sort_index().items()}
                                  for (sr, pi), g in d.groupby(["signal_ratio", "pi_layer"])}
    out["axis1"] = ax1
    # ---------------- axis 2 (a): envelope inclusion by record condition
    inc = {}
    for lab, src, ecol, rcol in (("lead10", cc, "engine_inside_lead10", "E_point10_inside_envelope"), ("all1to10", cc, "engine_inside_all_1to10", "E_point_inside_all_1to10"),
                                 ("lead30", e30, "engine_inside_lead30", "E_point30_inside_envelope")):
        r = dict(scenario_engine_total=f"{int(src[ecol].sum())}/{len(src)}", reference_total=f"{int(src[rcol].sum())}/{len(src)}")
        for fac in ("rest_ratio", "signal_ratio", "pi_layer", "pair"):
            r[f"scenario_engine_by_{fac}"] = {str(k): f"{int(g[ecol].sum())}/{len(g)}" for k, g in src.groupby(fac)}
        r["scenario_engine_by_SR_pi"] = {f"SR{sr:g}_pi{pi:g}": f"{int(g[ecol].sum())}/{len(g)}" for (sr, pi), g in src.groupby(["signal_ratio", "pi_layer"])}
        inc[lab] = r
    inc["_source"] = "figures/cell_contrasts.csv engine_inside_lead10, engine_inside_all_1to10, E_point10_inside_envelope, E_point_inside_all_1to10; figures/event30_contrasts.csv engine_inside_lead30, E_point30_inside_envelope"
    # inclusion split by whether the record determines the contrast (W10 <= delta)
    inc["lead10_engine_inside_by_determined"] = {str(k): f"{int(g.engine_inside_lead10.sum())}/{len(g)}" for k, g in cc.groupby("determined_lead10")}
    # sign of the scenario-engine contrast vs truth sign (P1 negative, P2 positive by construction; truth signs stored)
    cc2 = cc.merge(wt[["case_id", "pair", "E_true10"]], on=["case_id", "pair"])
    sign_ok = np.sign(cc2.engine_E10) == np.sign(cc2.E_true10)
    inc["lead10_engine_sign_matches_truth"] = dict(total=f"{int(sign_ok.sum())}/{len(cc2)}",
                                                   by_SR={f"SR{sr:g}": f"{int(sign_ok[cc2.signal_ratio == sr].sum())}/{int((cc2.signal_ratio == sr).sum())}" for sr in (0.5, 2.0)},
                                                   by_pair={p: f"{int(sign_ok[cc2.pair == p].sum())}/{int((cc2.pair == p).sum())}" for p in (P1, P2)})
    cc2["abs_err_over_absE"] = (cc2.engine_E10 - cc2.E_true10).abs() / cc2.E_true10.abs()
    cc2["ref_E10"] = np.nan
    inc["lead10_engine_abs_error_over_absEtrue_median"] = {f"SR{sr:g}": float(cc2[cc2.signal_ratio == sr].abs_err_over_absE.median()) for sr in (0.5, 2.0)}
    inc["_source_sign"] = "figures/cell_contrasts.csv engine_E10 joined to wb_repaired_v1_2/tables/W_table.csv E_true10 (evaluation only)"
    out["axis2_inclusion"] = inc
    # ---------------- axis 2 (b): width response
    wr = pd.read_csv(F / "width_response_blocks.csv")
    wresp = {}
    for tool, g in wr.groupby("tool"):
        v = g.spearman_W10_vs_width
        wresp[tool] = dict(blocks=len(g), undefined=int(g.spearman_undefined.sum()), positive=int((v > 0).sum()), negative=int((v < 0).sum()),
                           zero=int((v == 0).sum()), median=float(v.median()),
                           by_SR_pi={f"SR{sr:g}_pi{pi:g}": dict(positive=int((h.spearman_W10_vs_width > 0).sum()), negative=int((h.spearman_W10_vs_width < 0).sum()),
                                                                zero=int((h.spearman_W10_vs_width == 0).sum()), n=len(h), median=float(h.spearman_W10_vs_width.median()))
                                     for (sr, pi), h in g.groupby(["signal_ratio", "pi_layer"])})
    wresp["_source"] = "figures/width_response_blocks.csv spearman_W10_vs_width (5 rest levels per realization x SR x pi x pair block)"
    # width magnitudes by rest (P1), medians over 40 contrasts per rest level
    mag = {}
    for r, g in cc[cc.pair == P1].groupby("rest_ratio"):
        mag[f"rho{r:g}"] = dict(W10=float(g.W10.median()), proxy10=float(g.proxy10.median()), w_TF10=float(g.w_TF10.median()),
                                proxy_over_W10=float(g.proxy_over_W10.median()), w_TF_over_W10=float(g.w_TF_over_W10.median()))
    wresp["median_by_rest_P1_n40"] = mag
    wresp["median_by_SR_P1_n100"] = {f"SR{sr:g}": dict(W10=float(g.W10.median()), proxy10=float(g.proxy10.median()), w_TF10=float(g.w_TF10.median()))
                                     for sr, g in cc[cc.pair == P1].groupby("signal_ratio")}
    wresp["proxy_below_W10"] = f"{int(cc.proxy_below_W10.fillna(0).astype(bool).sum())}/{len(cc)}"
    wresp["proxy_over_W10_range_all"] = [float(cc.proxy_over_W10.min()), float(cc.proxy_over_W10.median()), float(cc.proxy_over_W10.max())]
    wresp["w_TF_over_W10_range_all"] = [float(cc.w_TF_over_W10.min()), float(cc.w_TF_over_W10.median()), float(cc.w_TF_over_W10.max())]
    wresp["median_by_determined_lead10"] = {str(k): dict(n=len(g), W10=float(g.W10.median()), proxy10=float(g.proxy10.median()), w_TF10=float(g.w_TF10.median()))
                                            for k, g in cc.groupby("determined_lead10")}
    wresp["_source_mag"] = "figures/cell_contrasts.csv W10, proxy10, w_TF10, proxy_over_W10, w_TF_over_W10, proxy_below_W10"
    out["axis2_width"] = wresp
    # ---------------- lead 30 magnitudes
    e = e30[e30.pair == P1]
    out["lead30"] = {f"rho{r:g}": dict(W30=float(g.W30.median()), proxy30=float(g.proxy30.median()), w_TF30=float(g.w_TF30.median())) for r, g in e.groupby("rest_ratio")}
    out["lead30_source"] = "figures/event30_contrasts.csv W30, proxy30, w_TF30 (P1, median over 40 contrasts per rest level)"
    # ---------------- numerical record
    v = man["verification"]
    pass
    out["numerics"] = dict(status=man["status_counts"], flags=man["W_flag_counts"], n_contrasts_with_W=man["n_contrasts_with_W"],
                           flags_by_SR_pi={f"SR{sr:g}_pi{pi:g}": dict(sparse_G=int(g["flags"].fillna("").str.contains("sparse_G").sum()),
                                                                      not_converged=int(g["flags"].fillna("").str.contains("not_converged").sum()), n=len(g))
                                           for (sr, pi), g in wt1.reset_index().groupby(["signal_ratio", "pi_layer"])},
                           n_accepted_x1_P1=dict(min=int(wt1.n_accepted_x1.min()), median=float(wt1.n_accepted_x1.median()), max=int(wt1.n_accepted_x1.max()),
                                                 n_below_200=int((wt1.n_accepted_x1 < 200).sum())),
                           global512_zero=v["global512_zero_accepted"], verification=v, runtime=man["runtime"],
                           source="wb_repaired_v1_2/wb_manifest.json; W_table.csv flags, n_accepted_x1 (P1 rows = one per case)")
    pc = comp["per_case"]
    mo = np.array([c["pairs"][P1].get("max_old_outside_new_m", 0.0) for c in pc if "old_W10" in c["pairs"][P1]])
    out["search_path"] = dict(summary=comp["summary"], n_cases_with_old=int(mo.size), n_old_outside_gt_1e6_m=int((mo > 1e-6).sum()),
                              n_old_outside_gt_1e4_m=int((mo > 1e-4).sum()), n_old_outside_gt_1e3_m=int((mo > 1e-3).sum()), max_m=float(mo.max()),
                              source="repair_comparison.json per_case[].pairs.P1_continue_vs_stop.max_old_outside_new_m (stored field, thresholds counted)")
    t1 = wt1.reset_index()
    ratio_tau = (t1.rmse_tol_m / t1.rmse_min_m) ** 2
    out["tolerance"] = dict(tau_over_sse_min=dict(min=float(ratio_tau.min()), median=float(ratio_tau.median()), max=float(ratio_tau.max())),
                            n_eff=dict(min=float(t1.n_eff.min()), median=float(t1.n_eff.median()), max=float(t1.n_eff.max())),
                            rmse_min_m=dict(min=float(t1.rmse_min_m.min()), median=float(t1.rmse_min_m.median()), max=float(t1.rmse_min_m.max())),
                            delta10_P1_median_by_SR={f"SR{sr:g}": float(g.delta10.median()) for sr, g in t1.groupby("signal_ratio")},
                            absE10_P1_median_by_SR={f"SR{sr:g}": float(g.E_true10.abs().median()) for sr, g in t1.groupby("signal_ratio")},
                            source="wb_repaired_v1_2/tables/W_table.csv P1 rows: rmse_tol_m, rmse_min_m (tau/SSE_min = (rmse_tol/rmse_min)^2), n_eff, delta10, E_true10")
    cl = pd.read_csv(WB / "tables/curves_long.csv")
    cl = cl[cl.lead <= 10].copy(); cl["fail"] = cl.W > cl.delta
    first = cl[cl.fail].groupby(["case_id", "pair"]).lead.min()
    l1 = cl[cl.lead == 1]
    out["first_lead_W_above_delta_1to10"] = dict(counts={int(k): int(v) for k, v in first.value_counts().sort_index().items()}, n_contrasts_failing=int(first.size),
                                                  W1_median_m=float(l1.W.median()), delta1_median_m=float(l1.delta.median()), n_W1_above_delta=int((l1.W > l1.delta).sum()),
                                                  source="wb_repaired_v1_2/tables/curves_long.csv W, delta, leads 1-10")
    (P / "report_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
