"""B independent recomputation of every number quoted in results/phase2/REPORT.md (D03 Phase 2), from persisted official
artifacts only: analysis/cell_metrics.csv (re-derived and byte-compared), wb/W/*.json, wb/truth_eval/*.json,
wb/reference/*.json, tools/timesfm_*_H*.npz, cases/derived_manifest.json. Protocol section 9 definitions; no threshold,
no significance test. Cross-checks C's analysis/SCIENTIFIC_READOUT.json where both compute the same quantity.

Writes results/phase2/analysis/B_REPORT_NUMBERS.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_phase2_metrics as mt  # noqa: E402

P2 = ROOT / "results/phase2"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
TRACKS = ("raw", "exp10", "exp30", "exp90")
LAYERS = ("confined_T50", "confined_T500", "leaky_T50", "leaky_T500", "unconfined_T50", "unconfined_T500")
RHOS, NS = (0.25, 1.0, 4.0), (2, 6, 18)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summ(x, ref=None):
    x = np.asarray(x, float)
    out = dict(n=int(x.size), median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
               frac_negative=float(np.mean(x < 0)), n_negative=int(np.sum(x < 0)), median_abs=float(np.median(np.abs(x))))
    if ref is not None:
        ref = np.asarray(ref, float)
        ok = np.isfinite(ref) & (ref > 1e-6)
        out["median_relative"] = float(np.median(x[ok] / ref[ok])) if ok.any() else None
        out["n_relative"] = int(ok.sum())
    return out


def main():
    # 0. cell_metrics.csv reproducibility: rerun the B joiner into /tmp and compare bytes
    tmp = Path("/tmp/p3_phase2_cellmetrics_recheck")
    subprocess.run([sys.executable, str(ROOT / "lib/p3_phase2_cell_metrics.py"), "--out", str(tmp), "--provenance", "official_analysis"], check=True,
                   capture_output=True)
    reproduced = sha(tmp / "cell_metrics.csv") == sha(P2 / "analysis/cell_metrics.csv")
    man = {m["case_id"]: m for m in json.load(open(P2 / "cases/derived_manifest.json"))}
    rows = list(csv.DictReader(open(P2 / "analysis/cell_metrics.csv")))
    assert len(rows) == 9720
    # index: (case, pair, track, quantity, lead) -> row
    idx = {(r["case_id"], r["pair_id"], r["track"], r["quantity"], r["lead_day"]): r for r in rows}
    f = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    # block keys
    blocks = sorted({(m["realization"], m["sid"]) for m in man.values()})
    cid = lambda real, sid, rho, N: f"r{real:02d}_{sid.rsplit('_T', 1)[0]}_T{sid.rsplit('_T', 1)[1]}_rho{rho:g}_N{N}"  # noqa: E731

    def val(real, sid, rho, N, pair, track, qty, lead, col):
        return f(idx[(cid(real, sid, rho, N), pair, track, qty, str(lead) if lead is not None else "")][col])

    def contrasts(col, pair, track, qty, lead, which, lo, hi, scope=None):
        """Paired contrasts within blocks. which='N': W(N=hi)-W(N=lo) at each rho; 'rho': W(rho=hi)-W(rho=lo) at each N."""
        d, ref, lay = [], [], []
        for real, sid in blocks:
            if scope and not scope(sid):
                continue
            others = RHOS if which == "N" else NS
            for o in others:
                if which == "N":
                    a, b = val(real, sid, o, hi, pair, track, qty, lead, col), val(real, sid, o, lo, pair, track, qty, lead, col)
                else:
                    a, b = val(real, sid, hi, o, pair, track, qty, lead, col), val(real, sid, lo, o, pair, track, qty, lead, col)
                if np.isfinite(a) and np.isfinite(b):
                    d.append(a - b); ref.append(b); lay.append(sid)
        return np.array(d), np.array(ref)

    scopes = dict(pooled=None, **{l: (lambda s, l=l: s == l) for l in LAYERS},
                  **{st: (lambda s, st=st: s.startswith(st + "_")) for st in ("confined", "leaky", "unconfined")})

    def readout(col, pair, track, qty, lead):
        out = {}
        for sname, sc in scopes.items():
            r = {}
            for which, pairs_ in (("N", ((2, 18), (2, 6), (6, 18))), ("rho", ((0.25, 4.0), (0.25, 1.0), (1.0, 4.0)))):
                for lo, hi in pairs_:
                    d, ref = contrasts(col, pair, track, qty, lead, which, lo, hi, sc)
                    r[f"{which}_{hi:g}_minus_{lo:g}"] = summ(d, ref)
            main_N, main_r = r["N_18_minus_2"], r["rho_4_minus_0.25"]
            r["ratio_medabs_N_over_rho"] = main_N["median_abs"] / main_r["median_abs"] if main_r["median_abs"] > 0 else None
            r["ratio_medabs_rho_over_N"] = main_r["median_abs"] / main_N["median_abs"] if main_N["median_abs"] > 0 else None
            out[sname] = r
        return out

    res = dict(inputs=dict(cell_metrics_sha256=sha(P2 / "analysis/cell_metrics.csv"), cell_metrics_reproduced_bytewise=reproduced,
                           derived_manifest_sha256=sha(P2 / "cases/derived_manifest.json"), wb_manifest_sha256=sha(P2 / "wb/wb_manifest.json"),
                           tool_archives={f"{t}_H{h}": sha(P2 / f"tools/timesfm_{t}_H{h}.npz") for t in TRACKS for h in (10, 30)},
                           protocol_sha256=(P2 / "protocol.sha256").read_text().split()[0]))
    # 1. Width: primary raw W(10), W(30) P1 (W is track-independent; raw rows), plus P2 and max 1-10
    res["W"] = {f"P1_lead{k}": readout("W_m", PAIRS[0], "raw", "lead", k) for k in (10, 30)}
    res["W"]["P2_lead10"] = readout("W_m", PAIRS[1], "raw", "lead", 10)
    res["W"]["P2_lead30"] = readout("W_m", PAIRS[1], "raw", "lead", 30)
    res["W"]["P1_max1to10_supp"] = readout("W_m", PAIRS[0], "raw", "max_days_1_10", None)
    # level medians (cell medians over realizations) per layer, P1
    lv = {}
    for sid in LAYERS:
        for k in (10, 30):
            for rho in RHOS:
                for N in NS:
                    ws = [val(r, sid, rho, N, PAIRS[0], "raw", "lead", k, "W_m") for r in range(10)]
                    et = [abs(val(r, sid, rho, N, PAIRS[0], "raw", "lead", k, "E_true_m")) for r in range(10)]
                    lv[f"{sid}|{k}|{rho:g}|{N}"] = dict(W_median_m=float(np.median(ws)), absEtrue_median_m=float(np.median(et)))
    res["W_cell_medians_P1"] = lv
    # 2. Tool (R3): abs error and signed ratio contrasts, every track, both leads, P1; plus P2 check of ratio equality
    r3 = {}
    for tr in TRACKS:
        for k in (10, 30):
            absrows = defaultdict(dict)
            for (c, p, t, q, l), r in idx.items():
                if t == tr and q == "lead" and l == str(k) and p == PAIRS[0]:
                    absrows[c] = abs(f(r["E_tool_m"]) - f(r["E_true_m"]))
            # write abs error into idx-like access
            for c, v in absrows.items():
                idx[(c, PAIRS[0], tr, "lead", str(k))]["_abs_err"] = v
            r3[f"{tr}|{k}"] = dict(abs_error=readout("_abs_err", PAIRS[0], tr, "lead", k), signed_ratio=readout("effect_ratio", PAIRS[0], tr, "lead", k))
    res["R3"] = r3
    # tool denominators and medians per track x lead x pair x layer
    den = {}
    for tr in TRACKS:
        for k in (10, 30):
            for p in PAIRS:
                for sname, sc in (("pooled", None), *((l, l) for l in LAYERS)):
                    rs = [r for (c, pp, t, q, l), r in idx.items() if t == tr and q == "lead" and l == str(k) and pp == p and (sc is None or man[c]["sid"] == sc)]
                    rm = [mt.row_metrics(f(r["E_true_m"]), f(r["E_tool_m"]), f(r["W_m"]), np.nan, np.nan) for r in rs]
                    dd = mt.denominators(rm)
                    ratios = np.array([x["effect_ratio"] for x in rm if x["ratio_defined"]], float)
                    inc = [r["envelope_includes"] for r in rs]
                    dd.update(n_envelope=sum(x != "" for x in inc), n_envelope_in=sum(x == "1" for x in inc),
                              median_ratio=float(np.median(ratios)), q25_ratio=float(np.percentile(ratios, 25)), q75_ratio=float(np.percentile(ratios, 75)),
                              frac_ratio_lt1=float(np.mean(ratios < 1)), n_ratio_negative=int(np.sum(ratios < 0)),
                              median_abs_error_m=float(np.median([x["abs_error_m"] for x in rm])),
                              median_error_over_W=float(np.median([x["error_over_W"] for x in rm if x["error_over_W"] is not None])))
                    den[f"{tr}|{k}|{p}|{sname}"] = dd
    res["tool_summary"] = den
    # filter vs raw, paired per row (P1): ratio difference, abs error difference
    fvr = {}
    for tr in TRACKS[1:]:
        for k in (10, 30):
            dr, de = [], []
            for c in man:
                a, b = idx[(c, PAIRS[0], tr, "lead", str(k))], idx[(c, PAIRS[0], "raw", "lead", str(k))]
                dr.append(f(a["effect_ratio"]) - f(b["effect_ratio"]))
                de.append(abs(f(a["E_tool_m"]) - f(a["E_true_m"])) - abs(f(b["E_tool_m"]) - f(b["E_true_m"])))
            fvr[f"{tr}|{k}"] = dict(ratio_minus_raw=summ(dr), abs_error_minus_raw_m=summ(de))
    res["filter_minus_raw_P1"] = fvr
    # 3. Collapse (protocol 9.4): cell medians log W on log SR, log pi_r, log rho_realized, +/- layer indicators
    col = {}
    for k in (10, 30):
        X0, X1, y, meta = [], [], [], []
        for sid in LAYERS:
            for rho in RHOS:
                for N in NS:
                    cs = [cid(r, sid, rho, N) for r in range(10)]
                    W = np.median([val(r, sid, rho, N, PAIRS[0], "raw", "lead", k, "W_m") for r in range(10)])
                    SR = np.median([man[c]["SR"] for c in cs]); pi = man[cs[0]]["pi_r"]; rr = np.median([man[c]["rho_realized"] for c in cs])
                    base = [1.0, np.log(SR), np.log(pi), np.log(rr)]
                    X0.append(base); X1.append(base + [1.0 if sid == l else 0.0 for l in LAYERS[1:]]); y.append(np.log(W)); meta.append((sid, rho, N, W, SR))
        y = np.array(y)

        def r2(X):
            X = np.array(X); b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
            return float(1 - e @ e / np.sum((y - y.mean()) ** 2)), int(np.linalg.matrix_rank(X))
        a0, k0 = r2(X0); a1, k1 = r2(X1)
        # identical-shape pair: confined_T50 vs leaky_T500 (same a, b; gain ratio)
        pr = [m[3] for m in meta if m[0] == "confined_T50"], [m[3] for m in meta if m[0] == "leaky_T500"]
        col[k] = dict(n_cells=len(y), r2_without_layers=a0, rank_without=k0, r2_with_layers=a1, rank_with=k1,
                      shape_pair_W_ratio_confinedT50_over_leakyT500=dict(median=float(np.median(np.array(pr[0]) / np.array(pr[1]))),
                                                                         min=float(np.min(np.array(pr[0]) / np.array(pr[1]))), max=float(np.max(np.array(pr[0]) / np.array(pr[1])))))
    lSR = [np.log(np.median([m["SR"] for m in man.values() if m["sid"] == l])) for l in LAYERS]
    lpi = [np.log(next(m["pi_r"] for m in man.values() if m["sid"] == l)) for l in LAYERS]
    gains = {l: next(m["gain_m_per_m3d"] for m in man.values() if m["sid"] == l) for l in LAYERS}
    res["collapse"] = dict(by_lead=col, pearson_logSR_logpi_layers=float(np.corrcoef(lSR, lpi)[0, 1]), gain_ratio_confinedT50_over_leakyT500=gains["confined_T50"] / gains["leaky_T500"])
    # 4. Volume / physical record
    V = defaultdict(dict)
    for m in man.values():
        V[(m["sid"], m["realization"], m["N_nominal"])][m["rho_nominal"]] = m["V_m3"]
    loss = [(v[0.25] - v[4.0]) for v in V.values()]
    lossf = [(v[0.25] - v[4.0]) / v[0.25] for v in V.values()]
    Nm = defaultdict(list)
    for m in man.values():
        Nm[(m["sid"], m["realization"], m["rho_nominal"])].append((m["V_m3"], m["mean_outside_rest_m3d"]))
    res["physical_record"] = dict(
        V_m3_range=[min(m["V_m3"] for m in man.values()), max(m["V_m3"] for m in man.values())],
        mean_outside_rest_range_m3d=[min(m["mean_outside_rest_m3d"] for m in man.values()), max(m["mean_outside_rest_m3d"] for m in man.values())],
        calendar_mean_range_m3d=[min(m["mean_context_m3d"] for m in man.values()), max(m["mean_context_m3d"] for m in man.values())],
        q0_range_m3d=[min(m["q0_m3d"] for m in man.values()), max(m["q0_m3d"] for m in man.values())],
        kappa_range=[min(m["kappa"] for m in man.values()), max(m["kappa"] for m in man.values())],
        cross_rho_V_loss_m3=[min(loss), max(loss)], cross_rho_V_loss_fraction=[min(lossf), max(lossf)],
        N_matching_max_rel_spread_V=max((max(a for a, _ in g) - min(a for a, _ in g)) / max(a for a, _ in g) for g in Nm.values()),
        N_matching_max_rel_spread_mean=max((max(b for _, b in g) - min(b for _, b in g)) / max(b for _, b in g) for g in Nm.values()),
        SR_range_by_layer={l: [min(m["SR"] for m in man.values() if m["sid"] == l), float(np.median([m["SR"] for m in man.values() if m["sid"] == l])),
                               max(m["SR"] for m in man.values() if m["sid"] == l)] for l in LAYERS},
        layer={l: {k: next(m[k] for m in man.values() if m["sid"] == l) for k in ("c_d", "a_d", "b", "lambda_m", "pi_r", "t50_d", "t95_d", "gain_m_per_m3d",
                                                                                     "layer_rest_end_day")} for l in LAYERS},
        R_days_by_layer={l: sorted({(m["rho_nominal"], m["R_days"]) for m in man.values() if m["sid"] == l}) for l in LAYERS},
        rho_realized_range={str(rho): [min(m["rho_realized"] for m in man.values() if m["rho_nominal"] == rho), max(m["rho_realized"] for m in man.values() if m["rho_nominal"] == rho)] for rho in RHOS},
        tail_over_t95={str(N): [min(m["tail_on_days"] / m["t95_d"] for m in man.values() if m["N_nominal"] == N), max(m["tail_on_days"] / m["t95_d"] for m in man.values() if m["N_nominal"] == N)] for N in NS})
    # 5. Pair identity and truth identity (from W / truth JSON)
    gap, half, tr_id = 0.0, 0.0, 0.0
    for c in man:
        W = json.loads((P2 / "wb/W" / f"{c}.json").read_text())["envelope"]
        T = json.loads((P2 / "wb/truth_eval" / f"{c}.json").read_text())
        w1, w2 = np.array(W[PAIRS[0]]["W"]), np.array(W[PAIRS[1]]["W"])
        gap = max(gap, float(np.max(np.abs(w1 - w2)))); half = max(half, float(np.max(np.abs(w2 - 0.5 * w1))))
        tr_id = max(tr_id, float(np.max(np.abs(np.array(T["E_true"][PAIRS[1]]) + 0.5 * np.array(T["E_true"][PAIRS[0]])))))
    res["pair_identity"] = dict(max_abs_W_P1_minus_W_P2_m=gap, max_abs_W_P2_minus_half_W_P1_m=half, max_abs_Etrue_P2_plus_half_P1_m=tr_id)
    # 6. Reference: ratio/sign/inclusion vs truth and W; draws; solver success; E_true within reference q10-q90 (supplement)
    ref = defaultdict(list)
    for c, m in man.items():
        R = json.loads((P2 / "wb/reference" / f"{c}.json").read_text())
        W = json.loads((P2 / "wb/W" / f"{c}.json").read_text())["envelope"]
        T = json.loads((P2 / "wb/truth_eval" / f"{c}.json").read_text())
        for p in PAIRS:
            for k in (10, 30):
                Er, Et = R["pairs"][p]["E_point"][k - 1], T["E_true"][p][k - 1]
                rm = mt.row_metrics(Et, Er, W[p]["W"][k - 1], W[p]["inf"][k - 1], W[p]["sup"][k - 1])
                q10, q90 = R["pairs"][p].get("E_q10", [np.nan] * 30)[k - 1], R["pairs"][p].get("E_q90", [np.nan] * 30)[k - 1]
                ref[(p, k)].append(dict(ratio=rm["effect_ratio"], sign=rm["sign_agreement"], inc=rm["envelope_includes"], truth_in_q=bool(min(q10, q90) <= Et <= max(q10, q90)),
                                        draws=R["n_param_draws_accepted"], solver=R.get("solver_success"), sid=m["sid"], case=c))
    res["reference"] = {f"{p}|{k}": dict(n=len(v), median_ratio=float(np.median([x["ratio"] for x in v])), sign_agree=sum(bool(x["sign"]) for x in v),
                                         envelope_in=sum(bool(x["inc"]) for x in v), outside_cases=[x["case"] for x in v if not x["inc"]],
                                         truth_in_q10_q90=sum(x["truth_in_q"] for x in v), draws_lt_1000=sum(x["draws"] < 1000 for x in v),
                                         solver_success_false=sum(x["solver"] is False for x in v)) for (p, k), v in ref.items()}
    # 7. Marginal-width proxy (supplement only; not a joint interval, not compared with W)
    mw = {}
    for tr in TRACKS:
        for k in (10, 30):
            with np.load(P2 / f"tools/timesfm_{tr}_H{k}.npz", allow_pickle=False) as z:
                mw[f"{tr}|{k}"] = dict(median_E_width_proxy_m=float(np.median(z["E_width_proxy"][:, k - 1])), median_marginal_width_m=float(np.median(z["marginal_widths"][:, k - 1])),
                                       E_point_equals_E_tool=bool(np.array_equal(z["E_point"], z["E_tool"])))
    res["marginal_proxy_supplement"] = mw
    # 8. Cross-check vs C readout
    C = json.load(open(P2 / "analysis/SCIENTIFIC_READOUT.json"))
    chk = {}
    try:
        chk["R1_median_N_day10"] = (res["W"]["P1_lead10"]["pooled"]["N_18_minus_2"]["median"], C["r1_day10"]["pooled"]["median"] if "pooled" in C["r1_day10"] else None)
    except Exception as e:  # structure differences are recorded, not hidden
        chk["R1_structure"] = repr(e)
    chk["C_tool_raw10_median_ratio"] = (res["tool_summary"]["raw|10|P1_continue_vs_stop|pooled"]["median_ratio"], C["tool_denominators_p1"]["raw|10"]["median_effect_ratio"])
    chk["C_tool_raw30_median_ratio"] = (res["tool_summary"]["raw|30|P1_continue_vs_stop|pooled"]["median_ratio"], C["tool_denominators_p1"]["raw|30"]["median_effect_ratio"])
    chk["C_collapse"] = (res["collapse"]["by_lead"], C.get("collapse"))
    res["cross_check_C"] = chk
    (P2 / "analysis/B_REPORT_NUMBERS.json").write_text(json.dumps(res, indent=1, default=float))
    print("reproduced cell_metrics:", reproduced)


if __name__ == "__main__":
    main()
