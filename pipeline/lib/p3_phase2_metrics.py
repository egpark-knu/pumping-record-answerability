"""D03 Phase 2 row metrics and figure-table mapping (B). Frozen before any official score; no thresholds are chosen
from results. Envelope inclusion uses the inherited v1.2 collector tolerance (lib/p3_wb_collect_repaired_v1_2.TOL_IN
= 1e-12 m, imported unchanged).

Per row (case x pair x track x lead):
  truth class      : 'nonfinite' | 'zero' (E_true == 0 exactly) | 'nonzero' (finite, != 0)
  small flags      : small_001 / small_02 = truth nonzero AND |E_true| <= 0.001 / 0.02 m (flags only; rows stay in denominators)
  ratio_defined    : truth nonzero AND E_tool finite; effect_ratio = E_tool / E_true (signed, never clipped; E_tool = 0 gives 0)
  sign_defined     : same mask; sign_agreement = sign(E_tool) == sign(E_true) (E_tool = 0 counts as disagreement)
  abs_error_m      : |E_tool - E_true| when both finite
  W_invalid        : W < 0 or W nonfinite (flagged); error_over_W defined only when W finite and W > 1e-6 m
  envelope_includes: inf - TOL_IN <= E_tool <= sup + TOL_IN (undefined when E_tool or the envelope is nonfinite)
All rows are retained; every summary reports total and each denominator.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from p3_wb_collect_repaired_v1_2 import TOL_IN  # inherited, 1e-12 m

EPS_SMALL = (0.001, 0.02)
W_FLOOR = 1e-6
CONTRACT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))) / "results/phase2/FIGURE_INPUT_CONTRACT.json"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def row_metrics(E_true, E_tool, W=float("nan"), env_inf=float("nan"), env_sup=float("nan")) -> dict:
    et, eo, w, lo, hi = map(_f, (E_true, E_tool, W, env_inf, env_sup))
    if not math.isfinite(et):
        tcls = "nonfinite"
    elif et == 0.0:
        tcls = "zero"
    else:
        tcls = "nonzero"
    nz = tcls == "nonzero"
    ratio_def = nz and math.isfinite(eo)
    out = dict(truth_class=tcls, small_001=bool(nz and abs(et) <= EPS_SMALL[0]), small_02=bool(nz and abs(et) <= EPS_SMALL[1]),
               ratio_defined=ratio_def, effect_ratio=(eo / et) if ratio_def else None,
               sign_defined=ratio_def, sign_agreement=(bool(np.sign(eo) == np.sign(et)) if ratio_def else None),
               abs_error_m=(abs(eo - et) if (math.isfinite(eo) and math.isfinite(et)) else None))
    w_invalid = (not math.isfinite(w)) or w < 0
    out["W_invalid"] = bool(w_invalid)
    out["W_negative"] = bool(math.isfinite(w) and w < 0)
    out["error_over_W"] = (out["abs_error_m"] / w) if (out["abs_error_m"] is not None and math.isfinite(w) and w > W_FLOOR) else None
    env_ok = math.isfinite(lo) and math.isfinite(hi) and lo <= hi
    out["envelope_includes"] = (bool(lo - TOL_IN <= eo <= hi + TOL_IN) if (env_ok and math.isfinite(eo)) else None)
    return out


def denominators(rows: list[dict]) -> dict:
    return dict(n_total=len(rows), n_truth_nonfinite=sum(r["truth_class"] == "nonfinite" for r in rows),
                n_truth_zero=sum(r["truth_class"] == "zero" for r in rows), n_truth_nonzero=sum(r["truth_class"] == "nonzero" for r in rows),
                n_small_001=sum(r["small_001"] for r in rows), n_small_02=sum(r["small_02"] for r in rows),
                n_ratio_defined=sum(r["ratio_defined"] for r in rows), n_sign=sum(r["sign_defined"] for r in rows),
                n_sign_agree=sum(bool(r["sign_agreement"]) for r in rows if r["sign_defined"]),
                n_error_over_W=sum(r["error_over_W"] is not None for r in rows), n_W_invalid=sum(r["W_invalid"] for r in rows),
                n_W_negative=sum(r["W_negative"] for r in rows), n_envelope=sum(r["envelope_includes"] is not None for r in rows),
                n_envelope_in=sum(bool(r["envelope_includes"]) for r in rows if r["envelope_includes"] is not None))


def contract_columns() -> list[str]:
    c = json.load(open(CONTRACT))
    cols = c.get("columns") or c.get("table", {}).get("columns")
    if isinstance(cols, dict):
        return list(cols)
    return [x["name"] if isinstance(x, dict) else x for x in cols]


def cell_metrics_row(d: dict, pair: str, track: str, quantity: str, lead, W_m, W_status, E_true, E_tool, env_inf, env_sup, data_status):
    """One figure-contract row (FIGURE_INPUT_CONTRACT.md). d is the case 'derived' record."""
    m = row_metrics(E_true, E_tool, W_m, env_inf, env_sup)
    b = lambda v: "" if v is None else int(bool(v))  # noqa: E731
    return dict(case_id=d["case_id"], storage_type=d["storage_type"], T_m2_d=d["T_m2_d"], rho=d["rho_nominal"], N_transitions=d["N_nominal"],
                realization=d["realization"], pair_id=pair, track=track, quantity=quantity, lead_day="" if lead is None else lead, W_m=W_m,
                W_status=W_status, SR=d["SR"], pi_r=d["pi_r"], rest_over_response=d["rest_over_t95"], E_true_m=E_true, E_tool_m=E_tool,
                effect_ratio="" if m["effect_ratio"] is None else m["effect_ratio"], ratio_defined=int(m["ratio_defined"]),
                sign_agreement=b(m["sign_agreement"]), error_over_W="" if m["error_over_W"] is None else m["error_over_W"],
                envelope_includes=b(m["envelope_includes"]), data_status=data_status)


def write_cell_metrics(rows: list[dict], path: Path):
    cols = contract_columns()
    missing = [c for c in cols if any(c not in r for r in rows)]
    if missing:
        raise ValueError(f"rows miss contract columns: {missing}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
