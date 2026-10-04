"""Phase 2 scientific aggregation for the frozen protocol section 9 (new code, worker C).

Reads an official cell_metrics.csv written by lib/p3_phase2_cell_metrics.py. Does not
compute envelopes, forecasts, filters, or the matched reference fit. Imports the frozen
row-metric module unchanged. Refuses a table that is not the full official grid.

The ratio used for R1 and R2 is the median of the absolute paired differences divided
by the median of the other contrast's absolute paired differences. It is not the
absolute value of the median difference. Both numbers are stored under different keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_phase2_metrics as mt  # noqa: E402

P2 = ROOT / "results/phase2"
PROTOCOL_SHA = "0b360edd4ecc9bc601376eeaba1f78d5511096c67cb2bd42810ffe010713b794"
TRACKS = ("raw", "exp10", "exp30", "exp90")
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
P1 = PAIRS[0]
RHOS = (0.25, 1.0, 4.0)
NS = (2, 6, 18)
LEADS = (10, 30)
W_FLOOR = mt.W_FLOOR
N_LEAD = 8640
N_MAX = 1080
LAYER_ORDER = (
    "confined_T50",
    "confined_T500",
    "leaky_T50",
    "leaky_T500",
    "unconfined_T50",
    "unconfined_T500",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(values):
    arr = np.asarray(list(values), float)
    return arr[np.isfinite(arr)]


def _median(values):
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _iqr(values):
    arr = _finite(values)
    if arr.size == 0:
        return None
    try:
        q75, q25 = np.percentile(arr, [75, 25], method="linear")
    except TypeError:
        q75, q25 = np.percentile(arr, [75, 25], interpolation="linear")
    return float(q75 - q25)


def _direction(median):
    if median is None:
        return "undefined"
    if median > 0:
        return "positive"
    if median < 0:
        return "negative"
    return "zero"


def summarise_deltas(deltas, relatives, n_blocks, n_missing):
    finite = _finite(deltas)
    rel = _finite(relatives)
    n_neg = int(np.sum(finite < 0)) if finite.size else 0
    med = _median(finite)
    med_abs = _median(np.abs(finite)) if finite.size else None
    return {
        "n_blocks": int(n_blocks),
        "n_missing_endpoint": int(n_missing),
        "n_finite": int(finite.size),
        "n_negative": n_neg,
        "fraction_negative": (None if finite.size == 0 else float(n_neg / finite.size)),
        "median_delta": med,
        "abs_of_median_delta": (None if med is None else float(abs(med))),
        "iqr_delta": _iqr(finite),
        "median_abs_delta": med_abs,
        "n_relative": int(rel.size),
        "median_relative": _median(rel),
    }


def _match_level(series, level):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.isclose(values, float(level), rtol=0, atol=1e-9)


def paired_contrast(frame, value_col, *, block_cols, condition_col, conditions, low, high, reference_is_low=True):
    """Δ = value(high) − value(low) inside each block, at each conditioning value."""
    rows = []
    for cond in conditions:
        part = frame.loc[_match_level(frame[condition_col], cond)].copy()
        low_rows = part.loc[_match_level(part["level"], low), block_cols + [value_col]]
        high_rows = part.loc[_match_level(part["level"], high), block_cols + [value_col]]
        merged = low_rows.merge(high_rows, on=block_cols, how="outer", suffixes=("_low", "_high"))
        n_missing = int(merged[f"{value_col}_low"].isna().sum() + merged[f"{value_col}_high"].isna().sum())
        # a block missing from one side has NaN; count blocks that lack either endpoint once
        lack = merged[f"{value_col}_low"].isna() | merged[f"{value_col}_high"].isna()
        n_missing = int(lack.sum())
        both = merged.loc[~lack]
        delta = both[f"{value_col}_high"].to_numpy(float) - both[f"{value_col}_low"].to_numpy(float)
        ref = both[f"{value_col}_low" if reference_is_low else f"{value_col}_high"].to_numpy(float)
        relative = np.full(delta.shape, np.nan)
        ok = np.isfinite(delta) & np.isfinite(ref) & (np.abs(ref) > W_FLOOR)
        relative[ok] = delta[ok] / ref[ok]
        rec = summarise_deltas(delta, relative, n_blocks=len(merged), n_missing=n_missing)
        rec["condition"] = float(cond) if not isinstance(cond, str) else cond
        rec["low"] = low
        rec["high"] = high
        rows.append(rec)
    return rows


def _with_level(frame, level_col):
    out = frame.copy()
    out["level"] = out[level_col].astype(float)
    return out


def scope_slices(frame):
    yield "pooled", frame
    for sid in LAYER_ORDER:
        yield f"layer:{sid}", frame.loc[frame["sid"] == sid]
    for storage in ("confined", "leaky", "unconfined"):
        yield f"storage:{storage}", frame.loc[frame["storage_type"] == storage]


def contrast_bundle(frame, value_col):
    """N contrasts at each rho, and rho contrasts at each N, for every scope."""
    block = ["sid", "realization"]
    out = {}
    for name, part in scope_slices(frame):
        if part.empty:
            out[name] = None
            continue
        n_part = _with_level(part, "N_transitions")
        r_part = _with_level(part, "rho")
        out[name] = {
            "N18_minus_N2_by_rho": paired_contrast(
                n_part, value_col, block_cols=block + ["rho"], condition_col="rho",
                conditions=RHOS, low=2, high=18,
            ),
            "rho4_minus_rho0.25_by_N": paired_contrast(
                r_part, value_col, block_cols=block + ["N_transitions"], condition_col="N_transitions",
                conditions=NS, low=0.25, high=4.0,
            ),
        }
    return out


def raw_contrasts(frame, value_col):
    """Return raw Δ arrays so medians are taken over blocks, not over medians."""
    block = ["sid", "realization"]
    collected = {}
    for name, part in scope_slices(frame):
        n_part = _with_level(part, "N_transitions")
        r_part = _with_level(part, "rho")
        n_deltas = []
        r_deltas = []
        for cond in RHOS:
            piece = n_part.loc[_match_level(n_part["rho"], cond)]
            low_rows = piece.loc[_match_level(piece["level"], 2), block + ["rho", value_col]]
            high_rows = piece.loc[_match_level(piece["level"], 18), block + ["rho", value_col]]
            merged = low_rows.merge(high_rows, on=block + ["rho"], suffixes=("_low", "_high"))
            lack = merged[f"{value_col}_low"].isna() | merged[f"{value_col}_high"].isna()
            delta = merged.loc[~lack, f"{value_col}_high"].to_numpy(float) - merged.loc[~lack, f"{value_col}_low"].to_numpy(float)
            n_deltas.append(delta)
        for cond in NS:
            piece = r_part.loc[_match_level(r_part["N_transitions"], cond)]
            low_rows = piece.loc[_match_level(piece["level"], 0.25), block + ["N_transitions", value_col]]
            high_rows = piece.loc[_match_level(piece["level"], 4.0), block + ["N_transitions", value_col]]
            merged = low_rows.merge(high_rows, on=block + ["N_transitions"], suffixes=("_low", "_high"))
            lack = merged[f"{value_col}_low"].isna() | merged[f"{value_col}_high"].isna()
            delta = merged.loc[~lack, f"{value_col}_high"].to_numpy(float) - merged.loc[~lack, f"{value_col}_low"].to_numpy(float)
            r_deltas.append(delta)
        n_all = np.concatenate(n_deltas) if n_deltas else np.array([])
        r_all = np.concatenate(r_deltas) if r_deltas else np.array([])
        collected[name] = {"N": n_all, "rho": r_all}
    return collected


def readout_from_raw(raw, *, primary: str):
    """primary 'N' for R1 and 'rho' for R2. Ratio is median|primary| / median|other|."""
    other = "rho" if primary == "N" else "N"
    views = {}
    for name, packs in raw.items():
        prim = _finite(packs[primary])
        comp = _finite(packs[other])
        med = _median(prim)
        med_abs_p = _median(np.abs(prim)) if prim.size else None
        med_abs_c = _median(np.abs(comp)) if comp.size else None
        ratio = None
        ratio_note = None
        if med_abs_p is None or med_abs_c is None:
            ratio_note = "undefined because one contrast has no finite paired difference"
        elif med_abs_c == 0.0:
            ratio_note = "undefined because the compared median absolute contrast is zero"
        else:
            ratio = float(med_abs_p / med_abs_c)
        views[name] = {
            "primary": primary,
            "direction_of_median": _direction(med),
            "median_primary": med,
            "abs_of_median_primary": None if med is None else float(abs(med)),
            "fraction_negative": None if prim.size == 0 else float(np.sum(prim < 0) / prim.size),
            "n_finite_primary": int(prim.size),
            "n_negative_primary": int(np.sum(prim < 0)) if prim.size else 0,
            "median_abs_primary": med_abs_p,
            "median_abs_compared": med_abs_c,
            "n_finite_compared": int(comp.size),
            "ratio_of_median_absolute_contrasts": ratio,
            "ratio_note": ratio_note,
        }
    directions = [views[f"layer:{sid}"]["direction_of_median"] for sid in LAYER_ORDER if f"layer:{sid}" in views]
    return {
        "pooled": views.get("pooled"),
        "by_layer": {sid: views.get(f"layer:{sid}") for sid in LAYER_ORDER},
        "by_storage": {s: views.get(f"storage:{s}") for s in ("confined", "leaky", "unconfined")},
        "layer_directions": directions,
        "layer_directions_differ": len(set(directions)) > 1,
        "definition": (
            "The ratio is median(|paired difference| of the primary contrast) divided by "
            "median(|paired difference| of the other contrast). It is not |median(difference)|. "
            "Primary differences are pooled over the 60 realization-by-layer blocks and the three "
            "conditioning values before the median is taken. Per-condition medians are in the contrast tables."
        ),
    }


def level_grid(frame, value_col):
    """Median of the cell value at each rho and N, including the intermediate levels."""
    out = {}
    for name, part in scope_slices(frame):
        cells = []
        for rho in RHOS:
            for count_n in NS:
                sl = part.loc[_match_level(part["rho"], rho) & _match_level(part["N_transitions"], count_n), value_col]
                finite = _finite(sl)
                cells.append({
                    "rho": rho,
                    "N": count_n,
                    "n_finite": int(finite.size),
                    "median": _median(finite),
                    "iqr": _iqr(finite),
                    "intermediate": bool(count_n == 6 or rho == 1.0),
                })
        out[name] = cells
    return out


def _design(frame, columns):
    blocks = [np.ones(len(frame))]
    for col in columns:
        blocks.append(frame[col].to_numpy(float))
    return np.column_stack(blocks)


def ols_r2(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if y.size == 0:
        return {"n": 0, "rank": 0, "r2": None, "note": "no rows"}
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    if ss_tot == 0.0:
        r2 = None
        note = "cell-median log W has zero variance on the rows used"
    else:
        r2 = float(1.0 - ss_res / ss_tot)
        note = None
    return {"n": int(y.size), "rank": int(rank), "n_columns": int(x.shape[1]), "r2": r2, "ss_tot": ss_tot, "ss_res": ss_res, "note": note}


def collapse_one(cells, lead):
    part = cells.loc[_match_level(cells["lead_day"], lead)].copy()
    n_cells = int(len(part))
    positive = part["median_W"].gt(0) & np.isfinite(part["median_W"]) & np.isfinite(part["log_SR"]) & np.isfinite(part["log_pi"]) & np.isfinite(part["log_rho"])
    n_nonpositive = int((~(part["median_W"].gt(0)) & np.isfinite(part["median_W"])).sum())
    n_nonfinite = int((~np.isfinite(part["median_W"])).sum())
    n_bad_predictor = int((np.isfinite(part["median_W"]) & part["median_W"].gt(0) & ~(np.isfinite(part["log_SR"]) & np.isfinite(part["log_pi"]) & np.isfinite(part["log_rho"]))).sum())
    used = part.loc[positive]
    y = np.log(used["median_W"].to_numpy(float))
    base_cols = ["log_SR", "log_pi", "log_rho"]
    x1 = _design(used, base_cols) if len(used) else np.zeros((0, 4))
    dummies = []
    dummy_names = []
    for sid in LAYER_ORDER[1:]:
        dummies.append((used["sid"] == sid).astype(float).to_numpy())
        dummy_names.append(sid)
    if len(used) and dummies:
        x2 = np.column_stack([x1] + dummies)
    else:
        x2 = x1
    m1 = ols_r2(y, x1)
    m2 = ols_r2(y, x2)
    pi_unique = {sid: int(part.loc[part["sid"] == sid, "pi_r"].nunique(dropna=True)) for sid in LAYER_ORDER}
    sr_unique = {sid: int(part.loc[part["sid"] == sid, "SR"].nunique(dropna=True)) for sid in LAYER_ORDER}
    pi_constant = all(v == 1 for v in pi_unique.values())
    return {
        "lead_day": lead,
        "n_cells": n_cells,
        "n_expected": 54,
        "n_nonpositive_median_W": n_nonpositive,
        "n_nonfinite_median_W": n_nonfinite,
        "n_nonfinite_predictor": n_bad_predictor,
        "n_used": int(len(used)),
        "model_logSR_logpi_logrho": m1,
        "model_plus_layer_indicators": m2,
        "layer_indicator_reference": LAYER_ORDER[0],
        "layer_indicators": dummy_names,
        "pi_r_nunique_within_layer": pi_unique,
        "SR_nunique_within_layer_cell_medians": sr_unique,
        "identification_note": (
            "Explained variance is the squared correlation of the projection, including when the design is rank deficient. "
            + ("Cell-median pi_r is constant within each layer, so log pi_r has no within-layer slope and is collinear with the layer indicators. " if pi_constant else "Cell-median pi_r is not constant within every layer. ")
            + "Cell-median SR is the median of the derived SR over the realizations in that cell. It is not forced to be constant inside a layer. "
            "Coefficients that share a column space are not reported as separate effects."
        ),
    }


def sr_only_pair(cells, manifest_layers):
    """confined T=50 against leaky T=500: same unit response, tenfold gain, SR-only description."""
    left = "confined_T50"
    right = "leaky_T500"
    left_meta = manifest_layers.get(left) or {}
    right_meta = manifest_layers.get(right) or {}
    gain_ratio = None
    if left_meta.get("gain_m_per_m3d") and right_meta.get("gain_m_per_m3d"):
        gain_ratio = float(left_meta["gain_m_per_m3d"] / right_meta["gain_m_per_m3d"])
    rows = []
    for lead in LEADS:
        for rho in RHOS:
            for count_n in NS:
                a = cells.loc[(cells["sid"] == left) & _match_level(cells["lead_day"], lead) & _match_level(cells["rho"], rho) & _match_level(cells["N_transitions"], count_n)]
                b = cells.loc[(cells["sid"] == right) & _match_level(cells["lead_day"], lead) & _match_level(cells["rho"], rho) & _match_level(cells["N_transitions"], count_n)]
                wa = None if a.empty else float(a["median_W"].iloc[0])
                wb = None if b.empty else float(b["median_W"].iloc[0])
                ratio = None
                if wa is not None and wb is not None and np.isfinite(wa) and np.isfinite(wb) and abs(wa) > W_FLOOR:
                    ratio = float(wb / wa)
                rows.append({"lead_day": lead, "rho": rho, "N": count_n, "median_W_confined_T50_m": wa, "median_W_leaky_T500_m": wb, "ratio_leakyT500_over_confinedT50": ratio})
    return {
        "layers": {"confined_T50": left_meta, "leaky_T500": right_meta},
        "gain_confinedT50_over_leakyT500": gain_ratio,
        "unit_response_a_equal": left_meta.get("a_d") == right_meta.get("a_d"),
        "unit_response_b_equal": left_meta.get("b") == right_meta.get("b"),
        "cells": rows,
        "statement": (
            "The protocol identifies confined T=50 and leaky T=500 as the pair with the same unit response and a tenfold gain difference. "
            "The measured gain ratio and the cell-median widths are reported here. "
            "This is an SR comparison. It is not a pure-storage contrast and it is not a universality claim."
        ),
    }


def layer_coupling(manifest):
    df = pd.DataFrame(manifest)
    layers = []
    for sid, part in df.groupby("sid"):
        sr_vals = part["SR"].to_numpy(float)
        pi_vals = part["pi_r"].to_numpy(float)
        layers.append({
            "sid": sid,
            "storage_type": part["storage_type"].iloc[0],
            "storage_value": float(part["storage_value"].iloc[0]),
            "T_m2_d": float(part["T_m2_d"].iloc[0]),
            "SR_median": float(np.median(sr_vals)),
            "SR_min": float(np.min(sr_vals)),
            "SR_max": float(np.max(sr_vals)),
            "SR_nunique": int(part["SR"].nunique()),
            "pi_r": float(np.median(pi_vals)) if np.nanmax(pi_vals) - np.nanmin(pi_vals) == 0 else None,
            "pi_r_nunique": int(part["pi_r"].nunique()),
            "a_d": float(part["a_d"].iloc[0]) if part["a_d"].nunique() == 1 else None,
            "b": float(part["b"].iloc[0]) if part["b"].nunique() == 1 else None,
            "gain_m_per_m3d": float(part["gain_m_per_m3d"].iloc[0]) if part["gain_m_per_m3d"].nunique() == 1 else None,
            "t95_d": float(part["t95_d"].iloc[0]) if part["t95_d"].nunique() == 1 else None,
        })
    layers = sorted(layers, key=lambda r: LAYER_ORDER.index(r["sid"]) if r["sid"] in LAYER_ORDER else 99)
    sr = np.array([r["SR_median"] for r in layers], float)
    pi = np.array([r["pi_r"] for r in layers], float)
    def corr(x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(np.log(x), np.log(y))[0, 1])
    return {
        "n_layers": len(layers),
        "layers": layers,
        "log_SR_log_pi_pearson": corr(sr, pi),
        "statement": (
            "The correlation uses one layer median of SR against that layer's pi_r. "
            "SR also varies inside a layer, across realizations and rest ratios, so the six-point correlation does not separate storage from resistance. "
            "pi_r is constant within a layer wherever pi_r_nunique is 1, and then a within-layer pi_r slope is not identified. "
            "No universality claim is made."
        ),
    }


def volume_report(manifest):
    df = pd.DataFrame(manifest)
    df["rho_key"] = df["rho_nominal"].astype(float).round(6)
    n_resid = []
    kappa_span = []
    for _, part in df.groupby(["sid", "realization", "rho_key"]):
        n_resid.append(float(part["V_m3"].max() - part["V_m3"].min()))
        kappa_span.append(float(part["kappa"].max() - part["kappa"].min()))
    cross = []
    for _, part in df.groupby(["sid", "realization", "N_nominal"]):
        got = {round(float(r.rho_nominal), 6): float(r.V_m3) for r in part.itertuples()}
        if not {0.25, 1.0, 4.0}.issubset(got):
            cross.append({"complete": False})
            continue
        cross.append({
            "complete": True,
            "V_rho0.25_minus_rho1": got[0.25] - got[1.0],
            "V_rho1_minus_rho4": got[1.0] - got[4.0],
            "V_rho0.25_minus_rho4": got[0.25] - got[4.0],
            "V_span_across_rho": max(got.values()) - min(got.values()),
        })
    ok = [c for c in cross if c["complete"]]

    def rng(key):
        vals = np.array([c[key] for c in ok], float)
        return {"min": float(vals.min()), "max": float(vals.max()), "n": int(vals.size)}

    return {
        "n_cases": int(len(df)),
        "n_matching_groups": int(len(n_resid)),
        "max_abs_V_difference_across_N_m3": float(np.max(np.abs(n_resid))) if n_resid else None,
        "kappa_span_across_N": {
            "min": float(np.min(kappa_span)) if kappa_span else None,
            "max": float(np.max(kappa_span)) if kappa_span else None,
            "median": float(np.median(kappa_span)) if kappa_span else None,
        },
        "cross_rho_volume_difference_m3": {k: rng(k) for k in (
            "V_rho0.25_minus_rho1", "V_rho1_minus_rho4", "V_rho0.25_minus_rho4", "V_span_across_rho",
        )},
        "n_incomplete_rho_groups": int(len(cross) - len(ok)),
        "reference_used": (
            "Within one layer, one realization, and one nominal rho, N=2, 6, and 18 are the matched schedules. "
            "The cross-rho range is the observed difference of total pumped volume V_m3. "
            "A longer rest lowers the total volume because the base schedule outside the rest is unchanged and the rest is zero."
        ),
    }


def _pearson_ok(x):
    return x


def p2_identity(frame):
    lead = frame.loc[frame["quantity"].eq("lead")].copy()
    keys = ["case_id", "track", "lead_day", "quantity"]
    a = lead.loc[lead["pair_id"] == PAIRS[0], keys + ["W_m", "E_true_m", "E_tool_m"]]
    b = lead.loc[lead["pair_id"] == PAIRS[1], keys + ["W_m", "E_true_m", "E_tool_m"]]
    m = a.merge(b, on=keys, suffixes=("_p1", "_p2"))
    w_gap = np.abs(m["W_m_p1"].to_numpy(float) - m["W_m_p2"].to_numpy(float))
    half_gap = np.abs(m["W_m_p2"].to_numpy(float) - 0.5 * m["W_m_p1"].to_numpy(float))
    ratio = m["W_m_p2"].to_numpy(float) / m["W_m_p1"].to_numpy(float)
    truth_gap = np.abs(m["E_true_m_p2"].to_numpy(float) + 0.5 * m["E_true_m_p1"].to_numpy(float))
    # W should not depend on the forecast track; check that too.
    raw = m.loc[m["track"] == "raw", ["case_id", "lead_day", "W_m_p1"]]
    track_gaps = []
    for track in TRACKS:
        if track == "raw":
            continue
        other = m.loc[m["track"] == track, ["case_id", "lead_day", "W_m_p1"]]
        both = raw.merge(other, on=["case_id", "lead_day"], suffixes=("_raw", "_other"))
        if len(both):
            track_gaps.append(float(np.nanmax(np.abs(both["W_m_p1_raw"] - both["W_m_p1_other"]))))
    return {
        "n_joined_rows": int(len(m)),
        "max_abs_W_p1_minus_W_p2_m": float(np.nanmax(w_gap)) if len(m) else None,
        "max_abs_W_p2_minus_half_W_p1_m": float(np.nanmax(half_gap)) if len(m) else None,
        "median_W_p2_over_W_p1": _median(ratio),
        "max_abs_Etrue_p2_plus_half_Etrue_p1_m": float(np.nanmax(truth_gap)) if len(m) else None,
        "n_nonfinite_W_gap": int(np.sum(~np.isfinite(w_gap))),
        "n_nonfinite_truth_gap": int(np.sum(~np.isfinite(truth_gap))),
        "max_abs_W_raw_minus_other_track_m": None if not track_gaps else float(np.max(track_gaps)),
        "assumption": (
            "P2 = -0.5 * P1 was not assumed. The equal-width gap, the half-width residual, and the truth residual are separate measurements."
        ),
    }


def attach_row_metrics(frame):
    metrics = [mt.row_metrics(et, eo, w) for et, eo, w in zip(frame["E_true_m"], frame["E_tool_m"], frame["W_m"])]
    extra = pd.DataFrame(metrics)
    extra = extra.add_prefix("re_")
    return pd.concat([frame.reset_index(drop=True), extra], axis=1)


def recompute_denominators(frame):
    lead = frame.loc[frame["quantity"].eq("lead")]
    out = {}
    for (sid, pair, track, lead_day), part in lead.groupby(["sid", "pair_id", "track", "lead_day"]):
        rows = part.filter(like="re_").rename(columns=lambda c: c[3:]).to_dict("records")
        # envelope comes from the table because inf and sup are not columns
        env = pd.to_numeric(part["envelope_includes"], errors="coerce")
        den = mt.denominators(rows)
        den["n_envelope"] = int(env.notna().sum())
        den["n_envelope_in"] = int((env == 1).sum())
        den["n_sign_agree"] = int(sum(bool(r["sign_agreement"]) for r in rows if r["sign_defined"]))
        key = f"{sid}|{pair}|{track}|{int(lead_day)}"
        out[key] = den
    return out


def compare_denominators(recomputed, official):
    keys = sorted(set(recomputed) | set(official))
    mismatches = []
    for key in keys:
        a = recomputed.get(key)
        b = official.get(key)
        if a is None or b is None:
            mismatches.append({"key": key, "missing": "recomputed" if a is None else "official"})
            continue
        for field in ("n_total", "n_truth_nonfinite", "n_truth_zero", "n_truth_nonzero", "n_small_001", "n_small_02",
                      "n_ratio_defined", "n_sign", "n_sign_agree", "n_error_over_W", "n_W_invalid", "n_envelope", "n_envelope_in"):
            if int(a[field]) != int(b[field]):
                mismatches.append({"key": key, "field": field, "recomputed": int(a[field]), "official": int(b[field])})
                break
    return mismatches


def normalised_supplement(frame):
    """W/q0, W/|E_true|, and log W. These do not replace W in metres."""
    sub = frame.loc[(frame["quantity"].eq("lead")) & (frame["pair_id"] == P1) & (frame["track"] == "raw")].copy()
    out = {}
    for lead in LEADS:
        part = sub.loc[_match_level(sub["lead_day"], lead)]
        w = part["W_m"].to_numpy(float)
        q0 = part["q0_m3d"].to_numpy(float)
        et = part["E_true_m"].to_numpy(float)
        w_over_q = np.full(w.shape, np.nan)
        q_ok = np.isfinite(w) & np.isfinite(q0) & (q0 != 0)
        w_over_q[q_ok] = w[q_ok] / q0[q_ok]
        w_over_e = np.full(w.shape, np.nan)
        e_ok = np.isfinite(w) & np.isfinite(et) & (et != 0)
        w_over_e[e_ok] = w[e_ok] / np.abs(et[e_ok])
        log_w = np.full(w.shape, np.nan)
        w_ok = np.isfinite(w) & (w > 0)
        log_w[w_ok] = np.log(w[w_ok])
        out[str(lead)] = {
            "n": int(len(part)),
            "median_W_m": _median(w),
            "median_W_over_q0_d_per_m2": _median(w_over_q),
            "n_W_over_q0": int(np.isfinite(w_over_q).sum()),
            "median_W_over_abs_Etrue": _median(w_over_e),
            "n_W_over_abs_Etrue": int(np.isfinite(w_over_e).sum()),
            "median_log_W": _median(log_w),
            "n_positive_W": int(np.isfinite(log_w).sum()),
        }
    out["note"] = "Supplementary normalised forms. They are not substituted for raw W in metres and they are not compared with marginal quantile widths."
    return out


def reference_matched(reference_dir: Path, truth_dir: Path, manifest):
    """Same row metrics on the reference point forecast. No ranking against the tool."""
    groups = {}
    n_files = 0
    n_missing = 0
    protocol_mismatch = []
    for case in manifest:
        cid = case["case_id"]
        rp = reference_dir / f"{cid}.json"
        tp = truth_dir / f"{cid}.json"
        if not rp.exists() or not tp.exists():
            n_missing += 1
            continue
        n_files += 1
        ref = json.loads(rp.read_text())
        truth = json.loads(tp.read_text())
        width = json.loads((P2 / "wb" / "W" / f"{cid}.json").read_text())
        if str(ref.get("protocol_sha256")) != PROTOCOL_SHA:
            protocol_mismatch.append(cid)
        for pair in PAIRS:
            et = truth["E_true"][pair]
            er = ref["pairs"][pair]["E_point"]
            sup = width["envelope"][pair]["sup"]
            inf = width["envelope"][pair]["inf"]
            for lead in LEADS:
                k = lead - 1
                key = f"{case['sid']}|{pair}|reference|{lead}"
                groups.setdefault(key, [])
                groups[key].append(mt.row_metrics(et[k], er[k], float(sup[k]) - float(inf[k]), inf[k], sup[k]))
    summarised = {}
    for key, rows in groups.items():
        den = mt.denominators(rows)
        ratios = [r["effect_ratio"] for r in rows if r["effect_ratio"] is not None]
        errors = [r["abs_error_m"] for r in rows if r["abs_error_m"] is not None]
        den["median_effect_ratio"] = _median(ratios)
        den["median_abs_error_m"] = _median(errors)
        summarised[key] = den
    # also a pooled pair x lead view
    pooled = {}
    for pair in PAIRS:
        for lead in LEADS:
            rows = []
            for key, vals in groups.items():
                sid, p, _kind, k = key.split("|")
                if p == pair and int(k) == lead:
                    rows.extend(vals)
            den = mt.denominators(rows)
            den["median_effect_ratio"] = _median([r["effect_ratio"] for r in rows if r["effect_ratio"] is not None])
            den["median_abs_error_m"] = _median([r["abs_error_m"] for r in rows if r["abs_error_m"] is not None])
            pooled[f"{pair}|{lead}"] = den
    return {
        "n_case_files": n_files,
        "n_missing": n_missing,
        "n_protocol_mismatch": len(protocol_mismatch),
        "by_layer": summarised,
        "pooled": pooled,
        "marginal_widths": "The reference marginal width w_TF was not read into this comparison and was not compared with W.",
        "role": "Structurally matched reference. These rows are not a ranking against the forecasting tool.",
    }


def max_day_supplement(frame):
    sub = frame.loc[(frame["quantity"].eq("max_days_1_10")) & (frame["track"].eq("raw")) & (frame["pair_id"].eq(P1))]
    day10 = frame.loc[(frame["quantity"].eq("lead")) & (frame["track"].eq("raw")) & (frame["pair_id"].eq(P1)) & _match_level(frame["lead_day"], 10)]
    return {
        "n_max_rows_p1": int(len(sub)),
        "median_max_days_1_10_m": _median(sub["W_m"]),
        "median_day10_m": _median(day10["W_m"]),
        "note": "The maximum over days 1-10 does not replace day 10 or day 30.",
    }


def value_frame(frame, value_col, *, pair, track, lead):
    part = frame.loc[
        frame["quantity"].eq("lead")
        & frame["pair_id"].eq(pair)
        & frame["track"].eq(track)
        & _match_level(frame["lead_day"], lead)
    ].copy()
    part[value_col] = pd.to_numeric(part[value_col], errors="coerce")
    return part


def build_cells(frame):
    sub = frame.loc[(frame["quantity"].eq("lead")) & (frame["pair_id"].eq(P1)) & (frame["track"].eq("raw"))].copy()
    rows = []
    for keys, part in sub.groupby(["sid", "storage_type", "T_m2_d", "rho", "N_transitions", "lead_day"]):
        sid, storage, level_t, rho, count_n, lead = keys
        w = part["W_m"].to_numpy(float)
        sr = part["SR"].to_numpy(float)
        pi = part["pi_r"].to_numpy(float)
        rows.append({
            "sid": sid,
            "storage_type": storage,
            "T_m2_d": float(level_t),
            "rho": float(rho),
            "N_transitions": int(count_n),
            "lead_day": int(lead),
            "n": int(len(part)),
            "median_W": _median(w),
            "SR": float(np.median(sr)),
            "SR_min": float(np.min(sr)),
            "SR_max": float(np.max(sr)),
            "pi_r": float(np.median(pi)),
            "rho_realized": float(np.median(part["rho_realized"].to_numpy(float))),
        })
    cells = pd.DataFrame(rows)
    cells["log_SR"] = np.where(cells["SR"] > 0, np.log(cells["SR"]), np.nan)
    cells["log_pi"] = np.where(cells["pi_r"] > 0, np.log(cells["pi_r"]), np.nan)
    cells["log_rho"] = np.where(cells["rho_realized"] > 0, np.log(cells["rho_realized"]), np.nan)
    return cells


def assert_full_official(frame):
    required = {"data_status", "quantity", "track", "pair_id", "realization", "sid", "lead_day", "case_id"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"cell_metrics is missing columns: {sorted(missing)}")
    if frame["data_status"].nunique() != 1 or frame["data_status"].iloc[0] != "official_analysis":
        raise RuntimeError("cell_metrics data_status is not official_analysis")
    lead = frame.loc[frame["quantity"].eq("lead")]
    maximum = frame.loc[frame["quantity"].eq("max_days_1_10") & frame["track"].eq("raw")]
    if len(lead) != N_LEAD:
        raise RuntimeError(f"lead rows {len(lead)} != {N_LEAD}")
    if len(maximum) != N_MAX:
        raise RuntimeError(f"max rows {len(maximum)} != {N_MAX}")
    if set(lead["realization"].astype(int)) != set(range(10)):
        raise RuntimeError("realizations are not 0-9")
    if set(lead["track"]) != set(TRACKS) or set(lead["pair_id"]) != set(PAIRS):
        raise RuntimeError("tracks or pairs are incomplete")
    if int(lead["sid"].nunique()) != 6:
        raise RuntimeError("expected 6 layers")
    keys = ["case_id", "pair_id", "track", "lead_day"]
    if lead.duplicated(keys).any():
        raise RuntimeError("duplicate lead rows")


def tool_summary(frame):
    """Sign, error/W, inclusion, and flag totals from the recomputed row metrics, by track and lead."""
    lead = frame.loc[frame["quantity"].eq("lead") & frame["pair_id"].eq(P1)]
    out = {}
    for track in TRACKS:
        for day in LEADS:
            part = lead.loc[lead["track"].eq(track) & _match_level(lead["lead_day"], day)]
            rows = part.filter(like="re_").rename(columns=lambda c: c[3:]).to_dict("records")
            den = mt.denominators(rows)
            env = pd.to_numeric(part["envelope_includes"], errors="coerce")
            den["n_envelope"] = int(env.notna().sum())
            den["n_envelope_in"] = int((env == 1).sum())
            den["median_abs_error_m"] = _median([r["abs_error_m"] for r in rows if r["abs_error_m"] is not None])
            den["median_effect_ratio"] = _median([r["effect_ratio"] for r in rows if r["effect_ratio"] is not None])
            den["median_error_over_W"] = _median([r["error_over_W"] for r in rows if r["error_over_W"] is not None])
            den["sign_agreement_rate"] = (None if den["n_sign"] == 0 else den["n_sign_agree"] / den["n_sign"])
            den["inclusion_rate"] = (None if den["n_envelope"] == 0 else den["n_envelope_in"] / den["n_envelope"])
            out[f"{track}|{day}"] = den
    out["note"] = (
        "small_001 and small_02 count finite nonzero truth at or below 0.001 m and 0.02 m. "
        "Those rows stay in every denominator. Tolerances remain TOL_IN = 1e-12 m and W floor = 1e-6 m."
    )
    return out


def r3_bundle(frame):
    """N and rho contrasts on absolute error and on the signed ratio, each track, both leads, P1."""
    out = {}
    for track in TRACKS:
        out[track] = {}
        for day in LEADS:
            part = value_frame(frame, "re_abs_error_m", pair=P1, track=track, lead=day)
            ratio_part = value_frame(frame, "re_effect_ratio", pair=P1, track=track, lead=day)
            err_raw = raw_contrasts(part, "re_abs_error_m")
            ratio_raw = raw_contrasts(ratio_part, "re_effect_ratio")
            out[track][str(day)] = {
                "absolute_error_contrasts": contrast_bundle(part, "re_abs_error_m"),
                "signed_ratio_contrasts": contrast_bundle(ratio_part, "re_effect_ratio"),
                "absolute_error_readout_N_primary": readout_from_raw(err_raw, primary="N"),
                "absolute_error_readout_rho_primary": readout_from_raw(err_raw, primary="rho"),
                "signed_ratio_readout_N_primary": readout_from_raw(ratio_raw, primary="N"),
                "signed_ratio_readout_rho_primary": readout_from_raw(ratio_raw, primary="rho"),
            }
    return out


def w_bundle(frame, pair):
    out = {}
    for day in LEADS:
        part = value_frame(frame, "W_m", pair=pair, track="raw", lead=day)
        raw = raw_contrasts(part, "W_m")
        out[str(day)] = {
            "levels": level_grid(part, "W_m"),
            "contrasts": contrast_bundle(part, "W_m"),
            "readout_N_primary": readout_from_raw(raw, primary="N"),
            "readout_rho_primary": readout_from_raw(raw, primary="rho"),
        }
    return out


def manifest_layers(manifest):
    df = pd.DataFrame(manifest)
    out = {}
    for sid, part in df.groupby("sid"):
        out[sid] = {
            "a_d": float(part["a_d"].iloc[0]),
            "b": float(part["b"].iloc[0]),
            "gain_m_per_m3d": float(part["gain_m_per_m3d"].iloc[0]),
            "SR": float(part["SR"].iloc[0]),
            "pi_r": float(part["pi_r"].iloc[0]),
            "a_nunique": int(part["a_d"].nunique()),
            "b_nunique": int(part["b"].nunique()),
        }
    return out


def build_readout(frame, manifest, *, denominators_official, provenance_paths):
    assert_full_official(frame)
    frame = attach_row_metrics(frame)
    recomputed = recompute_denominators(frame)
    mismatches = compare_denominators(recomputed, denominators_official)
    if mismatches:
        raise RuntimeError(f"row-metric denominators disagree with cell_denominators.json: {mismatches[:3]}")
    layers = manifest_layers(manifest)
    cells = build_cells(frame)
    w_p1 = w_bundle(frame, P1)
    w_p2 = w_bundle(frame, PAIRS[1])
    doc = {
        "role": "Phase 2 section 9 numerical readout. The parent report's first paragraph is written later from this file.",
        "provenance": provenance_paths,
        "relative_reference": (
            "For an N contrast the reference width is W at N=2 in the same block and at the same rho. "
            "For a rho contrast the reference width is W at rho=0.25 in the same block and at the same N. "
            "Relative change is supplementary. A nonfinite reference or a reference at or below the 1e-6 m floor is excluded from the relative median and counted in n_relative."
        ),
        "no_threshold": "No significance test, threshold, or positive-completion gate is applied. A negative or zero median is a result.",
        "marginal_widths": "Marginal quantile widths were not compared with W.",
        "volume": volume_report(manifest),
        "p2_identity": p2_identity(frame),
        "tool_denominators_p1": tool_summary(frame),
        "width_p1_raw": w_p1,
        "width_p2_raw": w_p2,
        "r1_day10": w_p1["10"]["readout_N_primary"],
        "r2_day30": w_p1["30"]["readout_rho_primary"],
        "r3": r3_bundle(frame),
        "collapse": {
            "day10": collapse_one(cells, 10),
            "day30": collapse_one(cells, 30),
            "coupling": layer_coupling(manifest),
            "sr_only_confinedT50_leakyT500": sr_only_pair(cells, layers),
        },
        "normalised_supplement": normalised_supplement(frame),
        "max_days_1_10_supplement": max_day_supplement(frame),
        "cell_median_count": int(len(cells)),
    }
    return doc


def _fmt(value):
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _sentence_readout(title, block):
    pooled = block["pooled"]
    lines = [
        f"{title}: the median paired difference is {_fmt(pooled['median_primary'])} "
        f"({pooled['direction_of_median']}).",
        f"The fraction of finite contrasts below zero is {_fmt(pooled['fraction_negative'])} "
        f"({pooled['n_negative_primary']} of {pooled['n_finite_primary']}).",
        f"The ratio of median absolute contrasts is {_fmt(pooled['ratio_of_median_absolute_contrasts'])}. "
        f"The absolute value of the median difference is {_fmt(pooled['abs_of_median_primary'])}, stored separately.",
    ]
    layer_bits = []
    for sid, view in block["by_layer"].items():
        layer_bits.append(
            f"{sid} median {_fmt(view['median_primary'])} ({view['direction_of_median']}), "
            f"negative fraction {_fmt(view['fraction_negative'])}, "
            f"median-absolute ratio {_fmt(view['ratio_of_median_absolute_contrasts'])}"
        )
    lines.append("Per layer: " + "; ".join(layer_bits) + ".")
    storage_bits = []
    for storage, view in block["by_storage"].items():
        storage_bits.append(
            f"{storage} median {_fmt(view['median_primary'])} ({view['direction_of_median']}), "
            f"negative fraction {_fmt(view['fraction_negative'])}, "
            f"median-absolute ratio {_fmt(view['ratio_of_median_absolute_contrasts'])}"
        )
    lines.append("Per storage label, pooling the two transmissivities: " + "; ".join(storage_bits) + ".")
    if block["layer_directions_differ"]:
        lines.append("The six layer directions of this median are not the same.")
    else:
        lines.append("The six layer directions of this median are the same.")
    return " ".join(lines)


def render_markdown(doc):
    r1 = _sentence_readout(
        "R1 at day 10 uses the N contrast (W at N=18 minus W at N=2), pooled over blocks and the three rest ratios, "
        "and compares its median absolute size with the rho contrast",
        doc["r1_day10"],
    )
    r2 = _sentence_readout(
        "R2 at day 30 uses the rho contrast (W at rho=4 minus W at rho=0.25), pooled over blocks and the three transition counts, "
        "and compares its median absolute size with the N contrast",
        doc["r2_day30"],
    )
    vol = doc["volume"]["cross_rho_volume_difference_m3"]["V_rho0.25_minus_rho4"]
    p2 = doc["p2_identity"]
    c10 = doc["collapse"]["day10"]
    c30 = doc["collapse"]["day30"]
    lines = [
        "# Phase 2 scientific readout",
        "",
        "This file records the section 9 numbers. It is not the parent final report.",
        "",
        "## R1 and R2",
        "",
        r1,
        "",
        r2,
        "",
        "## R3",
        "",
        "R3 applies the same N and rho contrasts to the tool absolute error and to the signed tool/truth ratio.",
        "The raw track is the main record. exp10, exp30, and exp90 are separate supplements. Both leads are reported.",
        "The numerical contrasts, denominators, and per-layer views are in the JSON block below under `r3` and `tool_denominators_p1`.",
        "",
        "## Volume and pair identity",
        "",
        f"Across rho, the pumped-volume difference V(rho=0.25) minus V(rho=4) ranges from {_fmt(vol['min'])} m3 to {_fmt(vol['max'])} m3 "
        f"over {vol['n']} layer-realization-N groups.",
        f"The largest absolute volume residual across N=2, 6, and 18 inside a matched layer, realization, and rho is "
        f"{_fmt(doc['volume']['max_abs_V_difference_across_N_m3'])} m3.",
        f"The observed maximum of |W(P1) - W(P2)| is {_fmt(p2['max_abs_W_p1_minus_W_p2_m'])} m.",
        f"The observed maximum of |W(P2) - 0.5 W(P1)| is {_fmt(p2['max_abs_W_p2_minus_half_W_p1_m'])} m, and the median of W(P2)/W(P1) is {_fmt(p2['median_W_p2_over_W_p1'])}.",
        f"The observed maximum of |E_true(P2) + 0.5 E_true(P1)| is {_fmt(p2['max_abs_Etrue_p2_plus_half_Etrue_p1_m'])} m.",
        "Those residuals were measured. The analysis did not assume that P2 equals minus one half of P1.",
        "",
        "## Collapse",
        "",
        f"Day 10 uses {c10['n_used']} of {c10['n_cells']} cell medians "
        f"(nonpositive {c10['n_nonpositive_median_W']}, nonfinite {c10['n_nonfinite_median_W']}). "
        f"Explained variance without layer indicators is {_fmt(c10['model_logSR_logpi_logrho']['r2'])} "
        f"(rank {c10['model_logSR_logpi_logrho']['rank']}). "
        f"With layer indicators it is {_fmt(c10['model_plus_layer_indicators']['r2'])} "
        f"(rank {c10['model_plus_layer_indicators']['rank']}).",
        f"Day 30 uses {c30['n_used']} of {c30['n_cells']} cell medians "
        f"(nonpositive {c30['n_nonpositive_median_W']}, nonfinite {c30['n_nonfinite_median_W']}). "
        f"Explained variance without layer indicators is {_fmt(c30['model_logSR_logpi_logrho']['r2'])} "
        f"(rank {c30['model_logSR_logpi_logrho']['rank']}). "
        f"With layer indicators it is {_fmt(c30['model_plus_layer_indicators']['r2'])} "
        f"(rank {c30['model_plus_layer_indicators']['rank']}).",
        doc["collapse"]["coupling"]["statement"],
        f"Pearson correlation of log SR and log pi_r across the six layers is {_fmt(doc['collapse']['coupling']['log_SR_log_pi_pearson'])}.",
        doc["collapse"]["sr_only_confinedT50_leakyT500"]["statement"],
        "",
        "## Limits carried into the readout",
        "",
        "There are six layers. SR and pi_r are anti-correlated across layers. pi_r is constant within a layer.",
        "Within one layer, one rest ratio, and one realization, N=2, 6, and 18 match in context volume and in the mean pumping rate outside the designated rest. Only a longer rest reduces the pumped volume. N moves with the recency of the last switch: the ON tail is 16 days longer at N=2 than at N=18, which is not a matched-volume difference.",
        "Storage and resistance are coupled. The confined T=50 and leaky T=500 pair is an SR comparison, not a pure-storage claim.",
        "Marginal quantile widths were not compared with W. The matched reference is not a ranking.",
        "Intermediate N=6 and rho=1 remain in the level grids and in the per-condition contrast tables.",
        "",
        "## Requested items and self-imposed checks",
        "",
        "Requested: the section 9.3 contrasts, the section 9.4 collapse, the volume range, the pair-identity measurement, the denominators, and this readout.",
        "Self-imposed, and not a completion gate: the synthetic check that median absolute contrasts are not replaced by the absolute median, and the exact recount of cell_denominators.json.",
        "",
        "## Full numerical record",
        "",
        "```json",
        json.dumps(doc, indent=1, allow_nan=False, default=_json_default),
        "```",
        "",
    ]
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))


def write_readout(doc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=1, allow_nan=False, default=_json_default)
    (out_dir / "SCIENTIFIC_READOUT.json").write_text(payload + "\n")
    (out_dir / "SCIENTIFIC_READOUT.md").write_text(render_markdown(doc))
    return {
        "json_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "md_sha256": sha256_file(out_dir / "SCIENTIFIC_READOUT.md"),
    }


def load_inputs(metrics_path: Path, manifest_path: Path):
    frame = pd.read_csv(metrics_path)
    manifest = json.loads(manifest_path.read_text())
    meta = pd.DataFrame(manifest)[
        ["case_id", "sid", "rho_realized", "V_m3", "q0_m3d", "a_d", "b", "gain_m_per_m3d", "kappa", "t95_d", "storage_value"]
    ]
    frame = frame.merge(meta, on="case_id", how="left")
    if frame["sid"].isna().any():
        raise RuntimeError("derived manifest does not cover every case_id in cell_metrics")
    return frame, manifest


def run_official(metrics_path: Path, manifest_path: Path, denominators_path: Path, out_dir: Path, reference_dir: Path, truth_dir: Path):
    frame, manifest = load_inputs(metrics_path, manifest_path)
    official = json.loads(denominators_path.read_text())
    if official.get("provenance") != "official_analysis":
        raise RuntimeError("cell_denominators provenance is not official_analysis")
    doc = build_readout(
        frame,
        manifest,
        denominators_official=official["denominators"],
        provenance_paths={
            "protocol_sha256": PROTOCOL_SHA,
            "analysis_code": str(Path(__file__).resolve()),
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "metrics_csv": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "denominators": str(denominators_path),
            "denominators_sha256": sha256_file(denominators_path),
            "derived_manifest": str(manifest_path),
            "derived_manifest_sha256": sha256_file(manifest_path),
            "tool_archives": official.get("tool_archives"),
            "wb_manifest_sha256": official.get("wb_manifest_sha256"),
            "frozen_metrics_sha256": sha256_file(ROOT / "lib/p3_phase2_metrics.py"),
            "command": (
                "env/.venv_pilot/bin/python lib/p3_phase2_analysis.py "
                f"--metrics {metrics_path} --manifest {manifest_path} --denominators {denominators_path} --out {out_dir}"
            ),
        },
    )
    doc["matched_reference"] = reference_matched(reference_dir, truth_dir, manifest)
    hashes = write_readout(doc, out_dir)
    return hashes


def _self_check():
    """Synthetic arithmetic only. Nothing here is a Phase 2 result."""
    # Two blocks. Deltas +1 and -3. median = -1, median of absolute values = 2.
    rows = []
    plan = {
        0: {0.25: {2: 4.0, 6: 4.0, 18: 5.0}, 1.0: {2: 4.0, 6: 4.0, 18: 4.0}, 4.0: {2: 4.0, 6: 4.0, 18: 1.0}},
        1: {0.25: {2: 10.0, 6: 10.0, 18: 7.0}, 1.0: {2: 10.0, 6: 10.0, 18: 10.0}, 4.0: {2: 10.0, 6: 10.0, 18: 10.0}},
    }
    for realization, by_rho in plan.items():
        for rho, by_n in by_rho.items():
            for count_n, width in by_n.items():
                rows.append({
                    "sid": "confined_T50", "storage_type": "confined", "realization": realization,
                    "rho": rho, "N_transitions": count_n, "W_m": width,
                })
    frame = pd.DataFrame(rows)
    raw = raw_contrasts(frame, "W_m")
    view = readout_from_raw(raw, primary="N")["pooled"]
    # Deltas are +1, 0, -3, -3, 0, 0. The median is 0 and the median of the absolute values is 0.5.
    if view["median_primary"] != 0.0 or view["abs_of_median_primary"] != 0.0:
        raise AssertionError(view)
    if view["median_abs_primary"] != 0.5:
        raise AssertionError(view)
    if view["median_abs_primary"] == view["abs_of_median_primary"]:
        raise AssertionError("absolute median replaced the median of absolute contrasts")
    if view["ratio_of_median_absolute_contrasts"] is not None:
        raise AssertionError(view)
    defined = []
    for rho in RHOS:
        for count_n, width in ((2, 10.0), (6, 10.0), (18, 12.0)):
            defined.append({"sid": "confined_T50", "storage_type": "confined", "realization": 0, "rho": rho, "N_transitions": count_n, "W_m": width if rho == 0.25 else width + 4.0})
    defined_view = readout_from_raw(raw_contrasts(pd.DataFrame(defined), "W_m"), primary="N")["pooled"]
    # |ΔN| is 2 at every rho. |Δrho| is 4 at every N. The ratio is 0.5.
    if defined_view["median_abs_primary"] != 2.0 or defined_view["median_abs_compared"] != 4.0:
        raise AssertionError(defined_view)
    if abs(defined_view["ratio_of_median_absolute_contrasts"] - 0.5) > 1e-12:
        raise AssertionError(defined_view)
    # Undefined ratio stays out of the finite median and the block is still counted in the condition table.
    ratio_rows = []
    for realization, ratio_n2, ratio_n18 in ((0, 0.5, 0.2), (1, None, 0.4)):
        for count_n, ratio in ((2, ratio_n2), (6, ratio_n2), (18, ratio_n18)):
            for rho in RHOS:
                ratio_rows.append({
                    "sid": "confined_T50", "storage_type": "confined", "realization": realization,
                    "rho": rho, "N_transitions": count_n, "re_effect_ratio": ratio,
                })
    ratio_frame = pd.DataFrame(ratio_rows)
    table = contrast_bundle(ratio_frame, "re_effect_ratio")["pooled"]["N18_minus_N2_by_rho"]
    finite_counts = [rec["n_finite"] for rec in table]
    if finite_counts != [1, 1, 1]:
        raise AssertionError(finite_counts)
    # Manifest volume function on a toy schedule, then on the real manifest if it is present.
    toy = []
    for sid in ("confined_T50",):
        for realization in (0,):
            for rho, volume in ((0.25, 100.0), (1.0, 90.0), (4.0, 70.0)):
                for count_n in NS:
                    toy.append({
                        "sid": sid, "realization": realization, "rho_nominal": rho, "N_nominal": count_n,
                        "V_m3": volume, "kappa": 1.0 + 0.01 * count_n,
                        "storage_type": "confined", "storage_value": 1e-4, "T_m2_d": 50.0,
                        "SR": 1.0, "pi_r": 0.02, "a_d": 20.0, "b": 0.001, "gain_m_per_m3d": 0.01, "t95_d": 10.0,
                    })
    toy_vol = volume_report(toy)
    if toy_vol["max_abs_V_difference_across_N_m3"] != 0.0:
        raise AssertionError(toy_vol)
    span = toy_vol["cross_rho_volume_difference_m3"]["V_rho0.25_minus_rho4"]
    if span["min"] != 30.0 or span["max"] != 30.0:
        raise AssertionError(span)
    manifest_path = P2 / "cases/derived_manifest.json"
    if manifest_path.exists():
        real = volume_report(json.loads(manifest_path.read_text()))
        if real["n_cases"] != 540:
            raise AssertionError(real["n_cases"])
        if real["max_abs_V_difference_across_N_m3"] > 1e-9:
            raise AssertionError(real["max_abs_V_difference_across_N_m3"])
        if real["cross_rho_volume_difference_m3"]["V_rho0.25_minus_rho4"]["min"] <= 0:
            raise AssertionError("expected a longer rest to reduce volume")
    # Refusal of a partial official table.
    try:
        assert_full_official(pd.DataFrame({"data_status": ["official_analysis"], "quantity": ["lead"]}))
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial table was accepted")
    print(json.dumps({"self_check": "ok", "median_abs": view["median_abs_primary"], "abs_of_median": view["abs_of_median_primary"]}))


def main():
    parser = argparse.ArgumentParser(description="Aggregate the official Phase 2 cell table under protocol section 9.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--metrics", type=Path, default=P2 / "analysis/cell_metrics.csv")
    parser.add_argument("--manifest", type=Path, default=P2 / "cases/derived_manifest.json")
    parser.add_argument("--denominators", type=Path, default=P2 / "analysis/cell_denominators.json")
    parser.add_argument("--reference", type=Path, default=P2 / "wb/reference")
    parser.add_argument("--truth", type=Path, default=P2 / "wb/truth_eval")
    parser.add_argument("--out", type=Path, default=P2 / "analysis")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    hashes = run_official(args.metrics, args.manifest, args.denominators, args.out, args.reference, args.truth)
    print(json.dumps(hashes))


if __name__ == "__main__":
    main()
