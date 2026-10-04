"""D04 analysis and figures from actual D03 and D04 files.

Refuses to write a readout until the phase3 envelope, both tool horizons,
and the installed-source normalization table are all present. D03 outputs
are read only. No new invariance cutoff is introduced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from scipy import stats

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_phase2_metrics as mt  # noqa: E402

P2 = ROOT / "results/phase2"
P3 = ROOT / "results/phase3"
TRACKS = ("raw", "exp10", "exp30", "exp90")
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
D03_PROTOCOL = "0b360edd4ecc9bc601376eeaba1f78d5511096c67cb2bd42810ffe010713b794"
D03_CELL = "410e6c68bc0fc77aeab19de264a2fc75628a06e70a57a74e31564169372d3b65"
FIELD_CSV = "4967465a29dbb5905baa57d2bc8b904fa22950f647d85025d0748959ea4f9e0f"
PHYSICAL = "61f7fa696cfbca26e8f46ace79ca5432e0c644a667ef9eccc7fc7cc27bd12bad"
FONT_REG = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
EXTRA = [
    "dataset", "experiment_group", "Q_scale", "recency_label", "sid", "storage_value", "t95_d",
    "env_inf_m", "env_sup_m", "delta_m", "sign_determined", "envelope_sign", "includes_zero",
    "record_contradiction", "ambiguous_truth_sign_error", "unconfined", "linear_approximation",
    "k_star", "flags", "d03_parent_case_id", "burst_switch_count", "short_off_intervals",
    "short_on_maximal_intervals", "tool_pair_id", "normalized_future_change", "normalized_future_size",
    "pumping_denominator", "pumping_cumulative_std", "pumping_detrended", "pumping_safe_std_replacement",
    "head_cumulative_std", "head_detrended",
]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(value):
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def sign_state(inf, sup) -> dict:
    """Closed interval. Determined only when 0 is outside, including the endpoints."""
    lo, hi = _num(inf), _num(sup)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        return dict(sign_determined="", envelope_sign="", includes_zero="")
    if lo > 0:
        return dict(sign_determined=1, envelope_sign=1, includes_zero=0)
    if hi < 0:
        return dict(sign_determined=1, envelope_sign=-1, includes_zero=0)
    return dict(sign_determined=0, envelope_sign=0, includes_zero=1)


def contradiction_flags(state, e_tool, e_true, storage_type) -> dict:
    eo, et = _num(e_tool), _num(e_true)
    determined = state["sign_determined"]
    out = dict(record_contradiction="", ambiguous_truth_sign_error="", unconfined=int(storage_type == "unconfined"))
    if determined == "" or not math.isfinite(eo):
        return out
    tool_sign = 0 if eo == 0 else (1 if eo > 0 else -1)
    truth_nonzero = math.isfinite(et) and et != 0
    truth_sign = 0 if not truth_nonzero else (1 if et > 0 else -1)
    if determined == 1:
        out["record_contradiction"] = int(tool_sign != int(state["envelope_sign"]))
        out["ambiguous_truth_sign_error"] = 0
    else:
        out["record_contradiction"] = 0
        out["ambiguous_truth_sign_error"] = int(truth_nonzero and tool_sign != truth_sign)
    return out


def load_tools(directory: Path, digest: str, n_cases: int, input_paths: dict) -> dict:
    out = {}
    n_pairs = n_cases * 2
    for track in TRACKS:
        for horizon in (10, 30):
            path = directory / f"timesfm_{track}_H{horizon}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            if str(data["protocol_sha256"]) != digest:
                raise ValueError(f"{path.name} protocol hash differs")
            if str(data["input_sha256"]) != sha(input_paths[horizon]):
                raise ValueError(f"{path.name} was not built from the official tool input")
            if "track" in data and str(data["track"]) != track:
                raise ValueError(f"{path.name} track label differs")
            point = np.asarray(data["E_point"], float)
            if point.shape != (n_pairs, horizon):
                raise ValueError(f"{path.name} E_point shape {point.shape}")
            mapping = {}
            for pid, row in zip(data["pair_id"].astype(str), point):
                if pid in mapping:
                    raise ValueError(f"duplicate pair id in {path.name}")
                mapping[pid] = row
            if len(mapping) != n_pairs:
                raise ValueError(f"{path.name} unique pairs {len(mapping)}")
            out[(track, horizon)] = mapping
            out[(track, horizon, "sha")] = sha(path)
    return out


def case_rows(record, dataset, wb: Path, tools, provenance: str) -> list[dict]:
    cid = record["case_id"]
    width = json.loads((wb / "W" / f"{cid}.json").read_text())
    truth = json.loads((wb / "truth_eval" / f"{cid}.json").read_text())
    rows = []
    flags = ",".join(width.get("flags") or [])
    for pair in PAIRS:
        pid = f"{cid}__{pair}"
        sup = np.asarray(width["envelope"][pair]["sup"], float)
        inf = np.asarray(width["envelope"][pair]["inf"], float)
        true = np.asarray(truth["E_true"][pair], float)
        delta = np.asarray(truth["delta"][pair], float)
        if sup.shape != (30,) or inf.shape != (30,) or true.shape != (30,) or delta.shape != (30,):
            raise ValueError(f"{cid} {pair} lead axis is not 30")
        curve = sup - inf
        for track in TRACKS:
            for lead in (10, 30):
                tool = float(tools[(track, lead)][pid][lead - 1])
                base = mt.cell_metrics_row(
                    record, pair, track, "lead", lead, float(curve[lead - 1]), width.get("W_status"),
                    float(true[lead - 1]), tool, float(inf[lead - 1]), float(sup[lead - 1]), provenance,
                )
                rows.append(_extend(base, record, dataset, inf[lead - 1], sup[lead - 1], delta[lead - 1], tool, true[lead - 1], flags, lead - 1, pid))
        k_star = int(np.argmax(curve[:10]))
        tool = float(tools[("raw", 10)][pid][k_star])
        base = mt.cell_metrics_row(
            record, pair, "raw", "max_days_1_10", None, float(curve[:10].max()), width.get("W_status"),
            float(true[k_star]), tool, float(inf[k_star]), float(sup[k_star]), provenance,
        )
        rows.append(_extend(base, record, dataset, inf[k_star], sup[k_star], delta[k_star], tool, true[k_star], flags, k_star, pid))
    return rows


def _extend(base, record, dataset, inf, sup, delta, tool, true, flags, k_star, pid) -> dict:
    state = sign_state(inf, sup)
    flags_row = contradiction_flags(state, tool, true, record.get("storage_type", ""))
    base.update(
        dataset=dataset,
        experiment_group=record.get("experiment_group") or dataset,
        Q_scale=record.get("Q_scale", 1.0),
        recency_label=record.get("recency_label") or "d03_schedule",
        sid=record["sid"],
        storage_value=record.get("storage_value", record.get("S")),
        t95_d=record["t95_d"],
        env_inf_m=float(inf),
        env_sup_m=float(sup),
        delta_m=float(delta),
        k_star=int(k_star),
        flags=flags,
        d03_parent_case_id=record.get("d03_parent_case_id", ""),
        burst_switch_count=record.get("burst_switch_count", ""),
        short_off_intervals=record.get("short_off_intervals", ""),
        short_on_maximal_intervals=record.get("short_on_maximal_intervals", ""),
        tool_pair_id=pid,
        linear_approximation=int(bool(record.get("linear_approximation", False))),
        normalized_future_change="",
        normalized_future_size="",
        pumping_denominator="",
        pumping_cumulative_std="",
        pumping_detrended="",
        pumping_safe_std_replacement="",
        head_cumulative_std="",
        head_detrended="",
    )
    base.update(state)
    base.update(flags_row)
    return base


def attach_normalization(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    use = [
        "case_id", "pair_id", "track", "horizon", "normalized_future_change", "normalized_future_size",
        "pumping_denominator", "pumping_cumulative_std", "pumping_detrended", "pumping_safe_std_replacement",
        "head_cumulative_std", "head_detrended",
    ]
    norm = pd.read_csv(path, usecols=use)
    if len(norm) != 19680 or norm.duplicated(["case_id", "pair_id", "track", "horizon"]).any():
        raise RuntimeError("normalization table is not the unique 1230 x 2 x 4 x 2 diagnostic")
    lead = frame["quantity"].eq("lead")
    keep = [c for c in frame.columns if c not in use[4:]]
    merged = frame.loc[lead, keep].merge(
        norm, left_on=["case_id", "tool_pair_id", "track", "lead_day"],
        right_on=["case_id", "pair_id", "track", "horizon"], how="left", suffixes=("", "_norm"),
    )
    if merged["normalized_future_size"].isna().any():
        raise RuntimeError("normalization rows do not match every lead row")
    merged = merged.drop(columns=["pair_id_norm", "horizon"], errors="ignore")
    # merge renamed the right pair_id; the left pair_id is the schedule-pair name.
    if "pair_id_norm" not in merged.columns and "pair_id" in norm.columns:
        pass
    other = frame.loc[~lead].copy()
    for column in use[4:]:
        if column not in other.columns:
            other[column] = ""
    common = [c for c in frame.columns if c in merged.columns or c in use[4:]]
    # Rebuild with the original pair_id preserved. The merge keeps left pair_id unless collision.
    if "pair_id_x" in merged.columns:
        merged = merged.rename(columns={"pair_id_x": "pair_id"}).drop(columns=["pair_id_y"], errors="ignore")
    out = pd.concat([merged, other], ignore_index=True, sort=False)
    return out


def summarise_diff(values) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out = dict(n=n, median=None, q25=None, q75=None, n_negative=0, n_positive=0, n_zero=0, wilcoxon_p=None)
    if n == 0:
        return out
    out.update(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
               n_negative=int((x < 0).sum()), n_positive=int((x > 0).sum()), n_zero=int((x == 0).sum()))
    if n >= 10 and np.any(x != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)
    out["wilcoxon_note"] = "self-imposed descriptive rank test, not a requested completion gate"
    return out


def _matched(frame, left, right, by, column):
    L = frame.loc[left].set_index(by)[column]
    R = frame.loc[right].set_index(by)[column]
    if L.index.has_duplicates or R.index.has_duplicates:
        raise RuntimeError(f"duplicate match key for {column}")
    both = L.index.intersection(R.index)
    return (R.loc[both] - L.loc[both]).to_numpy(float), int(len(both))


def design_a(frame, manifest_a) -> dict:
    convention = {
        "rule": "N is the burst switch count. The maximal-run census is N/2 one-day OFF runs and N/2-1 interior short ON runs because the first prescribed ON joins the preceding long ON.",
        "n_cases": int(len(manifest_a)),
        "switch_equals_N": bool((pd.Series([r["burst_switch_count"] for r in manifest_a]) == pd.Series([r["N_nominal"] for r in manifest_a])).all()),
        "short_off_equals_N_over_2": bool(all(r["short_off_intervals"] == r["N_nominal"] // 2 for r in manifest_a)),
        "short_on_maximal_equals_N_over_2_minus_1": bool(all(r["short_on_maximal_intervals"] == r["N_nominal"] // 2 - 1 for r in manifest_a)),
        "example_N2": {k: next(r[k] for r in manifest_a if r["N_nominal"] == 2) for k in ("burst_switch_count", "short_off_intervals", "short_on_maximal_intervals", "census")},
    }
    volumes = []
    for (_, _), part in pd.DataFrame(manifest_a).groupby(["sid", "realization"]):
        volumes.append(float(part["V_m3"].max() - part["V_m3"].min()))
        volumes.append(float(part["mean_outside_rest_m3d"].max() - part["mean_outside_rest_m3d"].min()))
    sub = frame.loc[frame["experiment_group"].eq("A")].copy()
    contrasts = {}
    for quantity, lead in (("lead", 10), ("max_days_1_10", None)):
        for pair in PAIRS:
            part = sub.loc[sub["quantity"].eq(quantity) & sub["pair_id"].eq(pair) & sub["track"].eq("raw")]
            if lead is not None:
                part = part.loc[pd.to_numeric(part["lead_day"]) == lead]
            by_count = ["sid", "realization", "recency_label", "pair_id"]
            for recency in ("recent", "old"):
                diff, n = _matched(part, part["N_transitions"].astype(int).eq(2) & part["recency_label"].eq(recency),
                                   part["N_transitions"].astype(int).eq(18) & part["recency_label"].eq(recency), by_count, "W_m")
                contrasts[f"W|{quantity}|{pair}|count|{recency}"] = summarise_diff(diff) | {"n_keys": n}
            by_rec = ["sid", "realization", "N_transitions", "pair_id"]
            for count in (2, 18):
                diff, n = _matched(part, part["N_transitions"].astype(int).eq(count) & part["recency_label"].eq("recent"),
                                   part["N_transitions"].astype(int).eq(count) & part["recency_label"].eq("old"), by_rec, "W_m")
                contrasts[f"W|{quantity}|{pair}|recency|N{count}"] = summarise_diff(diff) | {"n_keys": n}
    ratio = frame.loc[frame["experiment_group"].eq("A") & frame["quantity"].eq("lead")].copy()
    ratio["effect_ratio"] = pd.to_numeric(ratio["effect_ratio"], errors="coerce")
    for lead in (10, 30):
        for pair in PAIRS:
            for track in TRACKS:
                part = ratio.loc[ratio["lead_day"].eq(lead) & ratio["pair_id"].eq(pair) & ratio["track"].eq(track)]
                by_count = ["sid", "realization", "recency_label", "pair_id"]
                for recency in ("recent", "old"):
                    diff, n = _matched(part, part["N_transitions"].astype(int).eq(2) & part["recency_label"].eq(recency),
                                       part["N_transitions"].astype(int).eq(18) & part["recency_label"].eq(recency), by_count, "effect_ratio")
                    contrasts[f"ratio|{lead}|{pair}|{track}|count|{recency}"] = summarise_diff(diff) | {"n_keys": n}
                by_rec = ["sid", "realization", "N_transitions", "pair_id"]
                for count in (2, 18):
                    diff, n = _matched(part, part["N_transitions"].astype(int).eq(count) & part["recency_label"].eq("recent"),
                                       part["N_transitions"].astype(int).eq(count) & part["recency_label"].eq("old"), by_rec, "effect_ratio")
                    contrasts[f"ratio|{lead}|{pair}|{track}|recency|N{count}"] = summarise_diff(diff) | {"n_keys": n}
    return {"discrete_convention": convention, "max_abs_volume_or_mean_spread": float(np.max(volumes)), "paired": contrasts}


def design_b(frame) -> dict:
    d03 = frame.loc[frame["dataset"].eq("D03") & frame["N_transitions"].astype(int).eq(6)].copy()
    new = frame.loc[frame["experiment_group"].eq("B")].copy()
    out = {"n_scale1": int(d03["case_id"].nunique()), "n_scale_0p3": int(new.loc[new["Q_scale"].astype(float).eq(0.3), "case_id"].nunique()),
           "n_scale_3": int(new.loc[new["Q_scale"].astype(float).eq(3.0), "case_id"].nunique())}
    parents = set(new["d03_parent_case_id"]) - {""}
    if parents != set(d03["case_id"]):
        raise RuntimeError("scale-1 parents do not match the D03 N=6 case ids")
    rows = {}
    for lead in (10, 30):
        for pair in PAIRS:
            base = d03.loc[d03["quantity"].eq("lead") & d03["track"].eq("raw") & d03["pair_id"].eq(pair) & d03["lead_day"].eq(lead)]
            base = base.set_index("case_id")
            for scale in (0.3, 3.0):
                part = new.loc[new["quantity"].eq("lead") & new["track"].eq("raw") & new["pair_id"].eq(pair) & new["lead_day"].eq(lead) & new["Q_scale"].astype(float).eq(scale)]
                if len(part) != 180:
                    raise RuntimeError(f"scale {scale} rows {len(part)}")
                parent = base.loc[part["d03_parent_case_id"].to_numpy()]
                w_diff = part["W_m"].to_numpy(float) - parent["W_m"].to_numpy(float)
                delta = parent["delta_m"].to_numpy(float)
                e_ratio = np.abs(part["E_true_m"].to_numpy(float)) / np.abs(parent["E_true_m"].to_numpy(float)) / scale
                w_ok = (part["W_m"].to_numpy(float) > 1e-6) & (parent["W_m"].to_numpy(float) > 1e-6) & (part["E_true_m"].to_numpy(float) != 0) & (parent["E_true_m"].to_numpy(float) != 0)
                rel = np.full(len(part), np.nan)
                rel[w_ok] = (np.abs(part["E_true_m"].to_numpy(float)[w_ok]) / part["W_m"].to_numpy(float)[w_ok]) / (np.abs(parent["E_true_m"].to_numpy(float)[w_ok]) / parent["W_m"].to_numpy(float)[w_ok]) / scale
                key = f"{lead}|{pair}|{scale}"
                rows[key] = {
                    "W_difference_m": summarise_diff(w_diff),
                    "n_abs_W_difference_within_frozen_delta": int(np.sum(np.abs(w_diff) <= delta)),
                    "n_pairs": int(len(part)),
                    "frozen_delta_definition": "truth delta stored with the case, max(0.25|E_true|, 0.02 m)",
                    "n_abs_W_difference_within_inclusion_tolerance_1e-12": int(np.sum(np.abs(w_diff) <= 1e-12)),
                    "abs_E_ratio_over_scale": summarise_diff(e_ratio - 1.0) | {"median_ratio_over_scale": float(np.nanmedian(e_ratio))},
                    "abs_E_over_W_ratio_over_scale": summarise_diff(rel - 1.0) | {"median_ratio_over_scale": float(np.nanmedian(rel)), "n_defined": int(np.isfinite(rel).sum())},
                }
    ratio = pd.concat([
        d03.loc[d03["quantity"].eq("lead")],
        new.loc[new["quantity"].eq("lead")],
    ], ignore_index=True)
    ratio["effect_ratio"] = pd.to_numeric(ratio["effect_ratio"], errors="coerce")
    dependence = {}
    for lead in (10, 30):
        for pair in PAIRS:
            for track in TRACKS:
                part = ratio.loc[ratio["lead_day"].eq(lead) & ratio["pair_id"].eq(pair) & ratio["track"].eq(track)]
                dependence[f"{lead}|{pair}|{track}"] = spearman(part["SR"], part["effect_ratio"])
    out["paired"] = rows
    out["tool_ratio_vs_SR"] = dependence
    out["note"] = "Scale 1 is the existing D03 N=6 record. W is compared in metres. No new percent band is used as an invariance cutoff."
    return out


def ols_r2(y, x) -> dict:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y, x = y[mask], x[mask]
    n = int(len(y))
    if n < x.shape[1] + 2:
        return dict(n=n, r2=None)
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    tot = float(np.sum((y - y.mean()) ** 2))
    res = float(np.sum((y - pred) ** 2))
    return dict(n=n, r2=(None if tot == 0 else 1 - res / tot), coefficients=["intercept", "log_SR", "log_rho", "log_pi_r"])


def design_c(frame) -> dict:
    sub = frame.loc[frame["experiment_group"].eq("C") & frame["quantity"].eq("lead") & frame["track"].eq("raw")].copy()
    if sub["case_id"].nunique() != 90:
        raise RuntimeError("design C is not 90 cases")
    strata = []
    for (storage, lead, pair), part in sub.groupby(["storage_value", "lead_day", "pair_id"]):
        w = part["W_m"].to_numpy(float)
        et = part["E_true_m"].to_numpy(float)
        rel = np.full(len(part), np.nan)
        ok = np.isfinite(w) & np.isfinite(et) & (et != 0) & (w > 0)
        rel[ok] = w[ok] / np.abs(et[ok])
        strata.append(dict(storage_value=float(storage), lead=int(lead), pair=pair, n=int(len(part)),
                           median_W_m=float(np.median(w)), median_SR=float(np.median(part["SR"])), median_pi_r=float(np.median(part["pi_r"])),
                           median_W_over_abs_E=float(np.nanmedian(rel)), n_W_over_abs_E=int(np.isfinite(rel).sum()),
                           storage_type=sorted(set(part["storage_type"]))[0], linear_approximation=int(part["linear_approximation"].astype(int).max())))
    collapse = {}
    for lead in (10, 30):
        for pair in PAIRS:
            part = sub.loc[sub["lead_day"].eq(lead) & sub["pair_id"].eq(pair)]
            y = np.log(part["W_m"].to_numpy(float) / np.abs(part["E_true_m"].to_numpy(float)))
            x = np.column_stack([np.log(part["SR"].to_numpy(float)), np.log(part["rho"].to_numpy(float)), np.log(part["pi_r"].to_numpy(float))])
            bad = ~np.isfinite(y)
            y = np.where(np.isfinite(part["W_m"].to_numpy(float)) & (part["W_m"].to_numpy(float) > 0) & (part["E_true_m"].to_numpy(float) != 0), y, np.nan)
            collapse[f"{lead}|{pair}"] = ols_r2(y, x) | {
                "spearman_W_over_absE_vs_SR": spearman(part["SR"], part["W_m"] / part["E_true_m"].abs()),
                "spearman_W_over_absE_vs_rho": spearman(part["rho"], part["W_m"] / part["E_true_m"].abs()),
                "spearman_W_over_absE_vs_pi": spearman(part["pi_r"], part["W_m"] / part["E_true_m"].abs()),
                "n_nonpositive_excluded_from_log_fit_only": int(np.sum(~np.isfinite(y))),
            }
            del bad
    matched = {}
    p1 = sub.loc[sub["pair_id"].eq(PAIRS[0])]
    for lead in (10, 30):
        part = p1.loc[p1["lead_day"].eq(lead)]
        wide = part.pivot_table(index=["rho", "realization"], columns="storage_value", values="W_m", aggfunc="first")
        if list(wide.columns) != [0.0001, 0.001, 0.01] and set(np.round(wide.columns, 10)) != {0.0001, 0.001, 0.01}:
            wide.columns = [float(c) for c in wide.columns]
        cols = sorted(wide.columns)
        low, mid, high = cols
        matched[str(lead)] = {
            "n_triples": int(len(wide)),
            "median_W_high_minus_low_m": float(np.nanmedian(wide[high] - wide[low])),
            "median_W_mid_minus_low_m": float(np.nanmedian(wide[mid] - wide[low])),
        }
    lineage = frame.loc[frame["experiment_group"].eq("C") & frame["storage_value"].astype(float).eq(0.001) & frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0]) & frame["lead_day"].eq(10)]
    parent = frame.loc[frame["dataset"].eq("D03") & frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0]) & frame["lead_day"].eq(10)].set_index("case_id")
    parent_w = parent.loc[lineage["d03_parent_case_id"].to_numpy(), "W_m"].to_numpy(float)
    return {
        "n_cases": 90, "retained_S_0p001_even_if_lineage_matches": True, "strata": strata, "descriptive_collapse": collapse,
        "matched_storage_P1": matched,
        "S_0p001_vs_D03_parent_day10_P1_median_abs_W_m": float(np.median(np.abs(lineage["W_m"].to_numpy(float) - parent_w))),
        "n_lineage_pairs": int(len(lineage)),
        "claim_limit": "The fit and rank summaries describe this grid. They do not establish a universal storage mechanism.",
    }


def _rate(mask_num, mask_den) -> dict:
    den = int(np.sum(mask_den))
    num = int(np.sum(mask_num & mask_den))
    return dict(numerator=num, denominator=den, rate=(None if den == 0 else num / den))


def design_d(frame) -> dict:
    lead = frame.loc[frame["quantity"].eq("lead")].copy()
    lead["sign_determined"] = pd.to_numeric(lead["sign_determined"], errors="coerce")
    lead["record_contradiction"] = pd.to_numeric(lead["record_contradiction"], errors="coerce")
    lead["ambiguous_truth_sign_error"] = pd.to_numeric(lead["ambiguous_truth_sign_error"], errors="coerce")
    groups = {}
    for lead_day in (10, 30):
        for pair in PAIRS:
            for track in TRACKS:
                part = lead.loc[lead["lead_day"].eq(lead_day) & lead["pair_id"].eq(pair) & lead["track"].eq(track)]
                det = part["sign_determined"].eq(1)
                amb = part["sign_determined"].eq(0)
                key = f"{lead_day}|{pair}|{track}"
                groups[key] = {
                    "n": int(len(part)),
                    "sign_determined": _rate(det.to_numpy(), np.ones(len(part), bool)),
                    "record_contradiction": _rate(part["record_contradiction"].eq(1).to_numpy(), det.to_numpy()),
                    "ambiguous_truth_sign_error": _rate(part["ambiguous_truth_sign_error"].eq(1).to_numpy(), amb.to_numpy()),
                    "envelope_inclusion": _rate(pd.to_numeric(part["envelope_includes"], errors="coerce").eq(1).to_numpy(), pd.to_numeric(part["envelope_includes"], errors="coerce").notna().to_numpy()),
                }
                unc = part["unconfined"].astype(int).eq(1)
                groups[key]["unconfined_record_contradiction"] = _rate(part["record_contradiction"].eq(1).to_numpy(), (det & unc).to_numpy())
                groups[key]["unconfined_ambiguous_truth_sign_error"] = _rate(part["ambiguous_truth_sign_error"].eq(1).to_numpy(), (amb & unc).to_numpy())
                groups[key]["unconfined_n"] = int(unc.sum())
    boundary = []
    raw = lead.loc[lead["track"].eq("raw")]
    for lead_day in (10, 30):
        for pair in PAIRS:
            part = raw.loc[raw["lead_day"].eq(lead_day) & raw["pair_id"].eq(pair)]
            for rho, cell in part.groupby(part["rho"].astype(float)):
                det = cell["sign_determined"].eq(1)
                sr_d = cell.loc[det, "SR"].to_numpy(float)
                sr_u = cell.loc[~det, "SR"].to_numpy(float)
                overlap = None
                if len(sr_d) and len(sr_u):
                    overlap = not (sr_d.max() < sr_u.min() or sr_u.max() < sr_d.min())
                boundary.append(dict(lead=lead_day, pair=pair, rho=float(rho), n=int(len(cell)), n_determined=int(det.sum()),
                                     SR_determined=[None if len(sr_d) == 0 else float(sr_d.min()), None if len(sr_d) == 0 else float(sr_d.max())],
                                     SR_undetermined=[None if len(sr_u) == 0 else float(sr_u.min()), None if len(sr_u) == 0 else float(sr_u.max())],
                                     SR_ranges_overlap=overlap))
    rel = raw.loc[raw["pair_id"].eq(PAIRS[0])].copy()
    rel["W_over_absE"] = rel["W_m"] / rel["E_true_m"].abs()
    rel.loc[rel["E_true_m"].eq(0) | ~np.isfinite(rel["W_over_absE"]), "W_over_absE"] = np.nan
    relative = {}
    for lead_day in (10, 30):
        col = rel.loc[rel["lead_day"].eq(lead_day), "W_over_absE"]
        relative[str(lead_day)] = dict(n_defined=int(col.notna().sum()), n=int(len(col)), median=float(col.median()) if col.notna().any() else None)
    return {
        "definition": "Sign is determined only if inf > 0 or sup < 0. An endpoint at 0 is not determined. Record contradiction is a determined envelope whose tool sign differs, with a zero tool counted as disagreement. Ambiguous truth-sign error is counted only when the envelope includes 0. Envelope inclusion is a separate count. An unconfined truth error is not relabelled as a global contradiction.",
        "groups": groups, "boundary_by_observed_rho": boundary, "relative_width_P1_raw": relative,
    }


def spearman(x, y) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    out = dict(n=n, rho=None, p=None, n_negative_y=int(np.sum(y[mask] < 0)) if n else 0, n_zero_y=int(np.sum(y[mask] == 0)) if n else 0, n_omitted_nonfinite=int((~mask).sum()))
    if n >= 3:
        result = stats.spearmanr(x[mask], y[mask])
        out["rho"] = float(result.statistic)
        out["p"] = float(result.pvalue)
    out["p_note"] = "self-imposed descriptive probability, not a requested gate"
    return out


def design_e(frame) -> dict:
    lead = frame.loc[frame["quantity"].eq("lead")].copy()
    lead["effect_ratio"] = pd.to_numeric(lead["effect_ratio"], errors="coerce")
    lead["normalized_future_size"] = pd.to_numeric(lead["normalized_future_size"], errors="coerce")
    lead["normalized_future_change"] = pd.to_numeric(lead["normalized_future_change"], errors="coerce")
    overall = {}
    for lead_day in (10, 30):
        for pair in PAIRS:
            for track in TRACKS:
                part = lead.loc[lead["lead_day"].eq(lead_day) & lead["pair_id"].eq(pair) & lead["track"].eq(track)]
                overall[f"{lead_day}|{pair}|{track}"] = {
                    "size": spearman(part["normalized_future_size"], part["effect_ratio"]),
                    "signed_change": spearman(part["normalized_future_change"], part["effect_ratio"]),
                }
    raw = lead.loc[lead["track"].eq("raw") & lead["pair_id"].eq(PAIRS[0])]
    conditional = {"rho": {}, "recency_A": {}, "scale_B_and_D03_N6": {}}
    for lead_day in (10, 30):
        part = raw.loc[raw["lead_day"].eq(lead_day)]
        for rho, cell in part.groupby(part["rho"].astype(float)):
            conditional["rho"][f"{lead_day}|{rho}"] = spearman(cell["normalized_future_size"], cell["effect_ratio"])
        a = part.loc[part["experiment_group"].eq("A")]
        for recency, cell in a.groupby("recency_label"):
            conditional["recency_A"][f"{lead_day}|{recency}"] = spearman(cell["normalized_future_size"], cell["effect_ratio"])
        b = part.loc[part["experiment_group"].eq("B") | (part["dataset"].eq("D03") & part["N_transitions"].astype(int).eq(6))]
        for scale, cell in b.groupby(b["Q_scale"].astype(float)):
            conditional["scale_B_and_D03_N6"][f"{lead_day}|{scale}"] = spearman(cell["normalized_future_size"], cell["effect_ratio"])
    diag = {
        "n_lead_rows": int(len(lead)),
        "n_pumping_safe_std_replacement": int(pd.to_numeric(lead["pumping_safe_std_replacement"], errors="coerce").fillna(0).sum()),
        "n_pumping_detrended": int(pd.to_numeric(lead["pumping_detrended"], errors="coerce").fillna(0).sum()),
        "n_head_detrended": int(pd.to_numeric(lead["head_detrended"], errors="coerce").fillna(0).sum()),
        "n_negative_effect_ratio": int((lead["effect_ratio"] < 0).sum()),
        "n_zero_effect_ratio": int((lead["effect_ratio"] == 0).sum()),
        "n_undefined_effect_ratio": int(lead["effect_ratio"].isna().sum()),
    }
    return {
        "source": "A installed decode/_preprocess diagnostic. Future pumping change divided by the safe cumulative denominator after conditional detrending. Cases with a replaced denominator stay in the table.",
        "overall": overall, "conditional": conditional, "diagnostics": diag,
        "claim_limit": "Associations are conditional descriptions. They do not identify a cause.",
    }


def reference_summary(manifest_all) -> dict:
    groups = {f"{pair}|{lead}": [] for pair in PAIRS for lead in (10, 30)}
    missing = 0
    for record, wb in manifest_all:
        path = wb / "reference" / f"{record['case_id']}.json"
        truth_path = wb / "truth_eval" / f"{record['case_id']}.json"
        width_path = wb / "W" / f"{record['case_id']}.json"
        if not path.exists():
            missing += 1
            continue
        ref = json.loads(path.read_text())
        truth = json.loads(truth_path.read_text())
        width = json.loads(width_path.read_text())
        for pair in PAIRS:
            true = np.asarray(truth["E_true"][pair], float)
            point = np.asarray(ref["pairs"][pair]["E_point"], float)
            inf = np.asarray(width["envelope"][pair]["inf"], float)
            sup = np.asarray(width["envelope"][pair]["sup"], float)
            for lead in (10, 30):
                groups[f"{pair}|{lead}"].append(mt.row_metrics(true[lead - 1], point[lead - 1], float(sup[lead - 1] - inf[lead - 1]), inf[lead - 1], sup[lead - 1]))
    pooled = {}
    for key, rows in groups.items():
        den = mt.denominators(rows)
        den["median_effect_ratio"] = _median([r["effect_ratio"] for r in rows if r["effect_ratio"] is not None])
        pooled[key] = den
    return {"n_missing": missing, "pooled": pooled, "role": "Structurally matched reference. These rows are not a ranking against the forecasting tool."}


def _median(values):
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    return None if len(x) == 0 else float(np.median(x))


def field_bands(path: Path) -> dict:
    if sha(path) != FIELD_CSV:
        raise RuntimeError("field_pause_rho.csv hash changed; the figure band was not guessed")
    rows = list(csv.DictReader(path.open()))
    derived = [r for r in rows if r["figure1_role"] == "short_daily_off_derived_complement"]
    banned = [r for r in rows if r["figure1_role"] == "do_not_plot_as_observed_pause"]
    if any(r["observed_continuous_well_pause"] != "false" for r in rows):
        raise RuntimeError("an observed continuous pause appeared; the band rule must be reread")
    if any(r["interval_class"] == "administrative_accounting_remainder_not_observed_pause" and r in derived for r in rows):
        raise RuntimeError("an accounting remainder entered the derived band")
    spans = []
    for hours in sorted({r["documented_on_hours"] for r in derived}):
        part = [r for r in derived if r["documented_on_hours"] == hours]
        rhos = [r["rho_pause_over_t95"] for r in part]
        floats = [float(v) for v in rhos]
        spans.append(dict(documented_on_hours=hours, n=len(part), rho_min_csv=min(rhos, key=float), rho_max_csv=max(rhos, key=float),
                          rho_min=min(floats), rho_max=max(floats), sources=sorted(set(r["source_id"] for r in part)),
                          label="inferred from reported operating hours, not an observed metered pause"))
    return {"n_rows": len(rows), "n_derived_plotted": len(derived), "n_accounting_not_plotted": len(banned), "spans": spans,
            "observed_continuous_well_pause_rows": 0}


def required_missing() -> list[str]:
    missing = []
    needed = [
        P3 / "protocol_freeze.json", P3 / "protocol.sha256", P3 / "cases/generation_checks.json",
        P3 / "cases/derived_manifest.json", P3 / "cases/tool_inputs_H10.npz", P3 / "cases/tool_inputs_H30.npz",
        P3 / "normalization.csv", P3 / "field_pause_rho.csv",
    ]
    for path in needed:
        if not path.exists():
            missing.append(str(path))
    if (P3 / "cases/derived_manifest.json").exists():
        man = json.loads((P3 / "cases/derived_manifest.json").read_text())
        wb = P3 / "wb"
        for record in man:
            for folder in ("W", "truth_eval", "reference"):
                path = wb / folder / f"{record['case_id']}.json"
                if not path.exists():
                    missing.append(str(path))
                    break
            else:
                continue
            break
    for track in TRACKS:
        for horizon in (10, 30):
            path = P3 / "tools" / f"timesfm_{track}_H{horizon}.npz"
            if not path.exists():
                missing.append(str(path))
    return missing


def style():
    if Path(FONT_REG).exists():
        font_manager.fontManager.addfont(FONT_REG)
        font_manager.fontManager.addfont(FONT_BOLD)
        family = "Times New Roman"
    else:
        family = "DejaVu Serif"
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [family, "Times", "DejaVu Serif"],
        "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.labelsize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
    })
    return family


def _save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def draw_figure1(raw, bands, out: Path) -> dict:
    """Marker channels: shape is storage type, size is T, fill is sign, color is W/|E|."""
    family = style()
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.32, left=0.10, right=0.90, bottom=0.36, top=0.90)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[1].sharey(axes[0])
    cax = fig.add_subplot(grid[0, 2])
    positive = raw["ratio_w"].to_numpy(float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    cmap = mpl.colormaps["viridis"]
    shapes = (
        ("confined", "o", "Confined"),
        ("leaky", "s", "Leaky"),
        ("unconfined", "^", "Unconfined"),
        ("fixedc_confined", "v", "Fixed-c confined"),
        ("fixedc_leaky", "h", "Fixed-c leaky"),
        ("fixedc_linear", "D", "Fixed-c, S = 0.01, linear approximation"),
    )
    counts = {}
    for ax, lead, letter in zip(axes, (10, 30), ("a", "b")):
        part = raw.loc[raw["lead_day"].eq(lead)].copy()
        determined = part["sign_determined"].astype(str).isin(["1", "1.0"]) | part["sign_determined"].eq(1)
        finite = np.isfinite(part["ratio_w"].to_numpy(float))
        drawn = 0
        for storage, marker, _label in shapes:
            for level_t, size, layer in ((500.0, 64, 2), (50.0, 20, 3)):
                choose = part["storage_type"].eq(storage) & np.isclose(part["T_m2_d"].astype(float), level_t)
                cell = part.loc[choose.to_numpy() & finite & determined.to_numpy()]
                if len(cell):
                    ax.scatter(cell["rho"], cell["SR"], c=cell["ratio_w"], cmap=cmap, norm=norm, marker=marker, s=size, linewidths=0.3, edgecolors="0.15", zorder=layer)
                open_cell = part.loc[choose.to_numpy() & finite & ~determined.to_numpy()]
                if len(open_cell):
                    edges = cmap(norm(open_cell["ratio_w"].to_numpy(float)))
                    ax.scatter(open_cell["rho"], open_cell["SR"], marker=marker, s=size, facecolors="none", edgecolors=edges, linewidths=1.05, zorder=layer + 2)
                drawn += len(cell) + len(open_cell)
        missing = part.loc[~finite]
        if len(missing):
            ax.scatter(missing["rho"], missing["SR"], marker="x", c="0.35", s=16, linewidths=0.6, zorder=5)
        if drawn + len(missing) != len(part):
            raise RuntimeError(f"figure 1 dropped day-{lead} rows: drawn {drawn}, missing {len(missing)}, rows {len(part)}")
        counts[str(lead)] = {"drawn": drawn, "undefined_ratio": int(len(missing)), "rows": int(len(part))}
        for span in bands["spans"]:
            ax.axvspan(span["rho_min"], span["rho_max"], color="#cfcfcf" if span["documented_on_hours"] == "14" else "#8d8d8d", alpha=0.45, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.004, 8)
        ax.set_xlabel("Pause / response time", fontweight="bold", labelpad=6)
        ax.set_title(f"({letter}) Day {lead}", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Signal ratio", fontweight="bold", labelpad=6)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(positive)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Record width / |true effect|", fontweight="bold", labelpad=8)
    handles = [plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor="0.25", markeredgecolor="0.1", markersize=7, label=label) for _storage, marker, label in shapes]
    handles.extend([
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25", markeredgecolor="0.1", markersize=4.5, label="T = 50 m$^2$/d"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25", markeredgecolor="0.1", markersize=9, label="T = 500 m$^2$/d"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25", markeredgecolor="0.1", markersize=7, label="Filled: sign determined"),
        plt.Line2D([0], [0], marker="o", color="0.25", markerfacecolor="none", markeredgecolor="0.25", markersize=7, label="Open: sign not determined"),
        plt.Rectangle((0, 0), 1, 1, color="#cfcfcf", alpha=0.7, label="14 h stated on-time, inferred nightly off"),
        plt.Rectangle((0, 0), 1, 1, color="#8d8d8d", alpha=0.7, label="15 h stated on-time, inferred nightly off"),
    ])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.46, 0.012), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Answerability of the pumping record", fontsize=13, fontweight="bold")
    _save(fig, out / "figure1_answerability")
    return {"font_family_requested": family, "figure1": "figure1_answerability", "figure1_points": counts}


def draw_figures(frame, bands, out: Path) -> dict:
    family = style()
    out.mkdir(parents=True, exist_ok=True)
    records = {"font_family_requested": family}
    raw = frame.loc[frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0])].copy()
    raw["lead_day"] = raw["lead_day"]
    raw["ratio_w"] = raw["W_m"] / raw["E_true_m"].abs()
    raw.loc[raw["E_true_m"].eq(0) | ~np.isfinite(raw["ratio_w"]) | (raw["ratio_w"] <= 0), "ratio_w"] = np.nan
    records.update(draw_figure1(raw, bands, out))

    a = frame.loc[frame["experiment_group"].eq("A") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0])].copy()
    order = [("2", "recent"), ("2", "old"), ("18", "recent"), ("18", "old")]
    labels = ["N=2\nrecent", "N=2\nold", "N=18\nrecent", "N=18\nold"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=False)
    for ax, quantity, title in zip(axes, ("max_days_1_10", "lead"), ("Days 1–10 maximum", "Day 10")):
        part = a.loc[a["quantity"].eq(quantity)]
        if quantity == "lead":
            part = part.loc[part["lead_day"].eq(10)]
        data = []
        for n, recency in order:
            cell = part.loc[part["N_transitions"].astype(int).eq(int(n)) & part["recency_label"].eq(recency), "W_m"].to_numpy(float)
            data.append(cell)
        ax.boxplot(data, widths=0.6, showfliers=False)
        for i, cell in enumerate(data, start=1):
            ax.scatter(np.full(len(cell), i) + np.linspace(-0.12, 0.12, len(cell)), cell, s=10, c="#1b4f72", linewidths=0, zorder=3)
        ax.set_xticks(range(1, 5), labels)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Record width (m)", fontweight="bold", labelpad=6)
    fig.suptitle("Switch count and recency", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    _save(fig, out / "figure2_count_recency")
    records["figure2"] = "figure2_count_recency"

    tool = frame.loc[frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0])].copy()
    tool["effect_ratio"] = pd.to_numeric(tool["effect_ratio"], errors="coerce")
    tool["record_contradiction"] = pd.to_numeric(tool["record_contradiction"], errors="coerce").fillna(0).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.6), sharey=True)
    for row, lead in enumerate((10, 30)):
        part = tool.loc[tool["lead_day"].eq(lead)]
        contra = part["record_contradiction"].eq(1)
        axes[row, 0].scatter(part.loc[~contra, "rho"], part.loc[~contra, "effect_ratio"], s=12, c="#1b4f72", linewidths=0)
        axes[row, 0].scatter(part.loc[contra, "rho"], part.loc[contra, "effect_ratio"], s=16, c="#9b2226", marker="x", linewidths=0.7)
        axes[row, 0].set_xscale("log")
        axes[row, 0].set_xlim(0.18, 6)
        axes[row, 0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        axes[row, 0].set_xticks([0.25, 1, 4])
        axes[row, 0].set_xticklabels(["0.25", "1", "4"])
        axes[row, 0].set_xlabel("Pause / response time", fontweight="bold", labelpad=4)
        axes[row, 1].scatter(part.loc[~contra, "SR"], part.loc[~contra, "effect_ratio"], s=12, c="#1b4f72", linewidths=0)
        axes[row, 1].scatter(part.loc[contra, "SR"], part.loc[contra, "effect_ratio"], s=16, c="#9b2226", marker="x", linewidths=0.7)
        axes[row, 1].set_xscale("log")
        axes[row, 1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        axes[row, 1].set_xlabel("Signal ratio", fontweight="bold", labelpad=4)
        aa = part.loc[part["experiment_group"].eq("A")]
        xs, ys, cs = [], [], []
        for i, (n, recency) in enumerate(order):
            cell = aa.loc[aa["N_transitions"].astype(int).eq(int(n)) & aa["recency_label"].eq(recency)]
            jitter = np.linspace(-0.15, 0.15, len(cell)) if len(cell) else []
            axes[row, 2].scatter(np.full(len(cell), i) + jitter, cell["effect_ratio"], s=12,
                                 c=np.where(cell["record_contradiction"].eq(1), "#9b2226", "#1b4f72"), linewidths=0)
        axes[row, 2].set_xticks(range(4), ["2\nrecent", "2\nold", "18\nrecent", "18\nold"])
        axes[row, 2].set_xlabel("Count and placement", fontweight="bold", labelpad=6)
        axes[row, 0].set_ylabel(f"Day {lead} tool effect ratio", fontweight="bold", labelpad=4)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b4f72", markersize=6, label="Envelope sign not contradicted"),
        plt.Line2D([0], [0], marker="x", color="#9b2226", markersize=6, label="Tool sign contradicts the record"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False)
    fig.suptitle("Record features and the tool ratio", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.18, top=0.90, hspace=0.62, wspace=0.38)
    _save(fig, out / "figure3_tool_ratio")
    records["figure3"] = "figure3_tool_ratio"

    normed = tool.copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    for ax, lead in zip(axes, (10, 30)):
        part = normed.loc[normed["lead_day"].eq(lead)]
        size = pd.to_numeric(part["normalized_future_size"], errors="coerce")
        ok = size.gt(0) & part["effect_ratio"].notna()
        ax.scatter(size[ok], part.loc[ok, "effect_ratio"], s=12, c="#1b4f72", linewidths=0)
        ax.set_xscale("log")
        ax.set_xlabel("Normalized future pumping change", fontweight="bold", labelpad=6)
        ax.set_title(f"Day {lead}", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Tool effect ratio", fontweight="bold", labelpad=6)
    fig.suptitle("Normalized pumping change and the tool ratio", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    _save(fig, out / "figureS1_normalized_size")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=False, sharey=False)
    d03 = frame.loc[frame["dataset"].eq("D03") & frame["N_transitions"].astype(int).eq(6) & frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0])]
    new = frame.loc[frame["experiment_group"].eq("B") & frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0])]
    for row, lead in enumerate((10, 30)):
        base = d03.loc[d03["lead_day"].eq(lead)].set_index("case_id")
        for col, scale in enumerate((0.3, 3.0)):
            part = new.loc[new["lead_day"].eq(lead) & new["Q_scale"].astype(float).eq(scale)]
            parent = base.loc[part["d03_parent_case_id"].to_numpy()]
            rho = part["rho"].astype(float).to_numpy()
            for value, color, marker in ((0.25, "#1b4f72", "o"), (1.0, "#b08900", "s"), (4.0, "#9b2226", "D")):
                pick = np.isclose(rho, value)
                axes[row, col].scatter(parent["W_m"].to_numpy(float)[pick], part["W_m"].to_numpy(float)[pick], s=14, c=color, marker=marker, linewidths=0, label=f"Pause ratio {value:g}" if row == 0 and col == 0 else None)
            lims = [min(parent["W_m"].min(), part["W_m"].min()), max(parent["W_m"].max(), part["W_m"].max())]
            axes[row, col].plot(lims, lims, color="0.45", linewidth=0.7)
            axes[row, col].set_xlabel("Scale-1 record width (m)", fontweight="bold", labelpad=4)
            axes[row, col].set_ylabel(f"Scale-{scale:g} record width (m)", fontweight="bold", labelpad=4)
            axes[row, col].set_title(f"Day {lead}", loc="left", fontsize=11, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False)
    fig.suptitle("Pumping scale and record width", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.90, hspace=0.38, wspace=0.32)
    _save(fig, out / "figureS2_pumping_scale")

    c = frame.loc[frame["experiment_group"].eq("C") & frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    for ax, lead in zip(axes, (10, 30)):
        part = c.loc[c["lead_day"].eq(lead)]
        rel = part["W_m"] / part["E_true_m"].abs()
        for rho, marker, color in ((0.25, "o", "#1b4f72"), (1.0, "s", "#b08900"), (4.0, "D", "#2a9d8f")):
            cell = part["rho"].astype(float).eq(rho)
            ax.scatter(part.loc[cell, "storage_value"], rel[cell], marker=marker, c=color, s=18, linewidths=0, label=f"Pause ratio {rho:g}" if lead == 10 else None)
        ax.set_xscale("log")
        ax.set_xlabel("Storage coefficient", fontweight="bold", labelpad=6)
        ax.set_title(f"Day {lead}", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Record width / |true effect|", fontweight="bold", labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle("Storage grid at fixed transmissivity and resistance", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.86, wspace=0.18)
    _save(fig, out / "figureS3_storage")
    records["font_drawn"] = family
    records["display_note"] = "Logarithmic color and axis scales are display choices. They do not remove rows and they are not scientific thresholds."
    return records


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def first_paragraph(a, b, c, d, e, bands) -> str:
    count = a["paired"]["W|lead|P1_continue_vs_stop|count|recent"]
    rec = a["paired"]["W|lead|P1_continue_vs_stop|recency|N2"]
    r_count = a["paired"]["ratio|10|P1_continue_vs_stop|raw|count|recent"]
    r_rec = a["paired"]["ratio|10|P1_continue_vs_stop|raw|recency|N2"]
    r30_count = a["paired"]["ratio|30|P1_continue_vs_stop|raw|count|recent"]
    r30_rec = a["paired"]["ratio|30|P1_continue_vs_stop|raw|recency|N2"]
    scale_lo = b["paired"]["10|P1_continue_vs_stop|0.3"]
    scale_hi = b["paired"]["10|P1_continue_vs_stop|3.0"]
    store = c["matched_storage_P1"]["10"]
    det = d["groups"]["10|P1_continue_vs_stop|raw"]
    det30 = d["groups"]["30|P1_continue_vs_stop|raw"]
    size = e["overall"]["10|P1_continue_vs_stop|raw"]["size"]
    size30 = e["overall"]["30|P1_continue_vs_stop|raw"]["size"]
    span14 = next(s for s in bands["spans"] if s["documented_on_hours"] == "14")
    span15 = next(s for s in bands["spans"] if s["documented_on_hours"] == "15")
    return (
        f"On the separated grid, moving the burst from 2 to 18 switches changed the day-10 record width by a median {fmt(count['median'])} m "
        f"(recent placement, {count['n']} matched layer-realization pairs) and moving the same burst from the recent to the old placement changed it by a median {fmt(rec['median'])} m at N=2. "
        f"The raw tool-effect ratio shifted by a median {fmt(r_count['median'])} for that count contrast and {fmt(r_rec['median'])} for that recency contrast at day 10, and by {fmt(r30_count['median'])} and {fmt(r30_rec['median'])} at day 30. "
        f"Against the reused scale-1 records, day-10 width at scale 0.3 and scale 3 had median differences {fmt(scale_lo['W_difference_m']['median'])} m and {fmt(scale_hi['W_difference_m']['median'])} m; "
        f"{scale_lo['n_abs_W_difference_within_frozen_delta']}/{scale_lo['n_pairs']} and {scale_hi['n_abs_W_difference_within_frozen_delta']}/{scale_hi['n_pairs']} of those pairs fell inside the frozen effect delta. "
        f"With transmissivity and resistance fixed, day-10 width at storage 0.01 minus storage 0.0001 had median {fmt(store['median_W_high_minus_low_m'])} m across {store['n_triples']} matched pause-ratio and realization triples. "
        f"At day 10 the record determined a sign in {det['sign_determined']['numerator']}/{det['sign_determined']['denominator']} raw continue-versus-stop cases and the tool contradicted that sign in {det['record_contradiction']['numerator']}/{det['record_contradiction']['denominator']}; "
        f"the day-30 counts were {det30['sign_determined']['numerator']}/{det30['sign_determined']['denominator']} determined and {det30['record_contradiction']['numerator']}/{det30['record_contradiction']['denominator']} contradicted. "
        f"The rank correlation between normalized future pumping size and the raw tool ratio was {fmt(size['rho'])} at day 10 (n={size['n']}) and {fmt(size30['rho'])} at day 30 (n={size30['n']}). "
        f"Field pause/response ratios inferred from reported 14 h and 15 h operation span csv values {span14['rho_min_csv']} to {span14['rho_max_csv']} and {span15['rho_min_csv']} to {span15['rho_max_csv']}, all below the experiment minimum 0.25; no observed continuous well pause was plotted."
    )


def write_readout(doc, out_dir: Path):
    text = [doc["first_paragraph"], "", "## Designs", ""]
    text.append("The subjects of the comparisons are the pumping record and the aquifer. The forecasting tool is described only where the requested ratio and sign counts require it.")
    text.append("")
    text.append("### Count and recency")
    text.append(json.dumps(doc["A"]["discrete_convention"], default=str)[:2500])
    text.append("")
    text.append("Paired differences are the higher setting minus the lower setting: N=18 minus N=2, and old minus recent. Both schedule pairs and all four frozen tracks are in the JSON.")
    text.append("")
    text.append("### Pumping scale")
    text.append(doc["B"]["note"])
    text.append("")
    text.append("### Storage")
    text.append(doc["C"]["claim_limit"])
    text.append("")
    text.append("### Sign and answerability")
    text.append(doc["D"]["definition"])
    text.append("")
    text.append("### Normalization diagnostic")
    text.append(doc["E"]["source"] + " " + doc["E"]["claim_limit"])
    text.append("")
    text.append("### Field pause coverage")
    text.append("Derived nightly complements of stated operating hours are drawn. Administrative accounting remainders are not drawn. The searched sources did not provide an observed continuous well-pause distribution.")
    text.append("")
    text.append("### Reference")
    text.append(doc["reference"]["role"])
    text.append("")
    text.append("### Self-imposed checks")
    text.append("Wilcoxon probabilities and the logarithmic display scales are self-imposed. They are not requested completion gates and they do not remove rows.")
    (out_dir / "READOUT.md").write_text("\n".join(text) + "\n")
    (out_dir / "READOUT.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")


def compare_d03(rows) -> dict:
    official = pd.read_csv(P2 / "analysis/cell_metrics.csv")
    built = pd.DataFrame(rows)
    keys = ["case_id", "pair_id", "track", "quantity", "lead_day"]
    official["lead_day"] = official["lead_day"].fillna(-1).astype(int)
    built["lead_day"] = built["lead_day"].replace("", -1).astype(int)
    left = official.set_index(keys)
    right = built.set_index(keys)
    if set(left.index) != set(right.index):
        raise RuntimeError("D03 rebuilt keys differ from the frozen cell_metrics")
    mismatches = []
    for column in ("W_m", "E_true_m", "E_tool_m", "effect_ratio"):
        a = pd.to_numeric(left[column], errors="coerce")
        b = pd.to_numeric(right[column], errors="coerce")
        gap = (a - b).abs()
        both_missing = a.isna() & b.isna()
        bad = ~both_missing & (gap.fillna(np.inf) > 1e-8)
        if bad.any():
            mismatches.append({"column": column, "n": int(bad.sum()), "max_abs": float(gap[bad].max())})
    env_gap = (pd.to_numeric(left["envelope_includes"], errors="coerce").fillna(-1) != pd.to_numeric(right["envelope_includes"], errors="coerce").fillna(-1)).sum()
    return {"n_rows": int(len(built)), "value_mismatches": mismatches, "envelope_flag_mismatches": int(env_gap)}


def self_test() -> dict:
    checks = []
    endpoint = sign_state(0.0, 1.0)
    checks.append(endpoint["sign_determined"] == 0 and endpoint["includes_zero"] == 1)
    positive = sign_state(0.1, 1.0)
    flags = contradiction_flags(positive, -0.2, 0.4, "confined")
    checks.append(flags["record_contradiction"] == 1 and flags["ambiguous_truth_sign_error"] == 0)
    ambiguous = contradiction_flags(sign_state(-1.0, 1.0), -0.2, 0.4, "unconfined")
    checks.append(ambiguous["record_contradiction"] == 0 and ambiguous["ambiguous_truth_sign_error"] == 1 and ambiguous["unconfined"] == 1)
    zero_tool = contradiction_flags(positive, 0.0, 0.4, "confined")
    checks.append(zero_tool["record_contradiction"] == 1)
    bands = field_bands(P3 / "field_pause_rho.csv")
    checks.append(bands["n_accounting_not_plotted"] > 0 and all(s["rho_max"] < 0.25 for s in bands["spans"]))
    if not all(checks):
        raise RuntimeError(f"self-test failed: {checks}")
    return {"passed": True, "n_checks": len(checks), "field_spans": bands["spans"]}


def d03_loader_check() -> dict:
    digest = (P2 / "protocol.sha256").read_text().split()[0]
    if digest != D03_PROTOCOL or sha(P2 / "analysis/cell_metrics.csv") != D03_CELL:
        raise RuntimeError("frozen D03 protocol or cell_metrics hash changed")
    if sha(ROOT / "results/pilot/protocol.md") != PHYSICAL:
        raise RuntimeError("physical protocol hash changed")
    tools = load_tools(P2 / "tools", digest, 540, {10: P2 / "cases/tool_inputs_H10.npz", 30: P2 / "cases/tool_inputs_H30.npz"})
    manifest = json.loads((P2 / "cases/derived_manifest.json").read_text())
    rows = []
    for record in manifest:
        record = dict(record)
        record["experiment_group"] = "D03"
        record["Q_scale"] = 1.0
        record["recency_label"] = "d03_schedule"
        rows.extend(case_rows(record, "D03", P2 / "wb", tools, "official_analysis"))
    comparison = compare_d03(rows)
    if comparison["value_mismatches"] or comparison["envelope_flag_mismatches"]:
        raise RuntimeError(f"D03 loader does not reproduce frozen metrics: {comparison}")
    payload = {"passed": True, "comparison": comparison, "d03_protocol_sha256": digest, "cell_metrics_sha256": D03_CELL}
    (P3 / "C_D03_LOADER_CHECK.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def coverage_formulas(frame) -> dict:
    return {
        "unique_cases": int(frame["case_id"].nunique()),
        "n_d03": int(frame.loc[frame["dataset"].eq("D03"), "case_id"].nunique()),
        "n_d04": int(frame.loc[frame["dataset"].eq("D04"), "case_id"].nunique()),
        "n_rows": int(len(frame)),
        "n_lead": int(frame["quantity"].eq("lead").sum()),
        "n_max": int(frame["quantity"].eq("max_days_1_10").sum()),
        "formulas": {
            "W_m": "sup minus inf at the named lead, or the maximum of that width over leads 1-10",
            "effect_ratio": "E_tool / E_true when E_true is finite and nonzero and E_tool is finite; signed and unclipped",
            "sign_determined": "1 only when inf > 0 or sup < 0",
            "record_contradiction": "determined envelope and tool sign differs; tool zero counts as disagreement",
            "ambiguous_truth_sign_error": "envelope includes 0, truth is nonzero, and tool sign differs from truth",
            "envelope_includes": "inf - 1e-12 m <= E_tool <= sup + 1e-12 m",
            "normalized_future_size": "absolute installed-source normalized future pumping contrast at the lead",
            "frozen_delta": "max(0.25|E_true|, 0.02 m), taken from the stored truth delta",
        },
    }


def source_audit(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    for item in manifest["outputs"]:
        if sha(item["path"]) != item["sha256"]:
            mismatches.append(item["path"])
    frame = pd.read_csv(P3 / "cell_metrics.csv")
    if frame["case_id"].nunique() != 1230:
        mismatches.append("unique cases")
    readout = json.loads((P3 / "READOUT.json").read_text())
    part = frame.loc[frame["quantity"].eq("lead") & frame["track"].eq("raw") & frame["pair_id"].eq(PAIRS[0]) & frame["lead_day"].eq(10)]
    det = part["sign_determined"].astype(float).eq(1)
    recounted = int((part["record_contradiction"].astype(float).eq(1) & det).sum())
    quoted = readout["D"]["groups"]["10|P1_continue_vs_stop|raw"]["record_contradiction"]["numerator"]
    if recounted != quoted:
        mismatches.append("contradiction numerator")
    audit = {"verified": not mismatches, "mismatches": mismatches, "recounted_day10_raw_p1_contradictions": recounted}
    (P3 / "C_SOURCE_AUDIT.json").write_text(json.dumps(audit, indent=1) + "\n")
    if mismatches:
        raise RuntimeError(f"source audit failed: {mismatches}")
    return audit


def run_full():
    missing = required_missing()
    if missing:
        print(json.dumps({"status": "missing_inputs", "missing": missing[:12], "n_missing": len(missing)}))
        return 2
    checks = json.loads((P3 / "cases/generation_checks.json").read_text())
    freeze = json.loads((P3 / "protocol_freeze.json").read_text())
    digest = (P3 / "protocol.sha256").read_text().split()[0]
    if not checks.get("all_ok") or checks.get("fixture") or digest != freeze["protocol_sha256"] or checks["protocol_sha256"] != digest:
        raise RuntimeError("phase3 generation is not an official frozen grid")
    d03_check = d03_loader_check()
    d03_manifest = json.loads((P2 / "cases/derived_manifest.json").read_text())
    d04_manifest = json.loads((P3 / "cases/derived_manifest.json").read_text())
    if len(d04_manifest) != 690 or len(d03_manifest) != 540:
        raise RuntimeError("case counts are not 540 and 690")
    d03_tools = load_tools(P2 / "tools", D03_PROTOCOL, 540, {10: P2 / "cases/tool_inputs_H10.npz", 30: P2 / "cases/tool_inputs_H30.npz"})
    d04_tools = load_tools(P3 / "tools", digest, 690, {10: P3 / "cases/tool_inputs_H10.npz", 30: P3 / "cases/tool_inputs_H30.npz"})
    rows = []
    for record in d03_manifest:
        record = dict(record)
        record["experiment_group"] = "D03"
        record["Q_scale"] = 1.0
        record["recency_label"] = "d03_schedule"
        rows.extend(case_rows(record, "D03", P2 / "wb", d03_tools, "official_analysis"))
    for record in d04_manifest:
        rows.extend(case_rows(record, "D04", P3 / "wb", d04_tools, "official_analysis"))
    frame = pd.DataFrame(rows)
    frame = attach_normalization(frame, P3 / "normalization.csv")
    frame["lead_day"] = pd.to_numeric(frame["lead_day"].replace("", pd.NA), errors="coerce")
    if frame["case_id"].nunique() != 1230 or frame["quantity"].eq("lead").sum() != 19680:
        raise RuntimeError("combined coverage is not 1230 cases and 19680 lead rows")
    a = design_a(frame, [r for r in d04_manifest if r["experiment_group"] == "A"])
    b = design_b(frame)
    c = design_c(frame)
    d = design_d(frame)
    e = design_e(frame)
    reference = reference_summary([(r, P2 / "wb") for r in d03_manifest] + [(r, P3 / "wb") for r in d04_manifest])
    bands = field_bands(P3 / "field_pause_rho.csv")
    figures = draw_figures(frame, bands, P3 / "figures")
    frame.to_csv(P3 / "cell_metrics.csv", index=False)
    doc = {
        "first_paragraph": first_paragraph(a, b, c, d, e, bands),
        "A": a, "B": b, "C": c, "D": d, "E": e, "reference": reference, "field": bands,
        "coverage": coverage_formulas(frame), "d03_loader_check": d03_check, "figures": figures,
        "retained": "Failed flags, near-zero effects, zero ratios, and negative ratios stay in the denominators.",
    }
    write_readout(doc, P3)
    outputs = []
    for path in [P3 / "cell_metrics.csv", P3 / "READOUT.md", P3 / "READOUT.json", P3 / "field_pause_rho.csv"]:
        outputs.append({"path": str(path), "sha256": sha(path)})
    for path in sorted((P3 / "figures").glob("figure*.*")):
        outputs.append({"path": str(path), "sha256": sha(path)})
    manifest = {
        "task": "run", "d03_protocol_sha256": D03_PROTOCOL, "d04_protocol_sha256": digest,
        "physical_protocol_sha256": PHYSICAL, "cell_metrics_d03_sha256": D03_CELL, "field_csv_sha256": FIELD_CSV,
        "coverage": doc["coverage"], "outputs": outputs, "normalization_csv_sha256": sha(P3 / "normalization.csv"),
        "n_small_001_retained": int(pd.to_numeric(frame["ratio_defined"], errors="coerce").notna().sum()),
    }
    # small flags are inside row metrics but not separate columns. Recount from truth.
    small = frame.loc[frame["quantity"].eq("lead") & frame["E_true_m"].abs().gt(0) & frame["E_true_m"].abs().le(0.02)]
    manifest["n_lead_abs_E_at_or_below_0p02_retained"] = int(len(small))
    manifest["n_W_status_not_ok"] = int((frame["W_status"] != "ok").sum())
    (P3 / "C_EXECUTION_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (P3 / "FIGURE_MANIFEST.json").write_text(json.dumps({"font": figures, "files": [x for x in outputs if "/figures/" in x["path"]], "field_policy": bands}, indent=1, default=str) + "\n")
    audit = source_audit(P3 / "C_EXECUTION_MANIFEST.json")
    (P3 / "C_PROGRESS.md").write_text("# C analysis progress\n\nOfficial 1230-case readout, figures, and source audit are written.\n\nAudit verified: " + str(audit["verified"]) + "\n")
    print(json.dumps({"status": "written", "audit": audit["verified"], "cases": 1230}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--d03-loader-check", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if args.d03_loader_check:
        print(json.dumps(d03_loader_check()))
        return 0
    return run_full()


if __name__ == "__main__":
    sys.exit(main())
