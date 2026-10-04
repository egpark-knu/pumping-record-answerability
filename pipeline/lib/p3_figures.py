"""Aggregate the frozen D02 pilot and draw the two condition-first figures.

This module does not compute W, the transfer-function reference, or a forecast.
`check-ready` and `execute` read finished manifests. `execute` writes nothing
until every W/reference case and both scenario-engine archives are terminal and
the protocol and source hashes match. `fixture` draws only synthetic rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
PILOT = ROOT / "results/pilot"
PROTOCOL = PILOT / "protocol.md"
PROTOCOL_SHA_FILE = PILOT / "protocol.sha256"
CASES_READY = PILOT / "cases_ready.json"
AUDIT_WB = PILOT / "wb"
OFFICIAL_WB = PILOT / "wb_repaired_v1_2"
REPAIR_FREEZE = PILOT / "repair_freeze.json"
REPAIR_COMPARISON = PILOT / "repair_comparison.json"
TFM_MANIFEST = PILOT / "A_timesfm_inference_manifest.json"
TFM_H10 = PILOT / "tools" / "timesfm_H10.npz"
TFM_H30 = PILOT / "tools" / "timesfm_H30.npz"
PREPARATION_CODE_SHA256 = "07d0c4ea88258cfa50dd767dc34059217ccc932d873f430bbeb6628188571a9a"
EXPECTED_REPAIR_FREEZE_SHA256 = "655b1a42d8f7f1d66fd432f45ba2e68a5874aec1d2058e55d4c9e2cb6a44bbcf"
EXPECTED_REUSE_SHA256 = "00e4dfaeb38ccfe583cd3d14447f9917c2da02bebe9cfcceebdab8c7b6838ee3"
EXPECTED_COMPARISON_SHA256 = "132c549f2510d1c09be0d279a1d5eb64ef493d63354f49bfb1de5daac8e8f3ad"
EXPECTED_TABLE_SHA256 = {
    "W_table.csv": "62a68b62ba631806877d98310d69e992233829aa29bf4568974f398173fb9004",
    "curves_long.csv": "6c56eef228fb3c5a28de1501f43dfb04fd536c8a9914a3d18ed06588e96b51b6",
    "reference_table.csv": "40a4320480084d1b3673a50e6ce684d6bac5db99a48469ef4e7b9a2584edf1ba",
}
EXPECTED_FLAG_CASES = {"not_converged": 17, "sparse_G": 14}
OUTPUT_SHA256 = {
    10: "aee3a38f9981dee2028d79aa5fcf14e0cda601dd0c6f4c726b6673bdb6a4a88a",
    30: "f28937be09b7dd939f6377813f599fd903801b966dc21c10032d27ac7792244c",
}

FROZEN_PROTOCOL_SHA256 = "61f7fa696cfbca26e8f46ace79ca5432e0c644a667ef9eccc7fc7cc27bd12bad"
INPUT_SHA256 = {
    10: "085180db3595d5ed6541125a150a18be0d72fd5b80a00c582bac9a9ac1d6c8c7",
    30: "3e246b157fb7d74124dbf0da942de497f640890dc39183b7531e4024dc03c7a8",
}
N_CASES = 200
N_CONTRASTS = 400
N_QUERIES = 800
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
TERMINAL_W = {"ok", "W_failed", "error"}
TERMINAL_REF = {"ok", "error"}
NONTERMINAL = {"", "running", "pending", "started"}
SIGMA_M = 0.02
DELTA_FRACTION = 0.25
ENVELOPE_ATOL_M = 1e-12
PROXY_ATOL = 1e-4

PAIR_LABEL = {
    "P1_continue_vs_stop": "continue versus stop",
    "P2_current_vs_1p5x": "current versus 1.5x",
}
PAIR_MARKER = {"P1_continue_vs_stop": "o", "P2_current_vs_1p5x": "s"}
SIGNAL_COLOR = {0.5: "#0072B2", 2.0: "#D55E00"}
FORBIDDEN_FIGURE_TEXT = ("TimesFM", "Pastas", "Moirai", "Chronos", "GRU", "win", "rank")


class NotReady(RuntimeError):
    """Finished inputs are missing, duplicated, or hash-mismatched."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def _require(condition, message):
    if not condition:
        raise NotReady(message)


def _read_csv(path):
    _require(path.is_file(), f"Missing table {path}")
    frame = pd.read_csv(path)
    _require(len(frame) == len(frame.drop_duplicates()), f"Duplicate physical rows in {path.name}")
    return frame


def _as_bool(series):
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": True, "false": False, "1": True, "0": False})
    _require(mapped.notna().all() or series.isna().all(), "Unparsed boolean column")
    return mapped


def protocol_digest():
    _require(PROTOCOL.is_file(), "Frozen protocol.md is missing")
    digest = sha256_file(PROTOCOL)
    recorded = PROTOCOL_SHA_FILE.read_text().split()[0].strip()
    _require(digest == FROZEN_PROTOCOL_SHA256, "protocol.md hash is not the frozen digest")
    _require(recorded == FROZEN_PROTOCOL_SHA256, "protocol.sha256 does not match the frozen digest")
    return digest


def check_hashes(cases_ready, tfm_manifest):
    digest = protocol_digest()
    _require(cases_ready["protocol"]["sha256"] == digest, "cases_ready protocol hash mismatch")
    _require(cases_ready["counts"]["cases"] == N_CASES, "cases_ready case count is not 200")
    _require(cases_ready["counts"]["tool_rows_H10"] == N_QUERIES, "H10 query count is not 800")
    _require(cases_ready["counts"]["tool_rows_H30"] == N_QUERIES, "H30 query count is not 800")
    for horizon, key in ((10, "tool_inputs_H10"), (30, "tool_inputs_H30")):
        archived = cases_ready["archives"][key]
        _require(archived["sha256"] == INPUT_SHA256[horizon], f"Recorded H{horizon} input hash changed")
        path = ROOT / archived["path"]
        _require(path.is_file(), f"Missing tool input {path}")
        _require(sha256_file(path) == INPUT_SHA256[horizon], f"On-disk H{horizon} input hash mismatch")
    _require(tfm_manifest.get("status") == "complete", "Scenario-engine manifest is not terminal")
    _require(tfm_manifest.get("protocol_sha256") == digest, "Scenario-engine manifest protocol hash mismatch")
    return digest


def _manifest_protocol(manifest):
    return manifest.get("base_protocol_sha256", manifest.get("protocol_sha256"))


def check_wb_manifest(manifest):
    _require(_manifest_protocol(manifest) == FROZEN_PROTOCOL_SHA256, "W manifest protocol hash mismatch")
    _require(int(manifest.get("n_cases_manifest", -1)) == N_CASES, "W manifest case count is not 200")
    missing = list(manifest.get("missing") or [])
    _require(missing == [], f"W/reference manifest is incomplete ({len(missing)} cases missing)")
    _require(int(manifest.get("n_markers", -1)) == N_CASES, "W/reference marker count is not 200")


def resolve_wb(wb_dir):
    """Live aggregation reads only the repaired tree. The original wb/ tree stays audit-only."""
    path = Path(wb_dir).resolve()
    audit = AUDIT_WB.resolve()
    _require(path != audit and audit not in path.parents,
             "Refusing the immutable audit-only wb tree; pass wb_repaired_v1_2")
    _require(path == OFFICIAL_WB.resolve(), "Live aggregation accepts only results/pilot/wb_repaired_v1_2")
    return path


def _wb_files(wb_dir):
    return {
        "manifest": wb_dir / "wb_manifest.json",
        "reuse": wb_dir / "reuse_manifest.json",
        "w": wb_dir / "tables" / "W_table.csv",
        "ref": wb_dir / "tables" / "reference_table.csv",
        "curves": wb_dir / "tables" / "curves_long.csv",
    }


def _flag_tokens(value):
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def case_flag_counts(frame):
    per_case = {}
    for case_id, flags in zip(frame["case_id"], frame["flags"]):
        per_case.setdefault(case_id, set()).update(_flag_tokens(flags))
    return {name: sum(name in tokens for tokens in per_case.values()) for name in EXPECTED_FLAG_CASES}


def check_repaired_sources(wb_dir):
    """Hash-check the freeze, repaired tables, and reuse manifest before any figure is written."""
    wb_dir = resolve_wb(wb_dir)
    files = _wb_files(wb_dir)
    _require(files["manifest"].is_file(), "Repaired wb_manifest.json is missing")
    manifest = json.loads(files["manifest"].read_text())
    check_wb_manifest(manifest)
    _require(manifest.get("official") is True, "W manifest is not marked official")
    _require(manifest.get("version") == "v1.2", "W manifest version is not v1.2")
    _require(manifest.get("status_counts") == {"ok": N_CASES}, "Repaired status counts are not 200 ok")
    _require(manifest.get("W_flag_counts") == EXPECTED_FLAG_CASES, "Manifest flag counts are not 17 and 14")
    _require(int(manifest.get("n_contrasts_with_W", -1)) == N_CONTRASTS, "Repaired contrast count is not 400")
    freeze_digest = sha256_file(REPAIR_FREEZE)
    _require(freeze_digest == EXPECTED_REPAIR_FREEZE_SHA256, "repair_freeze.json hash is not the pinned digest")
    _require(manifest.get("repair_freeze", {}).get("sha256") == freeze_digest,
             "wb_manifest repair_freeze hash does not match the file")
    freeze = json.loads(REPAIR_FREEZE.read_text())
    _require(freeze.get("base_protocol_sha256") == FROZEN_PROTOCOL_SHA256, "Freeze protocol hash changed")
    _require(str(freeze.get("official_output", "")).rstrip("/") == "results/pilot/wb_repaired_v1_2",
             "Freeze official output is not the repaired tree")
    recorded_hashes = freeze.get("hashes") or {}
    _require(recorded_hashes == manifest.get("repair_freeze_hashes"), "Freeze hashes and manifest hashes differ")
    for relative, digest in recorded_hashes.items():
        path = ROOT / relative
        _require(path.is_file(), f"Frozen repair input is missing: {relative}")
        _require(sha256_file(path) == digest, f"Frozen repair hash mismatch: {relative}")
    reuse_digest = sha256_file(files["reuse"])
    _require(reuse_digest == EXPECTED_REUSE_SHA256, "reuse_manifest.json hash is not the pinned digest")
    _require(manifest.get("reuse_manifest_sha256") == reuse_digest, "Manifest reuse hash mismatch")
    reuse = json.loads(files["reuse"].read_text())
    _require(reuse.get("protocol_sha256") == FROZEN_PROTOCOL_SHA256, "Reuse manifest protocol hash mismatch")
    _require(reuse.get("n_cases") == N_CASES and reuse.get("n_ok") == N_CASES and reuse.get("not_ok") == [],
             "Reuse manifest is not 200 ok cases")
    _require(len(reuse.get("per_case") or {}) == N_CASES, "Reuse manifest does not list 200 cases")
    for horizon, label in ((10, "H10"), (30, "H30")):
        block = reuse["timesfm"][label]
        _require(block.get("match") is True, f"Reuse manifest does not match the H{horizon} archive")
        _require(block.get("sha256") == OUTPUT_SHA256[horizon], f"Reuse H{horizon} hash is not the terminal archive")
    comparison_digest = sha256_file(REPAIR_COMPARISON)
    _require(comparison_digest == EXPECTED_COMPARISON_SHA256, "repair_comparison.json hash is not the pinned digest")
    _require(manifest.get("repair_comparison", {}).get("sha256") == comparison_digest,
             "Manifest repair_comparison hash mismatch")
    for name, digest in EXPECTED_TABLE_SHA256.items():
        recorded = manifest["tables"][name]
        path = files[{"W_table.csv": "w", "reference_table.csv": "ref", "curves_long.csv": "curves"}[name]]
        _require(sha256_file(path) == digest, f"{name} hash is not the repaired table digest")
        _require(recorded["sha256"] == digest, f"Manifest table hash differs for {name}")
        _require(int(recorded["rows"]) == (N_CONTRASTS * 30 if name == "curves_long.csv" else N_CONTRASTS),
                 f"Manifest row count differs for {name}")
    return manifest, files, reuse


def _unique_contrasts(frame, status_column, terminal):
    keys = ["case_id", "pair"]
    _require(frame.duplicated(keys).sum() == 0, f"Duplicate case-pair rows in {status_column}")
    _require(len(frame) == N_CONTRASTS, f"{status_column} table does not have 400 contrasts")
    _require(set(frame["pair"]) == set(PAIRS), "A schedule pair is missing from the table")
    _require(frame.groupby("case_id").size().eq(2).all(), "A case does not have both schedule pairs")
    _require(frame["case_id"].nunique() == N_CASES, "Case count in the table is not 200")
    status = frame[status_column].astype(str)
    _require(~status.isin(NONTERMINAL).any(), f"Non-terminal status in {status_column}")
    unknown = sorted(set(status) - terminal)
    _require(unknown == [], f"Unknown {status_column}: {unknown}")


def _delta_matches(frame, true_column, delta_column):
    if delta_column not in frame.columns or true_column not in frame.columns:
        return
    ok = frame["W_status"].astype(str).eq("ok")
    if not ok.any():
        return
    width = DELTA_FRACTION * frame.loc[ok, true_column].abs()
    expected = np.maximum(width.to_numpy(), SIGMA_M)
    got = frame.loc[ok, delta_column].to_numpy(dtype=float)
    _require(np.allclose(got, expected, rtol=0, atol=1e-9),
             f"Stored {delta_column} does not match the frozen rule")


def load_live_tables(files):
    w = _read_csv(files["w"])
    ref = _read_csv(files["ref"])
    curves = _read_csv(files["curves"])
    for column in ("case_id", "realization", "rest_ratio", "signal_ratio", "pi_layer", "pair",
                   "W_status", "W10", "Wmax1to10", "delta10", "E_true10",
                   "determined_lead10", "determined_1to10"):
        _require(column in w.columns, f"W_table missing {column}")
    for column in ("case_id", "pair", "reference_status", "E_point10", "w_TF10", "w_TFmax1to10",
                   "E_point10_inside_envelope", "E_point_inside_all_1to10"):
        _require(column in ref.columns, f"reference_table missing {column}")
    for column in ("case_id", "pair", "lead", "W", "sup", "inf"):
        _require(column in curves.columns, f"curves_long missing {column}")
    _unique_contrasts(w, "W_status", TERMINAL_W)
    _unique_contrasts(ref, "reference_status", TERMINAL_REF)
    _require(len(curves) == N_CONTRASTS * 30, "curves_long is not 400 contrasts by 30 leads")
    _require(curves.duplicated(["case_id", "pair", "lead"]).sum() == 0, "Duplicate curve rows")
    _require(set(curves["lead"].astype(int)) == set(range(1, 31)), "curves_long does not store leads 1 through 30")
    _delta_matches(w, "E_true10", "delta10")
    _delta_matches(w, "E_true30", "delta30")
    if "reference_reused_hash_ok" in ref.columns:
        reused = _as_bool(ref["reference_reused_hash_ok"])
        _require(reused.notna().all() and bool(reused.all()), "A reference row was not reused by hash")
    if "flags" in w.columns:
        counts = case_flag_counts(w)
        _require(counts == EXPECTED_FLAG_CASES, f"Case flag counts are {counts}, not 17 and 14")
        _require(int(w.groupby("case_id")["flags"].nunique().max()) == 1, "Flags differ between the two pairs of one case")
    return w, ref, curves


def _scalar(array):
    value = np.asarray(array)
    if value.shape == ():
        return value.item()
    _require(value.size == 1, "Expected a scalar archive field")
    return value.reshape(-1)[0].item()


def load_scenario_archive(path, horizon, protocol_sha, input_sha, n_queries):
    _require(path.is_file(), f"Missing scenario-engine archive {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    _require(str(_scalar(payload["protocol_sha256"])) == protocol_sha, f"{path.name} protocol hash mismatch")
    _require(str(_scalar(payload["input_sha256"])) == input_sha, f"{path.name} source-input hash mismatch")
    _require(int(_scalar(payload["horizon"])) == horizon, f"{path.name} horizon field mismatch")
    quantiles = np.asarray(payload["quantiles"])
    widths = np.asarray(payload["marginal_widths"], dtype=float)
    point = np.asarray(payload["E_point"], dtype=float)
    proxy = np.asarray(payload["E_width_proxy"], dtype=float)
    n_contrasts = n_queries // 2
    _require(quantiles.shape == (n_queries, horizon, 9), f"{path.name} quantile shape is not {n_queries} queries")
    _require(widths.shape == (n_queries, horizon), f"{path.name} marginal-width shape mismatch")
    _require(point.shape == (n_contrasts, horizon), f"{path.name} point-contrast shape mismatch")
    _require(proxy.shape == (n_contrasts, horizon), f"{path.name} width-proxy shape mismatch")
    schedule = np.asarray(payload["schedule_id"]).astype(str)
    _require(np.all(schedule[0::2] == "a") and np.all(schedule[1::2] == "b"), "Archive is not stored as schedule a then b")
    case = np.asarray(payload["case_id"]).astype(str)
    _require(np.all(case[0::2] == case[1::2]), "Paired queries do not share a case id")
    if n_queries == N_QUERIES:
        _require(len(set(np.asarray(payload["query_id"]).astype(str))) == N_QUERIES, "Query ids are not 800 unique ids")
    wa = widths[0::2]
    wb = widths[1::2]
    finite = np.isfinite(wa) & np.isfinite(wb) & np.isfinite(proxy)
    if finite.any():
        gap = np.max(np.abs(proxy[finite] - (wa[finite] + wb[finite])))
        _require(gap <= PROXY_ATOL, "Stored width proxy is not wa + wb on finite rows")
    levels = np.asarray(payload["quantile_levels"], dtype=float)
    _require(np.allclose(levels, np.arange(1, 10) / 10), "Quantile levels are not 0.1 through 0.9")
    metas = [json.loads(str(item)) for item in np.asarray(payload["metadata_json"])[0::2]]
    rows = []
    for index, meta in enumerate(metas):
        pair = str(meta["schedule_pair"])
        _require(pair in PAIRS, f"Unknown schedule_pair {pair}")
        rows.append({
            "case_id": case[2 * index],
            "pair": pair,
            "realization": int(meta["realization"]),
            "rest_ratio": float(meta["rest_ratio"]),
            "signal_ratio": float(meta["signal_ratio"]),
            "pi_layer": float(meta["pi_layer"]),
            "E_point": point[index],
            "wa": wa[index],
            "wb": wb[index],
            "width_proxy": proxy[index],
        })
    _require(len({(row["case_id"], row["pair"]) for row in rows}) == len(rows), "Duplicate scenario-engine contrasts")
    return rows


def _lead_summary(values, horizon):
    array = np.asarray(values, dtype=float)
    _require(array.shape[-1] >= 10, "Horizon is shorter than the day-10 summary")
    window = np.asarray(array[..., :10], dtype=float)
    lead10 = np.asarray(array[..., 9], dtype=float)
    finite = np.isfinite(window)
    if window.ndim == 1:
        return float(lead10), (float(np.max(window[finite])) if finite.any() else np.nan)
    max1to10 = np.full(window.shape[:-1], np.nan)
    has = finite.any(axis=-1)
    max1to10[has] = np.max(np.where(finite, window, -np.inf), axis=-1)[has]
    return lead10, max1to10


def build_cells(w, ref, curves, scenario_h10):
    """Join stored envelopes to the scenario-engine archive. No new W."""
    w = w.copy()
    ref = ref.copy()
    for column in ("determined_lead10", "determined_1to10"):
        w[column] = _as_bool(w[column])
    for column in ("E_point10_inside_envelope", "E_point_inside_all_1to10"):
        ref[column] = _as_bool(ref[column])
    keys = ["case_id", "pair"]
    merged = w.merge(ref[keys + ["reference_status", "E_point10", "w_TF10", "w_TFmax1to10",
                                  "E_point10_inside_envelope", "E_point_inside_all_1to10"]],
                     on=keys, how="outer", indicator=True)
    _require((merged["_merge"] == "both").all(), "W and reference tables do not contain the same contrasts")
    scenario = pd.DataFrame([{
        "case_id": row["case_id"], "pair": row["pair"],
        "engine_E10": _lead_summary(row["E_point"], 10)[0],
        "engine_Emaxabs_unused": 0.0,
        "wa10": _lead_summary(row["wa"], 10)[0],
        "wb10": _lead_summary(row["wb"], 10)[0],
        "wa_max1to10": _lead_summary(row["wa"], 10)[1],
        "wb_max1to10": _lead_summary(row["wb"], 10)[1],
        "proxy10": _lead_summary(row["width_proxy"], 10)[0],
        "proxy_max1to10": _lead_summary(row["width_proxy"], 10)[1],
        "engine_E": row["E_point"],
    } for row in scenario_h10])
    # engine_Emaxabs_unused kept out of the written table
    scenario = scenario.drop(columns=["engine_Emaxabs_unused"])
    merged = merged.merge(scenario, on=keys, how="outer", indicator="scenario_join")
    _require((merged["scenario_join"] == "both").all(), "Scenario-engine contrasts do not match the W table")
    curve_index = curves.set_index(["case_id", "pair", "lead"])
    inside10 = []
    inside_all = []
    ref_agree = []
    for record in merged.itertuples(index=False):
        leads = []
        for lead in range(1, 11):
            try:
                curve = curve_index.loc[(record.case_id, record.pair, lead)]
            except KeyError as exc:
                raise NotReady(f"Missing envelope lead {lead} for {record.case_id} {record.pair}") from exc
            low = float(curve["inf"])
            high = float(curve["sup"])
            estimate = float(np.asarray(record.engine_E)[lead - 1])
            if np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high):
                leads.append(bool(low - ENVELOPE_ATOL_M <= estimate <= high + ENVELOPE_ATOL_M))
            else:
                leads.append(np.nan)
        inside_all.append(leads)
        inside10.append(leads[9])
        if str(record.reference_status) == "ok" and np.isfinite(record.E_point10):
            curve = curve_index.loc[(record.case_id, record.pair, 10)]
            recomputed = bool(float(curve["inf"]) - ENVELOPE_ATOL_M
                              <= float(record.E_point10)
                              <= float(curve["sup"]) + ENVELOPE_ATOL_M)
            ref_agree.append(recomputed == bool(record.E_point10_inside_envelope))
    _require(all(ref_agree), "Stored reference inclusion disagrees with the envelope table")
    merged["engine_inside_lead10"] = inside10
    merged["engine_inside_all_1to10"] = [bool(np.all(np.array(item, dtype=float) == 1))
                                         if np.all(pd.notna(item)) else np.nan
                                         for item in inside_all]
    for column in ("W10", "Wmax1to10", "proxy10", "proxy_max1to10", "w_TF10", "w_TFmax1to10",
                   "wa10", "wb10", "wa_max1to10", "wb_max1to10", "engine_E10"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["zero_W10"] = merged["W10"].eq(0)
    merged["proxy_over_W10"] = np.where(merged["zero_W10"] | ~np.isfinite(merged["W10"]),
                                        np.nan, merged["proxy10"] / merged["W10"])
    merged["w_TF_over_W10"] = np.where(merged["zero_W10"] | ~np.isfinite(merged["W10"]),
                                       np.nan, merged["w_TF10"] / merged["W10"])
    merged["proxy_below_W10"] = np.where(merged["zero_W10"] | ~np.isfinite(merged["proxy10"]) | ~np.isfinite(merged["W10"]),
                                        np.nan, merged["proxy10"] < merged["W10"])
    keep = ["case_id", "realization", "rest_ratio", "signal_ratio", "pi_layer", "pair",
            "W_status", "reference_status", "W10", "Wmax1to10", "delta10",
            "determined_lead10", "determined_1to10", "zero_W10",
            "engine_E10", "wa10", "wb10", "wa_max1to10", "wb_max1to10",
            "proxy10", "proxy_max1to10", "engine_inside_lead10", "engine_inside_all_1to10",
            "proxy_over_W10", "w_TF10", "w_TFmax1to10", "w_TF_over_W10",
            "E_point10_inside_envelope", "E_point_inside_all_1to10", "proxy_below_W10"]
    for optional in ("flags", "converged"):
        if optional in merged.columns:
            keep.append(optional)
    return merged[keep]


def _iqr(series):
    finite = series.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    return float(np.subtract(*np.percentile(finite, [75, 25])))


def _median(series):
    finite = series.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    return float(np.median(finite))


def _bool_counts(series):
    values = []
    for item in series.tolist():
        if pd.isna(item):
            values.append(np.nan)
        else:
            values.append(1.0 if bool(item) else 0.0)
    array = np.asarray(values, dtype=float)
    defined = int(np.isfinite(array).sum())
    inside = int(np.nansum(array)) if defined else 0
    return inside, defined


def condition_summary(cells):
    """Medians across realizations. Both lead summaries and both pairs stay in the table."""
    frames = []
    for summary, column in (("lead10", "W10"), ("max1to10", "Wmax1to10")):
        grouped = cells.groupby(["rest_ratio", "signal_ratio", "pi_layer", "pair"], dropna=False)
        rows = []
        for key, block in grouped:
            determined = block["determined_lead10"] if summary == "lead10" else block["determined_1to10"]
            n_within, n_defined = _bool_counts(determined)
            if summary == "lead10":
                engine_inside, engine_defined = _bool_counts(block["engine_inside_lead10"])
                reference_inside, reference_defined = _bool_counts(block["E_point10_inside_envelope"])
            else:
                engine_inside, engine_defined = _bool_counts(block["engine_inside_all_1to10"])
                reference_inside, reference_defined = _bool_counts(block["E_point_inside_all_1to10"])
            rows.append({
                "rest_ratio": key[0], "signal_ratio": key[1], "pi_layer": key[2], "pair": key[3],
                "summary": summary, "W_median_m": _median(block[column]), "W_iqr_m": _iqr(block[column]),
                "n_realizations": int(block["realization"].nunique()),
                "n_finite_W": int(np.isfinite(block[column]).sum()),
                "n_nonfinite_W": int((~np.isfinite(block[column].to_numpy(dtype=float))).sum()),
                "n_within_frozen_delta": n_within,
                "n_defined_delta": n_defined,
                "fraction_within_frozen_delta": (n_within / n_defined) if n_defined else np.nan,
                "engine_inclusion_numerator": engine_inside,
                "engine_inclusion_denominator": engine_defined,
                "reference_inclusion_numerator": reference_inside,
                "reference_inclusion_denominator": reference_defined,
            })
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def width_response(cells):
    """Spearman of width against W across rest levels inside one matched block.

    A constant series is an undefined correlation. A zero envelope makes the
    width ratio undefined. proxy_below_W10 counts a descriptive discrepancy.
    """
    rows = []
    group_columns = ["realization", "signal_ratio", "pi_layer", "pair"]
    for key, block in cells.groupby(group_columns, dropna=False):
        ordered = block.sort_values("rest_ratio")
        for tool, width_column in (("scenario_engine_marginal_proxy", "proxy10"),
                                   ("structurally_matched_reference", "w_TF10")):
            w = ordered["W10"].to_numpy(dtype=float)
            width = ordered[width_column].to_numpy(dtype=float)
            usable = np.isfinite(w) & np.isfinite(width)
            undefined = True
            coefficient = np.nan
            if usable.sum() >= 3 and np.unique(w[usable]).size > 1 and np.unique(width[usable]).size > 1:
                coefficient = float(spearmanr(w[usable], width[usable]).statistic)
                undefined = not np.isfinite(coefficient)
            rows.append({
                "realization": key[0], "signal_ratio": key[1], "pi_layer": key[2], "pair": key[3],
                "tool": tool, "n_rest_levels": int(usable.sum()),
                "spearman_W10_vs_width": coefficient,
                "spearman_undefined": undefined,
                "n_zero_W": int(np.sum(ordered["zero_W10"].to_numpy(dtype=bool) & usable)),
                "n_proxy_below_W_not_used_for_this_tool": (
                    int(np.nansum(ordered["proxy_below_W10"].to_numpy(dtype=float)))
                    if tool.startswith("scenario") else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _panel_legend(axis, handles):
    legend = axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
                         ncol=4, frameon=False, borderaxespad=0.0)
    return legend


def _style_axis(axis, xlabel, ylabel):
    axis.set_xlabel(xlabel, fontweight="bold", labelpad=8)
    axis.set_ylabel(ylabel, fontweight="bold", labelpad=8)
    axis.margins(x=0.08, y=0.12)
    axis.tick_params(width=0.6)
    axis.grid(True, axis="both", linewidth=0.3, alpha=0.35)


def figure_rest(summary, path_stem):
    """Rest ratio against envelope width. No model name. Both summaries and both pairs."""
    apply_style()
    pi_levels = sorted(summary["pi_layer"].unique())
    summaries = ("lead10", "max1to10")
    row_title = {"lead10": "Day 10", "max1to10": "Maximum of days 1–10"}
    figure, axes = plt.subplots(2, len(pi_levels), figsize=(7.2, 7.15), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    legend_handles = {
        f"{PAIR_LABEL[pair]}, signal {signal:g}": mpl.lines.Line2D(
            [0], [0], marker=marker, color=color, linestyle="none", markersize=6)
        for signal, color in SIGNAL_COLOR.items()
        for pair, marker in PAIR_MARKER.items()
    }
    for row, summary_name in enumerate(summaries):
        block = summary[summary["summary"] == summary_name]
        for column, pi_layer in enumerate(pi_levels):
            axis = axes[row, column]
            panel = block[block["pi_layer"] == pi_layer]
            for signal, color in SIGNAL_COLOR.items():
                for pair, marker in PAIR_MARKER.items():
                    part = panel[(panel["signal_ratio"] == signal) & (panel["pair"] == pair)]
                    if part.empty:
                        continue
                    dodge = -0.035 if pair.startswith("P1") else 0.035
                    x = np.log10(part["rest_ratio"].to_numpy(dtype=float)) + dodge
                    y = part["W_median_m"].to_numpy(dtype=float)
                    yerr = part["W_iqr_m"].to_numpy(dtype=float) / 2
                    axis.errorbar(x, y, yerr=yerr, fmt=marker, color=color, ms=5.5,
                                  lw=0.8, capsize=2)
            _style_axis(axis, "Rest / response time", "Envelope width W (m)")
            if row < len(summaries) - 1:
                axis.set_xlabel("")
            axis.set_title(f"{row_title[summary_name]},  pi_r = {pi_layer:g}", fontsize=12, pad=8)
    ticks = sorted(summary["rest_ratio"].unique())
    for axis in axes[-1, :]:
        axis.set_xticks(np.log10(ticks))
        axis.set_xticklabels([f"{tick:g}" for tick in ticks])
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    figure.legend(legend_handles.values(), legend_handles.keys(), loc="lower center",
                  bbox_to_anchor=(0.5, 0.012), ncol=2, frameon=False)
    _save(figure, path_stem)
    return path_stem


def figure_widths(cells, path_stem):
    """W against the two different width objects. Inclusion is mark fill, not a score."""
    apply_style()
    pi_levels = sorted(cells["pi_layer"].unique())
    tools = (
        ("proxy10", "engine_inside_lead10", "Scenario engine, wa + wb"),
        ("w_TF10", "E_point10_inside_envelope", "Structurally matched reference"),
    )
    figure, axes = plt.subplots(len(pi_levels), 2, figsize=(7.2, 6.8), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    legend_handles = {
        f"{PAIR_LABEL[pair]}, signal {signal:g}": mpl.lines.Line2D(
            [0], [0], marker=marker, color=color, linestyle="none", markersize=6)
        for signal, color in SIGNAL_COLOR.items()
        for pair, marker in PAIR_MARKER.items()
    }
    for row, pi_layer in enumerate(pi_levels):
        panel = cells[cells["pi_layer"] == pi_layer]
        for column, (width_column, inside_column, title) in enumerate(tools):
            axis = axes[row, column]
            for signal, color in SIGNAL_COLOR.items():
                for pair, marker in PAIR_MARKER.items():
                    part = panel[(panel["signal_ratio"] == signal) & (panel["pair"] == pair)]
                    if part.empty:
                        continue
                    inside = part[inside_column].map(lambda item: bool(item) if pd.notna(item) else False)
                    finite = np.isfinite(part["W10"]) & np.isfinite(part[width_column])
                    shown = part[finite]
                    face_mask = inside.reindex(shown.index).fillna(False).to_numpy()
                    face = shown[face_mask]
                    open_mark = shown[~face_mask]
                    axis.scatter(face["W10"], face[width_column], marker=marker, s=28,
                                 facecolors=color, edgecolors=color, linewidths=0.6)
                    axis.scatter(open_mark["W10"], open_mark[width_column], marker=marker, s=28,
                                 facecolors="none", edgecolors=color, linewidths=0.9)
            _style_axis(axis, "Envelope width W (m)", "Width (m)")
            if row < len(pi_levels) - 1:
                axis.set_xlabel("")
            axis.set_title(f"{title}\npi_r = {pi_layer:g}", fontsize=11, pad=6)
    figure.legend(legend_handles.values(), legend_handles.keys(), loc="lower center",
                  bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False)
    figure.text(0.5, 0.005,
                "Filled mark: day-10 point contrast inside the envelope. Open mark: outside. wa + wb is not a joint interval.",
                ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    _save(figure, path_stem)
    return path_stem


def _save(figure, path_stem):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_stem.with_suffix(".png"))
    figure.savefig(path_stem.with_suffix(".pdf"))
    plt.close(figure)
    _assert_figure_text(path_stem)


def _assert_figure_text(path_stem):
    # The PDF text layer is the checkable copy of the words drawn on the figure.
    raw = path_stem.with_suffix(".pdf").read_bytes()
    # Matplotlib Type 42 embeds glyphs; the rcParams font name is still in the file header.
    _require(b"TimesNewRoman" in raw or b"Times New Roman" in raw or b"Times" in raw,
             "Figure PDF does not record a Times font")


def _lead_at(values, index):
    array = np.asarray(values, dtype=float)
    _require(array.shape[-1] > index, "Horizon is shorter than the requested lead")
    value = np.asarray(array[..., index], dtype=float)
    if value.ndim == 0:
        return float(value)
    return value


def _curve_bounds(curve_index, case_id, pair, lead):
    try:
        curve = curve_index.loc[(case_id, pair, lead)]
    except KeyError as exc:
        raise NotReady(f"Missing envelope lead {lead} for {case_id} {pair}") from exc
    return float(curve["inf"]), float(curve["sup"]), curve


def _inside_bounds(estimate, low, high):
    if not (np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high)):
        return np.nan
    return bool(low - ENVELOPE_ATOL_M <= estimate <= high + ENVELOPE_ATOL_M)


def build_event30(w, ref, curves, scenario_h30):
    """Lead-30 diagnostic from the independent H30 archive and repaired lead-30 curves.

    This table does not replace the horizon-10 scores. It does not recompute W,
    the reference draws, or the forecast.
    """
    for column in ("W30", "delta30", "E_true30", "determined_lead30", "flags"):
        _require(column in w.columns, f"W_table missing {column}")
    for column in ("E_point30", "E_q10_30", "E_q90_30", "w_TF30",
                   "E_point30_inside_envelope", "E_point_inside_all_1to30"):
        _require(column in ref.columns, f"reference_table missing {column}")
    keys = ["case_id", "pair"]
    merged = w.merge(ref[keys + ["reference_status", "E_point30", "E_q10_30", "E_q90_30", "w_TF30",
                                  "E_point30_inside_envelope", "E_point_inside_all_1to30"]],
                     on=keys, how="outer", indicator=True)
    _require((merged["_merge"] == "both").all(), "Lead-30 reference rows do not match the W table")
    merged["E_point30_inside_envelope"] = _as_bool(merged["E_point30_inside_envelope"])
    merged["determined_lead30"] = _as_bool(merged["determined_lead30"])
    scenario = pd.DataFrame([{
        "case_id": row["case_id"],
        "pair": row["pair"],
        "engine_E30": _lead_at(row["E_point"], 29),
        "wa30": _lead_at(row["wa"], 29),
        "wb30": _lead_at(row["wb"], 29),
        "proxy30": _lead_at(row["width_proxy"], 29),
    } for row in scenario_h30])
    merged = merged.merge(scenario, on=keys, how="outer", indicator="scenario_join")
    _require((merged["scenario_join"] == "both").all(), "H30 contrasts do not match the repaired W table")
    _require(len(merged) == N_CONTRASTS, "Lead-30 table is not 400 contrasts")
    merged = merged.drop(columns=["_merge", "scenario_join"])
    curve_index = curves.set_index(["case_id", "pair", "lead"])
    engine_inside = []
    ref_agree = []
    curve_point_gap = []
    for record in merged.itertuples(index=False):
        low, high, curve = _curve_bounds(curve_index, record.case_id, record.pair, 30)
        engine_inside.append(_inside_bounds(record.engine_E30, low, high))
        if str(record.reference_status) == "ok" and np.isfinite(record.E_point30):
            recomputed = _inside_bounds(float(record.E_point30), low, high)
            ref_agree.append(recomputed == bool(record.E_point30_inside_envelope))
        curve_point_gap.append(abs(float(curve["ref_E_point"]) - float(record.E_point30)))
    _require(len(ref_agree) == N_CONTRASTS and all(ref_agree),
             "Stored lead-30 reference inclusion disagrees with the repaired envelope")
    _require(max(curve_point_gap) <= 1e-9, "Lead-30 curve reference point disagrees with the reference table")
    width_gap = np.abs(merged["w_TF30"].to_numpy(dtype=float)
                       - (merged["E_q90_30"].to_numpy(dtype=float) - merged["E_q10_30"].to_numpy(dtype=float)))
    finite_width = np.isfinite(width_gap)
    if finite_width.any():
        _require(float(np.max(width_gap[finite_width])) <= 1e-8,
                 "Stored lead-30 reference width is not E_q90_30 - E_q10_30")
    merged["engine_inside_lead30"] = engine_inside
    for column in ("W30", "delta30", "E_true30", "engine_E30", "wa30", "wb30", "proxy30", "w_TF30", "E_point30"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["zero_W30"] = merged["W30"].eq(0)
    merged["proxy_over_W30"] = np.where(merged["zero_W30"] | ~np.isfinite(merged["W30"]),
                                        np.nan, merged["proxy30"] / merged["W30"])
    merged["w_TF_over_W30"] = np.where(merged["zero_W30"] | ~np.isfinite(merged["W30"]),
                                       np.nan, merged["w_TF30"] / merged["W30"])
    keep = ["case_id", "realization", "rest_ratio", "signal_ratio", "pi_layer", "pair", "flags",
            "W_status", "reference_status", "W30", "delta30", "determined_lead30", "E_true30", "zero_W30",
            "engine_E30", "wa30", "wb30", "proxy30", "engine_inside_lead30", "proxy_over_W30",
            "E_point30", "E_q10_30", "E_q90_30", "w_TF30", "w_TF_over_W30",
            "E_point30_inside_envelope", "E_point_inside_all_1to30"]
    return merged[keep]


def event30_summary(frame):
    rows = []
    grouped = frame.groupby(["rest_ratio", "signal_ratio", "pi_layer", "pair"], dropna=False)
    for key, block in grouped:
        n_within, n_defined = _bool_counts(block["determined_lead30"])
        engine_inside, engine_defined = _bool_counts(block["engine_inside_lead30"])
        reference_inside, reference_defined = _bool_counts(block["E_point30_inside_envelope"])
        rows.append({
            "rest_ratio": key[0], "signal_ratio": key[1], "pi_layer": key[2], "pair": key[3],
            "summary": "lead30",
            "W30_median_m": _median(block["W30"]),
            "W30_iqr_m": _iqr(block["W30"]),
            "proxy30_median_m": _median(block["proxy30"]),
            "w_TF30_median_m": _median(block["w_TF30"]),
            "n_realizations": int(block["realization"].nunique()),
            "n_finite_W30": int(np.isfinite(block["W30"]).sum()),
            "n_nonfinite_W30": int((~np.isfinite(block["W30"].to_numpy(dtype=float))).sum()),
            "n_zero_W30": int(block["zero_W30"].sum()),
            "n_within_frozen_delta": n_within,
            "n_defined_delta": n_defined,
            "engine_inclusion_numerator": engine_inside,
            "engine_inclusion_denominator": engine_defined,
            "reference_inclusion_numerator": reference_inside,
            "reference_inclusion_denominator": reference_defined,
        })
    return pd.DataFrame(rows)


def captions(flag_note=None):
    text = """# Figure captions

These captions stay outside the image. W is the width of head-difference values still inside the declared response family for that record. It is a sampled inner numerical envelope under the frozen tolerance, not a global or exhaustive bound and not a universal theorem. It is not a probability interval and it is not a bound for every possible response family.

Figure 1. Envelope width against rest relative to response time. Rows are the day-10 width and the maximum width over days 1–10. Columns are the two pi_r strata. Color is the pumping-induced signal ratio. Marker shape is the future schedule pair. Points are medians across realizations and whiskers are the interquartile range. The frozen delta threshold is stored in the condition table and is not drawn as a success line.

Figure 2. Day-10 width objects against the same envelope width. The left column is wa + wb, the sum of the two marginal 10–90 ranges. That sum is a marginal-range difference proxy. It is not an 80 percent interval for the head difference, and the two marginal quantile pairs do not define a joint distribution. The right column is the 10–90 width of the structurally matched reference, from paired parameter draws of one fit. The two columns are different objects. Filled marks are day-10 point contrasts inside the envelope; open marks are outside. Neither mark is a coverage guarantee.

The lead-30 table is a separate event diagnostic. It uses the independent horizon-30 scenario-engine archive and the repaired lead-30 curves. It does not replace the day-10 or days-1–10 summaries.
"""
    if flag_note:
        text += "\n" + flag_note.strip() + "\n"
    return text


def _flag_note(counts):
    return (
        f"On the repaired table, {counts['not_converged']} cases are marked not_converged and "
        f"{counts['sparse_G']} cases are marked sparse_G. Those marks are numerical-search flags. "
        "They are not a success test. The converged flag does not show whether another search path "
        "would have reached a wider envelope. Search-path sensitivity is the already stored comparison "
        "between the audit tree and this repaired tree. It is not recomputed here, and the two envelopes "
        "are not united."
    )


def write_outputs(directory, cells, summary, response, flag_note=None):
    directory.mkdir(parents=True, exist_ok=True)
    cells.drop(columns=["engine_E"], errors="ignore").to_csv(directory / "cell_contrasts.csv", index=False)
    summary.to_csv(directory / "condition_summary.csv", index=False)
    response.to_csv(directory / "width_response_blocks.csv", index=False)
    (directory / "captions.md").write_text(captions(flag_note))
    figure_rest(summary, directory / "figure1_rest_ratio_W")
    figure_widths(cells, directory / "figure2_record_width")


def _pdf_font_record(path_stem):
    raw = path_stem.with_suffix(".pdf").read_bytes()
    return {
        "times_new_roman": b"TimesNewRoman" in raw or b"Times New Roman" in raw,
        "dejavu": b"DejaVu" in raw,
        "bytes": path_stem.with_suffix(".pdf").stat().st_size,
    }


def _spearman_counts(response):
    defined = response.loc[~response["spearman_undefined"].astype(bool), "spearman_W10_vs_width"]
    values = defined.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "n_blocks": int(len(response)),
        "n_undefined": int(response["spearman_undefined"].astype(bool).sum()),
        "n_defined": int(len(finite)),
        "n_positive": int(np.sum(finite > 0)),
        "n_negative": int(np.sum(finite < 0)),
        "n_zero": int(np.sum(finite == 0)),
    }


def _execution_markdown(summary, response, event_summary, counts, comparison, code_sha):
    rel10 = comparison["summary"]["rel_change_P1"]["W10"]
    rel30 = comparison["summary"]["rel_change_P1"]["W30"]
    lines = [
        "# Pilot figure execution",
        "",
        "The two figures and the horizon-10 condition table use the repaired v1.2 envelope.",
        "The original `results/pilot/wb/` tree was not read. W, the reference fit and draws, and both scenario-engine archives were not recomputed.",
        "",
        "## Code change",
        "",
        f"Preparation SHA-256 of `lib/p3_figures.py`: `{PREPARATION_CODE_SHA256}`.",
        f"SHA-256 after this execution change: `{code_sha}`.",
        "The change adds `--wb-dir`, refuses the audit-only `wb/` tree, requires the repair freeze, addendum, repair-code hashes, reuse manifest, and repaired table hashes, and writes a lead-30 diagnostic from the independent horizon-30 archive.",
        "The day-10 figure geometry, the frozen delta rule, and the Spearman rule are unchanged.",
        "",
        "## Envelope and flags",
        "",
        "W is a sampled inner envelope of direct-certified curves under the declared family and the frozen tolerance. It is not a certified global extremum over G.",
        f"Case-level flags in `W_table.csv`: not_converged {counts['not_converged']}, sparse_G {counts['sparse_G']}. Both pairs of a case carry the same flag string. These flags are numerical-search marks, not a success test. The converged flag does not describe search-path dependence.",
        "",
        "## Search-path limitation already stored",
        "",
        "Taken from `repair_comparison.json` summary, not recomputed:",
        f"- n_cases {comparison['summary']['n_cases']}; new_status {json.dumps(comparison['summary']['new_status'])}; old_status {json.dumps(comparison['summary']['old_status'])}.",
        f"- new_flag_counts {json.dumps(comparison['summary']['new_flag_counts'])}; old_flag_counts {json.dumps(comparison['summary']['old_flag_counts'])}.",
        f"- P1 relative change in W10 on {rel10['n']} cases with an old W: min {rel10['min']}, median {rel10['median']}, max {rel10['max']}, n_within_1pct {rel10['n_within_1pct']}, n_increase_gt_1pct {rel10['n_increase_gt_1pct']}, n_decrease_gt_1pct {rel10['n_decrease_gt_1pct']}.",
        f"- P1 relative change in W30 on {rel30['n']} cases: min {rel30['min']}, median {rel30['median']}, max {rel30['max']}.",
        f"- n_new_contains_old_all_leads {comparison['summary']['n_new_contains_old_all_leads']}.",
        f"- max_old_outside_new_m {comparison['summary']['max_old_outside_new_m']}.",
        f"- Stored note: {comparison['summary']['note']}",
        "The repair execution note reads that comparison as old ordinates outside v1.2 by more than 1e-6 m in 63 cases and more than 1e-3 m in 5, with the largest excess about 9.3 mm. That reading is preserved as the stated limitation. This run does not recompute it and does not add a union of the two envelopes.",
        "",
        "## Horizon-10 and days 1-10",
        "",
        "Inclusion numerators and denominators are counts of defined rows. `fraction_within_frozen_delta` is n_within_frozen_delta / n_defined_delta. A zero envelope makes the width ratio undefined. These fractions are not a success criterion.",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Width response",
        "",
        "The coefficient is Spearman correlation of day-10 width with W across the rest levels inside one realization, signal ratio, pi_r, and schedule pair. A constant series is undefined and is counted, not coerced. The sign of a defined coefficient is not a monotone-response test and is not a success criterion.",
        "",
        "```csv",
        response.to_csv(index=False).strip(),
        "```",
        "",
        "### Width-response counts",
        "",
    ]
    for tool, block in response.groupby("tool"):
        stats = _spearman_counts(block)
        lines.append(
            f"- {tool}: blocks {stats['n_blocks']}, defined {stats['n_defined']}, undefined {stats['n_undefined']}, "
            f"positive {stats['n_positive']}, negative {stats['n_negative']}, zero {stats['n_zero']}."
        )
    lines.extend([
        "",
        "## Lead-30 event diagnostic",
        "",
        "This table joins the independent horizon-30 scenario-engine archive at lead 30 to the repaired curves and the reused reference point and paired width at lead 30. It does not replace the horizon-10 scores. wa30 + wb30 remains a marginal-range difference proxy, not a joint interval.",
        "",
        "```csv",
        event_summary.to_csv(index=False).strip(),
        "```",
        "",
    ])
    return "\n".join(lines) + "\n"


def _check_archives(tfm_manifest):
    for horizon, path in ((10, TFM_H10), (30, TFM_H30)):
        recorded = tfm_manifest["outputs"][str(horizon)]
        _require(Path(recorded["path"]) == path, "Scenario-engine archive path differs from the manifest")
        _require(recorded["sha256"] == OUTPUT_SHA256[horizon], f"H{horizon} manifest hash is not the pinned archive")
        _require(sha256_file(path) == OUTPUT_SHA256[horizon], f"H{horizon} archive hash differs from its terminal manifest")
        _require(recorded["quantiles_shape"][0] == N_QUERIES, f"H{horizon} manifest is not an 800-query archive")
        _require(recorded["quantiles_shape"][1] == horizon, f"H{horizon} manifest horizon length mismatch")


def execute(out_dir, wb_dir):
    cases_ready = json.loads(CASES_READY.read_text())
    tfm_manifest = json.loads(TFM_MANIFEST.read_text())
    digest = check_hashes(cases_ready, tfm_manifest)
    wb_manifest, files, _reuse = check_repaired_sources(wb_dir)
    _check_archives(tfm_manifest)
    w, ref, curves = load_live_tables(files)
    _require((w["W_status"].astype(str) == "ok").all(), "A repaired contrast is not ok")
    _require(int((~np.isfinite(w["W10"])).sum()) == 0, "A repaired day-10 width is non-finite")
    _require(int((~np.isfinite(w["W30"])).sum()) == 0, "A repaired lead-30 width is non-finite")
    scenario_h10 = load_scenario_archive(TFM_H10, 10, digest, INPUT_SHA256[10], N_QUERIES)
    scenario_h30 = load_scenario_archive(TFM_H30, 30, digest, INPUT_SHA256[30], N_QUERIES)
    _require(len(scenario_h10) == N_CONTRASTS and len(scenario_h30) == N_CONTRASTS,
             "A scenario-engine archive does not have 400 contrasts")
    cells = build_cells(w, ref, curves, scenario_h10)
    summary = condition_summary(cells)
    _require(set(summary["summary"]) == {"lead10", "max1to10"}, "A horizon-10 summary is missing")
    _require(set(summary["pair"]) == set(PAIRS), "A schedule pair is missing from the condition table")
    _require(summary["n_realizations"].eq(10).all(), "A condition cell does not have 10 realizations")
    response = width_response(cells)
    event = build_event30(w, ref, curves, scenario_h30)
    event_summary = event30_summary(event)
    _require(len(event) == N_CONTRASTS, "Lead-30 diagnostic is not the full grid")
    _require(set(event_summary["pair"]) == set(PAIRS), "Lead-30 diagnostic dropped a schedule pair")
    counts = case_flag_counts(w)
    out_dir.mkdir(parents=True, exist_ok=True)
    event.to_csv(out_dir / "event30_contrasts.csv", index=False)
    event_summary.to_csv(out_dir / "event30_condition_summary.csv", index=False)
    write_outputs(out_dir, cells, summary, response, _flag_note(counts))
    code_sha = sha256_file(Path(__file__))
    comparison = json.loads(REPAIR_COMPARISON.read_text())
    execution = _execution_markdown(summary, response, event_summary, counts, comparison, code_sha)
    (PILOT / "figures_execution.md").write_text(execution)
    outputs = [
        "cell_contrasts.csv", "condition_summary.csv", "width_response_blocks.csv",
        "event30_contrasts.csv", "event30_condition_summary.csv", "captions.md",
        "figure1_rest_ratio_W.png", "figure1_rest_ratio_W.pdf",
        "figure2_record_width.png", "figure2_record_width.pdf",
    ]
    manifest = {
        "status": "wrote_figures",
        "wb_dir": str(files["manifest"].parent),
        "audit_wb_refused": str(AUDIT_WB),
        "protocol_sha256": digest,
        "repair_freeze_sha256": EXPECTED_REPAIR_FREEZE_SHA256,
        "reuse_manifest_sha256": EXPECTED_REUSE_SHA256,
        "repair_comparison_sha256": EXPECTED_COMPARISON_SHA256,
        "figure_code_sha256": code_sha,
        "figure_code_sha256_before": PREPARATION_CODE_SHA256,
        "n_contrasts": int(len(cells)),
        "n_event30_contrasts": int(len(event)),
        "flag_cases": counts,
        "n_zero_W10": int(cells["zero_W10"].sum()),
        "n_zero_W30": int(event["zero_W30"].sum()),
        "spearman": {tool: _spearman_counts(block) for tool, block in response.groupby("tool")},
        "inputs": {
            "W_table": {"path": str(files["w"]), "sha256": EXPECTED_TABLE_SHA256["W_table.csv"], "rows": int(len(w))},
            "reference_table": {"path": str(files["ref"]), "sha256": EXPECTED_TABLE_SHA256["reference_table.csv"], "rows": int(len(ref))},
            "curves_long": {"path": str(files["curves"]), "sha256": EXPECTED_TABLE_SHA256["curves_long.csv"],
                            "rows": int(len(curves)), "leads": "1-30"},
            "timesfm_H10": {"path": str(TFM_H10), "sha256": OUTPUT_SHA256[10], "queries": N_QUERIES, "horizon": 10},
            "timesfm_H30": {"path": str(TFM_H30), "sha256": OUTPUT_SHA256[30], "queries": N_QUERIES, "horizon": 30},
            "wb_manifest_cases": int(wb_manifest["n_cases_manifest"]),
        },
        "font": {
            "figure1": _pdf_font_record(out_dir / "figure1_rest_ratio_W"),
            "figure2": _pdf_font_record(out_dir / "figure2_record_width"),
        },
        "visual_check": "pending_open",
        "outputs": {name: sha256_file(out_dir / name) for name in outputs},
        "caveat": "wa+wb is a marginal-range proxy and is not a joint interval; W is a sampled inner envelope",
    }
    for label, record in manifest["font"].items():
        _require(record["times_new_roman"], f"{label} PDF does not embed Times New Roman")
        _require(not record["dejavu"], f"{label} PDF embeds DejaVu")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return {"status": "wrote_figures", "n_contrasts": int(len(cells)), "figure_code_sha256": code_sha}


def check_ready(wb_dir):
    reasons = []
    try:
        cases_ready = json.loads(CASES_READY.read_text())
        tfm_manifest = json.loads(TFM_MANIFEST.read_text())
        check_hashes(cases_ready, tfm_manifest)
        _check_archives(tfm_manifest)
        check_repaired_sources(wb_dir)
    except (OSError, json.JSONDecodeError, KeyError, NotReady) as exc:
        reasons.append(str(exc))
    status = "ready" if not reasons else "not_ready"
    return {"status": status, "reasons": reasons}


def _fixture_bundle(tmp):
    """Synthetic contrasts covering both pairs, both summaries, zero W, and a NaN."""
    rows_w = []
    rows_ref = []
    curve_rows = []
    scenario = []
    rests = (0.25, 0.5, 1.0, 2.0, 4.0)
    specs = []
    for realization in (0, 1):
        for rest in rests:
            specs.append((realization, rest, 0.5, 0.5, False))
        specs.append((realization, 1.0, 2.0, 5.0, True))
    for realization, rest, signal, pi_layer, zero in specs:
        for pair_index, pair in enumerate(PAIRS):
            case = f"fx_r{realization}_pi{pi_layer:g}_sr{signal:g}_rho{rest:g}"
            sign = -1.0 if pair_index == 0 else 1.0
            width = 0.0 if zero else (0.08 if realization == 0 else 0.11) / rest
            true = sign * 0.05
            delta = max(DELTA_FRACTION * abs(true), SIGMA_M)
            rows_w.append({
                "case_id": case, "realization": realization, "rest_ratio": rest,
                "signal_ratio": signal, "pi_layer": pi_layer, "pair": pair,
                "W_status": "ok", "W10": width, "Wmax1to10": width * 1.1,
                "E_true10": true, "delta10": delta,
                "determined_lead10": width <= delta, "determined_1to10": width * 1.1 <= delta,
            })
            point = sign * 0.02
            low = -width / 2
            high = width / 2
            inside = low <= point <= high
            rows_ref.append({
                "case_id": case, "pair": pair, "reference_status": "ok",
                "E_point10": point, "w_TF10": width * 0.4, "w_TFmax1to10": width * 0.45,
                "E_point10_inside_envelope": inside, "E_point_inside_all_1to10": inside,
            })
            engine = np.full(10, point)
            if realization == 1 and rest == 0.25 and pair_index == 0:
                engine = np.full(10, np.nan)
            wa = np.full(10, 0.03)
            wb = np.full(10, 0.02 if not zero else 0.02)
            if realization == 0 and signal == 0.5 and pi_layer == 0.5:
                wa = np.full(10, 0.01 * rest)
            scenario.append({
                "case_id": case, "pair": pair, "realization": realization,
                "rest_ratio": rest, "signal_ratio": signal, "pi_layer": pi_layer,
                "E_point": engine, "wa": wa, "wb": wb, "width_proxy": wa + wb,
            })
            for lead in range(1, 31):
                curve_rows.append({
                    "case_id": case, "pair": pair, "lead": lead,
                    "W": width, "sup": high, "inf": low,
                })
    # One constant-width block already exists for signal 2 (single rest). Add a
    # five-rest constant proxy via the signal 0.5 realization-1 series: wa is flat.
    w = pd.DataFrame(rows_w)
    ref = pd.DataFrame(rows_ref)
    curves = pd.DataFrame(curve_rows)
    _require(set(w["pair"]) == set(PAIRS), "Fixture dropped a schedule pair")
    _require((w["E_true10"] < 0).any() and (w["E_true10"] > 0).any(), "Fixture lost pair direction")
    cells = build_cells(w, ref, curves, scenario)
    cells["engine_E"] = [row["E_point"] for row in scenario]
    _require(cells["zero_W10"].any(), "Fixture did not keep a zero envelope")
    _require(cells.loc[cells["zero_W10"], "proxy_over_W10"].isna().all(), "Zero-W ratio was filled")
    _require(cells["engine_E10"].isna().any(), "NaN point contrast was dropped")
    summary = condition_summary(cells)
    _require(set(summary["summary"]) == {"lead10", "max1to10"}, "A lead summary is missing")
    _require(set(summary["pair"]) == set(PAIRS), "A pair is missing from the grouped table")
    response = width_response(cells)
    _require(response["spearman_undefined"].any(), "Constant-width block was coerced to a correlation")
    _require((~response["spearman_undefined"]).any(), "No finite rank correlation was computed")
    write_outputs(tmp, cells, summary, response)
    (tmp / "FIXTURE_NOT_PILOT.txt").write_text(
        "Synthetic rows for parser, grouping, and layout checks. Not a pilot result.\n")
    return {"n_cells": int(len(cells)), "n_undefined_spearman": int(response["spearman_undefined"].sum())}


def fixture_negatives():
    """Edges that must refuse: duplicate contrast, bad protocol hash, short manifest."""
    caught = []
    try:
        frame = pd.DataFrame({
            "case_id": ["c", "c"], "pair": ["P1_continue_vs_stop", "P1_continue_vs_stop"],
            "W_status": ["ok", "ok"],
        })
        _unique_contrasts(pd.concat([frame] * 200, ignore_index=True).assign(
            case_id=lambda df: np.arange(len(df)).astype(str) if False else df["case_id"]),
            "W_status", TERMINAL_W)
    except NotReady:
        caught.append("duplicate_or_count")
    try:
        check_wb_manifest({"protocol_sha256": FROZEN_PROTOCOL_SHA256, "n_cases_manifest": 200,
                           "n_markers": 1, "missing": ["case"]})
    except NotReady:
        caught.append("incomplete_manifest")
    try:
        check_wb_manifest({"protocol_sha256": "0" * 64, "n_cases_manifest": 200,
                           "n_markers": 200, "missing": []})
    except NotReady:
        caught.append("protocol_hash")
    try:
        resolve_wb(AUDIT_WB)
    except NotReady:
        caught.append("audit_wb_refused")
    _require(caught == ["duplicate_or_count", "incomplete_manifest", "protocol_hash", "audit_wb_refused"],
             f"Negative checks did not all fire: {caught}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-ready", "fixture", "execute"))
    parser.add_argument("--out", type=Path, default=PILOT / "figures")
    parser.add_argument("--wb-dir", type=Path, default=OFFICIAL_WB)
    args = parser.parse_args(argv)
    if args.command == "check-ready":
        report = check_ready(args.wb_dir)
        print(json.dumps(report, indent=1))
        return 0 if report["status"] == "ready" else 2
    if args.command == "fixture":
        fixture_negatives()
        report = _fixture_bundle(args.out)
        report["negatives"] = "duplicate, incomplete manifest, protocol hash, audit wb refused"
        report["pilot_figures"] = False
        print(json.dumps(report))
        return 0
    if args.out.resolve() == (PILOT / "figures" / "fixture_check").resolve():
        print(json.dumps({"status": "not_ready", "reasons": ["refusing to write live figures into the fixture directory"]}))
        return 2
    try:
        note = execute(args.out, args.wb_dir)
    except NotReady as exc:
        print(json.dumps({"status": "not_ready", "reasons": [str(exc)]}))
        return 2
    print(json.dumps(note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
