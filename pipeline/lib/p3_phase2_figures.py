"""Phase 2 figure pipeline. Layout only. No envelope, forecast, or score is computed.

Official analysis tables are read, checked, and drawn. This module does not
build those tables. The synthetic fixture is an invented grid used to prove
the layout; every fixture image is labeled as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "phase2-figure-contract-1"

STORAGE_ORDER = ["confined", "leaky", "unconfined"]
STORAGE_LABEL = {"confined": "Confined", "leaky": "Leaky", "unconfined": "Unconfined"}
STORAGE_COLOR = {"confined": "#0072B2", "leaky": "#E69F00", "unconfined": "#009E73"}
T_LEVELS = [50.0, 500.0]
T_MARKER = {50.0: "o", 500.0: "s"}
RHO_LEVELS = [0.25, 1.0, 4.0]
N_LEVELS = [2, 6, 18]
PAIRS = ["P1_continue_vs_stop", "P2_current_vs_1p5x"]
TRACKS = ["raw", "exp10", "exp30", "exp90"]
TRACK_LABEL = {
    "raw": "Unfiltered pumping",
    "exp10": "Exponential memory, 10 d",
    "exp30": "Exponential memory, 30 d",
    "exp90": "Exponential memory, 90 d",
}
LEADS = [10, 30]
QUANTITIES = ["lead", "max_days_1_10"]
REQUIRED_COLUMNS = [
    "case_id",
    "storage_type",
    "T_m2_d",
    "rho",
    "N_transitions",
    "realization",
    "pair_id",
    "track",
    "quantity",
    "lead_day",
    "W_m",
    "W_status",
    "SR",
    "pi_r",
    "rest_over_response",
    "E_true_m",
    "E_tool_m",
    "effect_ratio",
    "ratio_defined",
    "sign_agreement",
    "error_over_W",
    "envelope_includes",
    "data_status",
]
FORBIDDEN_TEXT = ("TimesFM", "Pastas", "Moirai", "Chronos", "GRU", "win", "rank")


class FigureContractError(ValueError):
    pass


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
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def font_report():
    path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    return {"requested": "Times New Roman", "resolved_path": path}


def _fail(message):
    raise FigureContractError(message)


def _check_text(text):
    for banned in FORBIDDEN_TEXT:
        if banned.lower() in text.lower():
            _fail(f"Figure text contains '{banned}': {text}")


def load_table(path):
    frame = pd.read_csv(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        _fail(f"Missing columns: {', '.join(missing)}")
    return frame


def _as_bool_series(series):
    if series.dtype == bool:
        return series
    mapped = series.map({
        "1": True, "0": False, "True": True, "False": False,
        "true": True, "false": False, 1: True, 0: False, 1.0: True, 0.0: False,
    })
    if mapped.isna().any() and series.isna().sum() != mapped.isna().sum():
        _fail("Boolean column has values other than 0/1 or true/false")
    return mapped


def validate_table(frame, *, require_full_grid=True):
    """Return a cleaned copy. Every physical cell must be present when the grid is required."""
    work = frame.copy()
    work["storage_type"] = work["storage_type"].astype(str)
    work["pair_id"] = work["pair_id"].astype(str)
    work["track"] = work["track"].astype(str)
    work["quantity"] = work["quantity"].astype(str)
    work["T_m2_d"] = work["T_m2_d"].astype(float)
    work["rho"] = work["rho"].astype(float)
    work["N_transitions"] = work["N_transitions"].astype(int)
    work["realization"] = work["realization"].astype(int)
    work["W_status"] = work["W_status"].astype(str)
    work["data_status"] = work["data_status"].astype(str)
    work["ratio_defined"] = _as_bool_series(work["ratio_defined"])
    for name in ("W_m", "SR", "pi_r", "rest_over_response", "E_true_m", "E_tool_m",
                 "effect_ratio", "error_over_W", "lead_day", "sign_agreement",
                 "envelope_includes"):
        work[name] = pd.to_numeric(work[name], errors="coerce")

    unknown_storage = sorted(set(work["storage_type"]) - set(STORAGE_ORDER))
    if unknown_storage:
        _fail(f"Unknown storage_type: {unknown_storage}")
    unknown_track = sorted(set(work["track"]) - set(TRACKS))
    if unknown_track:
        _fail(f"Unknown track: {unknown_track}")
    unknown_pair = sorted(set(work["pair_id"]) - set(PAIRS))
    if unknown_pair:
        _fail(f"Unknown pair_id: {unknown_pair}")
    unknown_quantity = sorted(set(work["quantity"]) - set(QUANTITIES))
    if unknown_quantity:
        _fail(f"Unknown quantity: {unknown_quantity}")
    if work["data_status"].nunique() != 1:
        _fail("data_status must be a single value for the whole table")

    lead_rows = work["quantity"].eq("lead")
    if lead_rows.any() and not set(work.loc[lead_rows, "lead_day"].dropna().astype(int)).issubset(set(LEADS)):
        _fail("lead_day on quantity=lead must be 10 or 30")
    defined = work["ratio_defined"].fillna(False)
    bad_defined = defined & work["effect_ratio"].isna()
    if bad_defined.any():
        _fail("ratio_defined is true where effect_ratio is empty")
    bad_undefined = (~defined) & work["effect_ratio"].notna() & lead_rows
    if bad_undefined.any():
        _fail("effect_ratio is filled where ratio_defined is false")

    if require_full_grid:
        _require_grid(work)
    return work


def _require_grid(work):
    """Each stratum, rest ratio, transition count, realization, and pair is kept."""
    lead = work[work["quantity"].eq("lead") & work["track"].isin(TRACKS)]
    expected = (
        len(STORAGE_ORDER) * len(T_LEVELS) * len(RHO_LEVELS) * len(N_LEVELS)
        * 10 * len(PAIRS) * len(TRACKS) * len(LEADS)
    )
    realizations = set(lead["realization"].unique())
    if realizations != set(range(10)):
        _fail(f"Lead rows must contain realizations 0-9, found {sorted(realizations)}")
    if len(lead) != expected:
        _fail(f"Lead grid has {len(lead)} rows; the contract expects {expected}")
    keys = ["storage_type", "T_m2_d", "rho", "N_transitions", "realization", "pair_id", "track", "lead_day"]
    if lead.duplicated(keys).any():
        _fail("Duplicate lead rows for one case, pair, track, and lead")
    max_rows = work[work["quantity"].eq("max_days_1_10") & work["track"].eq("raw")]
    max_expected = (
        len(STORAGE_ORDER) * len(T_LEVELS) * len(RHO_LEVELS) * len(N_LEVELS) * 10 * len(PAIRS)
    )
    if len(max_rows) not in (0, max_expected):
        _fail(
            f"max_days_1_10 raw rows are {len(max_rows)}; expected 0 or {max_expected}. "
            "Do not send a partial maximum grid."
        )


def aggregate(work):
    """Median display table. Flagged and undefined rows stay in the counts."""
    group = [
        "storage_type", "T_m2_d", "rho", "N_transitions", "pair_id", "track",
        "quantity", "lead_day",
    ]

    def _one(block):
        defined = block["ratio_defined"].fillna(False)
        flagged = block["W_status"].ne("ok")
        ratio = block.loc[defined, "effect_ratio"]
        signed = block.loc[defined, "sign_agreement"]
        error = block.loc[block["error_over_W"].notna(), "error_over_W"]
        included = block["envelope_includes"]
        return pd.Series({
            "n_rows": int(len(block)),
            "n_W": int(block["W_m"].notna().sum()),
            "median_W_m": float(block["W_m"].median()) if block["W_m"].notna().any() else np.nan,
            "n_flagged": int(flagged.sum()),
            "n_ratio_defined": int(defined.sum()),
            "n_ratio_undefined": int((~defined).sum()),
            "median_effect_ratio": float(ratio.median()) if len(ratio) else np.nan,
            "n_sign": int(signed.notna().sum()),
            "sign_agreement_rate": float(signed.mean()) if signed.notna().any() else np.nan,
            "n_error": int(len(error)),
            "median_error_over_W": float(error.median()) if len(error) else np.nan,
            "n_envelope": int(included.notna().sum()),
            "envelope_rate": float(included.mean()) if included.notna().any() else np.nan,
            "median_SR": float(block["SR"].median()) if block["SR"].notna().any() else np.nan,
            "median_pi_r": float(block["pi_r"].median()) if block["pi_r"].notna().any() else np.nan,
            "median_E_true_m": float(block["E_true_m"].median()) if block["E_true_m"].notna().any() else np.nan,
        })

    out = work.groupby(group, dropna=False).apply(_one, include_groups=False).reset_index()
    return out


def _matrix(summary, *, value, storage, transmissivity, pair, track, quantity, lead):
    block = summary[
        summary["storage_type"].eq(storage)
        & np.isclose(summary["T_m2_d"], transmissivity)
        & summary["pair_id"].eq(pair)
        & summary["track"].eq(track)
        & summary["quantity"].eq(quantity)
    ]
    if quantity == "lead":
        block = block[np.isclose(block["lead_day"].astype(float), lead)]
    values = np.full((len(N_LEVELS), len(RHO_LEVELS)), np.nan)
    counts = np.zeros_like(values)
    defined = np.zeros_like(values)
    flagged = np.zeros_like(values)
    for i, count_n in enumerate(N_LEVELS):
        for j, rho in enumerate(RHO_LEVELS):
            hit = block[block["N_transitions"].eq(count_n) & np.isclose(block["rho"], rho)]
            if hit.empty:
                _fail(f"Missing cell {storage} T={transmissivity} N={count_n} rho={rho} {track} {quantity} {lead}")
            row = hit.iloc[0]
            values[i, j] = row[value]
            counts[i, j] = row["n_rows"]
            defined[i, j] = row["n_ratio_defined"]
            flagged[i, j] = row["n_flagged"]
    return values, counts, defined, flagged


def _stamp(figure, provenance):
    if provenance == "synthetic_fixture":
        figure.text(
            0.5, 0.995, "Synthetic fixture. Not a Phase 2 result.",
            ha="center", va="top", fontsize=8, fontstyle="italic",
        )


def _span(values, *, ratio):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if ratio:
        if finite.size == 0:
            return TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
        low = min(0.0, float(finite.min()))
        high = max(2.0, float(finite.max()))
        if high <= 1.0:
            high = 1.0 + 1e-6
        if low >= 1.0:
            low = 1.0 - 1e-6
        return TwoSlopeNorm(vmin=low, vcenter=1.0, vmax=high)
    if finite.size == 0:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        high = low + 1e-6
    return mpl.colors.Normalize(vmin=low, vmax=high)


def _heatmap(axis, values, counts, defined, flagged, *, ratio, norm):
    shown = np.ma.masked_invalid(values)
    cmap = "RdBu_r" if ratio else "viridis"
    image = axis.imshow(shown, cmap=cmap, norm=norm, origin="upper", aspect="equal")
    axis.set_xticks(range(len(RHO_LEVELS)), [f"{rho:g}" for rho in RHO_LEVELS])
    axis.set_yticks(range(len(N_LEVELS)), [str(count_n) for count_n in N_LEVELS])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if ratio and defined[i, j] == 0:
                label, color = "undef.", "0.2"
            elif ratio and defined[i, j] < counts[i, j]:
                label, color = f"{int(defined[i, j])}/{int(counts[i, j])}", "black"
            elif flagged[i, j] > 0:
                label, color = f"flag {int(flagged[i, j])}", "white"
            else:
                continue
            axis.text(j, i, label, ha="center", va="center", fontsize=6, color=color)
    return image


def _plane(summary, *, value, pair, track, quantity, provenance, ratio, suptitle_note):
    apply_style()
    panels = []
    for storage in STORAGE_ORDER:
        for lead, transmissivity in ((lead, level) for lead in LEADS for level in T_LEVELS):
            panels.append(_matrix(
                summary, value=value, storage=storage, transmissivity=transmissivity,
                pair=pair, track=track, quantity=quantity, lead=lead,
            ))
    norms = {}
    for lead in LEADS:
        gathered = [panels[index][0] for index in range(len(panels)) if (index % 4) // 2 == LEADS.index(lead)]
        norms[lead] = _span(np.concatenate([item.ravel() for item in gathered]), ratio=ratio)
    figure, axes = plt.subplots(len(STORAGE_ORDER), 4, figsize=(7.2, 7.35), squeeze=False)
    images = {10: None, 30: None}
    panel_index = 0
    for row, storage in enumerate(STORAGE_ORDER):
        for col, (lead, transmissivity) in enumerate(
            (lead, level) for lead in LEADS for level in T_LEVELS
        ):
            axis = axes[row, col]
            values, counts, defined, flagged = panels[panel_index]
            panel_index += 1
            image = _heatmap(axis, values, counts, defined, flagged, ratio=ratio, norm=norms[lead])
            images[lead] = image
            if row == 0:
                day = "Days 1–10 maximum" if quantity == "max_days_1_10" else f"Day {lead}"
                axis.set_title(f"{day}\nT = {transmissivity:g} m²/d")
            if row == len(STORAGE_ORDER) - 1:
                axis.set_xlabel("Rest ratio ρ")
            if col == 0:
                axis.set_ylabel(f"{STORAGE_LABEL[storage]}\nTransition count N")
    figure.subplots_adjust(left=0.12, right=0.70, top=0.86, bottom=0.08, wspace=0.48, hspace=0.38)
    cax_left = figure.add_axes([0.74, 0.16, 0.018, 0.60])
    cax_right = figure.add_axes([0.86, 0.16, 0.018, 0.60])
    unit = "E_tool / E_true" if ratio else "Median W (m)"
    figure.colorbar(images[10], cax=cax_left)
    figure.colorbar(images[30], cax=cax_right)
    cax_left.set_ylabel(f"Day 10\n{unit}", fontweight="bold")
    cax_right.set_ylabel(f"Day 30\n{unit}", fontweight="bold")
    _stamp(figure, provenance)
    _check_text(suptitle_note)
    figure.text(0.5, 0.955, suptitle_note, ha="center", va="top", fontsize=11, fontweight="bold")
    return figure


def figure_a(summary, provenance):
    return _plane(
        summary, value="median_W_m", pair="P1_continue_vs_stop", track="raw",
        quantity="lead", provenance=provenance, ratio=False,
        suptitle_note="Envelope width on the transition-count and rest-ratio plane",
    )


def figure_a2(summary, provenance):
    apply_style()
    points = summary[
        summary["pair_id"].eq("P1_continue_vs_stop")
        & summary["track"].eq("raw")
        & summary["quantity"].eq("lead")
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharey="row")
    x_fields = [("median_SR", "Signal ratio"), ("median_pi_r", "π_r")]
    for row, (x_name, x_label) in enumerate(x_fields):
        for col, lead in enumerate(LEADS):
            axis = axes[row, col]
            block = points[np.isclose(points["lead_day"].astype(float), lead)]
            for storage in STORAGE_ORDER:
                for level, marker in T_MARKER.items():
                    hit = block[block["storage_type"].eq(storage) & np.isclose(block["T_m2_d"], level)]
                    axis.scatter(
                        hit[x_name], hit["median_W_m"],
                        c=STORAGE_COLOR[storage], marker=marker, s=28,
                        label=f"{STORAGE_LABEL[storage]}, T = {level:g}",
                        linewidths=0.3, edgecolors="white",
                    )
            axis.set_xscale("log")
            axis.set_xlabel(x_label, fontweight="bold")
            if col == 0:
                axis.set_ylabel("Median W (m)", fontweight="bold")
            axis.set_title(f"Day {lead}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.18, hspace=0.38, wspace=0.18)
    _stamp(figure, provenance)
    note = "Derived coordinates beside raw envelope width"
    _check_text(note)
    figure.text(0.5, 0.955, note, ha="center", va="top", fontsize=11, fontweight="bold")
    return figure


def figure_b(summary, provenance):
    return _plane(
        summary, value="median_effect_ratio", pair="P1_continue_vs_stop", track="raw",
        quantity="lead", provenance=provenance, ratio=True,
        suptitle_note="Point-contrast ratio on the same record plane",
    )


def figure_filter(summary, provenance, track):
    return _plane(
        summary, value="median_effect_ratio", pair="P1_continue_vs_stop", track=track,
        quantity="lead", provenance=provenance, ratio=True,
        suptitle_note=TRACK_LABEL[track],
    )


def figure_max(summary, provenance):
    present = summary["quantity"].eq("max_days_1_10")
    if not present.any():
        return None
    apply_style()
    figure, axes = plt.subplots(len(STORAGE_ORDER), 2, figsize=(7.2, 6.4), squeeze=False)
    gathered = []
    panels = []
    for storage in STORAGE_ORDER:
        for transmissivity in T_LEVELS:
            values, _c, defined, flagged = _matrix(
                summary, value="median_W_m", storage=storage, transmissivity=transmissivity,
                pair="P1_continue_vs_stop", track="raw", quantity="max_days_1_10", lead=None,
            )
            panels.append((values, defined, flagged))
            gathered.append(values)
    norm = _span(np.concatenate([item.ravel() for item in gathered]), ratio=False)
    image = None
    for row, storage in enumerate(STORAGE_ORDER):
        for col, transmissivity in enumerate(T_LEVELS):
            axis = axes[row, col]
            values, defined, flagged = panels[row * 2 + col]
            counts = np.full_like(values, 10.0)
            image = _heatmap(axis, values, counts, defined, flagged, ratio=False, norm=norm)
            if row == 0:
                axis.set_title(f"T = {transmissivity:g} m²/d")
            if row == len(STORAGE_ORDER) - 1:
                axis.set_xlabel("Rest ratio ρ")
            if col == 0:
                axis.set_ylabel(f"{STORAGE_LABEL[storage]}\nTransition count N")
    figure.subplots_adjust(left=0.16, right=0.84, top=0.82, bottom=0.1, wspace=0.28, hspace=0.32)
    cax = figure.add_axes([0.88, 0.18, 0.025, 0.55])
    figure.colorbar(image, cax=cax).set_label("Median maximum W (m)")
    _stamp(figure, provenance)
    note = "Maximum envelope width over days 1 to 10"
    _check_text(note)
    figure.text(0.5, 0.955, note, ha="center", va="top", fontsize=11, fontweight="bold")
    return figure


def figure_p2(summary, provenance):
    apply_style()
    raw = summary[summary["track"].eq("raw") & summary["quantity"].eq("lead")]
    left = raw[raw["pair_id"].eq("P1_continue_vs_stop")].set_index(
        ["storage_type", "T_m2_d", "rho", "N_transitions", "lead_day"]
    )
    right = raw[raw["pair_id"].eq("P2_current_vs_1p5x")].set_index(
        ["storage_type", "T_m2_d", "rho", "N_transitions", "lead_day"]
    )
    joined = left[["median_W_m", "median_E_true_m"]].join(
        right[["median_W_m", "median_E_true_m"]], lsuffix="_p1", rsuffix="_p2"
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.scatter(joined["median_W_m_p1"], joined["median_W_m_p2"], s=22, c="#0072B2", label="Cell median")
    limit = max(joined["median_W_m_p1"].max(), joined["median_W_m_p2"].max()) * 1.05
    grid = np.linspace(0, limit, 20)
    axis.plot(grid, grid, color="0.4", lw=0.8, label="Equal width")
    axis.set_xlabel("Median W, continue versus stop (m)", fontweight="bold")
    axis.set_ylabel("Median W, current versus 1.5× (m)", fontweight="bold")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.legend(frameon=False, loc="upper left")
    figure.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.16)
    _stamp(figure, provenance)
    note = "Pair identity check. Both pairs stay in the table."
    _check_text(note)
    figure.text(0.5, 0.955, note, ha="center", va="top", fontsize=11, fontweight="bold")
    return figure, joined.reset_index()


def save_figure(figure, stem, out_dir):
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    raw = pdf.read_bytes()
    return {
        "png": png.name,
        "pdf": pdf.name,
        "png_sha256": sha256_file(png),
        "pdf_sha256": sha256_file(pdf),
        "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size,
        "pdf_embeds_times": b"TimesNewRoman" in raw or b"Times New Roman" in raw,
    }


def render(frame, out_dir, *, provenance, require_full_grid=True):
    if provenance not in ("synthetic_fixture", "official_analysis"):
        _fail("provenance must be synthetic_fixture or official_analysis")
    work = validate_table(frame, require_full_grid=require_full_grid)
    if work["data_status"].iloc[0] != provenance:
        _fail("data_status does not match the requested provenance")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(work)
    summary_path = out_dir / "cell_median.csv"
    summary.to_csv(summary_path, index=False)
    records = {}
    records["figure_A"] = save_figure(figure_a(summary, provenance), "figure_A_width", out_dir)
    records["figure_A2"] = save_figure(figure_a2(summary, provenance), "figure_A2_derived", out_dir)
    records["figure_B"] = save_figure(figure_b(summary, provenance), "figure_B_ratio", out_dir)
    for track in TRACKS[1:]:
        records[f"figure_filter_{track}"] = save_figure(
            figure_filter(summary, provenance, track), f"figure_S_{track}", out_dir,
        )
    maximum = figure_max(summary, provenance)
    if maximum is not None:
        records["figure_max"] = save_figure(maximum, "figure_S_max_days_1_10", out_dir)
    else:
        records["figure_max"] = {"status": "absent", "reason": "quantity max_days_1_10 was not in the table"}
    diagnostic, joined = figure_p2(summary, provenance)
    records["figure_P2"] = save_figure(diagnostic, "figure_D_pair_identity", out_dir)
    joined.to_csv(out_dir / "pair_identity.csv", index=False)
    # Required numeric rows: both pairs, all strata, all tracks.
    pair_counts = summary.groupby("pair_id").size().to_dict()
    stratum_counts = summary.groupby(["storage_type", "T_m2_d"]).size().to_dict()
    return {
        "contract_version": CONTRACT_VERSION,
        "provenance": provenance,
        "n_input_rows": int(len(work)),
        "n_median_rows": int(len(summary)),
        "pair_row_counts": {str(key): int(value) for key, value in pair_counts.items()},
        "stratum_row_counts": {f"{a}|T={b:g}": int(v) for (a, b), v in stratum_counts.items()},
        "font": font_report(),
        "figures": records,
        "median_csv": summary_path.name,
        "pair_identity_csv": "pair_identity.csv",
    }


def build_synthetic_fixture():
    """Invented numbers. The shape is planted so the panels are not blank."""
    rows = []
    storage_w = {"confined": 1.0, "leaky": 0.55, "unconfined": 0.12}
    stratum_sr = {
        ("confined", 50.0): 2.4, ("confined", 500.0): 0.40,
        ("leaky", 50.0): 1.6, ("leaky", 500.0): 0.28,
        ("unconfined", 50.0): 0.12, ("unconfined", 500.0): 0.07,
    }
    stratum_pi = {
        ("confined", 50.0): 0.02, ("confined", 500.0): 0.002,
        ("leaky", 50.0): 0.20, ("leaky", 500.0): 0.02,
        ("unconfined", 50.0): 20.0, ("unconfined", 500.0): 2.0,
    }
    track_ratio = {"raw": 0.55, "exp10": 0.80, "exp30": 0.70, "exp90": 0.62}
    for storage in STORAGE_ORDER:
        for level in T_LEVELS:
            for rho in RHO_LEVELS:
                for count_n in N_LEVELS:
                    for realization in range(10):
                        sr = stratum_sr[(storage, level)] * (0.75 + 0.05 * realization)
                        pi_r = stratum_pi[(storage, level)]
                        base = 0.08 * storage_w[storage] * (1.0 if level == 50.0 else 0.35)
                        w10 = base * (1.0 - 0.08 * np.log(count_n / 2.0)) * (1.0 - 0.05 * np.log(rho / 0.25))
                        w30 = base * 1.4 * (1.0 - 0.04 * np.log(count_n / 2.0)) * (1.0 - 0.22 * np.log(rho / 0.25))
                        w10 *= 1.0 + 0.01 * realization
                        w30 *= 1.0 + 0.01 * realization
                        status = "ok"
                        if storage == "confined" and level == 50.0 and rho == 4.0 and count_n == 18 and realization == 0:
                            status = "not_converged"
                        for pair_index, pair in enumerate(PAIRS):
                            scale = 1.0 if pair_index == 0 else 0.5
                            for track in TRACKS:
                                for lead, width in ((10, w10), (30, w30)):
                                    e_true = (0.02 if storage != "unconfined" else 0.004) * scale
                                    blank = (
                                        storage == "unconfined" and level == 500.0
                                        and rho == 4.0 and count_n == 18 and lead == 30
                                    )
                                    partial = (
                                        storage == "unconfined" and level == 50.0
                                        and rho == 0.25 and count_n == 2 and lead == 10
                                        and realization >= 8
                                    )
                                    defined = not (blank or partial)
                                    ratio = track_ratio[track] if defined else np.nan
                                    if defined and lead == 30:
                                        ratio = ratio + 0.05
                                    e_tool = ratio * e_true if defined else np.nan
                                    rows.append({
                                        "case_id": f"{storage}_T{level:g}_r{rho:g}_N{count_n}_z{realization}",
                                        "storage_type": storage,
                                        "T_m2_d": level,
                                        "rho": rho,
                                        "N_transitions": count_n,
                                        "realization": realization,
                                        "pair_id": pair,
                                        "track": track,
                                        "quantity": "lead",
                                        "lead_day": lead,
                                        "W_m": width * scale,
                                        "W_status": status,
                                        "SR": sr,
                                        "pi_r": pi_r,
                                        "rest_over_response": rho,
                                        "E_true_m": 0.0 if not defined else e_true,
                                        "E_tool_m": e_tool,
                                        "effect_ratio": ratio,
                                        "ratio_defined": int(defined),
                                        "sign_agreement": 1 if defined else np.nan,
                                        "error_over_W": (abs(e_tool - e_true) / (width * scale)) if defined else np.nan,
                                        "envelope_includes": 1 if defined else np.nan,
                                        "data_status": "synthetic_fixture",
                                    })
                            rows.append({
                                "case_id": f"{storage}_T{level:g}_r{rho:g}_N{count_n}_z{realization}",
                                "storage_type": storage,
                                "T_m2_d": level,
                                "rho": rho,
                                "N_transitions": count_n,
                                "realization": realization,
                                "pair_id": pair,
                                "track": "raw",
                                "quantity": "max_days_1_10",
                                "lead_day": np.nan,
                                "W_m": w10 * 1.25 * scale,
                                "W_status": status,
                                "SR": sr,
                                "pi_r": pi_r,
                                "rest_over_response": rho,
                                "E_true_m": np.nan,
                                "E_tool_m": np.nan,
                                "effect_ratio": np.nan,
                                "ratio_defined": 0,
                                "sign_agreement": np.nan,
                                "error_over_W": np.nan,
                                "envelope_includes": np.nan,
                                "data_status": "synthetic_fixture",
                            })
    return pd.DataFrame(rows)


def png_size(path):
    raw = Path(path).read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        _fail(f"Not a PNG: {path}")
    width, height = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return width, height


def main():
    parser = argparse.ArgumentParser(description="Draw Phase 2 figures from a tidy analysis table.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", choices=["synthetic_fixture", "official_analysis"], required=True)
    parser.add_argument("--write-fixture-csv", action="store_true")
    args = parser.parse_args()
    if args.provenance == "synthetic_fixture" and "figure_fixtures" not in args.out.as_posix():
        _fail("Synthetic fixture output must live under a figure_fixtures directory")
    if args.write_fixture_csv:
        frame = build_synthetic_fixture()
        args.out.mkdir(parents=True, exist_ok=True)
        csv_path = args.out / "fixture_cell_metrics.csv"
        frame.to_csv(csv_path, index=False)
    elif args.input is None:
        _fail("Pass --input or --write-fixture-csv")
    else:
        frame = load_table(args.input)
        csv_path = args.input
    report = render(frame if not args.write_fixture_csv else pd.read_csv(args.out / "fixture_cell_metrics.csv"),
                    args.out, provenance=args.provenance)
    if args.write_fixture_csv:
        report["input_csv"] = "fixture_cell_metrics.csv"
        report["input_sha256"] = sha256_file(args.out / "fixture_cell_metrics.csv")
    else:
        report["input_csv"] = str(csv_path)
        report["input_sha256"] = sha256_file(csv_path)
    for key, record in report["figures"].items():
        if "png" in record:
            width, height = png_size(args.out / record["png"])
            record["width_px"] = width
            record["height_px"] = height
    report["code_sha256"] = sha256_file(Path(__file__))
    (args.out / "fixture_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n_input_rows": report["n_input_rows"],
                      "n_median_rows": report["n_median_rows"]}, indent=2))


if __name__ == "__main__":
    main()
