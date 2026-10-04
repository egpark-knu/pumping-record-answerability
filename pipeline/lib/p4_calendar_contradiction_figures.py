"""D05 Figures 3 and 4 with companions (non-scoring plot producer).

Reads only the official, already-computed D05 numerical tables
(results/phase4/tables/*.csv written by the official analysis) and
results/phase4/primary_rows.csv for an independent count cross-check.
No model, W, reference, truth, fit or bootstrap is computed here.
Every plotted number is written to a plot-input CSV beside the figure.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results/phase4"
T = P / "tables"
F = P / "figures"

FORMS = ["water_curtain", "paddy_irrigation", "domestic_continuous"]
FORM_LABEL = {"water_curtain": "Water curtain", "paddy_irrigation": "Paddy irrigation",
              "domestic_continuous": "Domestic continuous"}
STORAGES = ["confined", "leaky", "unconfined"]
LAYERS = ["confined_T50", "confined_T500", "leaky_T50", "leaky_T500", "unconfined_T50", "unconfined_T500"]
PAIRS = {"P1": "P1_continue_vs_stop", "P2": "P2_current_vs_1p5x"}
PAIR_LABEL = {"P1": "Continue vs stop (P1)", "P2": "Current vs 1.5 times pumping (P2)"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROUPS = ["confined", "leaky", "unconfined", "fixedc"]
GROUP_LABEL = {"confined": "Confined", "leaky": "Leaky", "unconfined": "Unconfined",
               "fixedc": "D04 fixed-coefficient storage sweep"}
GROUP_COLOR = {"confined": "#1b6ca8", "leaky": "#e08a1e", "unconfined": "#7a3e9d", "fixedc": "#8c8c8c"}
CMAP = plt.get_cmap("YlGnBu")
MARK = "#d7301f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_font():
    path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=path).get_name()
    if family != "Times New Roman":
        raise SystemExit(f"Times New Roman not resolved: {path} -> {family}")
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 220, "hatch.linewidth": 0.6})
    return path


def save(fig, stem, products):
    out = {}
    for ext in ("png", "pdf", "svg"):
        path = F / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
        out[ext] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    plt.close(fig)
    products[stem] = out


def pair_rows(df, pair):
    return df[df["pair_id"] == PAIRS[pair]]


# ---------------------------------------------------------------- Figure 3 core
def calendar_panel(ax, cells, rows, row_key, lead, active_only=False, denom_label=None):
    """cells: dataframe indexed by (row_value, month) with count columns."""
    norm = Normalize(0, 1)
    nrow = len(rows)
    for i, rv in enumerate(rows):
        y = nrow - 1 - i
        for m in range(1, 13):
            c = cells.get((rv, m))
            x = m - 1
            if c is None:
                continue
            den_all = c["denominator"]
            n_inactive = c["n_no_active_contrast"]
            n_active = c["n_active_contrast"]
            if active_only:
                den = n_active
                num_sign = c["active_sign_num"]
                num_mag = c["active_mag_num"]
            else:
                den = den_all
                num_sign = c["n_sign_determined"]
                num_mag = c["n_relative_width_below_one"]
            if den == 0 or (not active_only and n_active == 0):
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#e6e6e6", edgecolor="#555555",
                                       hatch="xxxx", linewidth=0.4))
                ax.text(x + 0.5, y + 0.5, "none" if active_only else f"{n_inactive}",
                        ha="center", va="center", fontsize=6.5, style="italic", color="#333333")
                continue
            rate = num_sign / den
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=CMAP(norm(rate)), edgecolor="white", linewidth=0.6))
            if (not active_only) and n_inactive > 0:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="none", edgecolor="#555555",
                                       hatch="////", linewidth=0.0))
                ax.text(x + 0.06, y + 0.08, f"{n_inactive}", ha="left", va="bottom", fontsize=6.2,
                        style="italic", color="black",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.85))
            if active_only:
                ax.text(x + 0.06, y + 0.08, f"{den}", ha="left", va="bottom", fontsize=6.2,
                        color="black", bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.85))
            frac = num_mag / den
            if num_mag > 0:
                ax.scatter([x + 0.70], [y + 0.55], s=10 + 110 * frac, color=MARK,
                           edgecolor="black", linewidth=0.5, zorder=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, nrow)
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(MONTHS)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{lead}-day lead", fontsize=11, fontweight="bold", loc="left")


def form_storage_axes(ax, rows_per_form, sublabels, label_x=-0.16):
    n = len(FORMS) * rows_per_form
    ticks, labels = [], []
    for fi, form in enumerate(FORMS):
        for si, sub in enumerate(sublabels):
            i = fi * rows_per_form + si
            ticks.append(n - 1 - i + 0.5)
            labels.append(sub)
        top = n - fi * rows_per_form
        if fi > 0:
            ax.axhline(top, color="black", linewidth=1.1)
        ax.text(label_x, top - rows_per_form / 2, FORM_LABEL[form].replace(" ", "\n"),
                transform=ax.get_yaxis_transform(), ha="center", va="center", rotation=90, fontsize=9,
                fontweight="bold", linespacing=1.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", pad=2)


def fig3_legend(fig, active_only, denom_text):
    handles = []
    for f in (0.25, 0.5, 1.0):
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor=MARK, markeredgecolor="black",
                              markeredgewidth=0.5, markersize=math.sqrt(10 + 110 * f),
                              label=f"W/|E$_\\mathrm{{true}}$| < 1 in {f:g} of {denom_text}"))
    if active_only:
        handles.append(Patch(facecolor="#e6e6e6", edgecolor="#555555", hatch="xxxx",
                             label="No active contrast in the cell (all origin rates zero)"))
        handles.append(Patch(facecolor="white", edgecolor="white", label="Small number: active cases (denominator)"))
    else:
        handles.append(Patch(facecolor="#9ecae1", edgecolor="#555555", hatch="////",
                             label="Some origins have zero origin-day rate (no active contrast)"))
        handles.append(Patch(facecolor="#e6e6e6", edgecolor="#555555", hatch="xxxx",
                             label="All origins zero rate: no active contrast, not poor answerability"))
        handles.append(Patch(facecolor="white", edgecolor="white",
                             label="Italic number: zero-rate origins in the cell"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005),
               fontsize=8.6, handletextpad=0.5, columnspacing=1.6)


def fig3_colorbar(fig, axes, label):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8.5)


def cells_dict(df, row_col):
    out = {}
    for _, r in df.iterrows():
        out[(r[row_col], int(r["origin_month"]))] = {
            "denominator": int(r["denominator"]), "n_sign_determined": int(r["n_sign_determined"]),
            "n_relative_width_below_one": int(r["n_relative_width_below_one"]),
            "n_no_active_contrast": int(r["n_no_active_contrast"]), "n_active_contrast": int(r["n_active_contrast"]),
            "active_sign_num": int(r["active_sign_rate.numerator"]),
            "active_mag_num": int(r["active_relative_width_rate.numerator"])}
    return out


def figure3(storage_cells, pair, products, plotinputs, active_only=False):
    d = pair_rows(storage_cells, pair).copy()
    d["row"] = d["calendar_type"] + "|" + d["storage_type"]
    rows = [f"{f}|{s}" for f in FORMS for s in STORAGES]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.6))
    fig.subplots_adjust(left=0.22, right=0.9, top=0.95, bottom=0.15, hspace=0.28)
    for ax, lead in zip(axes, (10, 30)):
        sub = d[d["lead_day"] == lead]
        calendar_panel(ax, cells_dict(sub, "row"), rows, "row", lead, active_only)
        form_storage_axes(ax, 3, [s.capitalize() for s in STORAGES])
    if active_only:
        fig3_colorbar(fig, axes, "Direction determined / active-contrast cases")
        fig3_legend(fig, True, "active cases")
    else:
        fig3_colorbar(fig, axes, "Direction determined / all 20 scheduled origins")
        fig3_legend(fig, False, "all 20")
    fig.suptitle(f"Operating calendar answerability, {PAIR_LABEL[pair]}", fontsize=11.5, fontweight="bold", y=0.995)
    stem = f"figure3_calendar_{'active_only_' if active_only else ''}{pair}"
    save(fig, stem, products)
    cols = ["origin_month", "calendar_type", "storage_type", "lead_day", "pair_id", "denominator",
            "n_sign_determined", "n_sign_undetermined", "n_sign_missing", "n_relative_width_below_one",
            "n_relative_width_defined", "n_relative_width_undefined", "n_no_active_contrast", "n_active_contrast",
            "sign_rate.rate", "relative_width_rate.rate", "active_sign_rate.numerator", "active_sign_rate.denominator",
            "active_sign_rate.rate", "active_relative_width_rate.numerator", "active_relative_width_rate.rate",
            "sign_rate.bootstrap.low", "sign_rate.bootstrap.high", "display_status"]
    path = T / f"{stem}_plotinput.csv"
    d[cols].sort_values(["lead_day", "calendar_type", "storage_type", "origin_month"]).to_csv(path, index=False)
    plotinputs[stem] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def figure3_layer(cells, pair, products, plotinputs):
    d = pair_rows(cells, pair).copy()
    d["row"] = d["calendar_type"] + "|" + d["layer_id"]
    rows = [f"{f}|{l}" for f in FORMS for l in LAYERS]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 7.4), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.93, bottom=0.15, wspace=0.06)
    for ax, lead in zip(axes, (10, 30)):
        sub = d[d["lead_day"] == lead]
        calendar_panel(ax, cells_dict(sub, "row"), rows, "row", lead, False)
        ax.set_xticklabels([m[0] for m in MONTHS])
        if lead == 10:
            form_storage_axes(ax, 6, [f"{l.split('_T')[0].capitalize()}, T = {l.split('_T')[1]} m²/d" for l in LAYERS],
                              label_x=-0.36)
    fig3_colorbar(fig, axes, "Direction determined / all 10 scheduled origins")
    fig3_legend(fig, False, "all 10")
    fig.suptitle(f"Operating calendar answerability by layer, {PAIR_LABEL[pair]}", fontsize=11.5,
                 fontweight="bold", y=0.995)
    stem = f"figure3_layer_detail_{pair}"
    save(fig, stem, products)
    path = T / f"{stem}_plotinput.csv"
    d.drop(columns=["row"]).to_csv(path, index=False)
    plotinputs[stem] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def figure3_variant(variant_cells, pair, products, plotinputs):
    d = pair_rows(variant_cells, pair)
    d = d[d["calendar_type"] == "water_curtain"].copy()
    keys = ["origin_month", "storage_type", "lead_day", "offseason_use_variant"]
    sums = ["denominator", "n_sign_determined", "n_relative_width_below_one", "n_no_active_contrast",
            "n_active_contrast", "active_sign_rate.numerator", "active_relative_width_rate.numerator",
            "n_relative_width_undefined"]
    g = d.groupby(keys, as_index=False)[sums].sum()
    g["variant_label"] = np.where(g["offseason_use_variant"].astype(str).str.lower() == "true",
                                  "Off-season use", "No off-season use")
    g["row"] = g["storage_type"] + "|" + g["variant_label"]
    rows = [f"{s}|{v}" for s in STORAGES for v in ("No off-season use", "Off-season use")]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4))
    fig.subplots_adjust(left=0.24, right=0.9, top=0.94, bottom=0.2, hspace=0.3)
    for ax, lead in zip(axes, (10, 30)):
        sub = g[g["lead_day"] == lead]
        calendar_panel(ax, cells_dict(sub, "row"), rows, "row", lead, False)
        ticks, labels = [], []
        for i, rv in enumerate(rows):
            s, v = rv.split("|")
            n = int(sub[sub["row"] == rv]["denominator"].iloc[0])
            ticks.append(len(rows) - 1 - i + 0.5)
            labels.append(f"{s.capitalize()}, {v.lower()} (n = {n})")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8.6)
        for k in (2, 4):
            ax.axhline(k, color="black", linewidth=1.0)
    fig3_colorbar(fig, axes, "Direction determined / scheduled origins in row")
    fig3_legend(fig, False, "scheduled origins")
    fig.suptitle(f"Water-curtain calendar with and without off-season use, {PAIR_LABEL[pair]}",
                 fontsize=11, fontweight="bold", y=0.995)
    stem = f"figure3_offseason_variant_{pair}"
    save(fig, stem, products)
    path = T / f"{stem}_plotinput.csv"
    g.drop(columns=["row"]).to_csv(path, index=False)
    plotinputs[stem] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


# ---------------------------------------------------------------- Figure 4
def bin_storage_inputs(bs, pair):
    d = pair_rows(bs, pair).copy()
    d = d[d["bin"] != "nonpositive_signal"].copy()
    d["bin"] = d["bin"].astype(int)
    d["group"] = np.where(d["storage_type"].str.startswith("fixedc_"), "fixedc", d["storage_type"])
    num = ["n", "determined_total", "strict_opposite_sign.numerator", "strict_opposite_sign.denominator",
           "zero_tool_prediction.numerator", "missing_tool.numerator", "reference_contradiction.numerator",
           "reference_contradiction.denominator", "reference_zero.numerator", "missing_reference.numerator",
           "undetermined_truth_error.numerator", "undetermined_truth_error.denominator",
           "reference_undetermined_truth_disagreement.numerator",
           "reference_undetermined_truth_disagreement.denominator", "n_zero_truth"]
    g = d.groupby(["bin", "group", "lead_day"], as_index=False)[num].sum()
    g["pair_id"] = PAIRS[pair]
    return g


def bar_panel(ax, g, lead, num, den, ref_num, ref_den, ylab, edges, ref_label):
    sub = g[g["lead_day"] == lead]
    width = 0.2
    ymax = 0
    for gi, grp in enumerate(GROUPS):
        for b in range(5):
            r = sub[(sub["bin"] == b) & (sub["group"] == grp)]
            n = int(r[num].iloc[0]) if len(r) else 0
            dn = int(r[den].iloc[0]) if len(r) else 0
            x = b + (gi - 1.5) * width
            if dn == 0:
                ax.text(x, 0.047, "n = 0", rotation=90, ha="center", va="bottom", fontsize=6.3, color="#777777")
                continue
            rate = n / dn
            ymax = max(ymax, rate)
            ax.bar(x, rate, width=width * 0.92, color=GROUP_COLOR[grp], edgecolor="black", linewidth=0.4,
                   hatch="///" if grp == "fixedc" else None)
            ax.text(x, max(rate, 0.035) + 0.012, f"{n}/{dn}", rotation=90, ha="center", va="bottom", fontsize=6.3)
    refs = []
    for b in range(5):
        r = sub[sub["bin"] == b]
        rn, rd = int(r[ref_num].sum()), int(r[ref_den].sum())
        refs.append(rn / rd if rd else np.nan)
    ax.plot(range(5), refs, color="black", marker="D", markersize=4.5, linewidth=1.4, label=ref_label, zorder=6)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{edges[i]:.3g}–\n{edges[i + 1]:.3g}" for i in range(5)], fontsize=8.4)
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylabel(ylab, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return refs


def figure4(bs, overall_rows, pair, edges, products, plotinputs):
    g = bin_storage_inputs(bs, pair)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.2, hspace=0.42, wspace=0.12)
    ref_rows = []
    for j, lead in enumerate((10, 30)):
        top = bar_panel(axes[0, j], g, lead, "strict_opposite_sign.numerator", "strict_opposite_sign.denominator",
                        "reference_contradiction.numerator", "reference_contradiction.denominator",
                        "Opposite-sign rate" if j == 0 else "", edges,
                        "Structurally matched reference, same rows (all storage)")
        bot = bar_panel(axes[1, j], g, lead, "undetermined_truth_error.numerator",
                        "undetermined_truth_error.denominator", "reference_undetermined_truth_disagreement.numerator",
                        "reference_undetermined_truth_disagreement.denominator",
                        "Truth-sign error rate" if j == 0 else "", edges,
                        "Structurally matched reference, same rows (all storage)")
        axes[0, j].set_title(f"{lead}-day lead: records that determine the direction", fontsize=10.5,
                             fontweight="bold", loc="left")
        axes[1, j].set_title(f"{lead}-day lead: records that leave the direction undetermined", fontsize=10.5,
                             fontweight="bold", loc="left")
        axes[1, j].set_xlabel("Signal-ratio bin (five pooled log-equal bins)", fontsize=10)
        o = overall_rows[overall_rows["lead_day"] == lead]
        det = int(o["determined_total"].sum())
        z = int(o["zero_tool_prediction.numerator"].sum())
        mi = int(o["missing_tool.numerator"].sum())
        zt = int(o["n_zero_truth"].sum())
        axes[0, j].text(0.99, 0.97, f"Tool zero outputs: {z} of {det}\nMissing tool outputs: {mi} of {det}",
                        transform=axes[0, j].transAxes, ha="right", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="#999999", linewidth=0.5, pad=2.5))
        axes[1, j].text(0.99, 0.97, f"Zero-effect records ({zt}, zero origin rate)\nhave no sign and are excluded",
                        transform=axes[1, j].transAxes, ha="right", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="#999999", linewidth=0.5, pad=2.5))
        for b in range(5):
            ref_rows.append({"pair_id": PAIRS[pair], "lead_day": lead, "bin": b, "edge_low": edges[b],
                             "edge_high": edges[b + 1], "reference_contradiction_rate_all_storage": top[b],
                             "reference_undetermined_truth_disagreement_rate_all_storage": bot[b]})
    handles = [Patch(facecolor=GROUP_COLOR[k], edgecolor="black", hatch="///" if k == "fixedc" else None,
                     label=GROUP_LABEL[k]) for k in GROUPS]
    handles.append(Line2D([], [], color="black", marker="D", markersize=4.5, linewidth=1.4,
                          label="Structurally matched reference on the same rows (all storage)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle(f"Tool sign against the record, {PAIR_LABEL[pair]}. Bar labels: count / denominator",
                 fontsize=11.5, fontweight="bold", y=0.985)
    stem = f"figure4_contradiction_{pair}"
    save(fig, stem, products)
    path = T / f"{stem}_plotinput.csv"
    g.to_csv(path, index=False)
    rpath = T / f"{stem}_reference_line_plotinput.csv"
    pd.DataFrame(ref_rows).to_csv(rpath, index=False)
    plotinputs[stem] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    plotinputs[stem + "_reference_line"] = {"path": str(rpath.relative_to(ROOT)), "sha256": sha(rpath)}


def variant_contradiction(rows):
    """Water-curtain contradiction split by the off-season Boolean, from official primary rows."""
    w = rows[rows["calendar_type"] == "water_curtain"].copy()
    w["variant"] = w["offseason_use_variant"].astype(str).str.lower() == "true"
    det = w[w["sign_determined"] == 1]
    det = det.assign(opp=(det["E_tool_m"] * det["env_sign"] < 0).astype(int))
    g = det.groupby(["pair_id", "lead_day", "origin_month", "variant"]).agg(
        n_opposite=("opp", "sum"), n_determined=("opp", "size")).reset_index()
    full = w.groupby(["pair_id", "lead_day", "origin_month", "variant"]).size().rename("n_rows").reset_index()
    return full.merge(g, how="left").fillna({"n_opposite": 0, "n_determined": 0})


def figure4_calendar(conc, vc, products, plotinputs):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.8), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.9, top=0.88, bottom=0.12, hspace=0.42, wspace=0.05)
    rows = ["water_curtain", "wc_base", "wc_variant", "paddy_irrigation", "domestic_continuous"]
    labels = ["Water curtain, all", "   no off-season use", "   off-season use", "Paddy irrigation",
              "Domestic continuous"]
    norm = Normalize(0, 0.6)
    cmap = plt.get_cmap("OrRd")
    records = []
    for i, pair in enumerate(("P1", "P2")):
        for j, lead in enumerate((10, 30)):
            ax = axes[i, j]
            c = conc[(conc["pair_id"] == PAIRS[pair]) & (conc["lead_day"] == lead)]
            v = vc[(vc["pair_id"] == PAIRS[pair]) & (vc["lead_day"] == lead)]
            for ri, rv in enumerate(rows):
                y = len(rows) - 1 - ri
                for m in range(1, 13):
                    if rv.startswith("wc_"):
                        r = v[(v["origin_month"] == m) & (v["variant"] == (rv == "wc_variant"))]
                        k, n = int(r["n_opposite"].sum()), int(r["n_determined"].sum())
                    else:
                        r = c[(c["calendar_type"] == rv) & (c["origin_month"] == m)]
                        k, n = int(r["strict_opposite_sign.numerator"].sum()), int(r["strict_opposite_sign.denominator"].sum())
                    records.append({"pair_id": PAIRS[pair], "lead_day": lead, "row": rv, "origin_month": m,
                                    "n_opposite": k, "n_determined": n, "rate": k / n if n else None})
                    if n == 0:
                        ax.add_patch(Rectangle((m - 1, y), 1, 1, facecolor="#e6e6e6", edgecolor="#555555",
                                               hatch="xxxx", linewidth=0.4))
                        continue
                    ax.add_patch(Rectangle((m - 1, y), 1, 1, facecolor=cmap(norm(k / n)), edgecolor="white",
                                           linewidth=0.6))
                    ax.text(m - 0.5, y + 0.5, f"{k}/{n}", ha="center", va="center", fontsize=6.6,
                            color="white" if k / n > 0.4 else "black")
            ax.axhline(2, color="black", linewidth=1.0)
            ax.set_xlim(0, 12)
            ax.set_ylim(0, len(rows))
            ax.set_xticks(np.arange(12) + 0.5)
            ax.set_xticklabels([mm[0] for mm in MONTHS])
            ax.set_yticks([len(rows) - 1 - k + 0.5 for k in range(len(rows))])
            ax.set_yticklabels(labels, fontsize=8.8)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{PAIR_LABEL[pair]}, {lead}-day lead", fontsize=10, fontweight="bold", loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015, extend="max")
    cb.set_label("Opposite-sign rate among direction-determined records", fontsize=9.5)
    fig.legend(handles=[Patch(facecolor="#e6e6e6", edgecolor="#555555", hatch="xxxx",
                              label="No direction-determined record in the cell"),
                        Patch(facecolor="white", edgecolor="white", label="Cell text: opposite-sign count / determined records")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("Where in the operating calendar the tool contradicts the record", fontsize=11.5,
                 fontweight="bold", y=0.985)
    stem = "figure4_calendar_concentration"
    save(fig, stem, products)
    path = T / f"{stem}_plotinput.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    plotinputs[stem] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


# ---------------------------------------------------------------- cross-check
def cross_check(rows, storage_cells, bs, edges):
    """Recount Figure 3/4 integers from primary_rows.csv and compare to the official tables."""
    r = rows.copy()
    r["sign_determined"] = ((r["env_inf_m"] > 0) | (r["env_sup_m"] < 0)).astype(int)
    r["env_sign"] = np.where(r["env_inf_m"] > 0, 1, np.where(r["env_sup_m"] < 0, -1, 0))
    e = r["E_true_m"]
    below = (e != 0) & e.notna() & (r["W_m"] < e.abs())
    r["below1"] = below.astype(int)
    b = r[r["calendar_type"].notna()].copy()
    b["origin_month"] = b["origin_month"].astype(int)
    b["no_active"] = (b["q_origin_m3d"] == 0).astype(int)
    mine = b.groupby(["origin_month", "calendar_type", "storage_type", "lead_day", "pair_id"]).agg(
        n=("case_id", "size"), det=("sign_determined", "sum"), below=("below1", "sum"),
        noact=("no_active", "sum")).reset_index()
    off = storage_cells.rename(columns={"denominator": "n", "n_sign_determined": "det",
                                        "n_relative_width_below_one": "below", "n_no_active_contrast": "noact"})
    m = mine.merge(off[["origin_month", "calendar_type", "storage_type", "lead_day", "pair_id", "n", "det", "below",
                        "noact"]], on=["origin_month", "calendar_type", "storage_type", "lead_day", "pair_id"],
                   suffixes=("_mine", "_official"), how="outer")
    cal_mismatch = int(sum((m[f"{k}_mine"] != m[f"{k}_official"]).sum() for k in ("n", "det", "below", "noact")))
    sr = r["SR"]
    idx = np.digitize(sr.where(sr > 0, np.nan).fillna(1.0), edges[1:-1], right=True)
    r["bin"] = np.where(sr > 0, idx.astype(str), "nonpositive_signal")
    d = r[r["sign_determined"] == 1]
    d = d.assign(opp=((d["E_tool_m"] * d["env_sign"]) < 0).astype(int),
                 refopp=((d["E_reference_m"] * d["env_sign"]) < 0).astype(int))
    mine4 = d.groupby(["bin", "storage_type", "pair_id", "lead_day"]).agg(
        opp=("opp", "sum"), det=("opp", "size"), refopp=("refopp", "sum")).reset_index()
    off4 = bs[["bin", "storage_type", "pair_id", "lead_day", "strict_opposite_sign.numerator",
               "strict_opposite_sign.denominator", "reference_contradiction.numerator"]].copy()
    off4 = off4[off4["strict_opposite_sign.denominator"] > 0]
    off4["bin"] = off4["bin"].astype(str)
    m4 = mine4.merge(off4, on=["bin", "storage_type", "pair_id", "lead_day"], how="outer")
    fig4_mismatch = int((m4["opp"] != m4["strict_opposite_sign.numerator"]).sum()
                        + (m4["det"] != m4["strict_opposite_sign.denominator"]).sum()
                        + (m4["refopp"] != m4["reference_contradiction.numerator"]).sum())
    return r, {"calendar_storage_cells_compared": int(len(m)), "calendar_count_mismatches": cal_mismatch,
               "bin_storage_cells_compared": int(len(m4)), "bin_storage_count_mismatches": fig4_mismatch,
               "rule": "sign determined iff env_inf>0 or env_sup<0; below1 iff E_true!=0 and W<|E_true|; "
                       "no active iff q_origin=0; bins by digitize(right=True) on official pooled edges"}


def main():
    stats_ready = json.loads((P / "STATS_READY.json").read_text())
    if sha(P / "statistics.json") != stats_ready["statistics_sha256"]:
        raise SystemExit("statistics.json hash differs from STATS_READY")
    if sha(P / "primary_rows.csv") != stats_ready["primary_rows_sha256"]:
        raise SystemExit("primary_rows.csv hash differs from STATS_READY")
    font = setup_font()
    stats = json.loads((P / "statistics.json").read_text())
    edges = stats["cohort_C"]["bins"]["edges"]
    storage_cells = pd.read_csv(T / "calendar_storage_cells.csv")
    cells = pd.read_csv(T / "calendar_cells.csv")
    variant_cells = pd.read_csv(T / "calendar_variant_cells.csv")
    bs = pd.read_csv(T / "contradiction_by_bin_storage.csv")
    conc = pd.read_csv(T / "calendar_monthly_contradiction_concentration.csv")
    src = pd.read_csv(T / "contradiction_by_source_signal_definition.csv")
    rows = pd.read_csv(P / "primary_rows.csv", low_memory=False)
    rows, check = cross_check(rows, storage_cells, bs, edges)
    products, plotinputs = {}, {}
    for pair in ("P1", "P2"):
        figure3(storage_cells, pair, products, plotinputs, active_only=False)
        figure3(storage_cells, pair, products, plotinputs, active_only=True)
        figure3_layer(cells, pair, products, plotinputs)
        figure3_variant(variant_cells, pair, products, plotinputs)
        figure4(bs, pair_rows(src, pair), pair, edges, products, plotinputs)
    vc = variant_contradiction(rows)
    figure4_calendar(conc, vc, products, plotinputs)
    inputs = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in {
        "statistics": P / "statistics.json", "primary_rows": P / "primary_rows.csv",
        "STATS_READY": P / "STATS_READY.json",
        "calendar_storage_cells": T / "calendar_storage_cells.csv", "calendar_cells": T / "calendar_cells.csv",
        "calendar_variant_cells": T / "calendar_variant_cells.csv",
        "contradiction_by_bin_storage": T / "contradiction_by_bin_storage.csv",
        "calendar_monthly_contradiction_concentration": T / "calendar_monthly_contradiction_concentration.csv",
        "contradiction_by_source_signal_definition": T / "contradiction_by_source_signal_definition.csv",
    }.items()}
    manifest = {"status": "rendered_pending_visual_inspection", "producer": str(Path(__file__).relative_to(ROOT)),
                "producer_sha256": sha(Path(__file__)), "non_scoring": True,
                "font": {"path": font, "sha256": sha(font), "family": "Times New Roman"},
                "bin_edges": edges, "inputs": inputs, "cross_check": check,
                "products": products, "plotinputs": plotinputs}
    (P / "FIGURE34_PRODUCTS.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cross_check": check, "n_products": len(products)}, indent=2))
    if check["calendar_count_mismatches"] or check["bin_storage_count_mismatches"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
