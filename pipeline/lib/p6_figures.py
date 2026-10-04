"""D07 figure revision (plotting only; no analysis recomputation, no new science numbers).

Reads unchanged D06 science tables/inputs and reuses the frozen p5_figures helpers (imported, never edited):
  * Figure 2 -> two panels: (a) sampled W against pumping scale, (b) AR(1) residual-recentred linearized width
    against sampled width (P1, 10-day lead). Data identical to the former Figure 2 panels a and e.
  * Figure S10 (former Figure 2 b, c): time-scale collapse scatter and bundle R2.
  * Figure S11 (former Figure 2 d, f): the other two linearized widths against sampled width.
  * Figure 5 -> same values as before; legend/axis names use descriptive record-set names.
  * Figure S4 -> redrawn from the saved summary results/phase4/figures/figure2_summary.csv (same medians/IQR
    the original image used) with descriptive set names instead of internal experiment labels.
Figures 1, 3, 4 and legacy S1-S3, S5-S9 contain no internal experiment name; their binaries are reused unchanged.
Internal experiment names never appear in any axis, legend or caption written here.

CLI: env/.venv_pilot/bin/python lib/p6_figures.py --out results/phase6/figures/render
"""
from __future__ import annotations

import os
for _n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_n, '1')
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / 'lib'))
import p5_figures as F  # noqa: E402  (frozen; helpers reused unchanged)
import p5_analysis as AN  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

P4 = ROOT / 'results/phase4'
P5 = ROOT / 'results/phase5'
AGG = P5 / 'aggregation'
INPUTS = AN.default_inputs()
SET_NAMES = {'D03': 'pause–transition set', 'D04A': 'count–recency set', 'D04B': 'pumping-scale set', 'D04C': 'storage set',
             'D05A': 'map-extension set', 'D05B': 'operating calendars at 200 m', 'D06A': 'operating calendars at 20 m'}
LEAD = 10


def _scatter_mode(ax, prim, mode, title, letter, ylab=True):
    ok = [r for r in prim if r['tolerance_mode'] == mode and r['status'] in AN.CERTIFIED and (AN._f(r['W_lin']) or 0) > 0 and (AN._f(r['W_sampled']) or 0) > 0]
    nq = sum(1 for r in prim if r['tolerance_mode'] == mode and AN._b(r.get('no_question')))
    other = sum(1 for r in prim if r['tolerance_mode'] == mode) - len(ok) - nq
    for st in F.STORAGES:
        pts = [r for r in ok if str(r.get('layer', '')).startswith(st) or str(r.get('layer', '')).startswith('fixedc_' + st)]
        if pts:
            ax.scatter([AN._f(r['W_sampled']) for r in pts], [AN._f(r['W_lin']) for r in pts], s=5, alpha=0.5, color=F.STORAGE_COLORS[st], lw=0)
    v = [AN._f(r['W_sampled']) for r in ok] + [AN._f(r['W_lin']) for r in ok]
    lo, hi = min(v) / 1.5, max(v) * 1.5
    ax.plot([lo, hi], [lo, hi], color='0.4', lw=0.7, ls=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Sampled W (m)')
    if ylab:
        ax.set_ylabel('Linearized W (m)')
    ax.text(0.03, 0.97, f'n = {len(ok)}\nno question = {nq}\nother status = {other}', transform=ax.transAxes, va='top', fontsize=8.5)
    F._label(ax, letter)
    return dict(mode=mode, n=len(ok), no_question=nq, other=other)


def _storage_handles():
    return [Line2D([], [], marker='o', ls='', color=c, label=s.capitalize()) for s, c in F.STORAGE_COLORS.items()]


def figure2(out, receipt):
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL_WIDTH_IN, 3.7))
    ax = axes[0]
    sc = [r for r in F._rows(AGG / 'D05_W_scale_table.csv') if r['pair_id'] == F.PAIRS[0] and int(r['lead_day']) == LEAD and r['layer'] != 'all']
    lay = sorted({r['layer'] for r in sc})
    for i, l in enumerate(lay):
        rs = sorted((r for r in sc if r['layer'] == l), key=lambda r: float(r['Q_scale']))
        x = [float(r['Q_scale']) * (1 + 0.04 * (i - len(lay) / 2)) for r in rs]
        y = [float(r['W_median_m']) for r in rs]
        err = [[yy - float(r['W_q25_m']) for yy, r in zip(y, rs)], [float(r['W_q75_m']) - yy for yy, r in zip(y, rs)]]
        ax.errorbar(x, y, yerr=err, marker='o' if l.endswith('T50') else 's', ms=4, lw=0.9, color=F.STORAGE_COLORS.get(l.split('_')[0], '0.3'), capsize=2)
    ax.set_xscale('log')
    ax.set_xticks([0.3, 1, 3])
    ax.set_xticklabels(['0.3', '1', '3'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('Pumping scale')
    ax.set_ylabel(f'W at {LEAD} d (m)')
    F._label(ax, 'a')
    prim = [r for r in F._rows(INPUTS['C_primary']) if r['pair'] == F.PAIRS[0] and int(float(r['lead'])) == LEAD]
    info = _scatter_mode(axes[1], prim, 'ar_locked', 'AR(1) metric, residual-recentred', 'b')
    h = _storage_handles() + [Line2D([], [], marker='o', ls='-', color='0.3', label='T = 50 m$^2$/d'), Line2D([], [], marker='s', ls='-', color='0.3', label='T = 500 m$^2$/d')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F._save(fig, out, 'figure2_ambiguity_linearized', False, receipt)
    receipt['figure2_ambiguity_linearized']['panel_b'] = info


def figureS10(out, receipt):
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL_WIDTH_IN, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    axc = axes[0]
    brows = [r for r in F._rows(INPUTS['B_rows']) if r['pair_id'] == F.PAIRS[0] and AN._b(r.get('mag_valid'))]
    for st in F.STORAGES:
        pts = [r for r in brows if st in r.get('sid', '')]
        if pts:
            axc.scatter([AN._f(r['Lk']) for r in pts], [AN._f(r['y_mag']) for r in pts], s=4, alpha=0.4, color=F.STORAGE_COLORS[st], lw=0)
    axc.axhline(0, color='0.5', lw=0.7)
    axc.set_xlabel('log(lead / t$_{95}$)')
    axc.set_ylabel('log(W / |E$_{true}$|)')
    F._label(axc, 'a')
    axd = axes[1]
    fits = [r for r in F._rows(AGG / 'B_bundle_fits.csv') if r['pair_id'] == F.PAIRS[0] and r['outcome'] == 'magnitude' and r['subset'] in ('full', 'D04C')]
    combos = [(s, lm) for s in ('full', 'D04C') for lm in ('pooled', '10', '30')]
    width = 0.26
    for k, b in enumerate(('i', 'ii', 'iii')):
        for j, (s, lm) in enumerate(combos):
            m = next((r for r in fits if r['subset'] == s and str(r['lead_mode']) == lm and r['bundle'] == b), None)
            y = AN._f(m['metric']) if m else None
            hatch = bool(m and m['interpretation_label'].startswith('fixed_lead_alias'))
            x = j + (k - 1) * width
            if y is None:
                axd.plot(x, 0.02, marker='x', color='0.3')
            else:
                axd.bar(x, y, width=width, color=('#cfcfcf', '#7f7f7f', '#2b2b2b')[k], hatch='//' if hatch else None, edgecolor='0.2', lw=0.4)
    axd.set_xticks(np.arange(len(combos)))
    axd.set_xticklabels([f'{"All" if s == "full" else "Storage"}\n{"pooled" if lm == "pooled" else lm + " d"}' for s, lm in combos], fontsize=8.5)
    axd.set_ylabel('R$^2$')
    axd.set_ylim(0, 1)
    F._label(axd, 'b')
    h2 = [Patch(facecolor=c, edgecolor='0.2', label=f'Bundle ({b})') for c, b in zip(('#cfcfcf', '#7f7f7f', '#2b2b2b'), ('i', 'ii', 'iii'))]
    h2.append(Patch(facecolor='white', edgecolor='0.2', hatch='//', label='Fixed-lead alias'))
    fig.legend(handles=_storage_handles() + h2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    F._save(fig, out, 'figureS10_timescale_collapse', False, receipt)


def figureS11(out, receipt):
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL_WIDTH_IN, 3.7))
    prim = [r for r in F._rows(INPUTS['C_primary']) if r['pair'] == F.PAIRS[0] and int(float(r['lead'])) == LEAD]
    i1 = _scatter_mode(axes[0], prim, 'sse_locked', 'Squared-error benchmark, saved tolerance', 'a')
    i2 = _scatter_mode(axes[1], prim, 'ar_prospective', 'AR(1) metric, conditional prospective', 'b', ylab=False)
    fig.legend(handles=_storage_handles(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F._save(fig, out, 'figureS11_linearized_other_metrics', False, receipt)
    receipt['figureS11_linearized_other_metrics']['panels'] = [i1, i2]


def figure5(out, receipt):
    d05 = F._d05_bin_rates(INPUTS['d05_stats'])
    r20 = [r for r in F._rows(AGG / 'A_contradictions.csv') if r['level'] == 'sr_bin_d05_fixed' and r['distance_m'] == '20']
    bins = ['0', '1', '2', '3', '4']
    xpos = {b: i for i, b in enumerate(bins + ['above_D05_range'])}
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL_WIDTH_IN, 3.4), sharey=True)
    for j, pair in enumerate(F.PAIRS):
        ax = axes[j]
        for lead, mk in ((10, 'o'), (30, 's')):
            for col, k in (('#b03a2e', 0), ('#1f4e79', 2)):
                xs, ys = [], []
                for b in bins:
                    v = d05.get((pair, lead, b))
                    if v is not None and v[k + 1] > 0:
                        xs.append(xpos[b]); ys.append(v[k] / v[k + 1])
                ax.plot(xs, ys, marker=mk, ms=4, color=col, lw=1.1, ls='-' if lead == 10 else '--', mfc=col if lead == 10 else 'white')
            pts = [(xpos.get(r['stratum']), AN._f(r['tool_opposite_rate_of_determined'])) for r in r20
                   if r['pair_id'] == pair and r['lead_day'] == str(lead) and r['stratum'] in xpos]
            pts = [p for p in pts if p[0] is not None and p[1] is not None]
            if pts:
                ax.scatter([p[0] + 0.15 for p in pts], [p[1] for p in pts], marker='*', s=55, facecolors='#e67e22' if lead == 10 else 'white',
                           edgecolors='#e67e22' if lead == 30 else 'k', lw=0.8, zorder=5)
        ax.set_xticks(range(len(xpos)))
        ax.set_xticklabels(['1', '2', '3', '4', '5', 'Above'])
        ax.set_xlabel('Signal-ratio bin (fixed edges)')
        if j == 0:
            ax.set_ylabel('Opposite-sign fraction')
        ax.set_title('Continue vs stop' if j == 0 else 'Current vs 1.5 times', fontsize=11)
        ax.set_ylim(bottom=0)
        F._label(ax, 'ab'[j])
    handles = [Line2D([], [], color='#b03a2e', marker='o', label='Scenario engine, design sets and 200 m calendars'),
               Line2D([], [], color='#1f4e79', marker='o', label='Structurally matched reference, same records'),
               Line2D([], [], color='0.3', ls='-', marker='o', label='Lead 10 d'), Line2D([], [], color='0.3', ls='--', marker='s', mfc='white', label='Lead 30 d'),
               Line2D([], [], marker='*', ls='', mfc='#e67e22', mec='k', ms=9, label='Scenario engine, 20 m calendars, 10 d'),
               Line2D([], [], marker='*', ls='', mfc='white', mec='#e67e22', ms=9, label='Scenario engine, 20 m calendars, 30 d')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.27), frameon=False, fontsize=9)
    fig.tight_layout()
    F._save(fig, out, 'figure5_record_contradictions', False, receipt)


def figureS4(out, receipt):
    rows = [r for r in F._rows(P4 / 'figures/figure2_summary.csv') if r['pair_id'] == F.PAIRS[0]]
    qcols = (('max_days_1_10', '', 'Days 1–10 maximum'), ('lead', '10', 'Day 10'), ('lead', '30', 'Day 30'))
    ncol = {'2': '#0072b2', '6': '#e69f00', '18': '#009e73'}
    fig = plt.figure(figsize=(F.FULL_WIDTH_IN, 11.0))
    gs = fig.add_gridspec(7, 3, height_ratios=[1, 1, 1, 0.42, 1, 1, 1], hspace=0.62, wspace=0.12, left=0.17, right=0.985, top=0.965, bottom=0.115)
    axes = np.empty((6, 3), dtype=object)
    for rr in range(6):
        g = rr if rr < 3 else rr + 1
        for cc in range(3):
            axes[rr, cc] = fig.add_subplot(gs[g, cc], sharey=axes[rr, 0] if cc else None)
            if cc:
                axes[rr, cc].tick_params(labelleft=False)
    for bi, (exp, xkey) in enumerate((('D03', 'rho'), ('D04', 'recency'))):
        for si, st in enumerate(F.STORAGES):
            for qi, (q, lead, title) in enumerate(qcols):
                ax = axes[bi * 3 + si, qi]
                sel = [r for r in rows if r['dataset'] == exp and r['storage_type'] == st and r['quantity'] == q and (r['lead_day'] or '') == lead]
                for T, mk, dx in (('50.0', 'o', -0.06), ('500.0', 's', 0.06)):
                    if exp == 'D03':
                        for N, c in ncol.items():
                            rs = sorted((r for r in sel if r['T_m2_d'] == T and r['N_transitions'] == N), key=lambda r: float(r['rho']))
                            x = np.log([float(r['rho']) for r in rs]) + dx
                            y = np.array([float(r['median_W_m']) for r in rs])
                            ax.errorbar(x, y, yerr=[y - [float(r['q25_W_m']) for r in rs], [float(r['q75_W_m']) for r in rs] - y], color=c, marker=mk, ms=3.5, lw=1, capsize=1.5)
                        ax.set_xticks(np.log([0.25, 1, 4]))
                        ax.set_xticklabels(['0.25', '1', '4'])
                    else:
                        for rec, ls, mfc, col in (('recent', '-', None, '0.1'), ('old', '--', 'white', '0.45')):
                            rs = sorted((r for r in sel if r['T_m2_d'] == T and r['recency_label'] == rec), key=lambda r: int(r['N_transitions']))
                            base = 0 if rec == 'recent' else 2.5
                            x = np.array([base + (0 if r['N_transitions'] == '2' else 1) for r in rs]) + dx
                            y = np.array([float(r['median_W_m']) for r in rs])
                            ax.errorbar(x, y, yerr=[y - [float(r['q25_W_m']) for r in rs], [float(r['q75_W_m']) for r in rs] - y], color=col, marker=mk, ms=3.5, lw=1,
                                        ls=ls, mfc=mfc or col, capsize=1.5)
                        ax.set_xticks([0, 1, 2.5, 3.5])
                        ax.set_xticklabels(['2\nrecent', '18\nrecent', '2\nold', '18\nold'], fontsize=8.5)
                ax.grid(axis='y', color='0.92', lw=0.6)
                if si == 0 and bi == 0:
                    ax.set_title(title, fontsize=11, fontweight='bold')
                if qi == 0:
                    ax.set_ylabel(f'{st.capitalize()}\nW (m)', fontsize=11)
                if si == 2:
                    ax.set_xlabel('Pause / response time, ρ' if exp == 'D03' else 'Transition count N', fontsize=11)
        top_ax, bot_ax = axes[bi * 3, 0].get_position(), axes[bi * 3 + 2, 0].get_position()
        fig.text(0.015, (top_ax.y1 + bot_ax.y0) / 2, ('Pause–transition set' if exp == 'D03' else 'Count–recency set'), rotation=90,
                 fontsize=12, fontweight='bold', ha='left', va='center')
    h = [Line2D([], [], color=c, label=f'Pause–transition set, N = {N}') for N, c in ncol.items()]
    h += [Line2D([], [], color='0.2', marker='o', ls='', label='T = 50 m$^2$/d'), Line2D([], [], color='0.2', marker='s', ls='', label='T = 500 m$^2$/d'),
          Line2D([], [], color='0.1', ls='-', marker='o', label='Count–recency set, recent'), Line2D([], [], color='0.45', ls='--', marker='o', mfc='white', label='Count–recency set, old')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.56, 0.0), frameon=False, fontsize=9.5)
    F._save(fig, out, 'figureS4_pause_transition_widths', False, receipt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'results/phase6/figures/render')
    a = ap.parse_args(argv)
    F.style()
    receipt = dict(font=F.font_preflight(require=True), figures={})
    for fn in (figure2, figure5, figureS4, figureS10, figureS11):
        fn(a.out, receipt['figures'])
    receipt['inputs'] = {k: dict(path=AN.rel(v), sha256=AN.sha(v)) for k, v in INPUTS.items() if Path(v).is_file()}
    receipt['extra_inputs'] = {p: AN.sha(ROOT / p) for p in ('results/phase4/figures/figure2_summary.csv', 'results/phase5/aggregation/TABLE_PROVENANCE.json')}
    (a.out / 'P6_RENDER_RECEIPT.json').write_text(json.dumps(receipt, indent=1, default=str) + '\n')
    print('P6_FIGURES_RENDERED', a.out)


if __name__ == '__main__':
    main()
