"""D06 five main quantitative figures (WRR full width, English, Times New Roman) from the p5_analysis tables.

Every plotted value is read from a file: aggregation tables (results/phase5/aggregation), official A/B/C rows,
frozen D05 rows/statistics and tf_input pumping arrays (inputs, not outcomes). Undefined/unavailable values are drawn
as explicit status marks (hatched "no question", "window absent", open markers) and never filled with zeros.
Captions live in FIGURE_CAPTIONS.md, not in the images. No model name appears in any figure.

  Figure 1  pumping-record timelines of representative records with the pause / transition / season-start windows,
            and the masked-to-full linearized width ratio over leads 1-30 (C mask information baseline).
  Figure 2  (a) sampled W(m) against pumping scale (D05 A), (b) relative ambiguity against log(lead/t95), (c) bundle R2
            (i)-(iii), (d-f) log W_lin against log sampled W: SSE benchmark, AR residual-recentred, AR conditional prospective.
  Figure 3  fixed D05 answerability map (P1, leads 10 and 30) with the frozen 0.5/0.9 contours, r = 20 m calendar
            overlays; points outside the D05 support are marked.
  Figure 4  operating calendars: active-origin determination fraction, r = 200 m beside r = 20 m (P1, leads 10/30);
            origins without pumping are hatched as "no question".
  Figure 5  record contradictions by fixed D05 signal bins for P1 and P2 (D05 tool and matched reference), with
            r = 20 m rates as separate markers.

CLI
  production: env/.venv_pilot/bin/python lib/p5_figures.py --production [--agg results/phase5/aggregation] [--out results/phase5/figures]
  fixture:    env/.venv_pilot/bin/python lib/p5_figures.py --fixture DIR    (watermarked; DIR under implementation_aggregation)
"""
from __future__ import annotations

import os
for _n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_n, '1')
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / 'lib'))
import p5_analysis as AN  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FULL_WIDTH_IN = 7.48  # WRR/AGU full width (190 mm)
PAIRS = AN.PAIRS
STORAGE_COLORS = {'confined': '#1b6ca8', 'leaky': '#e08e0b', 'unconfined': '#3a923a'}
FORMS = ('water_curtain', 'paddy_irrigation', 'domestic_continuous')
FORM_LABEL = {'water_curtain': 'Water curtain', 'paddy_irrigation': 'Paddy irrigation', 'domestic_continuous': 'Domestic'}
STORAGES = ('confined', 'leaky', 'unconfined')
WINDOW_STYLE = {'pause': ('#9e3d3d', 'Pause'), 'transition': ('#2f5d8a', 'Transition'), 'season_start': ('#5b8c3a', 'Season start')}
MODE_LABEL = {'sse_locked': 'SSE benchmark, saved tolerance', 'ar_locked': 'AR(1) metric, residual-recentred',
              'ar_prospective': 'AR(1) metric, conditional prospective'}
REPRESENTATIVES_FIG1 = (('phase2', 'r00_confined_T50_rho1_N6', 'Designed record'),
                        ('phase5', 'd06A_paddy_irrigation_m07_r00_confined_T50', 'Paddy, r = 20 m'),
                        ('phase5', 'd06A_water_curtain_m01_r00_confined_T50', 'Water curtain, r = 20 m'))


def font_preflight(require=True):
    try:
        path = font_manager.findfont('Times New Roman', fallback_to_default=False)
    except Exception as err:  # noqa: BLE001
        if require:
            raise RuntimeError('Times New Roman not resolvable: ' + repr(err))
        path = None
    bold = None
    try:
        bold = font_manager.findfont(font_manager.FontProperties(family='Times New Roman', weight='bold'), fallback_to_default=False)
    except Exception:  # noqa: BLE001
        pass
    if require and (path is None or 'Times New Roman' not in Path(path).name):
        raise RuntimeError(f'font resolved to {path}, not Times New Roman')
    return dict(regular=path, bold=bold, matplotlib=matplotlib.__version__)


def style():
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'axes.labelsize': 13, 'axes.titlesize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False, 'hatch.linewidth': 0.6})


def _label(ax, s):
    ax.text(-0.16, 1.03, s, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='left', va='bottom')


def _logclean(ax):
    from matplotlib.ticker import NullFormatter
    for a in (ax.xaxis, ax.yaxis):
        if a.get_scale() == 'log':
            a.set_minor_formatter(NullFormatter())


def _watermark(fig, fixture):
    if fixture:
        fig.text(0.5, 0.5, 'SYNTHETIC FIXTURE - NOT RESULTS', fontsize=34, color='red', alpha=0.25, ha='center', va='center',
                 rotation=25, fontweight='bold', zorder=100)


def _save(fig, out, name, fixture, receipt):
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        p = out / f'{name}.{ext}'
        fig.savefig(p, bbox_inches='tight', pad_inches=0.05)
        paths.append(p)
    w, h = fig.get_size_inches()
    plt.close(fig)
    receipt[name] = dict(files={AN.rel(p): AN.sha(p) for p in paths}, size_in=[float(w), float(h)], watermark_synthetic=bool(fixture),
                         visual_inspection='PENDING: open the rendered PNG at manuscript size; savefig is not inspection')


def _rows(path):
    return AN.read_csv(path)


def _tf_input_q(source_phase, case_id, root=ROOT):
    """Pumping context of a record (input). r20 calendar records reuse their r200 parent's byte-identical Q."""
    phase, cid = (('phase4', case_id.replace('d06A_', 'd05B_', 1)) if source_phase == 'phase5' else (source_phase, case_id))
    path = root / f'results/{phase}/cases/tf_inputs/{cid}.npz'
    with np.load(path, allow_pickle=False) as z:
        q = np.asarray(z['pumping_context'], float)
        dates = np.asarray(z['dates'])[:len(q)].astype('datetime64[D]')
    return q, dates, path


# ------------------------------------------------------------------ Figure 1
def figure1(agg, out, fixture, receipt, reps=REPRESENTATIVES_FIG1, span_days=400):
    curves = _rows(agg / 'C_mask_curves.csv')
    by = defaultdict(list)
    for r in curves:
        by[(r['source_phase'], r['case_id'])].append(r)
    fig, axes = plt.subplots(2, len(reps), figsize=(FULL_WIDTH_IN, 5.6), gridspec_kw=dict(height_ratios=[1, 1.15]))
    letters = 'abcdef'
    for j, (ph, cid, title) in enumerate(reps):
        q, dates, _ = _tf_input_q(ph, cid)
        n = len(q)
        lo = max(0, n - span_days)
        ax = axes[0, j]
        ax.step(np.arange(lo, n) - n, q[lo:], where='post', color='0.15', lw=0.8)
        rows = by.get((ph, cid), [])
        windows = {}
        for r in rows:
            if r['lead'] in ('1', 1) and r['pair'] == PAIRS[0] and r['mask_metric'] == 'ar_information':
                windows[r['window_kind']] = r
        for kind, (col, _lab) in WINDOW_STYLE.items():
            w = windows.get(kind)
            if w and w['window_status'] == 'present' and w['start'] not in ('', None):
                s, e = int(float(w['start'])), int(float(w['end']))
                ax.axvspan(s - n, e - n, color=col, alpha=0.28, lw=0)
        ax.set_xlim(lo - n, 0)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=10, loc='right')
        ax.set_xlabel('Day before origin' if j == 1 else '')
        if j == 0:
            ax.set_ylabel('Pumping (m$^3$/d)')
        _label(ax, letters[j])
        ax2 = axes[1, j]
        status_notes = []
        for kind, (col, lab) in WINDOW_STYLE.items():
            for metric, ls in (('ar_information', '-'), ('sse_information', '--')):
                pts = sorted(((int(float(r['lead'])), AN._f(r['ratio'])) for r in rows
                              if r['window_kind'] == kind and r['mask_metric'] == metric and r['pair'] == PAIRS[0]), key=lambda t: t[0])
                if not pts:
                    continue
                ld = [p[0] for p in pts if p[1] is not None]
                rv = [p[1] for p in pts if p[1] is not None]
                if rv:
                    ax2.plot(ld, rv, ls=ls, color=col, lw=1.4)
                elif metric == 'ar_information':
                    st = next((r['ratio_status'] for r in rows if r['window_kind'] == kind), 'undefined')
                    status_notes.append(f'{lab}: {st.replace("_", " ")}')
        ax2.axhline(1.0, color='0.6', lw=0.7)
        ax2.set_xlim(1, 30)
        ax2.set_xlabel('Lead (d)')
        if j == 0:
            ax2.set_ylabel('Masked / full width')
        if status_notes:
            ax2.text(0.98, 0.97, '\n'.join(status_notes), transform=ax2.transAxes, ha='right', va='top', fontsize=8.5, color='0.3')
        _label(ax2, letters[len(reps) + j])
    handles = [Patch(color=c, alpha=0.5, label=l) for c, l in WINDOW_STYLE.values()]
    handles += [Line2D([], [], color='0.2', ls='-', label='AR(1) information'), Line2D([], [], color='0.2', ls='--', label='SSE information')]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _watermark(fig, fixture)
    _save(fig, out, 'figure1_record_windows', fixture, receipt)


# ------------------------------------------------------------------ Figure 2
def figure2(agg, inputs, out, fixture, receipt, lead=10):
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    sc = [r for r in _rows(agg / 'D05_W_scale_table.csv') if r['pair_id'] == PAIRS[0] and int(r['lead_day']) == lead and r['layer'] != 'all']
    lay = sorted({r['layer'] for r in sc})
    for i, l in enumerate(lay):
        rs = sorted((r for r in sc if r['layer'] == l), key=lambda r: float(r['Q_scale']))
        x = [float(r['Q_scale']) * (1 + 0.04 * (i - len(lay) / 2)) for r in rs]
        y = [float(r['W_median_m']) for r in rs]
        err = [[yy - float(r['W_q25_m']) for yy, r in zip(y, rs)], [float(r['W_q75_m']) - yy for yy, r in zip(y, rs)]]
        st = l.split('_')[0]
        ax.errorbar(x, y, yerr=err, marker='o' if l.endswith('T50') else 's', ms=4, lw=0.9, color=STORAGE_COLORS.get(st, '0.3'), capsize=2)
    ax.set_xscale('log')
    ax.set_xticks([0.3, 1, 3])
    ax.set_xticklabels(['0.3', '1', '3'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Pumping scale')
    ax.set_ylabel(f'W at {lead} d (m)')
    _label(ax, 'a')
    prim = [r for r in _rows(inputs['C_primary']) if r['pair'] == PAIRS[0] and int(float(r['lead'])) == lead]
    for j, mode in enumerate(('sse_locked', 'ar_locked', 'ar_prospective')):
        axm = fig.add_subplot(gs[1, j])
        ok = [r for r in prim if r['tolerance_mode'] == mode and r['status'] in AN.CERTIFIED and (AN._f(r['W_lin']) or 0) > 0 and (AN._f(r['W_sampled']) or 0) > 0]
        nq = sum(1 for r in prim if r['tolerance_mode'] == mode and AN._b(r.get('no_question')))
        other = sum(1 for r in prim if r['tolerance_mode'] == mode) - len(ok) - nq
        for st in STORAGES:
            pts = [r for r in ok if str(r.get('layer', '')).startswith(st) or str(r.get('layer', '')).startswith('fixedc_' + st)]
            if pts:
                axm.scatter([AN._f(r['W_sampled']) for r in pts], [AN._f(r['W_lin']) for r in pts], s=5, alpha=0.5, color=STORAGE_COLORS[st], lw=0)
        if ok:
            v = [AN._f(r['W_sampled']) for r in ok] + [AN._f(r['W_lin']) for r in ok]
            lo, hi = min(v) / 1.5, max(v) * 1.5
            axm.plot([lo, hi], [lo, hi], color='0.4', lw=0.7, ls=':')
            axm.set_xlim(lo, hi)
            axm.set_ylim(lo, hi)
        axm.set_xscale('log')
        axm.set_yscale('log')
        _logclean(axm)
        axm.set_title(MODE_LABEL[mode].replace(', ', ',\n'), fontsize=9.5)
        axm.set_xlabel('Sampled W (m)')
        if j == 0:
            axm.set_ylabel('Linearized W (m)')
        axm.text(0.03, 0.97, f'n = {len(ok)}\nno question = {nq}\nother status = {other}', transform=axm.transAxes, va='top', fontsize=8)
        _label(axm, 'def'[j])
    axc = fig.add_subplot(gs[0, 1])
    brows = [r for r in _rows(inputs['B_rows']) if r['pair_id'] == PAIRS[0] and AN._b(r.get('mag_valid'))]
    stor = {}
    for st in STORAGES:
        pts = [r for r in brows if st in r.get('sid', '')]
        stor[st] = len(pts)
        if pts:
            axc.scatter([AN._f(r['Lk']) for r in pts], [AN._f(r['y_mag']) for r in pts], s=4, alpha=0.4, color=STORAGE_COLORS[st], lw=0)
    axc.axhline(0, color='0.5', lw=0.7)
    axc.set_xlabel('log(lead / t$_{95}$)')
    axc.set_ylabel('log(W / |E$_{true}$|)')
    _label(axc, 'b')
    axd = fig.add_subplot(gs[0, 2])
    fits = [r for r in _rows(agg / 'B_bundle_fits.csv') if r['pair_id'] == PAIRS[0] and r['outcome'] == 'magnitude' and r['subset'] in ('full', 'D04C')]
    combos = [(s, lm) for s in ('full', 'D04C') for lm in ('pooled', '10', '30')]
    width = 0.26
    for k, b in enumerate(('i', 'ii', 'iii')):
        ys, hatch = [], []
        for s, lm in combos:
            m = next((r for r in fits if r['subset'] == s and str(r['lead_mode']) == lm and r['bundle'] == b), None)
            ys.append(AN._f(m['metric']) if m else None)
            hatch.append(bool(m and m['interpretation_label'].startswith('fixed_lead_alias')))
        xs = np.arange(len(combos)) + (k - 1) * width
        for x, y, h in zip(xs, ys, hatch):
            if y is None:
                axd.plot(x, 0.02, marker='x', color='0.3')
            else:
                axd.bar(x, y, width=width, color=('#cfcfcf', '#7f7f7f', '#2b2b2b')[k], hatch='//' if h else None, edgecolor='0.2', lw=0.4)
    axd.set_xticks(np.arange(len(combos)))
    axd.set_xticklabels([f'{"All" if s == "full" else "C"}\n{"pool" if lm == "pooled" else lm}' for s, lm in combos], fontsize=8.5)
    axd.set_ylabel('R$^2$')
    axd.set_ylim(0, 1)
    _label(axd, 'c')
    h1 = [Line2D([], [], marker='o', ls='', color=c, label=s.capitalize()) for s, c in STORAGE_COLORS.items()]
    h2 = [Patch(facecolor=c, edgecolor='0.2', label=f'Bundle ({b})') for c, b in zip(('#cfcfcf', '#7f7f7f', '#2b2b2b'), ('i', 'ii', 'iii'))]
    h2.append(Patch(facecolor='white', edgecolor='0.2', hatch='//', label='Fixed-lead alias'))
    fig.legend(handles=h1 + h2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07), frameon=False)
    _watermark(fig, fixture)
    _save(fig, out, 'figure2_ambiguity_timescale', fixture, receipt)


# ------------------------------------------------------------------ Figure 3
def _d05_map_points(d05_rows_path, lead, pair):
    import p4_contracts
    keys = {(r['source_phase'], r['case_id']) for r in p4_contracts.cohort_manifests()['map']}
    pts = []
    for r in AN.read_csv(d05_rows_path):
        if (r['source_phase'], r['case_id']) in keys and r['pair_id'] == pair and int(float(r['lead_day'])) == lead:
            sr, rho = AN._f(r['SR']), AN._f(r['rho'])
            inf, sup = AN._f(r['env_inf_m']), AN._f(r['env_sup_m'])
            if sr and sr > 0 and rho is not None and inf is not None and sup is not None:
                pts.append((sr, rho + 0.05, inf > 0 or sup < 0))
    return pts


def figure3(agg, inputs, out, fixture, receipt):
    import p5_collect as C
    maps = C.d05_map(inputs['d05_stats'])
    a = AN.read_csv(inputs['A_rows'])
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.6), sharey=True)
    for j, lead in enumerate((10, 30)):
        ax = axes[j]
        bg = _d05_map_points(inputs['d05_rows'], lead, PAIRS[0])
        for det, mk in ((True, 'o'), (False, 'o')):
            p = [(x, y) for x, y, d in bg if d == det]
            if p:
                ax.scatter(*zip(*p), s=6, marker=mk, facecolors='0.65' if det else 'none', edgecolors='0.65', lw=0.4)
        c = maps['sign', lead, PAIRS[0]]
        yy = np.geomspace(0.05, 4.05, 200)
        if c['status'] == 'ok':
            k = c['coefficients']
            for prob, ls in ((0.5, '-'), (0.9, '--')):
                xs = np.exp((math.log(prob / (1 - prob)) - k['intercept'] - k['log_rho_plus_offset'] * np.log(yy)) / k['log_signal_ratio'])
                ax.plot(xs, yy, color='k', ls=ls, lw=1.1)
        rr = [r for r in a if r['pair_id'] == PAIRS[0] and int(float(r['lead_day'])) == lead and r.get('transition') != 'no_active_contrast']
        for r in rr:
            sr, rho, det, ins = AN._f(r['SR_origin']), AN._f(r['rho_last_off']), AN._i(r['det_r20']), AN._b(r.get('inside_d05_support'))
            if not sr or sr <= 0 or rho is None or det is None:
                continue
            mk = 'D' if ins is not False else 'X'
            ax.scatter(sr, rho + 0.05, s=12, marker=mk, facecolors='#c0392b' if det else 'none', edgecolors='#c0392b', lw=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        _logclean(ax)
        ax.set_xlabel('Signal ratio')
        if j == 0:
            ax.set_ylabel('Rest ratio + 0.05')
        ax.set_title(f'Lead {lead} d', fontsize=11)
        _label(ax, 'ab'[j])
    handles = [Line2D([], [], marker='o', ls='', mfc='0.65', mec='0.65', label='Map records, direction determined'),
               Line2D([], [], marker='o', ls='', mfc='none', mec='0.65', label='Map records, undetermined'),
               Line2D([], [], color='k', label='p = 0.5'), Line2D([], [], color='k', ls='--', label='p = 0.9'),
               Line2D([], [], marker='D', ls='', mfc='#c0392b', mec='#c0392b', label='r = 20 m calendar, determined'),
               Line2D([], [], marker='D', ls='', mfc='none', mec='#c0392b', label='r = 20 m calendar, undetermined'),
               Line2D([], [], marker='X', ls='', mfc='#c0392b', mec='#c0392b', label='Outside map support')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9)
    fig.tight_layout()
    _watermark(fig, fixture)
    _save(fig, out, 'figure3_answerability_map_r20', fixture, receipt)


# ------------------------------------------------------------------ Figure 4
def figure4(agg, out, fixture, receipt):
    det = [r for r in _rows(agg / 'A_determination.csv') if r['pair_id'] == PAIRS[0] and r['level'] == 'form_month_storage']
    idx = {(r['lead_day'], r['form'], r['origin_month'], r['storage_type']): r for r in det}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH_IN, 6.0), sharex=True, sharey=True)
    cmap = plt.get_cmap('viridis')
    rows_lab = [(f, s) for f in FORMS for s in STORAGES]
    im = None
    for i, lead in enumerate((10, 30)):
        for j, (dist, col) in enumerate(((200, 'frac_determined_r200_active'), (20, 'frac_determined_r20_active'))):
            ax = axes[i, j]
            M = np.full((len(rows_lab), 12), np.nan)
            noq = np.zeros_like(M, bool)
            for a_, (f, s) in enumerate(rows_lab):
                for m in range(1, 13):
                    r = idx.get((str(lead), f, str(m), s))
                    if r is None:
                        continue
                    if int(r['n_active']) == 0:
                        noq[a_, m - 1] = True
                    else:
                        M[a_, m - 1] = AN._f(r[col]) if AN._f(r[col]) is not None else np.nan
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
            for a_, m in zip(*np.where(noq)):
                ax.add_patch(plt.Rectangle((m - 0.5, a_ - 0.5), 1, 1, facecolor='0.92', edgecolor='0.6', hatch='///', lw=0.3))
            for b in (2.5, 5.5):
                ax.axhline(b, color='white', lw=1.5)
            ax.set_xticks(range(12))
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax.set_yticks(range(len(rows_lab)))
            ax.set_yticklabels([f'{FORM_LABEL[f]}, {s}' if s == 'leaky' else s for f, s in rows_lab], fontsize=8.5)
            ax.set_title(f'r = {dist} m, lead {lead} d', fontsize=11)
            if i == 1:
                ax.set_xlabel('Origin month')
            _label(ax, 'abcd'[2 * i + j])
    fig.tight_layout(rect=(0, 0.06, 0.9, 1))
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Direction determined (active origins)', fontweight='bold', fontsize=11)
    fig.legend(handles=[Patch(facecolor='0.92', edgecolor='0.6', hatch='///', label='No pumping at origin (no question)')], loc='lower center',
               bbox_to_anchor=(0.45, 0.0), frameon=False)
    _watermark(fig, fixture)
    _save(fig, out, 'figure4_operating_calendar_r200_r20', fixture, receipt)


# ------------------------------------------------------------------ Figure 5
def _d05_bin_rates(stats_path):
    t = json.loads(Path(stats_path).read_text())['cohort_C']['table']['by_bin_storage']
    acc = defaultdict(lambda: np.zeros(4))
    for r in t:
        b = r['bin']
        if b == 'nonpositive_signal':
            continue
        acc[(r['pair_id'], int(r['lead_day']), str(b))] += [r['tool_contradiction']['numerator'], r['tool_contradiction']['denominator'],
                                                             r['reference_contradiction']['numerator'], r['reference_contradiction']['denominator']]
    return acc


def figure5(agg, inputs, out, fixture, receipt):
    d05 = _d05_bin_rates(inputs['d05_stats'])
    r20 = [r for r in _rows(agg / 'A_contradictions.csv') if r['level'] == 'sr_bin_d05_fixed' and r['distance_m'] == '20']
    bins = ['0', '1', '2', '3', '4']
    xpos = {b: i for i, b in enumerate(bins + ['above_D05_range'])}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.4), sharey=True)
    for j, pair in enumerate(PAIRS):
        ax = axes[j]
        for lead, mk in ((10, 'o'), (30, 's')):
            for tag, col, k in (('tool', '#b03a2e', 0), ('reference', '#1f4e79', 2)):
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
        ax.set_xticklabels(['1', '2', '3', '4', '5', '>D05'])
        ax.set_xlabel('Signal-ratio bin (fixed D05 edges)')
        if j == 0:
            ax.set_ylabel('Opposite-sign fraction')
        ax.set_title('Continue vs stop' if j == 0 else 'Current vs 1.5 times', fontsize=11)
        ax.set_ylim(bottom=0)
        _label(ax, 'ab'[j])
    handles = [Line2D([], [], color='#b03a2e', marker='o', label='Scenario engine, D05 records'),
               Line2D([], [], color='#1f4e79', marker='o', label='Structurally matched reference, D05'),
               Line2D([], [], color='0.3', ls='-', marker='o', label='Lead 10 d'), Line2D([], [], color='0.3', ls='--', marker='s', mfc='white', label='Lead 30 d'),
               Line2D([], [], marker='*', ls='', color='#e67e22', mec='k', label='Scenario engine, r = 20 m calendars')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9)
    fig.tight_layout()
    _watermark(fig, fixture)
    _save(fig, out, 'figure5_record_contradictions', fixture, receipt)


CAPTIONS = """# D06 main figure captions (drafts; numbers belong in the manuscript text with numbers_sources.md lookups)

**Figure 1.** Pumping records of three representative records with the pause, transition and season-start windows marked (a-c), and the ratio of the masked to the full linearized width of the continue-versus-stop effect over leads 1-30 when the head observations inside each window are removed (d-f). Solid lines use the AR(1) noise information and dashed lines the unweighted squared-error information, both with the full-record budget held fixed. Windows that do not exist in a record are stated in the panel.

**Figure 2.** (a) Width W of the record ambiguity at a 10-day lead against pumping scale for the six layers (median and interquartile range). (b) Relative ambiguity against lead divided by the response time t95. (c) Explained variance of the three covariate bundles; hatched bars mark the fixed-lead alias in which bundle (ii) spans the same columns as bundle (i). (d-f) Linearized width against sampled width for the saved squared-error benchmark, the AR(1) metric recentred on the record residual, and the AR(1) metric computed without the sampled width.

**Figure 3.** Answerability map of the continue-versus-stop question from the frozen D05 fit at leads of 10 and 30 days, with the 0.5 and 0.9 contours. Grey circles are the map records. Red diamonds are the operating-calendar records at 20 m projected on the same coordinates; crosses lie outside the support of the D05 records.

**Figure 4.** Fraction of active origins whose record determines the direction of the continue-versus-stop effect, by origin month, operating calendar and storage type, at 200 m (left) and 20 m (right) and leads of 10 (top) and 30 days (bottom). Hatched cells have no pumping on the origin day and therefore no question.

**Figure 5.** Fraction of direction-determined records for which the scenario engine gives the opposite sign, by signal-ratio bin, for the continue-versus-stop (a) and the 1.5-times (b) questions. Lines show the D05 records and the structurally matched reference on the same rows; stars show the 20 m calendar records.
"""


def run(agg, inputs, out, fixture):
    style()
    receipt = dict(mode='SYNTHETIC_FIXTURE_NOT_RESULTS' if fixture else 'production', font=font_preflight(require=True),
                   rc=dict(axes_labelsize=13, tick_labelsize=10, labelweight='bold', width_in=FULL_WIDTH_IN, dpi=300), figures={})
    figure1(agg, out, fixture, receipt['figures'])
    figure2(agg, inputs, out, fixture, receipt['figures'])
    figure3(agg, inputs, out, fixture, receipt['figures'])
    figure4(agg, out, fixture, receipt['figures'])
    figure5(agg, inputs, out, fixture, receipt['figures'])
    (out / 'FIGURE_CAPTIONS.md').write_text(CAPTIONS)
    receipt['captions'] = AN.rel(out / 'FIGURE_CAPTIONS.md')
    receipt['inputs'] = {k: dict(path=AN.rel(v), sha256=AN.sha(v)) for k, v in inputs.items() if Path(v).is_file()}
    receipt['aggregation_provenance'] = AN.rel(agg / 'TABLE_PROVENANCE.json')
    (out / 'FIGURE_RENDER_RECEIPT.json').write_text(json.dumps(receipt, indent=1, default=str) + '\n')
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--production', action='store_true')
    g.add_argument('--fixture', type=Path)
    ap.add_argument('--agg', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    if a.production:
        import p5_contracts as K
        K.require_frozen(entries=K.RUNTIME_ENTRIES + AN.OWN_ENTRIES)
        agg = a.agg or AN.P5 / 'aggregation'
        prov = json.loads((agg / 'TABLE_PROVENANCE.json').read_text())
        if prov.get('mode') != 'production':
            raise PermissionError('aggregation tables are not production outputs')
        run(agg, AN.default_inputs(), a.out or AN.P5 / 'figures', fixture=False)
    else:
        fx = a.fixture.resolve()
        if AN.FIXTURE_ROOT.resolve() not in fx.parents:
            raise PermissionError('fixture directory must be under results/phase5/implementation_aggregation')
        spec = json.loads((fx / 'FIXTURE_SPEC.json').read_text())
        inputs = {k: fx / v for k, v in spec['inputs'].items()}
        run(a.agg or fx / 'aggregation', inputs, a.out or fx / 'figures', fixture=True)
    print('FIGURES_RENDERED')


if __name__ == '__main__':
    main()
