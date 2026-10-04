"""Main Figures 1-6 and Supporting Information Figure S19 from the archived figure data.

The plotting logic follows the code that produced the submitted figures; only
data loading is simplified. No fit, bootstrap, forecast or simulation is run.
Figure S19 is the former Figure 2 (W against pumping scale; linearized width),
moved to the Supporting Information before submission.
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import dates as mdates, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import DATA, INCREASE, LEADS, QUESTIONS, STOP, detected, letter, num, read_csv, read_json

STORAGES = ('confined', 'leaky', 'unconfined')
STORAGE_COLORS = {'confined': '#1b6ca8', 'leaky': '#e08e0b', 'unconfined': '#3a923a'}
FORMS = ('water_curtain', 'paddy_irrigation', 'domestic_continuous')
FORM_LABEL = {'water_curtain': 'Water curtain', 'paddy_irrigation': 'Paddy irrigation', 'domestic_continuous': 'Domestic'}
CERTIFIED = ('certified_interior', 'certified_constrained', 'bounded_by_family_only', 'zero_contrast', 'no_question')
_CASES = None


def cases():
    global _CASES
    if _CASES is None:
        _CASES = read_csv(DATA / 'cases/case_level_summary.csv')
    return _CASES


# ---------------------------------------------------------------- Figure 1
def figure1():
    ctx = 1024
    fig, axs = plt.subplots(3, 2, figsize=(13.2, 10.8), gridspec_kw={'height_ratios': [.8, 1.2, 1.]})
    fig.subplots_adjust(left=.08, right=.98, top=.94, bottom=.14, hspace=.72, wspace=.26)
    facts = []
    with np.load(DATA / 'figure1/figure1_candidate_curves.npz', allow_pickle=False) as z:
        sides = {side: {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(side + '__')} for side in ('left', 'right')}
    for col, side in enumerate(('left', 'right')):
        d = sides[side]
        dates = mdates.datestr2num(d['dates'].tolist()); past_dates = dates[624:ctx]; future_dates = dates[ctx:]
        qax, hax, eax = axs[:, col]
        qax.plot(past_dates, d['past_Q'][624:ctx], color='#354e67', lw=1.1, drawstyle='steps-post')
        qax.set_ylim(-5, 115); qax.set_ylabel('Pumping rate (m³/d)'); qax.set_xlabel('Calendar date')
        qax.set_title('20 m, confined aquifer: effect detected' if col == 0 else '200 m, unconfined aquifer: zero effect fits', pad=17)
        for ax in (qax, hax):
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.set_xlim(past_dates[0], future_dates[-1] if ax is hax else past_dates[-1])
        colors = plt.get_cmap('viridis')(np.linspace(.15, .85, len(d['candidate_id'])))
        null = set(d['null_indices'].tolist())
        for i in range(len(d['candidate_id'])):
            zero = i in null
            color = '#ad4d24' if zero else colors[i]
            # past candidate curves: orange zero-effect model, other acceptable candidates pale grey
            hax.plot(past_dates, d['past_head'][i, 624:ctx], color=color if zero else '#989898', alpha=.8 if zero else .23, lw=1.6 if zero else .65, zorder=3 if zero else 1)
            x = np.r_[dates[ctx - 1], future_dates]
            hax.plot(x, np.r_[d['past_head'][i, -1], d['head_continue'][i]], color=color, lw=1.7 if zero else .85, alpha=.95 if zero else .55)
            hax.plot(x, np.r_[d['past_head'][i, -1], d['head_stop'][i]], color=color, ls='--', lw=1.7 if zero else .85, alpha=.95 if zero else .55)
        hax.scatter(past_dates, d['observed_head'][624:ctx], s=3, color='#202020', alpha=.55, zorder=4, linewidths=0)
        hax.axvline(future_dates[0], color='#777777', ls=':', lw=.9)
        hax.axvspan(future_dates[0], future_dates[-1], color='#dddddd', alpha=.2, zorder=0)
        hax.text(.99, 1.035, 'Decision: 1 Jan 2010; future: 30 d', ha='right', va='bottom', transform=hax.transAxes, fontsize=10)
        hax.set_ylabel('Head (m)'); hax.set_xlabel('Calendar date')
        k = np.arange(1, 31)
        eax.fill_between(k, d['saved_inf'], d['saved_sup'], color='#809bbd', alpha=.3, zorder=1)
        for i in range(len(d['candidate_id'])):
            zero = i in null
            eax.plot(k, d['E'][i], color='#ad4d24' if zero else colors[i], lw=1.8 if zero else .85, alpha=.95 if zero else .6, zorder=4 if zero else 2)
        eax.plot(k, d['true_E'], color='black', lw=2.5, zorder=5)
        eax.annotate('', xy=(10, d['saved_sup'][9]), xytext=(10, d['saved_inf'][9]), arrowprops={'arrowstyle': '|-|', 'lw': 1.3, 'color': '#162b4b'}, zorder=7)
        width, truth, lo, hi = (float(d[n][9]) for n in ('saved_W', 'true_E', 'saved_inf', 'saved_sup'))
        relative = width / abs(truth); zero_status = 'zero included' if lo <= 0 <= hi else 'zero excluded'
        eax.text(.045, 1.035, rf'$W/|E_{{\mathrm{{true}}}}|$ (10 d) = {relative:.3f}', transform=eax.transAxes, fontsize=11, color='#162b4b', ha='left', va='bottom', zorder=8)
        eax.text(.98, 1.035, zero_status, transform=eax.transAxes, ha='right', va='bottom', fontsize=10, color='#ad4d24' if null else '#162b4b')
        eax.set_xlim(1, 30); eax.set_xticks([1, 10, 20, 30]); eax.set_ylabel('Head difference E (m)'); eax.set_xlabel('Days after the decision (d)')
        eax.margins(y=.13)
        for row, ax in enumerate((qax, hax, eax)):
            ax.text(-.06, 1.04, f'({chr(97 + row * 2 + col)})', transform=ax.transAxes, fontweight='bold', fontsize=13, va='bottom')
        facts.append(dict(panel='ef'[col], candidates=len(d['candidate_id']), W10_m=width, relative_width_10d=relative, zero_status=zero_status))
    legend = [Line2D([], [], marker='.', linestyle='', color='#202020', label='Observed head'),
              Line2D([], [], color='#989898', alpha=.5, label='Accepted candidates'),
              Line2D([], [], color='#354e67', label='Continue pumping'), Line2D([], [], color='#354e67', ls='--', label='Stop pumping'),
              Line2D([], [], color='black', lw=2.5, label='True effect'), Patch(facecolor='#809bbd', alpha=.3, label='Spread estimated by sampling'),
              Line2D([], [], color='#ad4d24', label='Accepted zero-effect model')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.53, .015), ncol=4, frameon=False, columnspacing=1.7, handlelength=2.1)
    return fig, facts


# ---------------------------------------------------------------- Figure 2
def figure2():
    rows = read_csv(DATA / 'masks/mask_curves.csv')
    pump = {}
    for r in read_csv(DATA / 'masks/representative_pumping.csv'):
        pump.setdefault(r['record'], []).append(float(r['pumping_m3_per_day']))
    reps = [('phase2', 'r00_confined_T50_rho1_N6', 'designed_confined_T50_rho1_N6', 'Designed record'),
            ('phase5', 'd06A_paddy_irrigation_m07_r00_confined_T50', 'paddy_m07_r00_confined_T50', 'Paddy, r = 20 m'),
            ('phase5', 'd06A_water_curtain_m01_r00_confined_T50', 'water_curtain_m01_r00_confined_T50', 'Water curtain, r = 20 m')]
    windows = {'pause': ('#9e3d3d', 'Pause'), 'transition': ('#2f5d8a', 'Transition'), 'season_start': ('#5b8c3a', 'Season start')}
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), gridspec_kw={'height_ratios': [1, 1.15]})
    facts = []
    for j, (stage, cid, name, title) in enumerate(reps):
        q = np.asarray(pump[name]); n = len(q); lo = max(0, n - 400); ax = axes[0, j]
        ax.step(np.arange(lo, n) - n, q[lo:], where='post', color='0.15', lw=.8)
        rs = [r for r in rows if (r['source_phase'], r['case_id']) == (stage, cid)]
        wrows = {r['window_kind']: r for r in rs if r['lead'] == '1' and r['pair'] == STOP and r['mask_metric'] == 'ar_information'}
        for kind, (col, _) in windows.items():
            w = wrows.get(kind)
            if w and w['window_status'] == 'present' and w['start'] not in ('', None):
                ax.axvspan(int(float(w['start'])) - n, int(float(w['end'])) - n, color=col, alpha=.28, lw=0)
        ax.set_xlim(lo - n, 0); ax.set_ylim(bottom=0); ax.set_title(title, fontsize=10, loc='right')
        ax.set_xlabel('Days before the decision' if j == 1 else '')
        if j == 0:
            ax.set_ylabel('Pumping (m$^3$/d)')
        ax.text(-.16, 1.03, 'abcdef'[j], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
        ax2 = axes[1, j]; notes = []
        for kind, (col, lab) in windows.items():
            for metric, ls in [('ar_information', '-'), ('sse_information', '--')]:
                pts = sorted([(int(float(r['lead'])), num(r['ratio'])) for r in rs if r['window_kind'] == kind and r['mask_metric'] == metric and r['pair'] == STOP])
                valid = [t for t in pts if t[1] is not None]
                if valid:
                    ax2.plot(*zip(*valid), ls=ls, color=col, lw=1.4)
                elif pts and metric == 'ar_information':
                    status = next((r['ratio_status'] for r in rs if r['window_kind'] == kind), 'undefined')
                    notes.append(lab + ': ' + status.replace('_', ' '))
        ax2.axhline(1.0, color='0.6', lw=.7); ax2.set_xlim(1, 30); ax2.set_xlabel('Lead (d)')
        if j == 0:
            ax2.set_ylabel('Masked / full width')
        if notes:
            ax2.text(.98, .97, '\n'.join(notes), transform=ax2.transAxes, ha='right', va='top', fontsize=8.5, color='0.3')
        ax2.text(-.16, 1.03, 'abcdef'[j + 3], transform=ax2.transAxes, fontsize=13, fontweight='bold', va='bottom')
        facts.append(dict(record=name, mask_rows=len(rs)))
    handles = [Patch(color=c, alpha=.5, label=l) for c, l in windows.values()] + [
        Line2D([], [], color='.2', ls='-', label='Correlated noise'), Line2D([], [], color='.2', ls='--', label='Unweighted squared error')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, .015), frameon=False)
    fig.tight_layout(rect=(0, .15, 1, 1))
    return fig, facts


# ---------------------------------------------------------------- Figure 3
def figure3():
    det = read_json(DATA / 'detection/detection_map_models.json')['primary']['no_layer']
    k = det['coefficients']
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.7), sharey=True)
    fig.subplots_adjust(left=.1, right=.98, bottom=.32, top=.9, wspace=.25)
    facts = []
    for j, lead in enumerate(LEADS):
        ax = axes[j]
        rs = [r for r in cases() if r['is_map'] == 'True' and r['pair_id'] == STOP and int(r['lead_day']) == lead]
        for flag in (True, False):
            pts = [r for r in rs if detected(r) == flag and num(r['SR']) and num(r['SR']) > 0]
            ax.scatter([num(r['SR']) for r in pts], [num(r['rho']) + .05 for r in pts], s=7, facecolor='.6' if flag else 'none', edgecolor='.6', lw=.4)
        yy = np.geomspace(.05, 4.05, 200)
        for prob, ls in [(.5, '-'), (.9, '--')]:
            xs = np.exp((math.log(prob / (1 - prob)) - k['intercept'] - k['log_rho_plus_offset'] * np.log(yy)) / k['log_signal_ratio'])
            ax.plot(xs, yy, 'k', ls=ls, lw=1.2)
        cal = [r for r in cases() if r['variant'] == 'original' and r['record_set'] == 'calendar20m' and r['pair_id'] == STOP
               and int(r['lead_day']) == lead and r['active'] == 'True']
        for r in cal:
            sr, rho = num(r['SR_origin']), num(r['rho_last_off'])
            if not sr or sr <= 0 or rho is None:
                continue
            mk = 'D' if r['inside_d05_support'] != 'False' else 'X'
            ax.scatter(sr, rho + .05, s=14, marker=mk, facecolors='#c0392b' if detected(r) else 'none', edgecolors='#c0392b', lw=.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Signal ratio'); ax.set_title(f"({'ab'[j]}) Lead {lead} d", loc='left')
        facts.append(dict(lead=lead, map_records=len(rs), map_detected=sum(detected(r) for r in rs), calendar20m_active=len(cal),
                          calendar20m_detected=sum(detected(r) for r in cal)))
    axes[0].set_ylabel('Pause ratio + 0.05')
    handles = [Line2D([], [], marker='o', ls='', mfc='.6', mec='.6', label='Map: effect detected'),
               Line2D([], [], marker='o', ls='', mfc='none', mec='.6', label='Map: effect not detected'),
               Line2D([], [], color='k', label='Detection probability 0.5'), Line2D([], [], color='k', ls='--', label='Detection probability 0.9'),
               Line2D([], [], marker='D', ls='', color='#c0392b', label='20 m calendar: effect detected'),
               Line2D([], [], marker='D', ls='', mfc='none', mec='#c0392b', label='20 m: effect not detected'),
               Line2D([], [], marker='X', ls='', color='#c0392b', label='Outside map support')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, frameon=False, fontsize=9)
    return fig, facts


# ---------------------------------------------------------------- Figure 4
def figure4():
    cal = read_csv(DATA / 'calendars/calendar_detection.csv')
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    fig.subplots_adjust(left=.25, right=.85, bottom=.16, top=.94, hspace=.35, wspace=.25)
    labels = [(f, s) for f in FORMS for s in STORAGES]
    facts = []
    for i, lead in enumerate(LEADS):
        for j, dist in enumerate((200, 20)):
            ax = axes[i, j]
            rs = [r for r in cases() if r['variant'] == 'original' and r['record_set'] == ('calendar200m' if dist == 200 else 'calendar20m')
                  and r['pair_id'] == STOP and int(r['lead_day']) == lead]
            matrix = np.full((len(labels), 12), np.nan); inactive = []
            for rr, (form, storage) in enumerate(labels):
                for month in range(1, 13):
                    cell = [r for r in rs if (r['calendar_type'], r['storage_type']) == (form, storage) and int(r['origin_month']) == month]
                    act = [r for r in cell if r['active'] == 'True']
                    k = sum(detected(r) for r in act)
                    cached = next(r for r in cal if r['stratum'] == 'distance_m+calendar_type+origin_month+storage_type' and int(r['distance_m']) == dist
                                  and r['calendar_type'] == form and r['storage_type'] == storage and int(r['origin_month']) == month)
                    assert int(cached['active_numerator']) == k and int(cached['active_denominator']) == len(act)
                    matrix[rr, month - 1] = num(cached['active_rate']) if act else np.nan
                    if not act:
                        inactive.append((rr, month - 1))
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto', interpolation='nearest')
            for rr, m in inactive:
                ax.add_patch(plt.Rectangle((m - .5, rr - .5), 1, 1, facecolor='.93', edgecolor='.65', hatch='///', lw=.3))
            ax.set_xticks(range(12), list('JFMAMJJASOND')); ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([FORM_LABEL[f] + ': ' + s for f, s in labels] if j == 0 else [])
            ax.set_xlabel('Origin month'); ax.set_title(f'({chr(97 + 2 * i + j)}) r = {dist} m, lead {lead} d', loc='left')
            facts.append(dict(distance=dist, lead=lead, records=len(rs), detected=sum(detected(r) for r in rs if r['active'] == 'True')))
    cax = fig.add_axes([.9, .25, .018, .48]); cb = fig.colorbar(image, cax=cax)
    cb.set_label('Effect detected / active origins', fontsize=13, fontweight='bold')
    fig.legend(handles=[Patch(facecolor='.93', edgecolor='.65', hatch='///', label='No active contrast (no question)')],
               loc='lower center', bbox_to_anchor=(.53, .025), ncol=1, frameon=False)
    return fig, facts


# ---------------------------------------------------------------- Figure 5
def figure5():
    edges = read_json(DATA / 'engine/signal_ratio_bins.json')['edges']
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.34, top=.9, wspace=.19)
    facts = []
    rows = [r for r in cases() if r['variant'] == 'original' and detected(r)]
    for j, pair in enumerate(QUESTIONS):
        ax = axes[j]
        for lead, mk in [(10, 'o'), (30, 's')]:
            for distance, color, reference in [(200, '#b03a2e', False), (200, '#1f4e79', True), (20, '#e67e22', False)]:
                series = []
                sel = [r for r in rows if r['pair_id'] == pair and int(r['lead_day']) == lead and int(r['distance_m']) == distance]
                for b in range(6):
                    valid = []
                    for r in sel:
                        sr = num(r['SR']); ef = num(r['E_reference_m'] if reference else r['E_tool_m'])
                        if not sr or sr <= 0 or ef is None:
                            continue
                        if min(int(np.searchsorted(edges[1:], sr, side='left')), 5) == b:
                            valid.append(r)
                    k = sum(r['reference_physically_impossible' if reference else 'engine_physically_impossible'] == 'True' for r in valid)
                    rate = k / len(valid) if valid else None
                    facts.append(dict(pair=pair, lead=lead, distance=distance, reference=reference, bin=b, numerator=k, denominator=len(valid), rate=rate))
                    if rate is not None:
                        series.append((b, rate))
                if series:
                    ax.plot(*zip(*series), marker=mk if distance == 200 else '*', color=color, ls='-' if lead == 10 else '--', mfc=color if lead == 10 else 'white', ms=5)
        labels = [f'{"[" if i == 0 else "("}{edges[i]:.6g},\n{edges[i + 1]:.6g}]' for i in range(5)] + [f'>{edges[-1]:.6g}']
        ax.set_xticks(range(6), labels); ax.tick_params(axis='x', labelsize=9)
        ax.set_xlabel('Signal-ratio interval (rounded edges)', labelpad=9)
        ax.set_title('(a) Continue vs stop' if j == 0 else '(b) Current vs 1.5 times', loc='left', fontsize=12, pad=9)
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Physically impossible sign fraction')
    handles = [Line2D([], [], color=c, marker='o', label=t) for c, t in [('#b03a2e', 'Forecasting program: design / 200 m'),
               ('#1f4e79', 'Transfer-function reference: same records'), ('#e67e22', 'Forecasting program: 20 m')]] + [
               Line2D([], [], color='.3', ls=l, marker=m, label=f'Lead {k} d') for l, m, k in [('-', 'o', 10), ('--', 's', 30)]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
    return fig, facts


# ---------------------------------------------------------------- Figure 6
def figure6():
    data = read_json(DATA / 'intervention/variant_effect_ratios.json')['rows']
    colors = ['#555555', '#0072B2', '#D55E00', '#009E73']
    variants = ['original', 'range', 'control', 'stop_control']
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    fig.subplots_adjust(left=.13, right=.98, bottom=.15, top=.93, wspace=.34, hspace=.7)
    facts = []
    for i, pair in enumerate((INCREASE, STOP)):
        for j, lead in enumerate(LEADS):
            ax = axes[i, j]
            for x, v in enumerate(variants):
                d = next(r for r in data if (r['pair'], r['lead'], r['variant']) == (pair, lead, v))['ratio']
                if d['point'] is None:
                    continue
                if d.get('lower') is not None:
                    ax.vlines(x, d['lower'], d['upper'], color=colors[x], lw=1)
                    ax.hlines([d['lower'], d['upper']], x - .04, x + .04, color=colors[x], lw=1)
                ax.plot(x, d['point'], marker='o', color=colors[x], ms=4.5)
                facts.append(dict(pair=pair, lead=lead, variant=v, median=d['point'], lower=d.get('lower'), upper=d.get('upper')))
            ax.set_xticks(range(4), ['Baseline', 'Higher rate\n150 m³/d', 'Nominal rate\n100 m³/d', 'Zero rate\n0 m³/d'])
            ax.set_xlabel('Context variant'); ax.set_title(f'({chr(97 + 2 * i + j)}) ' + ('Increase' if pair == INCREASE else 'Stop') + f' | {lead} d', loc='left')
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            ax.axhline(0, color='.4', lw=.7); ax.axhline(1, color='.4', lw=.7, ls='--')
            if j == 0:
                ax.set_ylabel('Median signed effect ratio')
    return fig, facts


# ---------------------------------------------------------------- Figure S19
def figureS19():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    ax = axes[0]
    sc = [r for r in read_csv(DATA / 'scale/w_scale_table.csv') if r['pair_id'] == STOP and int(r['lead_day']) == 10 and r['layer'] != 'all']
    layers = sorted({r['layer'] for r in sc})
    for i, layer in enumerate(layers):
        rs = sorted((r for r in sc if r['layer'] == layer), key=lambda r: float(r['Q_scale']))
        x = [float(r['Q_scale']) * (1 + 0.04 * (i - len(layers) / 2)) for r in rs]
        y = [float(r['W_median_m']) for r in rs]
        err = [[yy - float(r['W_q25_m']) for yy, r in zip(y, rs)], [float(r['W_q75_m']) - yy for yy, r in zip(y, rs)]]
        ax.errorbar(x, y, yerr=err, marker='o' if layer.endswith('T50') else 's', ms=4, lw=0.9, color=STORAGE_COLORS.get(layer.split('_')[0], '0.3'), capsize=2)
    ax.set_xscale('log'); ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(['0.3', '1', '3']); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('Pumping scale'); ax.set_ylabel('W at 10 d (m)'); letter(ax, 'a')
    prim = [r for r in read_csv(DATA / 'linearized/linearized_vs_sampled_lead10.csv') if r['tolerance_mode'] == 'ar_locked']
    ok = [r for r in prim if r['status'] in CERTIFIED and (num(r['W_lin']) or 0) > 0 and (num(r['W_sampled']) or 0) > 0]
    nq = sum(r['no_question'] == 'True' for r in prim); other = len(prim) - len(ok) - nq
    ax = axes[1]
    for st in STORAGES:
        pts = [r for r in ok if r['layer'].startswith(st) or r['layer'].startswith('fixedc_' + st)]
        if pts:
            ax.scatter([num(r['W_sampled']) for r in pts], [num(r['W_lin']) for r in pts], s=5, alpha=0.5, color=STORAGE_COLORS[st], lw=0)
    v = [num(r['W_sampled']) for r in ok] + [num(r['W_lin']) for r in ok]
    lo, hi = min(v) / 1.5, max(v) * 1.5
    ax.plot([lo, hi], [lo, hi], color='0.4', lw=0.7, ls=':'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title('AR(1) metric, residual-recentred', fontsize=10)
    ax.set_xlabel('Sampled W (m)'); ax.set_ylabel('Linearized W (m)'); letter(ax, 'b')
    ax.text(0.03, 0.97, f'n = {len(ok)}\nno question = {nq}\nother status = {other}', transform=ax.transAxes, va='top', fontsize=8.5)
    handles = [Line2D([], [], marker='o', ls='', color=c, label=s.capitalize()) for s, c in STORAGE_COLORS.items()] + [
        Line2D([], [], marker='o', ls='-', color='0.3', label='T = 50 m$^2$/d'), Line2D([], [], marker='s', ls='-', color='0.3', label='T = 500 m$^2$/d')]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, dict(panel_b_points=len(ok), no_question=nq, other_status=other)


FIGURES = {'figure1': figure1, 'figure2': figure2, 'figure3': figure3, 'figure4': figure4,
           'figure5': figure5, 'figure6': figure6, 'figureS19': figureS19}
