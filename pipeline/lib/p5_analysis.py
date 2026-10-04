"""D06 final aggregation tables (A distance calendars, B time-scale fits, C linearized widths) with provenance.

Reads ONLY the official component outputs and frozen D05 sources; never refits the D05 map, never reruns W/model/fits:
  A  results/phase5/A_rows.csv                      (p5_collect.run_official_a; twin-joined r20/r200 rows)
     results/phase4/primary_rows.csv                (frozen D05 rows: r200 reference outputs, map support, scale table)
     results/phase4/statistics.json                 (frozen cohort_A map coefficients, cohort_C fixed SR bin edges)
  B  results/phase5/B/B_fits.json, B_comparisons.csv, B_timescale_rows.csv (p5_timescale.run_official)
  C  results/phase5/C/C_linearized_rows.csv, C_primary_rows.csv, C_loglog.json, C_sign_agreement.csv,
     C_masking_rows.csv, C_window_masks.json, C_MERGE_RECEIPT.json  (p5_linearized merge)
Uncertainty: the shared D05 999 x 10 realization multiplicity matrix (logical sha 7520bcd1...), one draw for every table.

Rules carried from the frozen protocol draft v2 / RAW D06:
  * active origins (q_origin > 0) are the main calendar denominator; zero-origin = 'no question' counted separately;
    all-scheduled fractions are companions only;
  * strict determination inf > 0 or sup < 0 (computed upstream); r20 rows are matched to their r200 twin;
  * fixed D05 projection: sign = cohort_A no_layer only, magnitude = no_layer; no refit; support-hull flags kept;
  * contradictions: strictly opposite nonzero sign primary, exact-zero output separate, missing separate; fixed D05
    five log-SR bin edges with explicit below/above_D05_range strata (never folded into an end bin);
  * B: historical nominal-rho D04 C reproduction is a separate table from the realized-rho bundles; fixed-lead D04 C
    bundle (ii) alias is labelled, not read as recovery;
  * C: three required modes kept separate; every status (no_question, uncertified, unbounded, ...) is inventoried.
No acceptance threshold on any scientific value. Undefined values are written empty with a status column.

CLI
  production: env/.venv_pilot/bin/python lib/p5_analysis.py --production [--out results/phase5/aggregation]
  fixture:    env/.venv_pilot/bin/python lib/p5_analysis.py --fixture DIR   (DIR under results/phase5/implementation_aggregation)
  lookup:     env/.venv_pilot/bin/python lib/p5_analysis.py --lookup TABLE.csv --key col=value [...] --column COL [--out DIR]
"""
from __future__ import annotations

import os
for _n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_n, '1')
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
P4 = ROOT / 'results/phase4'
P5 = ROOT / 'results/phase5'
FIXTURE_ROOT = P5 / 'implementation_aggregation'
PAIRS = ('P1_continue_vs_stop', 'P2_current_vs_1p5x')
LEADS = (10, 30)
CAL_EDGES = tuple(round(0.1 * i, 1) for i in range(11))
THRESHOLDS = (0.5, 0.9)
BOOT_LOGICAL_SHA = '7520bcd1cd416b25f22e1e3c7ce6f3905209b837eac4d5a703a038b79f7aa30e'
CERTIFIED = ('certified_interior', 'certified_constrained', 'bounded_by_family_only', 'zero_contrast', 'no_question')
C_MODES = ('ar_locked', 'ar_prospective', 'sse_locked')
OWN_ENTRIES = ('p5_analysis', 'p5_figures')


def default_inputs(base: Path = P5, p4: Path = P4) -> dict:
    return dict(A_rows=base / 'A_rows.csv', d05_rows=p4 / 'primary_rows.csv', d05_stats=p4 / 'statistics.json',
                draws=p4 / 'bootstrap_draw_matrix.npy', B_fits=base / 'B/B_fits.json', B_comparisons=base / 'B/B_comparisons.csv',
                B_rows=base / 'B/B_timescale_rows.csv', C_rows=base / 'C/C_linearized_rows.csv', C_primary=base / 'C/C_primary_rows.csv',
                C_loglog=base / 'C/C_loglog.json', C_sign=base / 'C/C_sign_agreement.csv', C_masks=base / 'C/C_masking_rows.csv',
                C_windows=base / 'C/C_window_masks.json', C_merge=base / 'C/C_MERGE_RECEIPT.json')


# ------------------------------------------------------------------ parsing
def _f(v):
    if v is None or isinstance(v, bool):
        return None if v is None else float(v)
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    s = str(v).strip()
    if s in ('', 'None', 'nan', 'NaN', 'null'):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _i(v):
    x = _f(v)
    return None if x is None else int(round(x))


def _b(v):
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip()
    return {'True': True, 'true': True, '1': True, 'False': False, 'false': False, '0': False}.get(s)


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ shared draws
def shared_draws(path=P4 / 'bootstrap_draw_matrix.npy') -> np.ndarray:
    D = np.load(path, allow_pickle=False)
    if D.shape != (999, 10) or np.any(D < 0) or np.any(D.sum(1) != 10):
        raise ValueError('shared draw matrix malformed')
    if hashlib.sha256(np.asarray(D, '<i8').tobytes()).hexdigest() != BOOT_LOGICAL_SHA:
        raise PermissionError('shared draw matrix logical digest differs from D05')
    return np.asarray(D, float)


def _interval(vals, n_not):
    vals = [float(v) for v in vals if v is not None and math.isfinite(v)]
    out = dict(n_identified=len(vals), n_not_identified=int(n_not), low=None, high=None,
               status='conditional_on_identified_replicates' if vals else 'unidentifiable')
    if vals:
        out['low'], out['high'] = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return out


def ratio_interval(num_by_r, den_by_r, D):
    """Cluster-bootstrap percentile interval of sum(num)/sum(den); per-realization arrays of length 10."""
    if D is None:
        return None
    n, d = D @ np.asarray(num_by_r, float), D @ np.asarray(den_by_r, float)
    ok = d > 0
    return _interval(list(n[ok] / d[ok]), int((~ok).sum()))


def diff_interval(n1, d1, n2, d2, D):
    if D is None:
        return None
    a, b, c, e = (D @ np.asarray(x, float) for x in (n1, d1, n2, d2))
    ok = (b > 0) & (e > 0)
    return _interval(list(a[ok] / b[ok] - c[ok] / e[ok]), int((~ok).sum()))


def mean_interval(val_by_r, cnt_by_r, D):
    return ratio_interval(val_by_r, cnt_by_r, D)


def _by_r(rows, fn):
    out = np.zeros(10)
    for r in rows:
        v = fn(r)
        if v:
            out[r['realization_i']] += float(v)
    return out


# ------------------------------------------------------------------ provenance registry
class Registry:
    def __init__(self, out: Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tables = {}

    def write(self, name, rows, keys, sources, selection, method):
        path = self.out / name
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols or ['empty'])
            w.writeheader()
            for r in rows:
                w.writerow({k: ('' if v is None else (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v)) for k, v in r.items()})
        self.tables[name] = dict(path=rel(path), sha256=sha(path), n_rows=len(rows), columns=cols, key_columns=list(keys),
                                 sources={rel(s): sha(s) for s in sources if Path(s).is_file()}, selection=selection, method=method)
        return path

    def finish(self, extra):
        payload = dict(extra, tables=self.tables)
        (self.out / 'TABLE_PROVENANCE.json').write_text(json.dumps(payload, indent=1, default=str) + '\n')
        return payload


def lookup(out_dir, table, keys: dict, column):
    """Return value + provenance for one cell: (table file, 1-based data row, column, upstream sources)."""
    prov = json.loads((Path(out_dir) / 'TABLE_PROVENANCE.json').read_text())['tables'][table]
    hits = [(i + 2, r) for i, r in enumerate(read_csv(Path(out_dir) / table)) if all(str(r.get(k)) == str(v) for k, v in keys.items())]
    if len(hits) != 1:
        raise KeyError(f'{len(hits)} rows match {keys} in {table}')
    line, r = hits[0]
    return dict(value=r[column], table=prov['path'], table_sha256=prov['sha256'], csv_line=line, column=column, keys=keys,
                sources=prov['sources'], selection=prov['selection'], method=prov['method'])


# ------------------------------------------------------------------ A: distance calendars
def sr_bin(sr, edges):
    """Fixed D05 cohort_C edges; values outside the D05 range keep explicit strata (not folded into end bins)."""
    if sr is None or not sr > 0:
        return 'nonpositive_or_no_question'
    if sr < edges[0]:
        return 'below_D05_range'
    if sr > edges[-1]:
        return 'above_D05_range'
    return str(int(np.digitize([sr], edges[1:-1], right=True)[0]))


def prepare_a(rows, parents=None):
    """Type A rows; attach the r200 parent reference output (frozen D05 row) for reference contradictions."""
    out = []
    for r in rows:
        x = dict(r)
        x['realization_i'] = _i(r['realization'])
        x['lead'] = _i(r['lead_day'])
        x['no_q'] = r.get('transition') == 'no_active_contrast'
        for k in ('det_r20', 'det_r200', 'sign_r20', 'sign_r200', 'origin_month'):
            x[k] = _i(r.get(k))
        for k in ('W_m', 'E_true_m', 'E_tool_m', 'E_reference_m', 'W_m_r200', 'E_true_m_r200', 'E_tool_m_r200', 'SR_origin', 'SR_r200',
                  'rho_last_off', 'rho_last_off_r200', 'p_sign_map', 'p_sign_map_r200', 'log_ratio_map'):
            x[k] = _f(r.get(k))
        for k in ('magnitude_resolved_r20', 'magnitude_resolved_r200', 'inside_d05_support', 'inside_d05_support_r200'):
            x[k] = _b(r.get(k))
        par = None if parents is None else parents.get((r['twin_case_id'], r['pair_id'], x['lead']))
        x['E_reference_m_r200'] = None if par is None else _f(par.get('E_reference_m'))
        x['reference_status_r200'] = None if par is None else par.get('reference_status')
        out.append(x)
    return out


def _group_keys(r):
    yield ('all', '', '', '', '')
    yield ('form', r['form'], '', '', '')
    yield ('storage', '', '', r['storage_type'], '')
    yield ('layer', '', '', '', r['sid'])
    yield ('form_month', r['form'], r['origin_month'], '', '')
    yield ('form_month_storage', r['form'], r['origin_month'], r['storage_type'], '')
    yield ('form_month_layer', r['form'], r['origin_month'], '', r['sid'])


def a_determination(rows, D):
    groups = defaultdict(list)
    for r in rows:
        for g in _group_keys(r):
            groups[(r['pair_id'], r['lead']) + g].append(r)
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rs = groups[key]
        act = [r for r in rs if not r['no_q']]
        v20 = [r for r in act if r['det_r20'] is not None]
        v200 = [r for r in act if r['det_r200'] is not None]
        paired = [r for r in act if r['det_r20'] is not None and r['det_r200'] is not None]
        m20 = [r for r in act if r['magnitude_resolved_r20'] is not None]
        m200 = [r for r in act if r['magnitude_resolved_r200'] is not None]
        tr = Counter(r.get('transition') for r in rs)
        n20 = sum(r['det_r20'] for r in v20)
        n200 = sum(r['det_r200'] for r in v200)
        rec = dict(pair_id=key[0], lead_day=key[1], level=key[2], form=key[3], origin_month=key[4], storage_type=key[5], layer=key[6],
                   n_scheduled=len(rs), n_no_question=len(rs) - len(act), n_active=len(act), n_active_missing_r20=len(act) - len(v20),
                   n_active_missing_r200=len(act) - len(v200), n_active_paired=len(paired),
                   determined_r20=n20, determined_r200=n200,
                   frac_determined_r20_active=n20 / len(v20) if v20 else None, frac_determined_r200_active=n200 / len(v200) if v200 else None,
                   frac_r20_minus_r200_paired=(sum(r['det_r20'] for r in paired) - sum(r['det_r200'] for r in paired)) / len(paired) if paired else None,
                   frac_determined_r20_all_scheduled_companion=n20 / len(rs) if rs else None,
                   became_determined=tr['became_determined'], lost_determination=tr['lost_determination'], both_determined=tr['both_determined'],
                   neither=tr['neither'], transition_missing=tr['missing'],
                   magnitude_defined_r20=len(m20), magnitude_below1_r20=sum(r['magnitude_resolved_r20'] for r in m20),
                   magnitude_defined_r200=len(m200), magnitude_below1_r200=sum(r['magnitude_resolved_r200'] for r in m200),
                   denominator_rule='active origins (q_origin>0) with a determination status; no-question origins counted separately')
        if D is not None and key[2] in ('all', 'form', 'storage', 'layer', 'form_month', 'form_month_storage'):
            i20 = ratio_interval(_by_r(v20, lambda r: r['det_r20']), _by_r(v20, lambda r: 1), D)
            i200 = ratio_interval(_by_r(v200, lambda r: r['det_r200']), _by_r(v200, lambda r: 1), D)
            idf = diff_interval(_by_r(paired, lambda r: r['det_r20']), _by_r(paired, lambda r: 1), _by_r(paired, lambda r: r['det_r200']), _by_r(paired, lambda r: 1), D)
            for nm, iv in (('r20', i20), ('r200', i200), ('diff', idf)):
                rec[f'ci_{nm}_low'], rec[f'ci_{nm}_high'], rec[f'ci_{nm}_n_identified'] = iv['low'], iv['high'], iv['n_identified']
        out.append(rec)
    return out


def a_map_check(rows, maps, hull, D):
    """Fixed D05 projection against observed determination/magnitude at both distances (no refit)."""
    import p5_collect as C
    cal, summ, inv = [], [], []
    per = []
    for r in rows:
        for dist in (20, 200):
            base = dict(pair_id=r['pair_id'], lead_day=r['lead'], distance_m=dist, form=r['form'], storage_type=r['storage_type'],
                        realization_i=r['realization_i'], no_q=r['no_q'])
            if r['no_q']:
                per.append(dict(base, status='no_question'))
                continue
            if dist == 20:
                p, lr, inside, det = r['p_sign_map'], r['log_ratio_map'], r['inside_d05_support'], r['det_r20']
                W, E = _f(r['W_m']), _f(r['E_true_m'])
                status = r.get('map_status') or 'ok'
            else:
                pr = C.predict(maps, r['lead'], r['pair_id'], r['SR_r200'], r['rho_last_off_r200'], hull)
                p, lr, inside, det, status = pr.get('p_sign'), pr.get('log_ratio'), pr.get('inside_support'), r['det_r200'], pr['status']
                W, E = r['W_m_r200'], r['E_true_m_r200']
                if r['p_sign_map_r200'] is not None and p is not None and abs(p - r['p_sign_map_r200']) > 1e-12:
                    raise ValueError('r200 fixed-map recomputation differs from collector value: ' + r['case_id'])
            obs_lr = math.log(W / abs(E)) if W and W > 0 and E else None
            per.append(dict(base, status=status, p=p, det=det, inside=inside, pred_lr=lr, obs_lr=obs_lr))
    groups = defaultdict(list)
    for x in per:
        for g in (('all', ''), ('storage', x['storage_type']), ('form', x['form'])):
            groups[(x['pair_id'], x['lead_day'], x['distance_m']) + g].append(x)
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        xs = groups[key]
        head = dict(pair_id=key[0], lead_day=key[1], distance_m=key[2], level=key[3], stratum=key[4])
        ok = [x for x in xs if not x['no_q'] and x.get('p') is not None and x.get('det') is not None]
        inv.append(dict(head, n_rows=len(xs), n_no_question=sum(x['no_q'] for x in xs),
                        n_undefined_coordinates=sum(1 for x in xs if x.get('status') == 'undefined_coordinates'),
                        n_missing_observation=sum(1 for x in xs if not x['no_q'] and x.get('p') is not None and x.get('det') is None),
                        n_inside_support=sum(1 for x in ok if x['inside'] is True), n_outside_support_extrapolated=sum(1 for x in ok if x['inside'] is False),
                        n_support_unknown=sum(1 for x in ok if x['inside'] is None)))
        rec = dict(head, n=len(ok), fit='D05 cohort_A no_layer sign, frozen coefficients, no refit',
                   coordinates='SR_origin and rho_last_off at the case distance; operating-calendar projection')
        if ok:
            o, p = np.array([x['det'] for x in ok], float), np.array([x['p'] for x in ok])
            rec.update(brier=float(np.mean((o - p) ** 2)), mean_obs_minus_pred=float(np.mean(o - p)), observed_frac=float(o.mean()), mean_pred=float(p.mean()),
                       n_outside_support=sum(1 for x in ok if x['inside'] is False))
            for t in THRESHOLDS:
                rec.update({f'tp_{t}': int(np.sum((p >= t) & (o == 1))), f'fp_{t}': int(np.sum((p >= t) & (o == 0))),
                            f'fn_{t}': int(np.sum((p < t) & (o == 1))), f'tn_{t}': int(np.sum((p < t) & (o == 0)))})
            if D is not None:
                num = np.zeros(10); den = np.zeros(10); sq = np.zeros(10)
                for x in ok:
                    num[x['realization_i']] += x['det'] - x['p']; den[x['realization_i']] += 1; sq[x['realization_i']] += (x['det'] - x['p']) ** 2
                iv = ratio_interval(num, den, D); ib = ratio_interval(sq, den, D)
                rec.update(ci_mean_obs_minus_pred_low=iv['low'], ci_mean_obs_minus_pred_high=iv['high'], ci_brier_low=ib['low'], ci_brier_high=ib['high'])
            for i in range(len(CAL_EDGES) - 1):
                lo, hi = CAL_EDGES[i], CAL_EDGES[i + 1]
                sel = (p >= lo) & ((p < hi) if i < len(CAL_EDGES) - 2 else (p <= hi))
                cal.append(dict(head, bin_low=lo, bin_high=hi, n=int(sel.sum()), mean_pred=float(p[sel].mean()) if sel.any() else None,
                                observed_frac=float(o[sel].mean()) if sel.any() else None))
        mg = [x for x in xs if not x['no_q'] and x.get('pred_lr') is not None and x.get('obs_lr') is not None]
        rec['magnitude_n'] = len(mg)
        if mg:
            res = np.array([x['obs_lr'] - x['pred_lr'] for x in mg])
            rec.update(magnitude_residual_mean=float(res.mean()), magnitude_residual_median=float(np.median(res)),
                       magnitude_below1_agreement=float(np.mean([(x['pred_lr'] < 0) == (x['obs_lr'] < 0) for x in mg])),
                       magnitude_fit='D05 cohort_A magnitude no_layer, frozen coefficients')
        summ.append(rec)
    return summ, cal, inv


def a_contradictions(rows, edges, D):
    per = []
    for r in rows:
        for dist in (20, 200):
            if dist == 20:
                det, sg, tool, ref, sr = r['det_r20'], r['sign_r20'], r['E_tool_m'], r['E_reference_m'], r['SR_origin']
            else:
                det, sg, tool, ref, sr = r['det_r200'], r['sign_r200'], r['E_tool_m_r200'], r['E_reference_m_r200'], r['SR_r200']
            x = dict(pair_id=r['pair_id'], lead_day=r['lead'], distance_m=dist, form=r['form'], origin_month=r['origin_month'],
                     storage_type=r['storage_type'], realization_i=r['realization_i'], no_q=r['no_q'], det=det, bin=sr_bin(None if r['no_q'] else sr, edges))
            for tag, e in (('tool', tool), ('reference', ref)):
                if r['no_q'] or det != 1:
                    x[f'{tag}_opp'] = x[f'{tag}_zero'] = x[f'{tag}_missing'] = None
                elif e is None:
                    x[f'{tag}_opp'], x[f'{tag}_zero'], x[f'{tag}_missing'] = False, False, True
                else:
                    s = 1 if e > 0 else (-1 if e < 0 else 0)
                    x[f'{tag}_opp'], x[f'{tag}_zero'], x[f'{tag}_missing'] = s == -sg, s == 0, False
            per.append(x)
    groups = defaultdict(list)
    for x in per:
        for g in (('all', '', '', ''), ('sr_bin_d05_fixed', x['bin'], '', ''), ('storage', '', '', x['storage_type']), ('form_month', x['form'], x['origin_month'], '')):
            groups[(x['pair_id'], x['lead_day'], x['distance_m']) + g].append(x)
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        xs = groups[key]
        det = [x for x in xs if not x['no_q'] and x['det'] == 1]
        rec = dict(pair_id=key[0], lead_day=key[1], distance_m=key[2], level=key[3], stratum=key[4], origin_month=key[5], storage_type=key[6],
                   n_rows=len(xs), n_no_question=sum(x['no_q'] for x in xs), n_undetermined=sum(1 for x in xs if not x['no_q'] and x['det'] == 0),
                   n_status_missing=sum(1 for x in xs if not x['no_q'] and x['det'] is None), n_determined=len(det),
                   rule='strict opposite nonzero sign among determined rows; exact-zero output and missing output separate')
        for tag in ('tool', 'reference'):
            k_opp, k_zero, k_miss = (sum(1 for x in det if x[f'{tag}_{s}']) for s in ('opp', 'zero', 'missing'))
            rec.update({f'{tag}_opposite_sign': k_opp, f'{tag}_exact_zero': k_zero, f'{tag}_missing': k_miss,
                        f'{tag}_opposite_rate_of_determined': k_opp / len(det) if det else None})
            if D is not None and key[3] in ('all', 'sr_bin_d05_fixed', 'storage'):
                iv = ratio_interval(_by_r(det, lambda x, t=tag: x[f'{t}_opp']), _by_r(det, lambda x: 1), D)
                rec[f'ci_{tag}_low'], rec[f'ci_{tag}_high'] = iv['low'], iv['high']
        out.append(rec)
    return out


def d05_scale_table(d05_rows):
    """W (m) by pumping scale for the matched D05 A design (phase4 A, scales 0.3/1/3): RAW 1 scale-invariance facet."""
    g = defaultdict(list)
    for r in d05_rows:
        if r['source_phase'] == 'phase4' and r['experiment_group'] == 'A' and r['track'] == 'raw' and r['quantity'] == 'lead':
            W, s = _f(r['W_m']), _f(r['Q_scale'])
            if W is not None and s is not None:
                for lay in ('all', r['sid']):
                    g[(r['pair_id'], _i(r['lead_day']), lay, s)].append(W)
    return [dict(pair_id=k[0], lead_day=k[1], layer=k[2], Q_scale=k[3], n=len(v), W_median_m=float(np.median(v)),
                 W_q25_m=float(np.percentile(v, 25)), W_q75_m=float(np.percentile(v, 75))) for k, v in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


# ------------------------------------------------------------------ B
def b_tables(fits_path, comparisons_path):
    fits = json.loads(Path(fits_path).read_text())
    summary = []
    for c in fits['comparisons']:
        for b in ('i', 'ii', 'iii'):
            pt = c['point'][b]
            p = len(pt.get('columns') or [])
            rank = pt.get('rank')
            alias = rank is not None and p and rank < p
            label = ''
            if c['subset'] == 'D04C' and str(c['lead_mode']) in ('10', '30') and b == 'ii':
                label = 'fixed_lead_alias: Lk in span(1, log pi_r); identical column space to (i); not recovery'
            elif c['subset'] == 'D04C' and str(c['lead_mode']) == 'pooled' and b == 'ii':
                label = 'pooled lead contrast only (equals (i) plus a lead term in D04 C)'
            metric = pt.get('r2') if c['outcome'] == 'magnitude' else pt.get('log_likelihood')
            summary.append(dict(pair_id=c['pair_id'], lead_mode=c['lead_mode'], subset=c['subset'], outcome=c['outcome'], bundle=b,
                                pair_role=c.get('pair_role'), n_valid=c['point'].get('n_valid_rows'), n_excluded=c['point'].get('n_excluded'),
                                metric_name='R2' if c['outcome'] == 'magnitude' else 'log_likelihood', metric=metric,
                                mcfadden_r2=pt.get('mcfadden_r2'), status=pt.get('status'), rank=rank, n_columns=p, aliased=bool(alias),
                                schema_note=pt.get('schema_note'), interpretation_label=label, rho_definition='realized R_days/t95 (z0 for R=0)'))
    deltas = []
    for c in fits['comparisons']:
        for k, v in c['delta'].items():
            iv = (c.get('delta_interval') or {}).get(k) or {}
            deltas.append(dict(pair_id=c['pair_id'], lead_mode=c['lead_mode'], subset=c['subset'], outcome=c['outcome'], contrast=k, delta=v,
                               ci_low=iv.get('low'), ci_high=iv.get('high'), n_identified=iv.get('n_identified'), n_not_identified=iv.get('n_not_identified')))
    gate = fits.get('d04c_source_gate') or {}
    hist = [dict(key=k, n=v.get('n'), r2_reproduced=v.get('r2_reproduced'), r2_historical=v.get('r2_readout'), abs_diff=v.get('abs_diff'), rank=v.get('rank'),
                 rho_definition='historical nominal rho (cell_metrics rho_nominal)', source=gate.get('source'), source_sha256=gate.get('source_sha256'),
                 passed=gate.get('passed')) for k, v in sorted((gate.get('gate') or {}).items())]
    return summary, deltas, hist, dict(ranks=fits.get('ranks'), p2_identity=fits.get('p2_identity'), draws=fits.get('draws'))


# ------------------------------------------------------------------ C
def c_inventory(rows_path, n_cases):
    cnt, nq_cases, n_mode, prim = Counter(), defaultdict(set), Counter(), Counter()
    flags = Counter()
    with open(rows_path, newline='') as fh:
        for r in csv.DictReader(fh):
            m = r['tolerance_mode']
            n_mode[m] += 1
            cnt[(m, r['status'])] += 1
            if _b(r.get('primary')):
                prim[m] += 1
            if _b(r.get('no_question')):
                nq_cases[m].add((r['source_phase'], r['case_id']))
            for fl in ('linearization_failure', 'endpoint_numerical_uncertainty'):
                if _b(r.get(fl)):
                    flags[(m, fl)] += 1
    out = []
    for m in sorted(n_mode):
        out.append(dict(tolerance_mode=m, status='__ALL__', n_rows=n_mode[m], expected_rows=n_cases * 60, primary_rows=prim[m], expected_primary_rows=n_cases * 4,
                        no_question_cases=len(nq_cases[m]), linearization_failure_rows=flags[(m, 'linearization_failure')],
                        endpoint_numerical_uncertainty_rows=flags[(m, 'endpoint_numerical_uncertainty')]))
        for (mm, st), n in sorted(cnt.items()):
            if mm == m:
                out.append(dict(tolerance_mode=m, status=st, n_rows=n))
    return out


def c_layer_residuals(primary_path):
    g = defaultdict(list)
    excl = defaultdict(Counter)
    with open(primary_path, newline='') as fh:
        for r in csv.DictReader(fh):
            key = (r['tolerance_mode'], r['pair'], _i(r['lead']), r['layer'], r.get('distance_m'), r['source_phase'])
            wl, ws = _f(r.get('W_lin')), _f(r.get('W_sampled'))
            if r['status'] in CERTIFIED and wl and ws and wl > 0 and ws > 0:
                g[key].append(math.log(wl / ws))
            else:
                excl[key][r['status'] if r['status'] not in CERTIFIED else 'nonpositive_or_missing_width'] += 1
    out = []
    for key in sorted(set(g) | set(excl), key=lambda k: tuple(str(x) for x in k)):
        v = g.get(key, [])
        out.append(dict(tolerance_mode=key[0], pair=key[1], lead=key[2], layer=key[3], distance_m=key[4], source_phase=key[5], n_eligible=len(v),
                        log_Wlin_over_Wsampled_median=float(np.median(v)) if v else None, q25=float(np.percentile(v, 25)) if v else None,
                        q75=float(np.percentile(v, 75)) if v else None, excluded=dict(excl.get(key, {}))))
    return out


def c_loglog_flat(path):
    out = []
    for e in json.loads(Path(path).read_text()):
        g = e['group']
        rec = dict(tolerance_mode=g[0], pair=g[1], stratum_kind=g[2], stratum=g[3] if len(g) > 3 else '')
        rec.update({k: v for k, v in e.items() if k != 'group'})
        out.append(rec)
    return out


def c_mask_tables(path):
    rows = read_csv(path)
    curves = []
    for r in rows:
        curves.append(dict(source_phase=r['source_phase'], case_id=r['case_id'], window_kind=r['window_kind'], mask_metric=r['mask_metric'], pair=r['pair'],
                           lead=_i(r['lead']), window_status=r['window_status'], start=r.get('start'), end=r.get('end'), dates=r.get('dates'),
                           W_full=_f(r.get('W_full', r.get('full_width'))), W_mask=_f(r.get('W_mask', r.get('masked_width'))), ratio=_f(r.get('ratio')),
                           difference=_f(r.get('difference')), status=r.get('status'), monotonicity_audit_flag=_b(r.get('monotonicity_audit_flag')),
                           ratio_status='defined' if _f(r.get('ratio')) is not None else (r['window_status'] if r['window_status'] != 'present' else 'undefined_W_full_zero_or_uncertified')))
    summ = [c for c in curves if c['lead'] in (1, 10, 30)]
    return curves, summ


# ------------------------------------------------------------------ supplementary routes (existing sources only)
SUPP_ROUTES = {
    'layer_indicator_fits': ['results/phase4/statistics.json', 'results/phase4/figures/supplement_layer_10day_P1.png', 'results/phase4/figures/supplement_layer_30day_P1.png'],
    'positive_rho_sensitivity': ['results/phase4/statistics.json', 'results/phase4/figures/supplement_positive_rho_sensitivity.png'],
    'raw_pause_transition_width_D03_figureA': ['results/phase2/figures/figure_A_width.png', 'results/phase2/figures/cell_median.csv', 'results/phase2/analysis/cell_metrics.csv'],
    'raw_pause_transition_width_D04_figure2': ['results/phase3/figures/figure2_count_recency.png', 'results/phase4/figures/figure2_p1_pause_transitions.png', 'results/phase4/figures/figure2_plot_input.csv'],
    'recency_contrast': ['results/phase3/READOUT.json', 'results/phase4/figures/figure2_recency_contrasts.csv', 'results/phase4/figures/figure2_recency_summary.csv'],
    'normalization_diagnostic': ['results/phase3/normalization.csv', 'results/phase3/NORMALIZATION_SCHEMA.json', 'results/phase4/figures/supplement_raw_normalization.png'],
    'filter_tracks_D03_D04': ['results/phase2/analysis/cell_metrics.csv', 'results/phase3/cell_metrics.csv'],
    'reference_numerical_quality': ['results/phase4/primary_rows.csv', 'results/phase2/wb/reference', 'results/phase3/wb/reference', 'results/phase4/wb/reference'],
    'D05_r200_calendar_detail': ['results/phase4/figures/figure3_calendar_P1.png', 'results/phase4/figures/figure3_layer_detail_P1.png', 'results/phase4/figures/figure3_offseason_variant_P1.png', 'results/phase4/statistics.json'],
}


def supplementary_routes(root=ROOT):
    out = []
    for item, paths in SUPP_ROUTES.items():
        for p in paths:
            q = root / p
            out.append(dict(item=item, path=p, exists=q.exists(), kind='dir' if q.is_dir() else 'file', sha256=sha(q) if q.is_file() else None,
                            note='existing immutable source; reuse, do not re-execute'))
    return out


# ------------------------------------------------------------------ driver
def run(inputs: dict, out: Path, D, *, maps=None, hull=None, n_c_cases=6270, n_a_cases=2160, strict=True, mode='production'):
    import p5_collect as C
    reg = Registry(out)
    a_raw = read_csv(inputs['A_rows'])
    if strict and len(a_raw) != n_a_cases * 4:
        raise ValueError(f'A rows {len(a_raw)} != {n_a_cases * 4}')
    d05 = read_csv(inputs['d05_rows'])
    parents = {(r['case_id'], r['pair_id'], _i(r['lead_day'])): r for r in d05 if r['source_phase'] == 'phase4' and r['experiment_group'] == 'B'}
    A = prepare_a(a_raw, parents)
    edges = json.loads(Path(inputs['d05_stats']).read_text())['cohort_C']['bins']['edges']
    maps = maps or C.d05_map(inputs['d05_stats'])
    hull = hull if hull is not None else C.Hull(C.d05_support_points(inputs['d05_rows']))
    srcA = [inputs['A_rows'], inputs['d05_rows'], inputs['d05_stats'], inputs['draws']]
    reg.write('A_determination.csv', a_determination(A, D), ('pair_id', 'lead_day', 'level', 'form', 'origin_month', 'storage_type', 'layer'), srcA,
              'all A rows; active = q_origin>0', 'counts and fractions; r20 vs twin r200; shared-draw percentile intervals')
    ms, mc, mi = a_map_check(A, maps, hull, D)
    reg.write('A_map_prediction.csv', ms, ('pair_id', 'lead_day', 'distance_m', 'level', 'stratum'), srcA, 'active rows with prediction and observed status',
              'fixed D05 cohort_A no_layer projection; Brier, mean(obs-pred), confusion at .5/.9, magnitude residual')
    reg.write('A_map_calibration.csv', mc, ('pair_id', 'lead_day', 'distance_m', 'level', 'stratum', 'bin_low'), srcA, 'as A_map_prediction', 'fixed decile bins')
    reg.write('A_map_extrapolation_inventory.csv', mi, ('pair_id', 'lead_day', 'distance_m', 'level', 'stratum'), srcA, 'all A rows', 'D05 1260-case support hull flags')
    reg.write('A_contradictions.csv', a_contradictions(A, edges, D), ('pair_id', 'lead_day', 'distance_m', 'level', 'stratum', 'origin_month', 'storage_type'), srcA,
              'determined rows at each distance', 'D05 C rule; fixed D05 bins plus below/above_D05_range')
    reg.write('D05_W_scale_table.csv', d05_scale_table(d05), ('pair_id', 'lead_day', 'layer', 'Q_scale'), [inputs['d05_rows']], 'phase4 A raw lead rows', 'median/IQR of saved W')
    bs, bd, bh, bmeta = b_tables(inputs['B_fits'], inputs['B_comparisons'])
    srcB = [inputs['B_fits'], inputs['B_comparisons']]
    reg.write('B_bundle_fits.csv', bs, ('pair_id', 'lead_mode', 'subset', 'outcome', 'bundle'), srcB, 'all comparisons', 'pass-through of official fits with alias labels')
    reg.write('B_bundle_deltas.csv', bd, ('pair_id', 'lead_mode', 'subset', 'outcome', 'contrast'), srcB, 'all comparisons', 'paired deltas, shared-draw intervals')
    reg.write('B_D04C_historical_nominal_gate.csv', bh, ('key',), srcB, 'D04 C fixed-lead (i) nominal rho', 'historical reproduction, separate from realized-rho bundles')
    merge = json.loads(Path(inputs['C_merge']).read_text()) if Path(inputs['C_merge']).is_file() else {}
    if strict and merge.get('status') != 'execution_complete':
        raise ValueError('C merge receipt is not execution_complete: ' + str(merge.get('status')))
    srcC = [inputs[k] for k in ('C_rows', 'C_primary', 'C_loglog', 'C_sign', 'C_masks', 'C_windows', 'C_merge')]
    reg.write('C_status_inventory.csv', c_inventory(inputs['C_rows'], n_c_cases), ('tolerance_mode', 'status'), srcC, 'all curve rows', 'status counts; no_question cases distinct')
    reg.write('C_layer_residuals.csv', c_layer_residuals(inputs['C_primary']), ('tolerance_mode', 'pair', 'lead', 'layer', 'distance_m', 'source_phase'), srcC,
              'primary rows, certified with positive widths', 'log(W_lin/W_sampled) quantiles; exclusions by status')
    reg.write('C_loglog.csv', c_loglog_flat(inputs['C_loglog']), ('tolerance_mode', 'pair', 'stratum_kind', 'stratum'), srcC, 'C runtime groups', 'pass-through, x=log sampled W')
    reg.write('C_sign_agreement.csv', read_csv(inputs['C_sign']), ('group', 'active_only'), srcC, 'C runtime groups', 'pass-through')
    curves, msum = c_mask_tables(inputs['C_masks'])
    reg.write('C_mask_curves.csv', curves, ('source_phase', 'case_id', 'window_kind', 'mask_metric', 'pair', 'lead'), srcC, '18 representatives', 'full/masked widths and ratios, leads 1-30')
    reg.write('C_mask_lead_summary.csv', msum, ('source_phase', 'case_id', 'window_kind', 'mask_metric', 'pair', 'lead'), srcC, 'leads 1/10/30', 'subset of C_mask_curves')
    reg.write('SUPPLEMENTARY_ROUTES.csv', supplementary_routes(), ('item', 'path'), [], 'requested supplementary items', 'existing-source routing; no re-execution')
    return reg.finish(dict(mode=mode, draws_logical_sha256=BOOT_LOGICAL_SHA if D is not None else None, B_meta=bmeta, C_merge_status=merge.get('status'),
                           inputs={k: dict(path=rel(v), sha256=sha(v) if Path(v).is_file() else None) for k, v in inputs.items()}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--production', action='store_true')
    g.add_argument('--fixture', type=Path)
    g.add_argument('--lookup')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--key', action='append', default=[])
    ap.add_argument('--column')
    a = ap.parse_args(argv)
    if a.lookup:
        print(json.dumps(lookup(a.out or P5 / 'aggregation', a.lookup, dict(k.split('=', 1) for k in a.key), a.column), indent=1))
        return
    sys.path.insert(0, str(ROOT / 'lib'))
    if a.production:
        import p5_contracts as K
        digest = K.require_frozen(entries=K.RUNTIME_ENTRIES + OWN_ENTRIES)
        out = a.out or P5 / 'aggregation'
        res = run(default_inputs(), out, shared_draws(), mode='production')
        res['protocol_sha256'] = digest
        (Path(out) / 'TABLE_PROVENANCE.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
        K.require_frozen(entries=K.RUNTIME_ENTRIES + OWN_ENTRIES)
        print('AGGREGATION_COMPLETE', out)
    else:
        fx = a.fixture.resolve()
        if FIXTURE_ROOT.resolve() not in fx.parents:
            raise PermissionError('fixture directory must be under results/phase5/implementation_aggregation')
        spec = json.loads((fx / 'FIXTURE_SPEC.json').read_text())
        inputs = {k: fx / v for k, v in spec['inputs'].items()}
        res = run(inputs, a.out or fx / 'aggregation', shared_draws(), n_c_cases=spec['n_c_cases'], n_a_cases=spec['n_a_cases'], strict=True, mode='SYNTHETIC_FIXTURE_NOT_RESULTS')
        print('FIXTURE_AGGREGATION', len(res['tables']), 'tables')


if __name__ == '__main__':
    main()
