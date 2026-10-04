"""D06 collectors: A distance rows (r20 vs parent r200), the fixed D05 map projection, B raw rows, the shared C case schema.

A rows: the frozen p4_collect.new_rows applied to results/phase5 (W envelope, truth_eval, structurally matched reference,
both actor tool archives; strict=True raises on any missing output, strict=False keeps explicit statuses), relabelled
source_phase=phase5 / dataset=D06A, then joined by twin identity to the saved official D05 row (results/phase4/
primary_rows.csv). Determination is strict (inf > 0 or sup < 0). Zero-origin cases are 'no_active_contrast' (no question):
they carry W/reference/tool receipts but no prediction, outcome or relative width. Main calendar denominators are ACTIVE
origins; all-scheduled counts are kept as companions.

Fixed D05 map (never refit): statistics.json cohort_A.models.{sign,magnitude}[lead][pair].no_layer coefficients with the
original rho offset 0.05 at the operating-calendar projection coordinates (SR_origin, rho_last_off); layer fits are not
used to manufacture predictions. Support: convex hull of the D05 cohort-A design points (log SR, log(rho+0.05)) from the
saved primary rows; outside points are flagged extrapolated, never clipped.

Contradictions follow D05 C: among direction-determined records, a strictly opposite nonzero E_tool sign is primary and an
exact-zero output is separate; the reference is counted the same way on the same rows.
Official assembly requires the complete phase5 freeze; pure functions are fixture-tested.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

import p4_collect as F
import p4_statistics as S
import p5_contracts as K

LEADS = (10, 30)
RHO_OFFSET = S.RHO_OFFSET


def _f(v):
    if v is None or v == '' or v == 'None':
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ fixed D05 map
def d05_map(stats_path: Path = K.P4 / 'statistics.json') -> dict:
    s = json.loads(Path(stats_path).read_text())
    if s['prespecification']['rho_offset'] != RHO_OFFSET:
        raise ValueError('rho offset differs from the frozen D05 value')
    out = {}
    for kind in ('sign', 'magnitude'):
        for lead in LEADS:
            for p in K.PAIRS:
                m = s['cohort_A']['models'][kind][str(lead)][p]['no_layer']
                out[kind, lead, p] = dict(status=m['status'], coefficients=m.get('coefficients'))
    return out


def d05_support_points(primary=K.P4 / 'primary_rows.csv') -> np.ndarray:
    """(log SR, log(rho+0.05)) of the D05 cohort-A (map) cases, read from the saved primary rows of the cohort-A keys."""
    import p4_contracts
    keys = {(r['source_phase'], r['case_id']) for r in p4_contracts.cohort_manifests()['map']}
    pts = set()
    for r in read_csv(primary):
        if (r['source_phase'], r['case_id']) in keys and r['pair_id'] == K.PAIRS[0] and r['lead_day'] in ('10', '10.0'):
            sr, rho = _f(r['SR']), _f(r['rho'])
            if sr and sr > 0 and rho is not None:
                pts.add((math.log(sr), math.log(rho + RHO_OFFSET)))
    return np.array(sorted(pts))


class Hull:
    def __init__(self, pts):
        from scipy.spatial import Delaunay
        self.tri = Delaunay(pts)
        self.n_points = len(pts)

    def contains(self, x, y) -> bool:
        return bool(self.tri.find_simplex(np.array([[x, y]]))[0] >= 0)


def predict(maps, lead, pair, SR, rho, hull=None) -> dict:
    """Fixed-coefficient projection; returns statuses instead of values when undefined."""
    if SR is None or rho is None or not SR > 0 or rho < 0:
        return dict(status='undefined_coordinates', p_sign=None, log_ratio=None, inside_support=None)
    xs, xr = math.log(SR), math.log(rho + RHO_OFFSET)
    out = dict(status='ok', x_log_SR=xs, x_log_rho_offset=xr, inside_support=None if hull is None else hull.contains(xs, xr))
    sg, mg = maps['sign', lead, pair], maps['magnitude', lead, pair]
    if sg['status'] == 'ok':
        c = sg['coefficients']
        out['p_sign'] = float(expit(c['intercept'] + c['log_signal_ratio'] * xs + c['log_rho_plus_offset'] * xr))
    else:
        out['p_sign'] = None
        out['sign_fit_status'] = sg['status']
    if mg['status'] == 'ok':
        c = mg['coefficients']
        out['log_ratio'] = float(c['intercept'] + c['log_signal_ratio'] * xs + c['log_rho_plus_offset'] * xr)
    else:
        out['log_ratio'] = None
        out['magnitude_fit_status'] = mg['status']
    return out


# ------------------------------------------------------------------ row-level classification
def sign_det(inf, sup):
    det, sg, _ = S._sign_from_envelope(inf, sup)
    return det, sg


def transition(det20, det200, no_active):
    if no_active:
        return 'no_active_contrast'
    if det20 is None or det200 is None:
        return 'missing'
    return {(1, 0): 'became_determined', (0, 1): 'lost_determination', (1, 1): 'both_determined', (0, 0): 'neither'}[(det20, det200)]


def contradiction(det, sg, e):
    """(primary opposite-sign, exact-zero output) on determined rows; None when not applicable/missing."""
    if det != 1 or e is None:
        return None, None
    s = S._binary_sign(e)
    return (s == -sg), (s == 0)


def a_row(r20: dict, r200: dict, man: dict, maps, hull) -> dict:
    lead, pair = int(r20['lead_day']), r20['pair_id']
    no_active = bool(man['no_active_contrast'])
    d20, s20 = sign_det(_f(r20['env_inf_m']), _f(r20['env_sup_m']))
    d200, s200 = sign_det(_f(r200['env_inf_m']), _f(r200['env_sup_m'])) if r200 else (None, None)
    row = dict(r20, twin_case_id=man['twin_case_id'], distance_m=man['distance_m'], form=man['form'], origin_month=man['origin_month'],
               SR_origin=man['SR_origin'], rho_last_off=man['rho_last_off'], t95_d=man['t95_d'], gain_m_per_m3d=man['gain_m_per_m3d'],
               det_r20=d20, sign_r20=s20, det_r200=d200, sign_r200=s200, transition=transition(d20, d200, no_active),
               W_m_r200=_f(r200.get('W_m')) if r200 else None, E_true_m_r200=_f(r200.get('E_true_m')) if r200 else None,
               E_tool_m_r200=_f(r200.get('E_tool_m')) if r200 else None, SR_r200=_f(r200.get('SR')) if r200 else None,
               rho_last_off_r200=_f(r200.get('rho_last_off')) if r200 else None)
    W, E = _f(r20['W_m']), _f(r20['E_true_m'])
    row['magnitude_resolved_r20'] = None if no_active else S.magnitude_resolved(W, E)
    row['magnitude_resolved_r200'] = None if no_active or not r200 else S.magnitude_resolved(row['W_m_r200'], row['E_true_m_r200'])
    row['relative_width_r20'] = None if no_active or not W or not E else W / abs(E)
    for tag, d, sg, e in (('tool', d20, s20, _f(r20.get('E_tool_m'))), ('reference', d20, s20, _f(r20.get('E_reference_m'))),
                          ('tool_r200', d200, s200, row['E_tool_m_r200'])):
        c, z = (None, None) if no_active else contradiction(d, sg, e)
        row[f'{tag}_opposite_sign'], row[f'{tag}_exact_zero'] = c, z
    if no_active:
        row.update(map_status='no_active_contrast', p_sign_map=None, log_ratio_map=None, inside_d05_support=None, p_sign_map_r200=None)
    else:
        pr = predict(maps, lead, pair, man['SR_origin'], man['rho_last_off'], hull)
        pp = predict(maps, lead, pair, row['SR_r200'], row['rho_last_off_r200'], hull)
        row.update(map_status=pr['status'], p_sign_map=pr.get('p_sign'), log_ratio_map=pr.get('log_ratio'), inside_d05_support=pr.get('inside_support'),
                   p_sign_map_r200=pp.get('p_sign'), inside_d05_support_r200=pp.get('inside_support'))
    return row


# ------------------------------------------------------------------ assembly
def parent_rows(primary=K.P4 / 'primary_rows.csv') -> dict:
    return {(r['case_id'], r['pair_id'], int(float(r['lead_day']))): r for r in read_csv(primary)
            if r['source_phase'] == 'phase4' and r['experiment_group'] == 'B'}


def a_rows(strict=True, base: Path = K.P5, mode='production', maps=None, hull=None, parents=None) -> list[dict]:
    if mode == 'production':
        K.require_frozen()
    man = {m['case_id']: m for m in json.loads((Path(base) / 'cases/derived_manifest.json').read_text())}
    raw = F.new_rows(strict, base)
    maps = maps or d05_map()
    hull = hull if hull is not None else Hull(d05_support_points())
    parents = parents if parents is not None else parent_rows()
    out = []
    for r in raw:
        r = {k: (str(v) if v is not None else '') for k, v in r.items()}
        r.update(source_phase='phase5', dataset='D06A')
        m = man[r['case_id']]
        r200 = parents.get((m['twin_case_id'], r['pair_id'], int(r['lead_day'])))
        if r200 is None and strict:
            raise F.MissingOutput('parent D05 row ' + m['twin_case_id'])
        out.append(a_row(r, r200, m, maps, hull))
    if strict and len(out) != len(man) * 4:
        raise F.MissingOutput('A rows coverage')
    return out


def _wrate(rows, w, num, den):
    n = sum(w.get(r['realization_i'], 1) for r in rows if den(r))
    k = sum(w.get(r['realization_i'], 1) for r in rows if den(r) and num(r))
    return None if n == 0 else k / n


def summarize_a(rows, draws=None) -> dict:
    """Active-origin determination rates (main), all-scheduled companions, transitions, map calibration, contradictions."""
    edges = json.loads((K.P4 / 'statistics.json').read_text())['cohort_C']['bins']['edges']  # fixed D05 C SR bin edges
    for r in rows:
        r['realization_i'] = int(float(r['realization']))
        sr = r.get('SR_origin')
        r['sr_bin_d05_fixed'] = 'nonpositive_or_no_question' if not sr or sr <= 0 else str(S._bin_index(sr, edges))
    groups = defaultdict(list)
    for r in rows:
        k = (r['pair_id'], int(r['lead_day']))
        for key in (('all',), ('form', r['form']), ('layer', r['sid']), ('storage', r['storage_type']), ('form_month', r['form'], int(r['origin_month'])),
                    ('form_month_layer', r['form'], int(r['origin_month']), r['sid']), ('sr_bin_d05_fixed', r['sr_bin_d05_fixed'])):
            groups[k + key].append(r)
    out = {}
    act = lambda r: r['transition'] != 'no_active_contrast' and r['det_r20'] is not None  # noqa: E731
    for k, rs in groups.items():
        rec = dict(n_scheduled=len(rs), n_no_question=sum(r['transition'] == 'no_active_contrast' for r in rs),
                   n_active=sum(r['transition'] != 'no_active_contrast' for r in rs), n_missing=sum(r['transition'] == 'missing' for r in rs),
                   determined_r20_active=sum(1 for r in rs if act(r) and r['det_r20'] == 1),
                   determined_r200_active=sum(1 for r in rs if act(r) and r['det_r200'] == 1),
                   transitions={t: sum(r['transition'] == t for r in rs) for t in ('became_determined', 'lost_determination', 'both_determined', 'neither')},
                   magnitude_resolved_r20_active=sum(1 for r in rs if r['magnitude_resolved_r20'] is True),
                   tool_opposite_sign_r20=sum(1 for r in rs if r['tool_opposite_sign'] is True), tool_exact_zero_r20=sum(1 for r in rs if r['tool_exact_zero'] is True),
                   reference_opposite_sign_r20=sum(1 for r in rs if r['reference_opposite_sign'] is True),
                   tool_opposite_sign_r200=sum(1 for r in rs if r['tool_r200_opposite_sign'] is True))
        rec['rate_r20_active'] = None if not rec['n_active'] else rec['determined_r20_active'] / rec['n_active']
        rec['rate_r20_all_scheduled_companion'] = rec['determined_r20_active'] / rec['n_scheduled'] if rec['n_scheduled'] else None
        rec['rate_r200_active'] = None if not rec['n_active'] else rec['determined_r200_active'] / rec['n_active']
        pr = [r for r in rs if act(r) and r['p_sign_map'] is not None]
        if pr:
            obs, p = np.array([r['det_r20'] for r in pr], float), np.array([r['p_sign_map'] for r in pr])
            rec['map'] = dict(n=len(pr), brier=float(np.mean((obs - p) ** 2)), mean_obs_minus_pred=float(np.mean(obs - p)),
                              confusion={str(t): dict(tp=int(np.sum((p >= t) & (obs == 1))), fp=int(np.sum((p >= t) & (obs == 0))),
                                                      fn=int(np.sum((p < t) & (obs == 1))), tn=int(np.sum((p < t) & (obs == 0)))) for t in S.PROBABILITY_THRESHOLDS},
                              n_outside_support=sum(1 for r in pr if r['inside_d05_support'] is False))
            if draws is not None and len(k) == 3:
                reps = []
                for d in draws:
                    w = np.array([d[r['realization_i']] for r in pr], float)
                    reps.append(None if w.sum() == 0 else float(np.sum(w * (obs - p)) / w.sum()))
                rec['map']['mean_obs_minus_pred_interval'] = S.percentile_interval([x for x in reps if x is not None], sum(x is None for x in reps))
        if draws is not None and len(k) == 3:
            reps, nn = [], 0
            for d in draws:
                v = _wrate(rs, d, lambda r: r['det_r20'] == 1, act)
                (reps.append(v) if v is not None else None)
                nn += v is None
            rec['rate_r20_active_interval'] = S.percentile_interval(reps, nn)
        out['|'.join(map(str, k))] = rec
    return out


def b_raw_rows(strict=True, mode='production') -> list[dict]:
    """7800 B rows (1950 x 2 pairs x 2 leads) from the frozen D05 official combined table results/phase4/primary_rows.csv
    (it carries the D03/D04 raw lead rows used by D05 plus the D05 A rows). Only W, inf, sup and E_true are taken;
    covariates are joined later from the manifests."""
    if mode == 'production':
        K.require_frozen()
    want = {(u['source_phase'], u['case_id']) for u in K.b_universe()}
    out = []
    for r in read_csv(K.P4 / 'primary_rows.csv'):
        if (r['source_phase'], r['case_id']) in want and r['track'] == 'raw' and r['quantity'] == 'lead' and int(float(r['lead_day'])) in LEADS:
            out.append(dict(source_phase=r['source_phase'], case_id=r['case_id'], pair_id=r['pair_id'], lead_day=int(float(r['lead_day'])), W_m=_f(r['W_m']),
                            env_inf_m=_f(r['env_inf_m']), env_sup_m=_f(r['env_sup_m']), E_true_m=_f(r['E_true_m']), W_status=r['W_status']))
    keys = [(r['source_phase'], r['case_id'], r['pair_id'], r['lead_day']) for r in out]
    expect = {(p, c, q, k) for p, c in want for q in K.PAIRS for k in LEADS}
    if len(keys) != len(set(keys)) or (strict and set(keys) != expect):
        raise F.MissingOutput(f'B rows: n={len(keys)} unique={len(set(keys))} missing={len(expect - set(keys))} extra={len(set(keys) - expect)}')
    if mode != 'production':  # prefreeze: identity coverage only, no outcome value leaves this function
        return [dict(source_phase=a, case_id=b, pair_id=c, lead_day=d) for a, b, c, d in keys]
    return out


def write_c_cases(path: Path | None = None, mode='production') -> Path:
    """Shared C schema: production writes results/phase5/C_CASES.json after generation; otherwise a preview to evidence."""
    if mode == 'production':
        K.require_frozen()
        if not (K.P5 / 'cases/derived_manifest.json').is_file():
            raise F.MissingOutput('phase5 A cases not generated')
        path = path or K.P5 / 'C_CASES.json'
    else:
        path = path or K.EVIDENCE / 'C_CASES_SCHEMA_PREVIEW.json'
    recs = K.c_universe()
    payload = dict(fields=list(K.C_FIELDS), n=len(recs), counts={ph: sum(r['source_phase'] == ph for r in recs) for ph in ('phase2', 'phase3', 'phase4', 'phase5')},
                   official=mode == 'production', records=recs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=0) + '\n')
    return Path(path)


A_HEADERS = F.HEADERS + ('twin_case_id', 'distance_m', 'form', 'SR_origin', 'rho_last_off', 't95_d', 'gain_m_per_m3d', 'det_r20', 'sign_r20', 'det_r200',
                         'sign_r200', 'transition', 'W_m_r200', 'E_true_m_r200', 'E_tool_m_r200', 'SR_r200', 'rho_last_off_r200', 'magnitude_resolved_r20',
                         'magnitude_resolved_r200', 'relative_width_r20', 'tool_opposite_sign', 'tool_exact_zero', 'reference_opposite_sign',
                         'reference_exact_zero', 'tool_r200_opposite_sign', 'tool_r200_exact_zero', 'map_status', 'p_sign_map', 'log_ratio_map',
                         'inside_d05_support', 'p_sign_map_r200', 'inside_d05_support_r200')


def run_official_a():
    """Official A assembly after both actors complete (complete freeze, strict coverage)."""
    digest = K.require_frozen()
    from p5_timescale import shared_draws
    rows = a_rows(strict=True)
    dr = shared_draws()
    (K.P5 / 'tables').mkdir(exist_ok=True)
    with open(K.P5 / 'A_rows.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=A_HEADERS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    summ = summarize_a(rows, dr['draws'])
    (K.P5 / 'tables/A_distance_summary.json').write_text(json.dumps(dict(protocol_sha256=digest, draws={k: v for k, v in dr.items() if k != 'draws'},
                                                                          summary=summ), indent=1, default=str) + '\n')
    write_c_cases(mode='production')
    K.require_frozen()
    return rows


# interface names (IMPLEMENTATION_INTERFACES draft)
assemble_new = a_rows


def case_records(new_manifest=None) -> list[dict]:
    return K.c_universe(new_manifest)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('production_A', 'c_schema_preview'))
    a = ap.parse_args()
    if a.mode == 'production_A':
        run_official_a()
    else:
        p = write_c_cases(mode='preview')
        print('wrote', K.rel(p))
