"""D06 B: three-bundle time-scale reanalysis of the 1950 noncalendar D03-D05 cases (no new execution).

Covariates come from the authoritative derived manifests: actual R_days, t95_d, SR, pi_r; k is the lead (10/30).
  z0 = 1[R = 0];  Lrho = log(R/t95) (R > 0, else 0);  LRk = log(R/k) (R > 0, else 0);  Lk = log(k/t95)
The zero code always travels with its own indicator z0 (a separate intercept; log 0 is never set to 0 silently).
  (i)   [1, log SR, log pi_r, z0, Lrho]
  (ii)  (i) + Lk
  (iii) [1, log SR, log pi_r, z0, LRk]
z0 is dropped deterministically (schema note) only in a subset with no R = 0 row (e.g. D04 C, positive-rho sensitivity).
Outcomes: magnitude y = log(W/|E_true|) (finite W > 0, finite nonzero E_true); sign y = 1[inf > 0 or sup < 0] (finite
inf/sup). All bundles of one outcome/subset/lead mode use the same valid rows.

OLS: weighted rank-revealing SVD of the RMS-standardized design, relative cutoff 1e-10, unpenalized projection. Aliased
columns -> status coefficient_nonunique with the null basis; fitted values/SSE/R2 stay identified in the column space.
Logistic: the same estimable column space (orthonormal SVD scores), then the FROZEN p4_statistics.fit_logistic (complete/
quasi separation LP, unpenalized IRLS, original limits); separation/constant/numerical statuses are kept as null MLEs.
Uncertainty: the shared D05 999 x 10 realization-multiplicity matrix (statistics.json bootstrap.draw_matrix), same weights
for all bundles; percentile 2.5/97.5 over identified replicates (conditional denominators reported). P2 is reported with
its identity check, not pooled as an independent observation. No outcome-dependent repair of formula, rows or target.
Official fits require the complete phase5 freeze (run_official); fit functions themselves are pure and fixture-tested.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

import p4_statistics as S
import p5_contracts as K

LEADS = (10, 30)
LEAD_MODES = ('pooled', 10, 30)
BUNDLES = ('i', 'ii', 'iii')
RANK_TOL = 1e-10
SUBSETS = ('full', 'D04C', 'positive_rho')


# ------------------------------------------------------------------ design
def transforms(R_days: float, t95: float, k: int, SR: float, pi_r: float) -> dict:
    if not (t95 > 0 and SR > 0 and pi_r > 0 and k > 0 and R_days >= 0):
        raise ValueError('nonpositive design input')
    pos = R_days > 0
    return dict(log_SR=math.log(SR), log_pi=math.log(pi_r), z0=0.0 if pos else 1.0, Lrho=math.log(R_days / t95) if pos else 0.0,
                LRk=math.log(R_days / k) if pos else 0.0, Lk=math.log(k / t95))


BUNDLE_COLUMNS = {'i': ('intercept', 'log_SR', 'log_pi', 'z0', 'Lrho'), 'ii': ('intercept', 'log_SR', 'log_pi', 'z0', 'Lrho', 'Lk'),
                  'iii': ('intercept', 'log_SR', 'log_pi', 'z0', 'LRk')}


def design(rows, bundle: str):
    names = list(BUNDLE_COLUMNS[bundle])
    cols = {n: np.array([1.0 if n == 'intercept' else r[n] for r in rows], float) for n in names}
    note = None
    if 'z0' in cols and not np.any(cols['z0']):
        names.remove('z0')
        note = 'z0 dropped: no R=0 row in this subset'
    return names, np.column_stack([cols[n] for n in names]), note


# ------------------------------------------------------------------ fits
def _standardize(X):
    scale = np.sqrt(np.mean(X * X, axis=0))
    scale[scale == 0] = 1.0
    return X / scale, scale


def ols_svd(y, X, w=None, names=None) -> dict:
    y, X = np.asarray(y, float), np.asarray(X, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    keep = w > 0
    y, X, w = y[keep], X[keep], w[keep]
    names = list(names) if names is not None else [f'b{i}' for i in range(X.shape[1])]
    out = dict(n=int(len(y)), sum_w=float(w.sum()), p=X.shape[1], status='empty', r2=None, sse=None, sst=None, coefficients=None, rank=0)
    if not len(y):
        return out
    Xs, scale = _standardize(X)
    sw = np.sqrt(w)
    U, s, Vt = np.linalg.svd(Xs * sw[:, None], full_matrices=False)
    r = int(np.sum(s > RANK_TOL * s[0])) if s.size and s[0] > 0 else 0
    yw = y * sw
    fit_w = U[:, :r] @ (U[:, :r].T @ yw)
    sse = float(np.sum((yw - fit_w) ** 2))
    ybar = float(np.sum(w * y) / w.sum())
    sst = float(np.sum(w * (y - ybar) ** 2))
    out.update(rank=r, singular_values=s.tolist(), sse=sse, sst=sst, r2=None if sst == 0 else 1.0 - sse / sst, condition=float(s[0] / s[-1]) if s[-1] > 0 else float('inf'))
    if r == X.shape[1]:
        beta = (Vt.T @ ((U.T @ yw) / s)) / scale
        out.update(status='ok', coefficients=dict(zip(names, beta.tolist())))
    else:
        null = (Vt[r:].T / scale[:, None])
        null = null / np.linalg.norm(null, axis=0)
        out.update(status='coefficient_nonunique', nullspace=dict(names=names, basis=null.T.tolist()))
    return out


def logit_estimable(y, X, w=None, names=None) -> dict:
    y, X = np.asarray(y, float), np.asarray(X, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    keep = w > 0
    y, X, w = y[keep], X[keep], w[keep]
    names = list(names) if names is not None else [f'b{i}' for i in range(X.shape[1])]
    base = dict(n=int(len(y)), sum_w=float(w.sum()), p=X.shape[1], coefficients=None, log_likelihood=None, null_log_likelihood=None, mcfadden_r2=None, deviance=None)
    if not len(y):
        return dict(base, status='empty', rank=0)
    Xs, scale = _standardize(X)
    _, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    r = int(np.sum(s > RANK_TOL * s[0]))
    Z = Xs @ Vt[:r].T
    fit = S.fit_logistic(y, Z, sample_weight=w, column_names=[f'pc{i}' for i in range(r)])
    pbar = float(np.sum(w * y) / w.sum())
    ll0 = float(np.sum(w * (y * math.log(pbar) + (1 - y) * math.log(1 - pbar)))) if 0 < pbar < 1 else None
    out = dict(base, status=fit['status'], rank=r, column_space_dropped=X.shape[1] - r, null_log_likelihood=ll0,
               separation_diagnostic=fit.get('separation_diagnostic'), large_coefficient_diagnostic=fit.get('large_coefficient_diagnostic'))
    if r < X.shape[1]:
        null = Vt[r:].T / scale[:, None]
        out['nullspace'] = dict(names=names, basis=(null / np.linalg.norm(null, axis=0)).T.tolist())
    if fit['status'] == 'ok':
        ll = fit['log_likelihood']
        out.update(log_likelihood=ll, deviance=-2 * ll, mcfadden_r2=None if not ll0 else 1.0 - ll / ll0)
        if r == X.shape[1]:
            gamma = np.array([fit['coefficients'][f'pc{i}'] for i in range(r)])
            out['coefficients'] = dict(zip(names, ((Vt[:r].T @ gamma) / scale).tolist()))
        else:
            out['status_detail'] = 'coefficient_nonunique_likelihood_identified'
    return out


# ------------------------------------------------------------------ rows and valid sets
def analysis_rows(raw_rows, universe=None) -> list[dict]:
    """raw_rows: (source_phase, case_id, pair_id, lead_day, W_m, env_inf_m, env_sup_m, E_true_m). Covariates are joined from
    the B universe (manifests), never from the raw rows."""
    uni = {(u['source_phase'], u['case_id']): u for u in (universe or K.b_universe())}
    out = []
    for r in raw_rows:
        u = uni[(r['source_phase'], r['case_id'])]
        k = int(r['lead_day'])
        row = dict(source_phase=u['source_phase'], case_id=u['case_id'], experiment_group=u['experiment_group'], realization=u['realization'], sid=u['sid'],
                   d04c=u['d04c'], pair_id=r['pair_id'], lead_day=k, R_days=u['R_days'], t95_d=u['t95_d'], SR=u['SR'], pi_r=u['pi_r'],
                   **transforms(u['R_days'], u['t95_d'], k, u['SR'], u['pi_r']))
        W, E, lo, hi = (r.get(x) for x in ('W_m', 'E_true_m', 'env_inf_m', 'env_sup_m'))
        fin = lambda v: v is not None and isinstance(v, (int, float)) and math.isfinite(v)  # noqa: E731
        row['mag_valid'] = bool(fin(W) and W > 0 and fin(E) and E != 0)
        row['mag_reason'] = 'ok' if row['mag_valid'] else ('execution_missing_W' if not fin(W) else 'W_nonpositive_log_undefined' if W <= 0 else
                                                           'E_true_missing' if not fin(E) else 'E_true_zero_log_undefined')
        row['y_mag'] = math.log(W / abs(E)) if row['mag_valid'] else None
        row['sign_valid'] = bool(fin(lo) and fin(hi) and lo <= hi)
        row['sign_reason'] = 'ok' if row['sign_valid'] else 'execution_missing_bounds'
        row['y_sign'] = (1.0 if (lo > 0 or hi < 0) else 0.0) if row['sign_valid'] else None
        out.append(row)
    return out


def subset_rows(rows, subset, pair, lead_mode):
    sel = [r for r in rows if r['pair_id'] == pair and (lead_mode == 'pooled' or r['lead_day'] == lead_mode)]
    if subset == 'D04C':
        sel = [r for r in sel if r['d04c']]
    elif subset == 'positive_rho':
        sel = [r for r in sel if r['R_days'] > 0]
    return sel


def fit_bundles(rows, outcome, weights=None) -> dict:
    """Same valid rows for all bundles of the outcome. weights: realization -> multiplicity (bootstrap) or None."""
    key = 'mag' if outcome == 'magnitude' else 'sign'
    valid = [r for r in rows if r[key + '_valid']]
    y = np.array([r['y_' + key] for r in valid], float)
    w = None if weights is None else np.array([float(weights[r['realization']]) for r in valid])
    res = {}
    for b in BUNDLES:
        names, X, note = design(valid, b) if valid else (list(BUNDLE_COLUMNS[b]), np.zeros((0, len(BUNDLE_COLUMNS[b]))), None)
        f = ols_svd(y, X, w, names) if outcome == 'magnitude' else logit_estimable(y, X, w, names)
        f['schema_note'] = note
        f['columns'] = names
        res[b] = f
    res['n_valid_rows'] = len(valid)
    res['n_excluded'] = {reason: sum(1 for r in rows if r[key + '_reason'] == reason) for reason in sorted({r[key + '_reason'] for r in rows}) if reason != 'ok'}
    return res


def _metric(f, outcome):
    if outcome == 'magnitude':
        return f.get('r2')
    return f.get('log_likelihood') if f.get('status') == 'ok' else None


def deltas(fits, outcome) -> dict:
    out = {}
    for a, b in (('ii', 'i'), ('iii', 'i'), ('ii', 'iii')):
        x, y = _metric(fits[a], outcome), _metric(fits[b], outcome)
        out[f'{a}-{b}'] = None if x is None or y is None else x - y
        if outcome == 'sign':
            ma, mb = fits[a].get('mcfadden_r2'), fits[b].get('mcfadden_r2')
            out[f'mcfadden_{a}-{b}'] = None if ma is None or mb is None else ma - mb
    return out


# ------------------------------------------------------------------ shared draws
def shared_draws() -> dict:
    s = json.loads((K.P4 / 'statistics.json').read_text())['bootstrap']
    M = np.asarray(s['draw_matrix'], dtype='<i8')
    npy = K.P4 / 'bootstrap_draw_matrix.npy'
    N = np.load(npy)
    logical = K.sha_bytes(M.tobytes())
    canonical = [[d[i] for i in range(10)] for d in S._draws([], S.N_BOOTSTRAP, S.SEED)[1]]
    rec = dict(logical_sha256_le_i8=logical, logical_sha256_declared=s['draw_matrix_sha256'], logical_matches=logical == s['draw_matrix_sha256'],
               physical_npy_path=K.rel(npy), physical_npy_sha256=K.sha(npy), physical_equals_logical=bool(np.array_equal(N.astype('<i8'), M)),
               canonical_pcg64_seed20261001=bool(np.array_equal(np.asarray(canonical, '<i8'), M)), shape=list(M.shape), cluster_universe=s['cluster_universe'],
               shared_across_d05=s['shared_across'])
    if not (rec['logical_matches'] and rec['physical_equals_logical'] and rec['canonical_pcg64_seed20261001'] and M.shape == (999, 10)):
        raise ValueError('shared bootstrap matrix verification failed: ' + json.dumps(rec))
    rec['draws'] = [{i: int(v) for i, v in enumerate(row)} for row in M]
    return rec


def compare(rows, draws=None, n_boot=None) -> list[dict]:
    """Every pair x lead mode x subset x outcome: point fits, deltas and paired percentile intervals over shared draws."""
    out = []
    for pair in K.PAIRS:
        for lm in LEAD_MODES:
            for sub in SUBSETS:
                sel = subset_rows(rows, sub, pair, lm)
                for outcome in ('magnitude', 'sign'):
                    pt = fit_bundles(sel, outcome)
                    rec = dict(pair_id=pair, lead_mode=str(lm), subset=sub, outcome=outcome, n_rows=len(sel), point=pt, delta=deltas(pt, outcome),
                               pair_role='primary' if pair == K.PAIRS[0] else 'identity_companion_not_independent')
                    if draws is not None:
                        reps = {k: [] for k in rec['delta']}
                        nn = {k: 0 for k in rec['delta']}
                        for d in draws[:n_boot]:
                            dd = deltas(fit_bundles(sel, outcome, d), outcome)
                            for k, v in dd.items():
                                if v is None:
                                    nn[k] += 1
                                else:
                                    reps[k].append(v)
                        rec['delta_interval'] = {k: S.percentile_interval(reps[k], nn[k]) for k in reps}
                    out.append(rec)
    return out


# ------------------------------------------------------------------ identifiability (input only)
def rank_report(rows) -> dict:
    """Design ranks/conditions by subset and lead mode, plus the D04 C fixed-lead alias check (Lk in span[1, log pi])."""
    rep = {}
    for sub in SUBSETS:
        for lm in LEAD_MODES:
            sel = subset_rows(rows, sub, K.PAIRS[0], lm)
            if not sel:
                continue
            r = {}
            for b in BUNDLES:
                names, X, note = design(sel, b)
                Xs, _ = _standardize(X)
                s = np.linalg.svd(Xs, compute_uv=False)
                r[b] = dict(columns=names, rank=int(np.sum(s > RANK_TOL * s[0])), p=len(names), min_over_max_singular=float(s[-1] / s[0]), note=note)
            B = np.column_stack([np.ones(len(sel)), [x['log_pi'] for x in sel]])
            lk = np.array([x['Lk'] for x in sel])
            res = lk - B @ np.linalg.lstsq(B, lk, rcond=None)[0]
            r['Lk_residual_on_1_logpi_rel'] = float(np.linalg.norm(res) / max(np.linalg.norm(lk - lk.mean()), 1e-300))
            rep[f'{sub}|{lm}'] = r
    return rep


def input_rows_without_outcomes() -> list[dict]:
    raw = [dict(source_phase=u['source_phase'], case_id=u['case_id'], pair_id=p, lead_day=k) for u in K.b_universe() for p in K.PAIRS for k in LEADS]
    return analysis_rows(raw)


ROW_HEADERS = ('source_phase', 'case_id', 'experiment_group', 'realization', 'sid', 'd04c', 'pair_id', 'lead_day', 'R_days', 't95_d', 'SR', 'pi_r',
               'log_SR', 'log_pi', 'z0', 'Lrho', 'LRk', 'Lk', 'mag_valid', 'mag_reason', 'y_mag', 'sign_valid', 'sign_reason', 'y_sign')


def p2_identity(rows) -> dict:
    by = {(r['source_phase'], r['case_id'], r['lead_day'], r['pair_id']): r for r in rows}
    diffs, signs = [], 0
    for (ph, cid, k, p), r in by.items():
        if p != K.PAIRS[0]:
            continue
        q = by.get((ph, cid, k, K.PAIRS[1]))
        if q is None:
            continue
        if r['y_mag'] is not None and q['y_mag'] is not None:
            diffs.append(abs(r['y_mag'] - q['y_mag']))
        signs += int(r['y_sign'] != q['y_sign'])
    return dict(max_abs_log_ratio_difference=max(diffs) if diffs else None, sign_outcome_mismatches=signs)


def comparison_records(res):
    """Flatten saved fits only. Preserve generic metric and explicit outcome scales."""
    return [dict(pair_id=c['pair_id'], lead_mode=c['lead_mode'], subset=c['subset'], outcome=c['outcome'], n_valid=c['point']['n_valid_rows'],
                 **{f'{b}_metric': _metric(c['point'][b], c['outcome']) for b in BUNDLES}, **{f'{b}_status': c['point'][b]['status'] for b in BUNDLES},
                 **{f'{b}_rank': c['point'][b]['rank'] for b in BUNDLES},
                 **{f'{b}_{key}': c['point'][b].get(key) for b in BUNDLES for key in ('r2', 'sse', 'log_likelihood', 'deviance', 'mcfadden_r2')},
                 **{f'delta_{k}': v for k, v in c['delta'].items()},
                 **{f'delta_{k}_lo': (c.get('delta_interval') or {}).get(k, {}).get('low') for k in c['delta']},
                 **{f'delta_{k}_hi': (c.get('delta_interval') or {}).get(k, {}).get('high') for k in c['delta']}) for c in res['comparisons']]


def write_comparison_table(res, path):
    """Union of every record's fields; DictWriter retains its strict extras policy."""
    flat = comparison_records(res)
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(flat)
    return len(flat)


def validate_saved_results(out_dir, digest):
    """Read-only readiness check. Does not fit, regenerate draws, or authorize production."""
    receipt = json.loads((K.P5 / 'execution_B_timescale/COMPONENT_RECEIPT.json').read_text())
    for name in ('B_fits.json', 'B_timescale_rows.csv', 'B_CASES.json', 'D04C_SUBSET.json'):
        key = K.rel(out_dir / name)
        if K.sha(out_dir / name) != receipt['output_hashes'][key]:
            raise ValueError('saved B source changed: ' + name)
    for key, value in receipt['input_hashes'].items():
        if K.sha(K.ROOT / key) != value:
            raise ValueError('original B input changed: ' + key)
    for key, value in receipt['code_hashes'].items():
        if key != 'lib/p5_timescale.py' and K.sha(K.ROOT / key) != value:
            raise ValueError('original B computational source changed: ' + key)
    res = json.loads((out_dir / 'B_fits.json').read_text())
    with open(out_dir / 'B_timescale_rows.csv', newline='') as fh:
        rows = list(csv.DictReader(fh))
    universe = K.b_universe()
    expected = {(u['source_phase'], u['case_id'], pair, k) for u in universe for pair in K.PAIRS for k in LEADS}
    keys = [(r['source_phase'], r['case_id'], r['pair_id'], int(r['lead_day'])) for r in rows]
    if res['protocol_sha256'] != digest or res['n_rows'] != 7800 or len(keys) != 7800 or len(set(keys)) != 7800 or set(keys) != expected:
        raise ValueError('saved protocol/row identities mismatch')
    comps = res['comparisons']
    expected_comps = {(p, str(k), sub, outcome) for p in K.PAIRS for k in LEAD_MODES for sub in SUBSETS for outcome in ('magnitude', 'sign')}
    actual_comps = [(c['pair_id'], c['lead_mode'], c['subset'], c['outcome']) for c in comps]
    if len(comps) != 36 or set(actual_comps) != expected_comps:
        raise ValueError('saved comparison coverage mismatch')
    nfits = nci = 0
    for c in comps:
        if any(b not in c['point'] for b in BUNDLES):
            raise ValueError('saved bundle missing')
        nfits += len(BUNDLES)
        delta_keys = {'ii-i', 'iii-i', 'ii-iii'}
        if c['outcome'] == 'sign':
            delta_keys |= {'mcfadden_' + k for k in delta_keys}
        if set(c['delta']) != delta_keys or set(c['delta_interval']) != delta_keys:
            raise ValueError('saved delta/CI coverage mismatch')
        for ci in c['delta_interval'].values():
            if ci['n_identified'] + ci['n_not_identified'] != 999:
                raise ValueError('saved bootstrap accounting mismatch')
            nci += 1
    if nfits != 108 or nci != 162 or not res['d04c_source_gate']['passed']:
        raise ValueError('saved fits/intervals/historical source gate mismatch')
    for name, value in res['identity_manifests'].items():
        if K.sha(out_dir / name) != value:
            raise ValueError('saved identity manifest changed')
    stats = json.loads((K.P4 / 'statistics.json').read_text())['bootstrap']
    matrix = np.asarray(stats['draw_matrix'], dtype='<i8')
    npy = K.P4 / 'bootstrap_draw_matrix.npy'
    draws = res['draws']
    if (matrix.shape != (999, 10) or draws['shape'] != [999, 10]
            or K.sha_bytes(matrix.tobytes()) != stats['draw_matrix_sha256']
            or stats['draw_matrix_sha256'] != draws['logical_sha256_le_i8']
            or K.sha(npy) != draws['physical_npy_sha256']
            or not np.array_equal(np.load(npy).astype('<i8'), matrix)):
        raise ValueError('saved shared-draw hash mismatch')
    return res, dict(n_rows=len(rows), n_cases=len(universe), n_comparisons=len(comps), n_bundle_fits=nfits, n_intervals=nci,
                     shared_draws=999, protocol_sha256=digest, original_inputs_unchanged=True)


def finalize_existing(out_dir=K.P5 / 'B'):
    """Gate-bound serialization only; saved fits/rows stay byte-identical. No fit/draw loop."""
    digest = K.require_frozen()
    res, checks = validate_saved_results(out_dir, digest)
    preserved = {name: K.sha(out_dir / name) for name in ('B_fits.json', 'B_timescale_rows.csv')}
    path = out_dir / 'B_comparisons.csv'
    pending = path.with_suffix('.csv.pending')
    try:
        write_comparison_table(res, pending)
        if any(K.sha(out_dir / name) != value for name, value in preserved.items()):
            raise ValueError('saved B results changed during finalization')
        K.require_frozen()
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    receipt = dict(status='complete_saved_B_finalization', checks=checks, preserved_hashes=preserved,
                   comparison_csv_sha256=K.sha(path), source_sha256=K.sha(Path(__file__)), scientific_refits=0, bootstrap_reruns=0)
    (out_dir / 'FINALIZATION_RECEIPT.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


def run_official(out_dir: Path = K.P5 / 'B', n_boot=None):
    """Official B: complete phase5 freeze, strict 7800-row collection, shared draws, all comparisons, D04 C source gate."""
    digest = K.require_frozen()
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = write_identity_manifests(out_dir)
    gate = d04c_source_gate()
    if not gate['passed']:
        raise ValueError('D04 C historical source gate failed: ' + json.dumps(gate))
    import p5_collect
    rows = analysis_rows(p5_collect.b_raw_rows(strict=True))
    if len(rows) != K.B_UNIVERSE_COUNT * 4:
        raise ValueError('B rows != 7800')
    dr = shared_draws()
    res = dict(protocol_sha256=digest, n_rows=len(rows), draws={k: v for k, v in dr.items() if k != 'draws'}, ranks=rank_report(rows),
               p2_identity=p2_identity(rows), comparisons=compare(rows, dr['draws'], n_boot),
               identity_manifests=ident, d04c_source_gate=gate,
               historical_D04C_source='results/phase3/READOUT.json C.descriptive_collapse; lib/p3_phase3_analysis.py design_c (nominal rho)')
    with open(out_dir / 'B_timescale_rows.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=ROW_HEADERS, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    (out_dir / 'B_fits.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    write_comparison_table(res, out_dir / 'B_comparisons.csv')
    K.require_frozen()
    return res


# ------------------------------------------------------------------ D04 C historical source gate (PROTOCOL_DRAFT_AUDIT F_B1)
READOUT = K.P3 / 'READOUT.json'


def d04c_source_gate(tol=1e-9) -> dict:
    """Reproduce the versioned D04 C descriptive collapse (p3_phase3_analysis.design_c: OLS of log(W/|E|) on log SR,
    log rho_NOMINAL, log pi_r per lead/pair over the 90 C cases) from the saved rows, against READOUT.json C. This is a
    source-identity gate on an already published number, not a D06 fit: only the reproduced R2 and pass/fail leave it.
    The D06 main bundles keep realized R/t95 (Lrho); the D04 C fixed-lead (i) R2 is reported under both definitions."""
    import p5_collect as C
    ref = json.loads(READOUT.read_text())['C']['descriptive_collapse']
    uni = {u['case_id']: u for u in K.b_universe() if u['d04c']}
    rows = [r for r in C.read_csv(K.P4 / 'primary_rows.csv') if r['source_phase'] == 'phase3' and r['case_id'] in uni and r['track'] == 'raw'
            and r['quantity'] == 'lead']
    out = {}
    for key, want in ref.items():
        lead, pair = key.split('|')
        part = [r for r in rows if int(float(r['lead_day'])) == int(lead) and r['pair_id'] == pair]
        W = np.array([C._f(r['W_m']) for r in part], float)
        E = np.array([C._f(r['E_true_m']) for r in part], float)
        ok = np.isfinite(W) & (W > 0) & np.isfinite(E) & (E != 0)
        X = np.column_stack([np.ones(len(part)), np.log([float(r['SR']) for r in part]), np.log([float(r['rho']) for r in part]),
                             np.log([uni[r['case_id']]['pi_r'] for r in part])])
        assert all(float(r['rho']) == uni[r['case_id']]['rho_nominal'] for r in part), 'primary-row rho is not the nominal rho'
        f = ols_svd(np.log(W[ok] / np.abs(E[ok])), X[ok], None, ['intercept', 'log_SR', 'log_rho_nominal', 'log_pi_r'])
        out[key] = dict(n=int(ok.sum()), n_cases=len({r['case_id'] for r in part}), r2_reproduced=f['r2'], r2_readout=want['r2'],
                        abs_diff=None if f['r2'] is None else abs(f['r2'] - want['r2']), rank=f['rank'])
    passed = all(v['abs_diff'] is not None and v['abs_diff'] <= tol and v['n_cases'] == 90 for v in out.values())
    return dict(passed=passed, tol=tol, source=K.rel(READOUT), source_sha256=K.sha(READOUT), rows_source='results/phase4/primary_rows.csv', gate=out)


# ------------------------------------------------------------------ interface names (IMPLEMENTATION_INTERFACES draft)
def build_design(rows, bundle, pooled_leads=True):
    names, X, note = design(rows, bundle)
    return X, names, [note] if note else []


fit_ols, fit_logit = ols_svd, logit_estimable


def fit_compare(rows, draws, n_boot=None):
    return compare(rows, draws, n_boot)


def write_identity_manifests(out_dir: Path) -> dict:
    """B_CASES.json (1950) and D04C_SUBSET.json (90): explicit (source_phase, case_id) identity/covariate-source records."""
    uni = K.b_universe()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'B_CASES.json').write_text(json.dumps(dict(n=len(uni), records=uni), indent=0) + '\n')
    d4 = [u for u in uni if u['d04c']]
    (out_dir / 'D04C_SUBSET.json').write_text(json.dumps(dict(n=len(d4), source='phase3 experiment_group C', records=d4), indent=0) + '\n')
    return {p: K.sha(out_dir / p) for p in ('B_CASES.json', 'D04C_SUBSET.json')}


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('production', 'input_ranks', 'source_gate', 'finalize_existing'))
    a = ap.parse_args()
    if a.mode == 'production':
        run_official()
    elif a.mode == 'finalize_existing':
        print(json.dumps(finalize_existing()))
    elif a.mode == 'source_gate':
        g = d04c_source_gate()
        K.EVIDENCE.mkdir(parents=True, exist_ok=True)
        (K.EVIDENCE / 'B_D04C_SOURCE_GATE.json').write_text(json.dumps(g, indent=1) + '\n')
        print(json.dumps(g))
    else:
        rows = input_rows_without_outcomes()
        rep = dict(note='input-only design identifiability on the actual 1950-case universe; no W/E/sign outcome read; not scoring',
                   n_rows=len(rows), n_cases=len({(r['source_phase'], r['case_id']) for r in rows}),
                   n_R0_cases=len({(r['source_phase'], r['case_id']) for r in rows if r['z0'] == 1}),
                   n_D04C_cases=len({r['case_id'] for r in rows if r['d04c']}), ranks=rank_report(rows),
                   draws={k: v for k, v in shared_draws().items() if k != 'draws'})
        K.EVIDENCE.mkdir(parents=True, exist_ok=True)
        (K.EVIDENCE / 'B_INPUT_RANK_CHECKS.json').write_text(json.dumps(rep, indent=1) + '\n')
        print(json.dumps({k: v for k, v in rep.items() if k != 'ranks'}))


if __name__ == '__main__':
    _cli()
