"""D06 A: the 2160 D05 operating-calendar cases regenerated at r = 20 m (only geometry-derived quantities change).

Each new case is built from its hash-verified D05 B twin (results/phase4/cases): pumping context, the four future
schedules, rain, dates, natural head h_nat, AR(1) noise eps, realization/seeds/climate provenance, Q_pre = 100 and the
origin-day calendar rate are copied byte for byte. The pumping head is recomputed with the FROZEN p4_cases._physics
(hidden constant prehistory, exact Hantush block convolution, delta rule) on an r = 20 m layer, so h_star, head_context,
E_true and delta change only through r -> (b, lambda, gain, unit kernel, t95, pi_r, SR).

Guards against the two source traps identified in the design:
- p3_phase2_cases.layer caches by sid only and would silently return the r = 200 m kernel: never called here; the
  geometry cache below is keyed by the full physical tuple (storage_type, S, T, c, r, Q).
- p4_cases.build_B asserts the r = 200 m calendar t95: not called for r = 20; its feature code (p4_calendar.case_features,
  p4_cases.observed_rest_features) is reused with the r = 20 m t95.
Every parent is also re-derived through the frozen r = 200 m path from the same copied arrays and must reproduce the
stored D05 truth exactly, proving that the reused arrays and routine are the D05 generator.

Modes: --mode prefreeze_input builds all 2160 in memory, runs every identity/geometry/export check (FIXTURE_DIGEST stamp)
and writes only evidence under results/phase5/implementation_A_B/. --mode production requires the complete phase5
freeze and writes results/phase5/cases/{tf_inputs,truth}, derived_manifest.json, tool_inputs_H10/H30.npz, SHARD_*.json
and generation_checks.json. Optional --regen-parents re-runs the frozen D05 background/calendar generator (p4_cases.
all_B_cases) and requires byte equality with the stored parents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import p3_generator as g
import p3_phase2_physmap as pm
import p4_calendar as cal
import p4_cases as c4
from p3_kernels import hantush_block
from p3_make_cases import TRUTH_KEYS, load_tf_input, save_tf, save_truth
import p5_contracts as K

CTX, HMAX, PAIRS = g.CTX, g.HMAX, K.PAIRS
QPRE = 100.0
GEOMETRY_FIELDS = ('r_m', 'b', 'lambda_m', 'gain_m_per_m3d', 't50_d', 't95_d', 'pi_r', 'SR', 'SR_origin', 'SR_context_mean', 'lead_over_t95_10',
                   'lead_over_t95_30', 'rho_last_off', 'rho_last_off_projection', 'rho_longest_off')
IDENTITY_FIELDS = ('case_id', 'source_phase', 'experiment_group', 'tf_input', 'tf_sha256', 'truth', 'truth_sha256', 'census', 'shard_actor')
TF_BYTE_KEYS = ('rain', 'pumping_context', 'dates')

_GEOM: dict = {}


def stratum(sid: str, r: float) -> pm.Stratum:
    st = c4.strata_by_sid()[sid]
    return pm.Stratum(st.storage_type, st.S, st.T, st.c, float(r), st.Q)


def geometry(st: pm.Stratum) -> dict:
    """Layer record in the p3_phase2_cases.layer format, cached by the FULL physical tuple (never by sid)."""
    key = (st.storage_type, st.S, st.T, st.c, st.r, st.Q)
    if key not in _GEOM:
        d = pm.describe(st)
        B, S = hantush_block(d['a'], d['b'], CTX + HMAX)
        _GEOM[key] = dict(st=st, d=d, t95=d['t95'], B=B[0], S=S[0], key=key)
    return _GEOM[key]


def _bg(h_nat, eps, sigma_bg):
    pad = np.zeros(g.GEN_WARMUP)
    return dict(h_nat=np.concatenate([pad, h_nat]), eps=np.concatenate([pad, eps]), sigma_bg=sigma_bg)


def load_parent(pid: str, man_row: dict, verify_hash: bool = True) -> tuple[dict, dict]:
    tfp, trp = K.P4 / 'cases' / man_row['tf_input'], K.P4 / 'cases' / man_row['truth']
    if verify_hash and (K.sha(tfp) != man_row['tf_sha256'] or K.sha(trp) != man_row['truth_sha256']):
        raise PermissionError('D05 parent file differs from its manifest hash: ' + pid)
    ti = load_tf_input(tfp)
    with np.load(trp, allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    if ti['case_id'] != pid or str(tr['case_id']) != pid:
        raise ValueError('parent identity mismatch ' + pid)
    return ti, tr


def truth_in_family(ti: dict, theta: dict, t95: float, b_log10_min: float) -> dict:
    y = np.asarray(ti['head_context'], float)
    pos = np.asarray(ti['pumping_context'], float)
    pos = pos[pos > 0]
    amax = 10.0 * np.ptp(y) / float(np.median(pos)) if pos.size else float('nan')
    eb = 10.0 * np.ptp(y)
    lb = math.log10(theta['b'])
    return dict(log10_b=lb, log10_b_in_box=bool(b_log10_min - 1e-12 <= lb <= math.log10(25.0)), log10_b_on_lower_face=bool(abs(lb - b_log10_min) <= 1e-9),
                log10_a_in_box=bool(0.0 <= math.log10(theta['a']) <= 3.5), t95_le_1000=bool(t95 <= 1000.0),
                natural_in_box=bool(0.5 <= theta['n'] <= 5.0 and 1.0 <= theta['theta'] <= 100.0 and 5.0 <= theta['tau'] <= 500.0),
                A_over_Amax=float(theta['A'] / amax) if pos.size else None, eta_pump_over_bound=float(abs(theta['eta_pump_true']) / eb),
                context_has_positive_Q=bool(pos.size), all=bool(pos.size and b_log10_min - 1e-12 <= lb <= math.log10(25.0) and theta['A'] <= amax
                                                            and abs(theta['eta_pump_true']) <= eb and t95 <= 1000.0))


def build_from_parent(pid: str, man_row: dict, creal: dict, r: float = K.R_NEW_M, verify_hash: bool = True) -> tuple[dict, dict]:
    """Returns (case, checks). case has the p4_cases.build_B layout (case_id, meta, tf_input, truth, derived, base_N)."""
    ti, tr = load_parent(pid, man_row, verify_hash)
    meta0 = ti['meta']
    form, rr = meta0['calendar_form'], int(meta0['realization'])
    origin = date.fromisoformat(meta0['origin_date'])
    Qw = np.asarray(ti['pumping_context'], float)
    q_origin = float(meta0['q_origin_m3d'])
    rain_origin = float(ti['rain'][CTX])
    if q_origin != c4.declared_origin_rate(form, rr, meta0['origin_date'], rain_origin):
        raise ValueError('parent origin rate differs from the frozen calendar rule ' + pid)
    theta0 = json.loads(str(tr['theta_true_json']))
    h_nat, eps = np.asarray(tr['h_nat'], float), np.asarray(tr['eps'], float)
    bg = _bg(h_nat, eps, theta0['sigma_bg'])
    sid = man_row['sid']
    L200 = c4.physical(c4.strata_by_sid()[sid])  # frozen r=200 m layer (sid cache is correct for r=200 only)
    _f, _hn, _e, hp200, hs200, E200, d200 = c4._physics(bg, L200, Qw, q_origin, QPRE)
    st = stratum(sid, r)
    L = geometry(st)
    fut, hn, ep, hp_a, h_star, E, delta = c4._physics(bg, L, Qw, q_origin, QPRE)
    A, t95, sig = L['d']['gain'], L['t95'], theta0['sigma_bg']
    cid = K.new_id(pid)
    obs = c4.observed_rest_features(Qw, t95, origin - timedelta(days=1))
    feat = cal.case_features(form, creal, dict(sid=sid, storage_type=st.storage_type, t95_d=t95), origin, Qw, q_origin, float(Qw[-1]))
    V = float(Qw.sum())
    derived = {k: v for k, v in man_row.items() if k not in IDENTITY_FIELDS}
    derived.update(case_id=cid, source_phase='phase5', experiment_group=K.A_GROUP, twin_case_id=pid, parent_source_phase='phase4', parent_case_id=pid,
                   distance_m=float(r), r_m=float(r), b=L['d']['b'], lambda_m=L['d']['lam'], gain_m_per_m3d=A, t50_d=L['d']['t50'], t95_d=t95,
                   pi_r=L['d']['pi_r'], SR=A * q_origin / sig, SR_origin=A * q_origin / sig, SR_context_mean=A * V / CTX / sig,
                   lead_over_t95_10=10 / t95, lead_over_t95_30=30 / t95, rho_last_off=obs['rho_last_off'],
                   rho_last_off_projection=obs['rho_last_off'], rho_longest_off=obs['rho_longest_off'],
                   family_b_log10_min=K.B_LOG10_MIN_NEW, geometry_cache_key=list(L['key']))
    theta = dict(theta0, a=L['d']['a'], b=L['d']['b'], A=A, eta_pump_true=A * (QPRE - Qw[0]), r=float(r))
    meta = dict(meta0, experiment_group=K.A_GROUP, r_m=float(r), signal_ratio=float(derived['SR']), pi_layer=float(L['d']['pi_r']),
                t95_days=t95, rest_ratio=float(obs['rho_last_off']), twin_case_id=pid)
    tf_input = dict(case_id=cid, head_context=(h_star + eps)[:CTX], rain=ti['rain'], pumping_context=Qw, future_Q={k: fut_v for k, fut_v in
                    ((f'{p}_{s}', q) for p, (qa, qb) in fut.items() for s, q in (('a', qa), ('b', qb)))}, dates=ti['dates'], meta=meta)
    truth = dict(case_id=cid, h_star_a=h_star, h_nat=hn, h_pump_a=hp_a, eps=ep, E_true=E, delta=delta, theta_true=theta)
    tif = truth_in_family(tf_input, theta, t95, K.B_LOG10_MIN_NEW)
    derived['truth_in_family'] = tif
    ck = dict(
        twin_bytes_identical=bool(all(np.array_equal(ti[k], tf_input[k]) for k in TF_BYTE_KEYS)
                                  and all(np.array_equal(ti['future_Q'][k], tf_input['future_Q'][k]) for k in ti['future_Q'])
                                  and set(ti['future_Q']) == set(tf_input['future_Q'])
                                  and np.array_equal(hn, h_nat) and np.array_equal(ep, eps)),
        r200_route_reproduces_parent=bool(np.array_equal(hp200, tr['h_pump_a']) and np.array_equal(hs200, tr['h_star_a'])
                                          and all(np.array_equal(E200[p], tr['E_true__' + p]) and np.array_equal(d200[p], tr['delta__' + p]) for p in PAIRS)
                                          and np.array_equal((hs200 + eps)[:CTX], ti['head_context'])),
        head_context_changed=bool(not np.array_equal(tf_input['head_context'], ti['head_context'])),
        kernel_b_exact=bool(abs(L['d']['b'] - r * r / (4 * st.T * st.c)) <= 1e-15 * max(1.0, L['d']['b'])),
        geometry_is_r=bool(L['key'][4] == float(r) and L['d']['b'] != man_row['b']),
        calendar_features_invariant=bool(all(feat[k] == man_row[k] for k in ('last_off_length_days', 'last_off_distance_to_origin_days', 'n_zero_days',
                                                                           'q_origin_m3d', 'q_last_context_day_m3d', 'offseason_use_variant', 'origin_month'))
                                         and feat['n_on_off_switches'] == man_row['N_on_off_observed']
                                         and all(obs[k] == man_row[k] for k in ('longest_off_length_days', 'longest_off_distance_to_origin_days',
                                                                                'N_positive_rate_changes_observed', 'N_on_off_observed'))),
        origin_rate_rule=True,
        zero_origin_exact_zero=bool(q_origin != 0 or all(np.all(E[p] == 0) for p in PAIRS)),
        pair_identity_maxabs_m=float(np.max(np.abs(E[PAIRS[1]] + 0.5 * E[PAIRS[0]]))),
        nongeometry_derived_equal_parent=bool(all(derived[k] == man_row[k] for k in man_row if k not in IDENTITY_FIELDS + GEOMETRY_FIELDS)),
        meta_only_allowed_changes=sorted(k for k in set(meta) | set(meta0) if meta.get(k) != meta0.get(k)),
        theta_only_allowed_changes=sorted(k for k in theta if theta[k] != theta0.get(k)),
        truth_in_family=tif['all'])
    case = dict(case_id=cid, meta=meta, tf_input=tf_input, truth=truth, derived=derived, base_N=Qw)
    return case, ck


ALLOWED_META = {'experiment_group', 'r_m', 'signal_ratio', 'pi_layer', 't95_days', 'rest_ratio', 'twin_case_id'}
ALLOWED_THETA = {'a', 'b', 'A', 'eta_pump_true', 'r'}


def content_digest(case: dict) -> str:
    h = hashlib.sha256()
    ti, tr = case['tf_input'], case['truth']
    for a in (ti['head_context'], ti['rain'], ti['pumping_context'], *[ti['future_Q'][k] for k in sorted(ti['future_Q'])], tr['h_star_a'], tr['h_pump_a'],
              *[tr['E_true'][p] for p in PAIRS]):
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    h.update(json.dumps(case['meta'], sort_keys=True).encode())
    return h.hexdigest()


def build_all(verify_hash: bool = True, only=None, progress=None):
    man = {m['case_id']: m for m in json.loads((K.P4 / 'cases/derived_manifest.json').read_text())}
    reals = {int(r['realization']): r for r in cal.load_pilot_realizations()}
    cases, checks = [], {}
    for i, pid in enumerate(K.parent_ids()):
        if only is not None and K.new_id(pid) not in only:
            continue
        c, ck = build_from_parent(pid, man[pid], reals[int(man[pid]['realization'])], verify_hash=verify_hash)
        cases.append(c)
        checks[c['case_id']] = ck
        if progress and (i + 1) % 360 == 0:
            progress(f'{i + 1} built')
    return cases, checks, man


def manifest_row(c, tf, tr, out):
    return dict(c['derived'], census=None, tf_input=str(tf.relative_to(out)), tf_sha256=K.sha(tf), truth=str(tr.relative_to(out)),
                truth_sha256=K.sha(tr), shard_actor=K.actor_of(c['case_id']))


def generation_checks(cases, checks, man, digest) -> dict:
    ids = [c['case_id'] for c in cases]
    zero = {c['case_id'] for c in cases if c['derived']['q_origin_m3d'] == 0}
    variant = {c['case_id'] for c in cases if c['derived'].get('offseason_use_variant')}
    sh = K.shards()
    by_layer = {}
    for c in cases:
        d = c['derived']
        by_layer.setdefault(d['sid'], dict(b20=d['b'], gain20=d['gain_m_per_m3d'], t95_20=d['t95_d'], pi20=d['pi_r'],
                                           gain_ratio_20_over_200=d['gain_m_per_m3d'] / man[d['twin_case_id']]['gain_m_per_m3d'],
                                           b200=man[d['twin_case_id']]['b'], t95_200=man[d['twin_case_id']]['t95_d'],
                                           log10_b20_on_new_lower_face=d['truth_in_family']['log10_b_on_lower_face'],
                                           log10_b20_outside_old_box=bool(math.log10(d['b']) < K.B_LOG10_MIN_OLD - 1e-12)))
    agg = lambda k: sum(1 for v in checks.values() if v[k] is True)  # noqa: E731
    out = dict(n_cases=len(cases), unique=len(set(ids)) == len(ids), ids_match_identity=set(ids) == set(K.new_case_ids()),
               zero_origin_cases=len(zero), water_curtain_variant_cases=len(variant),
               actors={a: K.shard_balance(v, zero, variant) for a, v in sh.items()},
               twin_bytes_identical=agg('twin_bytes_identical'), r200_route_reproduces_parent=agg('r200_route_reproduces_parent'),
               head_context_changed=agg('head_context_changed'), kernel_b_exact=agg('kernel_b_exact'), geometry_is_r=agg('geometry_is_r'),
               calendar_features_invariant=agg('calendar_features_invariant'), zero_origin_exact_zero=agg('zero_origin_exact_zero'),
               nongeometry_derived_equal_parent=agg('nongeometry_derived_equal_parent'), truth_in_family=agg('truth_in_family'),
               meta_changes_allowed=all(set(v['meta_only_allowed_changes']) <= ALLOWED_META for v in checks.values()),
               theta_changes_allowed=all(set(v['theta_only_allowed_changes']) <= ALLOWED_THETA for v in checks.values()),
               pair_identity_maxabs_m=max(v['pair_identity_maxabs_m'] for v in checks.values()),
               geometry_by_layer=by_layer, max_A_over_Amax=max(c['derived']['truth_in_family']['A_over_Amax'] for c in cases),
               max_eta_pump_over_bound=max(c['derived']['truth_in_family']['eta_pump_over_bound'] for c in cases),
               protocol_sha256=digest, official=K.official_digest(digest))
    n = len(cases)
    out['all_ok'] = bool(n == K.A_COUNT and out['unique'] and out['ids_match_identity'] and out['zero_origin_cases'] == 918 and len(variant) == 216
                         and all(out[k] == n for k in ('twin_bytes_identical', 'r200_route_reproduces_parent', 'head_context_changed', 'kernel_b_exact',
                                                       'geometry_is_r', 'calendar_features_invariant', 'zero_origin_exact_zero',
                                                       'nongeometry_derived_equal_parent', 'truth_in_family'))
                         and out['meta_changes_allowed'] and out['theta_changes_allowed'] and out['pair_identity_maxabs_m'] < 1e-12
                         and all(a['n'] == K.A_PER_ACTOR and a['cell_min_max'] == [5, 5] and a['zero_origin'] == 459 and a['water_curtain_variant'] == 108
                                 for a in out['actors'].values()))
    return out


def export_checks(cases, digest) -> dict:
    import p5_export as ex
    res = {}
    for H in (10, 30):
        arr = ex.export_arrays(cases, H, digest)
        n = len(arr['query_id'])
        res[H] = dict(n_queries=n, n_cases=len(set(arr['case_id'].tolist())), per_case=n // max(1, len(set(arr['case_id'].tolist()))),
                      shapes={k: list(arr[k].shape) for k in ('head', 'pumping', 'rainfall')}, truth_keys=sorted(set(arr) & TRUTH_KEYS),
                      sha256_of_arrays=hashlib.sha256(b''.join(np.ascontiguousarray(arr[k]).tobytes() for k in ('head', 'pumping', 'rainfall'))).hexdigest(),
                      per_actor={a: len(ex.subset(arr, v)['query_id']) for a, v in K.shards().items()})
        del arr
    return res


def regen_parents(only=None) -> dict:
    """Frozen D05 generator (p4_cases.all_B_cases, requires the D05 freeze) vs stored parent npz, byte for byte."""
    man = {m['case_id']: m for m in json.loads((K.P4 / 'cases/derived_manifest.json').read_text())}
    want = None if only is None else {K.twin_of(c) for c in only}
    n = bad = 0
    bads = []
    for c in c4.all_B_cases(fixture=False, only=want):
        ti, tr = load_parent(c['case_id'], man[c['case_id']])
        ok = (np.array_equal(ti['head_context'], c['tf_input']['head_context']) and all(np.array_equal(ti[k], c['tf_input'][k]) for k in TF_BYTE_KEYS)
              and np.array_equal(tr['h_nat'], c['truth']['h_nat']) and np.array_equal(tr['eps'], c['truth']['eps'])
              and all(np.array_equal(tr['E_true__' + p], c['truth']['E_true'][p]) for p in PAIRS))
        n += 1
        if not ok:
            bad += 1
            bads.append(c['case_id'])
    return dict(n_regenerated=n, n_byte_identical=n - bad, mismatches=bads[:20])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('prefreeze_input', 'production'))
    ap.add_argument('--regen-parents', action='store_true', help='also regenerate every D05 parent from the frozen background/calendar path')
    a = ap.parse_args()
    t0 = time.time()
    digest = K.resolve_mode(a.mode)
    log = lambda m: print(f'[{time.time() - t0:7.1f}s] {m}', flush=True)  # noqa: E731
    cases, checks, man = build_all(progress=log)
    gen = generation_checks(cases, checks, man, digest)
    log('generation checks all_ok=' + str(gen['all_ok']))
    gen['export'] = export_checks(cases, digest)
    gen['export_ok'] = all(v['n_queries'] == 8 * K.A_COUNT // 2 and v['per_case'] == 4 and not v['truth_keys']
                           and all(n == 4 * K.A_PER_ACTOR for n in v['per_actor'].values()) for v in gen['export'].values())
    if a.regen_parents:
        gen['regen_parents'] = regen_parents()
        gen['regen_ok'] = gen['regen_parents']['n_byte_identical'] == K.A_COUNT
        log('regen ' + json.dumps(gen['regen_parents']))
    gen['runtime_s'] = time.time() - t0
    if a.mode == 'prefreeze_input':
        K.EVIDENCE.mkdir(parents=True, exist_ok=True)
        gen['note'] = 'prefreeze input-only identity/geometry/export check; FIXTURE_DIGEST stamp; no official file written; no W/model/scoring'
        (K.EVIDENCE / 'A_PREFREEZE_INPUT_CHECKS.json').write_text(json.dumps(gen, indent=1, default=str) + '\n')
        (K.EVIDENCE / 'A_PREFREEZE_CONTENT_DIGESTS.json').write_text(json.dumps({c['case_id']: content_digest(c) for c in cases}, indent=0) + '\n')
        log('wrote ' + K.rel(K.EVIDENCE / 'A_PREFREEZE_INPUT_CHECKS.json'))
        if not gen['all_ok'] or not gen['export_ok']:
            raise SystemExit('PREFREEZE_INPUT_CHECK_FAILED')
        return
    # ---------------- production (complete phase5 freeze verified by resolve_mode)
    if not gen['all_ok'] or not gen['export_ok']:
        raise SystemExit('GENERATION_CHECK_FAILED before any write')
    import p5_export as ex
    out = K.P5 / 'cases'
    for sub in ('tf_inputs', 'truth'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases:
        cid = c['case_id']
        tf, tr = out / 'tf_inputs' / f'{cid}.npz', out / 'truth' / f'{cid}.npz'
        save_tf(tf, c, digest)
        save_truth(tr, c, digest)
        with np.load(tf, allow_pickle=False) as z:
            if set(z.files) & TRUTH_KEYS:
                raise ValueError('truth leakage ' + cid)
        rows.append(manifest_row(c, tf, tr, out))
    for H in (10, 30):
        np.savez_compressed(out / f'tool_inputs_H{H}.npz', **ex.export_arrays(cases, H, digest))
    (out / 'derived_manifest.json').write_text(json.dumps(rows, indent=1))
    zero = {c['case_id'] for c in cases if c['derived']['q_origin_m3d'] == 0}
    variant = {c['case_id'] for c in cases if c['derived'].get('offseason_use_variant')}
    for actor, ids in K.shards().items():
        rec = dict(actor=actor, case_ids=ids, n_cases=len(ids), rule='actor_of(d06A id) = p4_partition.actor_of(twin d05B id)',
                   case_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(), balance=K.shard_balance(ids, zero, variant),
                   outputs=dict(wb_per_case='results/phase5/wb/{W,reference,truth_eval,markers}/<case_id>.*', tools=f'results/phase5/tools/{actor}/'))
        (K.P5 / f'SHARD_{actor.upper()}.json').write_text(json.dumps(rec, indent=1) + '\n')
    gen['tool_inputs_sha256'] = {H: K.sha(out / f'tool_inputs_H{H}.npz') for H in (10, 30)}
    gen['derived_manifest_sha256'] = K.sha(out / 'derived_manifest.json')
    (out / 'generation_checks.json').write_text(json.dumps(gen, indent=1, default=str))
    K.require_frozen()
    log('D06_A_CASES_GENERATED')


if __name__ == '__main__':
    main()
