"""Synthetic/source-linked correctness tests for the D06 A/B runtime (no official W, model query or scoring).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
     env/.venv_pilot/bin/python lib/p5_A_B_tests.py [--scratch DIR] [--skip-slow]
Writes results/phase5/implementation_A_B/TEST_RESULTS.json. Scratch outputs (fixture W/reference, fixture archives)
go only to --scratch, never to results/phase5/{cases,wb,tools}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

import p5_contracts as K

RESULTS = {}


def test(fn):
    def run(*a):
        t = time.time()
        try:
            detail = fn(*a)
            RESULTS[fn.__name__] = dict(ok=True, s=round(time.time() - t, 2), detail=detail)
        except Exception:
            RESULTS[fn.__name__] = dict(ok=False, s=round(time.time() - t, 2), error=traceback.format_exc()[-3000:])
        print(('PASS ' if RESULTS[fn.__name__]['ok'] else 'FAIL ') + fn.__name__, flush=True)
    run.__name__ = fn.__name__
    return run


def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc as e:
        return str(e)[:200]
    raise AssertionError(f'{f.__name__} did not raise {exc.__name__}')


# ------------------------------------------------------------------ contracts / freeze gate
@test
def t_freeze_gate(scratch):
    out = {}
    out['real_phase5_production_refused'] = raises(PermissionError, K.resolve_mode, 'production')
    base = scratch / 'freeze_fixture'
    base.mkdir(parents=True, exist_ok=True)
    out['absent'] = raises(PermissionError, K.require_frozen, base)
    proto = base / 'protocol.md'
    proto.write_text('# fixture protocol\n')
    dig = K.sha(proto)
    (base / 'protocol.sha256').write_text(dig + '  protocol.md\n')
    pinned = scratch / 'pinned_fixture.txt'
    pinned.write_text('v1')
    closure = K.local_import_closure()
    hashes = {p: K.sha(K.ROOT / p) for p in closure + list(K.EXTRA_PINS)}
    hashes[str(pinned)] = K.sha(pinned)
    def write(h, status=K.FREEZE_STATUS):
        (base / 'protocol_freeze.json').write_text(json.dumps(dict(status=status, protocol_sha256=dig, hashes=h)))
    write(hashes, 'draft')
    out['nonterminal'] = raises(PermissionError, K.require_frozen, base)
    write(dict(hashes, **{'lib/p5_contracts.py': 'TBD'}))
    out['placeholder'] = raises(PermissionError, K.require_frozen, base)
    write({k: v for k, v in hashes.items() if k != 'lib/p5_w_family.py'})
    out['closure_omission'] = raises(PermissionError, K.require_frozen, base)
    write(hashes)
    out['complete_ok'] = K.require_frozen(base) == dig
    pinned.write_text('v2')
    out['changed_pin'] = raises(PermissionError, K.require_frozen, base)
    pinned.write_text('v1')
    proto.write_text('# fixture protocol changed\n')
    out['protocol_changed'] = raises(PermissionError, K.require_frozen, base)
    out['closure'] = closure
    out['fixture_digest_not_official'] = not K.official_digest(K.FIXTURE_DIGEST)
    assert out['complete_ok'] and out['fixture_digest_not_official']
    for need in ('lib/p5_w_family.py', 'lib/p5_run_wb.py', 'lib/p4_cases.py', 'lib/p3_kernels.py', 'lib/p4_export.py', 'lib/p4_statistics.py'):
        assert need in closure or need in K.EXTRA_PINS, need
    return out


@test
def t_identities_shards(scratch):
    ids = K.new_case_ids()
    sh = K.shards()
    man = {m['case_id']: m for m in json.loads((K.P4 / 'cases/derived_manifest.json').read_text())}
    zero = {c for c in ids if man[K.twin_of(c)]['no_active_contrast']}
    var = {c for c in ids if man[K.twin_of(c)].get('offseason_use_variant')}
    bal = {a: K.shard_balance(v, zero, var) for a, v in sh.items()}
    assert len(ids) == 2160 and all(K.actor_of(c) == man[K.twin_of(c)]['shard_actor'] for c in ids)
    assert all(b['n'] == 1080 and b['cell_min_max'] == [5, 5] and b['zero_origin'] == 459 and b['water_curtain_variant'] == 108 for b in bal.values())
    assert K.new_id(K.twin_of(ids[7])) == ids[7]
    return bal


# ------------------------------------------------------------------ isolated family adapter
@test
def t_family_isolation(scratch):
    import p5_w_family as wf
    import p3_wenvelope as fw, p3_wenvelope_repaired_v1_2 as fr, p3_pastas as fp
    before = wf.frozen_module_state()
    ns6, ns4 = wf.load(-6.0), wf.load(-4.0)
    after = wf.frozen_module_state()
    assert before == after and fw.PUMP_BOX[1, 0] == -4.0 and fr.BOX[4, 0] == -4.0 and fp.FAMILY['b'][0] == 1e-4
    assert ns6.effective['consistent'] and ns4.effective['consistent']
    assert ns6.repaired.BOX[4, 0] == -6.0 and ns6.base.PUMP_BOX[1, 0] == -6.0 and ns6.pastas.FAMILY['b'] == (1e-6, 25.0)
    assert ns6.pastas.HANTUSH_STARTS == fp.HANTUSH_STARTS and np.array_equal(ns6.base.NAT_BOX, fw.NAT_BOX) and ns6.base.T95_MAX == fw.T95_MAX
    assert ns6.repaired.WEngineV12.__mro__[1] is ns6.base.WEngine and ns6.repaired.WEngineV12 is not fr.WEngineV12
    assert raises(ValueError, wf.load, -5.0)
    for k, r in ns6.receipts.items():
        assert r['frozen_sha256'] == wf.frozen_pins()[k]
    # reference: the isolated pastas model gets the new pmin, the frozen one keeps 1e-4
    from p3_make_cases import load_tf_input
    ti = load_tf_input(K.P4 / 'cases/tf_inputs/d05B_paddy_irrigation_m06_r00_confined_T500.npz')
    h, P, Q = fp._series(ti)
    pm_new = ns6.pastas.build_model(*ns6.pastas._series(ti)).parameters.loc['well_b', 'pmin']
    pm_old = fp.build_model(h, P, Q).parameters.loc['well_b', 'pmin']
    assert pm_new == 1e-6 and pm_old == 1e-4
    return dict(effective_new=ns6.effective, diffs={k: v['diff'] for k, v in ns6.receipts.items()}, pastas_pmin=dict(new=pm_new, frozen=pm_old))


@test
def t_w_identity_regression(scratch):
    """Isolated adapter at the frozen bound reproduces the frozen engine record exactly (reduced fixture budget)."""
    import p4_run_wb as F
    import p5_run_wb as R
    from p3_make_cases import load_tf_input
    ti = load_tf_input(K.P4 / 'cases/tf_inputs/d05B_water_curtain_m01_r00_confined_T50.npz')
    small = dict(global_levels=(128, 256), n_local=64, max_local_rounds=2)
    recF = F.w.WEngineV12(ti, seed=F.W_SEED).run(**small)
    b4 = R.bind(K.P4 / 'cases', scratch / 'unused', b_log10_min=-4.0, budget=small)
    _, rec4 = b4.frozen_fn['run_W'](ti)
    drop = {'runtime_s', 'wall_s', 'checks', 'seed', 'budget'}
    clean = lambda r: json.dumps(F.jsonable({k: v for k, v in r.items() if k not in drop}), sort_keys=True)  # noqa: E731
    assert clean(recF) == clean(rec4)
    b6 = R.bind(K.P4 / 'cases', scratch / 'unused', b_log10_min=-6.0, budget=small)
    _, rec6 = b6.frozen_fn['run_W'](ti)
    return dict(identical_at_frozen_bound=True, W10_P1_frozen=recF['envelope']['P1_continue_vs_stop']['W'][9],
                W10_P1_new_box_same_case=rec6['envelope']['P1_continue_vs_stop']['W'][9], note='reduced fixture budget; not an official W')


@test
def t_reference_identity_regression(scratch):
    """Isolated reference at the frozen bound reproduces the frozen p3_pastas fit (reduced draws), and -6 sets pmin 1e-6."""
    import warnings
    import p3_pastas as fp
    import p5_w_family as wf
    import p4_run_wb as F
    from p3_make_cases import load_tf_input
    ti = load_tf_input(K.P4 / 'cases/tf_inputs/d05B_water_curtain_m01_r00_confined_T50.npz')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = fp.fit_and_propagate(ti, seed=11, n_draws=40)
        b = wf.load(-4.0).pastas.fit_and_propagate(ti, seed=11, n_draws=40)
    a.pop('_model'), b.pop('_model')
    drop = {'runtime_s', 'wall_s'}
    ja = json.dumps(F.jsonable({k: v for k, v in a.items() if k not in drop}), sort_keys=True)
    jb = json.dumps(F.jsonable({k: v for k, v in b.items() if k not in drop}), sort_keys=True)
    assert ja == jb
    return dict(identical_at_frozen_bound=True, n_keys=len(a), note='reduced draws (40); not an official reference')


@test
def t_interfaces_gate(scratch):
    import p5_timescale as B, p5_collect as C, p5_w_family as wf, p5_run_wb, p5_timesfm_execution  # noqa: F401  (load full runtime before audit)
    g = B.d04c_source_gate()
    assert g['passed'] and all(v['n_cases'] == 90 for v in g['gate'].values())
    assert K.a_case_ids() == K.new_case_ids() and K.parent_of('d06A_x') == 'd05B_x' and C.assemble_new is C.a_rows
    u = K.build_case_universe()
    assert len(u['B_CASES']) == 1950 and len(u['D04C_SUBSET']) == 90 and len(u['C_CASES']) == 6270
    assert K.verify_old() > 10000
    aud = K.import_audit({p: 'x' for p in K.local_import_closure() + list(K.EXTRA_PINS) + ['lib/p5_A_B_tests.py', 'lib/tests/p3_fixture.py']})
    assert 'lib/p5_run_wb.py' in aud and 'lib/p4_run_wb.py' in aud
    unpinned = raises(PermissionError, K.import_audit, {})
    rec = wf.family_receipt(-6.0)
    assert rec['family_version'] == 'D06_new_r20_b1e-6' and rec['bounds']['log10_b'][0] == -6.0
    ident = B.write_identity_manifests(scratch / 'B_ident')
    return dict(d04c_gate=g, audited_modules=len(aud), unpinned_refused=unpinned, identity_manifest_sha=ident)


# ------------------------------------------------------------------ geometry / generation / export
@test
def t_geometry_cache(scratch):
    import p5_distance_cases as dc
    import p3_phase2_cases as d03
    g20 = dc.geometry(dc.stratum('confined_T500', 20.0))
    g200 = dc.geometry(dc.stratum('confined_T500', 200.0))
    assert g20['key'] != g200['key'] and g20['d']['b'] == 1e-6 and abs(g200['d']['b'] - 1e-4) < 1e-18
    frozen200 = d03.layer([s for s in d03.strata() if s.sid == 'confined_T500'][0])
    assert np.array_equal(frozen200['B'], g200['B']) and frozen200['t95'] == g200['t95']
    assert all(v['st'].r == 200.0 for v in d03._LAYER_CACHE.values()), 'sid-keyed frozen cache polluted with r20'
    return dict(b20=g20['d']['b'], t95_20=g20['t95'], t95_200=g200['t95'], frozen_cache_r=sorted({v['st'].r for v in d03._LAYER_CACHE.values()}))


def _sample_cases():
    import p5_distance_cases as dc
    ids = K.new_case_ids()
    pick = {'d06A_water_curtain_m01_r00_confined_T500', 'd06A_paddy_irrigation_m06_r01_unconfined_T50', 'd06A_domestic_continuous_m12_r00_leaky_T50',
            'd06A_water_curtain_m07_r00_confined_T50', 'd06A_paddy_irrigation_m06_r00_confined_T500', 'd06A_domestic_continuous_m03_r03_unconfined_T500'}
    assert pick <= set(ids)
    return dc.build_all(only=pick)


@test
def t_generation_export(scratch):
    import p4_export as fx
    import p5_export as ex
    from p3_make_cases import TRUTH_KEYS, save_tf, save_truth
    cases, checks, man = _sample_cases()
    bad = {c: [k for k, v in ck.items() if v is False] for c, ck in checks.items()}
    assert not any(bad.values()), bad
    zero = [c for c in cases if c['derived']['q_origin_m3d'] == 0]
    d = scratch / 'gen'
    (d / 'tf_inputs').mkdir(parents=True, exist_ok=True)
    (d / 'truth').mkdir(parents=True, exist_ok=True)
    for c in cases:
        save_tf(d / 'tf_inputs' / f"{c['case_id']}.npz", c, K.FIXTURE_DIGEST)
        save_truth(d / 'truth' / f"{c['case_id']}.npz", c, K.FIXTURE_DIGEST)
        with np.load(d / 'tf_inputs' / f"{c['case_id']}.npz") as z:
            assert not set(z.files) & TRUTH_KEYS
    arr = ex.export_arrays(cases, 30, K.FIXTURE_DIGEST)
    assert arr['head'].shape == (4 * len(cases), 1024) and arr['pumping'].shape == (4 * len(cases), 1054) and not set(arr) & TRUTH_KEYS
    meta = cases[0]['meta']
    pc, rain0 = cases[0]['tf_input']['pumping_context'], cases[0]['tf_input']['rain'][1024]
    rej_frozen = raises(ValueError, fx.declared_q_origin, meta, pc, rain0)
    rej_B = raises(ValueError, ex.declared_q_origin, dict(meta, experiment_group='B'), pc, rain0)
    rej_A = raises(ValueError, ex.declared_q_origin, dict(meta, experiment_group='A'), pc, rain0)
    rej_rate = raises(ValueError, ex.declared_q_origin, dict(meta, q_origin_m3d=meta['q_origin_m3d'] + 1.0), pc, rain0)
    assert ex.declared_q_origin(meta, pc, rain0) == meta['q_origin_m3d']
    assert all(v['same_code'] and v['globals_isolated'] for v in ex.rebinding_receipt().values())
    return dict(n=len(cases), n_zero_origin=len(zero), frozen_p4_export_rejects_new_group=rej_frozen, p5_rejects_B=rej_B, p5_rejects_A_misroute=rej_A,
                p5_rejects_wrong_rate=rej_rate, archive_rows=int(arr['head'].shape[0]))


@test
def t_inputs_only_runner(scratch):
    """TimesFM runner input path on fixture archives; no torch import, no model, no query."""
    import p5_export as ex
    import p5_timesfm_execution as T
    import p5_distance_cases as dc
    cases, checks, man = _sample_cases()
    d = scratch / 'tf_fixture_cases'
    d.mkdir(parents=True, exist_ok=True)
    rows = [dict(c['derived'], tf_input=f"tf_inputs/{c['case_id']}.npz", tf_sha256='fixture', shard_actor=K.actor_of(c['case_id'])) for c in cases]
    (d / 'derived_manifest.json').write_text(json.dumps(rows, default=str))
    for H in (10, 30):
        np.savez_compressed(d / f'tool_inputs_H{H}.npz', **ex.export_arrays(cases, H, K.FIXTURE_DIGEST))
    reps = {a: T.main(['--mode', 'inputs_only', '--actor', a, '--cases-dir', str(d)]) for a in K.ACTORS}
    assert 'torch' not in sys.modules and all(r['model_loaded'] is False and r['official'] is False for r in reps.values())
    assert sum(r['n_cases'] for r in reps.values()) == len(cases) and all(r['queries_per_horizon'] == 4 * r['n_cases'] for r in reps.values())
    refused = raises(PermissionError, T.main, ['--mode', 'production', '--actor', 'shard_b'])
    return dict(reports=reps, production_refused=refused)


@test
def t_run_wb_fixture(scratch):
    """End-to-end frozen driver bodies on one r20 fixture case (b on the new lower face), reduced budget; resume test."""
    import p5_run_wb as R
    from p3_make_cases import save_tf, save_truth
    cases, checks, man = _sample_cases()
    c = [x for x in cases if x['case_id'] == 'd06A_paddy_irrigation_m06_r00_confined_T500'][0]
    cd, od = scratch / 'wb_cases', scratch / 'wb_out'
    shutil.rmtree(od, ignore_errors=True)
    for sub in ('tf_inputs', 'truth'):
        (cd / sub).mkdir(parents=True, exist_ok=True)
    tf, tr = cd / 'tf_inputs' / f"{c['case_id']}.npz", cd / 'truth' / f"{c['case_id']}.npz"
    save_tf(tf, c, K.FIXTURE_DIGEST)
    save_truth(tr, c, K.FIXTURE_DIGEST)
    (cd / 'derived_manifest.json').write_text(json.dumps([dict(c['derived'], tf_sha256=K.sha(tf))], default=str))
    args = ['--mode', 'fixture', '--cases-dir', str(cd), '--out-dir', str(od), '--fixture-budget',
            json.dumps(dict(global_levels=[128, 256], n_local=64, max_local_rounds=2)), '--fixture-draws', '50']
    R.main(args)
    cid = c['case_id']
    W = json.loads((od / 'W' / f'{cid}.json').read_text())
    ref = json.loads((od / 'reference' / f'{cid}.json').read_text())
    te = json.loads((od / 'truth_eval' / f'{cid}.json').read_text())
    mk = json.loads((od / 'markers' / f'{cid}.done').read_text())
    assert W['family_box']['b_log10_min'] == -6.0 and ref['family_box']['b_log10_min'] == -6.0 and W['family_box']['family_version'] == 'D06_new_r20_b1e-6'
    assert W['family_box']['effective']['consistent'] and mk['protocol_sha256'] == K.FIXTURE_DIGEST and mk['driver_sha256'] == R.driver_sha()
    assert mk['reference_status'] == 'ok' and mk['truth_eval_ok'] and W.get('W_status') == 'ok'
    b = R.bind(cd, od)
    assert b.fresh(cid, K.sha(tf), K.FIXTURE_DIGEST)  # resume skip with identical hashes
    assert not b.fresh(cid, K.sha(tf), 'f' * 64)        # different protocol forces recompute
    rec = R.rebinding_receipt(b)
    assert all(rec[n]['same_code'] and rec[n]['globals_isolated'] for n in R.REBOUND) and rec['w_is_isolated'] and rec['frozen_module_globals_unchanged']
    assert rec['budget'] == rec['frozen_budget'] and rec['n_draws'] == rec['frozen_n_draws'] == 1000
    env = {v: '1' for v in R.THREAD_VARS + ('PYTHONDONTWRITEBYTECODE',)}
    old = {v: os.environ.get(v) for v in env}
    os.environ.update(env)
    try:
        refused = raises(PermissionError, R.main, ['--mode', 'production', '--actor', 'shard_b'])
    finally:
        for v, x in old.items():
            os.environ.pop(v) if x is None else os.environ.__setitem__(v, x)
    bad_out = raises(SystemExit, R.main, ['--mode', 'fixture', '--cases-dir', str(cd), '--out-dir', str(K.P5 / 'wb')])
    bud_prod = raises(SystemExit, R.main, ['--mode', 'production', '--actor', 'shard_b', '--fixture-draws', '5'])
    te_tc = te.get('truth_compatibility', {})
    return dict(case=cid, W_status=W['W_status'], W_flags=W.get('flags'), tolerance=W['tolerance'], xbest_log10_b=W['tolerance']['xbest'][4],
                ref_well_b=dict(zip(ref['param_names'], ref.get('params', ref.get('popt', [])))).get('well_b') if ref.get('param_names') else None,
                truth_full_vector_in_G=te_tc.get('truth_full_vector_in_G'), production_refused=refused, fixture_out_in_phase5_refused=bad_out,
                fixture_budget_refused_in_production=bud_prod, rebinding=rec, note='reduced fixture budget/draws; outputs only under scratch')


# ------------------------------------------------------------------ B timescale
def _synthetic_b(rng, n_real=10, d04c_like=False):
    rows = []
    for real in range(n_real):
        for j in range(12):
            for k in (10, 30):
                if d04c_like:
                    S_ = (1e-4, 1e-3, 1e-2)[j % 3]
                    t95, pi = 20.0 * S_ / 1e-3 * 1.3, 2e-2 * S_ / 1e-3  # t95 and pi_r both proportional to a = cS (b fixed)
                    R = (1.0, 2.0, 4.0)[(j // 3) % 3] * t95
                    SR = float(np.exp(rng.normal(-1, 1)))
                else:
                    t95, pi = float(np.exp(rng.normal(3, .4))), float(np.exp(rng.normal(-4, 1.2)))
                    R = 0.0 if j % 4 == 0 else float(np.exp(rng.normal(3, .6)))
                    SR = float(np.exp(rng.normal(-1, 1)))
                import p5_timescale as B
                tr = B.transforms(R, t95, k, SR, pi)
                y = -0.9 * tr['log_SR'] - 0.2 * tr['Lrho'] + 0.15 * tr['Lk'] + 0.3 * tr['z0'] + rng.normal(0, .3)
                inf = -1.0 + 0.5 * tr['log_SR'] + rng.normal(0, .8)
                for p in K.PAIRS:
                    rows.append(dict(source_phase='phase3' if d04c_like else 'phase2', case_id=f'syn{real}_{j}', experiment_group='C' if d04c_like else 'D03',
                                     realization=real, sid='x', d04c=d04c_like, pair_id=p, lead_day=k, R_days=R, t95_d=t95, SR=SR, pi_r=pi, **tr,
                                     mag_valid=True, mag_reason='ok', y_mag=y, sign_valid=True, sign_reason='ok', y_sign=float(inf > 0)))
    return rows


@test
def t_timescale_algebra(scratch):
    import p5_timescale as B
    import p4_statistics as S
    rng = np.random.default_rng(7)
    tr0 = B.transforms(0.0, 15.0, 10, .3, 1e-3)
    assert tr0['z0'] == 1 and tr0['Lrho'] == 0 and tr0['LRk'] == 0 and abs(tr0['Lk'] - math.log(10 / 15)) < 1e-15
    trp = B.transforms(30.0, 15.0, 10, .3, 1e-3)
    assert abs(trp['LRk'] - (trp['Lrho'] - trp['Lk'])) < 1e-12
    full = _synthetic_b(rng)
    d4 = _synthetic_b(rng, d04c_like=True)
    out = {}
    # D04C-like fixed lead: bundle (ii) aliased, all three bundles give the same R2 and likelihood
    for k in (10, 30):
        sel = [r for r in d4 if r['lead_day'] == k and r['pair_id'] == K.PAIRS[0]]
        fm, fs = B.fit_bundles(sel, 'magnitude'), B.fit_bundles(sel, 'sign')
        assert fm['ii']['rank'] == fm['ii']['p'] - 1 and fm['ii']['status'] == 'coefficient_nonunique'
        assert abs(fm['ii']['r2'] - fm['i']['r2']) < 1e-10 and abs(fm['iii']['r2'] - fm['i']['r2']) < 1e-10
        if fs['i']['status'] == 'ok':
            assert abs(fs['ii']['log_likelihood'] - fs['i']['log_likelihood']) < 1e-6 and abs(fs['iii']['log_likelihood'] - fs['i']['log_likelihood']) < 1e-6
        Xn = np.array(fm['ii']['nullspace']['basis'])
        names, X, note = B.design(sel, 'ii')
        assert np.max(np.abs(X @ Xn.T)) < 1e-8 * np.max(np.abs(X)) and note
        out[f'd04c_like_lead{k}'] = dict(ranks={b: fm[b]['rank'] for b in B.BUNDLES}, r2=fm['i']['r2'], sign_status=fs['i']['status'])
    pooled = [r for r in d4 if r['pair_id'] == K.PAIRS[0]]
    fp = B.fit_bundles(pooled, 'magnitude')
    assert fp['ii']['rank'] == fp['ii']['p']  # pooled leads add the lead contrast
    # bootstrap multiplicity weights == replicated rows
    sel = [r for r in full if r['pair_id'] == K.PAIRS[0]]
    draw = {i: int(v) for i, v in enumerate(rng.multinomial(10, [.1] * 10))}
    rep = [r for r in sel for _ in range(draw[r['realization']])]
    fw_, fr_ = B.fit_bundles(sel, 'magnitude', draw), B.fit_bundles(rep, 'magnitude')
    lw, lr = B.fit_bundles(sel, 'sign', draw), B.fit_bundles(rep, 'sign')
    for b in B.BUNDLES:
        assert abs(fw_[b]['r2'] - fr_[b]['r2']) < 1e-10 and abs(fw_[b]['sse'] - fr_[b]['sse']) < 1e-8 * max(1, fr_[b]['sse'])
        if lr[b]['status'] == 'ok':
            assert abs(lw[b]['log_likelihood'] - lr[b]['log_likelihood']) < 1e-6 * max(1, abs(lr[b]['log_likelihood']))
    # estimable-space logistic == frozen full-rank fit when the design is full rank
    names, X, _ = B.design(sel, 'ii')
    y = np.array([r['y_sign'] for r in sel])
    a, f = B.logit_estimable(y, X, None, names), S.fit_logistic(y, X, column_names=names)
    assert a['status'] == f['status'] == 'ok' and abs(a['log_likelihood'] - f['log_likelihood']) < 1e-7
    assert max(abs(a['coefficients'][n] - f['coefficients'][n]) for n in names) < 1e-5
    # OLS SVD == normal equations when full rank
    yy = np.array([r['y_mag'] for r in sel])
    o, fl = B.ols_svd(yy, X, None, names), S.fit_log_linear(yy, X, column_names=names)
    assert abs(o['r2'] - fl['r2']) < 1e-10 and max(abs(o['coefficients'][n] - fl['coefficients'][n]) for n in names) < 1e-8
    # R=0 rows retained with their own indicator; positive-only subset drops z0
    assert sum(r['z0'] for r in sel) > 0 and 'z0' in B.design(sel, 'i')[0]
    assert 'z0' not in B.design([r for r in sel if r['R_days'] > 0], 'i')[0]
    cmp = B.compare(full + d4, [draw, {i: 1 for i in range(10)}], None)
    assert len(cmp) == 2 * 3 * 3 * 2 and all('delta_interval' in c for c in cmp)
    out['n_comparisons'] = len(cmp)
    out['p2_identity'] = B.p2_identity(full)
    return out


@test
def t_shared_draws_and_input_ranks(scratch):
    import p5_timescale as B
    dr = B.shared_draws()
    rows = B.input_rows_without_outcomes()
    rr = B.rank_report(rows)
    assert len(rows) == 7800 and sum(r['z0'] for r in rows) == 180 * 4
    assert rr['D04C|10']['ii']['rank'] == 4 and rr['D04C|30']['ii']['rank'] == 4 and rr['D04C|10']['Lk_residual_on_1_logpi_rel'] < 1e-12
    assert rr['full|pooled']['i']['rank'] == 5 and rr['full|pooled']['ii']['rank'] == 6 and rr['D04C|pooled']['ii']['rank'] == 5
    assert all(sum(d.values()) == 10 for d in dr['draws'])
    return dict(draws={k: v for k, v in dr.items() if k != 'draws'}, d04c={k: v for k, v in rr.items() if k.startswith('D04C')})


# ------------------------------------------------------------------ collectors
@test
def t_collect(scratch):
    import p5_collect as C
    maps = C.d05_map()
    hull = C.Hull(C.d05_support_points())
    stats = json.loads((K.P4 / 'statistics.json').read_text())
    crossings = {}
    for lead in (10, 30):
        for p in K.PAIRS:
            cr = stats['cohort_A']['models']['sign'][str(lead)][p]['no_layer']['crossings']['0.0']['0.5']
            pr = C.predict(maps, lead, p, cr['signal_ratio'], 0.0, hull)
            crossings[f'{lead}|{p}'] = pr['p_sign']
            assert abs(pr['p_sign'] - 0.5) < 1e-9
    assert C.transition(1, 0, False) == 'became_determined' and C.transition(0, 1, False) == 'lost_determination'
    assert C.transition(None, 1, False) == 'missing' and C.transition(1, 1, True) == 'no_active_contrast'
    assert C.contradiction(1, 1, -0.2) == (True, False) and C.contradiction(1, -1, 0.0) == (False, True) and C.contradiction(0, 0, 1.0) == (None, None)
    assert C.predict(maps, 10, K.PAIRS[0], 0.0, 0.0)['status'] == 'undefined_coordinates'
    # synthetic A rows through a_row/summarize_a
    man = dict(no_active_contrast=False, twin_case_id='d05B_x', distance_m=20.0, form='paddy_irrigation', origin_month=6, SR_origin=0.5, rho_last_off=0.2,
               t95_d=9.0, gain_m_per_m3d=.002)
    manz = dict(man, no_active_contrast=True, SR_origin=0.0)
    base = dict(pair_id=K.PAIRS[0], lead_day='10', realization='0', sid='confined_T500', storage_type='confined', W_m='0.1', E_true_m='-0.3',
                env_inf_m='-0.4', env_sup_m='-0.3', E_tool_m='0.05', E_reference_m='-0.31')
    par = dict(env_inf_m='-0.3', env_sup_m='0.1', W_m='0.4', E_true_m='-0.2', E_tool_m='-0.1', SR='0.25', rho_last_off='0.1')
    rows = [C.a_row(dict(base, realization=str(i)), par, man, maps, hull) for i in range(10)] + [C.a_row(dict(base, realization='3'), par, manz, maps, hull)]
    s = C.summarize_a(rows, [{i: 1 for i in range(10)}, {i: (2 if i < 5 else 0) for i in range(10)}])
    al = s[f'{K.PAIRS[0]}|10|all']
    assert al['n_active'] == 10 and al['n_no_question'] == 1 and al['transitions']['became_determined'] == 10 and al['tool_opposite_sign_r20'] == 10
    assert al['rate_r20_active'] == 1.0 and abs(al['rate_r20_all_scheduled_companion'] - 10 / 11) < 1e-12 and rows[-1]['p_sign_map'] is None
    cov = C.b_raw_rows(strict=True, mode='coverage_only')
    assert len(cov) == 7800 and set(cov[0]) == {'source_phase', 'case_id', 'pair_id', 'lead_day'}
    prev = json.loads(C.write_c_cases(scratch / 'C_CASES_preview.json', mode='preview').read_text())
    assert prev['n'] == 6270 and prev['counts'] == {'phase2': 540, 'phase3': 690, 'phase4': 2880, 'phase5': 2160}
    assert set(K.C_FIELDS) <= set(prev['records'][0]) and all(r['family_bounds']['log10_b'][0] == (-6.0 if r['source_phase'] == 'phase5' else -4.0)
                                                              for r in prev['records'])
    assert raises(PermissionError, C.a_rows, True, K.P5, 'production')
    return dict(fixed_map_p_at_stored_crossings=crossings, hull_points=hull.n_points, summary_all=al)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', type=Path, default=None)
    ap.add_argument('--skip-slow', action='store_true')
    a = ap.parse_args()
    scratch = a.scratch or Path(tempfile.mkdtemp(prefix='p5AB_'))
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fast = [t_freeze_gate, t_interfaces_gate, t_identities_shards, t_family_isolation, t_geometry_cache, t_generation_export, t_inputs_only_runner,
            t_timescale_algebra, t_shared_draws_and_input_ranks, t_collect]
    slow = [t_w_identity_regression, t_reference_identity_regression, t_run_wb_fixture]
    for t in fast + ([] if a.skip_slow else slow):
        t(scratch)
    rep = dict(n=len(RESULTS), n_pass=sum(v['ok'] for v in RESULTS.values()), all_ok=all(v['ok'] for v in RESULTS.values()), runtime_s=time.time() - t0,
               scratch=str(scratch), python=sys.executable, threads={v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
               note='synthetic/source-linked correctness only: reduced-budget fixture W/reference in scratch, no official case/W/model/scoring', results=RESULTS)
    K.EVIDENCE.mkdir(parents=True, exist_ok=True)
    (K.EVIDENCE / 'TEST_RESULTS.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'results'}))
    if not rep['all_ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
