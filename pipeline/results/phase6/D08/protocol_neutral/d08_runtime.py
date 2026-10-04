"""D08 neutral execution entrypoints. Nothing executes on import.

Actual scientific stages require both frozen protocol pins and --execute.
The protocol worker only runs preflight/selftest, never scientific stages.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[4])))
HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution'
PAIRS = ('P1_continue_vs_stop', 'P2_current_vs_1p5x')
VARIANTS = {'range': 1.5, 'control': 1.0, 'stop_control': 0.0}
_GEN_HASHES = None
sys.path.insert(0, str(ROOT / 'lib'))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path)
    if not p.resolve().is_relative_to(HERE):
        raise PermissionError('All execution writes must stay under protocol_neutral')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def csvsave(path, rows):
    p = Path(path)
    assert p.resolve().is_relative_to(HERE)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def npz(path, **arrays):
    p = Path(path)
    assert p.resolve().is_relative_to(HERE)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def gate():
    frozen = read(HERE / 'FREEZE.json')
    digest = sha(ROOT / 'results/phase6/protocol.md')
    assert digest == frozen['protocol_sha256']
    assert (HERE / 'protocol.sha256').read_text().split()[0] == digest
    for p, expected in frozen['artifact_pins'].items():
        assert sha(p) == expected, p
    for p, expected in read(HERE / 'SOURCE_BASELINE.json').items():
        assert sha(p) == expected, p
    return digest


def index():
    return read(HERE / 'SOURCE_INDEX.json')


def source_csv():
    out = {}
    for path in (ROOT / 'results/phase4/primary_rows.csv', ROOT / 'results/phase5/A_rows.csv'):
        with path.open() as f:
            for line, row in enumerate(csv.DictReader(f), 2):
                k = row['source_phase'], row['case_id'], row['pair_id'], int(row['lead_day'])
                assert k not in out
                out[k] = dict(row, source_csv=str(path), source_csv_line=line)
    assert len(out) == 25080
    return out


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def record_sign(row):
    if str(row.get('no_active_contrast', '')).lower() == 'true':
        return None, 'no_question'
    lo, hi = number(row['env_inf_m']), number(row['env_sup_m'])
    if lo is None or hi is None:
        return None, 'envelope_missing'
    if lo > hi:
        raise ValueError('Invalid original envelope')
    return (1, 'determined') if lo > 0 else (-1, 'determined') if hi < 0 else (0, 'undetermined')


def reversal(effect, sign, state):
    if state != 'determined':
        return None, state
    if effect is None:
        return None, 'engine_missing'
    if effect == 0:
        return False, 'engine_zero'
    opposite = effect * sign < 0
    return opposite, 'opposite' if opposite else 'correct'


def signed_ratio(engine, truth):
    if truth is None:
        return None, 'truth_missing'
    if truth == 0:
        return None, 'truth_zero_no_question'
    if engine is None:
        return None, 'engine_missing'
    return engine / truth, 'ok'


def signal_bin(sr, edges, phase):
    if sr is None:
        return 'missing_signal'
    if sr <= 0:
        return 'nonpositive_signal'
    if phase == 'phase5' and sr < edges[0]:
        return 'below_D05_range'
    if phase == 'phase5' and sr > edges[-1]:
        return 'above_D05_range'
    return str(int(np.digitize([sr], edges[1:-1], right=True)[0]))


def range_metrics(q, qo):
    out = {}
    for label, v in [('full1024', q), ('recent365', q[-365:])]:
        assert len(v) == (1024 if label == 'full1024' else 365)
        assert np.isfinite(v).all() and np.all(v >= 0)
        pos = v[v > 0]
        out[label] = dict(max_positive=float(pos.max()) if pos.size else None,
                          min_positive=float(pos.min()) if pos.size else None,
                          min_rate=float(v.min()), zero_seen=bool(v.min() == 0),
                          increase_range_ratio=float(1.5 * qo / pos.max()) if qo > 0 and pos.size else None,
                          increase_status='ok' if qo > 0 and pos.size else 'no_question' if qo == 0 else 'no_positive_context')
    return out


def cluster_interval(values, realizations, draws, kind='median'):
    """Repeat every row of realization r D[b,r] times; median includes even-n midpoint."""
    v = np.asarray(values, float)
    r = np.asarray(realizations, int)
    assert v.shape == r.shape and np.isfinite(v).all()
    if not len(v):
        return dict(point=None, lower=None, upper=None, n_identified=0, n_unidentified=len(draws), status='unidentifiable')
    stat = np.median if kind == 'median' else np.mean
    samples = []
    for d in draws:
        z = np.repeat(v, np.asarray(d, int)[r])
        if len(z):
            samples.append(float(stat(z)))
    bounds = np.percentile(samples, [2.5, 97.5], method='linear') if samples else [None, None]
    return dict(point=float(stat(v)), lower=number(bounds[0]), upper=number(bounds[1]),
                n_identified=len(samples), n_unidentified=len(draws) - len(samples),
                status='conditional_on_identified_replicates' if samples else 'unidentifiable')


def author_rule(treatment, control, realizations, draws):
    t, c = np.asarray(treatment, float), np.asarray(control, float)
    assert len(t) == len(c) == len(realizations) and np.isfinite(t).all() and np.isfinite(c).all()
    interval = cluster_interval(t, realizations, draws)
    ctrl = cluster_interval(c, realizations, draws)
    a = interval['lower'] is not None and interval['lower'] > 0
    b = interval['point'] >= 2 * ctrl['point']
    return dict(a_interval_lower=interval['lower'], a_interval_upper=interval['upper'],
                treatment_median_change=interval['point'], control_median_change=ctrl['point'],
                doubled_control_median_change=2 * ctrl['point'], condition_a=bool(a), condition_b=bool(b),
                mechanical_decision='지지' if a and b else '지지되지 않음',
                treatment_bootstrap=interval, control_bootstrap=ctrl)


def draws():
    D = np.load(ROOT / 'results/phase4/bootstrap_draw_matrix.npy', allow_pickle=False)
    assert D.shape == (999, 10) and np.all(D >= 0) and np.all(D.sum(1) == 10)
    assert hashlib.sha256(np.asarray(D, '<i8').tobytes()).hexdigest() == '7520bcd1cd416b25f22e1e3c7ce6f3905209b837eac4d5a703a038b79f7aa30e'
    return D


def preflight():
    digest = gate()
    ix = index()
    assert len(ix['records']) == 6270 and len(ix['active20_ids']) == 1242
    assert len(set(ix['active20_ids'])) == 1242
    assert len(ix['W50_ids']) == len(set(ix['W50_ids'])) == 50
    assert set(ix['W50_ids']) <= set(ix['active20_ids'])
    for m in ix['records']:
        assert m['realization'] in range(10)
        for H in [10, 30]:
            n = m['native'][str(H)]
            assert n['a_index'] // 8 == (n['a_index'] + 1) // 8
            assert n['query_ids'][0].endswith('__a__H' + str(H))
    D = draws()
    report = dict(status='ready_no_scoring', protocol_sha256=digest, utc=now(),
                  A_records=6270, A_rows=25080, B_active_origins=1242, W_original_records=50,
                  bootstrap_shape=list(D.shape), scientific_stages_executed=[], model_loaded=False,
                  native_mapping_complete=True, all_source_hashes_checked=True)
    save(HERE / 'PREFLIGHT.json', report)
    print(json.dumps(report))


def run_a():
    digest = gate()
    rows, ix = source_csv(), index()
    result = []
    from p3_make_cases import load_tf_input
    for m in ix['records']:
        ti = load_tf_input(m['tf_path'])
        q = ti['pumping_context']
        metrics = range_metrics(q, m['q_origin_m3d'])
        for pair in PAIRS:
            for H in [10, 30]:
                original = rows[m['source_phase'], m['case_id'], pair, H]
                engine, truth, ref = [number(original[k]) for k in ['E_tool_m', 'E_true_m', 'E_reference_m']]
                sg, state = record_sign(original)
                rev, rev_state = reversal(engine, sg, state)
                ratio, ratio_state = signed_ratio(engine, truth)
                reference, ref_state = signed_ratio(ref, truth)
                out = dict(source_phase=m['source_phase'], case_id=m['case_id'], pair=pair, lead=H,
                           realization=m['realization'], record_set=m['record_set'], SR=number(original['SR']),
                           experiment_group=m['experiment_group'], sid=m['sid'], form=m['form'], origin_month=m['month'],
                           q_origin_m3d=m['q_origin_m3d'], Q_nominal_m3_d=m['Q_nominal_m3_d'],
                           signal_bin=signal_bin(number(original['SR']), ix['signal_edges'], m['source_phase']),
                           E_engine_m=engine, E_true_m=truth, E_reference_m=ref,
                           signed_engine_ratio=ratio, ratio_status=ratio_state, signed_reference_ratio=reference,
                           reference_ratio_status=ref_state, original_record_sign=sg, original_record_status=state,
                           original_reversal=rev, original_reversal_status=rev_state,
                           W_status=original['W_status'], tool_status=original['tool_status'], reference_status=original['reference_status'],
                           source_csv=original['source_csv'], source_csv_line=original['source_csv_line'],
                           reference_source_path=original['reference_source_path'], reference_sha256=original['reference_sha256'],
                           tf_path=m['tf_path'], tf_sha256=m['tf_sha256'], truth_path=m['truth_path'], truth_sha256=m['truth_sha256'],
                           native_archive=m['native'][str(H)]['archive_path'], native_archive_sha256=m['native'][str(H)]['archive_sha256'],
                           protocol_sha256=digest)
                for window, metric in metrics.items():
                    out.update({window + '_' + k: v for k, v in metric.items()})
                result.append(out)
    assert len(result) == 25080
    csvsave(OUT / 'A/source_frame.csv', result)
    groups = collections.defaultdict(list)
    for r in result:
        for window in ['full1024', 'recent365']:
            val = r[window + '_increase_range_ratio']
            category = ('no_question_or_missing' if val is None else 'within_range' if val <= 1 else 'outside_range') if r['pair'] == PAIRS[1] else ('zero_seen' if r[window + '_zero_seen'] else 'zero_absent')
            for bin_ in ['all_signal_bins', r['signal_bin']]:
                groups[r['record_set'], r['pair'], r['lead'], window, bin_, category].append(r)
    summaries = []
    for key, rs in sorted(groups.items()):
        summary = dict(zip(['record_set', 'pair', 'lead', 'window', 'signal_bin', 'range_category'], key))
        summary.update(rows=len(rs), ratio_status_counts=dict(collections.Counter(r['ratio_status'] for r in rs)),
                       reversal_status_counts=dict(collections.Counter(r['original_reversal_status'] for r in rs)))
        for metric in ['signed_engine_ratio', 'signed_reference_ratio', 'original_reversal']:
            valid = [r for r in rs if r[metric] is not None]
            summary[metric] = cluster_interval([r[metric] for r in valid], [r['realization'] for r in valid], draws(), 'mean' if metric == 'original_reversal' else 'median')
        summaries.append(summary)
    save(OUT / 'A/summary.json', dict(protocol_sha256=digest, groups=summaries, all_rows_retained=True))


def rebuilt(m, variant):
    """Same frozen physics; original rain, natural head and eps copied, never redrawn."""
    from p3_make_cases import load_tf_input
    import p4_cases as c4
    import p5_distance_cases as c5
    ti = load_tf_input(m['tf_path'])
    origin = datetime.date.fromisoformat(ti['meta']['origin_date'])
    assert str(ti['dates'][624]) == str(origin - datetime.timedelta(days=400))
    assert str(ti['dates'][630]) == str(origin - datetime.timedelta(days=394))
    assert str(ti['dates'][1023]) == str(origin - datetime.timedelta(days=1))
    with np.load(m['truth_path'], allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    theta = json.loads(str(tr['theta_true_json']))
    L = c5.geometry(c5.stratum(m['sid'], 20.0))
    bg = c5._bg(tr['h_nat'], tr['eps'], theta['sigma_bg'])
    q = ti['pumping_context'].copy()
    qo, qp = m['q_origin_m3d'], m['Q_pre_hidden_m3d']
    original = c4._physics(bg, L, q, qo, qp)
    assert np.array_equal((original[4] + original[2])[:1024], ti['head_context'])
    for p in PAIRS:
        assert np.array_equal(original[5][p], tr['E_true__' + p])
    q[624:631] = VARIANTS[variant] * m['Q_nominal_m3_d']
    fut, hn, eps, hp, hs, E, delta = c4._physics(bg, L, q, qo, qp)
    assert np.array_equal(hn, tr['h_nat']) and np.array_equal(eps, tr['eps'])
    checks = {}
    for p in PAIRS:
        for arm, expected in zip(['a', 'b'], fut[p]):
            assert np.array_equal(expected, ti['future_Q'][p + '_' + arm])
        residual = float(np.max(np.abs(E[p] - tr['E_true__' + p])))
        analytical = -L['d']['gain'] * np.convolve(fut[p][0] - fut[p][1], L['B'])[:30]
        closure = float(np.max(np.abs(E[p] - analytical)))
        assert residual <= 1e-12 and closure <= 1e-12, (m['case_id'], p, residual, closure)
        checks[p] = dict(context_truth_change_maxabs_m=residual, future_only_closure_maxabs_m=closure)
    changed = dict(ti, head_context=(hs + eps)[:1024], pumping_context=q)
    return changed, dict(case_id=m['case_id'], variant=variant, checks=checks, rain_natural_noise_future_unchanged=True)


def generate():
    digest = gate()
    records = {m['case_id']: m for m in index()['records'] if m['source_phase'] == 'phase5'}
    checks = []
    for variant in VARIANTS:
        for cid in index()['active20_ids']:
            m = records[cid]
            ti, ck = rebuilt(m, variant)
            ck['protocol_sha256'] = digest
            _save_generated(ti, variant, digest)
            ck['tf_input_path'] = str(OUT / 'generated' / variant / 'tf_inputs' / (cid + '.npz'))
            ck['tf_input_sha256'] = sha(ck['tf_input_path'])
            checks.append(ck)
    assert len(checks) == 3726
    save(OUT / 'generated/GENERATION_RECEIPT.json', dict(protocol_sha256=digest, records=checks, n=3726, max_true_effect_change_m=max(z['context_truth_change_maxabs_m'] for c in checks for z in c['checks'].values()), closure_tolerance_m=1e-12))


def _save_generated(ti, variant, digest):
    from p3_make_cases import save_tf
    target = OUT / 'generated' / variant / 'tf_inputs'
    target.mkdir(parents=True, exist_ok=True)
    save_tf(target / (ti['case_id'] + '.npz'), dict(case_id=ti['case_id'], tf_input=ti), digest)


def generated_ti(cid, variant, digest):
    global _GEN_HASHES
    from p3_make_cases import load_tf_input
    if _GEN_HASHES is None:
        receipt = read(OUT / 'generated/GENERATION_RECEIPT.json')
        assert receipt['protocol_sha256'] == digest and receipt['n'] == 3726
        _GEN_HASHES = {(r['case_id'], r['variant']): r['tf_input_sha256'] for r in receipt['records']}
    path = OUT / 'generated' / variant / 'tf_inputs' / (cid + '.npz')
    assert sha(path) == _GEN_HASHES[cid, variant], str(path)
    ti = load_tf_input(path)
    assert ti['protocol_sha256'] == digest
    return ti


def infer():
    digest = gate()
    receipt = read(OUT / 'generated/GENERATION_RECEIPT.json')
    assert receipt['protocol_sha256'] == digest and receipt['n'] == 3726
    settings = read(ROOT / 'results/phase6/engine/replay/SETTINGS.json')
    # Public edition: the checkpoint is located through TIMESFM_SNAPSHOT and identified by its frozen weight/config hashes.
    snapshot = Path(os.environ['TIMESFM_SNAPSHOT'])
    assert sha(snapshot / 'model.safetensors') == settings['checkpoint']['weights_sha256'] and sha(snapshot / 'config.json') == settings['checkpoint']['config_sha256']
    import torch
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    from p3_phase2_timesfm_execution import checked_prediction
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    assert torch.__version__ == settings['runtime']['torch'] and np.__version__ == settings['runtime']['numpy']
    locks = []
    for path in [ROOT / 'results/phase4/tools/MODEL_WAVE.lock', ROOT / 'results/phase5/tools/MODEL_WAVE.lock']:
        if path.exists():
            lk = path.open('r'); fcntl.flock(lk, fcntl.LOCK_EX); locks.append(lk)
    path = OUT / 'B/MODEL_WAVE.lock'; path.parent.mkdir(parents=True, exist_ok=True)
    lk = path.open('a'); fcntl.flock(lk, fcntl.LOCK_EX); locks.append(lk)
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(snapshot), local_files_only=True, force_download=False, device='mps', per_core_batch_size=8))
    ix = index()
    model_manifest = dict(protocol_sha256=digest, settings=settings, started_utc=now(), executor_sha256=sha(__file__),
                          generated_receipt_sha256=sha(OUT / 'generated/GENERATION_RECEIPT.json'), native_batches=True, completed=[])
    for H in [10, 30]:
        with np.load(ROOT / f'results/phase5/cases/tool_inputs_H{H}.npz', allow_pickle=False) as z:
            raw = {k: z[k] for k in ['query_id', 'head', 'pumping', 'rainfall', 'metadata_json']}
        lookup = {str(q): i for i, q in enumerate(raw['query_id'])}
        for variant in VARIANTS:
            for batch in ix['B_native_batches'][str(H)]:
                ids = batch['query_ids']
                indices = [lookup[q] for q in ids]
                head, pumping = raw['head'][indices].copy(), raw['pumping'][indices].copy()
                for i, qid in enumerate(ids):
                    cid = qid.split('__')[0]
                    if cid in ix['active20_ids']:
                        ti = generated_ti(cid, variant, digest)
                        head[i] = ti['head_context']; pumping[i, :1024] = ti['pumping_context']
                cov = np.stack([pumping, raw['rainfall'][indices]], axis=1).astype(np.float32)
                contexts = head.astype(np.float32)
                preds = list(model.predict_batch(contexts=list(contexts), horizon=H, past_future_covariates=list(cov), ts_ids=ids, **settings['API']))
                assert len(preds) == len(ids)
                quantiles = np.stack([checked_prediction(p.quantiles, p.ts_id, ids[i], H) for i, p in enumerate(preds)])
                target = OUT / 'B' / variant / f'H{H}' / (batch['batch_id'] + '.npz')
                npz(target, quantiles=quantiles, query_id=np.array(ids), contexts=contexts, covariates=cov, metadata_json=raw['metadata_json'][indices], horizon=H, protocol_sha256=digest, native_archive=batch['archive_path'], native_start=batch['start'])
                model_manifest['completed'].append(dict(path=str(target), sha256=sha(target), queries=len(ids), active_queries=batch['active_query_count']))
                save(OUT / 'B/EXECUTION_MANIFEST.json', model_manifest)
    model_manifest.update(completed_utc=now(), active_queries_expected=29808, all_done=True)
    save(OUT / 'B/EXECUTION_MANIFEST.json', model_manifest)


def run_w(shard, nshards):
    digest = gate()
    assert 0 <= shard < nshards <= 8
    from p5_w_family import engine, family_receipt
    from p4_run_wb import jsonable, BUDGET, W_SEED
    ix = index()
    rows = source_csv()
    family = family_receipt(-6.0)
    for variant in VARIANTS:
        for cid in ix['W50_ids'][shard::nshards]:
            ti = generated_ti(cid, variant, digest)
            target = OUT / 'W' / variant / (cid + '.json')
            try:
                eng = engine(ti, seed=W_SEED, b_log10_min=-6.0)
                rec = eng.run(**BUDGET)
                report = dict(case_id=cid, variant=variant, protocol_sha256=digest, tf_input_sha256=sha(OUT / 'generated' / variant / 'tf_inputs' / (cid + '.npz')), family_box=family, budget=BUDGET, W_seed=W_SEED, result=jsonable(rec), comparisons=[])
                for p in PAIRS:
                    for H in [10, 30]:
                        old = rows['phase5', cid, p, H]
                        env = (rec.get('envelope') or {}).get(p)
                        lo = number(env['inf'][H-1]) if env else None
                        hi = number(env['sup'][H-1]) if env else None
                        sg, status = record_sign(dict(env_inf_m=lo, env_sup_m=hi, no_active_contrast=False))
                        osg, ost = record_sign(old)
                        report['comparisons'].append(dict(pair=p, lead=H, original_W_m=number(old['W_m']), W_m=number(env['W'][H-1]) if env else None,
                            original_sign=osg, original_determination=ost, variant_sign=sg, variant_determination=status,
                            determination_changed=status != ost, original_W_status=old['W_status'], variant_W_status=rec.get('W_status'), flags=rec.get('flags', [])))
            except Exception:
                report = dict(case_id=cid, variant=variant, protocol_sha256=digest, W_status='error', error=traceback.format_exc())
            save(target, report)


def aggregate_b():
    digest = gate()
    ix, originals = index(), source_csv()
    manifest = read(OUT / 'B/EXECUTION_MANIFEST.json')
    assert manifest.get('all_done') and manifest['protocol_sha256'] == digest
    assert manifest['generated_receipt_sha256'] == sha(OUT / 'generated/GENERATION_RECEIPT.json')
    for output in manifest['completed']:
        assert sha(output['path']) == output['sha256'], output['path']
    effects = {}
    for variant in VARIANTS:
        for H in [10, 30]:
            for batch in ix['B_native_batches'][str(H)]:
                with np.load(OUT / 'B' / variant / f'H{H}' / (batch['batch_id'] + '.npz'), allow_pickle=False) as z:
                    assert str(z['protocol_sha256']) == digest
                    ids, q = z['query_id'], z['quantiles']
                    assert ids.tolist() == batch['query_ids']
                    for i in range(0, len(ids), 2):
                        cid, pair, arm, lead = str(ids[i]).split('__')
                        assert arm == 'a' and str(ids[i+1]) == str(ids[i]).replace('__a__', '__b__')
                        if cid in ix['active20_ids']:
                            effects[cid, pair, H, variant] = float(q[i, -1, 4] - q[i+1, -1, 4])
    assert len(effects) == 14904
    ms = {m['case_id']: m for m in ix['records'] if m['source_phase'] == 'phase5'}
    rows, decisions = [], []
    for pair in PAIRS:
        for H in [10, 30]:
            delta = {v: [] for v in VARIANTS}; reals = []
            for cid in ix['active20_ids']:
                old = originals['phase5', cid, pair, H]
                baseline, truth = number(old['E_tool_m']), number(old['E_true_m'])
                oldratio, status = signed_ratio(baseline, truth)
                # Incomplete original data are retained in A; B cannot silently shrink the author cohort.
                if status != 'ok':
                    raise ValueError('Full1242 author criterion unavailable; explicit repair required: ' + cid)
                sg, state = record_sign(old)
                reals.append(ms[cid]['realization'])
                for variant in VARIANTS:
                    e = effects[cid, pair, H, variant]
                    ratio, rstate = signed_ratio(e, truth)
                    assert rstate == 'ok'
                    change = ratio - oldratio
                    delta[variant].append(change)
                    rev, revstate = reversal(e, sg, state)
                    rows.append(dict(case_id=cid, pair=pair, lead=H, realization=ms[cid]['realization'], variant=variant,
                        original_E_engine_m=baseline, E_engine_m=e, E_true_m=truth, original_signed_ratio=oldratio,
                        signed_ratio=ratio, paired_ratio_change=change, original_record_status=state, original_record_sign=sg,
                        reversal=rev, reversal_status=revstate, denominator_is_frozen_D06_approximation=True, protocol_sha256=digest))
            rule = author_rule(delta['range'], delta['control'], reals, draws())
            rule.update(pair=pair, lead=H, n_original_records=1242, primary=pair == PAIRS[1] and H == 10, report_only=pair != PAIRS[1] or H != 10,
                stop_control_change=cluster_interval(delta['stop_control'], reals, draws()), negative_control_rule_applied_without_reinterpretation=True)
            decisions.append(rule)
    csvsave(OUT / 'B/source_frame.csv', rows)
    summaries = []
    for pair in PAIRS:
        for H in [10, 30]:
            for variant in VARIANTS:
                rs = [r for r in rows if r['pair'] == pair and r['lead'] == H and r['variant'] == variant]
                valid = [r for r in rs if r['reversal'] is not None]
                summaries.append(dict(pair=pair, lead=H, variant=variant, n=1242,
                    ratio=cluster_interval([r['signed_ratio'] for r in rs], [r['realization'] for r in rs], draws()),
                    reversal_numerator=sum(r['reversal'] for r in valid), reversal_denominator=len(valid),
                    reversal_rate=cluster_interval([r['reversal'] for r in valid], [r['realization'] for r in valid], draws(), 'mean'),
                    reversal_status_counts=dict(collections.Counter(r['reversal_status'] for r in rs)), denominator_approximation='frozen original D06 sign determination'))
    save(OUT / 'B/AUTHOR_CRITERION.json', dict(protocol_sha256=digest, decisions=decisions, summaries=summaries))


def aggregate_w():
    digest = gate()
    comparisons, errors = [], []
    for v in VARIANTS:
        for cid in index()['W50_ids']:
            r = read(OUT / 'W' / v / (cid + '.json'))
            assert r['protocol_sha256'] == digest
            if 'error' in r:
                errors.append(r)
            else:
                comparisons.extend(dict(case_id=cid, variant=v, **c) for c in r['comparisons'])
    csvsave(OUT / 'W/comparison_source_frame.csv', comparisons)
    save(OUT / 'W/COMPARISON_RECEIPT.json', dict(protocol_sha256=digest, selected_original_records=50, expected_variant_runs=150, successful_variant_runs=150-len(errors), errors=errors, rows=len(comparisons)))


def selftest():
    gate()
    D = np.array([[1, 1] + [0] * 8, [2, 0] + [0] * 8])
    assert cluster_interval([1, 3], [0, 1], D)['point'] == 2
    assert cluster_interval([1, 3], [0, 1], D)['lower'] == 1.025
    allten = np.ones((3, 10), dtype=int)
    r = author_rule([1]*10, [-1]*10, list(range(10)), allten)
    assert r['condition_a'] and r['condition_b'] and r['doubled_control_median_change'] == -2
    r = author_rule([0]*10, [-1]*10, list(range(10)), allten)
    assert not r['condition_a'] and r['condition_b'] and r['mechanical_decision'] == '지지되지 않음'
    assert signed_ratio(0, -1) == (0.0, 'ok')
    assert signed_ratio(1, 0)[0] is None
    assert reversal(0, -1, 'determined') == (False, 'engine_zero')
    q = np.ones(1024); q[624:631] = 1.5
    assert np.count_nonzero(q != 1) == 7 and q[630] == 1.5 and q[631] == 1
    metrics = range_metrics(q, 1)
    assert metrics['full1024']['increase_range_ratio'] == 1
    assert metrics['recent365']['increase_range_ratio'] == 1.5
    save(HERE / 'SELFTEST.json', dict(status='passed', utc=now(), tests=['cluster median/even midpoint/linear percentile', 'author negative control algebra', 'strict lower>0/equality false', 'engine zero and truth zero', 'exact seven indices', 'recent365 independent metrics'], scientific_data_scored=False, model_loaded=False))
    print('SELFTEST_OK: synthetic arrays only; no scientific scoring/model calls')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['preflight', 'selftest', 'A', 'generate', 'B', 'W', 'aggregate_B', 'aggregate_W'])
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    if a.stage not in ['preflight', 'selftest'] and not a.execute:
        ap.error('Scientific stage requires --execute after frozen protocol and independent release')
    if a.stage in ['preflight', 'selftest']:
        return globals()[a.stage]()
    OUT.mkdir(exist_ok=True)
    if a.stage == 'W':
        return run_w(a.shard, a.nshards)
    return {'A': run_a, 'generate': generate, 'B': infer, 'aggregate_B': aggregate_b, 'aggregate_W': aggregate_w}[a.stage]()


if __name__ == '__main__':
    main()
