"""Outcome-blind D08 inventory, backup, protocol freeze; no forecasts/effect scoring."""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from d08_runtime import ROOT, HERE, PAIRS, sha, save, now


def inspect_and_prepare():
    if (HERE / 'FREEZE.json').exists():
        raise PermissionError('Already frozen: preparation cannot regenerate the selection/index')
    records, active, sources = [], [], {}
    def pin(path, want=None):
        path = Path(path)
        h = sha(path)
        if want is not None:
            assert h == want, str(path)
        sources[str(path)] = h
        return h
    packet = json.loads((ROOT / 'results/phase6/D08/NEUTRAL_SOURCE_PACKET.json').read_text())
    for name in ['NEUTRAL_SECTION_2.md', 'NEUTRAL_SOURCE_PACKET.json', 'LATEST_STEERING.txt', 'PRE_REVISION_SHA256.json']:
        pin(ROOT / 'results/phase6/D08' / name)
    for phase in range(2, 6):
        base = ROOT / f'results/phase{phase}'
        mf = base / 'cases/derived_manifest.json'
        pin(mf)
        for m in json.loads(mf.read_text()):
            cid = m['case_id']
            tf, truth = base / 'cases' / m['tf_input'], base / 'cases' / m['truth']
            pin(tf, m['tf_sha256']); pin(truth, m['truth_sha256'])
            qo = float(m.get('q_origin_m3d', m['q0_m3d']))
            with np.load(tf, allow_pickle=False) as z:
                assert str(z['case_id']) == cid
                assert z['head_context'].shape == z['pumping_context'].shape == (1024,)
                assert z['rain'].shape == (1054,)
                assert np.isfinite(z['pumping_context']).all() and np.all(z['pumping_context'] >= 0)
                for pair in PAIRS:
                    assert np.all(z['future_Q__' + pair + '_a'] == qo)
                    assert np.all(z['future_Q__' + pair + '_b'] == (0 if pair == PAIRS[0] else 1.5 * qo))
            grp = m.get('experiment_group', 'D03')
            record_set = 'calendar20m' if phase == 5 else 'calendar200m' if phase == 4 and grp == 'B' else 'map' if phase == 4 else 'design'
            r = dict(source_phase=f'phase{phase}', case_id=cid, experiment_group=grp, record_set=record_set,
                     realization=int(m['realization']), sid=m['sid'], form=m.get('form', m.get('calendar_type')),
                     month=m.get('origin_month'), q_origin_m3d=qo, Q_nominal_m3_d=float(m['Q_nominal_m3_d']),
                     tf_path=str(tf), tf_sha256=m['tf_sha256'], truth_path=str(truth), truth_sha256=m['truth_sha256'],
                     Q_pre_hidden_m3d=float(m.get('Q_pre_hidden_m3d', 100)), native={})
            records.append(r)
            if phase == 5 and qo > 0:
                assert not m['no_active_contrast']
                active.append(cid)
    assert len(records) == 6270 and len(active) == 1242
    bykey = {(r['source_phase'], r['case_id']): r for r in records}
    native_batches = {'10': [], '30': []}
    settingspath = ROOT / 'results/phase6/engine/replay/SETTINGS.json'
    settings = json.loads(settingspath.read_text()); pin(settingspath)
    for phase in range(2, 6):
        base = ROOT / f'results/phase{phase}'
        for H in [10, 30]:
            pin(base / f'cases/tool_inputs_H{H}.npz')
        tools = base / 'tools'
        archives = sorted(tools.glob('timesfm_raw_H*.npz')) if phase in [2, 3] else sorted(tools.glob('*/timesfm_raw_H*.npz'))
        assert len(archives) == (2 if phase in [2, 3] else 4)
        for path in archives:
            pin(path)
            with np.load(path, allow_pickle=False) as z:
                qs, pids = z['query_id'].astype(str), z['pair_id'].astype(str)
                H = int(qs[0].rsplit('__H', 1)[1])
                prov = json.loads(str(z['provenance_json']))
                assert prov['batch'] == 8 and prov['checkpoint_revision'] == settings['checkpoint']['revision']
                assert prov['checkpoint_weights_sha256'] == settings['checkpoint']['weights_sha256']
                assert json.loads(prov['api_json']) == settings['API']
                # Preserve every original query index, pair order and companion batch member.
                for i in range(0, len(qs), 2):
                    cid, pair, arm, horizon = qs[i].split('__')
                    assert qs[i+1] == qs[i].replace('__a__', '__b__') and arm == 'a'
                    m = bykey[f'phase{phase}', cid]
                    native = m['native'].setdefault(str(H), dict(archive_path=str(path), archive_sha256=sources[str(path)], pairs={}))
                    assert native['archive_path'] == str(path)
                    native['pairs'][pair] = dict(a_index=i, b_index=i+1, query_ids=qs[i:i+2].tolist())
                    if pair == PAIRS[0]:
                        native.update(a_index=i, query_ids=qs[i:i+2].tolist())
                if phase == 5:
                    for start in range(0, len(qs), 8):
                        stop = min(start+8, len(qs))
                        qids = qs[start:stop].tolist()
                        nactive = sum(q.split('__')[0] in active for q in qids)
                        if nactive:
                            key = f'{path}:{start}:{stop}'
                            native_batches[str(H)].append(dict(batch_id='native_' + hashlib.sha256(key.encode()).hexdigest()[:16], archive_path=str(path), start=start, stop=stop, query_ids=qids, active_query_count=nactive, companion_zero_queries=len(qids)-nactive))
    for m in records:
        for H in ['10', '30']:
            assert set(m['native'][H]['pairs']) == set(PAIRS)
    assert sum(b['active_query_count'] for b in native_batches['10']) == 4968
    assert sum(b['active_query_count'] for b in native_batches['30']) == 4968
    # Equal base allocation across form/layer cells; rotating, outcome-blind realization balance.
    cells = collections.defaultdict(list)
    for cid in active:
        r = bykey['phase5', cid]
        cells[r['form'], r['sid']].append(r)
    order = sorted(cells, key=lambda k: hashlib.sha256(('D08_W50_20261001|' + '|'.join(k)).encode()).hexdigest())
    selected, rc = [], collections.Counter()
    for round_ in range(3):
        for cell in order:
            if len(selected) == 50:
                break
            candidates = [r for r in cells[cell] if r['case_id'] not in selected]
            r = min(candidates, key=lambda r: (rc[r['realization']], hashlib.sha256(('D08_W50_20261001|' + r['case_id']).encode()).hexdigest()))
            selected.append(r['case_id']); rc[r['realization']] += 1
    assert len(selected) == len(set(selected)) == 50
    selected_records = [bykey['phase5', c] for c in selected]
    save(HERE / 'W50_SELECTION.json', dict(selected_original_record_ids=selected, seed_label='D08_W50_20261001', algorithm='form/layer cells SHA-ordered; 2 per cell then first14 cells third; choose globally least-used realization then SHA256(caseID) tie-break', input_fields=['case_id', 'form', 'sid', 'realization'], outcome_fields_used=[], by_realization=dict(rc), by_form_layer={ '|'.join(k): sum((r['form'], r['sid']) == k for r in selected_records) for k in cells}, frozen_utc=now()))
    for cid in selected:
        pin(ROOT / 'results/phase5/wb/W' / (cid + '.json'))
    stats = ROOT / 'results/phase4/statistics.json'; pin(stats)
    edges = json.loads(stats.read_text())['cohort_C']['bins']['edges']
    for p in [ROOT / 'results/phase4/primary_rows.csv', ROOT / 'results/phase5/A_rows.csv', ROOT / 'results/phase4/bootstrap_draw_matrix.npy']:
        pin(p)
    # Exact transitive local source closure, including lazy imports and exec-loaded W sources.
    pending = ['p5_distance_cases', 'p5_export', 'p5_w_family', 'p4_run_wb', 'p3_phase2_timesfm_execution', 'timesfm_pilot_adapter', 'p3_wenvelope', 'p3_wenvelope_repaired_v1_2', 'p3_pastas']
    seen = set()
    while pending:
        mod = pending.pop()
        path = ROOT / 'lib' / (mod + '.py')
        if mod in seen or not path.exists():
            continue
        seen.add(mod); pin(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split('.')[0])
    for phase in range(2, 6):
        for name in ['protocol.md', 'protocol.sha256', 'protocol_freeze.json']:
            pin(ROOT / f'results/phase{phase}' / name)
    contract = ROOT / 'results/phase5/FROZEN_EXECUTION_CONTRACT.json'; pin(contract)
    envcontract = json.loads(contract.read_text())
    for kind in ['science_python', 'model_python']:
        for pkg in envcontract[kind]['packages'].values():
            for f, want in pkg['files'].items():
                pin(f, want)
    for path, want in settings['model_sources'].items():
        pin(path, want)
    for filename, key in [('model.safetensors', 'weights_sha256'), ('config.json', 'config_sha256')]:
        pin(Path(settings['checkpoint']['path']) / filename, settings['checkpoint'][key])
    # Copy bytes only: no manuscript text is decoded or inspected.
    prepath = ROOT / 'results/phase6/D08/PRE_REVISION_SHA256.json'
    backups = []
    for f, expected in json.loads(prepath.read_text()).items():
        p = Path(f); pin(p, expected)
        if p.parent == ROOT / 'manuscript' and p.name in ['draft.md', 'supplementary.md']:
            target = HERE / 'v2_backup' / (p.stem + '_v2.md')
        elif p.is_relative_to(ROOT / 'manuscript/figures'):
            target = HERE / 'v2_backup/figurehistory' / p.name
        else:
            target = HERE / 'v2_backup/other' / ('project1' if '/1_zero_shot/' in str(p) else 'project2') / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == expected, str(target)
        else:
            shutil.copy2(p, target)
        assert sha(target) == expected
        backups.append(dict(original=str(p), backup=str(target), sha256=expected))
    save(HERE / 'V2_PRESERVATION.json', dict(created_utc=now(), copied_as_binary=True, manuscript_narrative_read=False, baseline_manifest_sha256=sha(prepath), originals_verified=len(backups), backups=backups))
    save(HERE / 'SOURCE_INDEX.json', dict(records=records, active20_ids=sorted(active), W50_ids=selected, signal_edges=edges, B_native_batches=native_batches, index_source_fields_only=True, scientific_scores_computed=False, counts=dict(collections.Counter(m['record_set'] for m in records))))
    save(HERE / 'SOURCE_BASELINE.json', sources)
    save(HERE / 'PREPARATION_RECEIPT.json', dict(utc=now(), all_manifest_tf_truth_hashes_verified=True, counts=dict(collections.Counter(m['record_set'] for m in records)), local_source_closure=sorted(seen), score_computations=0, new_model_calls=0, inspected_sources=[str(p) for p in sources if '/lib/' in p or p.endswith(('NEUTRAL_SECTION_2.md', 'NEUTRAL_SOURCE_PACKET.json', 'LATEST_STEERING.txt'))], not_inspected=['full D08 original file', 'author instruction', 'root SOURCE_PACKET', 'VERBATIM_SECTIONS_1_3', 'D08 sections1/3', 'old executor design', 'manuscript narrative'], W50_frozen=True, native_batch_counts={H:len(bs) for H,bs in native_batches.items()}))
    print('PREPARED: 6270 original records;1242 active;50 IDs;source hashes+binary v2 backup;0 scores/0 calls')


if __name__ == '__main__':
    inspect_and_prepare()
