"""D06 A raw-only offline forecasts for one actor shard (1080 cases, 8640 queries): the D05 runner with phase5 inputs.

Usage (after the complete phase5 freeze and generation):
  env/.venv(0_zero_shot TimesFM env)/bin/python lib/p5_timesfm_execution.py --mode production --actor shard_b|shard_a
Same model, checkpoint revision, weights/config SHA256, API, raw track, batch 8, chunk 128 and independent H10/H30 runs
as D05 (p4_timesfm_execution); only inputs/outputs/locks move to results/phase5. Model-memory safety: exclusive flocks on
results/phase5/tools/MODEL_WAVE.lock AND (if it exists) the legacy results/phase4/tools/MODEL_WAVE.lock are held for the
process lifetime, so no two TimesFM instances (either phase) share MPS; the second actor waits (sequential waves).
Resume: chunk checkpoints are reused only when their run fingerprint (protocol, driver, inputs, actor, horizon, checkpoint)
matches. --mode inputs_only performs every input/provenance check and prepares the shard tracks without importing torch
or loading the model (no query is made); fixture inputs need --cases-dir and never write results/phase5/tools.
"""
import argparse, fcntl, json, hashlib, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import p5_export as prep
import p5_contracts as K
from p3_phase2_timesfm_execution import sha, atomic_npz, checked_prediction, load_chunk, pack_output
from p4_timesfm_execution import verify, progress
R = K.ROOT; P = K.P5; D = R / 'results/phase2'
BATCH = 8; CHUNK = 128
WEIGHTS_SHA = 'a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
CONFIG_SHA = 'ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6'
REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'


def load_shard(actor: str, base: Path = P) -> list[str]:
    """Same checks as p5_run_wb.load_shard, inlined: importing p5_run_wb pulls p4_run_wb -> p3_pastas -> pastas,
    which is absent from the read-only model environment (the W/reference runtime is not needed here)."""
    rec = json.loads((base / f'SHARD_{actor.upper()}.json').read_text())
    ids = rec['case_ids']
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != rec['case_ids_sha256'] or rec['n_cases'] != K.A_PER_ACTOR or len(set(ids)) != K.A_PER_ACTOR:
        raise ValueError('D06 shard manifest changed')
    if any(K.actor_of(c) != actor for c in ids) or set(ids) != set(K.shards()[actor]):
        raise ValueError('D06 shard rule mismatch')
    return ids


def shard_inputs(inputs, ids, H, digest):
    z = prep.subset(inputs, ids)
    if str(z['protocol_sha256']) != digest:
        raise PermissionError('Tool archive protocol stamp differs')
    return prep.prepare_track(z, H, 'raw', ids)


def check_model_closure():
    """Model source/checkpoint pins inherited from D05 (offline, read-only)."""
    source = json.loads((D / 'A_preflight_manifest.json').read_text())
    for path, want in source['source_hashes'].items():
        if '/timesfm3/torch/' in path:  # public edition: engine-source paths are relative to the TimesFM checkout (TIMESFM_SOURCE)
            assert sha(Path(__import__('os').environ.get('TIMESFM_SOURCE','timesfm/src/timesfm3')).parents[1] / path) == want, path
    assert prep.SNAPSHOT.name == REVISION, prep.SNAPSHOT
    assert sha(prep.SNAPSHOT / 'model.safetensors') == WEIGHTS_SHA
    assert sha(prep.SNAPSHOT / 'config.json') == CONFIG_SHA


def prepare(ids, cases_dir, digest):
    inputs, prepared = {}, {}
    for H in [10, 30]:
        with np.load(cases_dir / f'tool_inputs_H{H}.npz', allow_pickle=False) as arc:
            inputs[H] = {k: arc[k] for k in arc.files}
        assert str(inputs[H]['protocol_sha256']) == digest
        for track in prep.TRACKS:
            prepared[track, H] = shard_inputs(inputs[H], ids, H, digest)
    for t in prep.TRACKS:
        p10, p30 = prepared[t, 10], prepared[t, 30]
        for k in ['case_id', 'pair_id', 'schedule_id', 'metadata_json', 'contexts']:
            assert np.array_equal(p10[k], p30[k]), k
        assert np.array_equal(p10['covariates'], p30['covariates'][:, :, :1034])
        assert len(p10['query_id']) == 4 * len(ids) and len(set(p10['case_id'].tolist())) == len(ids)
    return prepared


def lock_waves(out_root):
    locks = []
    legacy = K.P4 / 'tools' / 'MODEL_WAVE.lock'
    if legacy.is_file():  # never create files in phase4; only wait on an existing legacy lock
        lk = open(legacy, 'r'); fcntl.flock(lk, fcntl.LOCK_EX); locks.append(lk)
    lk = open(out_root / 'MODEL_WAVE.lock', 'a'); fcntl.flock(lk, fcntl.LOCK_EX); locks.append(lk)
    return locks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('production', 'inputs_only'))
    ap.add_argument('--actor', required=True, choices=K.ACTORS)
    ap.add_argument('--cases-dir', type=Path, help='inputs_only fixture directory (derived_manifest + tool_inputs)')
    a = ap.parse_args(argv)
    started = time.time()
    if a.mode == 'production' or a.cases_dir is None:
        digest = K.require_frozen(); cases_dir = P / 'cases'
        gen = json.loads((cases_dir / 'generation_checks.json').read_text())
        assert gen['all_ok'] and gen['protocol_sha256'] == digest
        ids = load_shard(a.actor)
    else:
        digest, cases_dir = K.FIXTURE_DIGEST, a.cases_dir
        ids = [m['case_id'] for m in json.loads((cases_dir / 'derived_manifest.json').read_text()) if K.actor_of(m['case_id']) == a.actor]
    manifest = json.loads((cases_dir / 'derived_manifest.json').read_text()); known = {r['case_id'] for r in manifest}
    assert len(ids) == len(set(ids)) and set(ids) <= known
    prepared = prepare(ids, cases_dir, digest)
    if a.mode == 'inputs_only':
        rep = dict(actor=a.actor, n_cases=len(ids), queries_per_horizon=len(prepared['raw', 10]['query_id']),
                   contexts_shape=list(prepared['raw', 30]['contexts'].shape), covariates_shape=list(prepared['raw', 30]['covariates'].shape),
                   model_loaded=False, official=K.official_digest(digest))
        print('D06_INPUTS_ONLY_OK', json.dumps(rep), flush=True)
        return rep
    check_model_closure()
    out = P / 'tools' / a.actor; out.mkdir(parents=True, exist_ok=True)
    state = dict(task_id=f'D06_A_raw_execution_{a.actor}', actor=a.actor, status='waiting_model_wave_lock', completed_runs={}); progress(state, out)
    locks = lock_waves(P / 'tools')
    state['status'] = 'initializing'; progress(state, out)
    import torch
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(prep.SNAPSHOT), local_files_only=True, force_download=False, device='mps', per_core_batch_size=BATCH))
    runtime = dict(executable=sys.executable, python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, device='mps', threads=1)
    base = dict(actor=a.actor, shard_case_ids_sha256=json.loads((P / f'SHARD_{a.actor.upper()}.json').read_text())['case_ids_sha256'], protocol_sha256=digest,
                protocol_freeze_sha256=sha(P / 'protocol_freeze.json'), driver_sha256=sha(__file__), preparation_sha256=sha(R / 'lib/p5_export.py'),
                frozen_preparation_sha256=sha(R / 'lib/p4_export.py'), D05_driver_sha256=sha(R / 'lib/p4_timesfm_execution.py'),
                D03_driver_sha256=sha(R / 'lib/p3_phase2_timesfm_execution.py'), D03_adapter_sha256=sha(R / 'lib/p3_phase2_timesfm.py'),
                pilot_adapter_sha256=sha(R / 'lib/timesfm_pilot_adapter.py'), checkpoint_revision=prep.SNAPSHOT.name,
                checkpoint_weights_sha256=sha(prep.SNAPSHOT / 'model.safetensors'), checkpoint_config_sha256=sha(prep.SNAPSHOT / 'config.json'),
                api_json=json.dumps(prep.API, sort_keys=True), runtime_json=json.dumps(runtime, sort_keys=True), batch=BATCH, chunk_queries=CHUNK)

    def infer(z, start, stop, H):
        q = np.empty((stop - start, H, 9), np.float32)
        for first in range(start, stop, BATCH):
            last = min(first + BATCH, stop)
            preds = list(model.predict_batch(contexts=list(z['contexts'][first:last]), horizon=H, past_future_covariates=list(z['covariates'][first:last]),
                                             ts_ids=z['query_id'][first:last].tolist(), **prep.API))
            assert len(preds) == last - first
            for i, pred in enumerate(preds):
                q[first - start + i] = checked_prediction(pred.quantiles, pred.ts_id, z['query_id'][first + i], H)
        return q
    smoke = infer(prepared['raw', 10], 0, 4, 10); atomic_npz(out / 'smoke.npz', dict(quantiles=smoke, query_id=prepared['raw', 10]['query_id'][:4], smoke_only=True))
    for track in prep.TRACKS:
        for H in [10, 30]:
            key = f'{track}_H{H}'; z = prepared[track, H]; identity = dict(base, track=track, horizon=H, input_sha256=sha(P / f'cases/tool_inputs_H{H}.npz'))
            fp = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(); identity['run_fingerprint'] = fp
            state.update(status='running', active_run=key, active_queries_done=0); progress(state, out); runstart = time.time(); resumed = 0
            q = np.empty((len(z['query_id']), H, 9), np.float32)
            for start in range(0, len(z['query_id']), CHUNK):
                stop = min(start + CHUNK, len(z['query_id'])); cp = out / 'checkpoints' / key / f'{start:04d}_{stop:04d}.npz'
                if cp.exists():
                    values = load_chunk(cp, fp, z['query_id'][start:stop], start, stop, H); resumed += stop - start
                else:
                    values = infer(z, start, stop, H); atomic_npz(cp, dict(quantiles=values, query_id=z['query_id'][start:stop], start=start, stop=stop, fingerprint=fp))
                q[start:stop] = values; state['active_queries_done'] = stop; progress(state, out); print(key, stop, flush=True)
            path = out / f'timesfm_{key}.npz'; atomic_npz(path, pack_output(q, z, identity)); record = verify(path, z, identity)
            record.update(elapsed_s=time.time() - runstart, resumed_queries=resumed); state['completed_runs'][key] = record; progress(state, out)
    K.require_frozen(); state.update(status='inference_complete', active_run='none'); progress(state, out)
    result = dict(task_id=state['task_id'], actor=a.actor, inference_complete=True, outputs=state['completed_runs'], total_runs=2, total_cases_per_run=len(ids),
                  total_queries=8 * len(ids), total_contrasts=4 * len(ids), elapsed_s=time.time() - started, provenance=base, runtime=runtime, model_instances=1,
                  horizons_independent=True, truth_features=False, legacy_lock_held=len(locks) == 2,
                  environment={k: os.environ.get(k) for k in ['HF_HUB_OFFLINE', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                              'MKL_NUM_THREADS', 'HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME', 'TMPDIR']})
    (out / 'EXECUTION_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n'); print('D06_RAW_TWO_HORIZONS_COMPLETE', a.actor, flush=True)


if __name__ == '__main__':
    main()
