"""D05 raw-only offline forecasts for one actor shard, one model instance, exact provenance chunks.

Usage: env/.venv(0_zero_shot TimesFM env)/bin/python lib/p4_timesfm_execution.py --actor shard_b|shard_a
Outputs go only to results/phase4/tools/<actor>/ (chunks, packed raw H10/H30 archives, state, manifest). Model memory
safety: an exclusive flock on results/phase4/tools/MODEL_WAVE.lock admits one TimesFM instance at a time across both
actors (sequential model waves); the lock is held until the process exits, and the second actor blocks until then. H10 and H30 are independent runs.
"""
import argparse,fcntl,json,hashlib,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import p4_export as prep
from p3_phase2_timesfm_execution import sha,atomic_npz,checked_prediction,load_chunk,pack_output
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])));P=R/'results/phase4';D=R/'results/phase2'
BATCH=8;CHUNK=128

def hash_items(freeze):
 """Phase2 stores path hashes in named groups; phase3 stores a flat path map."""
 hashes=freeze.get('hashes') or {}
 if not hashes:return []
 sample=next(iter(hashes.values()))
 if isinstance(sample,dict):
  items=[]
  for group in hashes.values():items.extend(group.items())
  return items
 return list(hashes.items())

def frozen():
 from p4_contracts import require_frozen
 return require_frozen(),['PREPARE_OLD_HASHES.json + complete D05 freeze']

def verify(path,z,identity):
 with np.load(path,allow_pickle=False) as a:d={k:a[k] for k in a.files}
 assert json.loads(str(d['provenance_json']))==identity
 for k in ['query_id','pair_id','schedule_id','case_id','metadata_json']:assert np.array_equal(d[k],z[k]),k
 H=identity['horizon'];q=d['quantiles'];assert q.shape==(len(z['query_id']),H,9) and d['E_point'].shape==(len(z['pair_id']),H)
 assert len(set(d['case_id']))==len(set(z['case_id'])) and len(set(d['query_id']))==len(z['query_id']) and len(set(d['pair_id']))==len(z['pair_id'])
 for k in ['quantiles','E_point','E_tool','marginal_widths','E_proxy_low','E_proxy_high','E_width_proxy']:assert np.isfinite(d[k]).all(),k
 assert np.array_equal(d['E_point'],q[::2,:,4]-q[1::2,:,4]) and not np.any(np.diff(q,axis=2)<0)
 return dict(path=str(path),sha256=sha(path),cases=len(set(z['case_id'])),queries=len(z['query_id']),contrasts=len(z['pair_id']),shape=list(q.shape),nonfinite=0,missing=0)

def progress(state,out):
 state['updated_at']=datetime.now(timezone.utc).isoformat()
 (out/'EXECUTION_STATE.json').write_text(json.dumps(state,indent=2)+'\n')

def shard_inputs(inputs,ids,H,digest):
 import p4_export as ex
 z=ex.subset(inputs,ids)
 if str(z['protocol_sha256'])!=digest:raise PermissionError('Tool archive protocol stamp differs')
 return ex.prepare_track(z,H,'raw',ids)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--actor',required=True,choices=('shard_a','shard_b'));a=ap.parse_args()
 import p4_partition as part
 started=time.time();digest,checks=frozen();assert json.loads((P/'cases/generation_checks.json').read_text())['all_ok'];manifest=json.loads((P/'cases/derived_manifest.json').read_text())
 ids=part.load(a.actor);known={r['case_id'] for r in manifest}
 assert len(ids)==len(set(ids))==1440 and set(ids)<=known
 out=P/'tools'/a.actor;out.mkdir(parents=True,exist_ok=True)
 source=json.loads((D/'A_preflight_manifest.json').read_text())
 for path,want in source['source_hashes'].items():
  if '/timesfm3/torch/' in path:assert sha(Path(__import__('os').environ.get('TIMESFM_SOURCE','timesfm/src/timesfm3')).parents[1]/path)==want,path  # public edition: engine-source paths are relative to the TimesFM checkout (TIMESFM_SOURCE)
 assert sha(prep.SNAPSHOT/'model.safetensors')=='a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
 assert sha(prep.SNAPSHOT/'config.json')=='ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6'
 inputs={};prepared={}
 for H in [10,30]:
  with np.load(P/f'cases/tool_inputs_H{H}.npz',allow_pickle=False) as arc:inputs[H]={k:arc[k] for k in arc.files}
  assert str(inputs[H]['protocol_sha256'])==digest
  for track in prep.TRACKS:prepared[track,H]=shard_inputs(inputs[H],ids,H,digest)
 for t in prep.TRACKS:
  p10,p30=prepared[t,10],prepared[t,30]
  for k in ['case_id','pair_id','schedule_id','metadata_json','contexts']:assert np.array_equal(p10[k],p30[k]),k
  assert np.array_equal(p10['covariates'],p30['covariates'][:,:,:1034])
 state=dict(task_id=f'D05_raw_execution_{a.actor}',actor=a.actor,status='waiting_model_wave_lock',completed_runs={});progress(state,out)
 lock=open(P/'tools'/'MODEL_WAVE.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 state['status']='initializing';progress(state,out)
 import torch
 from timesfm3 import TimesFM3Evaluator,ModelConfig
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 model=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(prep.SNAPSHOT),local_files_only=True,force_download=False,device='mps',per_core_batch_size=BATCH))
 runtime=dict(executable=sys.executable,python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,device='mps',threads=1)
 base=dict(actor=a.actor,shard_case_ids_sha256=json.loads((P/f'SHARD_{a.actor.upper()}.json').read_text())['case_ids_sha256'],protocol_sha256=digest,protocol_freeze_sha256=sha(P/'protocol_freeze.json'),driver_sha256=sha(__file__),preparation_sha256=sha(R/'lib/p4_export.py'),D03_driver_sha256=sha(R/'lib/p3_phase2_timesfm_execution.py'),D03_adapter_sha256=sha(R/'lib/p3_phase2_timesfm.py'),pilot_adapter_sha256=sha(R/'lib/timesfm_pilot_adapter.py'),checkpoint_revision=prep.SNAPSHOT.name,checkpoint_weights_sha256=sha(prep.SNAPSHOT/'model.safetensors'),checkpoint_config_sha256=sha(prep.SNAPSHOT/'config.json'),api_json=json.dumps(prep.API,sort_keys=True),runtime_json=json.dumps(runtime,sort_keys=True),batch=BATCH,chunk_queries=CHUNK)
 def infer(z,start,stop,H):
  q=np.empty((stop-start,H,9),np.float32)
  for first in range(start,stop,BATCH):
   last=min(first+BATCH,stop);preds=list(model.predict_batch(contexts=list(z['contexts'][first:last]),horizon=H,past_future_covariates=list(z['covariates'][first:last]),ts_ids=z['query_id'][first:last].tolist(),**prep.API))
   assert len(preds)==last-first
   for i,pred in enumerate(preds):q[first-start+i]=checked_prediction(pred.quantiles,pred.ts_id,z['query_id'][first+i],H)
  return q
 smoke=infer(prepared['raw',10],0,4,10);atomic_npz(out/'smoke.npz',dict(quantiles=smoke,query_id=prepared['raw',10]['query_id'][:4],smoke_only=True))
 for track in prep.TRACKS:
  for H in [10,30]:
   key=f'{track}_H{H}';z=prepared[track,H];identity=dict(base,track=track,horizon=H,input_sha256=sha(P/f'cases/tool_inputs_H{H}.npz'))
   fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();identity['run_fingerprint']=fp
   state.update(status='running',active_run=key,active_queries_done=0);progress(state,out);runstart=time.time();resumed=0
   q=np.empty((len(z['query_id']),H,9),np.float32)
   for start in range(0,len(z['query_id']),CHUNK):
    stop=min(start+CHUNK,len(z['query_id']));cp=out/'checkpoints'/key/f'{start:04d}_{stop:04d}.npz'
    if cp.exists():values=load_chunk(cp,fp,z['query_id'][start:stop],start,stop,H);resumed+=stop-start
    else:
     values=infer(z,start,stop,H);atomic_npz(cp,dict(quantiles=values,query_id=z['query_id'][start:stop],start=start,stop=stop,fingerprint=fp))
    q[start:stop]=values;state['active_queries_done']=stop;progress(state,out);print(key,stop,flush=True)
   path=out/f'timesfm_{key}.npz';atomic_npz(path,pack_output(q,z,identity));record=verify(path,z,identity);record.update(elapsed_s=time.time()-runstart,resumed_queries=resumed);state['completed_runs'][key]=record;progress(state,out)
 frozen();state.update(status='inference_complete',active_run='none');progress(state,out)
 result=dict(task_id=state['task_id'],actor=a.actor,inference_complete=True,normalization_complete=False,outputs=state['completed_runs'],total_runs=2,total_cases_per_run=len(ids),total_queries=8*len(ids),total_contrasts=4*len(ids),elapsed_s=time.time()-started,provenance=base,runtime=runtime,frozen_files_verified=len(checks),model_instances=1,horizons_independent=True,truth_features=False,environment={k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','HF_HOME','HF_HUB_CACHE','TORCH_HOME','TMPDIR']})
 (out/'EXECUTION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print('D05_RAW_TWO_HORIZONS_COMPLETE',a.actor,flush=True)
if __name__=='__main__':main()
