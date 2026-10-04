"""Actual D04 all690 offline forecasts, serial one model, exact provenance chunks."""
import json,hashlib,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import p3_phase3_timesfm as prep
from p3_phase2_timesfm_execution import sha,atomic_npz,checked_prediction,load_chunk,pack_output
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])));P=R/'results/phase3';D=R/'results/phase2'
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
 digest=sha(P/'protocol.md')
 assert digest==(P/'protocol.sha256').read_text().split()[0]
 checks=[]
 for folder in [D,P]:
  f=json.loads((folder/'protocol_freeze.json').read_text())
  for path,want in hash_items(f):
   actual=sha(R/path);assert actual==want,path;checks.append(path)
 return digest,checks

def verify(path,z,identity):
 with np.load(path,allow_pickle=False) as a:d={k:a[k] for k in a.files}
 assert json.loads(str(d['provenance_json']))==identity
 for k in ['query_id','pair_id','schedule_id','case_id','metadata_json']:assert np.array_equal(d[k],z[k]),k
 H=identity['horizon'];q=d['quantiles'];assert q.shape==(2760,H,9) and d['E_point'].shape==(1380,H)
 assert len(set(d['case_id']))==690 and len(set(d['query_id']))==2760 and len(set(d['pair_id']))==1380
 for k in ['quantiles','E_point','E_tool','marginal_widths','E_proxy_low','E_proxy_high','E_width_proxy']:assert np.isfinite(d[k]).all(),k
 assert np.array_equal(d['E_point'],q[::2,:,4]-q[1::2,:,4]) and not np.any(np.diff(q,axis=2)<0)
 return dict(path=str(path),sha256=sha(path),cases=690,queries=2760,contrasts=1380,shape=list(q.shape),nonfinite=0,missing=0)

def progress(state):
 state['updated_at']=datetime.now(timezone.utc).isoformat()
 (P/'A_EXECUTION_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 (P/'A_PROGRESS.md').write_text('# A actual progress\n\n'+json.dumps(state,indent=2)+'\n\nOwn inference + normalization component only; parent W/reference/analysis/figures/report/review/delivery outside scope.\n')

def main():
 started=time.time();digest,checks=frozen();assert json.loads((P/'cases/generation_checks.json').read_text())['all_ok'];manifest=json.loads((P/'cases/derived_manifest.json').read_text());ids=[r['case_id'] for r in manifest]
 assert len(ids)==len(set(ids))==690
 source=json.loads((D/'A_preflight_manifest.json').read_text())
 for path,want in source['source_hashes'].items():
  if '/timesfm3/torch/' in path:assert sha(Path(__import__('os').environ.get('TIMESFM_SOURCE','timesfm/src/timesfm3')).parents[1]/path)==want,path  # public edition: engine-source paths are relative to the TimesFM checkout (TIMESFM_SOURCE)
 assert sha(prep.SNAPSHOT/'model.safetensors')=='a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
 assert sha(prep.SNAPSHOT/'config.json')=='ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6'
 inputs={};prepared={}
 for H in [10,30]:
  with np.load(P/f'cases/tool_inputs_H{H}.npz',allow_pickle=False) as a:inputs[H]={k:a[k] for k in a.files}
  assert str(inputs[H]['protocol_sha256'])==digest
  for track in prep.TRACKS:prepared[track,H]=prep.prepare_track(inputs[H],H,track,ids)
 for t in prep.TRACKS:
  a,b=prepared[t,10],prepared[t,30]
  for k in ['case_id','pair_id','schedule_id','metadata_json','contexts']:assert np.array_equal(a[k],b[k]),k
  assert np.array_equal(a['covariates'],b['covariates'][:,:,:1034])
 state=dict(task_id='run',status='initializing',completed_runs={});progress(state)
 import torch
 from timesfm3 import TimesFM3Evaluator,ModelConfig
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 model=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(prep.SNAPSHOT),local_files_only=True,force_download=False,device='mps',per_core_batch_size=BATCH))
 runtime=dict(executable=sys.executable,python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,device='mps',threads=1)
 base=dict(protocol_sha256=digest,protocol_freeze_sha256=sha(P/'protocol_freeze.json'),driver_sha256=sha(__file__),preparation_sha256=sha(R/'lib/p3_phase3_timesfm.py'),D03_driver_sha256=sha(R/'lib/p3_phase2_timesfm_execution.py'),D03_adapter_sha256=sha(R/'lib/p3_phase2_timesfm.py'),pilot_adapter_sha256=sha(R/'lib/timesfm_pilot_adapter.py'),checkpoint_revision=prep.SNAPSHOT.name,checkpoint_weights_sha256=sha(prep.SNAPSHOT/'model.safetensors'),checkpoint_config_sha256=sha(prep.SNAPSHOT/'config.json'),api_json=json.dumps(prep.API,sort_keys=True),runtime_json=json.dumps(runtime,sort_keys=True),batch=BATCH,chunk_queries=CHUNK)
 def infer(z,start,stop,H):
  q=np.empty((stop-start,H,9),np.float32)
  for first in range(start,stop,BATCH):
   last=min(first+BATCH,stop);preds=list(model.predict_batch(contexts=list(z['contexts'][first:last]),horizon=H,past_future_covariates=list(z['covariates'][first:last]),ts_ids=z['query_id'][first:last].tolist(),**prep.API))
   assert len(preds)==last-first
   for i,pred in enumerate(preds):q[first-start+i]=checked_prediction(pred.quantiles,pred.ts_id,z['query_id'][first+i],H)
  return q
 smoke=infer(prepared['raw',10],0,4,10);atomic_npz(P/'tools/smoke.npz',dict(quantiles=smoke,query_id=prepared['raw',10]['query_id'][:4],smoke_only=True))
 for track in prep.TRACKS:
  for H in [10,30]:
   key=f'{track}_H{H}';z=prepared[track,H];identity=dict(base,track=track,horizon=H,input_sha256=sha(P/f'cases/tool_inputs_H{H}.npz'))
   fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();identity['run_fingerprint']=fp
   state.update(status='running',active_run=key,active_queries_done=0);progress(state);runstart=time.time();resumed=0
   q=np.empty((2760,H,9),np.float32)
   for start in range(0,2760,CHUNK):
    stop=min(start+CHUNK,2760);cp=P/'tools/checkpoints'/key/f'{start:04d}_{stop:04d}.npz'
    if cp.exists():values=load_chunk(cp,fp,z['query_id'][start:stop],start,stop,H);resumed+=stop-start
    else:
     values=infer(z,start,stop,H);atomic_npz(cp,dict(quantiles=values,query_id=z['query_id'][start:stop],start=start,stop=stop,fingerprint=fp))
    q[start:stop]=values;state['active_queries_done']=stop;progress(state);print(key,stop,flush=True)
   path=P/'tools'/f'timesfm_{key}.npz';atomic_npz(path,pack_output(q,z,identity));record=verify(path,z,identity);record.update(elapsed_s=time.time()-runstart,resumed_queries=resumed);state['completed_runs'][key]=record;progress(state)
 frozen();state.update(status='inference_complete',active_run='none');progress(state)
 result=dict(task_id=state['task_id'],recovery_turn_id='run',recovery_slot='executor',inference_complete=True,normalization_complete=False,outputs=state['completed_runs'],total_runs=8,total_cases_per_run=690,total_queries=22080,total_contrasts=11040,elapsed_s=time.time()-started,provenance=base,runtime=runtime,frozen_files_verified=len(checks),model_instances=1,horizons_independent=True,truth_features=False,environment={k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','HF_HOME','HF_HUB_CACHE','TORCH_HOME','TMPDIR']})
 (P/'A_EXECUTION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print('ALL_EIGHT_NEW690_RUNS_COMPLETE',flush=True)
if __name__=='__main__':main()
