"""Actual D03 inference runner; imports frozen preparation/settings unchanged.
Single offline MPS model; all four tracks and both horizons; strict chunk resume.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
import p3_phase2_timesfm as prep

ROOT=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
P=ROOT/"results/phase2"
PROTOCOL="0b360edd4ecc9bc601376eeaba1f78d5511096c67cb2bd42810ffe010713b794"
BATCH=8
CHUNK=128


def now():return datetime.now(timezone.utc).isoformat()

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def atomic_npz(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp.npz')
 np.savez_compressed(tmp,**data);os.replace(tmp,path)

def checked_prediction(q,pred_id,expected_id,horizon):
 q=np.asarray(q)
 if pred_id!=expected_id or q.shape!=(horizon,9) or not np.isfinite(q).all():
  raise ValueError("Invalid prediction identity/quantile shape/finite values")
 q=q.astype(np.float32)
 if not np.isfinite(q).all():raise ValueError("Forecast overflows float32")
 return q

def load_chunk(path,fingerprint,ids,start,stop,horizon):
 with np.load(path,allow_pickle=False) as z:
  if str(z['fingerprint'])!=fingerprint or int(z['start'])!=start or int(z['stop'])!=stop or not np.array_equal(z['query_id'],ids):
   raise ValueError("Stale/mismatched chunk; no silent reuse")
  q=z['quantiles']
  if q.shape!=(stop-start,horizon,9) or q.dtype!=np.float32 or not np.isfinite(q).all():
   raise ValueError("Invalid saved quantiles")
 return q

def pack_output(q,z,identity):
 H=int(identity['horizon']);q=np.asarray(q)
 if q.shape!=(len(z['query_id']),H,9) or len(z['query_id'])!=2*len(z['pair_id']) or not np.isfinite(q).all():
  raise ValueError("Invalid completed forecast array")
 lo=q[0::2,:,0]-q[1::2,:,8];hi=q[0::2,:,8]-q[1::2,:,0]
 e=q[0::2,:,4]-q[1::2,:,4]
 out=dict(quantiles=q,marginal_widths=q[:,:,8]-q[:,:,0],E_point=e,E_tool=e,
   E_width_proxy=hi-lo,E_proxy_low=lo,E_proxy_high=hi,quantile_levels=np.arange(1,10)/10,
   query_id=z['query_id'],pair_id=z['pair_id'],schedule_id=z['schedule_id'],case_id=z['case_id'],
   contrast_case_id=z['case_id'][::2],metadata_json=z['metadata_json'],
   horizon=H,track=identity['track'],provenance_json=json.dumps(identity,sort_keys=True),
   marginal_proxy_not_joint_interval=True)
 for key,value in identity.items():
  if isinstance(value,(str,int,float,bool)):out[key]=value
 return out

def verify_archive(path,z,identity):
 with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
 if json.loads(str(d['provenance_json']))!=identity:raise ValueError("Archive identity mismatch")
 for key in ['query_id','pair_id','schedule_id','case_id','metadata_json']:
  if not np.array_equal(d[key],z[key]):raise ValueError(f"Archive {key} mismatch")
 q=d['quantiles'];H=identity['horizon']
 if q.shape!=(2160,H,9) or d['E_point'].shape!=(1080,H):raise ValueError("Full output counts/shape mismatch")
 if len(set(d['case_id'].astype(str)))!=540 or len(set(d['pair_id'].astype(str)))!=1080 or len(set(d['query_id'].astype(str)))!=2160:
  raise ValueError("Full output missing/duplicate IDs")
 for key in ['quantiles','E_point','marginal_widths','E_width_proxy','E_proxy_low','E_proxy_high']:
  if not np.isfinite(d[key]).all():raise ValueError(f"Nonfinite {key}")
 if not np.array_equal(d['E_point'],q[0::2,:,4]-q[1::2,:,4]):raise ValueError("Contrast reconstruction mismatch")
 if np.any(np.diff(q,axis=2)<0):raise ValueError("Quantile crossing despite frozen sorted output")
 return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,queries=2160,cases=540,contrasts=1080,
             quantiles_shape=list(q.shape),E_point_shape=list(d['E_point'].shape),nonfinite=0,missing=0,
             quantile_crossings=0,pair_ids_exact=True,median_contrast_reconstructed=True)

def progress(state):
 state['updated_at']=now()
 (P/'A_EXECUTION_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 done=state.get('completed_runs',{})
 note=['# A actual inference progress',f"Task run; updated {state['updated_at']}",
       f"Status: {state['status']}; completed {len(done)}/8 full runs",f"Active: {state.get('active_run','none')}; queries {state.get('active_queries_done',0)}/2160",
       f"Completed run keys: {', '.join(done) or 'none'}",'Checkpoint: exact local snapshot; model count 1; batch8; CPU threads1.',
       'Proof: A_EXECUTION_PREFLIGHT.json; strict chunk identity includes input/protocol/freeze/driver/adapter/model/API/runtime.',
       'Remaining: '+(', '.join(t+f'_H{h}' for t in prep.TRACKS for h in (10,30) if t+f'_H{h}' not in done) or 'none for A model component'),
       'Scope: only A forecasts; W/reference/metrics/figures/final D03 report remain parent-owned.']
 (P/'A_EXECUTION_PROGRESS.md').write_text('\n\n'.join(note)+'\n')


def main():
 started=now();wall=time.time();tools=P/'tools';tools.mkdir(exist_ok=True)
 state=dict(task_id='run',status='verifying',started_at=started,completed_runs={})
 progress(state)
 freeze=json.loads((P/'protocol_freeze.json').read_text());checks=0
 for group in freeze['hashes'].values():
  for path,expected in group.items():
   if sha(ROOT/path)!=expected:raise ValueError(f"Frozen file altered: {path}")
   checks+=1
 if sha(P/'protocol.md')!=PROTOCOL:raise ValueError("Protocol mismatch")
 # Public edition: the checkpoint and engine source are checked against the engine record A_preflight_manifest.json and the
 # tool inputs against the hashes written by p3_phase2_make_cases (generation_checks.json) instead of the original run records.
 pre_manifest=json.loads((P/'A_preflight_manifest.json').read_text())
 generated=json.loads((P/'cases/generation_checks.json').read_text())['tool']
 if sha(prep.SNAPSHOT/'model.safetensors')!=pre_manifest['checkpoint']['weights_sha256']:raise ValueError("Cached weight mismatch")
 if sha(prep.SNAPSHOT/'config.json')!=pre_manifest['checkpoint']['config_sha256']:raise ValueError("Cached config mismatch")
 sources=pre_manifest['source_hashes']
 for path,expected in sources.items():
  if '/timesfm3/torch/' in path and sha(Path(__import__('os').environ.get('TIMESFM_SOURCE','timesfm/src/timesfm3')).parents[1]/path)!=expected:raise ValueError("Installed TimesFM source changed")
 inputs={};prepared={};input_checks={}
 for H in (10,30):
  path=P/f'cases/tool_inputs_H{H}.npz'
  if sha(path)!=generated[str(H)]['sha256']:raise ValueError("Input differs from verified archive")
  with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
  if str(d['protocol_sha256'])!=PROTOCOL:raise ValueError("Input protocol stamp mismatch")
  inputs[H]=d
  for track in prep.TRACKS:prepared[(track,H)]=prep.prepare_track(d,H,track,require_full_grid=True)
  input_checks[str(H)]=dict(queries=2160,cases=540,contrasts=1080,history_pairs_weather_exact=True,full_grid_pass=True)
 # Cross-horizon identity uses case/pair/schedule keys; query IDs intentionally carry H.
 for track in prep.TRACKS:
  a,b=prepared[(track,10)],prepared[(track,30)]
  for k in ['case_id','pair_id','schedule_id','metadata_json']:
   if not np.array_equal(a[k],b[k]):raise ValueError("Independent-horizon identity mismatch")
  if not np.array_equal(a['contexts'],b['contexts']) or not np.array_equal(a['covariates'],b['covariates'][:,:,:1034]):
   raise ValueError("H10/H30 forcing prefix mismatch")
 import torch
 from timesfm3 import TimesFM3Evaluator,ModelConfig
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if not torch.backends.mps.is_available():raise RuntimeError("Requested MPS not available")
 runtime=dict(executable=sys.executable,python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,device='mps',torch_threads=torch.get_num_threads())
 model=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(prep.SNAPSHOT),local_files_only=True,force_download=False,device='mps',per_core_batch_size=BATCH))
 if list(model.config.quantiles)!=[i/10 for i in range(1,10)]:raise ValueError("Checkpoint quantiles mismatch")
 base=dict(protocol_sha256=PROTOCOL,protocol_freeze_sha256=sha(P/'protocol_freeze.json'),driver_sha256=sha(__file__),
   adapter_sha256=sha(ROOT/'lib/p3_phase2_timesfm.py'),pilot_adapter_sha256=sha(ROOT/'lib/timesfm_pilot_adapter.py'),
   checkpoint_revision=prep.SNAPSHOT.name,checkpoint_weights_sha256=pre_manifest['checkpoint']['weights_sha256'],checkpoint_config_sha256=sha(prep.SNAPSHOT/'config.json'),
   batch=BATCH,chunk_queries=CHUNK,device='mps',api_json=json.dumps(prep.API,sort_keys=True),runtime_json=json.dumps(runtime,sort_keys=True))
 def infer(z,start,stop,H):
  q=np.empty((stop-start,H,9),np.float32)
  for first in range(start,stop,BATCH):
   last=min(first+BATCH,stop)
   preds=list(model.predict_batch(contexts=[x for x in z['contexts'][first:last]],horizon=H,
              past_future_covariates=[x for x in z['covariates'][first:last]],ts_ids=z['query_id'][first:last].tolist(),**prep.API))
   if len(preds)!=last-first:raise ValueError("Incomplete batch")
   for i,pred in enumerate(preds):q[first-start+i]=checked_prediction(pred.quantiles,pred.ts_id,z['query_id'][first+i],H)
  return q
 state['status']='actual_smoke';progress(state)
 z=prepared[('raw',10)];sq=infer(z,0,4,10)
 smoke=dict(queries=4,horizon=10,track='raw',shape=list(sq.shape),finite=bool(np.isfinite(sq).all()),pair_ids=z['pair_id'][:2].tolist(),actual_inference=True,started_at=now())
 atomic_npz(tools/'smoke/timesfm_raw_H10_smoke.npz',dict(quantiles=sq,query_id=z['query_id'][:4],pair_id=z['pair_id'][:2],protocol_sha256=PROTOCOL,smoke_only=True))
 (P/'A_ACTUAL_SMOKE.json').write_text(json.dumps(smoke,indent=2)+'\n')
 state['status']='running';state['smoke']=smoke
 for track in prep.TRACKS:
  for H in (10,30):
   key=f'{track}_H{H}';z=prepared[(track,H)];input_sha=sha(P/f'cases/tool_inputs_H{H}.npz')
   identity=dict(base,track=track,horizon=H,input_sha256=input_sha)
   fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
   identity['run_fingerprint']=fingerprint
   state['active_run']=key;state['active_queries_done']=0;progress(state)
   out=tools/f'timesfm_{key}.npz';runstart=time.time();q=np.empty((2160,H,9),np.float32);resumed=0
   if out.exists():
    info=verify_archive(out,z,identity);info['existing_identical_archive_reused']=True
   else:
    for start in range(0,2160,CHUNK):
     stop=min(start+CHUNK,2160);chunk=tools/'checkpoints'/key/f'{start:04d}_{stop:04d}.npz'
     if chunk.exists():
      values=load_chunk(chunk,fingerprint,z['query_id'][start:stop],start,stop,H);resumed+=stop-start
     else:
      values=infer(z,start,stop,H)
      atomic_npz(chunk,dict(quantiles=values,query_id=z['query_id'][start:stop],start=start,stop=stop,fingerprint=fingerprint))
     q[start:stop]=values;state['active_queries_done']=stop;progress(state)
     print(json.dumps(dict(run=key,done=stop,total=2160,elapsed_s=round(time.time()-runstart,2))),flush=True)
    atomic_npz(out,pack_output(q,z,identity));info=verify_archive(out,z,identity)
   info.update(track=track,horizon=H,elapsed_s=time.time()-runstart,resumed_queries=resumed,run_fingerprint=fingerprint)
   state['completed_runs'][key]=info;progress(state)
 state['status']='complete';state['active_run']='none';state['active_queries_done']=0;progress(state)
 manifest=dict(task_id=state['task_id'],component='A model inference only',component_complete=True,started_at=started,finished_at=now(),elapsed_s=time.time()-wall,
   frozen_files_verified=checks,protocol_sha256=PROTOCOL,provenance=base,runtime=runtime,input_checks=input_checks,smoke=smoke,
   outputs=state['completed_runs'],total_runs=8,total_queries=17280,total_contrasts_across_runs=8640,total_cases_per_run=540,
   no_model_truth_or_W_inputs=True,model_instances=1,horizons_independent=True,all_three_filter_scales_retained=True,
   metric_rules='Frozen protocol section9; no ratios/sign/W scores computed by A',env={k:os.environ.get(k) for k in ['PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','HF_HUB_DISABLE_TELEMETRY','HF_HOME','HF_HUB_CACHE','TMPDIR','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
 (P/'A_EXECUTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('ALL_EIGHT_RUNS_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  error=dict(at=now(),error=repr(exc),driver_sha256=sha(__file__))
  (P/'A_EXECUTION_ERROR.json').write_text(json.dumps(error,indent=2)+'\n')
  raise
