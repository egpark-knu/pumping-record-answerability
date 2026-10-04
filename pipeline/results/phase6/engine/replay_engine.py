import sys,os,json,hashlib,time,datetime,fcntl,platform,subprocess
from pathlib import Path
import numpy as np
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3])));E=R/'results/phase6/engine';RE=E/'replay';sys.path.insert(0,str(R/'lib'))
from timesfm_pilot_adapter import API,SNAPSHOT
from p3_phase2_timesfm_execution import checked_prediction

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(f,d):Path(f).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
selection=json.loads((RE/'SELECTION.json').read_text());assert len(selection['selected'])==20
assert selection['protocol_sha256']==sha(E/'PROTOCOL.json')
preflight=json.loads((R/'results/phase2/A_preflight_manifest.json').read_text());sourcehashes={}
for f,want in preflight['source_hashes'].items():
 if '/timesfm3/torch/' in f:
  actual=sha(f);assert actual==want,f;sourcehashes[f]=actual
assert SNAPSHOT.name=='43046b85ec22d584a13f8098c2ed39c889e129c2'
assert sha(SNAPSHOT/'model.safetensors')=='a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
assert sha(SNAPSHOT/'config.json')=='ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6'
for bid,b in selection['batches'].items():
 assert sha(b['input_archive_path'])==b['input_archive_sha256']
 assert json.loads(b['original_provenance']['api_json'])==API
 assert b['original_provenance']['checkpoint_weights_sha256']==sha(SNAPSHOT/'model.safetensors')
 assert b['original_provenance']['checkpoint_revision']==SNAPSHOT.name
 assert b['original_provenance']['batch']==8
locks=[]
for f in [R/'results/phase4/tools/MODEL_WAVE.lock',R/'results/phase5/tools/MODEL_WAVE.lock']:
 if f.exists():
  lk=f.open('r');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lk)
lk=(RE/'MODEL_WAVE.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lk)
ps=subprocess.check_output(['ps','-axo','pid,comm,args'],text=True);save(RE/'PROCESS_PREFLIGHT.json',{'checked_utc':now(),'pid':os.getpid(),'legacy_read_only_locks_held':len(locks)-1,'inventory_python_model_processes':[l for l in ps.splitlines() if 'python' in l.split()[1].lower() and ('timesfm' in l or 'replay_engine.py' in l)]})
import torch
from timesfm3 import TimesFM3Evaluator,ModelConfig
torch.set_num_threads(1);torch.set_num_interop_threads(1)
runtime=dict(executable=sys.executable,python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,device='mps',threads=1)
for b in selection['batches'].values():
 original_runtime=json.loads(b['original_provenance']['runtime_json'])
 if 'torch_threads' in original_runtime:original_runtime['threads']=original_runtime.pop('torch_threads')
 assert runtime==original_runtime,(runtime,original_runtime)
settings={'started_utc':now(),'runtime':runtime,'machine':{'platform':platform.platform(),'processor':platform.processor(),'machine':platform.machine()},'checkpoint':{'path':str(SNAPSHOT),'revision':SNAPSHOT.name,'weights_sha256':sha(SNAPSHOT/'model.safetensors'),'config_sha256':sha(SNAPSHOT/'config.json')},'API':API,'batch':8,'model_sources':sourcehashes,'selection_sha256':sha(RE/'SELECTION.json'),'protocol_sha256':sha(E/'PROTOCOL.json'),'executor_sha256':sha(__file__),'env':{k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},'no_cache_forecast_load_for_new_inference':True}
save(RE/'SETTINGS.json',settings);print('MODEL_INITIALIZING',flush=True)
model=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(SNAPSHOT),local_files_only=True,force_download=False,device='mps',per_core_batch_size=8));assert list(model.config.quantiles)==[i/10 for i in range(1,10)]
print('MODEL_READY',flush=True);results={}
for pass_ in [1,2]:
 for bid,b in selection['batches'].items():
  with np.load(b['input_archive_path'],allow_pickle=False) as arc:z={k:arc[k] for k in arc.files}
  H=int(z['horizon']);start=time.monotonic();preds=list(model.predict_batch(contexts=list(z['contexts']),horizon=H,past_future_covariates=list(z['covariates']),ts_ids=z['query_id'].tolist(),**API));assert len(preds)==len(z['query_id'])
  q=np.stack([checked_prediction(pred.quantiles,pred.ts_id,z['query_id'][i],H) for i,pred in enumerate(preds)])
  file=RE/f'{bid}_fresh_pass{pass_}.npz';np.savez_compressed(file,quantiles=q,query_id=z['query_id'],horizon=H,fresh_model_inference=True)
  results[bid,pass_]=q
  print('FRESH_PASS',pass_,bid,'queries',len(q),'sec',round(time.monotonic()-start,3),flush=True)
comparisons=[]
for x in selection['selected']:
 bid=x['batch_id'];i=x['batch_local_index']
 with np.load(x['input_archive_path'],allow_pickle=False) as z:orig=z['original_quantiles']
 q1=results[bid,1];q2=results[bid,2]
 record={k:x[k] for k in ['row_id','cohort','lead','signal_bin','record_sign_status','batch_id','query_ids']};arms=[]
 for arm,j in [('a',i),('b',i+1)]:
  o=orig[j,:,4];a=q1[j,:,4];b=q2[j,:,4]
  arms.append({'arm':arm,'query_id':x['query_ids'][0 if arm=='a' else 1],'original_q50_sequence':o.tolist(),'fresh_pass1_q50_sequence':a.tolist(),'fresh_pass2_q50_sequence':b.tolist(),'original_endpoint':float(o[-1]),'fresh_pass1_endpoint':float(a[-1]),'fresh_pass2_endpoint':float(b[-1]),'pass1_exact_equal_original':bool(np.array_equal(o,a)),'pass2_exact_equal_original':bool(np.array_equal(o,b)),'fresh_passes_exact_equal':bool(np.array_equal(a,b)),'pass1_max_abs_change_original_m':float(np.max(np.abs(a.astype(float)-o.astype(float)))),'pass2_max_abs_change_original_m':float(np.max(np.abs(b.astype(float)-o.astype(float)))),'fresh_passes_max_abs_change_m':float(np.max(np.abs(b.astype(float)-a.astype(float))))})
 effects=[orig[i,:,4]-orig[i+1,:,4],q1[i,:,4]-q1[i+1,:,4],q2[i,:,4]-q2[i+1,:,4]];original,fresh1,fresh2=effects
 assert float(original[-1])==x['E_engine_m']
 record.update(arms=arms,original_pair_contrast_sequence=original.tolist(),fresh1_pair_contrast_sequence=fresh1.tolist(),fresh2_pair_contrast_sequence=fresh2.tolist(),original_effect_m=float(original[-1]),fresh1_effect_m=float(fresh1[-1]),fresh2_effect_m=float(fresh2[-1]),pass1_contrast_sequence_exact_equal=bool(np.array_equal(original,fresh1)),pass2_contrast_sequence_exact_equal=bool(np.array_equal(original,fresh2)),fresh_contrasts_exact_equal=bool(np.array_equal(fresh1,fresh2)),contrast_max_abs_change_original_m=max(float(np.max(np.abs(fresh1.astype(float)-original.astype(float)))),float(np.max(np.abs(fresh2.astype(float)-original.astype(float))))),endpoint_max_abs_change_original_m=max(abs(float(fresh1[-1])-float(original[-1])),abs(float(fresh2[-1])-float(original[-1]))),E_true_m=x['E_true_m'])
 record['endpoint_change_over_truth']=record['endpoint_max_abs_change_original_m']/abs(x['E_true_m']) if x['E_true_m'] else None
 comparisons.append(record)
summary={'status':'actual_fresh_replay20_complete','completed_utc':now(),'pair_rows':20,'selected_arm_requests':40,'fresh_passes':2,'batch_calls':len(selection['batches'])*2,'actual_requests_including_original_companions':sum(b['stop']-b['start'] for b in selection['batches'].values())*2,'point_definition':'q50, exactly original packed E_point definition; float32 arm subtraction','all_selected_arm_sequences_exact_equal_original':all(a['pass1_exact_equal_original'] and a['pass2_exact_equal_original'] for x in comparisons for a in x['arms']),'all_fresh_arm_sequences_exact_equal':all(a['fresh_passes_exact_equal'] for x in comparisons for a in x['arms']),'max_arm_sequence_abs_change_original_m':max(max(a['pass1_max_abs_change_original_m'],a['pass2_max_abs_change_original_m']) for x in comparisons for a in x['arms']),'max_fresh_arm_sequence_abs_change_m':max(a['fresh_passes_max_abs_change_m'] for x in comparisons for a in x['arms']),'all_pair_contrast_sequences_exact_equal_original':all(x['pass1_contrast_sequence_exact_equal'] and x['pass2_contrast_sequence_exact_equal'] for x in comparisons),'max_contrast_sequence_abs_change_original_m':max(x['contrast_max_abs_change_original_m'] for x in comparisons),'max_pair_endpoint_change_original_m':max(x['endpoint_max_abs_change_original_m'] for x in comparisons),'max_endpoint_change_over_truth':max(x['endpoint_change_over_truth'] for x in comparisons),'comparisons':comparisons,'settings_sha256':sha(RE/'SETTINGS.json'),'output_hashes':{str(f):sha(f) for f in RE.glob('*_fresh_pass*.npz')},'no_original_mutation':True}
save(RE/'REPLAY_RECEIPT.json',summary);print('REPLAY_COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ['comparisons','output_hashes']}),flush=True)
