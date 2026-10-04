"""Scoped frozen TimesFM execution and provenance driver; no W/truth scoring."""
import datetime, hashlib, importlib.metadata, importlib.util, json, os
from pathlib import Path
import subprocess, sys, time
import numpy as np
import torch
ROOT=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[2])))
P=ROOT/'results/pilot'
PYTHON=Path('python')
ADAPTER=ROOT/'lib/timesfm_pilot_adapter.py'
EXPECTED='61f7fa696cfbca26e8f46ace79ca5432e0c644a667ef9eccc7fc7cc27bd12bad'

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def save(path,obj): Path(path).write_text(json.dumps(obj,indent=2)+'\n')

proposal=json.loads((P/'model_interface_proposal.json').read_text())
ready=json.loads((P/'cases_ready.json').read_text())
freeze=json.loads((P/'protocol_freeze.json').read_text())
assert sha(P/'protocol.md')==EXPECTED==ready['protocol']['sha256']==freeze['protocol_sha256']
assert sha(ADAPTER)==freeze['hashes']['lib/timesfm_pilot_adapter.py']
assert sha(P/'model_interface_proposal.json')==freeze['hashes']['model_interface_proposal']
spec=importlib.util.spec_from_file_location('adapter',ADAPTER)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
assert a.API==proposal['model_api']['predict_batch_kwargs']

manifest={'status':'preflight','started_at':stamp(),'task_id':'run',
 'protocol_sha256':EXPECTED,'adapter_sha256':sha(ADAPTER),'driver_sha256':sha(__file__),
 'freeze_manifest_sha256':sha(P/'protocol_freeze.json'),'cases_ready_sha256':sha(P/'cases_ready.json'),
 'runtime':{'executable':sys.executable,'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','huggingface-hub','safetensors']},
 'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available(),'torch_threads':torch.get_num_threads(),
 'hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),'architecture':subprocess.check_output(['uname','-m'],text=True).strip()},
 'api':a.API,'checkpoint_revision':a.SNAPSHOT.name,'checkpoint_path':str(a.SNAPSHOT),
 'environment':{k:os.environ.get(k) for k in proposal['execution']['environment']},
 'inputs':{},'attempts':[],'outputs':{},'source_hashes':{},'negative_checks':{},
 'interpretation':'marginal-range width proxy; not a joint E interval or calibration proof'}
for filename in ['config.json','model.safetensors']:
 f=a.SNAPSHOT/filename
 manifest.setdefault('checkpoint_files',{})[filename]={'path':str(f),'resolved':str(f.resolve()),'bytes':f.stat().st_size,'sha256':sha(f)}
for filename in ['torch/evaluator.py','torch/timesfm3_forecaster.py','torch/model.py','torch/util.py','torch/transformer.py']:
 f=Path(__import__("os").environ.get("TIMESFM_SOURCE", "timesfm/src/timesfm3"))/filename
 manifest['source_hashes'][str(f)]=sha(f)
all_data={}
for H in [10,30]:
 inputpath=ROOT/ready['archives'][f'tool_inputs_H{H}']['path']
 assert sha(inputpath)==ready['archives'][f'tool_inputs_H{H}']['sha256']
 with np.load(inputpath,allow_pickle=False) as z: data={k:z[k] for k in z.files}
 assert set(data)=={'head','pumping','rainfall','query_id','case_id','pair_id','schedule_id','metadata_json','protocol_sha256'}
 order,ids=a.validate(data,H)
 assert len(order)==800 and len(set(ids['case_id']))==200 and len(set(ids['pair_id']))==400
 assert str(data['protocol_sha256'].item())==EXPECTED
 for i in range(0,800,2):
  ia,ib=order[i:i+2];meta=json.loads(ids['metadata_json'][ia]);q0=data['pumping'][ia,1023]
  assert q0>0 and np.all(data['pumping'][ia,1024:]==q0)
  expected_b=0 if meta['schedule_pair']=='P1_continue_vs_stop' else 1.5*q0
  assert meta['schedule_pair'] in ['P1_continue_vs_stop','P2_current_vs_1p5x']
  assert np.allclose(data['pumping'][ib,1024:],expected_b,rtol=1e-14,atol=1e-12)
 manifest['inputs'][str(H)]={'path':str(inputpath),'sha256':sha(inputpath),'rows':800,'cases':200,'contrasts':400,'head_shape':list(data['head'].shape),'pumping_shape':list(data['pumping'].shape),'rainfall_shape':list(data['rainfall'].shape),'finite_and_paired_dose_checked':True,'truth_keys_absent':True}
 all_data[H]=(data,order,ids)
assert np.array_equal(all_data[10][0]['head'],all_data[30][0]['head'])
assert np.array_equal(all_data[10][0]['pumping'],all_data[30][0]['pumping'][:,:1034])
assert np.array_equal(all_data[10][0]['rainfall'],all_data[30][0]['rainfall'][:,:1034])
manifest['cross_horizon_same_history_and_future_prefix']=True
(P/'tools').mkdir(exist_ok=True)
(P/'A_timesfm_logs').mkdir(exist_ok=True)
base=[str(PYTHON),'-B',str(ADAPTER),'--protocol',str(P/'protocol.md'),'--expected-protocol-sha256',EXPECTED,'--device','mps','--batch','8']
for H in [10,30]:
 cmd=base+['--cases',manifest['inputs'][str(H)]['path'],'--horizon',str(H),'--output',str(P/f'tools/timesfm_H{H}.npz'),'--validate-only']
 r=subprocess.run(cmd,capture_output=True,text=True)
 assert r.returncode==0,(r.stdout,r.stderr)
 manifest['inputs'][str(H)]['cli_validation']=r.stdout.strip()
cmd=base+['--cases',manifest['inputs']['10']['path'],'--horizon','10','--output',str(P/'tools/timesfm_H10.npz'),'--validate-only']
cmd[cmd.index('--expected-protocol-sha256')+1]='0'*64
r=subprocess.run(cmd,capture_output=True,text=True)
assert r.returncode!=0 and 'Frozen protocol hash mismatch' in r.stderr
manifest['negative_checks']['wrong_protocol_hash']={'rejected':True,'diagnostic':r.stderr.strip()}
save(P/'A_timesfm_inference_manifest.json',manifest)

def progress(state,H=None,attempt=None,elapsed=None):
 save(P/'A_timesfm_progress.json',{'updated_at':stamp(),'status':state,'horizon':H,'attempt':attempt,'elapsed_seconds':elapsed,'completed_horizons':list(manifest['outputs']),'protocol_sha256':EXPECTED})
 print(json.dumps({'status':state,'horizon':H,'elapsed_seconds':elapsed}),flush=True)

for H in [10,30]:
 out=P/f'tools/timesfm_H{H}.npz'
 if out.exists(): raise FileExistsError(f'Refusing to overwrite existing {out}')
 success=False
 for device,batch in [('mps',8),('mps',2),('cpu',2)]:
  cmd=[str(PYTHON),'-B',str(ADAPTER),'--cases',manifest['inputs'][str(H)]['path'],'--horizon',str(H),'--protocol',str(P/'protocol.md'),'--expected-protocol-sha256',EXPECTED,'--output',str(out),'--device',device,'--batch',str(batch)]
  log=P/f'A_timesfm_logs/H{H}_{device}_batch{batch}.log'
  attempt={'horizon':H,'device':device,'batch':batch,'argv':cmd,'started_at':stamp(),'log':str(log)}
  manifest['attempts'].append(attempt);manifest['status']='running';save(P/'A_timesfm_inference_manifest.json',manifest)
  t=time.monotonic();progress('running',H,attempt,0)
  with log.open('w') as lf:
   proc=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=ROOT)
   attempt['pid']=proc.pid
   while proc.poll() is None:
    time.sleep(30)
    progress('running',H,attempt,time.monotonic()-t)
   attempt.update(returncode=proc.returncode,elapsed_seconds=time.monotonic()-t,ended_at=stamp(),log_sha256=sha(log))
  save(P/'A_timesfm_inference_manifest.json',manifest)
  if proc.returncode==0:
   success=True;break
  diagnostic=log.read_text()
  attempt['failure_diagnostic_tail']=diagnostic[-6000:]
  save(P/'A_timesfm_inference_manifest.json',manifest)
  if not any(word in diagnostic.lower() for word in ['out of memory','mps','not implemented','unsupported','metal','allocation']):
   raise RuntimeError('Non-resource/API error requires targeted repair: '+diagnostic[-4000:])
 if not success: raise RuntimeError(f'All frozen device/batch fallbacks failed for H{H}')
 with np.load(out,allow_pickle=False) as z: o={k:z[k] for k in z.files}
 data,order,ids=all_data[H];q=o['quantiles']
 assert q.shape==(800,H,9) and np.isfinite(q).all()
 assert np.array_equal(o['query_id'],ids['query_id'][order])
 assert np.array_equal(o['case_id'],ids['case_id'][order])
 assert np.array_equal(o['metadata_json'],ids['metadata_json'][order])
 assert np.array_equal(o['pair_id'],ids['pair_id'][order][::2])
 assert str(o['input_sha256'].item())==manifest['inputs'][str(H)]['sha256']
 assert str(o['protocol_sha256'].item())==EXPECTED
 assert json.loads(str(o['api_json'].item()))==a.API
 assert np.all(np.diff(q,axis=-1)>=0)
 assert np.array_equal(o['E_point'],q[::2,:,4]-q[1::2,:,4])
 assert np.array_equal(o['marginal_widths'],q[:,:,8]-q[:,:,0])
 assert np.allclose(o['E_width_proxy'],o['marginal_widths'][::2]+o['marginal_widths'][1::2],rtol=1e-6,atol=2e-6)
 # Duplicate continue schedule across both pairs is a determinism diagnostic, no ranking/score.
 repeats=[]
 for case in set(o['case_id'].astype(str)):
  ia=np.flatnonzero((o['case_id'].astype(str)==case)&(o['schedule_id'].astype(str)=='a'))
  assert len(ia)==2
  repeats.append(float(np.max(np.abs(q[ia[0]]-q[ia[1]]))))
 manifest['outputs'][str(H)]={'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'quantiles_shape':list(q.shape),'E_point_shape':list(o['E_point'].shape),'all_rows_finite':True,'query_id_metadata_complete':True,'quantile_crossings_after_frozen_sort':int(np.count_nonzero(np.diff(q,axis=-1)<0)),'marginal_width_min':float(np.min(o['marginal_widths'])),'marginal_width_max':float(np.max(o['marginal_widths'])),'duplicate_continue_maxabs_m':max(repeats),'device':device,'batch':batch,'elapsed_seconds':attempt['elapsed_seconds'],'proxy_not_joint_interval':True}
 save(P/'A_timesfm_inference_manifest.json',manifest);progress('horizon_complete',H,attempt,attempt['elapsed_seconds'])
assert sha(P/'protocol.md')==EXPECTED and sha(ADAPTER)==manifest['adapter_sha256']
manifest.update(status='complete',ended_at=stamp(),validated_horizons=[10,30],adapter_changed=False,protocol_changed=False)
save(P/'A_timesfm_inference_manifest.json',manifest);progress('complete')
