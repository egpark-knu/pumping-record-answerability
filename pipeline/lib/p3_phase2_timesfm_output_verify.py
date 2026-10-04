"""Independent saved-output check; invokes only B's pure archive loader."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from datetime import datetime,timezone
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])));P=R/'results/phase2'
sys.path.insert(0,str(R/'lib'))
import p3_phase2_cell_metrics as collector
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((P/'A_EXECUTION_MANIFEST.json').read_text());freeze=json.loads((P/'protocol_freeze.json').read_text())
frozen=[]
for group in freeze['hashes'].values():
 for rel,expected in group.items():
  actual=sha(R/rel);assert actual==expected,rel
  frozen.append({'path':rel,'sha256':actual,'passed':True})
assert len(frozen)==62
assert sha(P/'protocol.md')==m['protocol_sha256']
inputs={}
for H in (10,30):
 with np.load(P/f'cases/tool_inputs_H{H}.npz',allow_pickle=False) as z:inputs[H]={k:z[k] for k in z.files}
loaded=collector.load_tools(P/'tools',m['protocol_sha256']);results={}
for key,rec in m['outputs'].items():
 p=Path(rec['path']);assert sha(p)==rec['sha256']
 with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 H=int(d['horizon']);tr=str(d['track']);a=inputs[H]
 # Input pair_id is per query; archive pair_id is per contrast (a,b).
 for k in ('query_id','schedule_id','case_id','metadata_json'):assert np.array_equal(d[k],a[k]),(key,k)
 assert np.array_equal(d['pair_id'],a['pair_id'][::2])
 assert np.array_equal(a['pair_id'][::2],a['pair_id'][1::2])
 assert len(set(d['case_id']))==540 and len(set(d['query_id']))==2160 and len(set(d['pair_id']))==1080
 q=d['quantiles'];assert q.shape==(2160,H,9) and d['E_point'].shape==(1080,H)
 for k in ('quantiles','E_point','E_tool','marginal_widths','E_width_proxy','E_proxy_low','E_proxy_high'):assert np.isfinite(d[k]).all(),(key,k)
 assert not np.any(np.diff(q,axis=2)<0)
 assert np.array_equal(d['E_point'],q[::2,:,4]-q[1::2,:,4])
 assert np.array_equal(d['E_tool'],d['E_point'])
 assert np.array_equal(d['marginal_widths'],q[:,:,8]-q[:,:,0])
 assert np.array_equal(d['E_proxy_low'],q[::2,:,0]-q[1::2,:,8])
 assert np.array_equal(d['E_proxy_high'],q[::2,:,8]-q[1::2,:,0])
 assert np.array_equal(d['E_width_proxy'],d['E_proxy_high']-d['E_proxy_low'])
 assert bool(d['marginal_proxy_not_joint_interval'])
 identity=json.loads(str(d['provenance_json']))
 for k,v in m['provenance'].items():assert identity[k]==v and str(d[k])==str(v),(key,k)
 assert str(d['input_sha256'])==sha(P/f'cases/tool_inputs_H{H}.npz')
 assert str(d['driver_sha256'])==sha(R/'lib/p3_phase2_timesfm_execution.py')
 assert set(loaded[(tr,H)])-{'_sha'}==set(a['pair_id'].astype(str))
 for pair,e in zip(d['pair_id'].astype(str),d['E_point']):assert np.array_equal(loaded[(tr,H)][pair],e)
 checkpoints=sorted((P/'tools/checkpoints'/key).glob('*.npz'));assert len(checkpoints)==17
 n=0
 for cp in checkpoints:
  with np.load(cp,allow_pickle=False) as c:
   start,stop=int(c['start']),int(c['stop']);assert start==n
   assert str(c['fingerprint'])==str(d['run_fingerprint'])
   assert np.array_equal(c['query_id'],d['query_id'][start:stop])
   assert np.array_equal(c['quantiles'],q[start:stop]);n=stop
 assert n==2160
 results[key]={'sha256':sha(p),'queries':2160,'cases':540,'pairs':1080,'shape':list(q.shape),'finite':True,'exact_ids':True,'median_contrast_exact':True,'collector_passed':True,'all_17_checkpoints_exact':True,'resumed_queries':rec['resumed_queries']}
v={'task_id':'run','verified_at':datetime.now(timezone.utc).isoformat(),'passed':True,'verification_method':'Independent Python NPZ reopen/reconstruct; actual B load_tools only; no W/metric computation','manifest_sha256':sha(P/'A_EXECUTION_MANIFEST.json'),'collector_sha256':sha(R/'lib/p3_phase2_cell_metrics.py'),'verifier_sha256':sha(__file__),'frozen_files_unchanged':frozen,'outputs':results,'total_runs':8,'total_queries':17280,'total_contrasts':8640,'nonfinite':0,'missing':0}
(P/'A_OUTPUT_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({k:v[k] for k in ('passed','total_runs','total_queries','total_contrasts','nonfinite','missing')}))
