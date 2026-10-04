from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,sys,hashlib,csv,datetime,time
from collections import Counter
import numpy as np
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3])));P=R/'results/phase8';sys.path.insert(0,str(R/'lib'))
from p3_wenvelope import nat_columns,t95_from_step
from p3_kernels import hantush_block
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def walk(v,path='$'):
 if isinstance(v,dict):
  if 'x' in v and 'lin' in v:yield path,v
  for k,q in v.items():yield from walk(q,path+'.'+k)
 elif isinstance(v,list):
  for k,q in enumerate(v):yield from walk(q,f'{path}[{k}]')
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
records=json.loads((P/'data/record_manifest.json').read_text());results=[];diag=[];types=Counter();changed=[];t=time.time();unique_count=0;member_count=0
for i,m in enumerate(records):
 assert sha(m['w_path'])==m['w_sha256'] and sha(m['tf_path'])==m['tf_sha256']
 w=json.loads(Path(m['w_path']).read_text());outerhash=w['tf_input_sha256'];assert outerhash==m['tf_sha256'];w=w if m['variant']=='original' else w['result']
 members=list(walk(w));member_count+=len(members);candidates={}
 for path,q in members:
  lin=q.get('lin');x=q.get('x')
  if not isinstance(lin,list) or not isinstance(x,list) or len(lin)!=5 or len(x)!=5:types['non_member_shape']+=1;continue
  if lin[0]!=0:types['nonzero_gain']+=1;continue
  k=(tuple(x),tuple(lin));entry=candidates.setdefault(k,dict(member=q,paths=[]));entry['paths'].append(path)
 with np.load(m['tf_path'],allow_pickle=False) as f:y=np.asarray(f['head_context'],float);rain=np.asarray(f['rain'],float)
 assert y.shape==(1024,) and rain.size>=1024
 eb=10*np.ptp(y);bounds=np.array([[.5,5],[0,2],[np.log10(5),np.log10(500)],[0,3.5],[m['b_floor'],np.log10(25)]])
 valid=[]
 for key,entry in candidates.items():
  q=entry['member'];x=np.array(q['x'],float);lin=np.array(q['lin'],float);reason=None;t95=None;sse=None
  if not np.isfinite(x).all() or not np.isfinite(lin).all():reason='nonfinite_parameters'
  elif q.get('A',0)!=0:reason='gain_field_inconsistent_with_zero_linear_gain'
  elif not ((x>=bounds[:,0])&(x<=bounds[:,1])).all():reason='nonlinear_bounds'
  elif abs(lin[1])>eb or abs(lin[4])>eb or lin[3]<0:reason='linear_bounds'
  else:
   _,S=hantush_block([10**x[3]],[10**x[4]],1054);t95=float(t95_from_step(S)[0]);t95=None if not np.isfinite(t95) else t95
   if t95 is None or t95>1000:reason='t95_outside_frozen_family'
   else:
    nc=nat_columns(x[None,:3],rain,float(rain[:1024].mean()))[0];X=np.column_stack([-(1-S[0,1:1025]),np.ones(1024),nc]);res=y-y.mean()-X@lin[1:];sse=float(res@res)
    if not np.isfinite(sse):reason='nonfinite_objective';sse=None
  unique_count+=1;types['valid' if reason is None else reason]+=1
  z=dict(source_phase=m['source_phase'],case_id=m['case_id'],record_set=m['record_set'],active=m['active'],tf_sha256=m['tf_sha256'],w_sha256=m['w_sha256'],witness_paths=json.dumps(entry['paths']),x=json.dumps(q['x']),lin=json.dumps(q['lin']),tau=m['tau'],valid=reason is None,rejection_reason=reason,direct_sse=sse,t95=t95,feasible_null=reason is None and sse<=m['tau'],saved_sse=q.get('sse'))
  diag.append(z)
  if reason is None:valid.append(z)
 raw=None
 if m['active']:raw=json.loads((P/'calculations/null_fits'/f"{m['source_phase']}__{m['case_id']}.json").read_text())
 raw_sse=None if raw is None else raw['SSE_min'];vbest=min(valid,key=lambda z:z['direct_sse']) if valid else None;vsse=None if vbest is None else vbest['direct_sse'];best=min([z for z in [raw_sse,vsse] if z is not None],default=None)
 detected=None if not m['active'] or best is None else bool(best>m['tau']);old=None if raw is None else raw['detected'];override=bool(old is True and detected is False)
 z=dict(source_phase=m['source_phase'],case_id=m['case_id'],record_set=m['record_set'],active=m['active'],tau=m['tau'],two_start_SSE_min=raw_sse,two_start_numerical_positive=old,saved_null_member_count=len(candidates),valid_saved_null_count=len(valid),minimum_valid_saved_null_sse=vsse,best_known_null_SSE=best,effect_detected=detected,feasible_null_override=override,best_saved_null_witness=None if vbest is None else vbest)
 results.append(z)
 if override:changed.append(z)
 if (i+1)%100==0 or i+1==len(records):
  v=dict(status='complete' if i+1==len(records) else 'working',records_scanned=i+1,total=len(records),unique_zero_members=unique_count,overrides=len(changed),elapsed_seconds=time.time()-t);atomic(P/'calculations/CORRECTION_PROGRESS.json',v);print(json.dumps(v),flush=True)
with (P/'data/A_saved_null_inventory.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(diag[0]));wr.writeheader();wr.writerows(diag)
atomic(P/'data/A_corrected_null_records.json',results);atomic(P/'data/A_detection_correction_delta.json',changed)
summary=dict(status='saved_null_inventory_fold_complete',records=6420,active=4584,saved_member_occurrences=member_count,unique_zero_gain_members=unique_count,member_status_counts=dict(types),corrected_overrides=len(changed),corrected_override_by_set=dict(Counter(z['record_set'] for z in changed)),corrected_active_detection=sum(z['effect_detected'] is True for z in results),original_corrected_active_detection=sum(z['effect_detected'] is True for z in results if not z['record_set'].startswith('D08')),raw_two_start_positive=sum(z['two_start_numerical_positive'] is True for z in results),new_optimizer_starts=0,new_fits=0,protocol_sha256=sha(P/'protocol.md'),inventory_rows=len(diag),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
atomic(P/'data/A_NULL_CORRECTION_SUMMARY.json',summary);print(json.dumps(summary),flush=True)
