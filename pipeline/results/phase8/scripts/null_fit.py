from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,datetime,traceback
import numpy as np
from scipy.optimize import minimize,lsq_linear
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3])));P=R/'results/phase8';sys.path.insert(0,str(R/'lib'))
from p3_wenvelope import nat_columns,t95_from_step
from p3_kernels import hantush_block
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DRIVER=sha(__file__)
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def fit(m):
 out=P/'calculations/null_fits'/f"{m['source_phase']}__{m['case_id']}.json"
 if out.exists():
  z=json.loads(out.read_text());assert z['driver_sha256']==DRIVER and z['tf_sha256']==m['tf_sha256'];return z['detected']
 assert sha(m['tf_path'])==m['tf_sha256'] and sha(m['w_path'])==m['w_sha256']
 with np.load(m['tf_path'],allow_pickle=False) as f:y=np.asarray(f['head_context'],float);rain=np.asarray(f['rain'],float)
 yc=y-y.mean();eb=10*np.ptp(y);lb=np.array([-eb,-np.inf,0,-eb]);ub=np.array([eb,np.inf,np.inf,eb]);bounds=np.array([[.5,5],[0,2],[np.log10(5),np.log10(500)],[0,3.5],[m['b_floor'],np.log10(25)]])
 starts=[np.clip(m['x_start'],bounds[:,0],bounds[:,1]),np.clip([2,1,np.log10(60),1,-3],bounds[:,0],bounds[:,1])];results=[]
 for j,x0 in enumerate(starts):
  best=None;evals=0;invalid=0;errors=0
  def objective(x):
   nonlocal best,evals,invalid,errors
   evals+=1
   try:
    _,S=hantush_block([10**x[3]],[10**x[4]],1054);t95=float(t95_from_step(S)[0])
    if not np.isfinite(t95) or t95>1000:invalid+=1;return 1e12
    nc=nat_columns(np.array(x[:3])[None,:],rain,float(rain[:1024].mean()))[0]
    X=np.column_stack([-(1-S[0,1:1025]),np.ones(1024),nc]);d=np.linalg.norm(X,axis=0);d[d==0]=1
    ls=lsq_linear(X/d,yc,bounds=(lb*d,ub*d),method='bvls');coef=np.clip(ls.x/d,lb,ub);res=yc-X@coef;sse=float(res@res)
    if np.isfinite(sse) and (best is None or sse<best['sse']):best=dict(sse=sse,x=list(map(float,x)),linear_coefficients=list(map(float,coef)),intercept=float(coef[1]+y.mean()),t95=t95,bvls_success=bool(ls.success),bvls_status=int(ls.status))
    return sse
   except Exception:errors+=1;return 1e12
  objective(x0)
  try:
   r=minimize(objective,x0,method='Powell',bounds=bounds,options={'maxfev':400,'xtol':1e-3,'ftol':1e-9})
   rr=dict(success=bool(r.success),status=int(r.status),message=str(r.message),nfev=int(r.nfev),returned_objective=float(r.fun),returned_x=list(map(float,r.x)))
  except Exception as e:rr=dict(success=False,status=None,message=repr(e),nfev=evals-1,returned_objective=None,returned_x=None)
  results.append(dict(start_number=j+1,start_x=list(map(float,x0)),optimizer=rr,evaluations_including_initial=evals,infeasible_evaluations=invalid,error_evaluations=errors,best_feasible=best))
 good=[s['best_feasible'] for s in results if s['best_feasible'] is not None];best=min(good,key=lambda q:q['sse']) if good else None
 z=dict(source_phase=m['source_phase'],case_id=m['case_id'],record_set=m['record_set'],tau=m['tau'],tf_sha256=m['tf_sha256'],w_sha256=m['w_sha256'],protocol_sha256=sha(P/'protocol.md'),driver_sha256=DRIVER,starts=results,best=best,SSE_min=None if best is None else best['sse'],detected=None if best is None else bool(best['sse']>m['tau']),status='finite_attained_numerical_minimum_not_global_proof' if best else 'no_finite_feasible_minimum')
 atomic(out,z);return z['detected']
def main():
 fr=json.loads((P/'PROTOCOL_FREEZE_RECEIPT.json').read_text())
 for p,s in fr['source_pins'].items():assert sha(p)==s,(p,'changed')
 records=[m for m in json.loads((P/'data/record_manifest.json').read_text()) if m['active']];(P/'calculations/null_fits').mkdir(exist_ok=True)
 t=time.time();counts={'true':0,'false':0,'missing':0};done=0
 with ProcessPoolExecutor(max_workers=8) as pool:
  fs={pool.submit(fit,m):m for m in records}
  for f in as_completed(fs):
   v=f.result();done+=1;counts['missing' if v is None else str(v).lower()]+=1
   if done%50==0 or done==len(records):
    z=dict(status='complete' if done==len(records) else 'working',completed=done,total=len(records),elapsed_seconds=time.time()-t,counts=counts,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),driver_sha256=DRIVER)
    atomic(P/'calculations/A_PROGRESS.json',z);print(json.dumps(z),flush=True)
if __name__=='__main__':main()
