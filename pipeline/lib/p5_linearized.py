"""D06 ten-parameter oracle affine uncertainty runtime, no W sampling.

Production: python lib/p5_linearized.py --production --manifest results/phase5/C_CASES.json
Input-only inventory: --dry-inventory --manifest ... (never evaluates outcomes).
Synthetic fixtures live in results/phase5/implementation_C/test_fixtures.py.

Manifest adapter accepts a list or {C:[...]}/{cases:[...]}/{records:[...]};
source_phase (alias phase) and case_id are mandatory. Paths default to the
frozen source phase's cases/{tf_inputs,truth} and wb/W. Explicit tf_input_path,
truth_path,w_path (or source_tf_input/source_truth/source_w) override them.
Hashes tf_sha256/truth_sha256/w_sha256 may be in each record; production
requires either those hashes or matching final-freeze file pins. Source
manifest derived fields are merged by exact (source_phase,case_id). No fuzzy
identity matching, case dropping or automatic source mutation.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[_name]='1'
import argparse,csv,hashlib,json,sys,time
sys.dont_write_bytecode=True
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.stats import f as f_dist
from p3_kernels import hantush_block,natural_kernel,causal_conv
from p5_linearized_support import tangent_support,least_squares,decomposition,CERT_TOL

ROOT=Path(__file__).resolve().parents[1]
P5=ROOT/'results/phase5'
PAIRS=('P1_continue_vs_stop','P2_current_vs_1p5x')
REQUIRED_MODES=('ar_locked','ar_prospective','sse_locked')
ROW_SCHEMA_VERSION='D06_C_AR_v2_3modes'
METRIC_DECISION=P5/'C_METRIC_DECISION.json'
PARAMETERS=('n','log10theta','log10tau','log10a','log10b','A','eta_pump','c0','beta_nat','eta_nat')
NUMERICAL_UNRESOLVED=('support_uncertified','feasibility_uncertified','local_fit_uncertified','derivative_unvalidated','gls_minimum_uncertified','invalid_transfer_budget','zero_information','invalid_residual_moments','derivative_invalid')
CERTIFIED=('certified_interior','certified_constrained','bounded_by_family_only','zero_contrast','no_question')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()


def jsonable(x):
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 if isinstance(x,np.ndarray):return jsonable(x.tolist())
 if isinstance(x,np.generic):return jsonable(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,Path):return str(x)
 return x


def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(jsonable(x),indent=2,allow_nan=False));tmp.replace(p)


@lru_cache(maxsize=256)
def _kernels(n,lt,lu,la,lb,length):
 B,S=hantush_block([10**la],[10**lb],length+30)
 K,Kc=natural_kernel([n],[10**lt],[10**lu],length)
 # Cached arrays are never returned to callers for mutation.
 for a in (B,S,K,Kc):a.flags.writeable=False
 return B,S,K,Kc


def family_value(v,ti):
 """Exact source daily affine family, all ten free declared parameters."""
 v=np.asarray(v,float);n=len(ti['pumping_context'])
 B,S,K,Kc=_kernels(*map(float,v[:5]),n)
 Q=np.asarray(ti['pumping_context'],float);P=np.asarray(ti['rain'],float)[:n]
 tail=1-S[0,1:n+1]
 pump=causal_conv(Q,B)[0]+Q[0]*tail
 natural=causal_conv(P,K)[0]+P.mean()*(1-Kc[0,1:n+1])
 decay=np.exp(-np.arange(1,n+1)/10**v[2])
 h=v[7]+v[8]*natural+v[9]*decay-v[5]*pump-v[6]*tail
 E={p:-v[5]*causal_conv(np.asarray(ti['future_Q'][p+'_a'])-np.asarray(ti['future_Q'][p+'_b']),B)[0,:30] for p in PAIRS}
 return h,E


def jacobians(v,ti,step=1e-4):
 v=np.asarray(v,float);h,E=family_value(v,ti);J=np.zeros((len(h),10));JE={p:np.zeros((30,10)) for p in PAIRS}
 audits=[]
 for j in range(5):
  d=np.zeros(10);d[j]=step
  hp,Ep=family_value(v+d,ti);hm,Em=family_value(v-d,ti)
  hph,Eph=family_value(v+d/2,ti);hmh,Emh=family_value(v-d/2,ti)
  full=(hp-hm)/(2*step);half=(hph-hmh)/step
  J[:,j]=full
  rel=float(np.linalg.norm(half-full)/max(np.linalg.norm(half),1e-7))
  audit=dict(parameter=PARAMETERS[j],step=step,maxabs=float(np.max(np.abs(half-full))),relative_norm=rel,validated=bool(rel<=1e-3 or np.max(np.abs(half-full))<=1e-7))
  for p in PAIRS:
   JE[p][:,j]=(Ep[p]-Em[p])/(2*step)
   er=float(np.linalg.norm((Eph[p]-Emh[p])/step-JE[p][:,j])/max(np.linalg.norm(JE[p][:,j]),1e-7))
   audit[p+'_relative']=er
   audit['validated'] &= er<=1e-3 or np.max(np.abs((Eph[p]-Emh[p])/step-JE[p][:,j]))<=1e-7
  audits.append(audit)
 # Exact source linear columns, no finite difference cancellation.
 B,S,K,Kc=_kernels(*map(float,v[:5]),len(h));Q=np.asarray(ti['pumping_context']);P=np.asarray(ti['rain'])[:len(h)]
 J[:,5]=-(causal_conv(Q,B)[0]+Q[0]*(1-S[0,1:len(h)+1]));J[:,6]=-(1-S[0,1:len(h)+1]);J[:,7]=1
 J[:,8]=causal_conv(P,K)[0]+P.mean()*(1-Kc[0,1:len(h)+1]);J[:,9]=np.exp(-np.arange(1,len(h)+1)/10**v[2])
 for p in PAIRS:
  JE[p][:,5]=-causal_conv(np.asarray(ti['future_Q'][p+'_a'])-np.asarray(ti['future_Q'][p+'_b']),B)[0,:30]
  JE[p][:,[0,1,2,6,7,8,9]]=0.
 # P2 reuse is permitted only after checking the actual future forcings.
 dq=[np.asarray(ti['future_Q'][p+'_a'])-np.asarray(ti['future_Q'][p+'_b']) for p in PAIRS]
 proportional=bool(np.array_equal(dq[1],-.5*dq[0]))
 discrepancy=float(np.max(np.abs(JE[PAIRS[1]]+.5*JE[PAIRS[0]])))
 return J,JE,dict(columns=audits,all_valid=all(a['validated'] for a in audits),pair_proportional=proportional,pair_derivative_discrepancy=discrepancy)


def ar1_whiten(values,times=None,phi=.8,sigma=.02):
 """Stationary gap innovations = inverse Cholesky of the kept covariance."""
 X=np.asarray(values,float);times=np.arange(len(X)) if times is None else np.asarray(times,int)
 if len(times)!=len(X) or np.any(np.diff(times)<=0) or not -1<phi<1 or sigma<=0:raise ValueError('Invalid stationary AR1 inputs')
 Y=np.empty_like(X)
 if not len(X):return Y
 Y[0]=X[0]/sigma
 a=phi**np.diff(times);den=sigma*np.sqrt(1-a*a)
 if X.ndim==2:Y[1:]=(X[1:]-a[:,None]*X[:-1])/den[:,None]
 else:Y[1:]=(X[1:]-a*X[:-1])/den
 return Y


def fisher_geometry(J,times=None,phi=.8,sigma=.02):
 J=np.asarray(J,float);W=ar1_whiten(J,times,phi,sigma)
 dec=decomposition(W);sc=dec['scale'];s=dec['s'][:dec['rank']]
 C=sc[:,None]*dec['V']/s[None,:]
 K=W@C;D=(J@C).T@(J@C);D=(D+D.T)/2
 U=np.linalg.qr(J@C,mode='reduced')[0] if dec['rank'] else np.empty((len(J),0))
 lam=float(np.linalg.eigvalsh(D)[-1]) if dec['rank'] else None
 return dict(lambda_max=lam,lambda_max_m2=lam,U=U,rank_shared=dec['rank'],Euclidean_orthogonality_error=float(np.linalg.norm(U.T@U-np.eye(dec['rank']))),H=W.T@W,C=C,K=K,D=D,null_basis=sc[:,None]*dec['N'],rank=dec['rank'],singular_values=dec['s'],parameter_scaling=sc,whiten_orthogonality_error=float(np.linalg.norm(K.T@K-np.eye(dec['rank']))),conversion='D=C.T G C; lambda_max supplies selected inscribed increment; SSE benchmark remains a separate region')



def expected_residual_moments(U,sigma=.02,phi=.8):
 """Source-author O(n r²) moments of (I-UU.T) Sigma (I-UU.T)."""
 U=np.asarray(U,float);n,r=U.shape
 if n<=20 or sigma<=0 or not -1<phi<1:raise ValueError('Invalid expected source tolerance inputs')
 if np.linalg.norm(U.T@U-np.eye(r))>1e-8:raise ValueError('U must be the retained orthonormal Euclidean tangent basis')
 forward=np.empty_like(U);backward=np.empty_like(U);carry=np.zeros(r)
 for i in range(n):carry=U[i]+phi*carry;forward[i]=carry
 carry=np.zeros(r)
 for i in range(n-1,-1,-1):carry=U[i]+phi*carry;backward[i]=carry
 S=sigma**2*(forward+backward-U);M=U.T@S;M=(M+M.T)/2;UM=U@M
 diag=sigma**2-2*np.sum(U*S,axis=1)+np.sum(UM*U,axis=1)
 lag=sigma**2*phi-np.sum(U[:-1]*S[1:],axis=1)-np.sum(S[:-1]*U[1:],axis=1)+np.sum(UM[:-1]*U[1:],axis=1)
 trace=float(n*sigma**2-np.trace(M));denom2=float((trace-diag[-1])*(trace-diag[0]))
 out=dict(trace_R=trace,trace_R_m2=trace,diag=diag,lag=lag,lag_sum=float(lag.sum()),lag_denominator_squared=denom2,trace_diagonal_error=float(abs(trace-diag.sum())),phi=phi,sigma=sigma,p=10,rank=r,algorithm='O(nr2) forward/backward stationary AR recurrence; Euclidean projection',dependency='J, declared stationary Sigma, expected tangent moments only')
 if trace<=1e-12*n*sigma**2 or denom2<=0:return dict(out,status='undefined_residual_moments')
 rho=float(lag.sum()/np.sqrt(denom2))
 if not np.isfinite(rho) or abs(rho)>1+1e-10:return dict(out,status='invalid_residual_moments')
 rho=float(np.clip(rho,-1,1));neff=float(n if rho==-1 else np.clip(n*(1-rho)/(1+rho),20,n))
 gamma=float(10*f_dist.ppf(.95,10,neff-10)/(neff-10))
 return dict(out,status='ok',r_pred=rho,n_eff_pred=neff,gamma_pred=gamma,Delta_P=gamma*trace,Delta_P_m2=gamma*trace)


def prospective_ar_radius(geometry,expected_receipt):
 lam=geometry.get('lambda_max')
 if not geometry['rank'] or lam is None or not np.isfinite(lam) or lam<=0:return dict(status='zero_information',c_P=None)
 if expected_receipt['status']!='ok':return dict(status=expected_receipt['status'],c_P=None)
 delta=float(expected_receipt['gamma_pred']*expected_receipt['trace_R'])
 if not np.isfinite(delta) or delta<0:return dict(status='invalid_transfer_budget',c_P=None)
 return dict(status='ok',c_P=delta/lam,Delta_P_m2=delta,lambda_max_m2=lam,dependency='J, declared stationary Sigma, expected tangent moments only',bounds_ownership='actual observed-head-range bounds passed separately; no pumping-log-only claim')


def locked_ar_radius(geometry,tolerance):
 lam=geometry.get('lambda_max');delta=float(tolerance['tau']-tolerance['sse_min'])
 if not np.isfinite(delta) or delta<0:return dict(status='invalid_transfer_budget',c_L=None,Delta_L_m2=delta)
 if not geometry['rank'] or lam is None or not np.isfinite(lam) or lam<=0:return dict(status='zero_information',c_L=None,Delta_L_m2=delta)
 return dict(status='ok',c_L=delta/lam,Delta_L_m2=delta,lambda_max_m2=lam,transfer_rule='inscribed maximal deterministic scalar; not exact legacy SSE; mean-SSE trace alternative remains dissent')


def bounded_gls_minimum(Jw,ew,lower_delta,upper_delta,linear=None):
 fit=least_squares(Jw,ew,lower_delta,upper_delta,linear)
 primal=fit.get('primal_upper');dual=fit.get('dual_lower')
 # Pad only the numerical dual lower, retaining its actual value and gap.
 lower=None if dual is None else min(dual,fit['sse'])-1e-12*(1+abs(dual))
 return dict(fit,Q_min_primal=primal,Q_min_dual_lower=lower,Q_min_gap=None if lower is None or primal is None else primal-lower,metric='AR1_precision',minimum_enclosure='numerical dual lower and feasible primal upper, not solver Boolean')


def inventory_receipt(case_count=6270):
 return dict(cases=case_count,pairs=2,leads=30,required_modes=REQUIRED_MODES,curve_rows_per_mode=case_count*60,primary_rows_per_mode=case_count*4,required_curve_total=case_count*60*len(REQUIRED_MODES),required_primary_total=case_count*4*len(REQUIRED_MODES),expected_zero_question_cases=1836)


def execution_fingerprint(digest,manifest,freeze):
 import scipy,p5_contracts
 closure=p5_contracts.local_import_closure(tuple(p5_contracts.RUNTIME_ENTRIES)+('p5_linearized',))
 return dict(protocol_sha256=digest,manifest_sha256=sha(manifest),metric_decision_sha256=sha(METRIC_DECISION),addendum_sha256=sha(P5/'C_AR_DESIGN_ADDENDUM.json'),required_modes=list(REQUIRED_MODES),row_schema_version=ROW_SCHEMA_VERSION,family_versions=['D03_D05_original_b1e-4','D06_new_r20_b1e-6'],driver_sha256=sha(__file__),support_sha256=sha(Path(__file__).with_name('p5_linearized_support.py')),transitive_module_hashes={p:sha(ROOT/p) for p in closure},runtime_versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),freeze_pins_sha256=hashlib.sha256(json.dumps(freeze.get('hashes',{}),sort_keys=True).encode()).hexdigest())


def validate_resume(saved,fingerprint,input_hashes=None):
 if saved.get('fingerprint')!=fingerprint:raise PermissionError('Resume mode/schema/metric/protocol/closure/runtime fingerprint mismatch; old SSE rows cannot be relabelled as AR')
 if input_hashes is not None and saved.get('input_hashes')!=input_hashes:raise PermissionError('Resume input receipts differ')
 return True


def local_tolerance(J,e,lower,upper,linear=None):
 fit=least_squares(J,e,lower,upper,linear);r=fit['residual']
 if len(r)<=20:raise ValueError('Tolerance needs n>20 with p=10')
 if np.std(r[:-1])==0 or np.std(r[1:])==0:
  return dict(tau=None,sse_min=fit['sse'],p=10,status='undefined_residual_autocorrelation',fit=fit)
 r1=float(np.corrcoef(r[:-1],r[1:])[0,1]);neff=float(np.clip(len(r)*(1-r1)/(1+r1),20,len(r)))
 gamma=float(10/(neff-10)*f_dist.ppf(.95,10,neff-10))
 return dict(tau=fit['sse']*(1+gamma),sse_min=fit['sse'],r1=r1,n_eff=neff,excess=gamma,p=10,status='certified' if fit['certified'] else 'local_fit_uncertified',fit=fit,dependency='observed head and oracle affine family only; no saved W tolerance or endpoints')


def strict_sign(lo,hi):
 if lo is None or hi is None:return None
 return 1 if lo>0 else (-1 if hi<0 else 0)



def certified_sign(s):
 sign=strict_sign(s['inf'],s['sup'])
 if s['status'] not in CERTIFIED:return None,True
 if s['status'] in ('zero_contrast','no_question'):return sign,False
 roundoff=10*np.finfo(float).eps*(1+max(abs(s['inf']),abs(s['sup'])))
 dl=s.get('lower_receipt',{}).get('dual_gap') or 0.
 du=s.get('upper_receipt',{}).get('dual_gap') or 0.
 lo=s['inf']-dl-roundoff;hi=s['sup']+du+roundoff
 uncertain=bool((s['inf']>0 and lo<=0) or (s['sup']<0 and hi>=0))
 return (None if uncertain else sign),uncertain

def true_vector(theta,head_mean=0.):
 return np.array([theta['n'],np.log10(theta['theta']),np.log10(theta['tau']),np.log10(theta['a']),np.log10(theta['b']),theta['A'],theta['eta_pump_true'],-head_mean,theta['nat_gain'],theta.get('eta_nat_true',0.)],float)


def family_bounds(ti,new=False):
 y=np.asarray(ti['head_context']);q=np.asarray(ti['pumping_context']);pos=q[q>0];span=float(np.ptp(y))
 if not len(pos) or span<=0:raise ValueError('Source family requires positive context pumping and nonconstant head')
 eb=10*span;amax=eb/float(np.median(pos))
 return (np.array([.5,0.,np.log10(5),0.,-6. if new else -4.,0.,-eb,-np.inf,0.,-eb]),np.array([5.,2.,np.log10(500),3.5,np.log10(25),amax,eb,np.inf,np.inf,eb]))


def _t95(v):
 _,S=hantush_block([10**v[3]],[10**v[4]],1054)
 hit=np.flatnonzero(S[0]>=.95)
 if not len(hit):return float('inf')
 k=int(hit[0]);return float(k-1+(.95-S[0,k-1])/(S[0,k]-S[0,k-1]))


def memory_tangent(v):
 t=_t95(v);grad=np.zeros(10)
 if not np.isfinite(t):raise ValueError('Truth exceeds source finite memory')
 for j in (3,4):
  d=np.zeros(10);d[j]=1e-4;grad[j]=(_t95(v+d)-_t95(v-d))/2e-4
 return (grad[None,:],np.array([1000.-t])),dict(t95=t,gradient=grad,slack=1000.-t)


def representative_keys():
 layers=('confined_T50','unconfined_T50');out=[]
 for l in layers:out.append(('phase2',f'r00_{l}_rho1_N6'))
 for l in layers:
  for rec in ('recent','old'):out.append(('phase3',f'd04A_r00_{l}_rho1_N18_rec{rec}'))
 for phase,prefix in (('phase4','d05B'),('phase5','d06A')):
  for form,month in (('water_curtain',1),('paddy_irrigation',7),('domestic_continuous',12)):
   for l in layers:out.append((phase,f'{prefix}_{form}_m{month:02d}_r00_{l}'))
 assert len(out)==18
 return out


def zero_runs(q):
 z=np.asarray(q)==0;change=np.diff(np.r_[False,z,False].astype(int));return list(zip(np.where(change==1)[0],np.where(change==-1)[0]))


def mask_windows(ti,derived,t95):
 """Input-only deterministic windows. No observed heads or W outcomes read."""
 q=np.asarray(ti['pumping_context']);n=len(q);dates=np.asarray(ti['dates'])[:n].astype('datetime64[D]');length=int(np.ceil(t95))
 form=derived.get('calendar_type') or derived.get('form') or ti.get('meta',{}).get('calendar_form')
 windows={}
 def add(kind,start=None,end=None,status=None):
  if status is None:status='present' if start is not None and 0<=start<end<=n else 'window_absent'
  windows[kind]=dict(status=status,start=None if start is None else int(start),end=None if end is None else int(end),dates=None if status!='present' else [str(dates[start]),str(dates[end-1])],definition='half-open observation indices; forcing/state arrays untouched')
 if not form:
  start=derived.get('rest_start_day');end=derived.get('layer_rest_end_day')
  if start is None and derived.get('R_days') is not None and end is not None:start=int(end)-int(derived['R_days'])
  add('pause',start,end)
  # The prescribed controlled OFF/ON burst lives after rest end. For D04 old
  # recency the final controlled boundary is supplied in the source manifest.
  endburst=derived.get('burst_end_boundary',derived.get('last_switch_boundary'))
  events=derived.get('controlled_event_indices') or derived.get('pulse_indices')
  if events:startburst=min(events)
  elif derived.get('burst_start_day') is not None:startburst=int(derived['burst_start_day'])+1
  else:
   N=int(derived.get('N_nominal') or 0)
   startburst=None if end is None or N==0 else int(end)+1
   if endburst is None and end is not None and N:endburst=int(end)+N
  add('transition',startburst,None if endburst is None else int(endburst)+1)
  add('season_start',status='not_applicable')
 else:
  runs=[(int(s),int(e)) for s,e in zero_runs(q) if e<n and e-s>=length and q[e]>0]
  add('pause',*(max(runs,key=lambda x:x[1]) if runs else (None,None)))
  monthday=np.array([str(d)[5:] for d in dates])
  if form in ('water_curtain','paddy_irrigation'):
   target='11-01' if form=='water_curtain' else '05-01'
   centers=[i for i,x in enumerate(monthday) if x==target]
   center=max(centers) if centers else None;add('season_start',None if center is None else center-length,None if center is None else center+length)
   months=np.array([int(str(d)[5:7]) for d in dates]);active=np.isin(months,(11,12,1,2,3) if form=='water_curtain' else (5,6,7,8))
   one=[(int(s),int(e+1)) for s,e in zero_runs(q) if e-s==1 and s>0 and e<n and q[s-1]>0 and q[e]>0 and np.all(active[s-1:e+1]) and all(str(dates[j])[5:]!=target for j in range(s-1,e+1))]
   add('transition',*(max(one,key=lambda x:x[1]) if one else (None,None)))
  else:
   add('season_start',status='not_applicable')
   maint=[(int(s),int(e+1)) for s,e in zero_runs(q) if s>0 and e<n and q[s-1]>0 and q[e]>0]
   add('transition',*(max(maint,key=lambda x:x[1]) if maint else (None,None)))
 return windows


def mask_information(J,JE,v,lower,upper,memory,excess_budget,times=None,*,c_L=None,E=None,dual=False,geometry_full=None):
 if dual:
  return dual_mask_information(J,JE,{p:np.zeros(30) for p in PAIRS} if E is None else E,v,lower,upper,memory,excess_budget,c_L,times)
 times=np.arange(len(J)) if times is None else np.asarray(times,int)
 f=fisher_geometry(J,times);prep=least_squares(J,np.zeros(len(J)),lower-v,upper-v,memory)
 intervals=solve_curves(J,np.zeros(len(J)),excess_budget,JE,lower-v,upper-v,memory,{p:np.zeros(30) for p in PAIRS},prep)
 return intervals,f


def solve_curves(J,e,tau,JE,lo,hi,linear,E,prepared=None):
 out={};primary=PAIRS[0];warm=None;reuse=bool(np.allclose(JE[PAIRS[1]],-.5*JE[primary],rtol=1e-12,atol=0.) and np.allclose(E[PAIRS[1]],-.5*E[primary],rtol=1e-12,atol=0.))
 if prepared is None:prepared=least_squares(J,e,lo,hi,linear)
 for p in PAIRS:
  if p==PAIRS[1] and reuse:
   out[p]=[]
   for s in out[primary]:
    z=dict(s);z.update(pair_reuse='exact P2=-0.5 P1; verified actual dQ, JE and E',inf=None if s['sup'] is None else -.5*s['sup'],sup=None if s['inf'] is None else -.5*s['inf'],width=None if s['width'] is None else .5*s['width'])
    if 'upper_receipt' in s:
     for dest,src in (('lower_receipt','upper_receipt'),('upper_receipt','lower_receipt')):
      r=dict(s[src]);r['value']=z['inf'] if dest=='lower_receipt' else z['sup']
      if r.get('dual_gap') is not None:r['dual_gap']*=.5
      r['pair_reuse']='exact proportional support reversal';z[dest]=r
    out[p].append(z)
   continue
  seq=[];warm=None
  for k in range(30):
   s=tangent_support(J,e,tau,JE[p][k],lo,hi,linear,E0=E[p][k],warm=warm,prepared=prepared)
   if 'lower_receipt' in s:warm=(s['lower_receipt']['x'],s['upper_receipt']['x'])
   seq.append(s)
  out[p]=seq
 return out


def require_frozen(phase5=P5):
 """Require final complete freeze, authoritative contract gate and own closure pins."""
 phase5=Path(phase5);f=json.loads((phase5/'protocol_freeze.json').read_text())
 if f.get('status')!='complete_protocol_frozen':raise PermissionError('Complete HIGH final protocol freeze required')
 protocol=phase5/'protocol.md';digest=sha(protocol)
 if f.get('protocol_sha256')!=digest or (phase5/'protocol.sha256').read_text().split()[0]!=digest:raise PermissionError('Protocol hash mismatch')
 hashes=f.get('hashes')
 if not isinstance(hashes,dict) or not hashes:raise PermissionError('Complete transitive closure pins required')
 for name,want in hashes.items():
  p=Path(name);p=p if p.is_absolute() else ROOT/p
  if len(want)!=64 or not p.is_file() or sha(p)!=want:raise PermissionError('Frozen source changed/missing: '+name)
 for file in ('lib/p5_linearized.py','lib/p5_linearized_support.py','lib/p3_kernels.py'):
  if hashes.get(file,hashes.get(str(ROOT/file)))!=sha(ROOT/file):raise PermissionError('Own implemented closure is not frozen: '+file)
 contract=phase5/'FROZEN_EXECUTION_CONTRACT.json'
 if not contract.is_file():raise PermissionError('Final execution contract missing')
 # Parent's canonical gate checks all additional role, inventory, closure,
 # immutable-old and numerical-runtime requirements; do not weaken it.
 import p5_contracts
 p5_contracts.require_frozen(entries=tuple(p5_contracts.RUNTIME_ENTRIES)+('p5_linearized',))
 return digest,f


def manifest_adapter(path,full=True):
 data=json.loads(Path(path).read_text())
 if isinstance(data,dict):
  for key in ('C','cases','records'):
   if key in data:data=data[key];break
 if not isinstance(data,list):raise ValueError('Manifest must be a list or C/cases/records list')
 result=[];seen=set();cache={}
 for rec in data:
  rec=dict(rec);phase=rec.get('source_phase',rec.get('phase'));cid=rec['case_id'];key=(phase,cid)
  if phase not in ('phase2','phase3','phase4','phase5') or key in seen:raise ValueError('Invalid/duplicate case identity '+str(key))
  seen.add(key)
  if phase not in cache:
   mp=ROOT/f'results/{phase}/cases/derived_manifest.json'
   cache[phase]={r['case_id']:r for r in json.loads(mp.read_text())} if mp.exists() else {}
  merged=dict(cache[phase].get(cid,{}));merged.update(rec);merged['source_phase']=phase
  base=ROOT/f'results/{phase}'
  for field,aliases,default in (('tf_input_path',('source_tf_input',),base/f'cases/tf_inputs/{cid}.npz'),('truth_path',('source_truth',),base/f'cases/truth/{cid}.npz'),('w_path',('source_w','W_path'),base/f'wb/W/{cid}.json')):
   val=merged.get(field)
   if val is None:val=next((merged[a] for a in aliases if a in merged),default)
   p=Path(val);merged[field]=str(p if p.is_absolute() else ROOT/p)
  if 'W_sha256' in merged:merged['w_sha256']=merged['W_sha256']
  result.append(merged)
 if full:
  expected={'phase2':540,'phase3':690,'phase4':2880,'phase5':2160}
  if Counter(r['source_phase'] for r in result)!=Counter(expected):raise ValueError('C full6270 universe mismatch')
  missing=set(representative_keys())-seen
  if missing:raise ValueError('Prespecified representative absent '+str(missing))
 return result


def load_record(rec,freeze=None):
 for field,hkey in (('tf_input_path','tf_sha256'),('truth_path','truth_sha256'),('w_path','w_sha256')):
  p=Path(rec[field]);got=sha(p);want=rec.get(hkey)
  if want is None and freeze is not None:want=freeze['hashes'].get(str(p),freeze['hashes'].get(str(p.relative_to(ROOT))))
  if want is None and freeze is not None and rec['source_phase']=='phase5':
   marker=ROOT/f"results/phase5/wb/markers/{rec['case_id']}.done"
   if field=='w_path' and marker.exists():want=json.loads(marker.read_text()).get('outputs',{}).get(str(p.relative_to(ROOT)))
  if freeze is not None and want is None:raise PermissionError('No frozen/source-generation/WB-marker input receipt for '+str(p))
  if want is not None and got!=want:raise PermissionError('Input receipt changed '+str(p))
 with np.load(rec['tf_input_path'],allow_pickle=False) as z:
  ti=dict(head_context=z['head_context'],rain=z['rain'],pumping_context=z['pumping_context'],dates=z['dates'],meta=json.loads(str(z['meta_json'])),future_Q={k[10:]:z[k] for k in z.files if k.startswith('future_Q__')})
  if str(z['case_id'])!=rec['case_id']:raise ValueError('tf identity mismatch')
 with np.load(rec['truth_path'],allow_pickle=False) as z:
  theta=json.loads(str(z['theta_true_json']));Et={p:z['E_true__'+p] for p in PAIRS}
  if str(z['case_id'])!=rec['case_id']:raise ValueError('truth identity mismatch')
 if len(ti['head_context'])!=1024:raise ValueError('Full frozen1024 context required')
 w=json.loads(Path(rec['w_path']).read_text())
 if freeze is not None:
  source_proto=freeze['protocol_sha256'] if rec['source_phase']=='phase5' else json.loads((ROOT/f"results/{rec['source_phase']}/protocol_freeze.json").read_text())['protocol_sha256']
  if w.get('protocol_sha256')!=source_proto or w.get('tf_input_sha256')!=sha(rec['tf_input_path']):raise PermissionError('W protocol/input lineage mismatch')
 if w['case_id']!=rec['case_id']:raise ValueError('W identity mismatch')
 return ti,theta,Et,w




def validate_family_alias(rec,lower,upper):
 bound=rec.get('family_bounds')
 if bound is None:return
 if isinstance(bound,dict):
  for key,index in (('n',0),('log10_theta',1),('log10_tau',2),('log10_a',3),('log10_b',4)):
   if key in bound and not np.allclose(bound[key],[lower[index],upper[index]],rtol=0,atol=1e-12):raise ValueError('Declared family_bounds alias disagrees with actual source '+key)
  if 't95_max_d' in bound and bound['t95_max_d']!=1000:raise ValueError('Finite memory bound alias differs')
 else:
  a=np.asarray(bound,float)
  if a.shape==(2,10) and not np.allclose(a,np.array([lower,upper]),rtol=0,atol=1e-12):raise ValueError('Actual absolute source bounds differ')


def error_rows(rec,error,modes=REQUIRED_MODES):
 out=[]
 for mode in modes:
  for pair in PAIRS:
   for lead in range(1,31):out.append(dict(source_phase=rec['source_phase'],case_id=rec['case_id'],realization=rec.get('realization'),layer=rec.get('layer_id',rec.get('sid')),distance_m=rec.get('r_m'),group=rec.get('experiment_group'),pair=pair,lead=lead,primary=lead in (10,30),tolerance_mode=mode,metric='SSE' if mode=='sse_locked' else 'AR1_precision',status='execution_error',error=str(error),W_lin=None,W_sampled=None,inf=None,sup=None,sign_linear=None,sign_sampled=None,sign_agrees=None,no_question=rec.get('no_active_contrast',False),linearization_failure=False,endpoint_numerical_uncertainty=True,row_schema_version=ROW_SCHEMA_VERSION))
 return out


def _array_sha(*arrays):
 h=hashlib.sha256()
 for a in arrays:
  a=np.asarray(a,dtype='<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()


def _failure_curves(status):
 return {p:[dict(status=status,inf=None,sup=None,width=None) for _ in range(30)] for p in PAIRS}


def compute_modes(J,e,JE,E,v,lower,upper,memory,tolerance,modes=REQUIRED_MODES):
 """Shared geometry, separate metric/residual/minimum preparations per mode."""
 geometry=fisher_geometry(J);moments=expected_residual_moments(geometry['U'])
 predicted=prospective_ar_radius(geometry,moments)
 locked=locked_ar_radius(geometry,tolerance)
 Jw=ar1_whiten(J);ew=ar1_whiten(e);ld=lower-v;ud=upper-v
 ols=least_squares(J,e,ld,ud,memory);gls=bounded_gls_minimum(Jw,ew,ld,ud,memory)
 true_ar_fit=least_squares(Jw,np.zeros(len(J)),ld,ud,memory)
 optional=local_tolerance(J,e,ld,ud,memory) if 'prospective_local' in modes else None
 out={};meta={}
 for mode in modes:
  canonical='sse_locked' if mode=='locked_original' else mode
  if canonical=='sse_locked':
   matrix,residual,threshold,prep=J,e,float(tolerance['tau']),ols
   fields=dict(metric='SSE',center_rule='legacy observed-residual affine family; saved nonlinear tolerance',transfer_rule='faithful saved SSE benchmark',metric_threshold=threshold,metric_threshold_lower=threshold,metric_threshold_upper=threshold,metric_threshold_units='m2',tau=threshold,sse_min_local=ols['sse'],status='ok')
  elif canonical=='ar_locked':
   status=locked['status'] if locked['status']!='ok' else ('ok' if gls['certified'] and gls['Q_min_primal'] is not None and gls['Q_min_dual_lower'] is not None else 'gls_minimum_uncertified')
   threshold=None if status!='ok' else gls['Q_min_primal']+locked['c_L']
   matrix,residual,prep=Jw,ew,gls
   fields=dict(metric='AR1_precision',center_rule='bounded affine GLS residual recentering; actual observed residual',transfer_rule='Delta_saved/lambda_max; inscribed maximal convention',metric_threshold=threshold,metric_threshold_lower=None if status!='ok' else gls['Q_min_dual_lower']+locked['c_L'],metric_threshold_upper=threshold,metric_threshold_units='dimensionless',tau=None,sse_min_local=None,status=status)
  elif canonical=='ar_prospective':
   matrix,residual,threshold,prep=Jw,np.zeros(len(J)),predicted.get('c_P'),true_ar_fit
   fields=dict(metric='AR1_precision',center_rule='oracle truth delta0; no actual residual in radius or support',transfer_rule='gamma_pred trace_R/lambda_max; expected Euclidean tangent residual convention',metric_threshold=threshold,metric_threshold_lower=threshold,metric_threshold_upper=threshold,metric_threshold_units='dimensionless',tau=None,sse_min_local=None,status=predicted['status'])
  elif canonical=='prospective_local':
   matrix,residual,threshold,prep=J,e,optional['tau'],ols
   fields=dict(metric='SSE',center_rule='optional observed-residual local SSE helper',transfer_rule='local affine p10/r1 tolerance; not no-sampling AR',metric_threshold=threshold,metric_threshold_lower=threshold,metric_threshold_upper=threshold,metric_threshold_units='m2',tau=threshold,sse_min_local=ols['sse'],status='ok' if optional['status']=='certified' else optional['status'])
  else:raise ValueError('Unknown explicit mode '+mode)
  curves=_failure_curves(fields['status']) if fields['status']!='ok' else solve_curves(matrix,residual,threshold,JE,ld,ud,memory,E,prep)
  # The exact ar_locked threshold lies in a numerical min enclosure. Supports
  # at upper threshold enclose it; retain lower-threshold supports when the
  # minimum gap is material, with no Boolean-only exact-min claim.
  if canonical=='ar_locked' and fields['status']=='ok':
   inner=None;gap=gls['Q_min_gap']
   if gap>1e-12*(1+abs(threshold)):
    inner=solve_curves(matrix,residual,fields['metric_threshold_lower'],JE,ld,ud,memory,E,prep)
   for pair in PAIRS:
    for k,c in enumerate(curves[pair]):
     c['minimum_threshold_uncertainty']=gap;c['threshold_support_rule']='upper min enclosure yields conservative region; lower supports bracket when material'
     if inner is not None:c['inner_threshold_support']={j:inner[pair][k].get(j) for j in ('inf','sup','width','status')}
  out[mode]=curves;meta[mode]=fields
 return out,meta,dict(geometry=geometry,expected_moments=moments,prospective_radius=predicted,locked_radius=locked,ols_minimum=ols,gls_minimum=gls,optional_local_tolerance=optional)


def dual_mask_information(J,JE,E,v,lower,upper,memory,Delta_L,c_L,times=None):
 """Two fixed full-budget masks; never recompute lambda or a masked radius."""
 times=np.arange(len(J)) if times is None else np.asarray(times,int);ld=lower-v;ud=upper-v;zero=np.zeros(len(J));out={}
 for metric,matrix,budget in (('ar_information',ar1_whiten(J,times),c_L),('sse_information',J,Delta_L)):
  if budget is None or not np.isfinite(budget) or budget<0:out[metric]=_failure_curves('invalid_transfer_budget');continue
  prep=least_squares(matrix,zero,ld,ud,memory);out[metric]=solve_curves(matrix,zero,budget,JE,ld,ud,memory,E,prep)
 return out


def run_case(rec,freeze,modes=REQUIRED_MODES):
 ti,theta,Et,w=load_record(rec,freeze);mean=float(np.mean(ti['head_context']));v=true_vector(theta,mean)
 y=np.asarray(ti['head_context'])-mean;h,E=family_value(v,ti);J,JE,ja=jacobians(v,ti)
 lower,upper=family_bounds(ti,rec['source_phase']=='phase5');validate_family_alias(rec,lower,upper);mem,ma=memory_tangent(v);e=y-h;tol=w['tolerance']
 allcurves,modeinfo,shared=compute_modes(J,e,JE,E,v,lower,upper,mem,tol,modes)
 geometry=shared['geometry'];locked=shared['locked_radius'];predicted=shared['prospective_radius'];moments=shared['expected_moments'];gls=shared['gls_minimum'];ols=shared['ols_minimum']
 bounds_hash=_array_sha(lower,upper,mem[0],mem[1]);rows=[]
 for mode,curves in allcurves.items():
  fields=modeinfo[mode];prospective=mode=='ar_prospective'
  for pair in PAIRS:
   env=w.get('envelope',{}).get(pair)
   no_question=not np.any(ti['future_Q'][pair+'_a']-ti['future_Q'][pair+'_b'])
   for k,support in enumerate(curves[pair]):
    if no_question:
     support=dict(support,status='no_question',inf=0.,sup=0.,width=0.,calculation_status=support['status']);curves[pair][k]=support
    slo=None if env is None else float(env['inf'][k]);shi=None if env is None else float(env['sup'][k]);sw=None if slo is None else shi-slo
    status=support['status'] if ja['all_valid'] or no_question else 'derivative_invalid'
    sign,uncertain=certified_sign(support) if status in CERTIFIED else (None,True);sampled=strict_sign(slo,shi)
    cone=bool(support['sup'] is not None and support['inf'] is not None and (support['sup']>CERT_TOL if pair==PAIRS[0] else support['inf']<-CERT_TOL))
    active=sorted(set(support.get('lower_receipt',{}).get('active',[])+support.get('upper_receipt',{}).get('active',[])))
    rows.append(dict(source_phase=rec['source_phase'],case_id=rec['case_id'],realization=rec.get('realization'),layer=rec.get('layer_id',rec.get('sid')),group=rec.get('experiment_group'),distance_m=rec.get('r_m'),pair=pair,lead=k+1,primary=k+1 in (10,30),tolerance_mode=mode,metric=fields['metric'],center_rule=fields['center_rule'],transfer_rule=fields['transfer_rule'],transfer_dissent='mean-SSE trace matching is defensible; no guaranteed legacy residual width ordering',lambda_max_m2=geometry['lambda_max'],Delta_L_m2=None if prospective else locked.get('Delta_L_m2'),c_L=None if prospective else locked.get('c_L'),c_P=predicted.get('c_P') if prospective else None,Q_min_B=None if prospective else gls['Q_min_primal'],Q_min_certified=None if prospective else gls['certified'],Q_min_dual_lower=None if prospective else gls['Q_min_dual_lower'],Q_min_gap=None if prospective else gls['Q_min_gap'],SSE_min_saved_m2=None if prospective else tol['sse_min'],tau_saved_m2=None if prospective else tol['tau'],SSE_min_affine_m2=None if prospective else ols['sse'],trace_R_m2=moments['trace_R'],r_pred=moments.get('r_pred'),n_eff_pred=moments.get('n_eff_pred'),gamma_pred=moments.get('gamma_pred'),Delta_P_m2=moments.get('Delta_P_m2'),rank_shared=geometry['rank'],rank_solver=support.get('rank'),rank_J=ols['rank'],rank_H=geometry['rank'],bounds_sha256=bounds_hash,rank_disagreement=ols['rank']!=geometry['rank'],metric_threshold=fields['metric_threshold'],metric_threshold_lower=fields['metric_threshold_lower'],metric_threshold_upper=fields['metric_threshold_upper'],metric_threshold_units=fields['metric_threshold_units'],tau=fields['tau'],sse_min_local=fields['sse_min_local'],inf=support['inf'],sup=support['sup'],W_lin=support['width'],W_sampled=sw,sampled_inf=slo,sampled_sup=shi,E_true=float(Et[pair][k]),sign_linear=sign,sign_sampled=sampled,sign_agrees=None if sign is None or sampled is None else sign==sampled,status=status,support_status=support['status'],linearization_failure=cone,no_question=bool(no_question),endpoint_numerical_uncertainty=uncertain,derivative_valid=ja['all_valid'],active_bound_labels=active,oracle_true_parameters=True,head_conditional_bounds=True,family_b_lower=-6 if rec['source_phase']=='phase5' else -4,source_W_status=w.get('W_status'),source_W_inner_envelope=True,row_schema_version=ROW_SCHEMA_VERSION))
 receipts=dict(identity={k:rec.get(k) for k in ('source_phase','case_id','realization','sid','layer_id','experiment_group','r_m')},oracle_true_parameters=True,parameters=dict(zip(PARAMETERS,v)),head_centering=dict(convention='y=h_obs-headmean, c0=-headmean; raw c0=0 equivalent e/J',mean=mean),bounds=[lower,upper],delta_bounds=[lower-v,upper-v],bounds_sha256=bounds_hash,jacobian_audit=ja,fisher=compact_fisher(geometry),memory=ma,locked_tolerance=tol,mode_fields=modeinfo,shared={key:(compact_fit(value) if key.endswith('minimum') else value) for key,value in shared.items() if key!='geometry'},inputs={f:dict(path=rec[f],sha256=sha(rec[f])) for f in ('tf_input_path','truth_path','w_path')},truth_E_match_maxabs=max(float(np.max(np.abs(E[p]-Et[p]))) for p in PAIRS),source_W_flags=w.get('flags',[]),source_W_status=w.get('W_status'),support_curves=compact_curves(allcurves),interpretation='ar_locked residual GLS vs sse_locked residual SSE centers differ; ar_prospective truth-centered, conditional on actual head bounds and oracle params',strongest_dissent='mean-SSE trace transfer defensible; selected inscribed transfer is a convention, not uniquely physical or extra information')
 masks=[];windows=None
 if (rec['source_phase'],rec['case_id']) in set(representative_keys()):
  delta=locked.get('Delta_L_m2');c=locked.get('c_L');windows=mask_windows(ti,rec,ma['t95']);base=dual_mask_information(J,JE,E,v,lower,upper,mem,delta,c)
  receipts['information_baselines']=compact_curves(base);receipts['masks']={}
  fulltimes=np.arange(len(J))
  for kind,window in windows.items():
   keep=np.ones(len(J),bool)
   if window['status']=='present':keep[window['start']:window['end']]=False
   times=np.flatnonzero(keep)
   curved=dual_mask_information(J[keep],JE,E,v,lower,upper,mem,delta,c,times) if window['status']=='present' else None
   if curved is not None:receipts['masks'][kind]=dict(window=window,keep_times=times,curves=compact_curves(curved),gap_precision='inverse stationary kept covariance; all original states/forcings retained')
   for metric in ('ar_information','sse_information'):
    for pair in PAIRS:
     for k,a in enumerate(base[metric][pair]):
      b=curved[metric][pair][k] if curved is not None else dict(inf=None,sup=None,width=None,status=window['status'])
      wa,wb=a['width'],b['width'];both=wa is not None and wb is not None
      gap=max([float(x.get('dual_gap') or 0) for z in (a,b) for x in (z.get('lower_receipt',{}),z.get('upper_receipt',{}))])
      masks.append(dict(source_phase=rec['source_phase'],case_id=rec['case_id'],pair=pair,lead=k+1,window_kind=kind,window=kind,mask_metric=metric,window_status=window['status'],start=window['start'],end=window['end'],dates=window['dates'],center_rule='oracle truth delta0',budget_source='full-record saved excess and lambda, held fixed',Delta_L_m2=delta,lambda_max_m2=geometry['lambda_max'],c_L=c,bounds_sha256=bounds_hash,full_times_sha256=_array_sha(fulltimes),keep_times_sha256=_array_sha(times),full_inf=a['inf'],full_sup=a['sup'],full_width=wa,masked_inf=b['inf'],masked_sup=b['sup'],masked_width=wb,W_full=wa,W_mask=wb,ratio=wb/wa if both and wa>0 else None,difference=wb-wa if both else None,full_support_status=a['status'],masked_support_status=b['status'],status=b['status'],certificate_gap=gap,monotonicity_audit_flag=bool(both and wb<wa-max(CERT_TOL,gap)),fixed_budget=c if metric=='ar_information' else delta,covariance='stationary AR1 exact gap marginal' if metric=='ar_information' else 'unweighted SSE'))
 return rows,masks,windows,receipts


def compact_fit(fit):return {k:v for k,v in fit.items() if k!='residual'}


def summary_tables(rows,draws=None):
 loggroups=defaultdict(list);signgroups=defaultdict(list)
 for r in rows:
  keys=[('overall',),('lead',r['lead']),('layer',r['layer']),('source',r['source_phase']),('distance',r['distance_m']),('group',r['group'])]
  for key in keys:
   group=(r['tolerance_mode'],r['pair'])+key;signgroups[group].append(r)
   if r['status'] in CERTIFIED and r['W_lin'] is not None and r['W_sampled'] is not None and r['W_lin']>0 and r['W_sampled']>0:loggroups[group].append(r)
 logs=[];signs=[]
 for group,rs in signgroups.items():
  valid=loggroups[group];entry=dict(group=group,inventory=len(rs),positive_certified_pairs=len(valid),excluded_statuses=dict(Counter(r['status'] for r in rs if not (r['status'] in CERTIFIED and r['W_lin'] is not None and r['W_sampled'] is not None and r['W_lin']>0 and r['W_sampled']>0))))
  if len(valid)>2:
   x=np.log([r['W_sampled'] for r in valid]);y=np.log([r['W_lin'] for r in valid]);X=np.column_stack((np.ones(len(x)),x));b=np.linalg.lstsq(X,y,rcond=1e-10)[0];res=y-X@b;sst=float(np.sum((y-y.mean())**2))
   entry.update(intercept=b[0],slope=b[1],R2=None if sst==0 else 1-float(res@res)/sst,log_ratio_mean=float(np.mean(y-x)),log_ratio_sd=float(np.std(y-x)),orientation='x=log sampled W; y=log W_lin')
  if draws is not None and len(valid)>2:entry.update(cluster_loglog(valid,draws))
  logs.append(entry)
  for active in (False,True):
   subset=[r for r in rs if not active or not r['no_question']];validsign=[r for r in subset if r['sign_linear'] is not None and r['sign_sampled'] is not None]
   counts=Counter((int(r['sign_linear']!=0),int(r['sign_sampled']!=0)) for r in validsign)
   directions=[r for r in validsign if r['sign_linear']!=0 and r['sign_sampled']!=0]
   signs.append(dict(group=group,active_only=active,inventory=len(subset),valid=len(validsign),missing_or_uncertified=len(subset)-len(validsign),both_undetermined=counts[0,0],linear_only_determined=counts[1,0],sampled_only_determined=counts[0,1],both_determined=counts[1,1],determination_agreement=None if not validsign else sum((r['sign_linear']!=0)==(r['sign_sampled']!=0) for r in validsign)/len(validsign),same_direction=None if not directions else sum(r['sign_linear']==r['sign_sampled'] for r in directions)/len(directions),affine_cone_failures=sum(r['linearization_failure'] for r in subset)))
   if draws is not None and validsign:signs[-1].update(cluster_sign(validsign,draws))
 return logs,signs



def shared_draws():
 path=ROOT/'results/phase4/bootstrap_draw_matrix.npy';draws=np.load(path,allow_pickle=False)
 if draws.shape!=(999,10) or np.any(draws<0) or np.any(draws.sum(axis=1)!=10):raise ValueError('Frozen999 shared cluster multiplicities malformed')
 # D05 logical digest is pinned by the selected HIGH design and actual source.
 want='7520bcd1cd416b25f22e1e3c7ce6f3905209b837eac4d5a703a038b79f7aa30e'
 if hashlib.sha256(np.asarray(draws,dtype='<i8').tobytes()).hexdigest()!=want:raise PermissionError('Shared logical draw digest differs from D05')
 return draws


def cluster_loglog(rows,draws):
 draws=np.asarray(draws)
 if draws.ndim!=2 or draws.shape[1]!=10 or np.any(draws<0) or np.any(draws.sum(axis=1)!=10):raise ValueError('Ten-cluster multiplicities required')
 stats=np.zeros((10,6))
 for r in rows:
  x=np.log(r['W_sampled']);y=np.log(r['W_lin']);stats[int(r['realization'])]+=np.array([1,x,y,x*x,x*y,y*y])
 total=draws@stats;N,sx,sy,sxx,sxy,syy=total.T
 xx=sxx-sx*sx/np.maximum(N,1);xy=sxy-sx*sy/np.maximum(N,1);yy=syy-sy*sy/np.maximum(N,1)
 ok=(N>2)&(xx>1e-12)&(yy>1e-12)
 slope=xy[ok]/xx[ok];intercept=(sy[ok]-slope*sx[ok])/N[ok];r2=xy[ok]**2/(xx[ok]*yy[ok])
 return dict(slope_ci=None if not len(slope) else np.quantile(slope,[.025,.975]),intercept_ci=None if not len(slope) else np.quantile(intercept,[.025,.975]),R2_ci=None if not len(slope) else np.quantile(r2,[.025,.975]),bootstrap_valid=int(ok.sum()),bootstrap_invalid=int((~ok).sum()),bootstrap='same D05 realization multiplicities; exact OLS sufficient statistics')


def cluster_sign(rows,draws):
 stats=np.zeros((10,5))
 for r in rows:
  a,b=r['sign_linear'],r['sign_sampled'];both=a!=0 and b!=0
  stats[int(r['realization'])]+=np.array([1,(a!=0)==(b!=0),both,both and a==b,r['linearization_failure']])
 total=np.asarray(draws)@stats;ok=total[:,0]>0;both=total[:,2]>0
 return dict(determination_agreement_ci=np.quantile(total[ok,1]/total[ok,0],[.025,.975]) if ok.any() else None,same_direction_ci=np.quantile(total[both,3]/total[both,2],[.025,.975]) if both.any() else None,bootstrap_valid=int(ok.sum()))


def compact_curves(modes):
 # Raw endpoints, active constraints and certificates retained. Shared residual
 # and matrix receipts are stored once, avoiding multi-GB repeated arrays.
 return {mode:{p:[{k:v for k,v in s.items() if k!='feasibility'} for s in seq] for p,seq in curves.items()} for mode,curves in modes.items()}


def compact_fisher(g):return {k:v for k,v in g.items() if k!='K'}


def merge_outputs(out,records,manifest_hash,digest,fingerprint=None):
 out=Path(out);paths=sorted(out.glob('C_linearized_rows_shard*of*.csv'))
 if not paths:raise FileNotFoundError('No C shard rows to merge')
 rows=[];masks=[];windows=[];seen=set()
 def decode(value):
  if value=='':return None
  if value in ('True','False'):return value=='True'
  try:return json.loads(value)
  except (ValueError,TypeError):return value
 for path in paths:
  tag=path.stem.removeprefix('C_linearized_rows_');state=json.loads((out/f'C_STATUS_{tag}.json').read_text())
  fp=state['fingerprint']
  if fingerprint is None:fingerprint=execution_fingerprint(digest,ROOT/'results/phase5/C_CASES.json',json.loads((P5/'protocol_freeze.json').read_text()))
  validate_resume(state,fingerprint)
  with path.open(newline='') as f:
   for raw in csv.DictReader(f):
    r={k:decode(v) for k,v in raw.items()};key=(r['source_phase'],r['case_id'],r['pair'],r['lead'],r['tolerance_mode'])
    if key in seen:raise ValueError('Overlapping shard row '+str(key))
    seen.add(key);rows.append(r)
  mp=out/f'C_masking_rows_{tag}.csv'
  if mp.exists():
   with mp.open(newline='') as f:masks.extend({k:decode(v) for k,v in raw.items()} for raw in csv.DictReader(f))
  windows.extend(json.loads((out/f'C_window_masks_{tag}.json').read_text()))
 expected={(r['source_phase'],r['case_id'],p,k,mode) for r in records for p in PAIRS for k in range(1,31) for mode in REQUIRED_MODES}
 if seen!=expected:raise ValueError(f'Incomplete full6270case/2pair/30lead/3mode merge: missing {len(expected-seen)}, extra {len(seen-expected)}')
 write_csv(out/'C_linearized_rows.csv',rows);write_csv(out/'C_primary_rows.csv',[r for r in rows if r['lead'] in (10,30)])
 write_csv(out/'C_masking_rows.csv',masks);write_json(out/'C_window_masks.json',windows)
 logs,signs=summary_tables(rows,shared_draws())
 write_json(out/'C_loglog.json',logs);write_csv(out/'C_loglog.csv',logs);write_csv(out/'C_sign_agreement.csv',signs)
 errors=[r for r in rows if r['status']=='execution_error'];unresolved=sum(r['status'] in NUMERICAL_UNRESOLVED for r in rows);write_json(out/'C_MERGE_RECEIPT.json',dict(protocol_sha256=digest,manifest_sha256=manifest_hash,total_rows=len(rows),required_modes=REQUIRED_MODES,required_curve_total=1128600,required_primary_total=75240,row_schema_version=ROW_SCHEMA_VERSION,rows_per_mode=376200,primary_rows_per_mode=25080,expected_cases=6270,representative_case_count=len(windows),expected_representatives=18,status='execution_incomplete' if errors or unresolved else 'execution_complete',error_rows=len(errors),numerical_unresolved_rows=unresolved,status_counts=dict(Counter(r['status'] for r in rows)),bootstrap_file_sha256=sha(ROOT/'results/phase4/bootstrap_draw_matrix.npy'),bootstrap_logical_sha256=hashlib.sha256(np.asarray(shared_draws(),dtype='<i8').tobytes()).hexdigest()))
 return 1 if errors or unresolved else 0


def write_csv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in rows:writer.writerow({k:json.dumps(jsonable(v)) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})


def main():
 ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--production',action='store_true');mode.add_argument('--dry-inventory',action='store_true');mode.add_argument('--fixture',action='store_true');mode.add_argument('--merge',action='store_true')
 ap.add_argument('--manifest','--cases',type=Path);ap.add_argument('--out',type=Path,default=P5);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=1);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 if a.fixture:
  import subprocess
  for script in ('test_fixtures.py','AR_DELTA_tests.py','AR_DELTA_pipeline.py','AR_DELTA_mask_pipeline.py','AR_DELTA_legacy_pipeline.py','AR_DELTA_quadrature.py'):
   rc=subprocess.call([sys.executable,str(P5/'implementation_C'/script)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
   if rc:return rc
  return 0
 if a.manifest is None:raise ValueError('--manifest/--cases required outside --fixture')
 if not 0<=a.shard<a.nshards:raise ValueError('Invalid shard')
 if not a.out.resolve().is_relative_to(P5.resolve()):raise PermissionError('Only phase5 output paths are permitted')
 digest,freeze=require_frozen() if a.production or a.merge else (None,None)
 records=manifest_adapter(a.manifest)
 if a.merge:return merge_outputs(a.out,records,sha(a.manifest),digest,execution_fingerprint(digest,a.manifest,freeze))
 if a.dry_inventory:
  print(json.dumps(dict(inventory_receipt(len(records)),mode='input_only',counts=dict(Counter(r['source_phase'] for r in records)),representatives=representative_keys())));return
 # Freeze must pin the exact inventory supplied by collector.
 frozen_keys={(r['source_phase'],r['case_id']) for r in __import__('p5_contracts').c_universe()}
 if {(r['source_phase'],r['case_id']) for r in records}!=frozen_keys:raise PermissionError('Collector identity differs from frozen6270 universe')
 out=a.out;out.mkdir(parents=True,exist_ok=True);receipt_dir=out/'C_receipts';receipt_dir.mkdir(exist_ok=True)
 tag=f'shard{a.shard:02d}of{a.nshards:02d}';rows=[];maskrows=[];windowrows=[];errors=[];fingerprint=execution_fingerprint(digest,a.manifest,freeze)
 if a.resume:
  for rec in records[a.shard::a.nshards]:
   cached=receipt_dir/f"{rec['source_phase']}__{rec['case_id']}.json"
   if cached.exists():validate_resume(json.loads(cached.read_text()),fingerprint,{f:sha(rec[f]) for f in ('tf_input_path','truth_path','w_path')})
 for i,rec in enumerate(records[a.shard::a.nshards]):
  p=receipt_dir/f"{rec['source_phase']}__{rec['case_id']}.json"
  try:
   reused=False
   input_hashes={f:sha(rec[f]) for f in ('tf_input_path','truth_path','w_path')}
   if a.resume and p.exists():
    old=json.loads(p.read_text())
    validate_resume(old,fingerprint,input_hashes)
    load_record(rec,freeze);rr=old['rows'];mm=old['mask_rows'];ww=old['windows'];reused=True
   if not reused:
    rr,mm,ww,rp=run_case(rec,freeze);write_json(p,dict(fingerprint=fingerprint,input_hashes=input_hashes,rows=rr,mask_rows=mm,windows=ww,receipt=rp))
   receipt_hash=sha(p)
   rr=[dict(r,receipt_path=str(p),receipt_sha256=receipt_hash) for r in rr]
   rows.extend(rr);maskrows.extend(mm)
   if ww is not None:windowrows.append(dict(source_phase=rec['source_phase'],case_id=rec['case_id'],windows=ww))
  except Exception as exc:
   import traceback
   error=dict(source_phase=rec['source_phase'],case_id=rec['case_id'],error=str(exc),traceback=traceback.format_exc());errors.append(error);write_json(p,dict(fingerprint=fingerprint,status='execution_error',**error))
   # Errors are explicit inventory records, never omitted from coverage.
   rows.extend(error_rows(rec,str(exc)))
  write_json(out/f'C_STATUS_{tag}.json',dict(fingerprint=fingerprint,processed=i+1,total=len(records[a.shard::a.nshards]),errors=errors,status='running'))
 write_csv(out/f'C_linearized_rows_{tag}.csv',rows);write_csv(out/f'C_masking_rows_{tag}.csv',maskrows);write_json(out/f'C_window_masks_{tag}.json',windowrows)
 logs,signs=summary_tables(rows,shared_draws())
 write_json(out/f'C_loglog_{tag}.json',logs);write_csv(out/f'C_sign_agreement_{tag}.csv',signs)
 unresolved=sum(r['status'] in NUMERICAL_UNRESOLVED for r in rows)
 write_json(out/f'C_STATUS_{tag}.json',dict(fingerprint=fingerprint,processed=len(records[a.shard::a.nshards]),rows=len(rows),errors=errors,numerical_unresolved_rows=unresolved,status='component_execution_complete' if not errors and not unresolved else 'component_execution_incomplete'))
 return 1 if errors or unresolved else 0


if __name__=='__main__':sys.exit(main())
