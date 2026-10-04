"""Preparation verification only: synthetic rain, schedules, contracts; no W/reference/model scoring."""
import ast,copy,json
from pathlib import Path
import numpy as np
from p4_contracts import ROOT,P4,sha,verify_old,reuse_cohorts,require_frozen
import p4_cases as cases
import p4_export as export
import p4_analysis as analysis

def main():
 checks={}
 for p in (ROOT/'lib').glob('p4_*.py'):ast.parse(p.read_text())
 checks['p4_syntax']=True
 cs=list(cases.all_A_cases(fixture=True))
 assert len(cs)==len({c['case_id'] for c in cs})==720
 checks['synthetic_A_unique_count']=len(cs)
 assert {c['derived']['rho_nominal'] for c in cs}=={0.,.1,.5,2.}
 assert {c['derived']['Q_scale'] for c in cs}=={.3,1.,3.}
 zeros=[c for c in cs if c['derived']['rho_nominal']==0]
 for c in zeros:
  q=c['tf_input']['pumping_context'];m=c['derived'];assert m['R_days']==m['rho_realized']==0
  assert np.count_nonzero(q==0)==3 and np.max(np.diff(np.flatnonzero(q==0)))==2
  assert not m['census']['designated_rest_identity'] and m['census']['designated_rest_absent']
 checks['rho0_no_designated_rest_short_off_switches']=len(zeros)
 for c in cs:
  m=c['derived'];t=c['truth'];q=c['tf_input']['pumping_context']
  assert q[-1]>0 and np.isclose(m['SR'],m['gain_m_per_m3d']*m['mean_outside_rest_m3d']/m['sigma_bg_m'])
  assert np.max(np.abs(t['E_true'][cases.PAIRS[1]]+.5*t['E_true'][cases.PAIRS[0]]))<1e-12
 checks['physical_SR_pair_identity']=True
 # Shared outside-rest schedule and scale lineage across all rho.
 grouped={}
 for c in cs:
  m=c['derived'];grouped.setdefault((m['realization'],m['sid']),[]).append(c)
 for group in grouped.values():
  base=[c for c in group if c['derived']['Q_scale']==1.][0]['base_N']
  for c in group:assert np.array_equal(c['base_N'],base*c['derived']['Q_scale'])
 checks['rho_base_identical_scale_lineage']=True
 for H in (10,30):
  arr=export.export_arrays(cs,H,'f'*64);prep=export.prepare_track(arr,H,'raw',[c['case_id'] for c in cs])
  assert len(prep['query_id'])==2880 and len(prep['pair_id'])==1440
  assert prep['covariates'].shape==(2880,2,1024+H)
 checks['A_queries_per_horizon']=2880;checks['raw_outputs_per_case_across_horizons']=8
 # Origin is actually zero; both future schedules are zero, no replacement rate.
 c=copy.deepcopy(cs[0]);c['tf_input']['pumping_context'][-1]=0.
 for key in c['tf_input']['future_Q']:c['tf_input']['future_Q'][key][:]=0.
 for H in (10,30):
  arr=export.export_arrays([c],H,'f'*64);assert np.all(arr['pumping'][:,1024:]==0)
  bad={k:v.copy() for k,v in arr.items()};bad['pumping'][0,1024:]=100.
  try:export.validate(bad,H)
  except ValueError:pass
  else:raise AssertionError('Nominal origin replacement accepted')
 try:export.prepare_track(arr,30,'exp10')
 except ValueError:pass
 else:raise AssertionError('Filtered track accepted')
 checks['zero_origin_kept_and_nominal_replacement_rejected']=True;checks['filtered_track_rejected']=True
 cohorts=reuse_cohorts();(P4/'PREPARE_REUSE_COHORTS.json').write_text(json.dumps(cohorts,indent=1))
 checks['old_map_cases']=len(cohorts['map']);checks['old_C_cases']=len(cohorts['contradiction'])
 # Synthetic statistical fixture with overlap, paired clusters and known finite threshold.
 rng=np.random.default_rng(1);sr=np.exp(rng.uniform(-2,2,400));rho=rng.choice([0.,.1,.5,2.],400)
 X=analysis.design(sr,rho);y=(rng.random(400)<1/(1+np.exp(-(X@np.array([.3,1.2,.5]))))).astype(int)
 fit=analysis.fit_logistic(X,y);assert fit['status']=='identified'
 assert analysis.threshold([0.,1.,0.],0.,.5,(.1,10))['status']=='finite'
 assert analysis.threshold([0.,1.,0.],0.,.9,(.1,2))['status']=='extrapolated'
 assert analysis.fit_logistic(X,np.ones(400))['status']=='unidentifiable'
 assert analysis.record_metrics(0.,0.,0.,0.,0.)['W_over_absE'] is None
 r=np.repeat(np.arange(10),40);ix=next(analysis.cluster_indices(r,1));assert all(np.sum(r[ix]==v)%40==0 for v in np.unique(r[ix]))
 idx,edges=analysis.log_signal_bins([.01,.1,1.,10.,100.,0.,np.nan]);assert len(edges)==6 and idx[-2:].tolist()==[-1,-1]
 checks['analysis_synthetic_overlap_thresholds_cluster_bins_zero']=True
 try:require_frozen()
 except (FileNotFoundError,PermissionError):checks['official_execution_blocked_before_freeze']=True
 else:raise AssertionError('Unexpected complete freeze already exists')
 # Numerical function bodies remain exactly the frozen execution driver implementations.
 def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
 old=functions(ROOT/'lib/p3_phase3_run_wb.py');new=functions(ROOT/'lib/p4_run_wb.py')
 for name in ('run_W','run_reference','truth_eval','fresh','run_case'):assert old[name]==new[name],name
 checks['W_reference_truth_functions_unchanged']=True
 # Reopen an actual synthetic packed raw-output archive; no model predictions are run.
 from p3_phase2_timesfm_execution import atomic_npz,pack_output
 from p4_timesfm_execution import verify
 observed=export.export_arrays([cs[0]],10,'f'*64);prepared=export.prepare_track(observed,10)
 q=np.broadcast_to(np.arange(9,dtype=np.float32),(4,10,9)).copy()
 identity={'horizon':10,'track':'raw','fixture_only':True}
 path=P4/'prepare_fixture/raw_output_fixture.npz';atomic_npz(path,pack_output(q,prepared,identity));verify(path,prepared,identity)
 checks['synthetic_model_output_roundtrip_no_inference']=True
 checks['old_hashes_unchanged']=verify_old();checks['all_ok']=True
 (P4/'PREPARE_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
