"""Post-freeze generation of all 2880 new D05 cases (A720 + B2160), zero-origin cases kept.

Run ONCE after the complete freeze, before either actor shard. Writes tf_inputs/truth NPZ per case, the derived
manifest, both horizon tool archives (all 2880 cases; shards subset them), explicit map/C cohort manifests and the
generation checks. Counters use the declared origin rate q_origin, never the last context day.
"""
import json
import numpy as np
from p4_contracts import ROOT,P4,require_frozen,sha,A_COUNT,RAW_B_COUNT,RAW_TOTAL,cohort_manifests,new_case_ids
from p4_cases import all_A_cases,all_B_cases,PAIRS
from p4_export import export_arrays
from p3_make_cases import save_tf,save_truth,TRUTH_KEYS
import p4_partition as part

def manifest_row(c,tf,tr,out):
 d={k:v for k,v in c['derived'].items() if k!='census'}
 d['census']=c['derived'].get('census')
 return dict(d,tf_input=str(tf.relative_to(out)),tf_sha256=sha(tf),truth=str(tr.relative_to(out)),truth_sha256=sha(tr),shard_actor=part.actor_of(c['case_id']))

def generation_checks(cases,digest):
 A=[c for c in cases if c['derived']['experiment_group']=='A'];B=[c for c in cases if c['derived']['experiment_group']=='B']
 ids=[c['case_id'] for c in cases];eA,eB=new_case_ids()
 zero=[c for c in B if c['derived']['q_origin_m3d']==0]
 checks=dict(n_cases=len(cases),n_A=len(A),n_B=len(B),unique=len(set(ids))==len(ids),ids_match_identity=set(ids)==set(eA)|set(eB),
             zero_origin_cases=len(zero),zero_origin_cases_A=sum(c['derived']['q_origin_m3d']==0 for c in A),
             zero_origin_truth_exact_zero=all(np.all(c['truth']['E_true'][p]==0) for c in zero for p in PAIRS),
             q_origin_ne_q_last=sum(c['derived']['q_origin_m3d']!=c['derived']['q_last_context_day_m3d'] for c in B),
             A_rho0_no_zero_day=all(np.all(c['tf_input']['pumping_context']>0) for c in A if c['derived']['rho_nominal']==0),
             water_curtain_variant_cases=sum(bool(c['derived'].get('offseason_use_variant')) for c in B),
             pair_identity_maxabs_m=float(max(np.max(np.abs(c['truth']['E_true'][PAIRS[1]]+.5*c['truth']['E_true'][PAIRS[0]])) for c in cases)),
             protocol_sha256=digest)
 checks['all_ok']=bool(checks['n_A']==A_COUNT and checks['n_B']==RAW_B_COUNT and checks['unique'] and checks['ids_match_identity'] and checks['zero_origin_truth_exact_zero']
                       and checks['zero_origin_cases_A']==0 and checks['A_rho0_no_zero_day'] and checks['water_curtain_variant_cases']==216 and checks['pair_identity_maxabs_m']<1e-12)
 return checks

def main():
 digest=require_frozen()
 cases=list(all_A_cases())+list(all_B_cases())
 if len(cases)!=RAW_TOTAL or len({c['case_id'] for c in cases})!=RAW_TOTAL:raise ValueError('Full grid coverage mismatch')
 out=P4/'cases';man=[]
 for sub in ('tf_inputs','truth'):(out/sub).mkdir(parents=True,exist_ok=True)
 for c in cases:
  cid=c['case_id'];tf=out/'tf_inputs'/f'{cid}.npz';tr=out/'truth'/f'{cid}.npz';save_tf(tf,c,digest);save_truth(tr,c,digest)
  with np.load(tf,allow_pickle=False) as z:
   if set(z.files)&TRUTH_KEYS:raise ValueError('Truth leakage')
  man.append(manifest_row(c,tf,tr,out))
 for H in (10,30):np.savez_compressed(out/f'tool_inputs_H{H}.npz',**export_arrays(cases,H,digest))
 (out/'derived_manifest.json').write_text(json.dumps(man,indent=1))
 (out/'cohort_manifests.json').write_text(json.dumps(cohort_manifests(),indent=1))
 checks=generation_checks(cases,digest)
 checks['tool_inputs_sha256']={H:sha(out/f'tool_inputs_H{H}.npz') for H in (10,30)}
 (out/'generation_checks.json').write_text(json.dumps(checks,indent=1))
 if not checks['all_ok']:raise SystemExit('GENERATION_CHECK_FAILED')
 require_frozen();print('D05_CASES_GENERATED',json.dumps({k:v for k,v in checks.items() if k!='tool_inputs_sha256'}))
if __name__=='__main__':main()
