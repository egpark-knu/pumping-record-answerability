"""Independent analytic truth/physical fixtures and D03-identical C golden comparisons."""
import json,datetime
from pathlib import Path
import numpy as np
import p3_phase3_cases as pc
from p3_phase2_physmap import unit_step
ROOT=pc.ROOT;P3=pc.P3
man=json.loads((P3/'cases/derived_manifest.json').read_text());err=0.;short_grid=0.;scaling=0.;gainC=set();piC={};golden=[]
for m in man:
    with np.load(P3/'cases'/m['truth'],allow_pickle=False) as z:
        E={p:z['E_true__'+p] for p in pc.PAIRS}
        ana=-m['q0_m3d']*m['gain_m_per_m3d']*unit_step(m['a_d'],m['b'],np.arange(0,pc.CTX+pc.HMAX+1))[1:31]
        short=-m['q0_m3d']*m['gain_m_per_m3d']*unit_step(m['a_d'],m['b'],np.arange(1,31))
        short_grid=max(short_grid,float(np.max(np.abs(short-ana))))
        err=max(err,float(np.max(np.abs(E[pc.PAIRS[0]]-ana))))
        if m['experiment_group']=='B':
            with np.load(pc.d03.P2/'cases/truth'/f"{m['d03_parent_case_id']}.npz",allow_pickle=False) as p:
                scaling=max(scaling,float(np.max(np.abs(E[pc.PAIRS[0]]-m['Q_scale']*p['E_true__'+pc.PAIRS[0]]))))
    if m['experiment_group']=='C':gainC.add(m['gain_m_per_m3d']);piC[m['storage_value']]=m['pi_r']
    if m['experiment_group']=='C' and m['storage_value']==.001:
        w=P3/'wb/W'/f"{m['case_id']}.json";r=P3/'wb/reference'/f"{m['case_id']}.json"
        if not w.exists() or not r.exists():continue
        w1=json.loads(w.read_text());r1=json.loads(r.read_text());w0=json.loads((pc.d03.P2/'wb/W'/f"{m['d03_parent_case_id']}.json").read_text());r0=json.loads((pc.d03.P2/'wb/reference'/f"{m['d03_parent_case_id']}.json").read_text())
        diffs={}
        for pair in pc.PAIRS:
            for k in ('inf','sup','W'):diffs[pair+'_'+k]=float(np.max(np.abs(np.array(w1['envelope'][pair][k])-w0['envelope'][pair][k])))
            diffs[pair+'_reference_E']=float(np.max(np.abs(np.array(r1['pairs'][pair]['E_point'])-r0['pairs'][pair]['E_point'])))
        golden.append(dict(case_id=m['case_id'],d03_parent=m['d03_parent_case_id'],diffs=diffs))
rec=dict(at=datetime.datetime.now().astimezone().isoformat(),analytic_E_P1_maxabs_m=err,independent_short_grid_interpolation_difference_m=short_grid,short_grid_note="Frozen kernel uses 4000 log nodes with max-time-dependent integration mesh. Algebraic identity uses same1054-day mesh; 30-day remesh discrepancy reported, not used to alter truth or algorithms.",B_scaled_truth_maxabs_m=scaling,C_identical_steady_gain=len(gainC)==1,C_pi_r_by_S=piC,C_001_golden_completed=len(golden),C_001_golden=golden,all_available_checks_ok=err<1e-12 and scaling<1e-12 and len(gainC)==1 and all(max(g['diffs'].values())<1e-12 for g in golden),audit_code_sha256=pc.sha(__file__))
(P3/'B_NUMERICAL_FIXTURES.json').write_text(json.dumps(rec,indent=1));print(json.dumps({k:v for k,v in rec.items() if k!='C_001_golden'}));assert rec['all_available_checks_ok']
