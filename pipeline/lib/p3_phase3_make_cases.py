"""Generate and verify all690 D04 cases before W/ref scoring; fixture mode never uses observed rain."""
import argparse,hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import p3_phase3_cases as pc
import p3_phase3_export as ex
from p3_make_cases import save_tf,save_truth,load_tf_input,TRUTH_KEYS
ROOT=pc.ROOT;P3=pc.P3

def invariants(cases,fixture=False):
    flags=defaultdict(lambda:True);groups=defaultdict(list);weather={};byid={c['case_id']:c for c in cases};maxmatch=0.;maxid=0.
    for c in cases:
        m=c['derived'];ti=c['tf_input'];t=c['truth'];cen=m['census'];group=m['experiment_group'];rr=m['realization'];groups[(group,m['sid'],rr)].append(c)
        if rr not in weather:weather[rr]=c
        b=weather[rr]
        flags['weather_noise_paired'] &= np.array_equal(ti['rain'],b['tf_input']['rain']) and np.array_equal(t['h_nat'],b['truth']['h_nat']) and np.array_equal(t['eps'],b['truth']['eps'])
        flags['longrest_exact'] &= cen['designated_rest_identity'] and np.all(ti['pumping_context'][m['rest_start_day']:m['layer_rest_end_day']]==0)
        flags['short_off_1day'] &= cen['N_OFF']==m['N_nominal']//2 and cen['all_short_1d_lt_t95']
        flags['positive_origin'] &= ti['pumping_context'][-1]>0
        flags['physical_mapping'] &= np.isclose(m['a_d'],m['c_d']*m['storage_value']) and np.isclose(m['b'],m['r_m']**2/(4*m['T_m2_d']*m['c_d']))
        flags['SR_definition'] &= np.isclose(m['SR'],m['gain_m_per_m3d']*m['mean_outside_rest_m3d']/m['sigma_bg_m'])
        flags['pair_origin'] &= all(np.array_equal(ti['future_Q'][f'{p}_a'],np.full(30,m['q0_m3d'])) for p in pc.PAIRS)
        maxid=max(maxid,float(np.max(np.abs(t['E_true'][pc.PAIRS[1]]+.5*t['E_true'][pc.PAIRS[0]]))))
        if group=='A':
            on=ti['pumping_context']>0;st,en=m['burst_start_day'],m['burst_end_boundary']
            switches=np.flatnonzero(on[1:]!=on[:-1])+1; bs=switches[(switches>st)&(switches<=en)]
            flags['A_actual_switch_count'] &= len(bs)==m['N_nominal'] and bs[-1]==en and np.array_equal(bs,np.arange(st+1,en+1))
            flags['A_recency_rounding'] &= pc.CTX-en==round(m['recency_target_ratio']*m['t95_d'])
            flags['A_actual_census'] &= cen['N_ON']==m['N_nominal']//2-1 and cen['N_short']==m['N_nominal']-1
            flags['A_outside_rest'] &= m['layer_rest_end_day']+mathceil(m['t95_d'])<=st
        else:
            flags['BC_actual_census'] &= cen['N_ON']==m['N_nominal']//2 and cen['N_short']==m['N_nominal']
        if not fixture and 'd03_parent_case_id' in m:
            par=m['d03_parent_case_id']; parent=load_tf_input(pc.d03.P2/'cases/tf_inputs'/f'{par}.npz')
            with np.load(pc.d03.P2/'cases/truth'/f'{par}.npz',allow_pickle=False) as z:
                flags['lineage_background'] &= np.array_equal(z['h_nat'],t['h_nat']) and np.array_equal(z['eps'],t['eps'])
            flags['lineage_Q_scale'] &= np.array_equal(ti['pumping_context'],parent['pumping_context']*m['Q_scale'])
            flags['lineage_rain'] &= np.array_equal(ti['rain'],parent['rain'])
            if group=='C':flags['C_001_head_exact_D03'] &= np.array_equal(ti['head_context'],parent['head_context'])
    for (group,sid,rr),cs in groups.items():
        if group=='A':
            v=np.array([c['derived']['V_m3'] for c in cs]);q=np.array([c['derived']['mean_outside_rest_m3d'] for c in cs]);rel=max(np.ptp(v)/v.mean(),np.ptp(q)/q.mean());maxmatch=max(maxmatch,float(rel))
            flags['A_fourway_match'] &= len(cs)==4 and rel<1e-12
            flags['A_common_rest_origin'] &= len({(c['derived']['rest_start_day'],c['derived']['layer_rest_end_day'],c['derived']['q0_m3d']) for c in cs})==1
    counts=Counter(c['derived']['experiment_group'] for c in cases)
    flags['full_unique_grid'] &= len(cases)==len(byid)==690 and dict(counts)=={'A':240,'B':360,'C':90}
    flags['truth_pair_identity'] &= maxid<1e-12
    return dict(invariants={k:bool(v) for k,v in flags.items()},case_counts=dict(counts),unique_ids=len(byid),matching_max_relative_spread=maxmatch,truth_pair_identity_maxabs_m=maxid,all_ok=bool(all(flags.values())))

def mathceil(x):return int(np.ceil(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');a=ap.parse_args();t0=time.time()
    digest='f'*64 if a.fixture else pc.check_frozen();out=P3/('B_fixture/cases' if a.fixture else 'cases')
    cases=list(pc.all_cases(a.fixture));checks=invariants(cases,a.fixture)
    if not checks['all_ok']:raise AssertionError(checks)
    for d in ('tf_inputs','truth'): (out/d).mkdir(parents=True,exist_ok=True)
    man=[]
    for c in cases:
        cid=c['case_id'];tf=out/'tf_inputs'/f'{cid}.npz';tr=out/'truth'/f'{cid}.npz';save_tf(tf,c,digest);save_truth(tr,c,digest)
        back=load_tf_input(tf)
        assert np.array_equal(back['head_context'],c['tf_input']['head_context']) and np.array_equal(back['pumping_context'],c['tf_input']['pumping_context'])
        with np.load(tf,allow_pickle=False) as z:assert not(set(z.files)&TRUTH_KEYS)
        man.append(dict(c['derived'],tf_input=str(tf.relative_to(out)),tf_sha256=pc.sha(tf),truth=str(tr.relative_to(out)),truth_sha256=pc.sha(tr)))
    tool={}
    for H in (10,30):
        arr=ex.export_arrays(cases,H,digest);p=out/f'tool_inputs_H{H}.npz';np.savez_compressed(p,**arr)
        with np.load(p,allow_pickle=False) as z:back={k:z[k] for k in z.files}
        ex.validate(back,H);assert not(set(back)&TRUTH_KEYS)
        tool[H]=dict(path=str(p.relative_to(ROOT)),sha256=pc.sha(p),rows=len(arr['query_id']),cases=len(set(arr['case_id'])),pairs=len(set(arr['pair_id'])),shapes={k:list(arr[k].shape) for k in ('head','pumping','rainfall')})
    # Negative/edge fixture: wrong future rate despite a valid generic tensor shape must be rejected.
    bad={k:v.copy() for k,v in arr.items()};bad['pumping'][0,1024:]+=1
    try:ex.validate(bad,H)
    except (ValueError,AssertionError):checks['wrong_origin_rate_rejected']=True
    else:raise AssertionError('Invalid origin-linked future accepted')
    checks.update(protocol_sha256=digest,fixture=a.fixture,tool=tool,runtime_s=time.time()-t0,code_sha256={str(p.relative_to(ROOT)):pc.sha(p) for p in (Path(__file__).resolve(),ROOT/'lib/p3_phase3_cases.py',ROOT/'lib/p3_phase3_export.py')})
    (out/'derived_manifest.json').write_text(json.dumps(man,indent=1,ensure_ascii=False));(out/'generation_checks.json').write_text(json.dumps(checks,indent=1))
    if not a.fixture:pc.check_frozen()
    print(json.dumps(checks));return 0
if __name__=='__main__':sys.exit(main())
