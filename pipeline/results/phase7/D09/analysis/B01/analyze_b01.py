"""Expand matrix B01 only; reuse version-matched prior paddy/changed540 intervals."""
from pathlib import Path
import csv,json,hashlib,collections,datetime,sys,statistics,bisect
import numpy as np
BASE=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[5])))
D=BASE/'results/phase7/D09'; OUT=Path(__file__).resolve().parent; PRIOR=OUT.parent
P=BASE/'results/phase6/D08/protocol_neutral';E=P/'execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def csvout(p,x):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
matrix=rd(D/'writing/ACTION_MATRIX.json');mr=rd(D/'writing/ACTION_MATRIX_RECEIPT.json');work=rd(OUT/'WORK_ITEMS.json')
assert mr['component_complete'] and mr['status']=='ACTION_MATRIX_COMPLETE' and mr['validation_result']=='pass'
assert sha(D/'writing/ACTION_MATRIX.json')==mr['matrix_artifact_sha256'][str(D/'writing/ACTION_MATRIX.json')]
job=next(x for x in matrix['calculation_work_items'] if x['id']=='B01');assert work==[job]
prior=rd(PRIOR/'FIRST_BATCH_RECEIPT.json');ps=rd(PRIOR/'SUMMARY_FIRST_BATCH.json');assert prior['validation_result']=='pass' and prior['direct_output_verification']['status']=='pass'
paths=list(map(Path,job['source_paths']))+[D/'author instruction',D/'writing/ACTION_MATRIX.json',D/'writing/ACTION_MATRIX_RECEIPT.json',OUT/'WORK_ITEMS.json',PRIOR/'FIRST_BATCH_RECEIPT.json',PRIOR/'SUMMARY_FIRST_BATCH.json',PRIOR/'SOURCE_FIRST_BATCH.csv',PRIOR/'BOOTSTRAP_FIRST_BATCH.csv',PRIOR/'VERIFICATION_FIRST_BATCH.json',P/'SOURCE_INDEX.json',P/'EXECUTION_PACKET.json',BASE/'results/phase6/protocol.md']
baseline={str(p):sha(p) for p in paths}
for p in paths:
    if str(p) in mr['source_sha256']:assert sha(p)==mr['source_sha256'][str(p)],p
    if str(p) in prior['source_current_sha256']:assert sha(p)==prior['source_current_sha256'][str(p)],p
    if str(p) in prior['output_sha256']:assert sha(p)==prior['output_sha256'][str(p)],p
cr=rd(D/'control/CONTROL_RECEIPT.json');cs=rd(D/'control/CONTROL_SUMMARY.json')
assert cr['component_complete'] and cr['validation_result']=='pass'
for p in [D/'control/CONTROL_SOURCE_FRAME.csv',D/'control/CONTROL_SUMMARY.json']:assert sha(p)==cr['evidence'][str(p)]
src=rows(PRIOR/'SOURCE_FIRST_BATCH.csv');control=rows(D/'control/CONTROL_SOURCE_FRAME.csv');bs=rows(E/'B/source_frame.csv')
active=set(rd(P/'SOURCE_INDEX.json')['active20_ids']);assert len(src)==len(control)==len(active)==1242
ci={x['case_id']:x for x in control};assert set(ci)=={x['case_id'] for x in src}==active
bi={(x['case_id'],x['pair'],int(x['lead']),int(x['realization']),x['variant']):(i+2,x) for i,x in enumerate(bs)};assert len(bi)==len(bs)==14904
for r in src:
    c=ci[r['case_id']];assert r['operation']==c['operation'] and int(r['realization'])==int(c['realization'])
    assert r['pair']=='P2_current_vs_1p5x' and int(r['lead'])==10 and float(r['q_origin_m3d'])>0 and float(r['geometry_r_m'])==20
    for v in ['range','control']:
        line,b=bi[r['case_id'],r['pair'],10,int(r['realization']),v]
        assert line==int(r[v+'_B_source_csv_line'])
        for k in ['signed_ratio','paired_ratio_change','E_engine_m']:assert float(r[v+'_'+k])==float(c[v+'_'+k])==float(b[k])
    changed=float(c['control_E_engine_m'])!=float(c['original_E_engine_m'])
    assert changed==(r['control_literal_changed']=='True')==(c['control_engine_contrast_changed']=='True')==(c['control_pumping_input_changed']=='True')
    assert float(r['range_minus_control_paired_contrast'])==float(r['range_signed_ratio'])-float(r['control_signed_ratio'])
csvout(OUT/'B01_recordwise_contrasts.csv',src)
draws=np.load(BASE/'results/phase4/bootstrap_draw_matrix.npy',allow_pickle=False)
assert draws.shape==(999,10) and sha(BASE/'results/phase4/bootstrap_draw_matrix.npy')==ps['bootstrap']['physical_sha256']
assert hashlib.sha256(np.asarray(draws,'<i8').tobytes()).hexdigest()==ps['bootstrap']['logical_le_i8_sha256']
groups=[('full1242',src,cs['full1242'],None),('domestic714',[r for r in src if r['operation']=='domestic_continuous'],cs['operating_table'][0],None),('paddy168',[r for r in src if r['operation']=='paddy_irrigation'],cs['operating_table'][1],'B01'),('water360',[r for r in src if r['operation']=='water_curtain'],cs['operating_table'][2],None),('changed540',[r for r in src if r['control_literal_changed']=='True'],cs['conditional_changed540'],'B02'),('changed_paddy162',[r for r in src if r['operation']=='paddy_irrigation' and r['control_pumping_input_changed']=='True'],cs['paddy_clean_input_changed'],None)]
metrics=['range_minus_control_paired_contrast','range_paired_ratio_change','control_paired_ratio_change']
results={};table=[];newboots=[];clusterrows=[]
def weightedmedian(vals,weights):
    pairs=sorted(zip(vals,weights));total=sum(weights)
    if not total:return None
    cum=[];n=0
    for v,w in pairs:n+=int(w);cum.append(n)
    get=lambda i:pairs[bisect.bisect_right(cum,i)][0]
    return (get((total-1)//2)+get(total//2))/2
def near(a,b):assert abs(a-b)<=1e-12*max(1,abs(a),abs(b)),(a,b)
new_checks=0
for name,rs,expected,reuse in groups:
    assert len(rs)==expected['n']
    for field,col in [('range_minus_original_paired_ratio_median','range_paired_ratio_change'),('control_minus_original_paired_ratio_median','control_paired_ratio_change'),('range_signed_ratio_median','range_signed_ratio'),('control_signed_ratio_median','control_signed_ratio'),('original_signed_ratio_median','original_signed_ratio')]:assert float(np.median([float(r[col]) for r in rs]))==expected[field]
    assert sum(r['control_literal_changed']=='True' for r in rs)==expected['control_engine_contrast_changed']
    real=np.array([int(r['realization']) for r in rs]);vals=np.array([[float(r[k]) for k in metrics] for r in rs]);assert np.isfinite(vals).all()
    samples=None
    if not reuse:
        samples=[]
        for b,draw in enumerate(draws):
            z=np.repeat(vals,draw[real],axis=0);med=np.median(z,axis=0) if len(z) else [np.nan]*3;samples.append(med)
            for j,k in enumerate(metrics):
                independent=weightedmedian(vals[:,j].tolist(),draw[real].tolist())
                if independent is not None:near(float(med[j]),independent);new_checks+=1
            newboots.append(dict(cohort=name,replicate=b,contributing_rows=len(z),contributing_distinct_realizations=int(np.sum(draw[np.unique(real)]>0)),**{k:'' if not np.isfinite(med[j]) else float(med[j]) for j,k in enumerate(metrics)}))
        samples=np.array(samples)
    stats={}
    for j,k in enumerate(metrics):
        v=vals[:,j];point=float(np.median(v));q=np.percentile(v,[10,25,50,75,90],method='linear')
        if reuse:
            old=ps['jobs'][reuse]['metrics'][k];assert point==old['median']
            lower=old['ci95_lower'];upper=old['ci95_upper'];ident=old['n_identified'];missing=old['n_unidentified'];status='conditional_on_identified_replicates'
        else:
            finite=samples[np.isfinite(samples[:,j]),j];ident=len(finite);missing=999-ident
            lower,upper=map(float,np.percentile(finite,[2.5,97.5],method='linear')) if ident else (None,None);status='conditional_on_identified_replicates' if ident else 'unidentifiable'
        stat=dict(median=point,ci95_lower=lower,ci95_upper=upper,n_identified=ident,n_unidentified=missing,interval_status=status,quantiles=dict(zip(['q10','q25','q50','q75','q90'],map(float,q))),positive_count=int(np.sum(v>0)),zero_count=int(np.sum(v==0)),negative_count=int(np.sum(v<0)),positive_fraction=float(np.mean(v>0)),interval_origin='reused_first_batch_'+reuse if reuse else 'new_missing_cohort_calculation')
        stats[k]=stat
        table.append(dict(cohort=name,n=len(rs),contributing_realization_count=len(set(real)),station_count=len({r['station'] for r in rs}),metric=k,**{x:stat[x] for x in ['median','ci95_lower','ci95_upper','n_identified','n_unidentified','interval_status','positive_count','zero_count','negative_count','positive_fraction','interval_origin']},**stat['quantiles']))
    for rID in sorted(set(real)):
        rr=[r for r in rs if int(r['realization'])==rID];cv=[float(r[metrics[0]]) for r in rr]
        clusterrows.append(dict(cohort=name,realization=int(rID),station=rr[0]['station'],n=len(rr),positive_contrast_count=sum(v>0 for v in cv),zero_contrast_count=sum(v==0 for v in cv),negative_contrast_count=sum(v<0 for v in cv),median_paired_contrast=statistics.median(cv)))
    results[name]={'n':len(rs),'contributing_realization_count':len(set(real)),'realization_ids':list(map(int,sorted(set(real)))),'station_rows':dict(collections.Counter(r['station'] for r in rs)),'operation_rows':dict(collections.Counter(r['operation'] for r in rs)),'unchanged_control_count':sum(r['control_literal_changed']!='True' for r in rs),'positive_median_realization_count':sum(x['median_paired_contrast']>0 for x in clusterrows if x['cohort']==name),'conditional_descriptive_subset':name in ['changed540','changed_paddy162'],'metrics':stats,'reused_first_batch_job':reuse,'control_summary_raw_medians_and_masks_reproduced':True}
csvout(OUT/'B01_operation_uncertainty.csv',table);csvout(OUT/'B01_new_bootstrap_statistics.csv',newboots);csvout(OUT/'B01_cluster_sharing.csv',clusterrows)
summary={'turn_id':'run','matrix_job':'B01','status':'complete','comment_ids':['R1-M5','R2-M6','R3-M2'],'cohorts':results,'definitions':{'paired_contrast':'median_i(R_range_i - R_control_i), no subtraction of group medians','changed540':'literal saved control_E_engine_m != original_E_engine_m; equal verified input-change mask; conditional descriptive','changed_paddy162':'paddy AND verified control pumping input changed; descriptive sensitivity; all168 main operation retained','full1242':'all original active20m cases; unchanged primary mechanical criterion','quantiles':'signed row distribution linear percentiles 10/25/50/75/90; exact zero without epsilon'},'bootstrap':ps['bootstrap'],'reuse':{'paddy168':'FIRST_BATCH B01','changed540':'FIRST_BATCH B02; not matrix B02','new_cohorts':['full1242','domestic714','water360','changed_paddy162'],'old_ci_recomputed':False,'all_cohort_raw_medians_reproduced':True},'fixed_primary':cr['primary'],'population':'Conditional on selected rainfall station records and twin design;10 realization IDs share3 stations; within-unit origins/layers and matched conditions remain paired. Not station-level/field probability or nonlinear-region coverage validation.','whole_parent_completion':False,'new_engine_queries':0,'new_W_searches':0,'new_simulations':0,'new_synthetic_draws':0}
save(OUT/'SUMMARY.json',summary);save(OUT/'ANALYSIS_SUMMARY.json',summary)
numbers={}
def walk(x,pointer):
    if isinstance(x,bool):return
    if isinstance(x,(float,int)):yield pointer,x
    elif isinstance(x,dict):
        for k,v in x.items():yield from walk(v,pointer+'/'+str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from walk(v,pointer+'/'+str(i))
for pointer,value in walk(results,'/cohorts'):
    name=pointer.split('/')[2]
    numbers[pointer]={'value':value,'summary_path':str(OUT/'SUMMARY.json'),'json_pointer':pointer,'source_csv':str(OUT/'B01_recordwise_contrasts.csv'),'cohort':name,'source_sha256':sha(OUT/'B01_recordwise_contrasts.csv'),'bootstrap_path':str(PRIOR/'BOOTSTRAP_FIRST_BATCH.csv') if name in ['paddy168','changed540'] else str(OUT/'B01_new_bootstrap_statistics.csv'),'definition_source':str(OUT/'SUMMARY.json')+' /definitions'}
save(OUT/'NEW_NUMBERS.json',{'values':numbers})
text=['# Matrix B01: paired higher-rate versus nominal control','','P2 increase, independent native10-day requests,20 m. Each contrast is the median of record-wise signed ratio differences.95% realization-cluster percentile intervals use the unchanged999 joint draws (PCG64 seed20261001). Each of six groups has10 contributing realization units and999 identified draws for each statistic.','','| Cohort | N | Range − original median [95% CI] | Control − original median [95% CI] | Paired range − control median [95% CI] | Positive contrast | Contrast Q10/Q25/Q50/Q75/Q90 |','|---|---|---|---|---|---|---|']
for name,d in results.items():
    ss=d['metrics'];fmt=lambda k:f"{ss[k]['median']:.6f} [{ss[k]['ci95_lower']:.6f}, {ss[k]['ci95_upper']:.6f}]";con=ss[metrics[0]]
    text.append(f"| {name} | {d['n']} | {fmt(metrics[1])} | {fmt(metrics[2])} | {fmt(metrics[0])} | {con['positive_count']}/{d['n']} ({con['positive_fraction']:.3%}) | "+' / '.join(f'{v:.6f}' for v in con['quantiles'].values())+' |')
text+=['','The primary full1242 mechanical result remains unchanged. Paddy168 includes six unchanged controls. Changed540 and changed_paddy162 are conditional descriptive subsets, not alternative primary endpoints. The nominal control expands the maximum for all360 water-curtain histories, and is range-neutral in domestic/paddy histories. A small nominal median does not establish equivalence. Paired benefit does not isolate a unique engine mechanism.','','Paddy168/changed540 intervals are reused at unchanged source/draw/output hashes from the prior terminal first batch; the other four sets are newly summarized. All six raw medians/masks reproduce CONTROL_SUMMARY. Full signed distributions, counts and source lines are in the CSV/JSON outputs. Realization-level sharing is in B01_cluster_sharing.csv.','','These intervals condition on three selected rainfall stations and the simulated design. They do not calibrate station-population uncertainty, nonlinear-region coverage, or individual field reliability. Required zero-face/global natural-shape profiling was not performed. This receipt completes only matrix B01, not matrix B02 or the whole D09.']
(OUT/'manuscript_ready_tables.md').write_text('\n'.join(text)+'\n')
assert {p:sha(p) for p in baseline}==baseline
verification={'status':'pass','all1242_case_pair_nativelead_realization_joins_checked':True,'prior_inputs_and_outputs_sha_match':True,'matrix_terminal_and_pin_verified':True,'all_six_raw_control_medians_and_masks_reproduced':True,'new_bootstrap_statistics_independently_checked':new_checks,'new_bootstrap_rows':len(newboots),'prior5994_statistic_verification_reused':True,'prior_bootstrap_verification_sha256':sha(PRIOR/'VERIFICATION_FIRST_BATCH.json'),'original_files_unchanged':True,'number_source_pointers':len(numbers)}
save(OUT/'B01_VERIFICATION.json',verification)
outputs=[OUT/x for x in ['B01_recordwise_contrasts.csv','B01_operation_uncertainty.csv','B01_new_bootstrap_statistics.csv','B01_cluster_sharing.csv','SUMMARY.json','ANALYSIS_SUMMARY.json','NEW_NUMBERS.json','manuscript_ready_tables.md','B01_VERIFICATION.json','analyze_b01.py']]
rec={'turn_id':'run','validation_type':'source_audit','validation_claim_id':'B01','validates_claim':'D09 B01 assigned stored-result analyses delivered','source_path':str(D/'author instruction'),'source_sha256':sha(D/'author instruction'),'audit_result':'pass','validation_result':'pass','component_complete':True,'whole_parent_completion':False,'assigned_jobs':[{'id':'B01','status':'complete','comment_ids':summary['comment_ids'],'cohorts':results}],'source_baseline_sha256':baseline,'source_current_sha256':{p:sha(p) for p in baseline},'sourcechecks':verification,'bootstrap':ps['bootstrap'],'output_sha256':{str(p):sha(p) for p in outputs},'reproduction_command':'python '+str(Path(__file__)),'sourcecode_sha256':sha(__file__),'software':{'python':sys.version,'numpy':np.__version__},'new_engine_queries':0,'new_W_searches':0,'new_simulations':0,'new_synthetic_draws':0,'blockers':[],'limits':summary['population'],'unexecuted_new_experiment':'zero-face/global natural-shape profiling remains(c); not assigned B01'}
save(OUT/'B01_SOURCE_RECEIPT.json',rec);save(OUT/'COMPONENT_RECEIPT.json',rec)
print(json.dumps({'status':'pass','new_weighted_order_statistic_checks':new_checks,'cohorts':{n:{'n':d['n'],'contrast':d['metrics'][metrics[0]]} for n,d in results.items()}},indent=2))
