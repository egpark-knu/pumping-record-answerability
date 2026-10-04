"""B01/B02 stored-result paired summaries. No hydrological/model/W execution."""
from pathlib import Path
import csv,json,hashlib,collections,datetime,sys
import numpy as np
BASE=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[4])))
OUT=Path(__file__).resolve().parent
P=BASE/'results/phase6/D08/protocol_neutral'
E=P/'execution'
C=BASE/'results/phase7/D09/control'
PAIR='P2_current_vs_1p5x'; LEAD=10; SEED=20261001
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def close(a,b):assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(a,b)
sources=[C/'CONTROL_RECEIPT.json',C/'CONTROL_SOURCE_FRAME.csv',C/'audit_control.py',E/'B/source_frame.csv',E/'B/AUTHOR_CRITERION.json',P/'SOURCE_INDEX.json',P/'EXECUTION_PACKET.json',BASE/'results/phase6/protocol.md',BASE/'results/phase4/bootstrap_draw_matrix.npy',BASE/'results/phase7/D09/author instruction']+[BASE/'manuscript/review_mock'/f'R{i}.md' for i in (1,2,3)]
baseline={str(p):sha(p) for p in sources}
receipt=rd(C/'CONTROL_RECEIPT.json');assert receipt['component_complete'] and receipt['validation_result']=='pass'
assert sha(C/'CONTROL_SOURCE_FRAME.csv')==receipt['evidence'][str(C/'CONTROL_SOURCE_FRAME.csv')]
assert sha(C/'audit_control.py')==receipt['evidence'][str(C/'audit_control.py')]
for p in [E/'B/source_frame.csv',E/'B/AUTHOR_CRITERION.json',P/'SOURCE_INDEX.json',P/'EXECUTION_PACKET.json',BASE/'results/phase6/protocol.md']:
    assert sha(p)==receipt['source_sha256_inventory'][str(p)]
index=rd(P/'SOURCE_INDEX.json');packet=rd(P/'EXECUTION_PACKET.json')
active=set(index['active20_ids']);assert len(active)==1242
ims={m['case_id']:m for m in index['records'] if m['source_phase']=='phase5' and m['case_id'] in active}
assert set(ims)==active
control=rows(C/'CONTROL_SOURCE_FRAME.csv');assert len(control)==1242 and len({r['case_id'] for r in control})==1242 and {r['case_id'] for r in control}==active
bs=rows(E/'B/source_frame.csv');assert len(bs)==14904
bi={(r['case_id'],r['pair'],int(r['lead']),int(r['realization']),r['variant']):(j+2,r) for j,r in enumerate(bs)}
assert len(bi)==14904
assert {(k[0],k[1],k[2],k[4]) for k in bi}=={(cid,p,h,v) for cid in active for p in ('P2_current_vs_1p5x','P1_continue_vs_stop') for h in (10,30) for v in ('range','control','stop_control')}
drawpath=BASE/'results/phase4/bootstrap_draw_matrix.npy'
assert sha(drawpath)=='d2c3ea3a3359fadf0a6d9d58420781f560a92c5c42a3c3c10d774c0e210ed5a7'
draws=np.load(drawpath,allow_pickle=False);assert draws.shape==(999,10)
logical=hashlib.sha256(np.asarray(draws,'<i8').tobytes()).hexdigest()
assert logical=='7520bcd1cd416b25f22e1e3c7ce6f3905209b837eac4d5a703a038b79f7aa30e'
rng=np.random.Generator(np.random.PCG64(SEED))
reconstructed=np.array([np.bincount(rng.choice(10,10,replace=True),minlength=10) for _ in range(999)])
assert np.array_equal(draws,reconstructed)
source=[];metadata_hash={}
for line,r in enumerate(control,2):
    cid=r['case_id'];m=ims[cid];real=int(r['realization']);assert real==m['realization'] and 0<=real<=9
    assert float(r['q_origin_m3d'])>0 and m['q_origin_m3d']>0
    assert r['operation']==m['form'] and int(r['origin_month'])==m['month']
    assert m['record_set']=='calendar20m' and m['experiment_group']=='A_distance20'
    # Reuse verified arrays, but read and preserve actual station/geometry labels.
    path=Path(r['original_tf_path']);actual=sha(path);assert actual==r['original_tf_sha256']==m['tf_sha256'];metadata_hash[str(path)]=actual
    with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['meta_json']))
    assert meta['realization']==real and meta['calendar_form']==r['operation'] and meta['origin_date']==r['origin_date'] and meta['r_m']==20
    for v in ('range','control'):
        bl,b=bi[cid,PAIR,LEAD,real,v];assert bl==int(r[v+'_source_csv_line'])
        assert b['protocol_sha256']==r['protocol_sha256']==packet['protocol_sha256']
        for k in ['E_engine_m','signed_ratio','paired_ratio_change']:assert float(b[k])==float(r[v+'_'+k])
        for k in ['original_E_engine_m','original_signed_ratio','E_true_m']:assert float(b[k])==float(r[k])
        truth=float(b['E_true_m']);assert truth!=0
        close(float(b['signed_ratio']),float(b['E_engine_m'])/truth)
        close(float(b['original_signed_ratio']),float(b['original_E_engine_m'])/truth)
        close(float(b['paired_ratio_change']),float(b['signed_ratio'])-float(b['original_signed_ratio']))
    changed=float(r['control_E_engine_m'])!=float(r['original_E_engine_m'])
    assert changed==(r['control_engine_contrast_changed']=='True')==(r['control_pumping_input_changed']=='True')
    contrast=float(r['range_signed_ratio'])-float(r['control_signed_ratio'])
    close(contrast,float(r['range_paired_ratio_change'])-float(r['control_paired_ratio_change']))
    source.append(dict(case_id=cid,pair=PAIR,lead=LEAD,realization=real,station=meta['rain_site'],sid=m['sid'],operation=r['operation'],origin_month=int(r['origin_month']),origin_date=r['origin_date'],geometry_r_m=meta['r_m'],q_origin_m3d=float(r['q_origin_m3d']),offseason_use_variant=r['offseason_use_variant'],E_true_m=float(r['E_true_m']),original_E_engine_m=float(r['original_E_engine_m']),range_E_engine_m=float(r['range_E_engine_m']),control_E_engine_m=float(r['control_E_engine_m']),original_signed_ratio=float(r['original_signed_ratio']),range_signed_ratio=float(r['range_signed_ratio']),control_signed_ratio=float(r['control_signed_ratio']),range_paired_ratio_change=float(r['range_paired_ratio_change']),control_paired_ratio_change=float(r['control_paired_ratio_change']),range_minus_control_paired_contrast=contrast,control_literal_changed=changed,control_pumping_input_changed=r['control_pumping_input_changed'],B01_all_paddy168=r['operation']=='paddy_irrigation',B02_literal_changed540=changed,control_source_csv_line=line,range_B_source_csv_line=int(r['range_source_csv_line']),control_B_source_csv_line=int(r['control_source_csv_line']),control_source_path=str(C/'CONTROL_SOURCE_FRAME.csv'),B_source_path=str(E/'B/source_frame.csv'),metadata_path=str(path),metadata_sha256=actual))
assert sum(r['B01_all_paddy168'] for r in source)==168
assert sum(r['B02_literal_changed540'] for r in source)==540
assert sum(r['control_literal_changed'] for r in source if r['B01_all_paddy168'])==162
writecsv(OUT/'SOURCE_FIRST_BATCH.csv',source)
METRICS=['range_minus_control_paired_contrast','range_paired_ratio_change','control_paired_ratio_change']
bootrows=[];jobs={};numbers={}
comments=['R1-major5','R2-major6','R3-major2']
for job,mask,label in [('B01','B01_all_paddy168','All 168 paddy-irrigation active origins'),('B02','B02_literal_changed540','540 origins conditional on literal changed control engine contrast')]:
    rs=[r for r in source if r[mask]];reals=np.array([r['realization'] for r in rs]);values=np.array([[r[k] for k in METRICS] for r in rs]);assert np.isfinite(values).all()
    samples=[]
    for b,draw in enumerate(draws):
        z=np.repeat(values,draw[reals],axis=0)
        med=np.median(z,axis=0) if len(z) else np.full(3,np.nan)
        samples.append(med)
        bootrows.append(dict(job=job,replicate=b,contributing_rows=len(z),contributing_unique_clusters=int(np.count_nonzero(draw[np.unique(reals)])),**{k:'' if not np.isfinite(med[j]) else float(med[j]) for j,k in enumerate(METRICS)}))
    samples=np.array(samples);stats={}
    for j,k in enumerate(METRICS):
        v=values[:,j];finite=samples[np.isfinite(samples[:,j]),j];lo,hi=np.percentile(finite,[2.5,97.5],method='linear');q=np.percentile(v,[0,25,50,75,100],method='linear')
        stats[k]={'median':float(np.median(v)),'ci95_lower':float(lo),'ci95_upper':float(hi),'n_identified':len(finite),'n_unidentified':999-len(finite),'distribution':dict(zip(['min','q25','median','q75','max'],map(float,q))),'positive_count':int(np.sum(v>0)),'zero_count':int(np.sum(v==0)),'negative_count':int(np.sum(v<0)),'positive_fraction':float(np.mean(v>0))}
    clusters=[]
    for real in sorted(set(reals)):
        cr=[r for r in rs if r['realization']==real];assert len({r['station'] for r in cr})==1
        clusters.append({'realization':int(real),'station':cr[0]['station'],'n':len(cr),'positive_contrast_count':sum(r[METRICS[0]]>0 for r in cr),'positive_contrast_fraction':sum(r[METRICS[0]]>0 for r in cr)/len(cr),'median_paired_contrast':float(np.median([r[METRICS[0]] for r in cr]))})
    jobs[job]={'status':'complete','comment_ids':comments,'cohort':label,'pair':PAIR,'native_lead_days':LEAD,'n':len(rs),'contributing_realization_count':len(clusters),'realization_ids':list(map(int,sorted(set(reals)))),'clusters':clusters,'station_rows':dict(collections.Counter(r['station'] for r in rs)),'operation_rows':dict(collections.Counter(r['operation'] for r in rs)),'control_unchanged_rows':sum(not r['control_literal_changed'] for r in rs),'metrics':stats,'not_difference_of_medians':True,'descriptive_not_primary':True,'median_difference_for_comparison_only':float(np.median(values[:,1])-np.median(values[:,2]))}
    for k,stat in stats.items():
        for field in ['median','ci95_lower','ci95_upper','n_identified','n_unidentified','positive_count','zero_count','negative_count','positive_fraction']:
            numbers[job+'.'+k+'.'+field]={'value':stat[field],'source_csv':str(OUT/'SOURCE_FIRST_BATCH.csv'),'cohort_mask_column':mask,'value_column':k,'summary_json_pointer':f'/jobs/{job}/metrics/{k}/{field}','bootstrap_source_csv':str(OUT/'BOOTSTRAP_FIRST_BATCH.csv'),'bootstrap_statistic':'row-weighted median after whole-realization multiplicity repetition; shared frozen draws','units':'dimensionless signed engine/truth ratio change' if field not in ['n_identified','n_unidentified','positive_count','zero_count','negative_count','positive_fraction'] else 'count or fraction'}
writecsv(OUT/'BOOTSTRAP_FIRST_BATCH.csv',bootrows)
def numeric_leaves(value,pointer):
    if isinstance(value,bool):return
    if isinstance(value,(int,float)):
        yield pointer,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from numeric_leaves(v,pointer+'/'+str(k))
    elif isinstance(value,list):
        for k,v in enumerate(value):yield from numeric_leaves(v,pointer+'/'+str(k))
for job,d in jobs.items():
    for pointer,value in numeric_leaves(d,'/jobs/'+job):
        key=pointer.removeprefix('/jobs/').replace('/','.')
        if key not in numbers:
            numbers[key]={'value':value,'summary_json_pointer':pointer,'source_csv':str(OUT/'SOURCE_FIRST_BATCH.csv'),'cohort_mask_column':'B01_all_paddy168' if job=='B01' else 'B02_literal_changed540','bootstrap_source_csv':str(OUT/'BOOTSTRAP_FIRST_BATCH.csv'),'definition':'Exact statistic, count or retained label at the supplied JSON pointer; descriptive conditional cohort.'}
primary=next(d for d in rd(E/'B/AUTHOR_CRITERION.json')['decisions'] if d['primary']);assert primary==receipt['primary'] or all(primary[k]==receipt['primary'][k] for k in primary)
summary={'turn_id':'run','jobs':jobs,'bootstrap':{'seed':SEED,'bit_generator':'PCG64','replicates':999,'draw_universe':list(range(10)),'draw_matrix_path':str(drawpath),'physical_sha256':sha(drawpath),'logical_le_i8_sha256':logical,'draw_reconstruction_verified':True,'percentile_method':'linear','interval_percentiles':[2.5,97.5],'statistic':'ordinary median over repeated paired record differences; even midpoint; shared cluster weights across all three metrics and both jobs'},'population':'Conditional on these stored active-origin twins, selected rainfall stations, calendars, layers and engine configuration. Realization IDs are bootstrap units, not independent stations; multiple origins/layers share each realization and station. No field probability or nonlinear admissible-region coverage calibration.','full_universe':{'nominal_calendar20_records':sum(m['record_set']=='calendar20m' for m in index['records']),'active_original_records':1242,'excluded_inactive_nominal_origins':sum(m['record_set']=='calendar20m' for m in index['records'])-1242,'B_source_rows':14904,'primary_rows_per_variant':1242,'source_rows_exported':len(source)},'fixed_primary_criterion_unchanged':primary,'source_sha256':baseline,'metadata_sha256':metadata_hash,'scope':{'whole_parent_completion':False,'new_hydrological_experiments':0,'engine_inference_calls':0,'W_profile_runs':0,'new_synthetic_draws':0,'statistical_bootstrap_on_stored_rows':True,'zero_face_global_natural_shape_profiling':'not run; new experiment outside B01/B02'}}
save(OUT/'SUMMARY_FIRST_BATCH.json',summary)
save(OUT/'NEW_NUMBERS_FIRST_BATCH.json',{'summary_path':str(OUT/'SUMMARY_FIRST_BATCH.json'),'values':numbers})
text=['# Stored-result paired uncertainty: B01 and B02','','All results are P2 current-versus-1.5-times-current, independent native 10-day requests at 20 m. The paired treatment-minus-control contrast is computed per record as R_range − R_control, then summarized by the row-weighted median. It is not a difference between condition medians. Intervals use the existing 999 shared realization-cluster draws (PCG64 seed 20261001); bounds are linear 2.5th/97.5th percentiles.','','| Cohort | N / realization clusters | Paired range − control median [95% CI] | Range − original median [95% CI] | Control − original median [95% CI] | Positive range − control | Contrast Q25 / Q75 |','|---|---|---|---|---|---|---|']
for j,d in jobs.items():
    s=d['metrics'];fmt=lambda k:f"{s[k]['median']:.6f} [{s[k]['ci95_lower']:.6f}, {s[k]['ci95_upper']:.6f}]"
    contrast=s[METRICS[0]]
    text.append(f"| {j}: {d['cohort']} | {d['n']} / {d['contributing_realization_count']} | {fmt(METRICS[0])} | {fmt(METRICS[1])} | {fmt(METRICS[2])} | {contrast['positive_count']}/{d['n']} ({contrast['positive_fraction']:.3%}) | {contrast['distribution']['q25']:.6f} / {contrast['distribution']['q75']:.6f} |")
text+=['','B01 retains all 168 paddy records, including the six unchanged nominal controls. B02 is conditional and descriptive: its 540 rows are selected by literal saved control effect unequal to original effect, without rounding or epsilon; the verified input-change mask coincides. Neither replaces the frozen primary 1,242-record criterion. No equivalence test follows from a small nominal median.','']
for j,d in jobs.items():
    text +=[f"## {j}: sharing across realization units",'', '| Realization | Rainfall station | Rows | Positive paired contrasts | Median paired contrast |','|---|---|---|---|---|']
    for c in d['clusters']:text.append(f"| {c['realization']} | {c['station']} | {c['n']} | {c['positive_contrast_count']} | {c['median_paired_contrast']:.6f} |")
    text +=['', 'Identified bootstrap medians: '+', '.join(f"{k}: {s['n_identified']}/999" for k,s in d['metrics'].items())+'.','']
text +=['The intervals describe variation conditional on the selected station records and simulated design. Realizations sharing a rainfall station do not become independent station samples; ten realization IDs are not a station-population calibration. These intervals do not establish nonlinear-region coverage, individual field reliability, a uniquely isolated coverage mechanism, or equivalence of the nominal control. The global zero-gain/natural-shape profiling requested elsewhere was not run. Exact values, cluster counts, source lines and hashes are supplied in SUMMARY_FIRST_BATCH.json, NEW_NUMBERS_FIRST_BATCH.json and SOURCE_FIRST_BATCH.csv.']
(OUT/'manuscript_ready_tables.md').write_text('\n'.join(text)+'\n')
current={p:sha(p) for p in baseline};assert current==baseline
for p,h in metadata_hash.items():assert sha(p)==h
outputs=[OUT/x for x in ['SOURCE_FIRST_BATCH.csv','BOOTSTRAP_FIRST_BATCH.csv','SUMMARY_FIRST_BATCH.json','NEW_NUMBERS_FIRST_BATCH.json','manuscript_ready_tables.md']]+[Path(__file__)]
final={'turn_id':'run','worker':'executor','validation_type':'source_audit','validation_claim_id':'analysis_first','validates_claim':'D09 shared paddy and changed540 paired uncertainty calculated from stored results','source_path':str(BASE/'results/phase7/D09/author instruction'),'source_sha256':sha(BASE/'results/phase7/D09/author instruction'),'audit_result':'pass','validation_result':'pass','assigned_jobs':jobs,'bootstrap':summary['bootstrap'],'whole_parent_completion':False,'scope':summary['scope'],'source_baseline_sha256':baseline,'source_current_sha256':current,'original_files_unchanged':True,'metadata_sha256':metadata_hash,'sourcechecks':{'terminal_control_receipt_current_hash_match':True,'all1242_complete_case_pair_lead_realization_join':True,'all14904_unique_B_keys':True,'active_vs_nominal_counts':summary['full_universe'],'all_primary_range_control_ratio_arithmetic_checked':True,'all1242_metadata_identity_and_station_read':True,'literal_changed_mask540_equals_verified_input_mask':True,'all168_paddy_including6_unchanged':True,'draw_matrix_reconstruction_exact':True,'fixed_primary_criterion_unchanged':True},'direct_output_verification':'Source/metadata hashes rechecked after calculations; paired CSV and bootstrap outputs will receive independent arithmetic verification before official response.','output_sha256':{str(p):sha(p) for p in outputs},'software':{'python':sys.version,'numpy':np.__version__},'reproduction_command':'python '+str(Path(__file__).resolve()),'blockers':[],'remaining_unassigned_analysis_not_claimed_complete':True}
save(OUT/'FIRST_BATCH_RECEIPT.json',final)
print(json.dumps({'jobs':{j:{'n':d['n'],'clusters':d['contributing_realization_count'],'metrics':d['metrics']} for j,d in jobs.items()},'source_checks':'pass','originals_unchanged':True},indent=2))
