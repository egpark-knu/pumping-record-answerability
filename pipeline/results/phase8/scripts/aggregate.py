from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,csv,hashlib,math,datetime,copy
from collections import Counter,defaultdict
import numpy as np
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3])));P=R/'results/phase8';D=R/'results/phase6/D08/protocol_neutral';sys.path.insert(0,str(R/'lib'));import p4_statistics as st
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def dump(name,obj): (P/'data'/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (P/'data'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def groups(rows,fields):
 g=defaultdict(list)
 for r in rows:g[tuple(r[k] for k in fields)].append(r)
 return [(dict(zip(fields,k)),v) for k,v in sorted(g.items(),key=lambda kv:str(kv[0]))]
def rate(rows,denfn,numfn):
 den=np.bincount([int(r['realization']) for r in rows if denfn(r)],minlength=10);num=np.bincount([int(r['realization']) for r in rows if denfn(r) and numfn(r)],minlength=10);db=M@den;nb=M@num;v=nb[db>0]/db[db>0]
 return dict(numerator=int(num.sum()),denominator=int(den.sum()),rate=float(num.sum()/den.sum()) if den.sum() else None,ci_low=float(np.percentile(v,2.5)) if len(v) else None,ci_high=float(np.percentile(v,97.5)) if len(v) else None,bootstrap_identified=len(v),bootstrap_missing=999-len(v))
def prefix(p,d):return {p+k:v for k,v in d.items()}
records=json.loads((P/'data/record_manifest.json').read_text());manifest={(r['source_phase'],r['case_id']):r for r in records};fits={}
for m in records:
 if m['active']:
  f=P/'calculations/null_fits'/f"{m['source_phase']}__{m['case_id']}.json";fits[(m['source_phase'],m['case_id'])]=json.loads(f.read_text())
assert len(fits)==4584
corrected={(r['source_phase'],r['case_id']):r for r in json.loads((P/'data/A_corrected_null_records.json').read_text())}
for key,f in fits.items():
 c=corrected[key];f['two_start_SSE_min']=f['SSE_min'];f['two_start_numerical_positive']=f['detected'];f['SSE_min']=c['best_known_null_SSE'];f['detected']=c['effect_detected'];f['feasible_null_override']=c['feasible_null_override'];f['status']='corrected_best_known_null_minimum_no_global_proof'

M=np.load(R/'results/phase4/bootstrap_draw_matrix.npy');assert M.shape==(999,10);draws=[dict(enumerate(map(int,r))) for r in M];st.validate_draws(draws)
rows=[dict(st.adapt_row(r),source_phase=r['source_phase']) for r in readcsv(R/'results/phase4/primary_rows.csv')+readcsv(R/'results/phase5/A_rows.csv')];orig={(r['source_phase'],r['case_id'],r['pair_id'],r['lead_day']):r for r in rows};assert len(orig)==25080
bv={(r['case_id'],r['variant'],r['pair'],int(r['lead'])):r for r in readcsv(D/'execution/B/source_frame.csv')}
for m in records:
 if m['variant']=='original':continue
 w=json.loads(Path(m['w_path']).read_text())['result']
 for pair in st.PAIRS:
  for lead in st.LEADS:
   r=copy.deepcopy(orig[('phase5',m['case_id'],pair,lead)]);v=bv[(m['case_id'],m['variant'],pair,lead)];e=w['envelope'][pair];r.update(source_phase=m['source_phase'],W_m=e['W'][lead-1],env_inf_m=e['inf'][lead-1],env_sup_m=e['sup'][lead-1],E_tool_m=float(v['E_engine_m']),E_reference_m=float(v['E_reference_m']) if v.get('E_reference_m') else None)
   r['sign_determined'],r['envelope_sign'],r['includes_zero']=st._sign_from_envelope(r['env_inf_m'],r['env_sup_m']);rows.append(r)
mapkeys={(m['source_phase'],m['case_id']) for m in json.loads((R/'results/phase4/COHORT_MANIFESTS.json').read_text())['map']};assert len(mapkeys)==1260
for r in rows:
 key=(r['source_phase'],r['case_id']);m=manifest[key];f=fits.get(key);r.update(record_set=m['record_set'],distance_m=m['distance_m'],active=m['active'],variant=m['variant'],null_SSE_min=None if f is None else f['SSE_min'],tau=m['tau'],detected=None if f is None else f['detected'],null_status='no_question' if f is None else f['status'],effect_detected=None if f is None else f['detected'],two_start_SSE_min=None if f is None else f['two_start_SSE_min'],two_start_numerical_positive=None if f is None else f['two_start_numerical_positive'],feasible_null_override=False if f is None else f['feasible_null_override'],is_map=key in mapkeys,old_sample_zero_excluded=r['sign_determined'],magnitude_resolved=r['W_m']<abs(r['E_true_m']) if r['E_true_m'] not in (None,0) else None)
 r['y_sign']=None if r['detected'] is None else float(r['detected'])
 expected=-1 if r['pair_id']==st.PAIRS[0] else 1;r['physical_expected_sign']=expected if r['active'] else 0
 r['engine_physically_impossible']=None if r['E_tool_m'] is None else r['E_tool_m']*expected<0
 r['reference_physically_impossible']=None if r['E_reference_m'] is None else r['E_reference_m']*expected<0
 r['engine_ratio']=None if r['E_true_m'] in (None,0) or r['E_tool_m'] is None else r['E_tool_m']/r['E_true_m'];r['reference_ratio']=None if r['E_true_m'] in (None,0) or r['E_reference_m'] is None else r['E_reference_m']/r['E_true_m']
assert len(rows)==25680
csvout('A_joined_rows.csv',rows)
recordrows=[r for r in rows if r['lead_day']==10 and r['pair_id']==st.PAIRS[0]];assert len(recordrows)==6420
concord=[]
for label,subset in [('all_existing',rows),('map1260',[r for r in rows if r['is_map']])]:
 fields=['record_set','distance_m','pair_id','lead_day'] if label=='all_existing' else ['distance_m','pair_id','lead_day']
 for k,v in groups(subset,fields):
  a=[r for r in v if r['active']];valid=[r for r in a if r['detected'] is not None and r['old_sample_zero_excluded'] is not None];both=sum(r['detected'] and r['old_sample_zero_excluded']==1 for r in valid);neither=sum(not r['detected'] and r['old_sample_zero_excluded']==0 for r in valid);oldonly=sum(not r['detected'] and r['old_sample_zero_excluded']==1 for r in valid);newonly=sum(r['detected'] and r['old_sample_zero_excluded']==0 for r in valid)
  concord.append(dict(scope=label,**k,records=len(v),active=len(a),no_question=len(v)-len(a),joint_defined=len(valid),both_detected=both,both_not_detected=neither,old_only=oldonly,null_only=newonly,agreement=both+neither,agreement_rate=(both+neither)/len(valid) if valid else None,new_detected=sum(r['detected'] is True for r in a),old_detected=sum(r['old_sample_zero_excluded']==1 for r in a),numerical_missing=len(a)-len(valid)))
csvout('A_concordance.csv',concord)
lookup={(r['source_phase'],r['case_id'],r['pair_id'],r['lead_day']):r for r in rows};discord=[]
for r in rows:
 if r['lead_day']!=10:continue
 q=lookup[(r['source_phase'],r['case_id'],r['pair_id'],30)]
 if r['old_sample_zero_excluded']!=q['old_sample_zero_excluded']:discord.append(dict(source_phase=r['source_phase'],case_id=r['case_id'],record_set=r['record_set'],distance_m=r['distance_m'],pair_id=r['pair_id'],is_map=r['is_map'],old10=r['old_sample_zero_excluded'],old30=q['old_sample_zero_excluded'],null_detected=r['detected'],SSE_min=r['null_SSE_min'],tau=r['tau'],inf10=r['env_inf_m'],sup10=r['env_sup_m'],inf30=q['env_inf_m'],sup30=q['env_sup_m']))
csvout('A_historical_lead_discordance.csv',discord)
startrows=[]
for key,f in fits.items():
 for s in f['starts']:
  b=s['best_feasible'];startrows.append(dict(source_phase=key[0],case_id=key[1],start_number=s['start_number'],tau=f['tau'],effect_detected=f['detected'],two_start_numerical_positive=f['two_start_numerical_positive'],success=s['optimizer']['success'],status=s['optimizer']['status'],message=s['optimizer']['message'],nfev=s['optimizer']['nfev'],evaluations=s['evaluations_including_initial'],infeasible_evaluations=s['infeasible_evaluations'],error_evaluations=s['error_evaluations'],returned_objective=s['optimizer']['returned_objective'],best_sse=None if b is None else b['sse'],best_t95=None if b is None else b['t95'],start_x=json.dumps(s['start_x']),best_x=json.dumps(None if b is None else b['x']),linear_coefficients=json.dumps(None if b is None else b['linear_coefficients'])))
csvout('A_start_diagnostics.csv',startrows)
cal=[r for r in recordrows if r['record_set'] in ('calendar200m','calendar20m')];assert len(cal)==4320;calout=[]
for fields in [['distance_m'],['distance_m','storage_type'],['distance_m','layer_id'],['distance_m','calendar_type'],['distance_m','origin_month'],['distance_m','calendar_type','origin_month','layer_id'],['distance_m','calendar_type','origin_month','storage_type']]:
 for k,v in groups(cal,fields):calout.append(dict(stratum='+'.join(fields),**k,scheduled=len(v),active=sum(r['active'] for r in v),no_question=sum(not r['active'] for r in v),**prefix('all_',rate(v,lambda r:True,lambda r:r['detected'] is True)),**prefix('active_',rate(v,lambda r:r['active'],lambda r:r['detected'] is True))))
csvout('A_calendar_detection.csv',calout)
trans=[]
for r in cal:
 if r['distance_m']!=20:continue
 q=lookup[('phase4',r['twin_case_id'],st.PAIRS[0],10)];assert r['active']==q['active'];trans.append(dict(case_id20=r['case_id'],case_id200=q['case_id'],realization=r['realization'],storage_type=r['storage_type'],layer_id=r['layer_id'],calendar_type=r['calendar_type'],origin_month=r['origin_month'],active=r['active'],detected20=r['detected'],detected200=q['detected'],transition='no_question' if not r['active'] else 'missing' if r['detected'] is None or q['detected'] is None else 'both_detected' if r['detected'] and q['detected'] else 'new_at20' if r['detected'] else 'lost_at20' if q['detected'] else 'neither_detected'))
csvout('A_calendar_200_20_transitions.csv',trans)
trsum=[]
for fields in [[],['storage_type'],['layer_id'],['calendar_type']]:
 for k,v in groups(trans,fields):trsum.append(dict(stratum='+'.join(fields) or 'all',**k,records=len(v),**Counter(r['transition'] for r in v)))
csvout('A_calendar_transition_summary.csv',trsum)
eng=[]
for scope,subset in [('all_existing',rows),('map1260',[r for r in rows if r['is_map']])]:
 fields=['record_set','distance_m','pair_id','lead_day'] if scope=='all_existing' else ['distance_m','pair_id','lead_day']
 for k,v in groups(subset,fields):
  rr=dict(scope=scope,**k,active=sum(r['active'] for r in v),detected=sum(r['detected'] is True for r in v),**prefix('engine_impossible_',rate(v,lambda r:r['detected'] is True and r['E_tool_m'] is not None,lambda r:r['engine_physically_impossible'])),**prefix('reference_impossible_',rate(v,lambda r:r['detected'] is True and r['E_reference_m'] is not None,lambda r:r['reference_physically_impossible'])),engine_zero=sum(r['detected'] is True and r['E_tool_m']==0 for r in v),engine_missing=sum(r['detected'] is True and r['E_tool_m'] is None for r in v))
  for tag in ['engine','reference']:
   a=[r[tag+'_ratio'] for r in v if r['detected'] is True and r[tag+'_ratio'] is not None]
   for stat,fn in [('min',min),('median',np.median),('max',max),('mean',np.mean)]:rr[tag+'_ratio_'+stat]=float(fn(a)) if a else None
  eng.append(rr)
csvout('A_engine_on_detected.csv',eng)
mag=[]
for fields in [['record_set','distance_m','pair_id','lead_day'],['record_set','distance_m','storage_type','pair_id','lead_day']]:
 for k,v in groups(rows,fields):mag.append(dict(stratum='+'.join(fields),**k,**prefix('magnitude_',rate(v,lambda r:r['active'] and r['magnitude_resolved'] is not None,lambda r:r['magnitude_resolved']))))
csvout('A_magnitude_unchanged.csv',mag)
maprows=[r for r in recordrows if r['is_map']];assert len(maprows)==1260 and all(r['y_sign'] is not None for r in maprows)
old=json.loads((R/'results/phase4/statistics.json').read_text());dump('magnitude_models_UNCHANGED.json',old['cohort_A']['models']['magnitude'])
def mapfit(sample,label):
 models={};full={};replicates={};exchange={};flat=[]
 for name,layers in [('no_layer',False),('layer',True)]:
  f=st._fit_rows(sample,'y_sign',layers);reps=[st._fit_rows(st.apply_draw(sample,d),'y_sign',layers,sample_weight=True) for d in draws];tree=st._crossings_for(f,layers);trees=[st._crossings_for(q,layers) for q in reps]
  for rho,layer,p,point in st._walk_points(tree,layers):
   rep=[t[rho][layer][p] if layers else t[rho][p] for t in trees];point['bootstrap']=st._bootstrap_payload(point,rep);flat.append(dict(model=label,layer_model=name,layer=layer,**{k:v for k,v in point.items() if k!='bootstrap'},**prefix('bootstrap_',point['bootstrap'])))
  models[name]={k:f.get(k) for k in ['status','coefficients','n','n_events','log_likelihood','large_coefficient_diagnostic']};models[name].update(crossings=tree,bootstrap_fit_status_counts=dict(Counter(q['status'] for q in reps)))
  full[name]=f;replicates[name]=reps
  if not layers:
   def ex(q):
    if q.get('status')!='ok' or not q.get('coefficients'):return None
    c=q['coefficients'];z=c['log_rho_plus_offset']/c['log_signal_ratio'];return dict(coefficient_exchange=z,SR4_over_SR0=math.exp(-z*math.log(4.05/.05)))
   point=ex(f);ee=[ex(q) for q in reps];exchange=dict(status=f['status'],point=point,bootstrap={k:st.percentile_interval([e[k] for e in ee if e is not None],sum(e is None for e in ee)) for k in ['coefficient_exchange','SR4_over_SR0']})
 comparison=st._comparison_with_deltas(full['no_layer'],full['layer'],st._null_log_likelihood(sample,'y_sign'));cb=[st._comparison_with_deltas(a,b,st._weighted_null(sample,'y_sign',draws[i])) for i,(a,b) in enumerate(zip(replicates['no_layer'],replicates['layer']))];comparison['bootstrap']={metric:st.percentile_interval([q[metric] for q in cb if q.get(metric) is not None],sum(q.get(metric) is None for q in cb)) for metric in ['lr_stat','delta_mcfadden']};comparison['bootstrap_status_counts']=dict(Counter(q['status'] for q in cb));models['layer_comparison']=comparison
 return models,exchange,flat
print('aggregate joins complete; map fitting',flush=True)
cache=P/'data/A_detection_map.json'
map_input_sha256=hashlib.sha256(json.dumps([(r['source_phase'],r['case_id'],r['y_sign'],r['SR'],r['rho'],r['layer_id'],r['realization']) for r in maprows],sort_keys=True).encode()).hexdigest()
if cache.exists() and json.loads(cache.read_text()).get('map_input_sha256')==map_input_sha256:
 c=json.loads(cache.read_text());assert c['n']==len(maprows) and c['primary']['no_layer']['n_events']==sum(r['y_sign'] for r in maprows) and c['bootstrap_draw_sha256']==sha(R/'results/phase4/bootstrap_draw_matrix.npy')
 model,exchange,sensitivity,sexchange=c['primary'],c['exchange'],c['positive_rho_sensitivity'],c['positive_rho_exchange'];print('reuse completed identical map/999 bootstrap results',flush=True)
else:
 model,exchange,flat=mapfit(maprows,'primary');sensitivity,sexchange,sflat=mapfit([r for r in maprows if r['rho']>0],'positive_rho_sensitivity');dump('A_detection_map.json',dict(primary=model,positive_rho_sensitivity=sensitivity,exchange=exchange,positive_rho_exchange=sexchange,bootstrap_draw_sha256=sha(R/'results/phase4/bootstrap_draw_matrix.npy'),lead_independent=True,pair_independent=True,n=1260,map_input_sha256=map_input_sha256));csvout('A_detection_thresholds.csv',flat+sflat)
assert len(trans)==2160 and sum(r['active'] for r in trans)==1242
table=[]
for rho in st.EVAL_RHO:
 k=st._rho_key(rho);a=model['no_layer']['crossings'][k];mag10=old['cohort_A']['models']['magnitude']['10'][st.PAIRS[0]]['no_layer']['crossings'][k];mag30=old['cohort_A']['models']['magnitude']['30'][st.PAIRS[0]]['no_layer']['crossings'][k]
 table.append(dict(rho=rho,**{f'detection_p{p}':a[str(p)]['signal_ratio'] for p in [.5,.9]},**{f'detection_p{p}_status':a[str(p)]['status'] for p in [.5,.9]},**{f'detection_p{p}_CI_{end}':a[str(p)]['bootstrap'][end] for p in [.5,.9] for end in ['low','high']},magnitude10=mag10['signal_ratio'],magnitude30=mag30['signal_ratio'],magnitude10_status=mag10['status'],magnitude30_status=mag30['status'],detection_p05_support=a['0.5']['reporting_status'],detection_p09_support=a['0.9']['reporting_status']))
csvout('Table3_D10.csv',table)
# One internal verification pass: identities, exact stored bounds/width/magnitude, starts, original immutable pins.
fr=json.loads((P/'PROTOCOL_FREEZE_RECEIPT.json').read_text())
for p,h in fr['source_pins'].items():assert sha(p)==h
for m in records:
 assert sha(m['w_path'])==m['w_sha256']
 w=json.loads(Path(m['w_path']).read_text());w=w if m['variant']=='original' else w['result']
 for pair in st.PAIRS:
  for lead in st.LEADS:
   r=lookup[(m['source_phase'],m['case_id'],pair,lead)];e=w['envelope'][pair]
   assert r['env_inf_m']==e['inf'][lead-1] and r['env_sup_m']==e['sup'][lead-1] and r['W_m']==e['W'][lead-1]
 if m['active']:
  f=fits[(m['source_phase'],m['case_id'])];assert len(f['starts'])==2 and f['tau']==m['tau'];assert f['detected'] is None or f['detected']==(f['SSE_min']>m['tau']);assert all(s['optimizer']['nfev']<=400 for s in f['starts'])
  for pair in st.PAIRS:
   for lead in st.LEADS:assert lookup[(m['source_phase'],m['case_id'],pair,lead)]['detected']==f['detected']
  assert f['SSE_min']<=f['two_start_SSE_min']
  c=corrected[(m['source_phase'],m['case_id'])]
  if c['minimum_valid_saved_null_sse'] is not None:assert f['SSE_min']<=c['minimum_valid_saved_null_sse']
  if c['minimum_valid_saved_null_sse'] is not None and c['minimum_valid_saved_null_sse']<=m['tau']:assert f['detected'] is False
# Audit only reused D09 evidence; no rerun of old studies.
audit={}
# Public edition: record the hashes of the reused supporting-analysis outputs that are present (no run-receipt comparison).
for name,files in [('B02',['ANALYSIS_SUMMARY.json','B02_source_rows.csv','B02_lead_discordance.csv']),('B03_B04',['ANALYSIS_SUMMARY.json','NEW_NUMBERS.json'])]:
 for fn in files:
  p=R/f'results/phase7/D09/analysis/{name}'/fn
  if p.is_file():audit[str(p.relative_to(R))]=sha(p)
summary=dict(status='A_corrected_detection_complete',records=6420,original_records=6270,active=4584,original_active=4434,starts=len(startrows),null_detected=sum(f['detected'] is True for f in fits.values()),original_null_detected=sum(f['detected'] is True for key,f in fits.items() if not key[0].startswith('phase6_D08')),two_start_numerical_positive=sum(f['two_start_numerical_positive'] is True for f in fits.values()),feasible_null_overrides=sum(f['feasible_null_override'] for f in fits.values()),missing_minima=sum(f['detected'] is None for f in fits.values()),start_status_counts=dict(Counter(str(s['status']) for s in startrows)),start_success_count=sum(s['success'] for s in startrows),error_evaluations=sum(s['error_evaluations'] for s in startrows),concordance=concord,Table3=table,exchange=exchange,calendar_overall=[r for r in calout if r['stratum']=='distance_m'],calendar_transitions=trsum,engine=eng,historical_discordant_unique_records=len(set((r['source_phase'],r['case_id']) for r in discord)),historical_discordant_map=len(set((r['source_phase'],r['case_id']) for r in discord if r['is_map'])),source_audit='D10 calc component delivered',reused_D09_hashes=audit,verification='one internal pass: exact input/output joins, unchanged W and tau, 2 starts/all active, <=400 evaluations/start, lead/pair identity, immutable source pins; passed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
dump('A_SUMMARY.json',summary);dump('A_INTERNAL_VERIFICATION.json',dict(status='passed',join_rows=25680,records=6420,active=4584,starts=len(startrows),errors=sum(s['error_evaluations'] for s in startrows),corrected_known_feasible_null_detected=0,feasible_null_overrides=sum(f['feasible_null_override'] for f in fits.values()),source_audit=audit));print(json.dumps({k:summary[k] for k in ['status','records','active','starts','null_detected','missing_minima','start_status_counts','error_evaluations','historical_discordant_unique_records','historical_discordant_map']}),flush=True)
