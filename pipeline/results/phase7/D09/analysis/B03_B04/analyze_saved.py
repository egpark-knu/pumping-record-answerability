"""B03/B04: saved-row analysis only. Never imports simulation/engine code."""
from pathlib import Path
from collections import Counter
from datetime import date,timedelta
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[5])))
OUT=Path(__file__).resolve().parent
D09=ROOT/'results/phase7/D09'
CERT=['certified_interior','certified_constrained','bounded_by_family_only','zero_contrast','no_question']
PARAMS=dict(primary_leads=[10,30],quantiles=[10,25,50,75,90],normalizer='abs(E_true), strictly nonzero only',bootstrap_replicates=999,bootstrap_seed=20261001,bin_edge_ownership='np.digitize(SR,edges[1:-1],right=True); phase5 outside-edge labels retained',new_simulations=0,new_engine_queries=0,new_W_searches=0)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(name,x):
 (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def clean(x):
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def csv(name,rows):
 df=pd.DataFrame(rows);df.to_csv(OUT/name,index=False);assert len(pd.read_csv(OUT/name))==len(df);return df
checks=[]
def check(label,condition):
 if not bool(condition):raise AssertionError(label)
 checks.append(label)
def quant(v,prefix):
 v=np.asarray(v,float);v=v[np.isfinite(v)];r={prefix+'_n':len(v)}
 for q in PARAMS['quantiles']:r[prefix+f'_q{q}']=float(np.percentile(v,q)) if len(v) else None
 r[prefix+'_mean']=float(np.mean(v)) if len(v) else None
 return r
print('pinning inputs',flush=True)
work=load(OUT/'WORK_ITEMS.json');matrix=load(D09/'writing/ACTION_MATRIX.json');mr=load(D09/'writing/ACTION_MATRIX_RECEIPT.json')
check('matrix terminal complete',mr['component_complete'] and mr['status']=='ACTION_MATRIX_COMPLETE')
paths={Path(p) for w in work for p in w['source_paths']}
paths.update([D09/'author instruction',D09/'writing/ACTION_MATRIX.json',D09/'writing/ACTION_MATRIX_RECEIPT.json',OUT/'WORK_ITEMS.json',D09/'control/CONTROL_RECEIPT.json',D09/'analysis/FIRST_BATCH_RECEIPT.json',ROOT/'results/phase6/D08/protocol_neutral/SOURCE_INDEX.json',ROOT/'results/phase6/D08/protocol_neutral/execution/A/summary.json',ROOT/'lib/p5_analysis.py',ROOT/'lib/p5_linearized_support.py',ROOT/'lib/p4_statistics.py',ROOT/'manuscript/supplementary.md'])
baseline={str(p):sha(p) for p in sorted(paths)}
for p,h in baseline.items():
 if p in matrix['source_sha256']:check('matrix pin '+p,h==matrix['source_sha256'][p])
comments={w['id']:[] for w in work}
for c in matrix['comments']:
 for s in c.get('subactions',[]):
  for wid in s.get('work_item_ids',[]):
   if wid in comments:comments[wid].append(s['id'])
ix=load(ROOT/'results/phase6/D08/protocol_neutral/SOURCE_INDEX.json')
meta=pd.DataFrame(ix['records'])[['source_phase','case_id','record_set','realization','sid','form']]
check('index unique source/case',not meta.duplicated(['source_phase','case_id']).any())
print('reading 1,128,600 saved linearized rows',flush=True)
cols=['source_phase','case_id','realization','layer','group','distance_m','pair','lead','primary','tolerance_mode','inf','sup','W_lin','W_sampled','sampled_inf','sampled_sup','E_true','sign_linear','sign_sampled','status','linearization_failure','active_bound_labels','rank_shared','rank_J','rank_solver','rank_disagreement','no_question','endpoint_numerical_uncertainty','receipt_path','receipt_sha256','source_W_inner_envelope']
c=pd.read_csv(ROOT/'results/phase5/C/C_linearized_rows.csv',usecols=cols,float_precision='round_trip')
merge=load(ROOT/'results/phase5/C/C_MERGE_RECEIPT.json')
check('curve row count',len(c)==merge['total_rows']==1128600)
check('3 modes x 376200',c.groupby('tolerance_mode').size().to_dict()=={k:376200 for k in merge['required_modes']})
check('curve exact key uniqueness',not c.duplicated(['source_phase','case_id','pair','lead','tolerance_mode']).any())
check('primary labels',np.array_equal(c['primary'],c.lead.isin([10,30])))
check('status inventory',c.status.value_counts().to_dict()==merge['status_counts'])
c['eligible']=c.status.isin(CERT)&np.isfinite(c.W_lin)&np.isfinite(c.W_sampled)&(c.W_lin>0)&(c.W_sampled>0)
c['valid_sign']=c.sign_linear.notna()&c.sign_sampled.notna()
# Reproduce every existing log eligible and sign count, no refitting.
lookup={'lead':'lead','layer':'layer','source':'source_phase','distance':'distance_m','group':'group'}
group_indices={'overall':c.groupby(['tolerance_mode','pair']).indices}
for kind,col in lookup.items():group_indices[kind]=c.groupby(['tolerance_mode','pair',col]).indices
def subset(mode,pair,kind,value):
 key=(mode,pair) if kind=='overall' else (mode,pair,float(value) if kind in ['lead','distance'] else value)
 return c.iloc[group_indices[kind][key]]

logs=pd.read_csv(ROOT/'results/phase5/aggregation/C_loglog.csv')
for r in logs.to_dict('records'):
 g=subset(r['tolerance_mode'],r['pair'],r['stratum_kind'],r['stratum'])
 check('eligible '+str((r['tolerance_mode'],r['pair'],r['stratum_kind'],r['stratum'])),len(g)==r['inventory'] and int(g.eligible.sum())==r['positive_certified_pairs'])
signs=pd.read_csv(ROOT/'results/phase5/aggregation/C_sign_agreement.csv')
def confusion(g):
 v=g[g.valid_sign];a=v.sign_linear!=0;b=v.sign_sampled!=0
 return dict(inventory=len(g),valid=len(v),missing_or_uncertified=len(g)-len(v),both_undetermined=int((~a&~b).sum()),linear_only_determined=int((a&~b).sum()),sampled_only_determined=int((~a&b).sum()),both_determined=int((a&b).sum()),affine_cone_failures=int(g.linearization_failure.sum()),same_direction_count=int(((v.sign_linear==v.sign_sampled)&a&b).sum()))
# cache base selection so active/all do not duplicate full-frame scans
for r in signs.to_dict('records'):
 k=json.loads(r['group']);g=subset(k[0],k[1],k[2],k[3] if len(k)>3 else None)
 if r['active_only']:g=g[~g.no_question]
 z=confusion(g);check('sign inventory '+r['group']+str(r['active_only']),all(z[k]==r[k] for k in z if k in r))
print('existing aggregate counts verified',flush=True)
p=c[c.primary].copy();del c,group_indices
check('primary 75240',len(p)==75240)
p=p.merge(meta[['source_phase','case_id','record_set']],on=['source_phase','case_id'],how='left',validate='many_to_one')
check('primary record-set join',p.record_set.notna().all())
# Every case receipt pin checked; exact primary identity and values compared to receipt rows.
receipt_evidence=[]
for path,g in p.groupby('receipt_path',sort=False):
 hs=g.receipt_sha256.unique();check('receipt single pin '+path,len(hs)==1 and sha(path)==hs[0]);j=load(path)
 rr=pd.DataFrame(j['rows']);rr=rr[rr.lead.isin([10,30])]
 key=['source_phase','case_id','pair','lead','tolerance_mode']
 z=g.merge(rr[key+['W_lin','W_sampled','inf','sup','status']],on=key,suffixes=('_csv','_receipt'),validate='one_to_one')
 check('receipt primary identity '+path,len(z)==len(g)==12)
 for field in ['W_lin','W_sampled','inf','sup']:
  check('receipt field '+field+' '+path,np.allclose(z[field+'_csv'].astype(float),z[field+'_receipt'].astype(float),rtol=1e-13,atol=0,equal_nan=True))
 check('receipt status '+path,(z.status_csv==z.status_receipt).all())
 receipt_evidence.append(dict(path=path,sha256=hs[0],source_phase=g.source_phase.iloc[0],case_id=g.case_id.iloc[0],primary_rows=len(g)))
save('B03_receipt_identity_checks.json',receipt_evidence)
keys=['tolerance_mode','pair','lead','layer','distance_m','source_phase','record_set']
errors=[];flags=[]
def summarize(g):
 r=dict(rows=len(g),width_eligible=int(g.eligible.sum()),no_question=int(g.no_question.sum()),zero_linear_width=int((g.W_lin==0).sum()),zero_sampled_width=int((g.W_sampled==0).sum()),negative_linear_width=int((g.W_lin<0).sum()),negative_sampled_width=int((g.W_sampled<0).sum()),missing_width=int((~np.isfinite(g.W_lin)|~np.isfinite(g.W_sampled)).sum()),truth_zero=int((g.E_true==0).sum()),status_counts=json.dumps(g.status.value_counts().to_dict(),sort_keys=True))
 v=np.log(g.loc[g.eligible,'W_lin']/g.loc[g.eligible,'W_sampled']);r.update(quant(v,'log_width_ratio'));r.update(quant(np.exp(v),'multiplicative_width_ratio'))
 for endpoint,field,ref in [('lower','inf','sampled_inf'),('upper','sup','sampled_sup')]:
  e=g[field]-g[ref];r.update(quant(e,endpoint+'_signed_error_m'));r.update(quant(abs(e),endpoint+'_absolute_error_m'))
  n=e[g.E_true.notna()&(g.E_true!=0)]/abs(g.loc[g.E_true.notna()&(g.E_true!=0),'E_true']);r.update(quant(n,endpoint+'_signed_normalized_error'));r.update(quant(abs(n),endpoint+'_absolute_normalized_error'))
 return r
for k,g in p.groupby(keys,dropna=False,sort=True):
 ident=dict(zip(keys,k));errors.append({**ident,**summarize(g)})
 counts=Counter()
 for x in g.active_bound_labels:
  for lab in json.loads(x):counts[str(lab)]+=1
 flags.append({**ident,**confusion(g), 'no_question':int(g.no_question.sum()),'rank_disagreement':int(g.rank_disagreement.sum()),'endpoint_numerical_uncertainty':int(g.endpoint_numerical_uncertainty.sum()),'active_affine_bound_rows':int((g.active_bound_labels!='[]').sum()),'active_affine_bound_label_counts':json.dumps(dict(counts),sort_keys=True),'rank_shared_counts':json.dumps(g.rank_shared.value_counts().to_dict(),sort_keys=True),'rank_J_counts':json.dumps(g.rank_J.value_counts().to_dict(),sort_keys=True),'rank_solver_counts':json.dumps(g.rank_solver.value_counts().to_dict(),sort_keys=True),'direction_confusion_counts':json.dumps({str(k):int(v) for k,v in g.groupby(['sign_linear','sign_sampled'],dropna=False).size().items()},sort_keys=True)})
e=csv('B03_primary_error_summary.csv',errors);fl=csv('B03_classification_flags.csv',flags)
check('B03 primary denominators reconciled',int(e.rows.sum())==75240 and int(fl.inventory.sum())==75240)
# Classification consequences conditional on actual flags and bound/rank states.
flagconditional=[]
for k,g in p.groupby(['tolerance_mode','pair','lead','record_set']):
 ident=dict(zip(['tolerance_mode','pair','lead','record_set'],k))
 axes={'physical_cone_failure':g.linearization_failure,'rank_disagreement':g.rank_disagreement,'endpoint_uncertainty':g.endpoint_numerical_uncertainty,'active_affine_bounds':g.active_bound_labels!='[]','rank_shared':g.rank_shared,'rank_J':g.rank_J,'rank_solver':g.rank_solver,'no_question':g.no_question}
 for axis,values in axes.items():
  for val,idx in values.groupby(values,dropna=False).groups.items():
   z=g.loc[idx];flagconditional.append({**ident,'flag_axis':axis,'flag_value':str(val),**confusion(z),**summarize(z)})
csv('B03_flag_conditioned_classification.csv',flagconditional)
# Existing layer residuals reconcile independently with new primary summaries.
old=pd.read_csv(ROOT/'results/phase5/aggregation/C_layer_residuals.csv')
for r in old.to_dict('records'):
 g=p
 for col in ['tolerance_mode','pair','lead','layer','distance_m','source_phase']:g=g[g[col]==r[col]]
 v=np.log(g.loc[g.eligible,'W_lin']/g.loc[g.eligible,'W_sampled'])
 check('layer eligible '+str(tuple(r[k] for k in ['tolerance_mode','pair','lead','layer','distance_m','source_phase'])),len(v)==r['n_eligible'])
 if len(v):check('layer median',np.isclose(np.median(v),r['log_Wlin_over_Wsampled_median'],rtol=1e-12,atol=1e-14))
roll=[]
for k,g in p.groupby(['tolerance_mode','pair','lead']):roll.append({**dict(zip(['tolerance_mode','pair','lead'],k)),**summarize(g),**confusion(g)})
csv('B03_primary_overall.csv',roll)
# Preserve association numbers without reinterpreting R2 as equality.
csv('B03_existing_associations.csv',logs[(logs.stratum_kind=='lead')&logs.stratum.isin(['10','30'])].to_dict('records'))
print('B03 primary errors and receipt identity verified; B04 range groups',flush=True)
a=pd.read_csv(ROOT/'results/phase6/D08/protocol_neutral/execution/A/source_frame.csv',dtype={'signal_bin':str},float_precision='round_trip')
check('A 25080 exact unique rows',len(a)==25080 and not a.duplicated(['source_phase','case_id','pair','lead']).any())
check('A index key and realization',len(a.merge(meta[['source_phase','case_id','realization']],on=['source_phase','case_id','realization'],validate='many_to_one'))==25080)
draws=np.load(ROOT/'results/phase4/bootstrap_draw_matrix.npy',allow_pickle=False)
check('draw999 x10',draws.shape==(999,10) and (draws.sum(axis=1)==10).all())
check('logical draws SHA',hashlib.sha256(np.asarray(draws,dtype='<i8').tobytes()).hexdigest()=='7520bcd1cd416b25f22e1e3c7ce6f3905209b837eac4d5a703a038b79f7aa30e')
rng=np.random.Generator(np.random.PCG64(20261001))
rebuilt=np.array([np.bincount(rng.choice(range(10),size=10,replace=True),minlength=10) for _ in range(999)])
check('draw matrix reconstruction PCG64 seed20261001',np.array_equal(draws,rebuilt))
# Verify declared edge ownership from saved SR; no new bins.
def signal(sr,phase):
 if pd.isna(sr):return 'missing_signal'
 if sr<=0:return 'nonpositive_signal'
 edges=ix['signal_edges']
 if phase=='phase5' and sr<edges[0]:return 'below_D05_range'
 if phase=='phase5' and sr>edges[-1]:return 'above_D05_range'
 return str(int(np.digitize([sr],edges[1:-1],right=True)[0]))
check('all saved signal bins',all(signal(r.SR,r.source_phase)==r.signal_bin for r in a.itertuples()))
summary=load(ROOT/'results/phase6/D08/protocol_neutral/execution/A/summary.json');groups=[]
for q in summary['groups']:
 g=a[(a.record_set==q['record_set'])&(a.pair==q['pair'])&(a.lead==q['lead'])]
 if q['signal_bin']!='all_signal_bins':g=g[g.signal_bin==q['signal_bin']]
 w=q['window']
 if q['pair']=='P2_current_vs_1p5x':cats=np.where(g[w+'_increase_range_ratio'].isna(),'no_question_or_missing',np.where(g[w+'_increase_range_ratio']<=1,'within_range','outside_range'))
 else:cats=np.where(g[w+'_zero_seen'],'zero_seen','zero_absent')
 g=g[cats==q['range_category']]
 check('range group row counts',len(g)==q['rows'])
 check('ratio statuses',g.ratio_status.value_counts().to_dict()==q['ratio_status_counts'])
 check('reversal statuses',g.original_reversal_status.value_counts().to_dict()==q['reversal_status_counts'])
 for metric in ['all_rows','signed_engine_ratio','signed_reference_ratio','original_reversal']:
  z=g if metric=='all_rows' else g[g[metric].notna()]
  cnt=z.groupby('realization').size().reindex(range(10),fill_value=0).to_numpy()
  identified=int((draws@cnt>0).sum());empty=999-identified
  if metric!='all_rows':
   oldmetric=q[metric];check('range identified/empty counts',identified==oldmetric['n_identified'] and empty==oldmetric['n_unidentified'])
   vals=z[metric].astype(float)
   point=float(vals.mean() if metric=='original_reversal' else vals.median()) if len(vals) else None
   check('range saved point',point is None and oldmetric['point'] is None or point is not None and np.isclose(point,oldmetric['point'],rtol=1e-12,atol=1e-14))
  r={k:q[k] for k in ['record_set','pair','lead','window','signal_bin','range_category']}
  r.update(metric=metric,group_rows=len(g),contributing_rows=len(z),contributing_clusters=int((cnt>0).sum()),identified_draws=identified,empty_draws=empty,realization_ids=json.dumps(np.flatnonzero(cnt).tolist()),ratio_status_counts=json.dumps(q['ratio_status_counts']),reversal_status_counts=json.dumps(q['reversal_status_counts']))
  r.update({f'r{i}_rows':int(cnt[i]) for i in range(10)})
  if metric!='all_rows':r.update({k:oldmetric[k] for k in ['point','lower','upper','status']})
  groups.append(r)
gc=csv('B04_group_cluster_counts.csv',groups)
check('316 groups x4 metrics',len(gc)==316*4)
for window in ['full1024','recent365']:
 z=gc[(gc.window==window)&(gc.signal_bin=='all_signal_bins')&(gc.metric=='all_rows')];check('range aggregate rows '+window,int(z.group_rows.sum())==25080)
# All frozen estimates, all statuses; form-level rows are operation-specific.
cal=pd.read_csv(ROOT/'results/phase5/aggregation/A_map_prediction.csv',float_precision='round_trip')
csv('B04_operation_calibration.csv',cal.to_dict('records'))
check('calibration operations retained',set(cal[cal.level=='form'].stratum)=={'domestic_continuous','paddy_irrigation','water_curtain'})
# Provenance from original input metadata and exact manifest/index IDs.
calendar={r['realization']:r for r in load(ROOT/'results/pilot/phase0_checks/realization_calendar.json')}
derived={(r['source_phase'],r['case_id']):r for r in load(ROOT/'results/phase5/cases/derived_manifest.json')}
provenance=[];tfpins={}
for rec in ix['records']:
 path=Path(rec['tf_path']);h=sha(path);check('original input pin '+rec['case_id'],h==rec['tf_sha256']);tfpins[str(path)]=h
 with np.load(path,allow_pickle=False) as z:
  md=json.loads(str(z['meta_json']));dates=z['dates'];start=str(dates[0]);end=str(dates[1023]);origin=md['origin_date']
 check('calendar origin exact dates '+rec['case_id'],end==str(date.fromisoformat(origin)-timedelta(days=1)) and start==str(date.fromisoformat(origin)-timedelta(days=1024)))
 cr=calendar[rec['realization']];station=md.get('site',md.get('rain_site'))
 check('station calendar join '+rec['case_id'],station==cr['site_stem'])
 if (rec['source_phase'],rec['case_id']) in derived:
  dm=derived[rec['source_phase'],rec['case_id']];check('derived origin metadata '+rec['case_id'],origin==dm['origin_date'] and dm['realization']==rec['realization'])
 row={k:rec[k] for k in ['source_phase','case_id','record_set','realization','sid','form']}
 row.update(station=station,kma_station=cr['kma_station'],context_start=start,context_end=end,origin_date=origin,calendar_reference_origin=cr['origin_first_forecast_day'],calendar_reference_context_start=cr['context_start'],calendar_reference_context_end=cr['context_end'],tf_path=str(path),tf_sha256=h)
 row.update({k:cr[k] for k in cr if k.startswith('seed_')});provenance.append(row)
pr=csv('B04_station_window_provenance.csv',provenance)
check('provenance6270',len(pr)==6270 and len(pr.realization.unique())==10 and len(pr.station.unique())==3)
# All published map group denominators and observed frequency reconstructed by exact provenance join.
calendarrows=a.merge(pr[['source_phase','case_id','context_start','origin_date']],on=['source_phase','case_id'],validate='many_to_one')
for r in cal.to_dict('records'):
 rs='calendar20m' if r['distance_m']==20 else 'calendar200m'
 z=calendarrows[(calendarrows.record_set==rs)&(calendarrows.pair==r['pair_id'])&(calendarrows.lead==r['lead_day'])&(calendarrows.ratio_status=='ok')]
 if r['level']=='form':z=z[z.form==r['stratum']]
 elif r['level']=='storage':z=z[z.sid.str.startswith(r['stratum']+'_')]
 check('map frozen group denominator',len(z)==r['n'])
 check('map observed fraction',np.isclose((z.original_record_status=='determined').mean(),r['observed_frac'],rtol=1e-12,atol=1e-14))
check('sources unchanged',{p:sha(p) for p in baseline}==baseline)
check('input metadata unchanged',all(sha(p)==h for p,h in tfpins.items()))
save('B04_input_pins.json',tfpins)
# Published Tables S9/S10 checked directly, including display stratum offset.
text=(ROOT/'manuscript/supplementary.md').read_text()
labels={'Operating calendars, 200 m':'calendar200m','Operating calendars, 20 m':'calendar20m','Designed records':'design','Map-extension records':'map'}
categories={'Zero observed':'zero_seen','Zero absent':'zero_absent','Within range':'within_range','Outside range':'outside_range','No question or missing':'no_question_or_missing'}
published=[]
for line in text.splitlines():
 if not line.startswith('| '):continue
 cells=[v.strip() for v in line.strip().strip('|').split('|')]
 if len(cells) not in [12,14] or cells[0] not in labels or cells[1] not in ['Stop','Increase']:continue
 try:lead=int(cells[2])
 except ValueError:continue
 window='full1024' if cells[3]=='1,024 d' else 'recent365'
 category=categories.get(cells[4])
 if category is None:
  # Inspect actual saved status terminology rather than discard a row.
  if 'question' in cells[4].lower():category='no_question_or_missing'
  else:raise AssertionError('unrecognized publication category '+cells[4])
 strat='all_signal_bins' if len(cells)==12 else str(int(cells[5])-1) if cells[5].isdigit() else {'Nonpositive':'nonpositive_signal','nonpositive signal':'nonpositive_signal','Below D05 range':'below_D05_range','Above D05 range':'above_D05_range','Missing':'missing_signal'}.get(cells[5],cells[5])
 z=gc[(gc.record_set==labels[cells[0]])&(gc.pair==('P1_continue_vs_stop' if cells[1]=='Stop' else 'P2_current_vs_1p5x'))&(gc.lead==lead)&(gc.window==window)&(gc.range_category==category)&(gc.signal_bin==strat)]
 off=0 if len(cells)==12 else 1
 check('published group exact identity '+str(cells[:6]),len(z)==4)
 check('published rows',int(z.iloc[0].group_rows)==int(cells[5+off]))
 engine=z[z.metric=='signed_engine_ratio'].iloc[0];check('published defined ratios',int(engine.contributing_rows)==int(cells[6+off]))
 ids=[int(v) for v in cells[11+off].split('/')]
 check('published identified draws',ids==[int(z[z.metric==m].iloc[0].identified_draws) for m in ['signed_engine_ratio','signed_reference_ratio','original_reversal']])
 published.append(dict(table='S9' if len(cells)==12 else 'S10',record_set=labels[cells[0]],lead=lead,window=window,signal_bin=strat,range_category=category,rows=int(cells[5+off]),identified_draws=cells[11+off],verified=True))
check('all316 published groups',len(published)==316)
csv('B04_published_table_reconciliation.csv',published)
# Runtime definitions independently read, never executed/imported.
code=(ROOT/'lib/p5_linearized.py').read_text();support=(ROOT/'lib/p5_linearized_support.py').read_text();dcode=(ROOT/'results/phase6/D08/protocol_neutral/d08_runtime.py').read_text();calendarcode=(ROOT/'lib/p4_calendar.py').read_text()
import re
ct=re.search(r'^CERT_TOL\s*=\s*([^\n]+)',support,re.M).group(1)
notes=['Endpoint errors compare affine endpoints with saved sampled inner-envelope endpoints; neither is a globally certified nonlinear extremum.','Affine support active labels describe affine constraints only; nonlinear envelope bound activity is not inferred.','Existing linearization_failure is the original physical-cone flag, using CERT_TOL='+ct+'; no new epsilon introduced.','Truth normalization excludes exactly zero E_true; missing and nonfinite widths and no-question records retain separate counts.','Rank, transfer and prospective radius definitions remain conditional on oracle centre and head-conditioned bounds; no centre/radius refit.','Range group intervals retain frozen 999 multiplicities and existing identified-only conditioning; ten realization clusters drawn jointly, selected three stations are not independent sampled station replicas.','Operation calibration rows retain frozen D05 coefficients; source estimates copied without logistic refit.','Calendar origin/context dates extracted from case NPZ, not substituted by the pilot reference window; station and seeds joined by realization.','Calendar schedules, natural component, noise and seeds are shared across matched hydraulic regimes and distances; clusters describe selected station/window conditions.','Parent all1242 mechanical control decision is unchanged; conditional540 and paddy168 retain descriptive status in parent.','Tables S9/S10 reconciliation is to their originating C and D08 A group tables, retaining all signal edge/status definitions.','C04 fitted-centre and new zero-face/global natural-shape profiling are not run.','Transfer uses D=(J C).T(J C), with C the retained scaled SVD whitening inverse; lambda_max is max eig(D). Locked increment c_L=(tau_saved-SSE_min_saved)/lambda_max. Prospective c_P=gamma_pred*trace_R/lambda_max, with expected stationary residual moments. Rank-deficient null directions and bounds remain in support; memory constraint is the local tangent of t95<=1000 d.','CSV read uses pandas float_precision=round_trip to preserve original receipt endpoint binary floats; no scientific zero tolerance is added.']
for wid in ['B03','B04']:
 save(wid+'_SOURCE_RECEIPT.json',dict(job=wid,status='complete',comment_ids=comments[wid],source_sha256=baseline,parameters=PARAMS,original_files_unchanged=True,whole_parent_completion=False,checks_count=len(checks),notes=notes,sourcecode_path=str(Path(__file__).resolve()),sourcecode_sha256=sha(__file__),reproduction_command=f'python {__file__}'))
source_for={'B03_flag_conditioned_classification.csv':str(ROOT/'results/phase5/C/C_linearized_rows.csv'),'B03_primary_error_summary.csv':str(ROOT/'results/phase5/C/C_linearized_rows.csv'),'B03_classification_flags.csv':str(ROOT/'results/phase5/C/C_linearized_rows.csv'),'B03_primary_overall.csv':str(ROOT/'results/phase5/C/C_linearized_rows.csv'),'B04_group_cluster_counts.csv':str(ROOT/'results/phase6/D08/protocol_neutral/execution/A/source_frame.csv'),'B04_operation_calibration.csv':str(ROOT/'results/phase5/aggregation/A_map_prediction.csv')}
nums=[]
for file,source in source_for.items():
 df=pd.read_csv(OUT/file,float_precision='round_trip')
 for n,r in enumerate(clean(df.to_dict('records')),start=2):
  nums.append(dict(output_path=str(OUT/file),csv_line=n,source_path=source,source_sha256=baseline[source],values=r))
save('NEW_NUMBERS.json',dict(jobs=['B03','B04'],entries=nums,notes=notes))
sparse=gc[(gc.metric!='all_rows')&(gc.empty_draws>0)]
summaryout=dict(status='complete',jobs={w:dict(status='complete',comment_ids=comments[w]) for w in comments},curve_rows=1128600,primary_rows=75240,rows_by_mode={k:376200 for k in merge['required_modes']},primary_by_mode={k:25080 for k in merge['required_modes']},range_source_rows=25080,range_groups=316,range_metric_rows=1264,calibration_rows=len(cal),calibration_operation_rows=int((cal.level=='form').sum()),provenance_cases=6270,realization_clusters=10,selected_stations=sorted(pr.station.unique()),sparse_metric_groups=len(sparse),maximum_empty_draws=int(gc.empty_draws.max()),overall_primary_errors=clean(roll),parameters=PARAMS,notes=notes,source_sha256=baseline,sourcecode_sha256=sha(__file__),whole_parent_completion=False,checks_passed=len(checks),blockers=[],residual_risk=['Saved inner envelopes do not establish global extremum or nonlinear region coverage.','Realization bootstrap does not calibrate individual field probability or a new station population.'])
for name in ['SUMMARY.json','ANALYSIS_SUMMARY.json']:save(name,summaryout)
save('VERIFICATION.json',dict(status='pass',checks=checks,all_sources_unchanged=True,original_input_count=len(tfpins)))
# Compact, directly usable tables plus full exact CSV references.
md=['# B03/B04 manuscript-ready stored-result analyses','',*['- '+n for n in notes],'','## Primary-lead width and endpoint errors','',pd.DataFrame(roll)[['tolerance_mode','pair','lead','rows','width_eligible','multiplicative_width_ratio_q50','multiplicative_width_ratio_q25','multiplicative_width_ratio_q75','lower_absolute_error_m_q50','upper_absolute_error_m_q50','linear_only_determined','sampled_only_determined','affine_cone_failures']].to_markdown(index=False),'','## Frozen operation-specific calibration','',cal[cal.level=='form'][['pair_id','lead_day','distance_m','stratum','n','observed_frac','mean_pred','mean_obs_minus_pred','brier','ci_mean_obs_minus_pred_low','ci_mean_obs_minus_pred_high']].to_markdown(index=False),'','## Sparse range summaries','',f'{len(sparse)} metric/group entries have empty bootstrap draws. Complete rows, ten per-realization counts, identified/empty counts, status and original intervals are in B04_group_cluster_counts.csv.','','Every numeric output row and exact source SHA is recorded in NEW_NUMBERS.json. Complete endpoint quantiles and flags stratified by mode/layer/distance/source/record_set/pair/lead are in B03_primary_error_summary.csv and B03_classification_flags.csv. All 6270 station/window case origins and seeds are in B04_station_window_provenance.csv.']
(OUT/'manuscript_ready_tables.md').write_text('\n'.join(md)+'\n')
outputs={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['COMPONENT_RECEIPT.json','WORK_ITEMS.json']}
save('COMPONENT_RECEIPT.json',dict(validation_type='source_audit',validation_claim_id='B03_B04',validates_claim='D09 B03_B04 assigned stored-result analyses delivered',source_path=str(D09/'author instruction'),audit_result='pass',validation_result='pass',component_complete=True,whole_parent_completion=False,turn_id='run',worker='executor',jobs=summaryout['jobs'],source_sha256=baseline,output_sha256=outputs,parameters=PARAMS,original_files_unchanged=True,new_engine_queries=0,new_simulations=0,new_W_searches=0,new_experiments=0,sourcecode_path=str(Path(__file__).resolve()),sourcecode_sha256=sha(__file__),reproduction_command=f'python {__file__}',checks_passed=len(checks),blockers=[],residual_risk=summaryout['residual_risk']))
print(json.dumps({k:v for k,v in summaryout.items() if k not in ['overall_primary_errors','source_sha256','notes','parameters']},ensure_ascii=False),flush=True)
