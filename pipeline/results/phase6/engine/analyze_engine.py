from pathlib import Path
import numpy as np,json,csv,hashlib,math,random,collections,datetime
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3])));E=R/'results/phase6/engine';RE=E/'replay';RE.mkdir(exist_ok=True)
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ah(a):a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def save(f,d):Path(f).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def finite(v):
 try:x=float(v);return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def writecsv(f,rows):
 with Path(f).open('w',newline='') as s:
  w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
protocol=json.loads((E/'PROTOCOL.json').read_text());sources={str(E/'PROTOCOL.json'):sha(E/'PROTOCOL.json')};rows=[]
stats_path=R/'results/phase4/statistics.json';stats=json.loads(stats_path.read_text());edges=stats['cohort_C']['bins']['edges'];sources[str(stats_path)]=sha(stats_path)
for cohort,file in [('old4110',R/'results/phase4/primary_rows.csv'),('new2160',R/'results/phase5/A_rows.csv')]:
 sources[str(file)]=sha(file)
 with file.open() as s:
  for line,x in enumerate(csv.DictReader(s),2):
   sr=finite(x['SR']); inf=finite(x['env_inf_m']);sup=finite(x['env_sup_m']);et=finite(x['E_tool_m']);truth=finite(x['E_true_m']);noq=x.get('no_active_contrast','').lower()=='true'
   envsign=None if inf is None or sup is None else (1 if inf>0 else -1 if sup<0 else 0)
   status='no_question' if noq else 'envelope_missing' if envsign is None else 'undetermined' if envsign==0 else 'engine_missing' if et is None else 'engine_zero' if et==0 else 'opposite' if et*envsign<0 else 'correct'
   truthstatus='truth_missing' if truth is None else 'truth_zero' if truth==0 else 'engine_missing' if et is None else 'engine_zero' if et==0 else 'truth_opposite' if et*truth<0 else 'truth_correct'
   bin_='missing_signal' if sr is None else 'nonpositive_signal' if sr<=0 else 'below_D05_range' if cohort=='new2160' and sr<edges[0] else 'above_D05_range' if cohort=='new2160' and sr>edges[-1] else str(int(np.digitize([sr],edges[1:-1],right=True)[0]))
   rows.append(dict(row_id=f"{x['source_phase']}::{x['case_id']}::{x['pair_id']}::H{x['lead_day']}",cohort=cohort,source_phase=x['source_phase'],case_id=x['case_id'],pair=x['pair_id'],lead=int(x['lead_day']),signal_bin=bin_,SR=sr,record_sign_status=status,truth_sign_status=truthstatus,envelope_sign=envsign,E_engine_m=et,E_true_m=truth,storage=x['storage_type'],source_csv=str(file),source_csv_line=line,tool_status=x['tool_status'],truth_status=x['truth_status'],W_status=x['W_status']))
assert len(rows)==25080 and len({x['row_id'] for x in rows})==25080
assert len({x['case_id'] for x in rows if x['cohort']=='old4110'})==4110
assert len({x['case_id'] for x in rows if x['cohort']=='new2160'})==2160
rng=random.Random(20261001);selected=[];availability=[]
for status in ['opposite','correct']:
 for cohort,lead,quota in [('old4110',10,3),('old4110',30,2),('new2160',10,3),('new2160',30,2)]:
  pool=[x for x in rows if x['cohort']==cohort and x['lead']==lead and x['pair']=='P2_current_vs_1p5x' and x['record_sign_status']==status]; chosen=[]
  for bin_ in ['4','above_D05_range','3','2','1','0','below_D05_range']:
   cand=sorted([x for x in pool if x['signal_bin']==bin_],key=lambda x:x['row_id']);rng.shuffle(cand);chosen+=cand[:quota-len(chosen)]
   if len(chosen)==quota:break
  assert len(chosen)==quota,(cohort,lead,status,len(pool));selected+=chosen;availability.append(dict(cohort=cohort,lead=lead,status=status,available=len(pool),highest_bin_available=sum(x['signal_bin']=='4' for x in pool),selected_row_ids=[x['row_id'] for x in chosen]))
assert len(selected)==20 and len({x['row_id'] for x in selected})==20
selected=[dict(x) for x in selected]
save(RE/'SELECTION_PRE_INPUT.json',{'protocol_sha256':sha(E/'PROTOCOL.json'),'selected':selected,'availability':availability,'magnitude_width_replay_outcomes_not_used':True})
# Direct archived forecast mapping and batch provenance; no new inference.
archives={}; lookup={};batch_defs={};raw_schema={}
for phase in [2,3,4,5]:
 root=R/f'results/phase{phase}/tools'; paths=sorted(root.glob('timesfm_raw_H*.npz')) if phase in [2,3] else sorted(root.glob('*/timesfm_raw_H*.npz'))
 for f in paths:
  sources[str(f)]=sha(f)
  with np.load(f,allow_pickle=False) as z:arc={k:z[k] for k in ['quantiles','pair_id','query_id','schedule_id','case_id','metadata_json','provenance_json','quantile_levels']}
  H=arc['quantiles'].shape[1];assert H in [10,30] and np.array_equal(arc['quantile_levels'],np.arange(1,10)/10)
  assert np.isfinite(arc['quantiles']).all() and not (np.diff(arc['quantiles'],axis=2)<0).any()
  raw_schema[str(f)]={'keys':list(arc),'quantiles_shape':list(arc['quantiles'].shape),'provenance':json.loads(str(arc['provenance_json']))}
  archives[str(f)]=arc
  for i,pid in enumerate(arc['pair_id']):
   key=(f'phase{phase}',str(pid),H);assert key not in lookup;lookup[key]=(str(f),2*i)
for x in selected:
 arcpath,index=lookup[x['source_phase'],x['case_id']+'__'+x['pair'],x['lead']];arc=archives[arcpath];start=(index//8)*8;stop=min(start+8,len(arc['query_id']));assert index+1<stop
 key=(arcpath,start,stop);batchid='batch_'+hashlib.sha256(repr(key).encode()).hexdigest()[:16]
 x.update(archive_path=arcpath,archive_pair_query_index=index,batch_id=batchid,batch_local_index=index-start,query_ids=[str(arc['query_id'][index]),str(arc['query_id'][index+1])])
 batch_defs[batchid]={'archive_path':arcpath,'source_phase':x['source_phase'],'horizon':x['lead'],'start':start,'stop':stop,'query_ids':arc['query_id'][start:stop].tolist(),'original_provenance':raw_schema[arcpath]['provenance']}
# Preserve original batch membership, ordering and request bytes. Only selected batches copied.
for phase in sorted({x['source_phase'] for x in selected}):
 for H in sorted({x['lead'] for x in selected if x['source_phase']==phase}):
  f=R/'results'/phase/'cases'/f'tool_inputs_H{H}.npz';sources[str(f)]=sha(f)
  with np.load(f,allow_pickle=False) as z:data={k:z[k] for k in ['query_id','head','pumping','rainfall','metadata_json','protocol_sha256']}
  queryindex={str(q):i for i,q in enumerate(data['query_id'])}
  for bid,b in batch_defs.items():
   if b['source_phase']!=phase or b['horizon']!=H:continue
   ix=[queryindex[q] for q in b['query_ids']];context=data['head'][ix].astype(np.float32);cov=np.stack([data['pumping'][ix],data['rainfall'][ix]],axis=1).astype(np.float32);arc=archives[b['archive_path']]
   assert np.array_equal(data['metadata_json'][ix],arc['metadata_json'][b['start']:b['stop']])
   file=RE/f'{bid}_inputs.npz';np.savez_compressed(file,contexts=context,covariates=cov,query_id=np.array(b['query_ids']),head_original=data['head'][ix],pumping_original=data['pumping'][ix],rainfall_original=data['rainfall'][ix],metadata_json=data['metadata_json'][ix],original_quantiles=arc['quantiles'][b['start']:b['stop']],horizon=H)
   b.update(input_source_path=str(f),input_source_sha256=sources[str(f)],input_archive_path=str(file),input_archive_sha256=sha(file),contexts_sha256=ah(context),covariates_sha256=ah(cov))
   print('INPUT_SNAPSHOT',bid,H,len(ix),flush=True)
for x in selected:
 b=batch_defs[x['batch_id']];x.update(input_archive_path=b['input_archive_path'],input_archive_sha256=b['input_archive_sha256'],contexts_sha256=b['contexts_sha256'],covariates_sha256=b['covariates_sha256'])
save(RE/'SELECTION.json',{'protocol_sha256':sha(E/'PROTOCOL.json'),'selected':selected,'availability':availability,'batches':batch_defs,'fresh_outcomes_not_used':True});save(E/'ARCHIVED_SCHEMA.json',raw_schema)
# Only now calculate new magnitude and actual same-request forecast-head-width ratios.
for x in rows:
 f,index=lookup[x['source_phase'],x['case_id']+'__'+x['pair'],x['lead']];arc=archives[f];q=arc['quantiles'];a=q[index,-1];b=q[index+1,-1];effect=float(a[4]-b[4]);assert effect==x['E_engine_m'],(x['row_id'],effect,x['E_engine_m'])
 wa=float(a[8]-a[0]);wb=float(b[8]-b[0]);primary=max(wa,wb);et=x['E_engine_m'];truth=x['E_true_m']
 x.update(archive_path=f,archive_sha256=sources[f],query_id_a=str(arc['query_id'][index]),query_id_b=str(arc['query_id'][index+1]),q10_a_m=float(a[0]),q50_a_m=float(a[4]),q90_a_m=float(a[8]),q10_b_m=float(b[0]),q50_b_m=float(b[4]),q90_b_m=float(b[8]),width_a_m=wa,width_b_m=wb,width_primary_max_m=primary,truth_ratio_status='ok' if truth is not None and truth!=0 and et is not None else 'undefined_zero_or_missing_truth',width_status='ok' if wa>0 and wb>0 else 'nonpositive_width',abs_engine_over_truth=abs(et)/abs(truth) if truth is not None and truth!=0 and et is not None else None,abs_engine_over_width_a=abs(et)/wa if wa>0 and et is not None else None,abs_engine_over_width_b=abs(et)/wb if wb>0 and et is not None else None,abs_engine_over_width_primary=abs(et)/primary if primary>0 and et is not None else None)
writecsv(E/'ENGINE_ROBUSTNESS_ROWS.csv',rows)
def desc(values):
 v=np.array([x for x in values if x is not None],float)
 if len(v)==0:return dict(n=0,min=None,q10=None,median=None,q90=None,max=None)
 q=np.quantile(v,[0,.1,.5,.9,1],method='linear');return dict(n=len(v),min=float(q[0]),q10=float(q[1]),median=float(q[2]),q90=float(q[3]),max=float(q[4]))
metrics=['abs_engine_over_truth','abs_engine_over_width_a','abs_engine_over_width_b','abs_engine_over_width_primary']; summaries=[];groups=collections.defaultdict(list)
for x in rows:
 for scope in ['all_bins',x['signal_bin']]:groups[x['cohort'],x['pair'],x['lead'],scope,x['record_sign_status']].append(x)
for (cohort,pair,lead,bin_,status),rs in sorted(groups.items()):
 summaries.append(dict(cohort=cohort,pair=pair,lead=lead,signal_bin=bin_,status=status,rows=len(rs),metrics={m:desc([x[m] for x in rs]) for m in metrics},truth_sign_status_counts=dict(collections.Counter(x['truth_sign_status'] for x in rs)),undefined_truth_ratios=sum(x['abs_engine_over_truth'] is None for x in rs),undefined_width_ratios=sum(x['abs_engine_over_width_primary'] is None for x in rs)))
flat=[dict(cohort=s['cohort'],pair=s['pair'],lead=s['lead'],signal_bin=s['signal_bin'],status=s['status'],row_count=s['rows'],metric=m,**d) for s in summaries for m,d in s['metrics'].items()];writecsv(E/'ENGINE_DISTRIBUTIONS.csv',flat)
validation=[]
for lead in [10,30]:
 for pair in ['P1_continue_vs_stop','P2_current_vs_1p5x']:
  for bin_ in range(5):
   rs=[x for x in rows if x['cohort']=='old4110' and x['lead']==lead and x['pair']==pair and x['signal_bin']==str(bin_)];opp=sum(x['record_sign_status']=='opposite' for x in rs);den=sum(x['record_sign_status'] in ['correct','opposite','engine_zero'] for x in rs)
   saved=[x for x in stats['cohort_C']['table']['by_bin_storage'] if x['pair_id']==pair and x['lead_day']==lead and x['bin']==bin_];savedopp=sum(x['tool_contradiction']['numerator'] for x in saved);saveden=sum(x['tool_contradiction']['denominator'] for x in saved)
   assert (opp,den)==(savedopp,saveden),(lead,pair,bin_,opp,den,savedopp,saveden);validation.append(dict(lead=lead,pair=pair,bin=bin_,opposite=opp,denominator=den,rate=opp/den if den else None))
for f in [R/'lib/p3_phase2_timesfm.py',R/'lib/p3_phase2_timesfm_execution.py',R/'lib/p3_phase3_timesfm_execution.py',R/'lib/p4_export.py',R/'lib/p5_export.py',R/'lib/p4_timesfm_execution.py',R/'lib/p5_timesfm_execution.py',R/'lib/timesfm_pilot_adapter.py',R/'results/phase5/protocol.md',R/'results/phase5/protocol.sha256',R/'results/phase5/protocol_freeze.json',R/'results/phase5/FINAL_SOURCE_AUDIT.json',R/'results/phase2/A_preflight_manifest.json']:
 if f.is_file():sources[str(f)]=sha(f)
save(E/'SOURCE_SHA_BEFORE.json',sources);save(E/'ENGINE_ROBUSTNESS_SUMMARY.json',{'status':'archived_distributions_complete_replay_pending','protocol_sha256':sha(E/'PROTOCOL.json'),'cohort_counts':{'old4110':16440,'new2160':8640},'rows':len(rows),'case_count':6270,'signal_bin_edges':edges,'all_archived_quantiles_saved':True,'all_engine_CSV_endpoints_exactly_match_archived_q50_subtraction':True,'Figure5_old_bins_exact_numerator_denominator_match':validation,'summaries':summaries,'width_warning':'forecast-head width, not calibrated effect confidence interval','replay_selection':str(RE/'SELECTION.json')});print('DISTRIBUTIONS_COMPLETE',len(rows),'batches',len(batch_defs),flush=True)
