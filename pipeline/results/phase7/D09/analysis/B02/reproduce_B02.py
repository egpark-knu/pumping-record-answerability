#!/usr/bin/env python3
"""B02: immutable stored-result extraction only. No engine imports/calls or random draws."""
import sys
sys.dont_write_bytecode=True
import csv,json,hashlib,math,platform
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[5])))
OUT=ROOT/'results/phase7/D09/analysis/B02'
D09=ROOT/'results/phase7/D09'
PAIRS=('P1_continue_vs_stop','P2_current_vs_1p5x')
HASH={}
def sha(p):
 p=Path(p); h=hashlib.sha256(p.read_bytes()).hexdigest();HASH[str(p)]=h;return h
def load(p):
 p=Path(p);raw=p.read_bytes();HASH[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def dump(name,d): (OUT/name).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(name,rows):
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def f(v):
 if v in (None,''):return None
 x=float(v);return x if math.isfinite(x) else None
def truth(v):return v is True or str(v).lower()=='true'
def rows(p):
 sha(p)
 with Path(p).open(newline='') as h:return list(csv.DictReader(h))
raw=load(D09/'writing/ACTION_MATRIX_RECEIPT.json'); matrix=load(D09/'writing/ACTION_MATRIX.json')
assert raw['status']=='ACTION_MATRIX_COMPLETE' and raw['component_complete'] and raw['validation_result']=='pass'
assert HASH[str(D09/'writing/ACTION_MATRIX.json')]==raw['matrix_artifact_sha256'][str(D09/'writing/ACTION_MATRIX.json')]
items=load(OUT/'WORK_ITEMS.json');assert [i['id'] for i in items]==['B02']
for p in [D09/'author instruction',D09/'control/CONTROL_RECEIPT.json',D09/'analysis/FIRST_BATCH_RECEIPT.json']:
 if p.suffix=='.json':load(p)
 else:sha(p)
for p in items[0]['source_paths']:sha(p)
for p in ('lib/p4_collect.py','lib/p4_contracts.py','lib/p3_kernels.py'):sha(ROOT/p)
for p,h in list(HASH.items()):
 if p in raw['source_sha256']:assert h==raw['source_sha256'][p],('dispatch input SHA mismatch',p)
# Safe statistics-only import: no envelope/scoring modules or source writes.
sys.path.insert(0,str(ROOT/'lib'))
from p4_statistics import build_cohort_manifest
man={ph:load(ROOT/'results'/ph/'cases/derived_manifest.json') for ph in ('phase2','phase3','phase4','phase5')}
cohort=build_cohort_manifest(man['phase2'],man['phase3'],man['phase4'])
original=load(ROOT/'results/phase4/COHORT_MANIFESTS.json')
print('Original manifest keys',list(original),flush=True)
orig_map=original['map'];assert {(r['source_phase'],r['case_id']) for r in orig_map}=={(r['source_phase'],r['case_id']) for r in cohort['records'] if r['cohort_A']}
mi={(ph,r['case_id']):r for ph,mm in man.items() for r in mm}
ci={(r['source_phase'],r['case_id']):r for r in cohort['records']}
index=load(ROOT/'results/phase6/D08/protocol_neutral/SOURCE_INDEX.json');si={(r['source_phase'],r['case_id']):r for r in index['records']};assert len(si)==len(index['records'])
primary=rows(ROOT/'results/phase4/primary_rows.csv');a20=rows(ROOT/'results/phase5/A_rows.csv')
assert len(primary)==4110*4 and len(a20)==2160*4
selected=[]
for r in primary:
 k=(r['source_phase'],r['case_id']);c=ci[k]
 if c['cohort_A'] or c['cohort_B']:
  selected.append(dict(r,record_set='map1260' if c['cohort_A'] else 'calendars2160',distance_m=200))
selected += [dict(r,record_set='calendars2160',distance_m=20) for r in a20]
keys=[(r['source_phase'],r['case_id'],r['pair_id'],int(r['lead_day'])) for r in primary+a20];assert len(keys)==len(set(keys))
# One read per case; CSV is always compared with original W arrays, lead-1.
groups=defaultdict(list)
for r in selected:groups[(r['source_phase'],r['case_id'])].append(r)
endpoint=[];diagnostics=[];stages=[];max_endpoint_diff=0.;max_width_diff=0.;pin_count=0
for n,((phase,cid),rr) in enumerate(groups.items()):
 p=ROOT/'results'/phase/'wb/W'/f'{cid}.json';w=load(p);assert w['case_id']==cid
 src=si[phase,cid];m=mi[phase,cid]
 assert int(src['realization'])==int(rr[0]['realization'])==int(m['realization'])
 for kind in ('tf','truth'):
  pin=src.get(kind+'_sha256');path=src[kind+'_path'];assert sha(path)==pin;pin_count+=1
 assert w['tf_input_sha256']==src['tf_sha256']
 tol=w.get('tolerance',{}); tau=tol.get('tau')
 diag=dict(source_phase=phase,case_id=cid,record_set=rr[0]['record_set'],distance_m=rr[0]['distance_m'],realization=rr[0]['realization'],layer_id=rr[0]['layer_id'],w_path=str(p),w_sha256=HASH[str(p)],W_status=w.get('W_status'),converged=w.get('converged'),flags=w.get('flags'),seed=w.get('seed'),W_seed=w.get('W_seed'),budget=w.get('budget'),tolerance=tol,bestfit_seed=w.get('bestfit_seed'),global_stages=w.get('global'),local_stages=w.get('local'),refined_leads=w.get('refined_leads'),refinement_increments=w.get('refinement_increments'),refinement_terminals=w.get('refinement_terminals'),direct_audit=w.get('direct_audit'),witness_check=w.get('witness_check'),n_accepted_total=w.get('n_accepted_total'),family_box=w.get('family_box'),stage_lead_extrema_available=False,stage_note='Global/local saved summaries are W10/Wmax, not per-lead inf/sup. No missing extrema reconstructed.')
 diagnostics.append(diag)
 for stage,objs in [('global',w.get('global',{})),('local',w.get('local',[]))]:
  for label,obj in (objs.items() if isinstance(objs,dict) else enumerate(objs,1)):
   stages.append(dict(source_phase=phase,case_id=cid,w_path=str(p),w_sha256=HASH[str(p)],stage=stage,stage_label=str(label),saved_values=obj))
 for r in rr:
  pair=r['pair_id'];lead=int(r['lead_day']); assert r['track']=='raw' and r['quantity']=='lead' and int(r['tool_horizon'])==lead
  lo=f(r['env_inf_m']);hi=f(r['env_sup_m']);e=f(r['E_true_m']);width=f(r['W_m']);assert lo is not None and hi is not None and e is not None and lo<=hi
  env=w['envelope'][pair];dl=abs(lo-env['inf'][lead-1]);dh=abs(hi-env['sup'][lead-1]);max_endpoint_diff=max(max_endpoint_diff,dl,dh);assert dl==dh==0
  max_width_diff=max(max_width_diff,abs(width-(hi-lo)))
  active=not truth(r.get('no_active_contrast'));assert active==(float(src['q_origin_m3d'])!=0) if r['record_set']=='calendars2160' else active
  sign=1 if lo>0 else -1 if hi<0 else 0
  nearest_side='inf' if abs(lo)<=abs(hi) else 'sup';nearest=lo if nearest_side=='inf' else hi
  wit=w.get('witnesses',{}).get(pair,{});wl=wit.get('inf',[]);wh=wit.get('sup',[])
  wi=wl[lead-1] if len(wl)>=lead else None;ws=wh[lead-1] if len(wh)>=lead else None
  row=dict(r,lead_day=lead,active=active,env_inf_m=lo,env_sup_m=hi,E_true_m=e,W_m=width,truth_inside=(lo<=e<=hi),truth_excluded=not(lo<=e<=hi),strict_zero_exclusion=(sign!=0),envelope_sign=sign,includes_exact_zero=(lo<=0<=hi),exact_zero_endpoint=(lo==0 or hi==0),width_effect_ratio=width/abs(e) if e!=0 else None,magnitude_resolved=(width/abs(e)<1) if e!=0 else None,nearest_zero_side=nearest_side,nearest_zero_effect_m=nearest,nearest_zero_abs_m=abs(nearest),nearest_zero_relative=abs(nearest)/abs(e) if e!=0 else None,w_path=str(p),w_sha256=HASH[str(p)],tau=tau,inf_witness=wi,sup_witness=ws,inf_A=wi.get('A') if wi else None,sup_A=ws.get('A') if ws else None,inf_sse_over_tau=wi.get('sse')/tau if wi and tau else None,sup_sse_over_tau=ws.get('sse')/tau if ws and tau else None,station=m.get('site',m.get('rain_site',m.get('station'))),declared_N_nominal=m.get('N_nominal'),declared_rho_nominal=m.get('rho_nominal'),tf_path=src['tf_path'],tf_sha256=src['tf_sha256'],truth_path=src['truth_path'],truth_sha256=src['truth_sha256'],family_version=w.get('family_box',{}).get('family_version','frozen_D03_D05'),witness_interpretation='feasibility at saved nonlinear shapes; not globally certified extrema')
  endpoint.append(row)
 if (n+1)%500==0:print('Processed saved cases',n+1,flush=True)
# Rank ALL endpoints by proximity (descriptive only), avoiding arbitrary near-zero epsilon.
ranks=defaultdict(list)
for r in endpoint:ranks[r['record_set'],r['distance_m'],r['pair_id'],r['lead_day']].append(r)
for rr in ranks.values():
 for rank,r in enumerate(sorted(rr,key=lambda r:(r['nearest_zero_abs_m'],r['case_id'])),1):r['nearest_zero_rank_all_origins']=rank
summary=[]
for (co,dist,pair,lead),rr in sorted(ranks.items()):
 aa=[r for r in rr if r['active']];nq=[r for r in rr if not r['active']]
 summary.append(dict(record_set=co,distance_m=dist,pair_id=pair,lead_day=lead,total_origins=len(rr),active_origins=len(aa),no_question_origins=len(nq),contributing_realization_clusters=len({r['realization'] for r in aa}),truth_inside_active=sum(r['truth_inside'] for r in aa),truth_excluded_active=sum(r['truth_excluded'] for r in aa),truth_inside_no_question=sum(r['truth_inside'] for r in nq),strict_zero_exclusion_active=sum(r['strict_zero_exclusion'] for r in aa),includes_exact_zero_active=sum(r['includes_exact_zero'] for r in aa),exact_zero_endpoint_active=sum(r['exact_zero_endpoint'] for r in aa),magnitude_ratio_defined_active=sum(r['width_effect_ratio'] is not None for r in aa),width_effect_below1_active=sum(r['magnitude_resolved'] is True for r in aa),width_effect_below1_truth_excluded_active=sum(r['magnitude_resolved'] is True and r['truth_excluded'] for r in aa)))
# exact paired lead and question identities
ix={(r['source_phase'],r['case_id'],r['pair_id'],r['lead_day']):r for r in endpoint}
max_half=0.;discord=[];case_comparisons=[]
for (phase,cid),rr in groups.items():
 for pair in PAIRS:
  a=ix[phase,cid,pair,10];b=ix[phase,cid,pair,30]
  dd=(a['envelope_sign']!=b['envelope_sign']);md=(a['magnitude_resolved']!=b['magnitude_resolved']);td=(a['truth_inside']!=b['truth_inside'])
  cr=dict(source_phase=phase,case_id=cid,record_set=a['record_set'],distance_m=a['distance_m'],pair_id=pair,realization=a['realization'],active=a['active'],direction_discordant=dd,magnitude_discordant=md,truth_inclusion_discordant=td)
  case_comparisons.append(cr)
  if dd or md or td:
   for key in ('env_inf_m','env_sup_m','E_true_m','W_m','envelope_sign','strict_zero_exclusion','magnitude_resolved','truth_inside','nearest_zero_effect_m','nearest_zero_abs_m','nearest_zero_relative','inf_A','sup_A','inf_sse_over_tau','sup_sse_over_tau','inf_witness','sup_witness','w_path','w_sha256','tau','Q_scale','N_transitions','rho','no_active_contrast','nearest_zero_rank_all_origins'):
    cr[key+'_10']=a.get(key);cr[key+'_30']=b.get(key)
   discord.append(cr)
 for lead in (10,30):
  p1=ix[phase,cid,PAIRS[0],lead];p2=ix[phase,cid,PAIRS[1],lead]
  assert p1['strict_zero_exclusion']==p2['strict_zero_exclusion'] and p1['magnitude_resolved']==p2['magnitude_resolved']
  max_half=max(max_half,abs(p2['env_inf_m']+p1['env_sup_m']/2),abs(p2['env_sup_m']+p1['env_inf_m']/2),abs(p2['E_true_m']+p1['E_true_m']/2),abs(p2['W_m']-p1['W_m']/2))
assert max_half<1e-12, max_half # verification roundoff only; classification uses exact zero
for co,dist,counts in [('map1260',200,(856,865)),('calendars2160',200,(669,676)),('calendars2160',20,(1087,1087))]:
 for lead,count in zip((10,30),counts):
  for pair in PAIRS:assert next(r for r in summary if (r['record_set'],r['distance_m'],r['pair_id'],r['lead_day'])==(co,dist,pair,lead))['strict_zero_exclusion_active']==count
write('B02_truth_inclusion.csv',summary);write('B02_source_rows.csv',endpoint);write('B02_lead_discordance.csv',discord);write('B02_all_lead_comparisons.csv',case_comparisons);write('B02_saved_search_diagnostics.csv',diagnostics);write('B02_saved_budget_trajectories.csv',stages)
# Direct reread output verification plus immutable source rehash.
for name,expected in [('B02_truth_inclusion.csv',len(summary)),('B02_source_rows.csv',len(endpoint)),('B02_lead_discordance.csv',len(discord)),('B02_saved_search_diagnostics.csv',len(diagnostics)),('B02_saved_budget_trajectories.csv',len(stages))]:
 with (OUT/name).open() as h:assert len(list(csv.DictReader(h)))==expected
changed=[p for p,h in HASH.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed,changed
comments=[c['id'] for c in matrix['comments'] if 'B02' in json.dumps(c)]
boundary=[r for r in discord if r['direction_discordant'] and r['pair_id']==PAIRS[0]]
boundary_diagnosis={'unique_records':len(boundary),'all_30_only':all(not r['strict_zero_exclusion_10'] and r['strict_zero_exclusion_30'] for r in boundary),'nearest_endpoint10_exact_zero':sum(r['nearest_zero_effect_m_10']==0 for r in boundary),'nearest_endpoint10_positive':sum(r['nearest_zero_effect_m_10']>0 for r in boundary),'all_nearest_witness_gain_positive_both_leads':all(r['sup_A_10']>0 and r['sup_A_30']>0 for r in boundary),'nearest_endpoint10_abs_max_m':max(abs(r['nearest_zero_effect_m_10']) for r in boundary),'nearest_endpoint30_abs_min_m':min(abs(r['nearest_zero_effect_m_30']) for r in boundary),'nearest_endpoint30_abs_max_m':max(abs(r['nearest_zero_effect_m_30']) for r in boundary),'implementation_source':'lib/p3_wenvelope.py pump_columns: u=-causal_conv(dq,B); lib/p3_kernels.py causal_conv: FFT irfft(rfft(k)*rfft(x)); repaired endpoint_curves: E=A*u; saved endpoints are not clipped to hydraulic sign','interpretation':'Positive/tiny zero 10-day endpoints at positive A are compatible with finite precision convolution behavior, rather than a demonstration of a lead-dependent zero-gain admissibility boundary. This is a stored-output/code inference; no kernels or engine were rerun. Saved witnesses do not exclude unsampled feasible A=0 natural shapes.'}
write('B02_direction_boundary_records.csv',boundary)
dc=[]
for co,dist in [('map1260',200),('calendars2160',200),('calendars2160',20)]:
 for pair in PAIRS:
  rr=[r for r in case_comparisons if (r['record_set'],r['distance_m'],r['pair_id'])==(co,dist,pair)]
  dc.append(dict(record_set=co,distance_m=dist,pair_id=pair,n_comparisons=len(rr),direction_discordant=sum(r['direction_discordant'] for r in rr),determined10_only=sum(ix[r['source_phase'],r['case_id'],pair,10]['strict_zero_exclusion'] and not ix[r['source_phase'],r['case_id'],pair,30]['strict_zero_exclusion'] for r in rr),determined30_only=sum(not ix[r['source_phase'],r['case_id'],pair,10]['strict_zero_exclusion'] and ix[r['source_phase'],r['case_id'],pair,30]['strict_zero_exclusion'] for r in rr),magnitude_discordant=sum(r['magnitude_discordant'] for r in rr),truth_inclusion_discordant=sum(r['truth_inclusion_discordant'] for r in rr)))
conv=Counter((r['record_set'],r['distance_m'],str(r['converged'])) for r in diagnostics)
ss=dict(assigned_job='B02',status='complete',comment_ids=comments,cohort_definitions={'map1260':'phase2 N_nominal6:180 + phase3 B:360 + phase4 A:720; original map identity exact','calendars2160':'all phase4 B at200m and phase5 A_distance20 at20m; inactive origins separately retained','track':'raw; quantity lead; native horizon=lead10/30','near_zero':'No arbitrary epsilon or threshold. All endpoint distances and ranks supplied; exact zero separately counted.'},summary=summary,lead_discordance=dc,direction_boundary_diagnosis=boundary_diagnosis,diagnostics_cases=len(diagnostics),converged_counts=[dict(record_set=k[0],distance_m=k[1],converged=k[2],count=v) for k,v in sorted(conv.items())],maximum_endpoint_csv_W_difference_m=max_endpoint_diff,maximum_W_width_difference_m=max_width_diff,maximum_half_width_identity_roundoff_m=max_half,source_index_note='SOURCE_INDEX has tf/truth/native pins but no w_path. W resolved by original p4_collect/p5_collect wb/W convention, case ID and exact array match verified.',bounded_diagnosis='Official calls reproduced. Per-lead stored endpoint witnesses are feasible saved members. Lead discordance is present in the computed inner envelope; saved search summaries do not provide a global null-face test or prove a physical lead-dependent direction boundary. Global natural-shape zero profiling remains C01, unexecuted. Global/local W10/Wmax trajectories cannot reconstruct unsaved per-lead extrema.',no_global_certification_claim=True,new_engine_queries=0,new_W_searches=0,new_simulations=0,new_experiments=0,new_random_draws=0,bootstrap='not requested by B02; none used',whole_parent_completion=False,blockers=[])
dump('SUMMARY.json',ss);dump('ANALYSIS_SUMMARY.json',ss)
nums=[]
for key,value in boundary_diagnosis.items():
 if isinstance(value,(int,float)) and not isinstance(value,bool):nums.append(dict(metric=key,value=value,source_json=str(OUT/'SUMMARY.json'),json_section='direction_boundary_diagnosis'))
for r in summary:
 for k,v in r.items():
  if isinstance(v,(int,float)) and k not in ('distance_m','lead_day'):nums.append(dict(metric=k,value=v,record_set=r['record_set'],distance_m=r['distance_m'],pair_id=r['pair_id'],lead_day=r['lead_day'],source_csv=str(OUT/'B02_truth_inclusion.csv'),source_filter={kk:r[kk] for kk in ('record_set','distance_m','pair_id','lead_day')},column=k))
for r in dc:
 for k,v in r.items():
  if isinstance(v,int) and k!='distance_m':nums.append(dict(metric=k,value=v,source_json=str(OUT/'SUMMARY.json'),json_section='lead_discordance',source_filter={kk:r[kk] for kk in ('record_set','distance_m','pair_id')}))
dump('NEW_NUMBERS.json',nums)
md=['Saved-result boundary diagnostics (B02). Counts describe sampled inner envelopes and saved feasible witnesses; they do not certify global extrema or a calibrated 95% region.','', '| Population | Distance (m) | Question | Lead | All | Active | No question | Truth inside (active) | Truth excluded (active) | Strict zero excluded | W/abs(truth)<1 | Ratio<1 but truth excluded |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:md.append('| '+' | '.join(str(r[k]) for k in ('record_set','distance_m','pair_id','lead_day','total_origins','active_origins','no_question_origins','truth_inside_active','truth_excluded_active','strict_zero_exclusion_active','width_effect_below1_active','width_effect_below1_truth_excluded_active'))+' |')
md+=['','| Population | Distance | Question | Direction discordant | 10 only | 30 only | Magnitude discordant | Truth inclusion discordant |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in dc:md.append('| '+' | '.join(str(r[k]) for k in ('record_set','distance_m','pair_id','direction_discordant','determined10_only','determined30_only','magnitude_discordant','truth_inclusion_discordant'))+' |')
md+=['',json.dumps(boundary_diagnosis,ensure_ascii=False),'',ss['bounded_diagnosis'],'','Exact IDs and both lead witnesses are in B02_lead_discordance.csv (direction_discordant=True selects direction calls). B02_source_rows.csv supplies all endpoint distances and ranks without changing zero. B02_saved_budget_trajectories.csv reports actual stages only.','', 'Source: B02_truth_inclusion.csv, SUMMARY.json and B02_SOURCE_RECEIPT.json; each number has a filter/column or JSON section in NEW_NUMBERS.json.']
(OUT/'manuscript_ready_tables.md').write_text('\n'.join(md)+'\n')
receipt=dict(validation_type='source_audit',validation_claim_id='B02',validates_claim='D09 B02 assigned stored-result analyses delivered',source_path=str(D09/'author instruction'),audit_result='pass',validation_result='pass',status='B02_COMPLETE',turn_id='run',worker='executor',utc=datetime.now(timezone.utc).isoformat(),assigned_jobs=[dict(id='B02',status='complete',comment_ids=comments)],whole_parent_completion=False,component_complete=True,source_sha256=HASH,sourcechecks={'matrix_terminal_complete':True,'dispatch_pins_match':True,'original_map_IDs_exact':True,'cohort_manifest_counts':cohort['counts'],'row_duplicates':0,'native_horizon_matches_lead':True,'CSV_W_endpoint_match_exact':True,'original_tf_truth_pins_verified':pin_count,'original_sources_unchanged':True,'official_direction_counts_reproduced':True,'two_questions_classification_identical':True,'maximum_half_identity_roundoff_m':max_half},direct_output_verification={'reopened_csv_cardinalities':True,'source_rows':len(endpoint),'case_diagnostics':len(diagnostics),'discordance_rows':len(discord),'stage_rows':len(stages)},software={'python':sys.version,'platform':platform.platform()},parameters={'leads':[10,30],'track':'raw','quantity':'lead','zero_rule':'inf>0 or sup<0; no epsilon','truth_inclusion':'inf<=E_true<=sup','magnitude':'W/abs(E_true)<1 only E_true!=0','random_seed':None,'bootstrap_draws':0},reproduction_command=f'python {OUT}/reproduce_B02.py',scope={'new_engine_queries':0,'new_W_searches':0,'new_simulations':0,'new_experiments':0,'manuscript_edits':0},residual_risk=[ss['bounded_diagnosis']],blockers=[],untried_in_scope_alternatives=[],next_agent_action='Writer integrates saved-result tables and exact discordant IDs; C01 new global zero-face experiment requires separate assignment.',sourcecode_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
receipt['output_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name not in ('B02_SOURCE_RECEIPT.json','COMPONENT_RECEIPT.json')}
dump('B02_SOURCE_RECEIPT.json',receipt);dump('COMPONENT_RECEIPT.json',receipt)
print(json.dumps({'status':'B02_COMPLETE','summary':summary,'discordance':dc,'case_diagnostics':len(diagnostics),'source_files':len(HASH),'comment_ids':comments},indent=2),flush=True)
