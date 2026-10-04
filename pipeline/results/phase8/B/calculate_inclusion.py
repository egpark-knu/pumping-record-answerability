#!/usr/bin/env python3
"""D10 B: deterministic aggregation of saved envelopes and saved truth only."""
import csv, hashlib, json, math, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3]))); P=R/'results/phase8'; B=P/'B'
PAIRS=('P1_continue_vs_stop','P2_current_vs_1p5x')
hashes={}
def sha(path):
    path=str(path)
    if path not in hashes: hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashes[path]
def readj(path):
    sha(path); return json.loads(Path(path).read_text())
def rows(path):
    sha(path)
    with open(path,newline='') as f: return list(csv.DictReader(f))
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
freeze=readj(P/'PROTOCOL_FREEZE_RECEIPT.json')
for path,pin in freeze['source_pins'].items(): assert sha(path)==pin,(path,'frozen pin mismatch')
manifest=readj(P/'data/record_manifest.json')
idxpath=R/'results/phase6/D08/protocol_neutral/SOURCE_INDEX.json'
index=readj(idxpath)['records']; idx={(r['source_phase'],r['case_id']):r for r in index}
assert len(idx)==6270
orig={(r['source_phase'],r['case_id']) for r in manifest if r['variant']=='original'}
assert orig==set(idx)
assert len(manifest)==6420 and len({(r['source_phase'],r['case_id']) for r in manifest})==6420
mapids={(r['source_phase'],r['case_id']) for r in readj(R/'results/phase4/COHORT_MANIFESTS.json')['map']}; assert len(mapids)==1260
primary={}; metadata={}
for path in (R/'results/phase4/primary_rows.csv',R/'results/phase5/A_rows.csv'):
    for r in rows(path):
        if r['track']!='raw' or r['quantity']!='lead': continue
        key=(r['source_phase'],r['case_id'],r['pair_id'],int(r['lead_day']))
        assert key not in primary; primary[key]=(r,str(path)); metadata[key[:2]]=r
assert len(primary)==25080
D8=R/'results/phase6/D08/protocol_neutral'
vtruth={(r['variant'],r['case_id'],r['pair'],int(r['lead'])):r for r in rows(D8/'execution/B/source_frame.csv')}
vwidth={(r['variant'],r['case_id'],r['pair'],int(r['lead'])):r for r in rows(D8/'execution/W/comparison_source_frame.csv')}
gen={(r['variant'],r['case_id']):r for r in readj(D8/'execution/generated/GENERATION_RECEIPT.json')['records']}
d09={(r['source_phase'],r['case_id'],r['pair_id'],int(r['lead_day'])):r for r in rows(R/'results/phase7/D09/analysis/B02/B02_source_rows.csv')}
# Public edition: the run receipts of the supporting analyses are not part of the release; their code is hashed.
for rel in ('results/phase7/D09/analysis/B02/reproduce_B02.py','results/phase7/D09/analysis/B03_B04/analyze_saved.py','results/phase6/D08/protocol_neutral/d08_runtime.py'): sha(R/rel)
agg=defaultdict(Counter); joins=Counter(); records=Counter()
fields=['source_phase','case_id','variant','population','record_set','experiment_group','main_cohort','main_manuscript_eligible','layer_id','storage_type','operation','calendar_month','distance_m','realization','active_state','pair_id','lead_day','env_inf_m','env_sup_m','E_true_m','inclusion_status','truth_exact_zero','W_status','truth_provenance','w_path','w_sha256','truth_path','truth_sha256','primary_source_path','primary_joined']
def add(key,status,zero):
    c=agg[key]; c['n']+=1;c[status]+=1;c['truth_exact_zero']+=zero
with open(B/'source_frame.csv','w',newline='') as f:
    out=csv.DictWriter(f,fieldnames=fields);out.writeheader()
    for m in manifest:
        phase,cid,var=m['source_phase'],m['case_id'],m['variant']; original=var=='original'
        sourcekey=(phase,cid) if original else ('phase5',cid)
        s=idx[sourcekey]; md=metadata[sourcekey]
        for name in ('tf','w'): assert sha(m[name+'_path'])==m[name+'_sha256'],(cid,name)
        w=readj(m['w_path']); assert w['case_id']==cid and w['tf_input_sha256']==m['tf_sha256']
        if not original:
            g=gen[(var,cid)]; assert g['tf_input_sha256']==m['tf_sha256']; assert g['rain_natural_noise_future_unchanged']
            w=w['result']
        tp=s['truth_path']; assert sha(tp)==s['truth_sha256']
        with np.load(tp,allow_pickle=False) as truth: es={pair:truth['E_true__'+pair].tolist() for pair in PAIRS}
        cohort=('map1260' if sourcekey in mapids else 'calendar200m' if m['record_set']=='calendar200m' else 'calendar20m' if m['record_set']=='calendar20m' else 'other_design') if original else 'sensitivity_'+var
        pop='original' if original else 'sensitivity'; active='active' if m['active'] else 'no_question'; eligible=original and cohort!='other_design'
        layer=md.get('layer_id','') or m.get('layer_id',''); storage=md.get('storage_type',''); operation=md.get('calendar_type','') or md.get('form',''); month=md.get('origin_month','')
        group=md.get('experiment_group','') or s.get('experiment_group','')
        base=dict(source_phase=phase,case_id=cid,variant=var,population=pop,record_set=m['record_set'],experiment_group=group,main_cohort=cohort,main_manuscript_eligible=eligible,layer_id=layer,storage_type=storage,operation=operation,calendar_month=month,distance_m=m['distance_m'],realization=m['realization'],active_state=active,W_status=w.get('W_status',''),truth_provenance='saved_original_truth' if original else 'frozen_D06_parent_effect_as_used_by_D08',w_path=m['w_path'],w_sha256=m['w_sha256'],truth_path=tp,truth_sha256=s['truth_sha256'])
        scopes=['all_original',cohort,'set:'+phase+':'+m['record_set']+':'+group] if original else [cohort]
        strata=[('overall','all'),('layer',layer or 'unknown'),('storage',storage or 'unknown'),('operation',operation or 'not_available'),('calendar_month',month or 'not_available'),('layer_storage_operation_calendar','|'.join((layer,storage,operation,month)))]
        records[(pop,phase,m['record_set'],active)]+=1
        for pair in PAIRS:
            env=w.get('envelope',{}).get(pair,{})
            assert len(es[pair])==30
            for lead in range(1,31):
                lo=env.get('inf',[None]*30)[lead-1];hi=env.get('sup',[None]*30)[lead-1];e=es[pair][lead-1]
                valid=all(x is not None and math.isfinite(float(x)) for x in (lo,hi,e)) and lo<=hi
                status=('included' if lo<=e<=hi else 'excluded') if valid else 'missing_or_invalid'
                zero=e==0; pk=(phase,cid,pair,lead); primarypath='';joined=False
                if lead in (10,30):
                    if original:
                        pr,primarypath=primary[pk]
                        assert (lo,hi,e)==tuple(float(pr[k]) for k in ('env_inf_m','env_sup_m','E_true_m')),pk
                        joins['original_primary']+=1
                        if pk in d09:
                            dr=d09[pk]; assert (lo,hi,e)==tuple(float(dr[k]) for k in ('env_inf_m','env_sup_m','E_true_m'));assert (status=='included')==(dr['truth_inside']=='True'); joins['D09_B02']+=1
                    else:
                        vk=(var,cid,pair,lead);assert e==float(vtruth[vk]['E_true_m']);assert hi-lo==float(vwidth[vk]['W_m']);joins['D08_variant_truth']+=1;joins['D08_variant_width']+=1;primarypath=str(D8/'execution/B/source_frame.csv')
                    joined=True
                row=base|dict(pair_id=pair,lead_day=lead,env_inf_m=lo,env_sup_m=hi,E_true_m=e,inclusion_status=status,truth_exact_zero=zero,primary_source_path=primarypath,primary_joined=joined)
                out.writerow(row)
                for scope in scopes:
                    for st,val in strata: add((pop,scope,st,val,active,pair,lead),status,zero)
assert dict(joins)==dict(original_primary=25080,D09_B02=22320,D08_variant_truth=600,D08_variant_width=600),joins
cols=['population','scope','stratum','stratum_value','active_state','pair_id','lead_day','n','included','excluded','missing_or_invalid','valid_n','truth_exact_zero','inclusion_rate_valid','inclusion_rate_all']
with open(B/'inclusion_by_set_stratum_lead.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader()
    for key,c in sorted(agg.items()):
        valid=c['included']+c['excluded'];wr.writerow(dict(zip(cols[:7],key))|{k:c[k] for k in ('n','included','excluded','missing_or_invalid','truth_exact_zero')}|dict(valid_n=valid,inclusion_rate_valid=c['included']/valid if valid else '',inclusion_rate_all=c['included']/c['n']))
# One deterministic local check: re-read complete output and recompute exact predicates/counts.
check=Counter(); basecheck=defaultdict(Counter)
with open(B/'source_frame.csv',newline='') as f:
    for row in csv.DictReader(f):
        vals=[float(row[k]) if row[k] else math.nan for k in ('env_inf_m','env_sup_m','E_true_m')];lo,hi,e=vals
        valid=all(math.isfinite(x) for x in vals) and lo<=hi
        expected=('included' if lo<=e<=hi else 'excluded') if valid else 'missing_or_invalid'
        assert expected==row['inclusion_status'];check[row['population']]+=1;check[expected]+=1
        scope='all_original' if row['population']=='original' else row['main_cohort']
        k=(row['population'],scope,'overall','all',row['active_state'],row['pair_id'],int(row['lead_day']));basecheck[k]['n']+=1;basecheck[k][expected]+=1;basecheck[k]['truth_exact_zero']+=row['truth_exact_zero']=='True'
assert check['original']==376200 and check['sensitivity']==9000
for k,c in basecheck.items(): assert c==agg[k],(k,c,agg[k])
for path,pin in freeze['source_pins'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin
scopes=['all_original','map1260','calendar200m','calendar20m','other_design','sensitivity_range','sensitivity_control','sensitivity_stop_control']
text=['# D10 B — saved envelope inclusion','', 'Exact closed rule: `inf <= E_true <= sup`, with no epsilon. These are empirical inclusion counts in saved envelopes, not estimated 95% confidence coverage. Active and no-question records remain separate. Original denominators exclude sensitivity variants.','', 'All 6,270 originals and 150 existing variants are represented at both pairs and every lead 1–30. Main manuscript cohorts are map1260, calendar200m, and calendar20m (5,580 originals). Other design records (690) remain in the complete original population.','', '| Set | State | Pair | Lead | Included / records | Missing or invalid | Rate |','|---|---|---|---:|---:|---:|---:|']
for scope in scopes:
    pop='sensitivity' if scope.startswith('sensitivity') else 'original'
    for active in ('active','no_question'):
        for pair in PAIRS:
            for lead in (10,30):
                c=agg.get((pop,scope,'overall','all',active,pair,lead))
                if c: text.append(f"| {scope} | {active} | {pair} | {lead} | {c['included']} / {c['n']} | {c['missing_or_invalid']} | {100*c['included']/c['n']:.3f}% |")
text+=['','`source_frame.csv` contains unique `(source_phase, case_id, pair_id, lead_day)` keys; variants have distinct source phases. Aggregation scopes overlap intentionally; sum only all_original plus the three sensitivity scopes for full-population totals. Layer/storage/operation/calendar strata are descriptive stored metadata; unavailable fields are explicitly labeled.','', 'Variant truth is the frozen phase5 parent effect, exactly as used in the existing D08 B comparison frame. No variant-specific truth NPZ is present. Existing D08 generation receipts document unchanged future forcing; reported parent-truth inclusion is a sensitivity diagnostic.','', f"One deterministic check re-read {sum(check[x] for x in ('original','sensitivity')):,} rows, recomputed the exact rule, and reconciled aggregate counts. Joins: {dict(joins)}."]
(B/'readytable.md').write_text('\n'.join(text)+'\n')
sha(B/'calculate_inclusion.py');dump(B/'INPUT_HASHES.json',hashes)
outputs={name:sha(B/name) for name in ('source_frame.csv','inclusion_by_set_stratum_lead.csv','readytable.md','INPUT_HASHES.json','calculate_inclusion.py')}
receipt=dict(status='pass',validation_type='source_audit',validates_claim='D10 B inclusion component delivered',whole_parent_completion=False,utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=freeze['protocol_sha256'],record_manifest_sha256=sha(P/'data/record_manifest.json'),input_hash_manifest='INPUT_HASHES.json',outputs_sha256=outputs,records=6420,original_records=6270,sensitivity_records=150,active_records=4584,no_question_records=1836,all_lead_rows=dict(check),source_frame_unique_key=['source_phase','case_id','pair_id','lead_day'],record_counts=[dict(population=k[0],source_phase=k[1],record_set=k[2],active_state=k[3],n=v) for k,v in sorted(records.items())],joins=dict(joins),main_manuscript_original_records=5580,inclusion_rule='exact finite closed inf <= E_true <= sup; no epsilon',missing_definition='missing/nonfinite endpoints or truth, or inverted endpoints',check=dict(type='single deterministic output re-read and predicate/aggregate reconciliation',status='pass',rows=385200,source_pins_unchanged=True),source_coverage=dict(original_index_complete=True,all_leads=list(range(1,31)),D09_B02_matching_rows=22320,D09_B03_B04_receipt_and_frame_code_inspected=True,reused_claim_scope='Only pinned stored frames and independently matched values; no whole historical receipt revalidation'),variant_truth_provenance='frozen phase5 parent E_true used in existing D08 B frame; no new truth evaluation',execution=dict(python=sys.executable,numpy=np.__version__,command=f'{sys.executable} {B}/calculate_inclusion.py',new_fits=False,new_engine_queries=False,new_W_queries=False,new_bootstraps=False))
dump(B/'B_RECEIPT.json',receipt)
component=dict(schema_version='1.0',component='D10_B',validation_claim_id='D10_B_inclusion',audit_result='pass',component_complete=True,turn_id='run',worker='executor',validation_type='source_audit',validates_claim='D10 B inclusion component delivered',validation_result='pass',status='pass',whole_parent_completion=False,source_path=str(P/'orchestration/author instruction'),protocol_sha256=freeze['protocol_sha256'],receipt_path=str(B/'B_RECEIPT.json'),receipt_sha256=sha(B/'B_RECEIPT.json'),artifacts=outputs,evidence=['6420 records; 385200 pair-lead rows','25080 original primary + 600 variant truth + 600 width + 22320 D09 joins exact','one deterministic complete-output check passed'])
dump(B/'COMPONENT_RECEIPT.json',component)
print(json.dumps(dict(status='pass',counts=dict(check),joins=dict(joins),outputs=outputs),indent=2))
