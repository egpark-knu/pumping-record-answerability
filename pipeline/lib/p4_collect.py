"""Observed/evaluation row serializer for the D05 C and calendar statistics (post-scoring use only).

One row per (source_phase, case_id, pair_id, lead_day in {10,30}), raw track, quantity 'lead', with the headers that
p4_statistics.adapt_row reads. The structurally matched reference E_reference_m is MANDATORY for every C row: old D03/D04
references are rejoined from their own immutable wb/reference files (old cell_metrics never carried the column), new
references come from results/phase4/wb/reference. strict=True raises on any missing W/truth/reference/tool output;
strict=False keeps the row with an explicit *_status of 'missing'/'error' and null value, never a zero or a dropped row.
Primary tool values: lead10 from the independent H10 run, lead30 from H30; H30 day10 is a separately named diagnostic.
"""
import csv,json
from pathlib import Path
import numpy as np
from p4_contracts import ROOT,P4,cohort_manifests,sha

PAIRS=('P1_continue_vs_stop','P2_current_vs_1p5x');LEADS=(10,30);ACTORS=('shard_a','shard_b')
HEADERS=('source_phase','case_id','experiment_group','dataset','layer_id','sid','storage_type','T_m2_d','realization','Q_scale','rho','N_transitions',
         'recency_label','SR','SR_definition','SR_context_mean','q_origin_m3d','q_last_context_day_m3d','calendar_type','offseason_use_variant','origin_month',
         'no_active_contrast','last_off_length_days','last_off_distance_to_origin_days','rho_last_off','N_on_off_observed','schedule_template',
         'pair_id','track','quantity','lead_day','W_m','W_status','W_flags','env_inf_m','env_sup_m','E_true_m','truth_status',
         'E_tool_m','tool_status','tool_horizon','E_tool_H30_day10_diagnostic_m','E_reference_m','reference_status','reference_source_path','reference_sha256','data_status')
SOURCE_PHASE={'D03':'phase2','D04':'phase3'}

class MissingOutput(RuntimeError):pass

def _f(v):
 if v is None or v=='':return None
 x=float(v);return x if np.isfinite(x) else None

_REF_SHA={}
def reference_value(path,pair,lead):
 """(value, status) from a frozen-driver reference JSON; an error record or absent file is never a value."""
 path=Path(path)
 if not path.is_file():return None,'missing'
 r=json.loads(path.read_text())
 if r.get('error') or pair not in r.get('pairs',{}):return None,'error'
 v=_f(r['pairs'][pair]['E_point'][lead-1]);return (v,'ok') if v is not None else (None,'nonfinite')

def reference_receipt(path):
 """Path relative to the project root and raw-byte SHA256 of the reference file (None when absent)."""
 path=Path(path);rel=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
 if not path.is_file():return rel,None
 if str(path) not in _REF_SHA:_REF_SHA[str(path)]=sha(path)
 return rel,_REF_SHA[str(path)]

def _need(strict,status,what):
 if strict and status!='ok':raise MissingOutput(what+': '+status)

def old_rows(strict=True,metrics=ROOT/'results/phase3/cell_metrics.csv'):
 """All 1230 D03/D04 cases x 2 pairs x 2 leads from the frozen combined table, with rejoined references."""
 out=[]
 with open(metrics,newline='') as fh:
  for r in csv.DictReader(fh):
   if r['track']!='raw' or r['quantity']!='lead' or _f(r['lead_day']) not in LEADS:continue
   phase=SOURCE_PHASE[r['dataset']];lead=int(float(r['lead_day']))
   rp=ROOT/'results'/phase/'wb/reference'/f"{r['case_id']}.json";ref,rs=reference_value(rp,r['pair_id'],lead);rpath,rsha=reference_receipt(rp)
   _need(strict,rs,f"reference {phase}/{r['case_id']}")
   out.append(dict(source_phase=phase,case_id=r['case_id'],experiment_group=r['experiment_group'],dataset=r['dataset'],layer_id=r['sid'],sid=r['sid'],
                   storage_type=r['storage_type'],T_m2_d=_f(r['T_m2_d']),realization=int(float(r['realization'])),Q_scale=_f(r['Q_scale']),rho=_f(r['rho']),
                   N_transitions=_f(r['N_transitions']),recency_label=r['recency_label'],SR=_f(r['SR']),SR_definition='outside_designated_rest',
                   pair_id=r['pair_id'],track='raw',quantity='lead',lead_day=lead,W_m=_f(r['W_m']),W_status=r['W_status'],W_flags=r.get('flags'),
                   env_inf_m=_f(r['env_inf_m']),env_sup_m=_f(r['env_sup_m']),E_true_m=_f(r['E_true_m']),truth_status='ok',E_tool_m=_f(r['E_tool_m']),
                   tool_status='ok' if _f(r['E_tool_m']) is not None else 'missing',tool_horizon=lead,E_reference_m=ref,reference_status=rs,reference_source_path=rpath,reference_sha256=rsha,data_status=r['data_status']))
 return out

def _tool_tables(tools_dir):
 tabs={}
 for H in (10,30):
  tabs[H]={}
  for a in ACTORS:
   p=Path(tools_dir)/a/f'timesfm_raw_H{H}.npz'
   if not p.is_file():continue
   with np.load(p,allow_pickle=False) as z:
    for i,pid in enumerate(z['pair_id']):
     if str(pid) in tabs[H]:raise ValueError('Two actors produced the same contrast '+str(pid))
     tabs[H][str(pid)]=np.asarray(z['E_tool'][i],float)
 return tabs

def new_rows(strict=True,base=P4):
 """2880 new cases x 2 pairs x 2 leads from cases/derived_manifest, wb (W, truth_eval, reference) and both actor tool dirs."""
 base=Path(base);man=json.loads((base/'cases/derived_manifest.json').read_text());tabs=_tool_tables(base/'tools');out=[]
 for m in man:
  cid=m['case_id'];wp=base/'wb/W'/f'{cid}.json';tp=base/'wb/truth_eval'/f'{cid}.json'
  W=json.loads(wp.read_text()) if wp.is_file() else None;T=json.loads(tp.read_text()) if tp.is_file() else None
  env=(W or {}).get('envelope');ws=(W or {}).get('W_status','missing') if W is not None else 'missing'
  if W is not None and W.get('error'):ws='error'
  for p in PAIRS:
   for lead in LEADS:
    rp=base/'wb/reference'/f'{cid}.json';ref,rs=reference_value(rp,p,lead);rpath,rsha=reference_receipt(rp)
    inf=_f(env[p]['inf'][lead-1]) if env else None;sup=_f(env[p]['sup'][lead-1]) if env else None
    et=_f(T['E_true'][p][lead-1]) if T else None;ts='ok' if et is not None else 'missing'
    pid=f'{cid}__{p}';tool=tabs[lead].get(pid);diag=tabs[30].get(pid)
    eo=_f(tool[lead-1]) if tool is not None else None;tos='ok' if eo is not None else 'missing'
    for st,what in ((rs,'reference'),(ts,'truth'),(tos,'tool'),('ok' if env else ws,'W envelope')):_need(strict,st,f'{what} phase4/{cid}/{p}/{lead}')
    out.append(dict(source_phase='phase4',case_id=cid,experiment_group=m['experiment_group'],dataset='D05',layer_id=m['layer_id'],sid=m['sid'],
                    storage_type=m['storage_type'],T_m2_d=m['T_m2_d'],realization=m['realization'],Q_scale=m['Q_scale'],rho=m.get('rho'),
                    N_transitions=m.get('N_transitions'),recency_label=m.get('recency_label'),SR=m['SR'],SR_definition=m['SR_definition'],
                    SR_context_mean=m['SR_context_mean'],q_origin_m3d=m['q_origin_m3d'],q_last_context_day_m3d=m['q_last_context_day_m3d'],
                    calendar_type=m.get('calendar_type'),offseason_use_variant=m.get('offseason_use_variant'),origin_month=m.get('origin_month'),
                    no_active_contrast=m['no_active_contrast'],last_off_length_days=m['last_off_length_days'],
                    last_off_distance_to_origin_days=m['last_off_distance_to_origin_days'],rho_last_off=m['rho_last_off'],N_on_off_observed=m['N_on_off_observed'],
                    schedule_template=m['schedule_template'],pair_id=p,track='raw',quantity='lead',lead_day=lead,
                    W_m=(sup-inf) if inf is not None and sup is not None else None,W_status=ws,W_flags=';'.join((W or {}).get('flags',[])),
                    env_inf_m=inf,env_sup_m=sup,E_true_m=et,truth_status=ts,E_tool_m=eo,tool_status=tos,tool_horizon=lead if tool is not None else None,
                    E_tool_H30_day10_diagnostic_m=_f(diag[9]) if (diag is not None and lead==10) else None,
                    E_reference_m=ref,reference_status=rs,reference_source_path=rpath,reference_sha256=rsha,data_status='official_analysis' if all(s=='ok' for s in (rs,ts,tos)) and env else 'incomplete_output'))
 return out

def assemble(strict=True,base=P4):
 """C rows exactly matching the explicit 4110-case cohort x 2 pairs x 2 leads; duplicates/omissions are errors."""
 rows=old_rows(strict)+new_rows(strict,base);C=cohort_manifests()['C']
 want={(c['source_phase'],c['case_id'],p,l) for c in C for p in PAIRS for l in LEADS}
 got=[(r['source_phase'],r['case_id'],r['pair_id'],r['lead_day']) for r in rows]
 if len(got)!=len(set(got)):raise ValueError('Duplicate C rows')
 if set(got)!=want:raise MissingOutput(f'C coverage: missing={len(want-set(got))} extra={len(set(got)-want)}')
 return rows

def write(rows,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,'w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=HEADERS,extrasaction='raise');w.writeheader();w.writerows({k:r.get(k) for k in HEADERS} for r in rows)
 return path
