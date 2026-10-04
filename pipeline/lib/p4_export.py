"""Observed-only D05 export permits exact zero origin rates; never substitutes epsilon or nominal pumping.

Origin contract (PROTOCOL_DESIGN section 4): both futures hold the declared operational origin rate q_origin.
A: q_origin must equal the last context rate. B: q_origin is read from truth-free metadata and re-derived from the
deterministic calendar rule (form, realization, origin date, observed origin-day rain); it may differ from q_last.
"""
import json
import numpy as np
import p3_phase2_export as old
import p3_phase2_timesfm as base
import timesfm_pilot_adapter as pilot
API=base.API;SNAPSHOT=base.SNAPSHOT;TRACKS={'raw':None}
GROUPS=('A','B')

def declared_q_origin(meta,pumping_context,rain_origin):
 """Authoritative origin rate from observed/declared inputs only; raises on any inconsistent declaration."""
 group=meta.get('experiment_group')
 if group not in GROUPS:raise ValueError('Unknown D05 experiment group')
 if 'q_origin_m3d' not in meta:raise ValueError('Missing declared origin rate')
 q=float(meta['q_origin_m3d']);last=float(pumping_context[1023])
 if not np.isfinite(q) or q<0:raise ValueError('Invalid declared origin rate')
 if group=='A':
  if q!=last:raise ValueError('A origin rate must equal the last context rate')
  return q
 from p4_cases import declared_origin_rate
 if meta.get('q_last_context_day_m3d') is not None and float(meta['q_last_context_day_m3d'])!=last:raise ValueError('B last-context declaration mismatch')
 if q!=declared_origin_rate(meta['calendar_form'],meta['realization'],meta['origin_date'],float(rain_origin)):raise ValueError('B origin rate differs from deterministic calendar')
 return q

def validate(data,H,expected_ids=None):
 if set(data)!=base.INPUT_KEYS:raise ValueError('Truth-free observed schema required')
 ph=np.asarray(data['protocol_sha256'])
 if ph.shape!=() or len(str(ph.item()))!=64:raise ValueError('Scalar protocol hash required')
 int(str(ph.item()),16)
 order,ids=pilot.validate(data,H)
 if np.any(data['pumping']<0):raise ValueError('Negative pumping')
 cases={}
 for i,c in enumerate(ids['case_id']):cases.setdefault(c,[]).append(i)
 if expected_ids is not None and (set(cases)!=set(expected_ids) or len(set(expected_ids))!=len(list(expected_ids))):raise ValueError('Case coverage mismatch')
 for cid,ix in cases.items():
  if len(ix)!=4:raise ValueError('Four queries per horizon required')
  ms=[json.loads(ids['metadata_json'][i]) for i in ix]
  if sorted(m['schedule_pair'] for m in ms)!=sorted(list(base.PAIRS)*2):raise ValueError('Pair mismatch')
  if any({k:v for k,v in m.items() if k!='schedule_pair'}!={k:v for k,v in ms[0].items() if k!='schedule_pair'} for m in ms):raise ValueError('Case metadata differs')
  for key in ('head','rainfall'):
   if any(not np.array_equal(data[key][i],data[key][ix[0]]) for i in ix):raise ValueError('Paired channels differ')
  if any(not np.array_equal(data['pumping'][i,:1024],data['pumping'][ix[0],:1024]) for i in ix):raise ValueError('Paired histories differ')
  q0=declared_q_origin(ms[0],data['pumping'][ix[0],:1024],data['rainfall'][ix[0],1024])
  for i,m in zip(ix,ms):
   if (m['head_unit'],m['pumping_unit'],m['rainfall_unit'],m['daily_alignment'])!=('m','m3/d','mm/d',True):raise ValueError('Units/alignment')
   want=q0 if ids['schedule_id'][i]=='a' else (0. if m['schedule_pair']==base.PAIRS[0] else 1.5*q0)
   if not np.all(data['pumping'][i,1024:]==want):raise ValueError('Origin future mismatch')
 return order,ids

def export_arrays(cases,H,digest):
 rows=[]
 for c in cases:
  ti=c['tf_input'];q0=declared_q_origin(c['meta'],ti['pumping_context'],ti['rain'][1024]);future=old.forced_futures(q0,H)
  for p in base.PAIRS:
   for s in ('a','b'):
    if not np.array_equal(ti['future_Q'][p+'_'+s][:H],future[p+'_'+s]):raise ValueError('Case origin future mismatch')
    pid=c['case_id']+'__'+p
    rows.append(dict(head=ti['head_context'],pumping=np.r_[ti['pumping_context'],future[p+'_'+s]],rainfall=ti['rain'][:1024+H],case_id=c['case_id'],pair_id=pid,query_id=pid+'__'+s+'__H'+str(H),schedule_id=s,metadata_json=json.dumps(dict(c['meta'],schedule_pair=p),sort_keys=True)))
 out={k:np.stack([r[k] for r in rows]) for k in ('head','pumping','rainfall')}
 out.update({k:np.array([r[k] for r in rows],dtype=str) for k in ('case_id','pair_id','query_id','schedule_id','metadata_json')});out['protocol_sha256']=np.array(digest)
 validate(out,H,[c['case_id'] for c in cases]);return out

def subset(data,case_ids):
 """Rows of an exported archive for one disjoint shard, original order kept; every requested case must exist."""
 want=set(case_ids)
 if len(want)!=len(list(case_ids)):raise ValueError('Duplicate shard case id')
 keep=np.isin(data['case_id'],sorted(want))
 if set(data['case_id'][keep])!=want:raise ValueError('Shard case ids missing from archive')
 out={k:(v if k=='protocol_sha256' else v[keep]) for k,v in data.items()};return out

def prepare_track(data,H,track='raw',expected_case_ids=None):
 if track!='raw':raise ValueError('D05 raw track only')
 order,ids=validate(data,H,expected_case_ids)
 return dict(contexts=data['head'][order].astype(np.float32),covariates=np.stack([data['pumping'][order],data['rainfall'][order]],axis=1).astype(np.float32),query_id=ids['query_id'][order],pair_id=ids['pair_id'][order][::2],schedule_id=ids['schedule_id'][order],case_id=ids['case_id'][order],metadata_json=ids['metadata_json'][order])
