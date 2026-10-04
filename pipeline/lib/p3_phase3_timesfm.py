"""D04 input validation/preparation, same frozen pilot adapter/API/filter.
D03 grid-label validator cannot accept D04 fixed-c storage layers; this validates
observed channels and schedule semantics, not D03's former physical grid.
"""
import json
import numpy as np
import p3_phase2_timesfm as base
import timesfm_pilot_adapter as pilot
TRACKS=base.TRACKS;API=base.API;SNAPSHOT=base.SNAPSHOT

def prepare_track(data,H,track,expected_case_ids=None):
 if set(data)!=base.INPUT_KEYS:raise ValueError('Unexpected model input field (truth leakage)')
 if track not in TRACKS:raise ValueError('Unprespecified track')
 order,ids=pilot.validate(data,H)
 cases={c:np.flatnonzero(ids['case_id']==c) for c in set(ids['case_id'])}
 if expected_case_ids is not None and set(cases)!=set(expected_case_ids):raise ValueError('Full new690 case coverage mismatch')
 if (data['pumping']<0).any():raise ValueError('Negative pumping')
 for cid,ix in cases.items():
  if len(ix)!=4:raise ValueError('Two schedule pairs required')
  for k in ['head','rainfall']:
   if any(not np.array_equal(data[k][i],data[k][ix[0]]) for i in ix):raise ValueError('Paired target/weather mismatch')
  if any(not np.array_equal(data['pumping'][i,:1024],data['pumping'][ix[0],:1024]) for i in ix):raise ValueError('Paired pumping history mismatch')
  metas=[json.loads(ids['metadata_json'][i]) for i in ix]
  if {m['schedule_pair'] for m in metas}!=set(base.PAIRS):raise ValueError('Pair names mismatch')
  if any((m['head_unit'],m['pumping_unit'],m['rainfall_unit'],m['daily_alignment'])!=('m','m3/d','mm/d',True) for m in metas):raise ValueError('Units/alignment mismatch')
  last=float(data['pumping'][ix[0],1023])
  for pair in base.PAIRS:
   rows=[i for i in ix if json.loads(ids['metadata_json'][i])['schedule_pair']==pair]
   for i in rows:
    want=last if ids['schedule_id'][i]=='a' else (0 if pair==base.PAIRS[0] else 1.5*last)
    if not np.all(data['pumping'][i,1024:]==want):raise ValueError('Future pumping does not continue exact observed origin')
 q=data['pumping'][order];feature=q if TRACKS[track] is None else base.exp_filter(q,TRACKS[track])
 return dict(contexts=data['head'][order].astype(np.float32),covariates=np.stack([feature,data['rainfall'][order]],axis=1).astype(np.float32),
  query_id=ids['query_id'][order],pair_id=ids['pair_id'][order][::2],schedule_id=ids['schedule_id'][order],case_id=ids['case_id'][order],metadata_json=ids['metadata_json'][order])
