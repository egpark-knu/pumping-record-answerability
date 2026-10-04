"""Disjoint equal D05 execution shards: shard_a and shard_b each get A360 + B1080 = 1440 unique new cases.

Assignment is a fixed parity Latin rule decided before any outcome exists:
  A: actor = (realization + layer + rho_index + scale_index) mod 2
  B: actor = (realization + layer + form_index + month) mod 2
so every (layer, rho, scale) and (form, month, layer) cell splits 5/5 over realizations, every realization splits evenly,
and the six layers of one (form, month, realization) calendar split 3/3 (zero-origin cases therefore split exactly).
Both actors run the same frozen runtime; per-case output files are keyed by case_id, so disjoint sets never overwrite.
"""
import hashlib,json,re
from collections import Counter
from p4_contracts import P4,NEW_RHOS,SCALES,A_COUNT,RAW_B_COUNT,new_case_ids
import p4_calendar as cal
ACTORS=('shard_a','shard_b')
LAYERS=[l['sid'] for l in cal.LAYERS]

def parse(cid):
 if cid.startswith('d05A_'):
  m=re.fullmatch(r'd05A_r(\d\d)_(\w+_T\d+)_rho([0-9.]+)_N6_scale([0-9.]+)',cid)
  return dict(group='A',realization=int(m[1]),sid=m[2],rho=float(m[3]),scale=float(m[4]))
 m=re.fullmatch(r'd05B_(water_curtain|paddy_irrigation|domestic_continuous)_m(\d\d)_r(\d\d)_(\w+_T\d+)',cid)
 return dict(group='B',form=m[1],month=int(m[2]),realization=int(m[3]),sid=m[4])

def actor_of(cid):
 k=parse(cid);r,l=k['realization'],LAYERS.index(k['sid'])
 if k['group']=='A':return ACTORS[(r+l+NEW_RHOS.index(k['rho'])+SCALES.index(k['scale']))%2]
 return ACTORS[(r+l+cal.FORMS.index(k['form'])+k['month'])%2]

def shards():
 A,B=new_case_ids();out={a:dict(A=[],B=[]) for a in ACTORS}
 for cid in A:out[actor_of(cid)]['A'].append(cid)
 for cid in B:out[actor_of(cid)]['B'].append(cid)
 return out

def balance(ids,zero_origin=None):
 ks=[parse(c) for c in ids];A=[k for k in ks if k['group']=='A'];B=[k for k in ks if k['group']=='B']
 rep=dict(n=len(ids),A=len(A),B=len(B),by_realization=dict(Counter(k['realization'] for k in ks)),by_layer=dict(Counter(k['sid'] for k in ks)),
          A_by_rho=dict(Counter(str(k['rho']) for k in A)),A_by_scale=dict(Counter(str(k['scale']) for k in A)),
          B_by_form=dict(Counter(k['form'] for k in B)),B_by_month=dict(Counter(k['month'] for k in B)),
          A_cell_min_max=[min(Counter((k['sid'],k['rho'],k['scale']) for k in A).values()),max(Counter((k['sid'],k['rho'],k['scale']) for k in A).values())],
          B_cell_min_max=[min(Counter((k['form'],k['month'],k['sid']) for k in B).values()),max(Counter((k['form'],k['month'],k['sid']) for k in B).values())])
 if zero_origin is not None:rep['B_zero_origin']=sum(c in zero_origin for c in ids)
 return rep

def write(zero_origin=None):
 sh=shards();A,B=new_case_ids();man={}
 allids=[c for a in ACTORS for g in 'AB' for c in sh[a][g]]
 if len(allids)!=A_COUNT+RAW_B_COUNT or set(allids)!=set(A)|set(B):raise ValueError('Shards do not cover D05 exactly once')
 for a in ACTORS:
  ids=sh[a]['A']+sh[a]['B']
  if len(sh[a]['A'])!=A_COUNT//2 or len(sh[a]['B'])!=RAW_B_COUNT//2:raise ValueError('Unequal shard')
  rec=dict(actor=a,case_ids=ids,n_cases=len(ids),n_A=len(sh[a]['A']),n_B=len(sh[a]['B']),rule=__doc__.split('\n')[2:5],
           case_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),balance=balance(ids,zero_origin),
           outputs=dict(wb_per_case='results/phase4/wb/{W,reference,truth_eval,markers}/<case_id>.*',wb_logs=f'results/phase4/wb/logs/{a}_proc<k>.log',
                        tools=f'results/phase4/tools/{a}/',state=f'results/phase4/tools/{a}/EXECUTION_STATE.json'))
  path=P4/f'SHARD_{a.upper()}.json';path.write_text(json.dumps(rec,indent=1)+'\n');man[a]=dict(path=str(path.relative_to(P4.parents[1])),case_ids_sha256=rec['case_ids_sha256'],n=len(ids))
 if set(sh['shard_a']['A']+sh['shard_a']['B'])&set(sh['shard_b']['A']+sh['shard_b']['B']):raise ValueError('Shards overlap')
 return man

def load(actor):
 rec=json.loads((P4/f'SHARD_{actor.upper()}.json').read_text())
 if hashlib.sha256('\n'.join(rec['case_ids']).encode()).hexdigest()!=rec['case_ids_sha256'] or rec['n_cases']!=1440:raise ValueError('Shard manifest changed')
 if any(actor_of(c)!=actor for c in rec['case_ids']):raise ValueError('Shard rule mismatch')
 return rec['case_ids']
