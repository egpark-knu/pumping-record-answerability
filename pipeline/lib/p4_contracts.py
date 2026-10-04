"""D05 immutable inheritance, cohort identity and complete-freeze execution gate."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
P4=ROOT/'results/phase4'
NEW_RHOS=(0.,.1,.5,2.)
SCALES=(.3,1.,3.)
ALL_RHOS=(0.,.1,.25,.5,1.,2.,4.)
A_COUNT=720
RAW_B_COUNT=2160
RAW_TOTAL=A_COUNT+RAW_B_COUNT
MAP_COUNT=540+A_COUNT
C_COUNT=1230+RAW_TOTAL
# COUNT_CLARIFICATION.md: the earlier packet's "2880 B" was a root wording error. Raw D05 is A720+B2160=2880.
# Do not invent a fourth form or inflate the water-curtain variant.
CALENDAR_READY=('complete','ready_for_freeze')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def verify_old():
 snap=json.loads((P4/'PREPARE_OLD_HASHES.json').read_text())
 bad=[p for p,h in snap['hashes'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
 if bad:raise PermissionError('Prior artifacts changed: '+str(bad[:10]))
 return len(snap['hashes'])

def require_frozen():
 f=json.loads((P4/'protocol_freeze.json').read_text())
 if f.get('status')!='complete_protocol_frozen':raise PermissionError('Complete D05 protocol not frozen')
 digest=sha(P4/'protocol.md')
 if digest!=f['protocol_sha256'] or digest!=(P4/'protocol.sha256').read_text().split()[0]:raise PermissionError('Protocol hash mismatch')
 if int(f.get('b_case_count',-1))!=RAW_B_COUNT or int(f.get('a_case_count',A_COUNT))!=A_COUNT:raise PermissionError('Freeze counts differ from raw D05 A720+B2160')
 calendar=P4/'CALENDAR_PREPARATION.json'
 if not calendar.is_file() or sha(calendar)!=f['calendar_preparation_sha256']:raise PermissionError('Terminal calendar preparation missing/changed')
 c=json.loads(calendar.read_text())
 # 'preparation_only' is never silently accepted: readiness needs the recorded integration evidence.
 if c.get('status') not in CALENDAR_READY or not c.get('integration_evidence'):raise PermissionError('Calendar preparation is nonterminal')
 for p,h in f['hashes'].items():
  if sha(ROOT/p)!=h:raise PermissionError('Frozen dependency changed: '+p)
 verify_old()
 return digest

def reuse_cohorts():
 cohorts={'map':[],'contradiction':[]};seen=set()
 for phase in ('phase2','phase3'):
  for m in json.loads((ROOT/'results'/phase/'cases/derived_manifest.json').read_text()):
   key=(phase,m['case_id'])
   if key in seen:raise ValueError('Duplicate source case')
   seen.add(key)
   rec={'phase':phase,'case_id':m['case_id'],'tf_sha256':m['tf_sha256'],'truth_sha256':m['truth_sha256'],'realization':m['realization'],'map_included':False}
   include=(phase=='phase2' and m['N_nominal']==6) or (phase=='phase3' and m.get('experiment_group')=='B')
   rec['map_included']=include
   rec['inclusion_reason']='D03 N6 scale1 / D04 B N6 scales.3,3' if include else 'outside comparable A fitted-map design; retained in C'
   cohorts['contradiction'].append(rec)
   if include:cohorts['map'].append(rec)
 assert len(cohorts['map'])==540 and len(cohorts['contradiction'])==1230
 return cohorts

def new_case_ids():
 """Deterministic D05 identities without generating any head series (A order: realization, layer, rho, scale)."""
 import p3_phase2_cases as d03
 import p4_calendar as cal
 A=[f'd05A_r{r:02d}_{st.sid}_rho{rho:g}_N6_scale{s:g}' for r in range(10) for st in d03.strata() for rho in NEW_RHOS for s in SCALES]
 B=[f"d05B_{form}_m{m:02d}_r{r:02d}_{l['sid']}" for r in range(10) for form in cal.FORMS for m in range(1,13) for l in cal.LAYERS]
 if len(A)!=A_COUNT or len(set(A))!=A_COUNT or len(B)!=RAW_B_COUNT or len(set(B))!=RAW_B_COUNT or set(A)&set(B):raise ValueError('D05 identity count')
 return A,B

def cohort_manifests():
 """Explicit (source_phase, case_id) joins: map = D03 N6 + D04 B + new A (1260); C = all D03 + D04 + new (4110)."""
 old=reuse_cohorts();A,B=new_case_ids()
 key=lambda r:{'source_phase':r['phase'],'case_id':r['case_id'],'realization':int(r['realization'])}
 new=lambda cid,g:{'source_phase':'phase4','case_id':cid,'realization':int(re.search(r'_r(\d\d)_',cid).group(1)),'experiment_group':g}
 mp=[key(r) for r in old['map']]+[new(c,'A') for c in A]
 C=[key(r) for r in old['contradiction']]+[new(c,'A') for c in A]+[new(c,'B') for c in B]
 for name,rows,n in (('map',mp,MAP_COUNT),('C',C,C_COUNT)):
  ks=[(r['source_phase'],r['case_id']) for r in rows]
  if len(ks)!=n or len(set(ks))!=n:raise ValueError(name+' cohort identity')
 if any(r['source_phase']=='phase4' and r['experiment_group']=='B' for r in mp):raise ValueError('B never enters map calibration')
 inmap={(r['source_phase'],r['case_id']) for r in mp}
 # p4_statistics.validate_inventory schema: cohort_A = map calibration, cohort_B = operating calendar, cohort_C = contradiction
 records=[dict(source_phase=r['source_phase'],case_id=r['case_id'],realization=r['realization'],cohort_A=(r['source_phase'],r['case_id']) in inmap,
               cohort_B=r.get('experiment_group')=='B' and r['source_phase']=='phase4',cohort_C=True) for r in C]
 return {'map':mp,'C':C,'records':records}
