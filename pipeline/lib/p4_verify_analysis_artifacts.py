"""Read-only substantive verification of completed analysis/exports; no fit or scoring."""
from pathlib import Path
import csv,json,sys,hashlib,math
from collections import Counter
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];P=R/'results/phase4';sys.path.insert(0,str(R/'lib'))
from p4_contracts import require_frozen
import p4_statistics as st
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=require_frozen();s=json.loads((P/'statistics.json').read_text());assert s['status']=='official_analysis'
with (P/'primary_rows.csv').open() as f:rows=[st.adapt_row(r) for r in csv.DictReader(f)]
records=json.loads((P/'COHORT_MANIFESTS.json').read_text())['records'];idx,counts=st.validate_inventory(rows,records,official=True)
ref_paths={r['reference_source_path']:r['reference_sha256'] for r in rows};assert len(ref_paths)==4110
for p,h in ref_paths.items():assert sha(R/p)==h,p
assert Counter(r['source_phase'] for r in rows)==Counter(phase2=2160,phase3=2760,phase4=11520)
assert all(r['tool_horizon']=='' or int(float(r['tool_horizon']))==r['lead_day'] for r in rows)
m=np.load(P/'bootstrap_draw_matrix.npy');assert m.shape==(999,10) and np.all(m.sum(axis=1)==10)
assert hashlib.sha256(m.astype('<i8').tobytes()).hexdigest()==s['bootstrap']['draw_matrix_sha256']
universe,draws=st._draws([],999,20261001);st.validate_draws(draws);assert m.tolist()==[[d[i] for i in universe] for d in draws]
with (P/'tables/threshold_crossings.csv').open() as f:cross=list(csv.DictReader(f))
assert len(cross)==1176
keys=['analysis','outcome','lead_day','pair_id','model','layer_id','rho','threshold'];assert len({tuple(r[k] for k in keys) for r in cross})==1176
nulls=[r for r in cross if r['reporting_status']=='unidentifiable'];assert all(not r['signal_ratio'] and not r.get('bootstrap.low') and not r.get('bootstrap.high') for r in nulls)
for r in cross:
 a=s['cohort_A']['models' if r['analysis']=='primary' else 'positive_rho_sensitivity'][r['outcome']][r['lead_day']][r['pair_id']][r['model']]['crossings'][r['rho']]
 if r['model']=='layer':a=a[r['layer_id']]
 if r['outcome']=='sign':a=a[r['threshold']]
 assert r['status']==a['status'] and r['reporting_status']==a['reporting_status']
 if a['signal_ratio'] is not None:assert math.isclose(float(r['signal_ratio']),a['signal_ratio'],rel_tol=1e-14)
 assert int(r['bootstrap.n_identified'])+int(r['bootstrap.n_not_identified'])==999
B=[r for r in rows if idx[(r['source_phase'],r['case_id'])]['cohort_B']];assert len(B)==8640
zero=[r for r in B if r['q_origin_m3d']==0];assert len(zero)==3672 and all(r['W_m']==0 and r['E_true_m']==0 and st.width_event(r) is None for r in zero)
cal=s['cohort_B']['storage_cells'];assert len(cal)==432 and all(c['denominator']==20 for c in cal)
assert sum(c['n_no_active_contrast'] for c in cal)==3672
with (P/'tables/calendar_A_map_projections.csv').open() as f:projections=list(csv.DictReader(f))
assert len(projections)==8640 and sum(r['support_status']=='nonpositive_origin_signal' for r in projections)==3672
assert all(r['projection_only']=='True' and r['map_prediction_used_for_calendar_outcome']=='False' for r in projections)
with (P/'tables/normalization_raw_plotinput.csv').open() as f:normal=list(csv.DictReader(f))
assert len(normal)==4920 and len({(r['case_id'],r['pair_id'],r['lead_day']) for r in normal})==4920
products=json.loads((P/'MAP_SUPPLEMENT_PRODUCTS.json').read_text());visual=json.loads((P/'MAP_SUPPLEMENT_VISUAL_INSPECTION.json').read_text());assert len(visual['notes'])==12
for r in products['products']:
 p=Path(r['path']);assert sha(p)==r['sha256']
 if p.suffix=='.png':
  with Image.open(p) as im:im.verify()
 elif p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF') and b'TimesNewRoman' in p.read_bytes() and b'%%EOF' in p.read_bytes()[-1024:]
 elif p.suffix=='.svg':assert '<svg' in p.read_text() and 'Times New Roman' in p.read_text()
for r in visual['notes']:assert sha(r['path'])==r['sha256'] and not r['material_issue_remaining']
for r in products['plotinputs'].values():assert sha(r['path'])==r['sha256']
old=json.loads((P/'OLD_FIGURE2_MANIFEST.json').read_text())
def checktree(j):
 if isinstance(j,dict):
  if 'path' in j and 'sha256' in j:assert sha(j['path'])==j['sha256'],j['path']
  for v in j.values():checktree(v)
 elif isinstance(j,list):
  for v in j:checktree(v)
checktree(old['products']);checktree(old['tables'])
diag=json.loads((P/'REFERENCE_DIAGNOSTICS.json').read_text());assert diag['n_cases']==4110 and len(diag['solver_false'])==3 and len(diag['partial_draws'])==126
out=dict(status='pass',protocol_sha256=protocol,frozen_dependency_count=136,old_snapshot_files=10789,coverage=counts,n_rows=len(rows),n_cases=len(idx),reference_receipts_rehashed=len(ref_paths),bootstrap_draw_matrix_sha256=s['bootstrap']['draw_matrix_sha256'],bootstrap_shape=list(m.shape),threshold_rows=len(cross),retained_unidentifiable_crossings=len(nulls),calendar_no_active_primary_rows=len(zero),calendar_storage_cells=len(cal),projection_primary_rows=len(projections),normalization_rows=len(normal),new_figures=len(visual['notes']),new_format_files=len(products['products']),all_PNG_decodes=True,all_PDF_TimesNewRoman=True,all_SVG_TimesNewRoman=True,all_actual_visual_notes_current=True,old_Figure2_and_recency_reused_unchanged=True,reference_solver_false_by_phase=dict(Counter(r['source_phase'] for r in diag['solver_false'])),reference_partial_draws_by_phase=dict(Counter(r['source_phase'] for r in diag['partial_draws'])),negative_checks=['Unidentifiable full-sample crossings have no fabricated estimates/CIs','All918 zero-origin cases retained as3672 primary rows and relative-width undefined','No diagnostic H30day10 duplicate primary rows','Calendar projection outputs never substitute map predictions for actual outcomes','All999 bootstrap multiplicities exactly canonical across components','Full16440 point-reference cohort retained despite numerical flags'])
(P/'ANALYSIS_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
