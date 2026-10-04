"""All-lead reference/truth metrics and reusable D03+D04 data index. No tool scoring or scientific report."""
import csv,datetime,json
from pathlib import Path
import p3_phase3_cases as pc
import p3_phase2_metrics as mt
P3=pc.P3;ROOT=pc.ROOT
man=json.loads((P3/'cases/derived_manifest.json').read_text());by={m['case_id']:m for m in man}
source=P3/'wb/tables/curves_long.csv';rows=[]
for r in csv.DictReader(source.open()):
    m=by[r['case_id']];E,Er,W,lo,hi=[float(r[k]) for k in ('E_true_m','ref_E_m','W_m','inf_m','sup_m')]
    metrics=mt.row_metrics(E,Er,W,lo,hi)
    rows.append(dict(case_id=r['case_id'],pair_id=r['pair_id'],lead_day=int(r['lead_day']),experiment_group=m['experiment_group'],sid=m['sid'],realization=m['realization'],
                     storage_type=m['storage_type'],storage_value=m['storage_value'],T_m2_d=m['T_m2_d'],c_d=m['c_d'],Q_scale=m['Q_scale'],recency_label=m['recency_label'],recency_ratio=m['recency_ratio'],
                     SR=m['SR'],pi_r=m['pi_r'],rho_realized=m['rho_realized'],t95_d=m['t95_d'],lead_over_t95=int(r['lead_day'])/m['t95_d'],
                     E_true_m=E,ref_E_m=Er,inf_m=lo,sup_m=hi,W_m=W,sign_determined=bool(lo>0 or hi<0),W_over_abs_E_true=W/abs(E) if E!=0 else None,**{'ref_'+k:v for k,v in metrics.items()}))
assert len(rows)==41400
out=P3/'wb/tables/reference_metrics_long.csv'
with out.open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
index=dict(at=datetime.datetime.now().astimezone().isoformat(),n_new_cases=690,n_reused_D03_cases=540,n_combined_unique_ids=1230,n_new_pairs=1380,n_new_lead_rows=41400,
           D03_scale1_N6_reuse_count=180,n_duplicate_ids=0,notes=['D03 results read-only; no regenerated old scores','C .001 new requested cells remain distinct IDs despite identical D03 inputs', 'Tool inference/normalization A-owned; aggregation/figures C-owned'],
           files={str(p.relative_to(ROOT)):pc.sha(p) for p in [source,out,P3/'wb/tables/W_table.csv',P3/'cases/derived_manifest.json',pc.d03.P2/'wb/tables/curves_long.csv',pc.d03.P2/'wb/tables/W_table.csv',pc.d03.P2/'cases/derived_manifest.json']},code_sha256=pc.sha(__file__))
old=json.loads((pc.d03.P2/'cases/derived_manifest.json').read_text());assert len({m['case_id'] for m in man+old})==1230
(P3/'B_D_E_INPUT_INDEX.json').write_text(json.dumps(index,indent=1));print(json.dumps({k:v for k,v in index.items() if k!='files'}))
