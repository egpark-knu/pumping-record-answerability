"""B independent report aggregation from actual immutable C cells and W/truth curves; no model rerun."""
import json,hashlib,datetime
from pathlib import Path
import pandas as pd,numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/phase3';f=pd.read_csv(P/'cell_metrics.csv');lead=f[f.quantity.eq('lead')];raw=lead[lead.track.eq('raw')];r=json.loads((P/'READOUT.json').read_text())
def stat(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)];return dict(n=len(v),median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),negative=int((v<0).sum()))
def difference(d,index,col,a,b,value):
 x=d.pivot(index=index,columns=col,values=value);return stat(x[b]-x[a])
A={}
for pair in raw.pair_id.unique():
 for k in (10,30):
  d=raw[(raw.experiment_group=='A')&(raw.pair_id==pair)&(raw.lead_day==k)]
  for rec in ('recent','old'):
   x=d[d.recency_label==rec]
   A[f'{pair}|{k}|N18-N2|{rec}']={v:difference(x,['sid','realization'],'N_transitions',2,18,v) for v in ['W_m','effect_ratio']}
  for N in (2,18):
   x=d[d.N_transitions==N];A[f'{pair}|{k}|old-recent|N{N}']={v:difference(x,['sid','realization'],'recency_label','recent','old',v) for v in ['W_m','effect_ratio']}
 d=f[(f.experiment_group=='A')&(f.track=='raw')&(f.pair_id==pair)&(f.quantity=='max_days_1_10')]
 for rec in ('recent','old'):A[f'{pair}|max1-10|N18-N2|{rec}']=difference(d[d.recency_label==rec],['sid','realization'],'N_transitions',2,18,'W_m')
 for N in (2,18):A[f'{pair}|max1-10|old-recent|N{N}']=difference(d[d.N_transitions==N],['sid','realization'],'recency_label','recent','old','W_m')
S={};G={};Norm={}
for (pair,track,k),d in lead.groupby(['pair_id','track','lead_day']):
 key=f'{pair}|{track}|{int(k)}';truth=d.E_true_m.to_numpy();tool=d.E_tool_m.to_numpy();finite=np.isfinite(truth)&np.isfinite(tool);nonzero=finite&(truth!=0);err=nonzero&(np.sign(tool)!=np.sign(truth));det=d.sign_determined.eq(1);con=d.record_contradiction.eq(1)
 S[key]=dict(n=len(d),truth_nonfinite=int((~np.isfinite(truth)).sum()),truth_zero=int((truth==0).sum()),truth_nonzero=int((np.isfinite(truth)&(truth!=0)).sum()),ratio_defined=int(d.ratio_defined.sum()),sign_denominator=int(nonzero.sum()),sign_correct=int((nonzero&~err).sum()),truth_wrong=int(err.sum()),contradictions=int(con.sum()),truth_wrong_and_contradiction=int((err&con).sum()),fraction_all_errors_record_contradictions=float((err&con).sum()/err.sum()) if err.sum() else None,ratio=stat(d.effect_ratio),abs_error=stat(abs(d.E_tool_m-d.E_true_m)),error_over_W=stat(d.error_over_W),envelope_in=int(d.envelope_includes.sum()),envelope_denominator=int(np.isfinite(d.env_inf_m).sum()),small_001=int((abs(truth)<=.001).sum()),small_02=int((abs(truth)<=.02).sum()),unconfined_n=int(d.unconfined.sum()))
 for name,ix in [('determined',det),('ambiguous',~det),('unconfined_determined',det&d.unconfined.eq(1)),('unconfined_ambiguous',~det&d.unconfined.eq(1)),('fixedc_linear',d.storage_type.eq('fixedc_linear'))]:
  z=d[ix];G[key+'|'+name]=dict(n=len(z),truth_errors=int(err[ix].sum()),contradictions=int(con[ix].sum()),envelope_in=int(z.envelope_includes.sum()),ratio=stat(z.effect_ratio) if len(z) else None)
 sp=spearmanr(d.normalized_future_size,d.effect_ratio);Norm[key]=dict(n=len(d),spearman_rho=float(sp.statistic))
B={}
b=raw[((raw.experiment_group=='B')|((raw.dataset=='D03')&(raw.N_transitions==6)))]
for (pair,k),d in b.groupby(['pair_id','lead_day']):
 for scale in (.3,3):
  B[f'{pair}|{int(k)}|scale{scale}']={v:difference(d[d.Q_scale.isin([1,scale])],['sid','realization','rho'],'Q_scale',1,scale,v) for v in ['W_m','effect_ratio']}
Truth={}
for phase in ('phase2','phase3'):
 rows=[];miss=[]
 for file in (ROOT/'results'/phase/'wb/W').glob('*.json'):
  w=json.loads(file.read_text());t=json.loads((ROOT/'results'/phase/'wb/truth_eval'/file.name).read_text())
  for pair,env in w['envelope'].items():
   for idx,(e,lo,hi) in enumerate(zip(t['E_true'][pair],env['inf'],env['sup'])):
    inside=lo-1e-12<=e<=hi+1e-12;rows.append(inside)
    if not inside:miss.append(dict(case_id=file.stem,pair=pair,lead=idx+1,true=e,inf=lo,sup=hi))
 Truth[phase]=dict(n_curves_values=len(rows),inside=sum(rows),misses=miss)
# Existing C readout facts must agree; existing W lead pairing is day10 only.
check={}
for pair in raw.pair_id.unique():
 for k in (10,30):
  key=f'{k}|{pair}|raw';g=r['D']['groups'][key];s=S[f'{pair}|raw|{k}'];check[key]=s['contradictions']==g['record_contradiction']['numerator'] and s['envelope_in']==g['envelope_inclusion']['numerator'] and abs(Norm[f'{pair}|raw|{k}']['spearman_rho']-r['E']['overall'][key]['size']['rho'])<1e-12
assert all(check.values())
out=dict(at=datetime.datetime.now().astimezone().isoformat(),A=A,B=B,tool_summaries=S,sign_groups=G,normalization=Norm,truth_actual_inclusion=Truth,C_readout_reproduced=check,coverage=dict(cases=f.case_id.nunique(),lead_rows=len(lead),max_rows=len(f)-len(lead)),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'cell_metrics.csv',P/'READOUT.json',Path(__file__).resolve()]})
(P/'B_REPORT_NUMBERS.json').write_text(json.dumps(out,indent=1));print(json.dumps({'coverage':out['coverage'],'C_readout_reproduced':check,'truth':Truth,'raw_summaries':{k:v for k,v in S.items() if '|raw|' in k}},indent=1))
