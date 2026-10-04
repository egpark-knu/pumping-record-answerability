"""Non-scoring D05 table exporter and standalone figure producer. Never fits or scores."""
from pathlib import Path
import csv,json,hashlib,math,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.special import expit
R=Path(__file__).resolve().parents[1];P=R/'results/phase4';sys.path.insert(0,str(R/'lib'))
import p4_statistics as st
from p4_contracts import require_frozen
S=json.loads((P/'statistics.json').read_text());assert S['bootstrap']['n']==999
require_frozen()
F=P/'figures';F.mkdir(exist_ok=True);T=P/'tables';T.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(p,j): p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def flat(j,prefix=''):
 out={}
 for k,v in j.items():
  key=prefix+k
  if isinstance(v,dict):out.update(flat(v,key+'.'))
  elif isinstance(v,list):out[key]=json.dumps(v,allow_nan=False)
  else:out[key]=v
 return out
def table(name,rows):
 rows=[flat(r) for r in rows];keys=list(dict.fromkeys(k for r in rows for k in r))
 with (T/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 return rows
with (P/'primary_rows.csv').open() as f: rows=[st.adapt_row(r) for r in csv.DictReader(f)]
records=json.loads((P/'COHORT_MANIFESTS.json').read_text())['records'];idx={(r['source_phase'],r['case_id']):r for r in records}
A=[r for r in rows if idx[(r['source_phase'],r['case_id'])]['cohort_A']];B=[r for r in rows if idx[(r['source_phase'],r['case_id'])]['cohort_B']]
assert len(A)==5040 and len(B)==8640
thresholds=[];comparisons=[];modelrows=[]
for analysis,models in [('primary',S['cohort_A']['models']),('positive_rho',S['cohort_A']['positive_rho_sensitivity'])]:
 for outcome,block in models.items():
  for lead,pairs in block.items():
   for pair,payload in pairs.items():
    comparisons.append(dict(analysis=analysis,outcome=outcome,lead_day=lead,pair_id=pair,**payload['layer_comparison']))
    for name in ('no_layer','layer'):
     model=payload[name];modelrows.append(dict(analysis=analysis,outcome=outcome,lead_day=lead,pair_id=pair,model=name,**{k:v for k,v in model.items() if k!='crossings'}))
     for rho,node in model['crossings'].items():
      layers=node if name=='layer' else {'pooled':node}
      for layer,points in layers.items():
       probs=points if outcome=='sign' else {'width1':points}
       for prob,point in probs.items(): thresholds.append(dict(point,analysis=analysis,outcome=outcome,lead_day=lead,pair_id=pair,model=name,layer_id=layer,rho=float(rho),threshold=prob))
table('threshold_crossings',thresholds);table('layer_explanatory_gain',comparisons);table('model_parameters_and_status',modelrows)
for name,cells in S['cohort_B'].items():
 if name.endswith('cells'):table('calendar_'+name,cells)
C=S['cohort_C']['table']
for name,cells in C.items():
 if name.startswith('by_'):table('contradiction_'+name,cells)
table('undetermined_truth_errors',C['undetermined_absolute_errors']);table('contradiction_overall',[{k:v for k,v in C.items() if not isinstance(v,list)}])
table('coverage_dispositions',[dict(section='inventory',**S['inventory']),dict(section='map',**{k:v for k,v in S['cohort_A'].items() if k.startswith('n_')})])
table('row_dispositions',[dict(source_phase=r['source_phase'],case_id=r['case_id'],pair_id=r['pair_id'],lead_day=r['lead_day'],SR=r['SR'],SR_definition=r['SR_definition'],W_status=r['W_status'],W_flags=r.get('W_flags'),sign_determined=r['sign_determined'],zero_truth=r['E_true_m']==0,zero_width=r['W_m']==0,relative_width_undefined=st.width_event(r) is None,zero_tool=r['E_tool_m']==0,tool_missing=r['E_tool_m'] is None,reference_status=r['reference_status'],reference_missing=r['E_reference_m'] is None) for r in rows])
np.save(P/'bootstrap_draw_matrix.npy',np.asarray(S['bootstrap']['draw_matrix'],dtype='<i8'))
assert sha(P/'bootstrap_draw_matrix.npy') # .npy container hash distinct from prescribed raw matrix hash
font=font_manager.findfont('Times New Roman',fallback_to_default=False);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':'Times New Roman','axes.labelweight':'bold','axes.labelsize':13,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':200})
products=[];plotinputs={}
def export(fig,name,data):
 inp=T/(name+'_plotinput.json');savejson(inp,data);plotinputs[name]=dict(path=str(inp),sha256=sha(inp))
 for ext in ('png','pdf','svg'):
  p=F/(name+'.'+ext);fig.savefig(p,bbox_inches='tight');products.append(dict(path=str(p),sha256=sha(p)))
 plt.close(fig)
colors=['#2c6d99','#ab5038','#426d45'];ticks=[-1.65,*np.log10(st.EVAL_RHO[1:])];labels=['0','.1','.25','.5','1','2','4']
ZERO_X=-1.65
def coordinate(rho):return ZERO_X if rho==0 else math.log10(rho)
def support(subset,layer=None):return st._support_from_rows([r for r in subset if r['SR'] and r['SR']>0 and (layer is None or r['layer_id']==layer)])
def fitdict(model,subset):
 return dict(model,support=support(subset),layer_support={l:support(subset,l) for l in st.LAYER_ORDER})
def frame(ax,subset,lead,ylabel=True):
 ax.axvspan(ZERO_X-.15,ZERO_X+.25,color='#f2f2f2',zorder=-3);ax.axvline(ZERO_X+.3,color='.7',ls=':');ax.set_xlim(-1.8,.68);ax.set_xticks(ticks,labels);
 if ZERO_X < -2: ax.set_xlim(-2.8,1.4);ax.set_xticks([ZERO_X,*np.log10([.01,.03,.1,.3,1,3,10,20])],['0','.01','.03','.1','.3','1','3','10','20'])
 ax.set_yscale('log');ax.set_ylim(.001,20);ax.set_xlabel('Pause / response time, ρ');ax.set_ylabel('Signal ratio' if ylabel else '');ax.set_title(f'{lead}-day lead',fontsize=12,fontweight='bold');ax.grid(alpha=.12)
 good=[r for r in subset if r['SR'] and r['SR']>0]
 ax.scatter([coordinate(r['rho']) for r in good],[r['SR'] for r in good],s=8,color='.6',alpha=.16,zorder=-1)
def lines(ax,subset,models,layer=None,sensitivity=False):
 inputs=[];notes=[]
 for outcome,prob,color,ls in [('sign',.5,colors[0],'-'),('sign',.9,colors[1],'--'),('magnitude',None,colors[2],'-.')]:
  fit=fitdict(models[outcome],subset)
  if fit['status']!='ok':notes.append(f'{outcome}: {fit["status"]}');continue
  values=[];rhos=np.geomspace(.1,4,240)
  for rho in rhos:
   p=st.crossing_from_fit(fit,prob,float(rho),layer) if outcome=='sign' else st.magnitude_crossing_from_fit(fit,float(rho),layer)
   y=p.get('signal_ratio');inside=p.get('reporting_status')=='finite_inside'
   values.append(y if inside and y is not None and .001<=y<=20 else np.nan);inputs.append(dict(p,outcome=outcome,probability=prob,rho=float(rho),layer_id=layer))
  if any(np.isfinite(values)):ax.plot(np.log10(rhos),values,color=color,ls=ls,lw=1.8)
  else:notes.append(f'{"p="+str(prob) if prob else "width=1"}: outside support')
  if not sensitivity:
   p=st.crossing_from_fit(fit,prob,0.,layer) if outcome=='sign' else st.magnitude_crossing_from_fit(fit,0.,layer)
   if p.get('reporting_status')=='finite_inside' and p.get('signal_ratio') is not None:ax.plot([ZERO_X-.1,ZERO_X+.15],[p['signal_ratio']]*2,color=color,ls=ls,lw=1.8)
   inputs.append(dict(p,outcome=outcome,probability=prob,rho=0.,layer_id=layer))
 if notes:ax.text(.02,.02,'\n'.join(dict.fromkeys(notes)),transform=ax.transAxes,fontsize=8,va='bottom',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
 return inputs
legend=[Line2D([],[],color=c,ls=l,label=n) for c,l,n in zip(colors,['-','--','-.'],['Direction probability 0.5','Direction probability 0.9','Relative width = 1'])]
projection=[]
for r in B:
 subset=[a for a in A if a['pair_id']==r['pair_id'] and a['lead_day']==r['lead_day']];sup=support(subset);rho=st._float(r.get('rho_last_off'));sr=r['SR'];inside=None if sr is None or sr<=0 or rho is None else sup.contains(math.log(sr),rho)
 projection.append(dict(case_id=r['case_id'],pair_id=r['pair_id'],lead_day=r['lead_day'],calendar_type=r['calendar_type'],origin_month=r['origin_month'],layer_id=r['layer_id'],rho_last_off=rho,SR_origin=sr,SR_context=st._float(r.get('SR_context_mean')),map_SR_definition='outside_designated_rest',projection_SR_definition=r['SR_definition'],projection_only=True,map_prediction_used_for_calendar_outcome=False,support_status='nonpositive_origin_signal' if sr is not None and sr<=0 else 'unknown_coordinate' if inside is None else 'inside_pooled_A_hull' if inside else 'outside_pooled_A_hull',sign_determined=r['sign_determined'],relative_width_below_one=st.width_event(r)))
table('calendar_A_map_projections',projection)
for pair in st.PAIRS:
 tag='P1' if pair==st.PAIRS[0] else 'P2'
 for proj in (False,True):
  ZERO_X=-2.65 if proj else -1.65
  fig,axes=plt.subplots(1,2,figsize=(10,4.7));fig.subplots_adjust(bottom=.26,wspace=.26)
  data=[]
  for ax,lead in zip(axes,st.LEADS):
   subset=[r for r in A if r['pair_id']==pair and r['lead_day']==lead];frame(ax,subset,lead)
   models={o:S['cohort_A']['models'][o][str(lead)][pair]['no_layer'] for o in ('sign','magnitude')};data+=lines(ax,subset,models)
   if proj:
    for form,mark,c in zip(st.CALENDAR_TYPES,['o','s','^'],colors):
     ps=[p for p in projection if p['pair_id']==pair and p['lead_day']==lead and p['calendar_type']==form and p['SR_origin']>0 and p['rho_last_off'] is not None ]
     ax.scatter([coordinate(p['rho_last_off']) for p in ps],[p['SR_origin'] for p in ps],marker=mark,facecolors='none',edgecolors=c,s=20,alpha=.25)
    ax.set_ylabel('Origin-rate signal ratio (projection)');ax.set_xlabel('Last observed pause / response time')
    hidden=sum(p['pair_id']==pair and p['lead_day']==lead and (p['SR_origin']<=0 or p['rho_last_off'] is None ) for p in projection)
    ax.text(.99,.98,f'{hidden} zero-rate origins (not plotted)',ha='right',va='top',transform=ax.transAxes,fontsize=9)
  handles=legend if not proj else legend+[Line2D([],[],marker=m,color=c,ls='',markerfacecolor='none',label=l) for m,c,l in zip(['o','s','^'],colors,['Water curtain','Paddy irrigation','Domestic continuous'])]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=3)
  export(fig,('calendar_projection_' if proj else 'figure1_answerability_')+tag,dict(contours=data,points=[dict(case_id=r['case_id'],rho=r['rho'],SR=r['SR']) for r in A if r['pair_id']==pair] if not proj else [p for p in projection if p['pair_id']==pair],coordinate_caution='B origin-rate SR and last observed pause are projections on A controlled outside-rest SR/prescribed-pause contours; not predictions.' if proj else 'Primary controlled map; zero-pause strip separate from positive logarithmic axis.'))
 # six-layer offset maps
 ZERO_X=-1.65
 fig,axes=plt.subplots(3,2,figsize=(10,11));fig.subplots_adjust(bottom=.11,hspace=.5,wspace=.28);data=[]
 for ax,layer in zip(axes.flat,st.LAYER_ORDER):
  subset=[r for r in A if r['pair_id']==pair and r['layer_id']==layer and r['lead_day']==10]
  frame(ax,subset,10);ax.set_title(layer.replace('_',' '),fontsize=12)
  # show day10 solid requested contours; day30 in table and second producer below
  models={o:S['cohort_A']['models'][o]['10'][pair]['layer'] for o in ('sign','magnitude')};data+=lines(ax,subset,models,layer=layer)
 fig.legend(handles=legend,loc='lower center',ncol=3)
 export(fig,'supplement_layer_10day_'+tag,dict(contours=data,lead_day=10,model='five layer indicators; day30 companion and all thresholds in CSV'))
 fig,axes=plt.subplots(3,2,figsize=(10,11));fig.subplots_adjust(bottom=.11,hspace=.5,wspace=.28);data=[]
 for ax,layer in zip(axes.flat,st.LAYER_ORDER):
  subset=[r for r in A if r['pair_id']==pair and r['layer_id']==layer and r['lead_day']==30];frame(ax,subset,30);ax.set_title(layer.replace('_',' '),fontsize=12)
  models={o:S['cohort_A']['models'][o]['30'][pair]['layer'] for o in ('sign','magnitude')};data+=lines(ax,subset,models,layer=layer)
 fig.legend(handles=legend,loc='lower center',ncol=3);export(fig,'supplement_layer_30day_'+tag,dict(contours=data,lead_day=30))
# Scale widths and normalized physical answerability, full A; descriptive medians/IQR only.
scale=[]
for pair in st.PAIRS:
 for lead in st.LEADS:
  for layer in st.LAYER_ORDER:
   for q in st.MAP_SCALES:
    rs=[r for r in A if r['pair_id']==pair and r['lead_day']==lead and r['layer_id']==layer and r['Q_scale']==q]
    w=[r['W_m'] for r in rs];v=[abs(r['E_true_m'])/r['W_m']/q for r in rs if r['W_m']>0]
    scale.append(dict(pair_id=pair,lead_day=lead,layer_id=layer,Q_scale=q,n=len(rs),W_median=float(np.median(w)),W_q25=float(np.percentile(w,25)),W_q75=float(np.percentile(w,75)),scaled_answerability_median=float(np.median(v))))
table('scale_width_summary',scale)
fig,axes=plt.subplots(2,2,figsize=(10,8));fig.subplots_adjust(bottom=.18,hspace=.35)
for pi,pair in enumerate(st.PAIRS):
 for li,lead in enumerate(st.LEADS):
  ax=axes[pi,li]
  for i,layer in enumerate(st.LAYER_ORDER):
   rs=[s for s in scale if s['pair_id']==pair and s['lead_day']==lead and s['layer_id']==layer];x=[s['Q_scale'] for s in rs];y=[s['W_median'] for s in rs];ax.errorbar(x,y,yerr=[[s['W_median']-s['W_q25'] for s in rs],[s['W_q75']-s['W_median'] for s in rs]],marker='o',capsize=2,label=layer.replace('_',' '))
  ax.set_xscale('log');ax.set_xticks([.3,1,3],['0.3','1','3']);ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_xlabel('Pumping-rate multiplier');ax.set_ylabel('Envelope width (m)');ax.set_title(f'P{pi+1}, {lead}-day lead');ax.grid(alpha=.15)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3);export(fig,'supplement_pumping_scale',dict(summary=scale,summary_type='Median and interquartile range over all controlled-map rhos and realizations; no new fit.'))
old=json.loads((R/'results/phase3/READOUT.json').read_text());storage=old['C']['strata'];table('old_D04C_storage_summary',storage)
table('old_D04B_paired_scale',[dict(contrast=k,**v) for k,v in old['B']['paired'].items()]);table('calendar_monthly_contradiction_concentration',[r for r in S['cohort_C']['table']['by_calendar_month_form'] if r.get('calendar_type') is not None]);table('old_normalization_correlations',[dict(contrast=k,**v) for k,v in old['E']['overall'].items() if k.endswith('|raw')])
fig,axes=plt.subplots(2,2,figsize=(10,7));fig.subplots_adjust(hspace=.4,bottom=.12)
for pi,pair in enumerate(st.PAIRS):
 for li,lead in enumerate(st.LEADS):
  rs=[s for s in storage if s['pair']==pair and s['lead']==lead];rs.sort(key=lambda r:r['storage_value']);ax=axes[pi,li];ax.plot([s['storage_value'] for s in rs],[s['median_W_over_abs_E'] for s in rs],marker='o');ax.set_xscale('log');ax.set_xlabel('Storage coefficient, S');ax.set_ylabel('Median relative width');ax.set_title(f'P{pi+1}, {lead}-day lead');ax.grid(alpha=.15)
export(fig,'supplement_storage_coefficient',dict(exact_old_D04C=storage,source='results/phase3/READOUT.json',source_sha256=sha(R/'results/phase3/READOUT.json'),interpretation='T, c and gain fixed; no coupled six-regime substitution.'))
# Original raw normalization diagnostic, retain both-pair correlation reversal; no new estimator.
normal=pd.read_csv(R/'results/phase3/normalization.csv');metrics=pd.read_csv(R/'results/phase3/cell_metrics.csv');normal=normal[normal.track=='raw'];metrics=metrics[(metrics.track=='raw')&(metrics.quantity=='lead')]
normal['lead_day']=normal['horizon'];normal['pair_id']=normal['schedule_pair'];joined=normal.merge(metrics[['case_id','pair_id','lead_day','effect_ratio']],on=['case_id','pair_id','lead_day'],validate='one_to_one');assert len(joined)==4920
joined[['case_id','pair_id','lead_day','normalized_future_size','normalized_future_change','effect_ratio']].to_csv(T/'normalization_raw_plotinput.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(10,8));fig.subplots_adjust(hspace=.4,wspace=.28)
for pi,pair in enumerate(st.PAIRS):
 for li,lead in enumerate(st.LEADS):
  rs=joined[(joined.pair_id==pair)&(joined.lead_day==lead)];ax=axes[pi,li];ax.scatter(rs.normalized_future_size,rs.effect_ratio,s=5,alpha=.18);ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1);ax.axhline(0,color='.7',lw=.7);ax.set_xlabel('Normalized future pumping size');ax.set_ylabel('Raw effect / true effect');corr=old['E']['overall'][f'{lead}|{pair}|raw']['size']['rho'];ax.set_title(f'P{pi+1}, {lead}-day lead; Spearman ρ = {corr:.3f}')
export(fig,'supplement_raw_normalization',dict(source='tables/normalization_raw_plotinput.csv',source_sha256=sha(T/'normalization_raw_plotinput.csv'),old_correlations=old['E']['overall'],interpretation='Retained D03/D04 hypothesis diagnostic; opposite correlations are not a normalization mechanism conclusion.'))
# Requested prespecified positive-rho sensitivity, no new fit.
fig,axes=plt.subplots(2,2,figsize=(10,8));fig.subplots_adjust(bottom=.15,hspace=.4);data=[]
for pi,pair in enumerate(st.PAIRS):
 for li,lead in enumerate(st.LEADS):
  ax=axes[pi,li];subset=[r for r in A if r['pair_id']==pair and r['lead_day']==lead and r['rho']>0];frame(ax,subset,lead);ax.set_title(f'P{pi+1}, {lead}-day lead');models={o:S['cohort_A']['positive_rho_sensitivity'][o][str(lead)][pair]['no_layer'] for o in ('sign','magnitude')};data+=lines(ax,subset,models,sensitivity=True)
fig.legend(handles=legend,loc='lower center',ncol=3);export(fig,'supplement_positive_rho_sensitivity',dict(contours=data,excluded_boundary='rho0 changed to positive-rate pulses; prescribed sensitivity uses same999 draws.'))
savejson(P/'MAP_SUPPLEMENT_PRODUCTS.json',dict(status='rendered_pending_visual_inspection',font=dict(path=font,sha256=sha(font),family=font_manager.FontProperties(fname=font).get_name()),products=products,plotinputs=plotinputs,tables=[dict(path=str(p),sha256=sha(p)) for p in T.glob('*')],primary_input_sha256=sha(P/'primary_rows.csv'),statistics_sha256=sha(P/'statistics.json'),producer_sha256=sha(Path(__file__))))
print('RENDERED',len(products)//3,'figures',len(list(T.glob('*'))),'tables/input artifacts',flush=True)
