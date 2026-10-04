"""D05 new schedules only; frozen background, kernels, truth formulas and serialization retained.

A: positive rho keep the inherited D03 burst/rest formula byte for byte. rho=0 is the PROTOCOL_DESIGN section 3
positive low-rate pulse template: no zero day, no designated rest, six controlled rate edges, same total volume.
B: the p4_calendar daily operating calendars (unchanged module) on the same frozen background function with the
monthly context start; futures hold the calendar ORIGIN-DAY rate (including zero), not the last context rate.
Both groups share one head/truth routine (_physics), so the hydraulic generator cannot diverge between A and B.
"""
import hashlib,math
from datetime import date,timedelta
import numpy as np
import pandas as pd
import p3_generator as g
import p3_phase2_cases as d03
import p4_calendar as cal
from p4_contracts import ROOT,NEW_RHOS,SCALES,A_COUNT,RAW_B_COUNT,require_frozen
CTX=g.CTX;HMAX=g.HMAX;PAIRS=d03.PAIRS
PULSE_FACTOR=.5
FIXTURE_ANCHOR_YEAR=2004  # fixture rain spans 2000-2006; official anchors stay 2010+realization

def physical(st):
 L=dict(d03.layer(st));L['R']={rho:int(round(rho*L['t95'])) for rho in NEW_RHOS}
 # Rmax stays the D03 maximum 4*t95: unchanged compensation window and comparable lineage.
 return L

def schedule(mult,L,rho):
 if rho not in NEW_RHOS:raise ValueError('Unprespecified D05 rho')
 R=int(round(rho*L['t95']));e=L['e'];Rmax=L['Rmax'];base0=100*np.asarray(mult,float)
 C=np.zeros(CTX,bool);C[:e-Rmax]=True
 if not (0<e-Rmax and e+18<CTX):raise ValueError('Inherited schedule does not fit')
 if rho==0:
  # PROTOCOL_DESIGN section 3: three one-day half-rate pulses at e+1,e+3,e+5; C compensates the removed volume.
  J=np.zeros(CTX,bool);J[e+1:e+6:2]=True
  kappa=1+(1-PULSE_FACTOR)*base0[J].sum()/base0[C].sum();q=base0.copy();q[C]*=kappa;q[J]*=PULSE_FACTOR
  if R!=0 or not np.all(q>0):raise AssertionError('rho0 pulse template must be strictly positive with no rest')
  return dict(q=q,base_N=q.copy(),base0=base0,kappa=float(kappa),off=np.zeros(CTX,bool),pulse=J,C=C,e=e,R=0,Rmax=Rmax,template='positive_low_pulse_rho0')
 off=np.zeros(CTX,bool);off[e+1:e+6:2]=True
 kappa=1+base0[off].sum()/base0[C].sum();base=base0.copy();base[C]*=kappa;base[off]=0
 q=base.copy();q[e-R:e]=0
 return dict(q=q,base_N=base,base0=base0,kappa=float(kappa),off=off,pulse=np.zeros(CTX,bool),C=C,e=e,R=R,Rmax=Rmax,template='inherited_burst_off_rest')

def mask_edges(mask):
 m=np.asarray(mask,bool);return int(np.sum(m[1:]!=m[:-1]))

def positive_rate_changes(q):
 q=np.asarray(q,float);both=(q[1:]>0)&(q[:-1]>0);return int(np.sum(both&(q[1:]!=q[:-1])))

def zero_runs(q):
 """Maximal zero runs [start,end) of the observed context."""
 off=np.asarray(q,float)==0.;runs=[];i=0
 while i<off.size:
  if off[i]:
   j=i
   while j<off.size and off[j]:j+=1
   runs.append((i,j));i=j
  else:i+=1
 return runs

def observed_rest_features(q,t95,context_end=date(2000,1,1)):
 """Observed last and longest maximal zero runs (p4_calendar definition); null distance when no zero day."""
 last=cal.last_off_features(context_end,np.asarray(q,float),t95);runs=zero_runs(q)
 if runs:
  s,e=max(runs,key=lambda r:(r[1]-r[0],r[1]));longest=dict(longest_off_length_days=e-s,longest_off_distance_to_origin_days=CTX-e,rho_longest_off=(e-s)/t95)
 else:longest=dict(longest_off_length_days=0,longest_off_distance_to_origin_days=None,rho_longest_off=0.)
 return dict(last,**longest,n_zero_days=int(np.sum(np.asarray(q)==0)),N_on_off_observed=cal.on_off_switches(q),N_positive_rate_changes_observed=positive_rate_changes(q))

def _physics(bg,L,Qw,q_origin,Qpre):
 """Frozen D03 head/truth routine: hidden constant prehistory, exact Hantush block convolution, delta rule."""
 A=L['d']['gain'];B,S=L['B'],L['S']
 fut={PAIRS[0]:(np.full(HMAX,q_origin),np.zeros(HMAX)),PAIRS[1]:(np.full(HMAX,q_origin),np.full(HMAX,1.5*q_origin))}
 def hp(f):return A*(Qpre*(1-S[1:CTX+HMAX+1])+np.convolve(np.concatenate([Qw,f]),B)[:CTX+HMAX])
 w=slice(g.GEN_WARMUP,g.GEN_WARMUP+CTX+HMAX);h_nat,eps=bg['h_nat'][w],bg['eps'][w];hp_a=hp(fut[PAIRS[0]][0]);h_star=h_nat-hp_a
 E,delta={},{}
 for p,(qa,qb) in fut.items():
  ha,hb=h_nat-hp(qa),h_nat-hp(qb)
  assert np.array_equal(ha[:CTX],h_star[:CTX]) and np.array_equal(hb[:CTX],h_star[:CTX])
  E[p]=(ha-hb)[CTX:];delta[p]=np.maximum(g.Design().delta_rel*np.abs(E[p]),g.Design().sigma_eps)
 return fut,h_nat,eps,hp_a,h_star,E,delta

def _theta(L,bg,st,Qpre,q_origin,Qw,**extra):
 A=L['d']['gain']
 return dict(a=L['d']['a'],b=L['d']['b'],A=A,Q_pre=Qpre,q0=q_origin,Q_start=float(Qw[0]),eta_pump_true=A*(Qpre-Qw[0]),
             eta_nat_true=0.,sigma_bg=bg['sigma_bg'],nat_gain=bg['nat_gain'],**bg['nat'],T=st.T,S=st.S,r=st.r,c=st.c,**extra)

UNITS=dict(head='m',pumping='m3/d',rain='mm/d',T='m2/d',r='m',c='d',gain='m per m3/d',time='d')

def build(bg,st,rho,scale=1.,group='A',schedule_record=None):
    N=6;recency='none'
    L=physical(st)
    sc=schedule(bg['mult'],L,rho) if schedule_record is None else schedule_record
    Qw=sc['q']*scale; A=L['d']['gain']; t95=L['t95']; q0=float(Qw[-1]);Qpre=100*scale
    fut,h_nat,eps,hp_a,h_star,E,delta=_physics(bg,L,Qw,q0,Qpre)
    real=bg['real']; rr=int(real['realization']);suffix=f'_scale{scale:g}'
    cid=f'd05{group}_r{rr:02d}_{st.sid}_rho{rho:g}_N{N}{suffix}'
    rest=(sc['e']-sc['R'],sc['e']);cen=d03.census(Qw,t95,rest)
    if sc['R']==0:cen.update(designated_rest_identity=False,designated_rest_absent=True)
    outside=np.ones(CTX,bool);outside[slice(*rest)]=False
    V=float(Qw.sum());mean=float(Qw[outside].mean())
    ctrl=sc['pulse'] if rho==0 else sc['off']
    lastctl=int(np.flatnonzero(ctrl)[-1]);burstend=lastctl+1
    obs=observed_rest_features(Qw,t95)
    derived=dict(case_id=cid,sid=st.sid,layer_id=st.sid,source_phase='phase4',storage_type=st.storage_type,storage_value=st.S,T_m2_d=st.T,r_m=st.r,Q_nominal_m3_d=100*scale,
                 a_star_d=L['d']['a'],c_d=st.c,a_d=L['d']['a'],b=L['d']['b'],lambda_m=L['d']['lam'],gain_m_per_m3d=A,t50_d=L['d']['t50'],t95_d=t95,
                 pi_r=L['d']['pi_r'],layer_rest_end_day=sc['e'],rest_start_day=rest[0],R_days=sc['R'],R_max_days=sc['Rmax'],rho_nominal=rho,rho=rho,
                 rho_realized=sc['R']/t95,N_nominal=N,N_transitions=N,N_controlled_rate_edges=mask_edges(ctrl),kappa=sc['kappa'],V_m3=V,mean_outside_rest_m3d=mean,mean_context_m3d=V/CTX,q0_m3d=q0,
                 q_origin_m3d=q0,q_last_context_day_m3d=q0,calendar_mean_m3d=float(Qw.mean()),base_mult_mean=float(bg['mult'].mean()),base_N_sha256=hashlib.sha256(np.ascontiguousarray(sc['base_N']*scale).tobytes()).hexdigest(),
                 SR=A*mean/bg['sigma_bg'],SR_definition='outside_designated_rest',SR_context_mean=A*V/CTX/bg['sigma_bg'],sigma_bg_m=bg['sigma_bg'],lead_over_t95_10=10/t95,lead_over_t95_30=30/t95,
                 rest_over_t95=sc['R']/t95,last_short_event_day=lastctl,last_switch_boundary=burstend,recency_days=CTX-burstend,recency_ratio=(CTX-burstend)/t95,
                 recency_basis='controlled_rate_pulse' if rho==0 else 'off_switch',schedule_template=sc['template'],
                 pulse_factor=PULSE_FACTOR if rho==0 else None,pulse_indices=[int(i) for i in np.flatnonzero(sc['pulse'])],
                 tail_on_days=cen['tail_len'],census=cen,realization=rr,site=real['site_stem'],experiment_group=group,Q_scale=scale,recency_label=recency,
                 calendar_type=None,no_active_contrast=bool(q0==0),
                 linear_approximation=bool(group=='C' and st.S==.01),units=UNITS,**obs)
    derived.update(designated_rest_present=bool(sc['R']>0), signal_relative_defined=bool(q0!=0))
    meta=dict(origin_date=str(bg['dates'][g.GEN_WARMUP+CTX].date()),rain_site=real['site_stem'],realization=rr,
              rest_ratio=float(rho),signal_ratio=float(derived['SR']),pi_layer=float(derived['pi_r']),head_unit='m',pumping_unit='m3/d',rainfall_unit='mm/d',daily_alignment=True,
              storage_type=st.storage_type,storage_value=float(st.S),T_m2_d=float(st.T),r_m=float(st.r),Q_m3_d=100*scale,transition_count=N,t95_days=t95,
              experiment_group=group,Q_scale=scale,recency_label=recency,recency_days=derived['recency_days'],recency_ratio=derived['recency_ratio'],
              c_d=st.c,linear_approximation=derived['linear_approximation'],q_origin_m3d=q0,schedule_template=sc['template'])
    ti=dict(case_id=cid,head_context=(h_star+eps)[:CTX],rain=bg['P'][g.GEN_WARMUP:],pumping_context=Qw,
            future_Q={f'{p}_{s}':q for p,(qa,qb) in fut.items() for s,q in [('a',qa),('b',qb)]},dates=np.array([str(x.date()) for x in bg['dates'][g.GEN_WARMUP:]]),meta=meta)
    truth=dict(case_id=cid,h_star_a=h_star,h_nat=h_nat,h_pump_a=hp_a,eps=eps,E_true=E,delta=delta,
               theta_true=_theta(L,bg,st,Qpre,q0,Qw,R_days=float(sc['R'])))
    return dict(case_id=cid,meta=meta,tf_input=ti,truth=truth,derived=derived,base_N=sc['base_N']*scale)

def all_A_cases(fixture=False):
 if not fixture:require_frozen()
 reals,rains=d03.realizations(fixture)
 for real in reals:
  bg=g.realization_background(real,rains[real['site_stem']],g.Design())
  for st in d03.strata():
   for rho in NEW_RHOS:
    for scale in SCALES:yield build(bg,st,rho,scale)

# ------------------------------------------------------------------ B operating calendars

def strata_by_sid():
 return {st.sid:st for st in d03.strata()}

def declared_origin_rate(form,realization,origin_date,rain_origin_mm):
 """Deterministic calendar rate on the origin day from truth-free inputs (rule identity, date, observed rain)."""
 rr=int(realization);d=date.fromisoformat(str(origin_date))
 row=dict(realization=rr,offseason_use_variant=rr in cal.OFFSEASON_REALIZATIONS)
 maint=cal.maintenance_dayset(rr,range(d.year,d.year+1)) if form=='domestic_continuous' else set()
 if not np.isfinite(rain_origin_mm):raise ValueError('Nonfinite origin rain is an input failure')
 return float(cal.rate_on(form,d,row,float(rain_origin_mm),maint))

def fixture_climate():
 import sys;sys.path.insert(0,str(ROOT/'lib/tests'))
 from p3_fixture import fixture_rain
 s=fixture_rain();return pd.DataFrame({'date':s.index,'RAIN':s.values}),s

def build_B(bg,st,creal,form,origin,q,q_origin,climate=None):
 L=physical(st);A=L['d']['gain'];t95=L['t95'];Qpre=100.
 if bg['dates'][g.GEN_WARMUP+CTX]!=pd.Timestamp(origin) or bg['dates'][g.GEN_WARMUP]!=pd.Timestamp(origin-timedelta(days=CTX)):
  raise ValueError('B background window is not aligned to the calendar origin')
 Qw=np.asarray(q,float).copy();q_origin=float(q_origin)
 if Qw.shape!=(CTX,) or np.any(Qw<0) or not np.all(np.isfinite(Qw)):raise ValueError('Invalid B context pumping')
 if form=='paddy_irrigation':
  # Calendar rain and head-generator rain are the same daily array: dry paddy-season days pump, wet days stop.
  P=bg['P'][g.GEN_WARMUP:g.GEN_WARMUP+CTX];mo=np.array([d.month for d in bg['dates'][g.GEN_WARMUP:g.GEN_WARMUP+CTX]])
  season=np.isin(mo,cal.PADDY_MONTHS)
  if not (np.all(Qw[season&(P>cal.RAIN_OFF_THRESHOLD_MM)]==0) and np.all(Qw[season&(P<=cal.RAIN_OFF_THRESHOLD_MM)]==cal.Q_NOMINAL_M3D) and np.all(Qw[~season]==0)):
   raise ValueError('Paddy calendar is not aligned with the background rain array')
 if q_origin!=declared_origin_rate(form,creal['realization'],origin.isoformat(),float(bg['P'][g.GEN_WARMUP+CTX])):
  raise ValueError('Origin rate differs from the deterministic calendar rule')
 fut,h_nat,eps,hp_a,h_star,E,delta=_physics(bg,L,Qw,q_origin,Qpre)
 feat=cal.case_features(form,creal,dict(sid=st.sid,storage_type=st.storage_type,t95_d=t95),origin,Qw,q_origin,float(Qw[-1]))
 if abs(t95-[l for l in cal.LAYERS if l['sid']==st.sid][0]['t95_d'])>1e-9:raise ValueError('Calendar t95 differs from frozen physical map')
 cid=feat['case_id'];obs=observed_rest_features(Qw,t95,origin-timedelta(days=1))
 for k in ('last_off_length_days','last_off_distance_to_origin_days','rho_last_off','n_zero_days'):
  if feat[k]!=obs[k]:raise AssertionError('Calendar feature mismatch '+k)
 if feat['n_on_off_switches']!=obs['N_on_off_observed']:raise AssertionError('Switch count mismatch')
 V=float(Qw.sum());sig=bg['sigma_bg'];zero=q_origin==0
 derived=dict(case_id=cid,sid=st.sid,layer_id=st.sid,source_phase='phase4',experiment_group='B',calendar_type=form,form=form,origin_date=origin.isoformat(),origin_month=origin.month,
              anchor_year=int(creal['anchor_year']),realization=int(creal['realization']),site=creal['site_stem'],kma_station=creal.get('kma_station'),
              offseason_use_variant=feat['offseason_use_variant'],storage_type=st.storage_type,storage_value=st.S,T_m2_d=st.T,r_m=st.r,c_d=st.c,
              a_d=L['d']['a'],b=L['d']['b'],lambda_m=L['d']['lam'],gain_m_per_m3d=A,t50_d=L['d']['t50'],t95_d=t95,pi_r=L['d']['pi_r'],
              Q_nominal_m3_d=cal.Q_NOMINAL_M3D,Q_scale=1.,Q_pre_hidden_m3d=Qpre,q_origin_m3d=q_origin,q0_m3d=q_origin,q_last_context_day_m3d=float(Qw[-1]),
              future_uses='q_origin_including_zero',V_m3=V,mean_context_m3d=V/CTX,sigma_bg_m=sig,
              SR=A*q_origin/sig,SR_origin=A*q_origin/sig,SR_definition='origin_rate',SR_context_mean=A*V/CTX/sig,
              rho=None,rho_nominal=None,N_nominal=None,N_transitions=None,rho_last_off_projection=obs['rho_last_off'],
              zero_effect_case=zero,no_active_contrast=zero,relative_width_status='undefined_no_active_contrast' if zero else 'defined_after_scoring',
              lead_over_t95_10=10/t95,lead_over_t95_30=30/t95,schedule_template='operating_calendar',recency_label='none',
              linear_approximation=False,units=UNITS,**obs)
 meta=dict(origin_date=origin.isoformat(),rain_site=creal['site_stem'],realization=int(creal['realization']),rest_ratio=float(obs['rho_last_off']),
           signal_ratio=float(derived['SR']),pi_layer=float(L['d']['pi_r']),head_unit='m',pumping_unit='m3/d',rainfall_unit='mm/d',daily_alignment=True,
           storage_type=st.storage_type,storage_value=float(st.S),T_m2_d=float(st.T),r_m=float(st.r),Q_m3_d=cal.Q_NOMINAL_M3D,
           transition_count=int(obs['N_on_off_observed']),t95_days=t95,experiment_group='B',Q_scale=1.,recency_label='none',c_d=st.c,
           linear_approximation=False,calendar_form=form,offseason_use_variant=bool(feat['offseason_use_variant']),q_origin_m3d=q_origin,
           q_last_context_day_m3d=float(Qw[-1]),schedule_template='operating_calendar')
 ti=dict(case_id=cid,head_context=(h_star+eps)[:CTX],rain=bg['P'][g.GEN_WARMUP:],pumping_context=Qw,
         future_Q={f'{p}_{s}':v for p,(qa,qb) in fut.items() for s,v in [('a',qa),('b',qb)]},dates=np.array([str(x.date()) for x in bg['dates'][g.GEN_WARMUP:]]),meta=meta)
 truth=dict(case_id=cid,h_star_a=h_star,h_nat=h_nat,h_pump_a=hp_a,eps=eps,E_true=E,delta=delta,
            theta_true=_theta(L,bg,st,Qpre,q_origin,Qw,q_last_context=float(Qw[-1]),last_off_length_days=float(obs['last_off_length_days'])))
 return dict(case_id=cid,meta=meta,tf_input=ti,truth=truth,derived=derived,base_N=Qw)

def B_realizations(fixture=False):
 """Pilot rows (seeds/sites) with the calendar anchor year; fixture keeps seeds but uses synthetic rain and 2004."""
 reals=cal.load_pilot_realizations()
 if not fixture:return reals
 return [dict(r,site_stem='FIXTURE_SYNTHETIC',anchor_year=FIXTURE_ANCHOR_YEAR) for r in reals]

def all_B_cases(fixture=False,only=None):
 """Yield the 2160 B cases in calendar order (realization, form, month, layer). only: optional set of case ids."""
 if not fixture:require_frozen()
 reals=B_realizations(fixture);byid=strata_by_sid()
 if fixture:clim,rain_s=fixture_climate();climates={'FIXTURE_SYNTHETIC':(clim,rain_s)}
 else:climates={s:(cal.load_climate(s),g.load_rain(s,p)) for s,p in d03.SITE_SHA.items()}
 for creal in reals:
  clim,rain=climates[creal['site_stem']]
  for form in cal.FORMS:
   series=cal.build_rate_series(form,creal,clim)
   for month in range(1,13):
    origin=date(creal['anchor_year'],month,1);_end,q,q_origin=cal.slice_context(series,origin)
    ids=[f"d05B_{form}_m{month:02d}_r{creal['realization']:02d}_{l['sid']}" for l in cal.LAYERS]
    if only is not None and not set(ids)&set(only):continue
    real=dict(realization=creal['realization'],site_stem=creal['site_stem'],context_start=(origin-timedelta(days=CTX)).isoformat(),
              seed_natural=creal['seed_natural'],seed_noise=creal['seed_noise'],seed_schedule=creal['seed_schedule'],seed_pastas_param_sample=creal['seed_pastas_param_sample'])
    bg=g.realization_background(real,rain,g.Design())
    for layer,cid in zip(cal.LAYERS,ids):
     if only is None or cid in only:yield build_B(bg,byid[layer['sid']],creal,form,origin,q,q_origin)

def all_cases(fixture=False):
 yield from all_A_cases(fixture)
 yield from all_B_cases(fixture)
