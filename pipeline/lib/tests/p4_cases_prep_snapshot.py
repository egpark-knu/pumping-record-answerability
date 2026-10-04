"""D05 new schedules only; frozen background, kernels, truth formulas and serialization retained."""
import hashlib,math
import numpy as np
import p3_generator as g
import p3_phase2_cases as d03
from p4_contracts import ROOT,NEW_RHOS,SCALES,require_frozen
CTX=g.CTX;HMAX=g.HMAX;PAIRS=d03.PAIRS

def physical(st):
 L=dict(d03.layer(st));L['R']={rho:int(round(rho*L['t95'])) for rho in NEW_RHOS}
 # Rmax stays the D03 maximum 4*t95: unchanged compensation window and comparable lineage.
 return L

def schedule(mult,L,rho):
 if rho not in NEW_RHOS:raise ValueError('Unprespecified D05 rho')
 R=int(round(rho*L['t95']));e=L['e'];Rmax=L['Rmax'];base0=100*np.asarray(mult,float)
 off=np.zeros(CTX,bool);off[e+1:e+6:2]=True
 C=np.zeros(CTX,bool);C[:e-Rmax]=True
 if not (0<e-Rmax and e+18<CTX):raise ValueError('Inherited schedule does not fit')
 kappa=1+base0[off].sum()/base0[C].sum();base=base0.copy();base[C]*=kappa;base[off]=0
 q=base.copy();q[e-R:e]=0
 if rho==0:
  assert R==0 and np.count_nonzero(q==0)==3
  # Three isolated one-day switch OFF intervals; no designated sustained rest.
 return dict(q=q,base_N=base,base0=base0,kappa=float(kappa),off=off,C=C,e=e,R=R,Rmax=Rmax)

def build(bg,st,rho,scale=1.,group='A',schedule_record=None):
    N=6;recency='none'
    L=physical(st)
    sc=schedule(bg['mult'],L,rho) if schedule_record is None else schedule_record
    Qw=sc['q']*scale; A=L['d']['gain']; t95=L['t95']; q0=float(Qw[-1]);Qpre=100*scale;B,S=L['B'],L['S']
    fut={PAIRS[0]:(np.full(HMAX,q0),np.zeros(HMAX)),PAIRS[1]:(np.full(HMAX,q0),np.full(HMAX,1.5*q0))}
    def hp(f):return A*(Qpre*(1-S[1:CTX+HMAX+1])+np.convolve(np.concatenate([Qw,f]),B)[:CTX+HMAX])
    w=slice(g.GEN_WARMUP,g.GEN_WARMUP+CTX+HMAX);h_nat,eps=bg['h_nat'][w],bg['eps'][w];hp_a=hp(fut[PAIRS[0]][0]);h_star=h_nat-hp_a
    E,delta={},{}
    for p,(qa,qb) in fut.items():
        ha,hb=h_nat-hp(qa),h_nat-hp(qb)
        assert np.array_equal(ha[:CTX],h_star[:CTX]) and np.array_equal(hb[:CTX],h_star[:CTX])
        E[p]=(ha-hb)[CTX:];delta[p]=np.maximum(g.Design().delta_rel*np.abs(E[p]),g.Design().sigma_eps)
    real=bg['real']; rr=int(real['realization']);suffix=f'_scale{scale:g}'
    cid=f'd05{group}_r{rr:02d}_{st.sid}_rho{rho:g}_N{N}{suffix}'
    rest=(sc['e']-sc['R'],sc['e']);cen=d03.census(Qw,t95,rest)
    if sc['R']==0:cen.update(designated_rest_identity=False,designated_rest_absent=True)
    outside=np.ones(CTX,bool);outside[slice(*rest)]=False
    V=float(Qw.sum());mean=float(Qw[outside].mean());lastoff=int(np.flatnonzero(sc['off'])[-1]) if np.any(sc['off']) else -1;burstend=lastoff+1
    derived=dict(case_id=cid,sid=st.sid,storage_type=st.storage_type,storage_value=st.S,T_m2_d=st.T,r_m=st.r,Q_nominal_m3_d=100*scale,
                 a_star_d=L['d']['a'],c_d=st.c,a_d=L['d']['a'],b=L['d']['b'],lambda_m=L['d']['lam'],gain_m_per_m3d=A,t50_d=L['d']['t50'],t95_d=t95,
                 pi_r=L['d']['pi_r'],layer_rest_end_day=sc['e'],rest_start_day=rest[0],R_days=sc['R'],R_max_days=sc['Rmax'],rho_nominal=rho,
                 rho_realized=sc['R']/t95,N_nominal=N,kappa=sc['kappa'],V_m3=V,mean_outside_rest_m3d=mean,mean_context_m3d=V/CTX,q0_m3d=q0,
                 calendar_mean_m3d=float(Qw.mean()),base_mult_mean=float(bg['mult'].mean()),base_N_sha256=hashlib.sha256(np.ascontiguousarray(sc['base_N']*scale).tobytes()).hexdigest(),
                 SR=A*mean/bg['sigma_bg'],SR_context_mean=A*V/CTX/bg['sigma_bg'],sigma_bg_m=bg['sigma_bg'],lead_over_t95_10=10/t95,lead_over_t95_30=30/t95,
                 rest_over_t95=sc['R']/t95,last_short_event_day=lastoff,last_switch_boundary=burstend,recency_days=CTX-burstend,recency_ratio=(CTX-burstend)/t95,
                 tail_on_days=cen['tail_len'],census=cen,realization=rr,site=real['site_stem'],experiment_group=group,Q_scale=scale,recency_label=recency,
                 linear_approximation=bool(group=='C' and st.S==.01),units=dict(head='m',pumping='m3/d',rain='mm/d',T='m2/d',r='m',c='d',gain='m per m3/d',time='d'))
    derived.update(designated_rest_present=bool(sc['R']>0), signal_relative_defined=bool(q0!=0))
    meta=dict(origin_date=str(bg['dates'][g.GEN_WARMUP+CTX].date()),rain_site=real['site_stem'],realization=rr,
              rest_ratio=float(rho),signal_ratio=float(derived['SR']),pi_layer=float(derived['pi_r']),head_unit='m',pumping_unit='m3/d',rainfall_unit='mm/d',daily_alignment=True,
              storage_type=st.storage_type,storage_value=float(st.S),T_m2_d=float(st.T),r_m=float(st.r),Q_m3_d=100*scale,transition_count=N,t95_days=t95,
              experiment_group=group,Q_scale=scale,recency_label=recency,recency_days=derived['recency_days'],recency_ratio=derived['recency_ratio'],
              c_d=st.c,linear_approximation=derived['linear_approximation'])
    ti=dict(case_id=cid,head_context=(h_star+eps)[:CTX],rain=bg['P'][g.GEN_WARMUP:],pumping_context=Qw,
            future_Q={f'{p}_{s}':q for p,(qa,qb) in fut.items() for s,q in [('a',qa),('b',qb)]},dates=np.array([str(x.date()) for x in bg['dates'][g.GEN_WARMUP:]]),meta=meta)
    truth=dict(case_id=cid,h_star_a=h_star,h_nat=h_nat,h_pump_a=hp_a,eps=eps,E_true=E,delta=delta,
               theta_true=dict(a=L['d']['a'],b=L['d']['b'],A=A,Q_pre=Qpre,q0=q0,R_days=float(sc['R']),Q_start=float(Qw[0]),eta_pump_true=A*(Qpre-Qw[0]),
                               eta_nat_true=0.,sigma_bg=bg['sigma_bg'],nat_gain=bg['nat_gain'],**bg['nat'],T=st.T,S=st.S,r=st.r,c=st.c))
    return dict(case_id=cid,meta=meta,tf_input=ti,truth=truth,derived=derived,base_N=sc['base_N']*scale)

def all_A_cases(fixture=False):
 if not fixture:require_frozen()
 reals,rains=d03.realizations(fixture)
 for real in reals:
  bg=g.realization_background(real,rains[real['site_stem']],g.Design())
  for st in d03.strata():
   for rho in NEW_RHOS:
    for scale in SCALES:yield build(bg,st,rho,scale)
