"""D04 new physical/schedule grid; frozen D03 background, kernel and algorithms imported unchanged."""
from __future__ import annotations
import hashlib,json,math
from functools import lru_cache
from pathlib import Path
import numpy as np
import p3_generator as g
import p3_phase2_cases as d03
from p3_phase2_physmap import Stratum,describe
from p3_kernels import hantush_block
ROOT=d03.ROOT; P3=ROOT/'results/phase3'; CTX=g.CTX; HMAX=g.HMAX; PAIRS=d03.PAIRS

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_frozen():
    f=json.loads((P3/'protocol_freeze.json').read_text())
    assert sha(P3/'protocol.md')==f['protocol_sha256']==(P3/'protocol.sha256').read_text().split()[0]
    for p,h in f['hashes'].items():
        if sha(ROOT/p)!=h: raise PermissionError('Frozen file changed: '+p)
    return f['protocol_sha256']

@lru_cache(None)
def physical(st):
    d=describe(st); B,S=hantush_block(d['a'],d['b'],CTX+HMAX)
    e=CTX-(math.floor(d['t95'])+1)-18; R={rho:int(round(rho*d['t95'])) for rho in d03.RHOS}
    if e-R[4.]<=0: e=CTX-19  # preserve requested long rest, one-day burst and positive origin; disclose shortened tail
    if e-R[4.]<=0: raise ValueError('Requested rest cannot fit 1024-day context')
    return dict(st=st,d=d,t95=d['t95'],e=e,R=R,Rmax=R[4.],B=B[0],S=S[0])

def schedule_A(mult,L,N,recency):
    t95=L['t95']; R=int(round(t95)); ends={r:CTX-int(round(f*t95)) for r,f in [('recent',.25),('old',2.)]}
    e=min(b-18 for b in ends.values())-math.ceil(t95); b=ends[recency]; start=b-N
    base0=100*np.asarray(mult,float); off=np.zeros(CTX,bool); off[start+1:b:2]=True
    C=np.zeros(CTX,bool); C[:e-R]=True
    assert 0<e-R<e<=start-math.ceil(t95) and b<CTX and N in (2,18)
    kappa=1+base0[off].sum()/base0[C].sum(); base=base0.copy();base[C]*=kappa;base[off]=0
    q=base.copy();q[e-R:e]=0
    return dict(q=q,base_N=base,base0=base0,kappa=float(kappa),off=off,C=C,e=e,R=R,Rmax=R,
                burst_start_day=start,burst_end_boundary=b,recency_label=recency,recency_target_ratio=.25 if recency=='recent' else 2.,
                recency_days=CTX-b,recency_ratio=(CTX-b)/t95)


def build(bg,st,rho,N,group,scale=1.,recency='none'):
    L=physical(st)
    sc=schedule_A(bg['mult'],L,N,recency) if group=='A' else d03.schedule(bg['mult'],L,rho,N)
    Qw=sc['q']*scale; A=L['d']['gain']; t95=L['t95']; q0=float(Qw[-1]);Qpre=100*scale;B,S=L['B'],L['S']
    fut={PAIRS[0]:(np.full(HMAX,q0),np.zeros(HMAX)),PAIRS[1]:(np.full(HMAX,q0),np.full(HMAX,1.5*q0))}
    def hp(f):return A*(Qpre*(1-S[1:CTX+HMAX+1])+np.convolve(np.concatenate([Qw,f]),B)[:CTX+HMAX])
    w=slice(g.GEN_WARMUP,g.GEN_WARMUP+CTX+HMAX);h_nat,eps=bg['h_nat'][w],bg['eps'][w];hp_a=hp(fut[PAIRS[0]][0]);h_star=h_nat-hp_a
    E,delta={},{}
    for p,(qa,qb) in fut.items():
        ha,hb=h_nat-hp(qa),h_nat-hp(qb)
        assert np.array_equal(ha[:CTX],h_star[:CTX]) and np.array_equal(hb[:CTX],h_star[:CTX])
        E[p]=(ha-hb)[CTX:];delta[p]=np.maximum(g.Design().delta_rel*np.abs(E[p]),g.Design().sigma_eps)
    real=bg['real']; rr=int(real['realization']);suffix=f'_rec{recency}' if group=='A' else f'_scale{scale:g}' if group=='B' else f'_S{st.S:g}'
    cid=f'd04{group}_r{rr:02d}_{st.sid}_rho{rho:g}_N{N}{suffix}'
    rest=(sc['e']-sc['R'],sc['e']);cen=d03.census(Qw,t95,rest);outside=np.ones(CTX,bool);outside[slice(*rest)]=False
    V=float(Qw.sum());mean=float(Qw[outside].mean());lastoff=int(np.flatnonzero(sc['off'])[-1]);burstend=lastoff+1
    derived=dict(case_id=cid,sid=st.sid,storage_type=st.storage_type,storage_value=st.S,T_m2_d=st.T,r_m=st.r,Q_nominal_m3_d=100*scale,
                 a_star_d=L['d']['a'],c_d=st.c,a_d=L['d']['a'],b=L['d']['b'],lambda_m=L['d']['lam'],gain_m_per_m3d=A,t50_d=L['d']['t50'],t95_d=t95,
                 pi_r=L['d']['pi_r'],layer_rest_end_day=sc['e'],rest_start_day=rest[0],R_days=sc['R'],R_max_days=sc['Rmax'],rho_nominal=rho,
                 rho_realized=sc['R']/t95,N_nominal=N,kappa=sc['kappa'],V_m3=V,mean_outside_rest_m3d=mean,mean_context_m3d=V/CTX,q0_m3d=q0,
                 calendar_mean_m3d=float(Qw.mean()),base_mult_mean=float(bg['mult'].mean()),base_N_sha256=hashlib.sha256(np.ascontiguousarray(sc['base_N']*scale).tobytes()).hexdigest(),
                 SR=A*mean/bg['sigma_bg'],SR_context_mean=A*V/CTX/bg['sigma_bg'],sigma_bg_m=bg['sigma_bg'],lead_over_t95_10=10/t95,lead_over_t95_30=30/t95,
                 rest_over_t95=sc['R']/t95,last_short_event_day=lastoff,last_switch_boundary=burstend,recency_days=CTX-burstend,recency_ratio=(CTX-burstend)/t95,
                 tail_on_days=cen['tail_len'],census=cen,realization=rr,site=real['site_stem'],experiment_group=group,Q_scale=scale,recency_label=recency,
                 linear_approximation=bool(group=='C' and st.S==.01),units=dict(head='m',pumping='m3/d',rain='mm/d',T='m2/d',r='m',c='d',gain='m per m3/d',time='d'))
    if group=='A':
        derived.update({k:sc[k] for k in ('burst_start_day','burst_end_boundary','recency_target_ratio')})
        derived.update(burst_switch_count=N,short_off_intervals=N//2,short_on_maximal_intervals=N//2-1,prescribed_on_intervals=N//2)
    if group=='B' or (group=='C' and st.S==.001):
        parent=f'r{rr:02d}_{st.sid if group=="B" else "leaky_T50"}_rho{rho:g}_N{N}'
        derived['d03_parent_case_id']=parent
        derived['d03_parent_tf_sha256']=sha(d03.P2/'cases/tf_inputs'/f'{parent}.npz')
        derived['d03_parent_truth_sha256']=sha(d03.P2/'cases/truth'/f'{parent}.npz')
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


def all_cases(fixture=False):
    reals,rains=d03.realizations(fixture)
    for real in reals:
        bg=g.realization_background(real,rains[real['site_stem']],g.Design())
        for st in d03.strata():
            for N in (2,18):
                for rec in ('recent','old'):yield build(bg,st,1.,N,'A',recency=rec)
            for rho in d03.RHOS:
                for scale in (.3,3.):yield build(bg,st,rho,6,'B',scale=scale)
        for S,label in [(1e-4,'fixedc_confined'),(1e-3,'fixedc_leaky'),(1e-2,'fixedc_linear')]:
            st=Stratum(label,S,50.,20000.,200.,100.)
            for rho in d03.RHOS:yield build(bg,st,rho,6,'C')
