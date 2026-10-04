"""Convex affine SSE support with explicit bounds and numerical dual receipts.

No ridge is used. Coordinates are column-normalized; rank uses 1e-10 SVD.
Certificate means a numerical primal/dual gap at the reported tolerance,
not an interval-arithmetic proof. Failures remain support_uncertified.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import minimize, lsq_linear, nnls, linprog

RANK_RTOL=1e-10
CERT_TOL=2e-7


def problem_digest(J,e,lower=None,upper=None,linear=None):
 n=np.asarray(J).shape[1];h=hashlib.sha256()
 lo=np.full(n,-np.inf) if lower is None else lower;hi=np.full(n,np.inf) if upper is None else upper
 for value in (J,e,lo,hi):
  a=np.asarray(value,dtype='<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
 if linear is not None:
  for value in linear:
   a=np.asarray(value,dtype='<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def decomposition(J):
 J=np.asarray(J,float)
 norms=np.linalg.norm(J,axis=0)
 scale=1/np.where(norms>1e-14,norms,1.)
 U,s,Vt=np.linalg.svd(J*scale,full_matrices=False)
 rank=int(np.sum(s>RANK_RTOL*(s[0] if s.size else 1.)))
 # For observations fewer than columns retain the missing null vectors.
 if J.shape[0]<J.shape[1]:
  _,_,Vt=np.linalg.svd(J*scale,full_matrices=True)
 return dict(scale=scale,U=U[:,:rank],s=s,rank=rank,V=Vt[:rank].T,N=Vt[rank:].T)

def constraints(lower,upper,linear,n):
 rows=[];rhs=[];labels=[]
 for j in range(n):
  if np.isfinite(upper[j]):
   a=np.zeros(n);a[j]=1;rows.append(a);rhs.append(upper[j]);labels.append('upper_'+str(j))
  if np.isfinite(lower[j]):
   a=np.zeros(n);a[j]=-1;rows.append(a);rhs.append(-lower[j]);labels.append('lower_'+str(j))
 if linear is not None:
  A,b=linear
  for i,(a,v) in enumerate(zip(np.atleast_2d(A),np.atleast_1d(b))):
   rows.append(a);rhs.append(v);labels.append('memory_'+str(i))
 return np.asarray(rows,float).reshape(-1,n),np.asarray(rhs,float),labels

def least_squares(J,e,lower=None,upper=None,linear=None):
 J=np.asarray(J,float);e=np.asarray(e,float);n=J.shape[1]
 lower=np.full(n,-np.inf) if lower is None else np.asarray(lower,float)
 upper=np.full(n,np.inf) if upper is None else np.asarray(upper,float)
 dec=decomposition(J);sc=dec['scale'];A=J*sc
 M,b,labels=constraints(lower/sc,upper/sc,None if linear is None else (np.asarray(linear[0])*sc,linear[1]),n)
 if np.any(lower>=upper):raise ValueError('Strictly ordered bounds required')
 fit=lsq_linear(A,e,bounds=(lower/sc,upper/sc),tol=1e-12,lsq_solver='exact',max_iter=500)
 x=fit.x
 if M.size and np.max(M@x-b)>1e-9:
  r=minimize(lambda z:float(np.sum((e-A@z)**2)),x,jac=lambda z:2*A.T@(A@z-e),method='SLSQP',bounds=list(zip(lower/sc,upper/sc)),constraints=[dict(type='ineq',fun=lambda z:b-M@z,jac=lambda z:-M)],options=dict(maxiter=1000,ftol=1e-13))
  x=r.x
 residual=e-A@x;g=-2*A.T@residual
 active=np.where(b-M@x<=1e-7*(1+np.abs(b)))[0]
 mu=nnls(M[active].T,-g,maxiter=1000)[0] if active.size else np.zeros(0)
 station=g+(M[active].T@mu if active.size else 0)
 # Dual of bounded least squares, including memory faces.
 bvec=-2*A.T@e+(M[active].T@mu if active.size else 0)
 dual=_quadratic_dual(A,e,0.,np.zeros(n),1.,M[active],b[active],mu)
 sse=float(residual@residual)
 feasible=not M.size or np.max(M@x-b)<CERT_TOL
 certified=feasible and dual is not None and sse-dual<CERT_TOL*(1+sse)
 gap=None if dual is None else max(0.,sse-dual)
 violation=max(0.,float(np.max(M@x-b))) if M.size else 0.
 complement=float(np.max(np.abs(mu*(M[active]@x-b[active])))) if len(mu) else 0.
 certified=certified and np.linalg.norm(station)<=2e-6 and complement<=2e-6
 return dict(problem_digest=problem_digest(J,e,lower,upper,linear),primal_upper=sse if feasible else None,dual_gap=gap,primal_violation=violation,complementarity=complement,multipliers=mu,x=x*sc,sse=sse,residual=residual,rank=dec['rank'],singular_values=dec['s'],scale=sc,certified=bool(certified),dual_lower=dual,kkt_residual=float(np.linalg.norm(station)),active=[labels[i] for i in active])

def _quadratic_dual(A,e,T,c,lam,M,b,mu):
 """Lower bound on min c.x with SSE<=T, Mx<=b; unconstrained Lagrangian infimum."""
 g=c-2*lam*A.T@e+(M.T@mu if len(mu) else 0)
 if lam<=1e-14:
  if np.linalg.norm(g)>2e-8:return None
  return float(-mu@b)
 _,s,Vt=np.linalg.svd(A,full_matrices=False)
 r=int(np.sum(s>RANK_RTOL*(s[0] if len(s) else 1)))
 proj=Vt[:r].T@(Vt[:r]@g)
 if np.linalg.norm(g-proj)>2e-8*(1+np.linalg.norm(g)):return None
 invg=Vt[:r].T@((Vt[:r]@g)/(lam*s[:r]**2))
 return float(lam*(e@e-T)-mu@b-.25*g@invg)

def _active_face_polish(A,e,T,c,M,b,active):
 """Exact ellipsoid endpoint restricted to the SLSQP active face; no ridge."""
 n=A.shape[1]
 # Full-rank only: leave every existing null/recession path untouched.
 if decomposition(A)['rank']!=n:return None
 F=M[active]
 if len(active):
  _,sf,Vf=np.linalg.svd(F,full_matrices=True)
  rf=int(np.sum(sf>RANK_RTOL*sf[0]))
  x0=np.linalg.lstsq(F,b[active],rcond=RANK_RTOL)[0];Z=Vf[rf:].T
 else:x0=np.zeros(n);Z=np.eye(n)
 if not Z.shape[1]:return None
 B=A@Z;U,s,Vt=np.linalg.svd(B,full_matrices=False)
 if len(s)!=Z.shape[1] or s[-1]<=RANK_RTOL*s[0]:return None
 center=x0+Z@(Vt.T@((U.T@(e-A@x0))/s))
 radius2=T-float(np.sum((e-A@center)**2));direction=Z@(Vt.T@((Vt@(Z.T@c))/s**2));v=float(c@direction)
 if radius2<0 or v<=0:return None
 return center-np.sqrt(radius2/v)*direction


def tangent_support(J,e,T,q,lower=None,upper=None,linear=None,E0=0.,warm=None,prepared=None):
 """Support interval over ||e-Jd||²<=T and all frozen linear constraints.

 All bounds are on delta, not absolute parameters. Returns endpoints and
 certificates, never infers optimality from a solver Boolean. `prepared` may
 hold a previously calculated least-squares receipt for identical constraints.
 """
 J=np.asarray(J,float);e=np.asarray(e,float);q=np.asarray(q,float);n=J.shape[1]
 if J.shape[0]!=e.size or q.shape!=(n,) or not np.isfinite(T):raise ValueError('Invalid support input')
 lo=np.full(n,-np.inf) if lower is None else np.asarray(lower,float)
 hi=np.full(n,np.inf) if upper is None else np.asarray(upper,float)
 dec=decomposition(J);sc=dec['scale'];A=J*sc;qs=q*sc
 M,b,labels=constraints(lo/sc,hi/sc,None if linear is None else (np.asarray(linear[0])*sc,linear[1]),n)
 if prepared is not None and prepared.get('problem_digest')!=problem_digest(J,e,lo,hi,linear):raise ValueError('Prepared minimum belongs to a different J/residual/bounds/memory problem')
 fit=least_squares(J,e,lo,hi,linear) if prepared is None else prepared
 if fit['sse']>T+CERT_TOL*(1+abs(T)):
  return dict(status='linearization_incompatible' if fit.get('dual_lower') is not None and fit['dual_lower']>T else 'feasibility_uncertified',inf=None,sup=None,width=None,feasibility=fit,rank=dec['rank'])
 if fit['sse']>T:
  return dict(status='support_uncertified',inf=None,sup=None,width=None,reason='near_empty_feasibility_boundary',feasibility=fit,rank=dec['rank'])
 if not np.any(q):
  return dict(status='zero_contrast',inf=float(E0),sup=float(E0),width=0.,rank=dec['rank'],feasibility=fit)
 N=dec['N'];nullq=N.T@qs
 null_present=np.linalg.norm(nullq)>1e-9*max(np.linalg.norm(qs),1e-20)
 # Recession LP finds an actual null direction compatible with every finite face.
 if null_present:
  for sign in (-1.,1.):
   lp=linprog(sign*nullq,A_ub=M@N if M.size else None,b_ub=np.zeros(len(b)) if M.size else None,bounds=[(-1.,1.)]*N.shape[1],method='highs')
   if lp.success and lp.fun < -1e-9*np.linalg.norm(qs):
    witness=sc*(N@lp.x)
    return dict(status='unbounded',inf=None,sup=None,width=None,null_witness=witness,unbounded_direction='upper' if sign<0 else 'lower',rank=dec['rank'],feasibility=fit)
 # Exact closed form if the contrast annihilates nullspace and both profile
 # endpoints satisfy every box/memory inequality. No nullspace regularization.
 V=dec['V'];s=dec['s'][:dec['rank']]
 xhat=V@((dec['U'].T@e)/s) if len(s) else np.zeros(n)
 r2=T-float(np.sum((e-A@xhat)**2))
 vec=V@((V.T@qs)/s**2) if len(s) else np.zeros(n)
 v=float(qs@vec)
 if not null_present and r2>=0 and v>0:
  dx=np.sqrt(r2/v)*vec;ends=(xhat-dx,xhat+dx)
  if all(not M.size or np.max(M@x-b)<=1e-10 for x in ends):
   c=float(E0+qs@xhat);r=np.sqrt(r2*v)
   return dict(status='certified_interior',inf=c-r,sup=c+r,width=2*r,rank=dec['rank'],lower_receipt=dict(x=ends[0]*sc,sse=float(np.sum((e-A@ends[0])**2)),dual_gap=0.,kkt_residual=0.,method='audited_closed_form'),upper_receipt=dict(x=ends[1]*sc,sse=float(np.sum((e-A@ends[1])**2)),dual_gap=0.,kkt_residual=0.,method='audited_closed_form'),feasibility=fit)
 xstart=fit['x']/sc
 normq=max(np.linalg.norm(qs),1e-30)
 receipts=[]
 for sign in (1.,-1.):
  c=sign*qs/normq
  # Normalized objective avoids disparate E units across leads/layers.
  def fun(x):return float(c@x)
  def cf(x):return float(T-np.sum((e-A@x)**2))
  def cg(x):return 2*A.T@(e-A@x)
  cons=[dict(type='ineq',fun=cf,jac=cg)]
  if M.size:cons.append(dict(type='ineq',fun=lambda x:b-M@x,jac=lambda x:-M))
  start=xstart if warm is None else np.asarray(warm[len(receipts)])/sc
  if cf(start)<-1e-10 or (M.size and np.max(M@start-b)>1e-10):start=xstart
  opt=minimize(fun,start,jac=lambda x:c,method='SLSQP',constraints=cons,options=dict(maxiter=800,ftol=1e-12))
  x=opt.x;polished=False
  for certificate_attempt in range(2):
   sse=float(np.sum((e-A@x)**2));slack=b-M@x
   active=np.where(slack<=2e-6*(1+np.abs(b)))[0]
   columns=[]
   include=sse>=T-2e-6*(1+abs(T))
   if include:columns.append(2*A.T@(A@x-e))
   columns.extend(M[active])
   mult=nnls(np.asarray(columns).T,-c,maxiter=2000)[0] if columns else np.zeros(0)
   lam=float(mult[0]) if include else 0.;mu=mult[int(include):]
   station=c+(2*lam*A.T@(A@x-e))+(M[active].T@mu if len(active) else 0)
   dual=_quadratic_dual(A,e,T,c,lam,M[active],b[active],mu)
   primal=fun(x);gap=None if dual is None else max(0.,primal-dual)*normq
   violation=max(0.,sse-T,float(np.max(-slack)) if len(slack) else 0.)
   complement=max(abs(lam*(sse-T)),float(np.max(np.abs(mu*slack[active]))) if len(mu) else 0.)
   certified=dual is not None and violation<=CERT_TOL*(1+abs(T)) and gap<=CERT_TOL*(1+abs(qs@x)) and np.linalg.norm(station)<=2e-6 and complement<=2e-6
   if certified or certificate_attempt or not include:break
   candidate=_active_face_polish(A,e,T,c,M,b,active)
   if candidate is None:break
   x=candidate;polished=True
  receipts.append(dict(x=x*sc,value=float(E0+qs@x),sse=sse,primal_violation=violation,dual_gap=gap,kkt_residual=float(np.linalg.norm(station)),lambda_sse=lam,multipliers=mu,active=[labels[i] for i in active],complementarity=complement,certified=bool(certified),solver_success=bool(opt.success),solver_message=str(opt.message),iterations=int(opt.nit),method='SLSQP_active_face_polish_dual' if polished else 'SLSQP_exact_gradient_dual'))
 status='certified_constrained' if all(r['certified'] for r in receipts) else 'support_uncertified'
 if status=='certified_constrained' and null_present:status='bounded_by_family_only'
 return dict(status=status,inf=receipts[0]['value'],sup=receipts[1]['value'],width=receipts[1]['value']-receipts[0]['value'],rank=dec['rank'],null_contrast_norm=float(np.linalg.norm(nullq)),lower_receipt=receipts[0],upper_receipt=receipts[1],feasibility=fit)
