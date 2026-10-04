"""Compatibility API routed to canonical p4_statistics; no fitting on import."""
import numpy as np
import p4_statistics as canonical
OFFSET=canonical.RHO_OFFSET
BOOTSTRAP_DRAWS=canonical.N_BOOTSTRAP
BOOTSTRAP_SEED=canonical.SEED


def design(sr,rho,layers=None,levels=None):
    sr,rho=np.asarray(sr,float),np.asarray(rho,float)
    if np.any(~np.isfinite(sr)) or np.any(sr<=0) or np.any(~np.isfinite(rho)) or np.any(rho<0): raise ValueError("Undefined log coordinates; retain/count separately")
    cols=[np.ones(len(sr)),np.log(sr),np.log(rho+OFFSET)]
    if layers is not None:
        levels=list(canonical.LAYER_ORDER) if levels is None else list(levels)
        cols.extend((np.asarray(layers)==label).astype(float) for label in levels[1:])
    return np.column_stack(cols)


def _legacy_fit(fit):
    result=dict(fit)
    result["status"]="identified" if fit["status"]=="ok" else "unidentifiable"
    result["reason"]=fit["status"]
    result["coefficients"]=None if fit.get("coefficients") is None else list(fit["coefficients"].values())
    return result


def fit_logistic(X,y):
    return _legacy_fit(canonical.fit_logistic(y,X))


def threshold(coefficients,rho,target,sr_support,layer_offset=0.,link="logistic"):
    b=np.asarray(coefficients,float)
    names=["intercept","log_signal_ratio","log_rho_plus_offset"]
    fit=canonical.ok_fit(dict(zip(names,b[:3])),canonical.Support(np.log(sr_support[0]),np.log(sr_support[1]),rho,rho,[rho]))
    fit["coefficients"]["intercept"]+=layer_offset
    if link=="logistic": out=canonical.crossing_from_fit(fit,target,rho,None)
    else:
        if target!=1.: raise ValueError("canonical magnitude contour target is1")
        out=canonical.magnitude_crossing_from_fit(fit,rho,None)
    return dict(out,status="finite" if out["reporting_status"]=="finite_inside" else out["reporting_status"],value=out["signal_ratio"],observed_support=list(sr_support))


def fit_width(X,width,truth):
    X,width,truth=np.asarray(X,float),np.asarray(width,float),np.asarray(truth,float)
    ok=np.isfinite(width)&np.isfinite(truth)&(width>0)&(truth!=0)
    fit=canonical.fit_log_linear(np.log(width[ok])-np.log(np.abs(truth[ok])),X[ok])
    return dict(_legacy_fit(fit),n_defined=int(ok.sum()),n_undefined=int((~ok).sum()),R2=fit.get("r2"))


def cluster_indices(realizations,draws=BOOTSTRAP_DRAWS,seed=BOOTSTRAP_SEED):
    r=np.asarray(realizations)
    if not np.isin(r,np.arange(10)).all(): raise ValueError("realizations must be0..9")
    _,multiplicities=canonical._draws([],draws,seed)
    for d in multiplicities:
        parts=[np.tile(np.flatnonzero(r==c),d[c]) for c in range(10)]
        yield np.concatenate(parts)


def bootstrap_contour(X,y,r,fit_function,contour_function,draws=BOOTSTRAP_DRAWS):
    values=[];statuses={};failures=0
    for ix in cluster_indices(r,draws):
        fit=fit_function(np.asarray(X)[ix],np.asarray(y)[ix])
        if fit["status"]!="identified":failures+=1;continue
        point=contour_function(fit["coefficients"]);statuses[point["status"]]=statuses.get(point["status"],0)+1
        if point.get("value") is not None: values.append(point["value"])
    interval=canonical.percentile_interval(values,draws-len(values))
    return dict(interval95=None if not values else [interval["low"],interval["high"]],draws=draws,n_defined=len(values),fit_failures=failures,threshold_status_counts=statuses,interval_status=interval["status"])


def record_metrics(inf,sup,true,tool,reference):
 vals=np.asarray([inf,sup,true,tool,reference],float)
 valid=np.isfinite(vals[:2]).all() and inf<=sup
 direction=1 if valid and inf>0 else -1 if valid and sup<0 else 0
 width=sup-inf if valid else np.nan
 opposite=lambda e: bool(direction!=0 and np.isfinite(e) and direction*e<0)
 return dict(sign_determined=direction!=0,record_direction=direction,envelope_valid=bool(valid),truth_exact_zero=bool(np.isfinite(true) and true==0),signal_relative_defined=bool(np.isfinite(true) and true!=0),W_over_absE=None if not(valid and np.isfinite(true) and true!=0) else width/abs(true),tool_opposite=opposite(tool),reference_opposite=opposite(reference),tool_zero=bool(np.isfinite(tool) and tool==0),tool_abs_error=None if not np.isfinite([true,tool]).all() else abs(tool-true),undetermined_tool_abs_error=None if direction or not np.isfinite([true,tool]).all() else abs(tool-true))

def log_signal_bins(sr):
    bins=canonical.log_equal_bins(sr)
    indices=np.array([-1 if canonical._bin_index(canonical._float(x),bins["edges"]) in (None,"nonpositive_signal") else canonical._bin_index(float(x),bins["edges"]) for x in sr],int)
    return indices,bins["edges"] if bins["edges"][0] is not None else None
