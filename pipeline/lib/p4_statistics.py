"""D05 prescore statistics.

The formulas, cohort rules, seed, bootstrap budget and the decision to use no
penalty are fixed here, before D05 outcomes exist. The functions accept a
table of already computed rows. They do not score envelopes, modify phase2 or
phase3 files, or choose a completion cutoff.

Sign is recomputed from the closed envelope; inconsistent supplied labels are
rejected. Primary record contradiction is a strict opposite nonzero sign. A tool
or reference exact zero is reported as separate disagreement. W/|E_true| < 1 is the magnitude
event. Equality is not inside that event. Rows that cannot enter a logarithm
or a ratio stay in the inventory and in the calendar denominator.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.special import expit
from collections import Counter

import numpy as np

SEED = 20261001
N_BOOTSTRAP = 999
RHO_OFFSET = 0.05
PROBABILITY_THRESHOLDS = (0.5, 0.9)
PRIMARY_PENALTY = None
REFERENCE_LAYER = "confined_T50"
LAYER_ORDER = (
    "confined_T50",
    "confined_T500",
    "leaky_T50",
    "leaky_T500",
    "unconfined_T50",
    "unconfined_T500",
)
EVAL_RHO = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
MAP_SCALES = (0.3, 1.0, 3.0)
N_SIGNAL_BINS = 5
SEPARATION_ABS_COEF = 20.0
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-8
PROBABILITY_CLIP = 1e-12
CI_PERCENTILES = (2.5, 97.5)
LEADS = (10, 30)
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
CALENDAR_TYPES = ("water_curtain", "paddy_irrigation", "domestic_continuous")
EXCLUDED_RECENCY = ("recent", "old")
ZERO_SLOPE = 1e-12

PHASE2_CELL_COLUMNS = (
    "case_id", "storage_type", "T_m2_d", "rho", "N_transitions", "realization",
    "pair_id", "track", "quantity", "lead_day", "W_m", "W_status", "SR", "pi_r",
    "rest_over_response", "E_true_m", "E_tool_m", "effect_ratio", "ratio_defined",
    "sign_agreement", "error_over_W", "envelope_includes", "data_status",
)
PHASE3_CELL_EXTRAS = (
    "dataset", "experiment_group", "Q_scale", "recency_label", "sid", "storage_value",
    "t95_d", "env_inf_m", "env_sup_m", "delta_m", "sign_determined", "envelope_sign",
    "includes_zero", "record_contradiction", "ambiguous_truth_sign_error", "unconfined",
    "linear_approximation",
)
OPTIONAL_LATER_COLUMNS = ( "calendar_type", "origin_month", "offseason_use", "layer_id",
)

PRESPECIFICATION = {
    "seed": SEED,
    "n_bootstrap": N_BOOTSTRAP,
    "rng": "numpy.random.Generator(PCG64)",
    "ci_percentiles": list(CI_PERCENTILES),
    "rho_offset": RHO_OFFSET,
    "probability_thresholds": list(PROBABILITY_THRESHOLDS),
    "primary_penalty": PRIMARY_PENALTY,
    "large_coefficient_diagnostic": SEPARATION_ABS_COEF,
    "separation_method": "signed-design complete/quasi LP; no coefficient cutoff",
    "irls_max_iter": IRLS_MAX_ITER,
    "irls_tol": IRLS_TOL,
    "reference_layer": REFERENCE_LAYER,
    "layer_order": list(LAYER_ORDER),
    "eval_rho": list(EVAL_RHO),
    "map_scales": list(MAP_SCALES),
    "n_signal_bins": N_SIGNAL_BINS,
    "bin_rule": "one pooled log-equal partition of finite SR>0; minimum included in the first bin; SR<=0 is an extra stratum",
    "magnitude_contour": "log(W/abs(E_true)) on log(SR) and log(rho+0.05), with and without layer indicators; crossing where the fitted ratio equals 1",
    "magnitude_calendar_event": "W/abs(E_true)<1 when W and E_true are finite and E_true!=0; W=0 and E_true=0 stay in the denominator and are undefined in the log contour",
    "sign_rule": "inf>0 or sup<0; an endpoint at 0 is not determined",
    "cluster": "realization id, one draw applied to every stratum, pair, lead, scale and month",
    "phase2_cell_columns": list(PHASE2_CELL_COLUMNS),
    "phase3_cell_extras": list(PHASE3_CELL_EXTRAS),
    "optional_later_columns": list(OPTIONAL_LATER_COLUMNS),
}


class Support:
    """Empirical transformed-covariate hull; legacy boxes only for helper callers."""
    def __init__(self, log_sr_min, log_sr_max, rho_min, rho_max, rho_observed, points=None):
        self.log_sr_min, self.log_sr_max = float(log_sr_min), float(log_sr_max)
        self.rho_min, self.rho_max = float(rho_min), float(rho_max)
        self.rho_observed = tuple(float(v) for v in rho_observed)
        self.points = None if points is None else np.unique(np.asarray(points, float), axis=0)
        self.hull = None
        if self.points is not None and len(self.points) >= 3 and np.linalg.matrix_rank(self.points-self.points[0]) == 2:
            self.hull = ConvexHull(self.points)

    def contains(self, log_sr, rho):
        x = np.array([log_sr, math.log(rho + RHO_OFFSET)])
        if self.points is None:
            return self.log_sr_min-1e-12 <= log_sr <= self.log_sr_max+1e-12 and self.rho_min-1e-12 <= rho <= self.rho_max+1e-12
        if self.hull is not None:
            return bool(np.all(self.hull.equations[:,:-1] @ x + self.hull.equations[:,-1] <= 1e-10))
        if not len(self.points):
            return False
        v = self.points-self.points[0]
        _, sigma, vh = np.linalg.svd(v, full_matrices=False)
        if not len(sigma) or sigma[0] < 1e-12:
            return bool(np.linalg.norm(x-self.points[0]) <= 1e-10)
        axis = vh[0]
        z = (x-self.points[0]) @ axis
        residual = (x-self.points[0])-z*axis
        t = v @ axis
        return bool(np.linalg.norm(residual)<=1e-10 and t.min()-1e-10<=z<=t.max()+1e-10)


def ok_fit(coefficients, support):
    return {
        "status": "ok",
        "coefficients": dict(coefficients),
        "support": support,
        "n": None,
        "log_likelihood": None,
    }


def _blank(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def _float(value):
    if _blank(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first(row, names):
    for name in names:
        if name in row and not _blank(row.get(name)):
            return row.get(name)
    return None


ROW_SCHEMA = {
    "identity": ["source_phase", "case_id", "pair_id", "lead_day", "track", "quantity"],
    "source_phases": ["phase2", "phase3", "phase4"],
    "metadata": ["realization", "layer_id", "storage_type", "SR", "SR_definition", "rho", "Q_scale", "N_transitions", "experiment_group"],
    "outcomes": ["env_inf_m", "env_sup_m", "W_m", "W_status", "E_true_m", "E_tool_m"],
    "reference_required": ["E_reference_m", "reference_status", "reference_source_path", "reference_sha256"],
    "calendar_required": ["calendar_type", "origin_month", "q_origin_m3d", "offseason_use_variant"],
    "undefined_numeric": "None, never epsilon/zero substitution",
    "execution_guard": "official=True calls require_frozen; explicit complete cohort manifest and actual reference receipts required",
    "primary_horizons": {"10":10,"30":30},
    "reference_file_contract": "source_phase/case_id/reference_path -> JSON pairs[pair].E_point[lead-1], complete 30-day vector; raw-byte hash receipt",
}


def source_phase(row):
    value=_first(row,("source_phase","dataset"))
    return {"D03":"phase2","D04":"phase3","D05":"phase4","phase2":"phase2","phase3":"phase3","phase4":"phase4"}.get(value)


def _case_key(row):
    return source_phase(row),str(row.get("case_id"))


def _row_key(row):
    return (*_case_key(row),row.get("pair_id"),int(row.get("lead_day")))


def build_cohort_manifest(phase2, phase3, phase4, official=True):
    """Consume actual derived manifests; no outcome-based selection or file writes."""
    records=[]
    for phase,items in (("phase2",phase2),("phase3",phase3),("phase4",phase4)):
        for item in items:
            r=dict(item,source_phase=phase)
            n=_first(r,("N_nominal","N_transitions"));group=r.get("experiment_group")
            A=(phase=="phase2" and _float(n)==6) or (phase=="phase3" and group=="B") or (phase=="phase4" and group=="A")
            B=phase=="phase4" and group=="B"
            records.append({"source_phase":phase,"case_id":str(r["case_id"]),"cohort_A":A,"cohort_B":B,"cohort_C":True,"realization":int(r["realization"])})
    index={_case_key(r):r for r in records}
    if len(index)!=len(records): raise ValueError("duplicate source case in cohort_manifest")
    counts={"C":len(records),"A":sum(r["cohort_A"] for r in records),"B":sum(r["cohort_B"] for r in records)}
    if official and (counts!={"C":4110,"A":1260,"B":2160} or (len(phase2),len(phase3),len(phase4))!=(540,690,2880)):
        raise ValueError(f"incomplete cohort_manifest coverage {counts}")
    if any(r["realization"] not in range(10) for r in records): raise ValueError("realization must be0..9")
    return {"records":records,"counts":counts,"official":official}


def reference_rows_from_files(entries):
    """Read already computed references only; never fit or score anything.

    Each entry: source_phase, case_id, reference_path, optional expected_sha256.
    Return four keyed raw lead rows/case with actual file hash/status receipts.
    A missing/failed reference is explicit, never an optional absent column.
    """
    rows=[]
    for entry in entries:
        path=Path(entry["reference_path"]);digest=None;status="ok";payload={};error=None
        try:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if entry.get("expected_sha256") not in (None,digest): raise ValueError("reference hash mismatch")
            payload=json.loads(raw)
            if str(payload.get("case_id"))!=str(entry["case_id"]): raise ValueError("reference case mismatch")
            if payload.get("error") is not None: status="error"
        except FileNotFoundError: status="missing"
        except (OSError,ValueError,TypeError) as exc: status="error";error=str(exc)
        for pair in PAIRS:
            values=payload.get("pairs",{}).get(pair,{}).get("E_point",[])
            for lead in LEADS:
                value=_float(values[lead-1]) if status=="ok" and isinstance(values,list) and len(values)==30 else None
                rows.append({"source_phase":source_phase(entry),"case_id":str(entry["case_id"]),"pair_id":pair,"lead_day":lead,"E_reference_m":value,"reference_status":status if status!="ok" else "ok" if value is not None else "nonfinite_or_incomplete","reference_source_path":str(path),"reference_sha256":digest,"reference_error":error})
    return rows


def join_reference_rows(rows, reference_rows=None):
    if reference_rows is None:
        joined=[dict(r) for r in rows]
    else:
        refs={}
        for ref in reference_rows:
            key=_row_key(ref)
            if key in refs: raise ValueError("duplicate reference key")
            refs[key]=ref
        joined=[]
        for r in rows:
            key=_row_key(r)
            if key not in refs: raise ValueError(f"mandatory reference row absent: {key}")
            joined.append(dict(r,**{k:v for k,v in refs[key].items() if k.startswith("reference_") or k=="E_reference_m"}))
    for r in joined:
        if "reference_status" not in r or "E_reference_m" not in r: raise ValueError("mandatory reference estimate/status columns absent")
        if r["reference_status"]=="ok" and _float(r["E_reference_m"]) is None: raise ValueError("reference status ok with nonfinite value")
        if r["reference_status"]!="ok" and _float(r["E_reference_m"]) is not None: raise ValueError("failed reference must not carry invented finite estimate")
    return joined


def validate_inventory(rows, cohort_manifest, official=True):
    if cohort_manifest is None: raise ValueError("explicit cohort_manifest required")
    entries=cohort_manifest["records"] if isinstance(cohort_manifest,dict) else cohort_manifest
    index={_case_key(r):r for r in entries}
    if len(index)!=len(entries): raise ValueError("duplicate cohort_manifest case")
    if official and (len(index)!=4110 or sum(r["cohort_A"] for r in entries)!=1260 or sum(r["cohort_B"] for r in entries)!=2160 or not all(r["cohort_C"] for r in entries) or Counter(r["source_phase"] for r in entries)!=Counter({"phase2":540,"phase3":690,"phase4":2880})):
        raise ValueError("incomplete or extraneous official cohort_manifest")
    found=set();keys=set();counts={"A":0,"B":0,"C":0}
    for row in rows:
        key=_case_key(row)
        if key not in index: raise ValueError(f"case absent from cohort_manifest: {key}")
        if not _lead_row(row) or row["pair_id"] not in PAIRS: raise ValueError("only primary raw lead rows may enter statistics")
        rk=_row_key(row)
        if rk in keys: raise ValueError("duplicate primary row")
        keys.add(rk);found.add(key)
        entry=index[key]
        if row["realization"]!=entry["realization"]: raise ValueError("cohort realization mismatch")
        for c in counts: counts[c]+=bool(entry[f"cohort_{c}"])
        if entry["cohort_A"] and not _map_family(row): raise ValueError("map metadata inconsistent with explicit manifest")
        if entry["cohort_B"] and (row["calendar_type"] not in CALENDAR_TYPES or row["origin_month"] not in range(1,13) or row.get("q_origin_m3d") is None or row["q_origin_m3d"]<0): raise ValueError("calendar schema incomplete")
        if official:
            if row["realization"] not in range(10): raise ValueError("realization must be0..9")
            if row.get("tool_horizon") is not None and int(row["tool_horizon"])!=row["lead_day"]: raise ValueError("primary tool horizon mismatch")
            if not row.get("reference_source_path") or (row.get("reference_status")=="ok" and not row.get("reference_sha256")): raise ValueError("actual reference provenance required")
    if official:
        expected={(ph,cid,p,l) for ph,cid in index for p in PAIRS for l in LEADS}
        if keys!=expected or counts!={"A":5040,"B":8640,"C":16440} or {r["realization"] for r in rows}!=set(range(10)):
            raise ValueError(f"incomplete requested statistics inventory {counts}")
    return index,counts


def _layer_id(row):
    explicit = _first(row, ("layer_id",))
    if explicit is not None:
        return str(explicit)
    sid = _first(row, ("sid",))
    if sid is not None:
        return str(sid)
    storage = _first(row, ("storage_type",))
    transmissivity = _float(_first(row, ("T_m2_d", "T")))
    if storage is None or transmissivity is None:
        return None
    if abs(transmissivity - round(transmissivity)) < 1e-8:
        return f"{storage}_T{int(round(transmissivity))}"
    return f"{storage}_T{transmissivity}"


def _sign_from_envelope(inf, sup):
    if inf is None or sup is None or inf > sup:
        return None, None, None
    if inf > 0:
        return 1, 1, 0
    if sup < 0:
        return 1, -1, 0
    return 0, 0, 1


def _binary_sign(value):
    if value is None:
        return None
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def adapt_row(row):
    out = dict(row)
    out["case_id"] = None if _blank(row.get("case_id")) else str(row.get("case_id"))
    out["source_phase"] = source_phase(row)
    out["layer_id"] = _layer_id(row)
    out["realization"] = _float(_first(row, ("realization", "realization_id")))
    if out["realization"] is not None and abs(out["realization"] - round(out["realization"])) < 1e-8:
        out["realization"] = int(round(out["realization"]))
    out["SR"] = _float(_first(row, ("SR", "signal_ratio", "sr")))
    out["rho"] = _float(_first(row, ("rho", "rho_nominal")))
    out["N_transitions"] = _float(_first(row, ("N_transitions", "N_nominal", "N")))
    scale = _float(_first(row, ("Q_scale", "scale")))
    out["Q_scale"] = 1.0 if scale is None and source_phase(row) == "phase2" else scale
    out["Q_scale_imputed"] = scale is None
    out["pair_id"] = None if _blank(row.get("pair_id")) else str(row.get("pair_id"))
    out["track"] = None if _blank(row.get("track")) else str(row.get("track"))
    out["quantity"] = None if _blank(row.get("quantity")) else str(row.get("quantity"))
    lead = _float(_first(row, ("lead_day", "lead")))
    out["lead_day"] = None if lead is None else int(round(lead))
    out["dataset"] = None if _blank(row.get("dataset")) else str(row.get("dataset"))
    out["experiment_group"] = None if _blank(row.get("experiment_group")) else str(row.get("experiment_group"))
    out["recency_label"] = None if _blank(row.get("recency_label")) else str(row.get("recency_label"))
    out["storage_type"] = None if _blank(row.get("storage_type")) else str(row.get("storage_type"))
    out["W_m"] = _float(_first(row, ("W_m", "W")))
    out["W_status"] = None if _blank(row.get("W_status")) else str(row.get("W_status"))
    out["E_true_m"] = _float(_first(row, ("E_true_m", "E_true")))
    out["E_tool_m"] = _float(_first(row, ("E_tool_m", "E_tool")))
    reference = _first(row, ("E_reference_m", "ref_E_m", "E_ref_m"))
    out["E_reference_m"] = _float(reference)
    out["reference_present"] = reference is not None
    calendar = _first(row, ("calendar_type", "operating_form", "calendar_form", "form"))
    out["calendar_type"] = None if calendar is None else {"paddy":"paddy_irrigation", "domestic":"domestic_continuous", "water_curtain_offseason":"water_curtain"}.get(str(calendar), str(calendar))
    out["offseason_use_variant"] = str(calendar) == "water_curtain_offseason" or row.get("offseason_use_variant") in (True,1,"1","true","True")
    out["SR_definition"] = row.get("SR_definition") or ("origin_rate" if calendar is not None else "outside_designated_rest")
    out["q_origin_m3d"] = _float(_first(row,("q_origin_m3d","q0_m3d")))
    month = _float(_first(row, ("origin_month", "month", "forecast_origin_month")))
    out["origin_month"] = None if month is None else int(round(month))
    inf = _float(_first(row, ("env_inf_m", "inf")))
    sup = _float(_first(row, ("env_sup_m", "sup")))
    recomputed, envelope_sign, includes_zero = _sign_from_envelope(inf, sup)
    for key, expected in (("sign_determined",recomputed),("envelope_sign",envelope_sign)):
        supplied = row.get(key)
        if not _blank(supplied) and _float(supplied) != expected:
            raise ValueError(f"{key} inconsistent with actual envelope for {out['case_id']}")
    supplied_width = out["W_m"]
    if inf is not None and sup is not None and inf<=sup and supplied_width is not None and not math.isclose(sup-inf,supplied_width,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f"W_m inconsistent with actual envelope for {out['case_id']}")
    out["sign_determined"] = recomputed
    out["envelope_sign"] = envelope_sign
    out["includes_zero"] = includes_zero
    out["env_inf_m"] = inf
    out["env_sup_m"] = sup
    return out


def _scale_allowed(scale):
    return scale is not None and any(math.isclose(scale, item, rel_tol=0.0, abs_tol=1e-9) for item in MAP_SCALES)


def _lead_row(row):
    if row["lead_day"] not in LEADS:
        return False
    if row["track"] != "raw":
        return False
    return row["quantity"] in (None, "lead")


def _map_family(row):
    # Fixture-only lineage predicate. Official execution overrides with manifest membership.
    phase = row.get("source_phase")
    eligible = phase == "phase2" or (phase == "phase3" and row.get("experiment_group")=="B") or (phase == "phase4" and row.get("experiment_group")=="A")
    return bool(eligible and row.get("calendar_type") is None and _lead_row(row) and row.get("N_transitions")==6 and row.get("layer_id") in LAYER_ORDER and _scale_allowed(row.get("Q_scale")) and row.get("rho") is not None and row["rho"]>=0 and row.get("recency_label") not in EXCLUDED_RECENCY)


def magnitude_resolved(W, E_true):
    if W is None or E_true is None or E_true == 0 or W < 0:
        return None
    return abs(W) / abs(E_true) < 1.0


def width_event(row):
    if row.get("W_status") in ("error", "missing", "invalid"):
        return None
    return magnitude_resolved(row.get("W_m"), row.get("E_true_m"))


def classify_row(row):
    cohort_A = bool(row.get("cohort_A", _map_family(row)))
    sign_exclusion = None
    magnitude_exclusion = None
    if not cohort_A:
        sign_exclusion = "not_cohort_A"
        magnitude_exclusion = "not_cohort_A"
    elif row["sign_determined"] is None or row.get("W_status") in ("error","missing","invalid"):
        sign_exclusion = "invalid_envelope"
        magnitude_exclusion = "invalid_envelope"
    elif row["SR"] is None:
        sign_exclusion = "nonfinite_signal"
        magnitude_exclusion = "nonfinite_signal"
    elif row["SR"] <= 0:
        sign_exclusion = "nonpositive_signal"
        magnitude_exclusion = "nonpositive_signal"
    else:
        if row["sign_determined"] not in (0, 1):
            sign_exclusion = "missing_sign"
        if row["E_true_m"] == 0:
            magnitude_exclusion = "zero_true_effect"
        elif row["W_m"] == 0:
            magnitude_exclusion = "zero_width"
        elif row["W_status"] in ("error", "missing", "invalid") or row["W_m"] is None or row["W_m"] < 0 or row["E_true_m"] is None:
            magnitude_exclusion = "undefined_width"
    sign_likelihood = cohort_A and sign_exclusion is None
    magnitude_likelihood = cohort_A and magnitude_exclusion is None
    cohort_B = bool(row.get("cohort_B", row["calendar_type"] is not None or row["origin_month"] is not None))
    cohort_C = bool(row.get("cohort_C", _lead_row(row)))
    return {
        "cohort_A": cohort_A,
        "cohort_B": cohort_B and _lead_row(row),
        "cohort_C": cohort_C,
        "sign_likelihood": sign_likelihood,
        "sign_exclusion": sign_exclusion,
        "magnitude_likelihood": magnitude_likelihood,
        "magnitude_exclusion": magnitude_exclusion,
        "dropped": False,
    }


def logit(probability):
    if not 0 < probability < 1:
        raise ValueError("probability must lie strictly between 0 and 1")
    return math.log(probability / (1.0 - probability))


def separation_diagnostic(y, X):
    """Finite-MLE existence diagnostic on a full-rank standardized design."""
    Z = (2*np.asarray(y)-1)[:,None]*X
    complete = linprog(np.zeros(X.shape[1]), A_ub=-Z, b_ub=-np.ones(len(Z)), bounds=[(None,None)]*X.shape[1], method="highs")
    if complete.success:
        return {"status":"complete_separation"}
    if complete.status != 2:
        return {"status":"separation_solver_failure", "solver_status":int(complete.status)}
    quasi = linprog(-Z.mean(axis=0), A_ub=-Z, b_ub=np.zeros(len(Z)), bounds=[(-1,1)]*X.shape[1], method="highs")
    if not quasi.success:
        return {"status":"separation_solver_failure", "solver_status":int(quasi.status)}
    margin = float(-quasi.fun)
    return {"status":"quasi_separation" if margin>1e-9 else "overlap", "mean_margin":margin}


def fit_logistic(y, X, sample_weight=None, column_names=None, max_iter=None):
    y, X = np.asarray(y,float), np.asarray(X,float)
    if X.ndim!=2 or y.ndim!=1 or len(y)!=len(X) or not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isin(y,[0,1]).all():
        raise ValueError("finite design and binary response required")
    names = tuple(column_names) if column_names is not None else tuple(f"b{i}" for i in range(X.shape[1]))
    if len(names)!=X.shape[1]: raise ValueError("column names do not match design")
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,float)
    if sw.shape!=y.shape or not np.isfinite(sw).all() or (sw<0).any(): raise ValueError("invalid sample weights")
    keep=sw>0; y,X,sw=y[keep],X[keep],sw[keep]
    failed={"status":"empty","coefficients":None,"n":len(y),"n_events":float(sw@y),"log_likelihood":None,"converged":False}
    if not len(y): return failed
    if np.all(y==y[0]): return dict(failed,status="constant_outcome")
    scale=np.sqrt(np.mean(X*X,axis=0));scale[scale==0]=1.;XX=X/scale
    sv=np.linalg.svd(XX,compute_uv=False)
    if len(sv)<X.shape[1] or sv[-1]<=1e-10*sv[0]: return dict(failed,status="rank_deficient")
    diag=separation_diagnostic(y,XX)
    if diag["status"]!="overlap": return dict(failed,**diag)
    beta=np.zeros(X.shape[1]);converged=False
    for _ in range(IRLS_MAX_ITER if max_iter is None else max_iter):
        mu=np.clip(expit(XX@beta),PROBABILITY_CLIP,1-PROBABILITY_CLIP)
        hessian=XX.T@((sw*mu*(1-mu))[:,None]*XX)
        try: step=np.linalg.solve(hessian,XX.T@(sw*(y-mu)))
        except np.linalg.LinAlgError: return dict(failed,status="singular")
        if not np.isfinite(step).all(): return dict(failed,status="numerical_failure")
        beta+=step
        if np.max(np.abs(step))<IRLS_TOL: converged=True;break
    if not converged: return dict(failed,status="nonconvergence")
    mu=np.clip(expit(XX@beta),PROBABILITY_CLIP,1-PROBABILITY_CLIP)
    hessian=XX.T@((sw*mu*(1-mu))[:,None]*XX);eig=np.linalg.eigvalsh(hessian)
    if eig[0]<=1e-8*max(1.,float(eig[-1])): return dict(failed,status="ill_conditioned")
    coefficients=beta/scale
    if not np.isfinite(coefficients).all(): return dict(failed,status="numerical_failure")
    eta=XX@beta;ll=float(np.sum(sw*(y*eta-np.logaddexp(0.,eta))))
    return dict(failed,status="ok",coefficients=dict(zip(names,coefficients.tolist())),log_likelihood=ll,converged=True,hessian_min_eig=float(eig[0]),large_coefficient_diagnostic=bool(np.max(np.abs(coefficients))>SEPARATION_ABS_COEF),separation_diagnostic=diag)


def _rho_key(rho):
    return str(float(rho))


def _crossing(fit, target, rho, layer_id):
    result={"rho":float(rho),"layer_id":layer_id,"signal_ratio":None,"log_signal_ratio":None,"status":fit.get("status","missing"),"reporting_status":"unidentifiable","out_of_support":False,"sr_out_of_observed_support":False,"rho_not_observed":False,"rho_out_of_observed_range":False,"rho_interpolated":False}
    if rho<0 or not math.isfinite(rho): raise ValueError("rho must be finite nonnegative")
    coef=fit.get("coefficients")
    if fit.get("status")!="ok" or not coef: return result
    slope=coef.get("log_signal_ratio")
    if slope is None or not math.isfinite(slope) or abs(slope)<=ZERO_SLOPE:
        return dict(result,status="zero_slope")
    if fit.get("with_layers") and layer_id not in fit.get("layers_present",[]): return dict(result,status="layer_absent")
    layer=0.
    if layer_id is not None and layer_id!=REFERENCE_LAYER:
        key=f"layer:{layer_id}"
        if key not in coef: return dict(result,status="layer_absent")
        layer=coef[key]
    if not all(math.isfinite(v) for v in coef.values()): return dict(result,status="nonfinite_coefficients")
    log_sr=(target-coef["intercept"]-coef["log_rho_plus_offset"]*math.log(rho+RHO_OFFSET)-layer)/slope
    result["log_signal_ratio"]=float(log_sr)
    if not math.isfinite(log_sr): return dict(result,status="nonfinite_root")
    try: value=math.exp(log_sr)
    except OverflowError: return dict(result,status="exponent_overflow")
    if value==0: return dict(result,status="exponent_underflow")
    result.update(signal_ratio=value,status="ok")
    support=fit.get("layer_support",{}).get(layer_id) if fit.get("with_layers") else fit.get("support")
    if support is not None:
        result["sr_out_of_observed_support"]=log_sr<support.log_sr_min-1e-12 or log_sr>support.log_sr_max+1e-12
        result["rho_not_observed"]=not any(math.isclose(rho,v,abs_tol=1e-9,rel_tol=0.) for v in support.rho_observed)
        result["rho_out_of_observed_range"]=rho<support.rho_min-1e-12 or rho>support.rho_max+1e-12
        result["out_of_support"]=not support.contains(log_sr,rho)
        result["rho_interpolated"]=result["rho_not_observed"] and not result["out_of_support"]
    result["reporting_status"]="extrapolated" if result["out_of_support"] else "finite_inside"
    return result


def crossing_from_fit(fit, probability, rho, layer_id):
    return dict(_crossing(fit,logit(probability),rho,layer_id),probability=probability)


def magnitude_crossing_from_fit(fit, rho, layer_id):
    return dict(_crossing(fit,0.,rho,layer_id),contour="W_over_abs_E_equals_1",target_log_ratio=0.)


def fit_log_linear(y, X, sample_weight=None, column_names=None):
    """Weighted least squares for log(W/|E_true|). No penalty is added."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    names = tuple(column_names) if column_names is not None else tuple(f"b{i}" for i in range(X.shape[1]))
    if len(names) != X.shape[1] or len(y) != len(X):
        raise ValueError("log-linear design does not match y")
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    keep = np.isfinite(y) & np.isfinite(sw) & (sw > 0)
    y, X, sw = y[keep], X[keep], sw[keep]
    failed = {"status": "empty", "coefficients": None, "n": int(len(y)), "r2": None}
    if len(y) == 0:
        return failed
    xtw = X.T * sw
    gram = xtw @ X
    try:
        beta = np.linalg.solve(gram, xtw @ y)
    except np.linalg.LinAlgError:
        failed["status"] = "singular"
        return failed
    if not np.all(np.isfinite(beta)):
        failed["status"] = "singular"
        return failed
    eigenvalues = np.linalg.eigvalsh(gram)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) <= 1e-10 * scale:
        failed["status"] = "singular"
        return failed
    fitted = X @ beta
    resid = y - fitted
    ss_res = float(np.sum(sw * resid ** 2))
    weight_sum = float(np.sum(sw))
    center = float(np.sum(sw * y) / weight_sum)
    ss_tot = float(np.sum(sw * (y - center) ** 2))
    return {
        "status": "ok",
        "coefficients": {name: float(value) for name, value in zip(names, beta)},
        "n": int(len(y)),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "r2": None if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "converged": True,
    }


def percentile_interval(values, n_not_identified):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    out = {
        "n_identified": len(finite),
        "n_not_identified": int(n_not_identified),
        "low": None,
        "high": None,
        "status": "conditional_on_identified_replicates" if finite else "unidentifiable",
    }
    if finite:
        out["low"] = float(np.percentile(finite, CI_PERCENTILES[0]))
        out["high"] = float(np.percentile(finite, CI_PERCENTILES[1]))
    return out


def log_equal_bins(signal_ratios):
    values = np.asarray(list(signal_ratios), dtype=float)
    finite = values[np.isfinite(values)]
    nonpositive = finite[finite <= 0]
    positive = finite[finite > 0]
    if len(positive) == 0:
        edges = [None] * (N_SIGNAL_BINS + 1)
        counts = [0] * N_SIGNAL_BINS
    else:
        logs = np.linspace(math.log(float(positive.min())), math.log(float(positive.max())), N_SIGNAL_BINS + 1)
        edges = [float(math.exp(v)) for v in logs]
        if math.isclose(edges[0], edges[-1], rel_tol=0.0, abs_tol=1e-15):
            counts = [int(len(positive))] + [0] * (N_SIGNAL_BINS - 1)
        else:
            index = np.digitize(positive, edges[1:-1], right=True)
            counts = [int(np.sum(index == i)) for i in range(N_SIGNAL_BINS)]
    return {
        "n_bins": N_SIGNAL_BINS,
        "pooled": True,
        "edges": edges,
        "counts": counts,
        "n_positive_signal": int(len(positive)),
        "n_nonpositive_signal": int(len(nonpositive)),
        "rule": PRESPECIFICATION["bin_rule"],
    }


def _bin_index(signal, edges):
    if signal is None:
        return None
    if signal <= 0:
        return "nonpositive_signal"
    if not edges or edges[0] is None:
        return None
    if math.isclose(edges[0], edges[-1], rel_tol=0.0, abs_tol=1e-15):
        return 0
    return int(np.digitize([signal], edges[1:-1], right=True)[0])


def _rate(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": None if denominator == 0 else numerator / denominator,
    }


def _comparison_rates(rows, draws=None):
    determined=lambda r:r.get("sign_determined")==1
    tool_ok=lambda r:determined(r) and r.get("E_tool_m") is not None
    ref_ok=lambda r:determined(r) and r.get("E_reference_m") is not None and r.get("reference_status","ok")=="ok"
    paired=lambda r:tool_ok(r) and ref_ok(r)
    und=lambda r:r.get("sign_determined")==0 and r.get("E_true_m") not in (None,0)
    def opposite(r,col): return r.get(col) is not None and r[col]*r["envelope_sign"]<0
    def disagree(r,col): return _binary_sign(r.get(col))!=r.get("envelope_sign")
    def rate(den,yes):
        out=_rate(sum(den(r) and yes(r) for r in rows),sum(den(r) for r in rows))
        if draws: out["bootstrap"]=_conditional_bootstrap(rows,draws,den,yes)
        return out
    out={
        "determined_total":sum(determined(r) for r in rows),
        "tool_contradiction":rate(tool_ok,lambda r:opposite(r,"E_tool_m")),
        "tool_disagreement_including_zero":rate(tool_ok,lambda r:disagree(r,"E_tool_m")),
        "zero_tool_prediction":rate(tool_ok,lambda r:r["E_tool_m"]==0),
        "missing_tool":rate(determined,lambda r:r.get("E_tool_m") is None),
        "reference_contradiction":rate(ref_ok,lambda r:opposite(r,"E_reference_m")),
        "reference_disagreement_including_zero":rate(ref_ok,lambda r:disagree(r,"E_reference_m")),
        "reference_zero":rate(ref_ok,lambda r:r["E_reference_m"]==0),
        "reference_agreement":rate(ref_ok,lambda r:not disagree(r,"E_reference_m")),
        "missing_reference":rate(determined,lambda r:not ref_ok(r)),
        "paired_tool_contradiction":rate(paired,lambda r:opposite(r,"E_tool_m")),
        "paired_reference_contradiction":rate(paired,lambda r:opposite(r,"E_reference_m")),
        "undetermined_truth_error":rate(lambda r:und(r) and r.get("E_tool_m") is not None,lambda r:_binary_sign(r["E_tool_m"])!=_binary_sign(r["E_true_m"])),
        "reference_undetermined_truth_disagreement":rate(lambda r:und(r) and r.get("E_reference_m") is not None and r.get("reference_status","ok")=="ok",lambda r:_binary_sign(r["E_reference_m"])!=_binary_sign(r["E_true_m"])),
        "n_zero_truth":sum(r.get("E_true_m")==0 for r in rows),
        "n_nonfinite_truth":sum(r.get("E_true_m") is None for r in rows),
        "undetermined_absolute_errors":[{"source_phase":r.get("source_phase"),"case_id":r.get("case_id"),"pair_id":r.get("pair_id"),"lead_day":r.get("lead_day"),"tool_absolute_error":None if r.get("E_tool_m") is None or r.get("E_true_m") is None else abs(r["E_tool_m"]-r["E_true_m"])} for r in rows if r.get("sign_determined")==0],
    }
    out["strict_opposite_sign"]=out["tool_contradiction"]
    return out


def contradiction_table(rows, edges, draws=None):
    """Strict opposite sign primary; tool/reference zero and missing are distinct."""
    out=_comparison_rates(rows,draws)
    def grouped(keys):
        groups={}
        for r in rows:
            key=tuple(_bin_index(r.get("SR"),edges) if k=="bin" else r.get(k) for k in keys)
            groups.setdefault(key,[]).append(r)
        return [dict(zip(keys,key),n=len(rs),**_comparison_rates(rs,draws)) for key,rs in sorted(groups.items(),key=lambda item:str(item[0]))]
    out.update(undetermined_cases=[{"case_id":r.get("case_id"),"source_phase":r.get("source_phase"),"layer_id":r.get("layer_id"),"bin":_bin_index(r.get("SR"),edges)} for r in rows if r.get("sign_determined")==0],by_bin_layer=grouped(("bin","layer_id","pair_id","lead_day")),by_bin_storage=grouped(("bin","storage_type","pair_id","lead_day")),by_source_signal_definition=grouped(("source_phase","SR_definition","pair_id","lead_day")),by_calendar_month_form=grouped(("origin_month","calendar_type","pair_id","lead_day")) if any(r.get("calendar_type") for r in rows) else [],role="Structurally matched reference on the same rows; no ranking.")
    return out


def _conditional_bootstrap(rows, draws, denominator_fn, success_fn):
    # Sufficient counts per realization preserve every paired draw while avoiding
    # a 999-fold scan of the full4110-case row table for each reporting cell.
    numerator=np.zeros(10);denominator=np.zeros(10)
    for row in rows:
        rr=row.get("realization")
        if rr is None or rr not in range(10) or not denominator_fn(row): continue
        denominator[int(rr)]+=1
        if success_fn(row): numerator[int(rr)]+=1
    matrix=np.asarray([[draw.get(i,0) for i in range(10)] for draw in draws],float).reshape((-1,10))
    den=matrix@denominator;num=matrix@numerator
    defined=den>0
    return percentile_interval((num[defined]/den[defined]).tolist(),int(np.sum(~defined)))


def bootstrap_rate(rows, draws, numerator_fn):
    return _conditional_bootstrap(rows,draws,lambda row:True,numerator_fn)


def calendar_rates(rows, draws=None):
    rows=[r for r in rows if r.get("calendar_type") is not None]
    def aggregate(keys):
        groups={}
        for r in rows: groups.setdefault(tuple(r.get(k) for k in keys),[]).append(r)
        out=[]
        for key,rs in sorted(groups.items(),key=lambda item:str(item[0])):
            no_active=lambda r:r.get("q_origin_m3d")==0
            active=lambda r:r.get("q_origin_m3d") is not None and r["q_origin_m3d"]>0
            determined=lambda r:r.get("sign_determined")==1
            below=lambda r:width_event(r) is True
            def rate(den,yes):
                item=_rate(sum(den(r) and yes(r) for r in rs),sum(den(r) for r in rs))
                if draws: item["bootstrap"]=_conditional_bootstrap(rs,draws,den,yes)
                return item
            cell=dict(zip(keys,key))
            cell.update(denominator=len(rs),relative_width_denominator=len(rs),n_sign_determined=sum(determined(r) for r in rs),n_sign_undetermined=sum(r.get("sign_determined")==0 for r in rs),n_sign_missing=sum(r.get("sign_determined") is None for r in rs),n_zero_signal=sum(r.get("SR")==0 for r in rs),n_nonpositive_signal=sum(r.get("SR") is not None and r["SR"]<=0 for r in rs),n_zero_true_effect=sum(r.get("E_true_m")==0 for r in rs),n_no_active_contrast=sum(no_active(r) for r in rs),n_active_contrast=sum(active(r) for r in rs),n_origin_rate_unknown=sum(r.get("q_origin_m3d") is None for r in rs),n_relative_width_below_one=sum(below(r) for r in rs),n_relative_width_undefined=sum(width_event(r) is None for r in rs),n_relative_width_defined=sum(width_event(r) is not None for r in rs),sign_rate=rate(lambda r:True,determined),relative_width_rate=rate(lambda r:True,below),active_sign_rate=rate(active,determined),active_relative_width_rate=rate(active,below),defined_relative_width_rate=rate(lambda r:width_event(r) is not None,below))
            cell["display_status"]="origin_rate_unknown" if cell["n_origin_rate_unknown"] else "no_active_contrast" if not cell["n_active_contrast"] else "active_and_inactive" if cell["n_no_active_contrast"] else "active"
            out.append(cell)
        return out
    keys=("origin_month","calendar_type","layer_id","storage_type","lead_day","pair_id")
    return {"cells":aggregate(keys),"storage_cells":aggregate(("origin_month","calendar_type","storage_type","lead_day","pair_id")),"variant_cells":aggregate(keys+("offseason_use_variant",)),"expected_calendar_types":list(CALENDAR_TYPES),"expected_months":list(range(1,13))}


def apply_draw(rows, multiplicity):
    weighted = []
    for row in rows:
        copied = dict(row)
        realization = row.get("realization")
        copied["bootstrap_weight"] = float(multiplicity.get(realization, 0))
        weighted.append(copied)
    return weighted


def _support_from_rows(rows):
    if not rows:
        return None
    logs = [math.log(row["SR"]) for row in rows]
    rhos = [row["rho"] for row in rows]
    return Support(min(logs), max(logs), min(rhos), max(rhos), tuple(sorted(set(rhos))), points=list(zip(logs,[math.log(v+RHO_OFFSET) for v in rhos])))


def _design(rows, with_layers):
    names = ["intercept", "log_signal_ratio", "log_rho_plus_offset"]
    columns = [
        np.ones(len(rows)),
        np.array([math.log(row["SR"]) for row in rows]),
        np.array([math.log(row["rho"] + RHO_OFFSET) for row in rows]),
    ]
    present = {row["layer_id"] for row in rows}
    if with_layers:
        for layer in LAYER_ORDER:
            if layer == REFERENCE_LAYER or layer not in present:
                continue
            names.append(f"layer:{layer}")
            columns.append(np.array([1.0 if row["layer_id"] == layer else 0.0 for row in rows]))
    return names, np.column_stack(columns), present


def _fit_rows(rows, y_name, with_layers, sample_weight=None, kind="logit"):
    if len(rows) == 0:
        return {"status": "empty", "coefficients": None, "n": 0, "support": None, "with_layers": with_layers}
    rows = [r for r in rows if sample_weight is None or r.get("bootstrap_weight",1.)>0]
    if not rows: return {"status":"empty","coefficients":None,"n":0,"support":None,"with_layers":with_layers}
    names, X, present = _design(rows, with_layers)
    y = np.array([row[y_name] for row in rows], dtype=float)
    weight = None if sample_weight is None else np.array([row.get("bootstrap_weight", 1.0) for row in rows])
    fitter = fit_logistic if kind == "logit" else fit_log_linear
    fit = fitter(y, X, sample_weight=weight, column_names=names)
    fit["support"] = _support_from_rows(rows)
    fit["layer_support"] = {layer:_support_from_rows([r for r in rows if r["layer_id"]==layer]) for layer in present}
    fit["with_layers"] = with_layers
    fit["layers_present"] = sorted(present)
    return fit


def _crossings_for(fit, with_layers):
    layers = [None] if not with_layers else list(LAYER_ORDER)
    found = {}
    for rho in EVAL_RHO:
        per_layer = {}
        for layer in layers:
            per_threshold = {}
            for probability in PROBABILITY_THRESHOLDS:
                per_threshold[str(probability)] = crossing_from_fit(fit, probability, rho, layer)
            per_layer["pooled" if layer is None else layer] = per_threshold
        found[_rho_key(rho)] = per_layer if with_layers else per_layer["pooled"]
    return found


def _attach_outcome(row):
    out = dict(row)
    out["y_sign"] = None if row["sign_determined"] not in (0, 1) else float(row["sign_determined"])
    if row["W_m"] is not None and row["W_m"] > 0 and row["E_true_m"] not in (None, 0):
        out["y_log_width"] = math.log(row["W_m"]) - math.log(abs(row["E_true_m"]))
    else:
        out["y_log_width"] = None
    return out


def _null_log_likelihood(rows, y_name):
    if not rows:
        return None
    y = np.array([row[y_name] for row in rows], dtype=float)
    fit = fit_logistic(y, np.ones((len(rows), 1)), column_names=("intercept",))
    return fit.get("log_likelihood")


def _layer_comparison(no_layer, with_layer, null_ll):
    if no_layer.get("status") != "ok" or with_layer.get("status") != "ok":
        return {"status": "unidentifiable", "mcfadden_no_layer": None, "mcfadden_layer": None, "lr_stat": None}
    ll0 = no_layer.get("log_likelihood")
    ll1 = with_layer.get("log_likelihood")
    if ll0 is None or ll1 is None:
        return {"status": "not_a_logit", "r2_no_layer": no_layer.get("r2"), "r2_layer": with_layer.get("r2"),
                "ss_res_no_layer": no_layer.get("ss_res"), "ss_res_layer": with_layer.get("ss_res")}
    def mcfadden(ll):
        if null_ll is None or null_ll == 0 or not math.isfinite(null_ll):
            return None
        return 1.0 - ll / null_ll
    n_indicators = len([key for key in (with_layer.get("coefficients") or {}) if key.startswith("layer:")])
    return {
        "status": "ok",
        "log_likelihood_no_layer": ll0,
        "log_likelihood_layer": ll1,
        "null_log_likelihood": null_ll,
        "deviance_no_layer": -2.0 * ll0,
        "deviance_layer": -2.0 * ll1,
        "mcfadden_no_layer": mcfadden(ll0),
        "mcfadden_layer": mcfadden(ll1),
        "lr_stat": 2.0 * (ll1 - ll0),
        "lr_df": n_indicators,
        "note": "descriptive layer comparison, not a completion gate",
    }


def _bootstrap_payload(point, replicates):
    values=[r["signal_ratio"] for r in replicates if r.get("status")=="ok" and r.get("signal_ratio") is not None]
    payload=percentile_interval(values,len(replicates)-len(values))
    payload["status_counts"]=dict(Counter(r.get("status","missing") for r in replicates))
    payload["n_extrapolated"]=sum(r.get("reporting_status")=="extrapolated" for r in replicates)
    if point.get("status")!="ok":
        payload.update(low=None,high=None,published=False,status="full_sample_unidentifiable")
    return payload


def _comparison_with_deltas(no_layer, with_layer, null_ll):
    comparison=_layer_comparison(no_layer,with_layer,null_ll)
    if comparison.get("status")=="ok" and comparison.get("mcfadden_no_layer") is not None and comparison.get("mcfadden_layer") is not None:
        comparison["delta_mcfadden"]=comparison["mcfadden_layer"]-comparison["mcfadden_no_layer"]
    if comparison.get("status")=="not_a_logit":
        if comparison.get("r2_layer") is not None and comparison.get("r2_no_layer") is not None:
            comparison["delta_r2"]=comparison["r2_layer"]-comparison["r2_no_layer"]
        comparison["delta_ss_res"]=comparison["ss_res_no_layer"]-comparison["ss_res_layer"]
    return comparison


def _weighted_null(rows, y_name, draw):
    n=sum(draw.get(r["realization"],0) for r in rows)
    events=sum(draw.get(r["realization"],0)*r[y_name] for r in rows)
    if not n or events in (0,n): return 0. if n else None
    p=events/n
    return events*math.log(p)+(n-events)*math.log1p(-p)


def _model_block(rows, y_name, draws, kind="logit"):
    block={};grouped={}
    for row in rows: grouped.setdefault((row["lead_day"],row["pair_id"]),[]).append(row)
    for lead in LEADS:
        lead_block={}
        for pair in PAIRS:
            subset=[r for r in grouped.get((lead,pair),[]) if r.get(y_name) is not None]
            pair_block={};fits={};replicate_fits={}
            for label,with_layers in (("no_layer",False),("layer",True)):
                fit=_fit_rows(subset,y_name,with_layers,kind=kind)
                get_crossings=_crossings_for if kind=="logit" else _width_crossings
                crossings=get_crossings(fit,with_layers)
                refits=[_fit_rows(apply_draw(subset,d),y_name,with_layers,sample_weight=True,kind=kind) for d in draws]
                trees=[get_crossings(f,with_layers) for f in refits]
                if kind=="logit":
                    for rho,layer,p,point in _walk_points(crossings,with_layers):
                        rep=[t[rho][layer][p] if with_layers else t[rho][p] for t in trees]
                        point["bootstrap"]=_bootstrap_payload(point,rep)
                else:
                    for rho,node in crossings.items():
                        for layer,point in (node.items() if with_layers else (("pooled",node),)):
                            rep=[t[rho][layer] if with_layers else t[rho] for t in trees]
                            point["bootstrap"]=_bootstrap_payload(point,rep)
                fits[label]=fit;replicate_fits[label]=refits
                pair_block[label]={k:fit.get(k) for k in ("status","coefficients","n","n_events","log_likelihood","r2","layers_present","large_coefficient_diagnostic")}
                pair_block[label].update(crossings=crossings,bootstrap_fit_status_counts=dict(Counter(f["status"] for f in refits)))
            comparison=_comparison_with_deltas(fits["no_layer"],fits["layer"],_null_log_likelihood(subset,y_name) if kind=="logit" else None)
            paired=[_comparison_with_deltas(a,b,_weighted_null(subset,y_name,draws[i]) if kind=="logit" else None) for i,(a,b) in enumerate(zip(replicate_fits["no_layer"],replicate_fits["layer"]))]
            comparison["bootstrap"]={}
            for metric in (("lr_stat","delta_mcfadden") if kind=="logit" else ("delta_r2","delta_ss_res")):
                values=[c[metric] for c in paired if _float(c.get(metric)) is not None]
                interval=percentile_interval(values,len(draws)-len(values))
                if _float(comparison.get(metric)) is None: interval.update(low=None,high=None,published=False,status="full_sample_unidentifiable")
                comparison["bootstrap"][metric]=interval
            comparison["bootstrap_status_counts"]=dict(Counter(c["status"] for c in paired))
            pair_block["layer_comparison"]=comparison;lead_block[pair]=pair_block
        block[str(lead)]=lead_block
    return block


def _width_crossings(fit, with_layers):
    layers = [None] if not with_layers else list(LAYER_ORDER)
    found = {}
    for rho in EVAL_RHO:
        per_layer = {}
        for layer in layers:
            crossing = magnitude_crossing_from_fit(fit, rho, layer)
            crossing["contour"] = "W_over_abs_E_equals_1"
            per_layer["pooled" if layer is None else layer] = crossing
        found[_rho_key(rho)] = per_layer if with_layers else per_layer["pooled"]
    return found


def _walk_points(tree, with_layers):
    points = []
    for rho_key, node in tree.items():
        if with_layers:
            for layer, thresholds in node.items():
                for probability, payload in thresholds.items():
                    points.append((rho_key, layer, probability, payload))
        else:
            for probability, payload in node.items():
                points.append((rho_key, "pooled", probability, payload))
    return points


def _draws(rows, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    universe=list(range(10));rng=np.random.Generator(np.random.PCG64(seed))
    draws=[]
    for _ in range(n_bootstrap):
        counts=np.bincount(rng.choice(universe,size=10,replace=True),minlength=10)
        draws.append({i:int(counts[i]) for i in universe})
    return universe,draws


def validate_draws(draws, official=True):
    universe=set(range(10))
    if any(set(d)!=universe or sum(d.values())!=10 or any(not isinstance(v,(int,np.integer)) or v<0 for v in d.values()) for d in draws):
        raise ValueError("draws require ten nonnegative integer cluster multiplicities summing to ten")
    if official and draws!=_draws([],N_BOOTSTRAP,SEED)[1]:
        raise ValueError("official draw matrix must be canonical PCG64 seed20261001/999 draws")


def run_statistics(rows, n_bootstrap=None, seed=None, draws=None, *, cohort_manifest=None, reference_rows=None, official=True):
    """Post-freeze analysis API; explicit official inventory and reference receipts required.

    official=False is for synthetic fixtures only and labels output accordingly.
    No function here runs an envelope, model forecast, or reference estimator.
    """
    if official and cohort_manifest is None: raise ValueError("explicit cohort_manifest required")
    protocol_digest=None
    if official:
        from p4_contracts import require_frozen
        protocol_digest=require_frozen()
    if official and (n_bootstrap not in (None,N_BOOTSTRAP) or seed not in (None,SEED)): raise ValueError("official bootstrap budget/seed fixed")
    if not official and reference_rows is None:
        rows=[dict(r,reference_status=r.get("reference_status","ok" if _float(r.get("E_reference_m")) is not None else "missing"),E_reference_m=r.get("E_reference_m")) for r in rows]
    rows=join_reference_rows(rows,reference_rows)
    adapted = [_attach_outcome(adapt_row(row)) for row in rows]
    if cohort_manifest is not None:
        membership,_=validate_inventory(adapted,cohort_manifest,official)
        for row in adapted:
            for c in ("A","B","C"): row[f"cohort_{c}"]=bool(membership[_case_key(row)][f"cohort_{c}"])
    flags = [classify_row(row) for row in adapted]
    if draws is None:
        universe, draws = _draws(adapted, N_BOOTSTRAP if n_bootstrap is None else n_bootstrap, SEED if seed is None else seed)
    else:
        universe = list(range(10))
        if official: validate_draws(draws,official=True)
    sign_rows = []
    magnitude_rows = []
    calendar_rows = []
    contradiction_rows = []
    n_nonpositive = 0
    n_zero_true = 0
    for row, flag in zip(adapted, flags):
        if flag["cohort_A"] and flag["sign_exclusion"] == "nonpositive_signal":
            n_nonpositive += 1
        if flag["cohort_A"] and row.get("E_true_m") == 0:
            n_zero_true += 1
        if flag["sign_likelihood"]:
            sign_rows.append(row)
        if flag["magnitude_likelihood"]:
            magnitude_rows.append(row)
        if flag["cohort_B"]:
            calendar_rows.append(row)
        if flag["cohort_C"]:
            contradiction_rows.append(row)
    bins = log_equal_bins([row["SR"] for row in contradiction_rows if row.get("SR") is not None])
    prespec = dict(PRESPECIFICATION)
    prespec["seed"] = SEED if seed is None else seed
    prespec["n_bootstrap"] = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    return {
        "status": "official_analysis_with_input_gaps" if official and any(r.get("reference_status")!="ok" or r.get("E_tool_m") is None or r.get("sign_determined") is None for r in adapted) else "official_analysis" if official else "synthetic_fixture",
        "protocol_sha256": protocol_digest,
        "row_schema": ROW_SCHEMA,
        "prespecification": prespec,
        "inventory": {
            "n_input_rows": len(adapted),
            "n_dropped": sum(flag["dropped"] for flag in flags),
            "dispositions": dict(Counter(flag["sign_exclusion"] or "sign_likelihood" for flag in flags)),
        },
        "cohort_A": {
            "n_rows": sum(flag["cohort_A"] for flag in flags),
            "n_sign_likelihood": len(sign_rows),
            "n_magnitude_likelihood": len(magnitude_rows),
            "n_nonpositive_signal": n_nonpositive,
            "n_zero_true_effect": n_zero_true,
            "models": {
                "sign": _model_block(sign_rows, "y_sign", draws, kind="logit"),
                "magnitude": _model_block(magnitude_rows, "y_log_width", draws, kind="loglinear"),
            },
            "positive_rho_sensitivity": {"sign":_model_block([r for r in sign_rows if r["rho"]>0],"y_sign",draws,kind="logit"),"magnitude":_model_block([r for r in magnitude_rows if r["rho"]>0],"y_log_width",draws,kind="loglinear")},
            "formula": {
                "sign": "logit Pr(sign determined) = b0 + b_s log(SR) + b_rho log(rho+0.05) [+ five layer indicators; confined_T50 baseline]",
                "sign_crossing": "SR(p, rho, layer) = exp((logit(p) - b0 - b_rho log(rho+0.05) - b_layer) / b_s) for p in {0.5, 0.9}",
                "magnitude": "log(W/abs(E_true)) = g0 + g_s log(SR) + g_rho log(rho+0.05) [+ the same layer indicators]",
                "magnitude_crossing": "SR_1 = exp((-(g0 + g_rho log(rho+0.05) + g_layer)) / g_s), the signal ratio where the fitted ratio is 1",
            },
        },
        "cohort_B": calendar_rates(calendar_rows, draws=draws),
        "cohort_C": {
            "bins": bins,
            "table": contradiction_table(contradiction_rows, bins["edges"], draws=draws),
            "n_rows": len(contradiction_rows),
        },
        "bootstrap": {
            "seed": prespec["seed"],
            "n": len(draws),
            "cluster_universe": universe,
            "draw_matrix": [[int(d.get(i,0)) for i in universe] for d in draws],
            "draw_matrix_sha256": hashlib.sha256(np.asarray([[d.get(i,0) for i in universe] for d in draws],dtype="<i8").tobytes()).hexdigest(),
            "shared_across": ["cohort_A", "cohort_B", "cohort_C", "pairs", "leads", "rho", "scales", "months"],
        },
    }


def self_test():
    suite_path = __file__.replace("p4_statistics.py", "p4_statistics_tests.py")
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName("p4_statistics_tests")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {"tests_path": suite_path, "passed": result.wasSuccessful(), "ran": result.testsRun}
