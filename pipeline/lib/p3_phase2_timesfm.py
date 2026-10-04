"""D03 observed-input/filter preflight only: no model import or inference entrypoint.
Proposed interface pending protocol owner freeze and author schedule-matching answer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import timesfm_pilot_adapter as pilot

PILOT_ADAPTER_SHA256 = "e90a9f906e1411e47b77e1da80387016913f0516f921eb265a1ab99584843c6e"
TRACKS = {"raw": None, "exp10": 10, "exp30": 30, "exp90": 90}
API = dict(pilot.API)
SNAPSHOT = pilot.SNAPSHOT
CONTEXT = 1024
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
PHYSICAL = {"confined": 1e-4, "leaky": 1e-3, "unconfined": 0.1}
INPUT_KEYS = {"head", "pumping", "rainfall", "query_id", "case_id", "pair_id", "schedule_id", "metadata_json", "protocol_sha256"}


def exp_filter(q, tau_days):
    """Daily causal exponential EWMA; z[0]=observed Q[0], no hidden prehistory.
    z[t]=exp(-1/tau)*z[t-1]+(1-exp(-1/tau))*Q[t]. Future uses same recurrence.
    Output has original Q units. No centring/scaling is applied here.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] == 0 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Q must be finite nonnegative [N,L], L>0")
    if tau_days not in (10, 30, 90):
        raise ValueError("Only prespecified 10/30/90 day scales are permitted")
    decay = np.exp(-1.0 / tau_days)
    weight = -np.expm1(-1.0 / tau_days)
    z = np.empty_like(q)
    z[:, 0] = q[:, 0]
    for t in range(1, q.shape[1]):
        z[:, t] = decay * z[:, t-1] + weight * q[:, t]
    return z


def validate(data, horizon, require_full_grid=False):
    if hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest() != PILOT_ADAPTER_SHA256:
        raise ValueError("Frozen pilot adapter hash mismatch")
    if set(data) != INPUT_KEYS:
        raise ValueError("Only observed forcing, head and declared metadata are allowed")
    ph = np.asarray(data["protocol_sha256"])
    if ph.shape != () or ph.dtype.kind not in "US" or len(str(ph.item())) != 64:
        raise ValueError("Scalar protocol SHA-256 required; preflight may use placeholder")
    try:
        int(str(ph.item()), 16)
    except ValueError as exc:
        raise ValueError("Invalid protocol SHA-256") from exc
    order, ids = pilot.validate(data, horizon)
    if (np.asarray(data["pumping"]) < 0).any():
        raise ValueError("Pumping must be nonnegative extraction")
    metas = [json.loads(s) for s in ids["metadata_json"]]
    cases = {}
    for i, meta in enumerate(metas):
        for key in ("storage_type", "storage_value", "T_m2_d", "r_m", "Q_m3_d", "transition_count", "t95_days"):
            if key not in meta:
                raise ValueError(f"Missing physical metadata {key}")
        if meta["storage_type"] not in PHYSICAL or not np.isclose(float(meta["storage_value"]), PHYSICAL[meta["storage_type"]], rtol=0, atol=1e-12):
            raise ValueError("Storage type/value must preserve D03 physical strata")
        for key in ("T_m2_d", "r_m", "Q_m3_d", "t95_days", "rest_ratio", "signal_ratio", "pi_layer"):
            if isinstance(meta[key], bool) or not np.isfinite(float(meta[key])) or float(meta[key]) <= 0:
                raise ValueError(f"Positive finite physical/derived label required: {key}")
        if meta["rest_ratio"] not in (.25, 1, 4) or meta["transition_count"] not in (2, 6, 18):
            raise ValueError("D03 rest/transition grid required")
        if isinstance(meta["realization"], bool) or meta["realization"] not in range(10):
            raise ValueError("Realization must be integer 0..9")
        if (meta["head_unit"], meta["pumping_unit"], meta["rainfall_unit"]) != ("m", "m3/d", "mm/d") or meta["daily_alignment"] is not True:
            raise ValueError("Daily metre/m3-per-day/mm-per-day interface required")
        cases.setdefault(ids["case_id"][i], []).append(i)
    for cid, ix in cases.items():
        if len(ix) != 4 or {metas[i]["schedule_pair"] for i in ix} != set(PAIRS):
            raise ValueError(f"{cid}: two contrasts/four queries required")
        shared = [{k: v for k, v in metas[i].items() if k != "schedule_pair"} for i in ix]
        if any(m != shared[0] for m in shared[1:]):
            raise ValueError(f"{cid}: physical labels/weather/noise metadata must match across contrasts")
        for key in ("head", "rainfall"):
            if any(not np.array_equal(data[key][i], data[key][ix[0]]) for i in ix[1:]):
                raise ValueError(f"{cid}: {key} differs across contrasts")
        if any(not np.array_equal(data["pumping"][i,:CONTEXT],data["pumping"][ix[0],:CONTEXT]) for i in ix[1:]):
            raise ValueError(f"{cid}: pumping history differs across contrasts")
        a_futures = []
        for pair in PAIRS:
            rows = [i for i in ix if metas[i]["schedule_pair"] == pair]
            if len({ids["pair_id"][i] for i in rows}) != 1:
                raise ValueError("Pair identity inconsistent with schedule pair")
            ia = next(i for i in rows if ids["schedule_id"][i] == "a")
            ib = next(i for i in rows if ids["schedule_id"][i] == "b")
            qa = data["pumping"][ia,CONTEXT:]; qb = data["pumping"][ib,CONTEXT:]
            if not np.all(qa == qa[0]) or qa[0] <= 0:
                raise ValueError("Future a must continue a fixed positive Q")
            want = np.zeros_like(qa) if pair == PAIRS[0] else 1.5 * qa
            if not np.array_equal(qb, want):
                raise ValueError("Future pairs must be continue/stop and current/1.5x")
            a_futures.append(qa)
        if not np.array_equal(*a_futures):
            raise ValueError("Future current Q differs across contrasts")
    if require_full_grid:
        if len(cases) != 540 or len(order) != 2160:
            raise ValueError("Official full grid requires 540 cases/1080 contrasts/2160 queries")
        cell_meta = [metas[ix[0]] for ix in cases.values()]
        tvals = {float(m["T_m2_d"]) for m in cell_meta}
        if len(tvals) != 2 or len({float(m["r_m"]) for m in cell_meta}) != 1 or len({float(m["Q_m3_d"]) for m in cell_meta}) != 1:
            raise ValueError("Two T levels and common declared r/Q across layers required")
        cells = {(m["storage_type"],float(m["T_m2_d"]),float(m["rest_ratio"]),m["transition_count"],m["realization"]) for m in cell_meta}
        expected = {(s,t,r,n,i) for s in PHYSICAL for t in tvals for r in (.25,1.,4.) for n in (2,6,18) for i in range(10)}
        if cells != expected:
            raise ValueError("Full physical/rest/transition/realization Cartesian grid required")
    return order, ids


def prepare_track(data, horizon, track, require_full_grid=False):
    if track not in TRACKS:
        raise ValueError("All tracks are prespecified; no best-filter selection")
    order, ids = validate(data, horizon, require_full_grid)
    q = np.asarray(data["pumping"])[order]
    pumping = q if TRACKS[track] is None else exp_filter(q, TRACKS[track])
    # Only these arrays enter predict_batch; physical labels remain output metadata.
    return dict(contexts=np.asarray(data["head"])[order].astype(np.float32),
                covariates=np.stack([pumping, np.asarray(data["rainfall"])[order]], axis=1).astype(np.float32),
                query_id=ids["query_id"][order], pair_id=ids["pair_id"][order][::2],
                schedule_id=ids["schedule_id"][order], case_id=ids["case_id"][order],
                metadata_json=ids["metadata_json"][order], track=track, horizon=horizon,
                api=dict(API), checkpoint_revision=SNAPSHOT.name,
                filter_initialization="z[0]=observed Q[0]; no unseen prehistory")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases",type=Path,required=True)
    ap.add_argument("--horizon",type=int,choices=[10,30],required=True)
    ap.add_argument("--track",choices=list(TRACKS),required=True)
    ap.add_argument("--require-full-grid",action="store_true")
    args=ap.parse_args()
    with np.load(args.cases,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    out=prepare_track(d,args.horizon,args.track,args.require_full_grid)
    print(json.dumps(dict(status="validated_only",official_inference=False,track=args.track,horizon=args.horizon,
                         queries=len(out["query_id"]),contrasts=len(out["pair_id"]),cases=len(set(out["case_id"])),
                         context=CONTEXT,covariates_shape=list(out["covariates"].shape),api=API)))


if __name__=="__main__":main()
