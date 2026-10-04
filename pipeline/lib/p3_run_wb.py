"""D02 pilot runner: W envelope and structurally matched reference on the frozen cases (B).

Frozen method code is imported unchanged (p3_wenvelope, p3_pastas, p3_make_cases loader); this file only
orchestrates, saves and verifies. Truth is loaded ONLY in the separate evaluation step after W and the
reference are saved (truth_eval/), never passed to WEngine acceptance or the reference fit.

Usage: OMP_NUM_THREADS=1 ... python lib/p3_run_wb.py --shard 0 --nshards 2
Outputs under results/pilot/wb/: W/{case}.json, reference/{case}.json + reference/{case}_draws.npz,
truth_eval/{case}.json, markers/{case}.done (json with output hashes), logs/shard{k}.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_pastas as pt  # noqa: E402
import p3_wenvelope as w  # noqa: E402
from p3_make_cases import load_tf_input  # noqa: E402

PILOT = ROOT / "results/pilot"
CASES = PILOT / "cases"
OUT = PILOT / "wb"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
W_SEED = 20260930  # WEngine default; identical fixed Sobol sequences for every case (protocol §8)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def frozen_digest():
    d = (PILOT / "protocol.sha256").read_text().split()[0]
    if sha(PILOT / "protocol.md") != d:
        raise PermissionError("protocol.md does not match protocol.sha256")
    return d


def run_case(case_id, digest, seeds, log):
    f = CASES / "tf_inputs" / f"{case_id}.npz"
    ti = load_tf_input(f)
    if ti["protocol_sha256"] != digest:
        raise ValueError("case protocol hash mismatch")
    real = int(ti["meta"]["realization"])
    res = {"case_id": case_id, "tf_input_sha256": sha(f), "protocol_sha256": digest, "meta": ti["meta"]}
    checks = {}
    # ---------------- W (observed record only)
    t0 = time.time()
    try:
        eng = w.WEngine(ti, seed=W_SEED)
        rec = eng.run()
        rec["wall_s"] = time.time() - t0
        rec["seed"] = W_SEED
        env = rec.get("envelope")
        if env:
            Wm = {p: np.array(env[p]["W"]) for p in PAIRS}
            checks["W_finite"] = bool(all(np.all(np.isfinite(Wm[p])) and np.all(Wm[p] >= -1e-15) for p in PAIRS))
            checks["W_pair_identity_maxabs_m"] = float(max(
                np.max(np.abs(Wm[PAIRS[1]] - 0.5 * Wm[PAIRS[0]])),
                np.max(np.abs(np.array(env[PAIRS[1]]["sup"]) + 0.5 * np.array(env[PAIRS[0]]["inf"]))),
                np.max(np.abs(np.array(env[PAIRS[1]]["inf"]) + 0.5 * np.array(env[PAIRS[0]]["sup"])))))
        (OUT / "W" / f"{case_id}.json").write_text(json.dumps(jsonable(dict(res, **rec)), ensure_ascii=False))
        res["W_status"] = "ok" if env else "W_failed"
    except Exception:
        eng = None
        res["W_status"] = "error"
        (OUT / "W" / f"{case_id}.json").write_text(json.dumps(dict(res, error=traceback.format_exc(), flags=["W_failed"])))
        log(f"{case_id} W ERROR\n{traceback.format_exc()}")
    # ---------------- reference (observed record only)
    t1 = time.time()
    seed = int(seeds[real])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pt.fit_and_propagate(ti, seed=seed)
            cons = pt.simulate_consistency(ti, fit)
        ml = fit.pop("_model")
        names = fit["param_names"]
        iw = [names.index(k) for k in ("well_A", "well_a", "well_b")]
        # reproduce the identical joint draws (same call, same seed) to store the paired E samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            draws = ml.solver.get_parameter_sample(name=None, n=1000, max_iter=50, seed=seed)
        Es = {}
        for p in PAIRS:
            dQ = np.asarray(ti["future_Q"][f"{p}_a"], float) - np.asarray(ti["future_Q"][f"{p}_b"], float)
            Es[p] = np.array([pt.unit_E(d[iw], dQ) for d in draws]) if draws.shape[0] else np.empty((0, 30))
        rep = {}
        for p in PAIRS:
            if Es[p].shape[0] and "E_q10" in fit["pairs"][p]:
                q10, q90 = np.quantile(Es[p], [0.1, 0.9], axis=0)
                rep[p] = float(max(np.max(np.abs(q10 - fit["pairs"][p]["E_q10"])), np.max(np.abs(q90 - fit["pairs"][p]["E_q90"]))))
        checks["reference_draws_reproduced_maxabs_m"] = max(rep.values()) if rep else None
        checks["reference_simulate_consistency_m"] = cons
        e1, e2 = np.array(fit["pairs"][PAIRS[0]]["E_point"]), np.array(fit["pairs"][PAIRS[1]]["E_point"])
        checks["reference_pair_identity_maxabs_m"] = float(np.max(np.abs(e2 + 0.5 * e1)))
        checks["reference_finite"] = bool(np.all(np.isfinite(e1)) and np.all(np.isfinite(e2)))
        try:
            pcov = ml.solver.pcov.values.tolist()
        except Exception:
            pcov = None
        fit.update(pcov_full=pcov, seed=seed, n_draws_requested=1000, wall_s=time.time() - t1,
                   solver_success=bool(getattr(ml.solver, "result", None) is None or getattr(ml.solver.result, "success", True)))
        np.savez_compressed(OUT / "reference" / f"{case_id}_draws.npz", draws=draws, param_names=np.array(names, dtype=str),
                            **{f"E_draws__{p}": Es[p] for p in PAIRS})
        (OUT / "reference" / f"{case_id}.json").write_text(json.dumps(jsonable(dict(res, **fit)), ensure_ascii=False))
        res["reference_status"] = "ok"
    except Exception:
        res["reference_status"] = "error"
        (OUT / "reference" / f"{case_id}.json").write_text(json.dumps(dict(res, error=traceback.format_exc())))
        log(f"{case_id} REFERENCE ERROR\n{traceback.format_exc()}")
    # ---------------- evaluation only (after both outputs are saved)
    try:
        with np.load(CASES / "truth" / f"{case_id}.npz", allow_pickle=False) as z:
            tr = {k: z[k] for k in z.files}
        theta = json.loads(str(tr["theta_true_json"]))
        E_true = {p: tr["E_true__" + p] for p in PAIRS}
        ev = {"case_id": case_id, "note": "evaluation only; not used by W acceptance or reference fit"}
        if eng is not None and res["W_status"] == "ok":
            recW = json.loads((OUT / "W" / f"{case_id}.json").read_text())
            tc = w.truth_compatibility(eng, recW, theta, E_true)
            ev["truth_compatibility"] = jsonable(tc)
        ev["E_true"] = {p: E_true[p].tolist() for p in PAIRS}
        ev["delta"] = {p: tr["delta__" + p].tolist() for p in PAIRS}
        (OUT / "truth_eval" / f"{case_id}.json").write_text(json.dumps(jsonable(ev)))
    except Exception:
        log(f"{case_id} TRUTH-EVAL ERROR\n{traceback.format_exc()}")
    res["checks"] = checks
    outs = [OUT / "W" / f"{case_id}.json", OUT / "reference" / f"{case_id}.json", OUT / "truth_eval" / f"{case_id}.json"]
    res["outputs"] = {str(p.relative_to(PILOT)): (sha(p) if p.exists() else None) for p in outs}
    res["total_wall_s"] = time.time() - t0
    (OUT / "markers" / f"{case_id}.done").write_text(json.dumps(jsonable(res), ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for d in ("W", "reference", "truth_eval", "markers", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    digest = frozen_digest()
    man = json.load(open(CASES / "observed_manifest.json"))
    cal = json.load(open(PILOT / "phase0_checks/realization_calendar.json"))
    seeds = {int(r["realization"]): int(r["seed_pastas_param_sample"]) for r in cal}
    ids = [m["case_id"] for m in man]
    if a.only:
        ids = [i for i in ids if i in set(a.only)]
    ids = ids[a.shard::a.nshards]
    logf = open(OUT / "logs" / f"shard{a.shard}.log", "a")

    def log(msg):
        logf.write(time.strftime("%H:%M:%S ") + msg + "\n"); logf.flush()
    log(f"start shard {a.shard}/{a.nshards} n={len(ids)} protocol={digest[:16]}")
    for i, cid in enumerate(ids):
        if (OUT / "markers" / f"{cid}.done").exists():
            continue
        r = run_case(cid, digest, seeds, log)
        log(f"{i + 1}/{len(ids)} {cid} W={r.get('W_status')} ref={r.get('reference_status')} {r['total_wall_s']:.1f}s")
    log("shard done")


if __name__ == "__main__":
    main()
