"""D05 Phase 4 execution driver (unchanged D03/D04 algorithms): W envelope (frozen v1.2 engine), structurally matched reference (frozen
p3_pastas.fit_and_propagate) and a SEPARATE truth evaluation for the official D05 cases in results/phase4/cases.

Frozen presets (results/phase2/protocol.md section 6 and 8; not changeable here):
  W   : WEngineV12(tf_input, seed=20260930).run(global_levels=(128,256,512,1024), n_local=512, max_local_rounds=12)
        (5% / >=200 accepted / 4 restarts are module constants of the frozen engine)
  ref : fit_and_propagate(tf_input, seed=calendar seed_pastas_param_sample)  (defaults n_draws=1000, max_iter=50, 12 starts)
        + simulate_consistency; the same joint draws are regenerated (same call, same seed) to store paired E samples
Truth is read only after the W record and the reference record are written; nothing from truth reaches the engines.
Resume: a case is skipped only if its marker has the same tf_input / protocol / driver hashes and all outputs exist
with the recorded hashes; otherwise it is recomputed.

D05 actor shards: --actor shard_a|shard_b restricts the run to results/phase4/SHARD_<ACTOR>.json (1440 disjoint case ids);
--shard/--nshards then split that actor's list over independent single-thread processes. Per-case outputs are keyed by
case_id and logs by actor/process, so the two actors never write the same file. Every one of the 2880 cases (including
the 918 zero-origin B cases) needs W, reference and truth evaluation; a reference error stays an explicit error status.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
       env/.venv_pilot/bin/python lib/p4_run_wb.py --actor shard_b --shard 0 --nshards 4
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
import p3_wenvelope as w0  # noqa: E402  (frozen)
import p3_wenvelope_repaired_v1_2 as w  # noqa: E402  (frozen)
import p3_pastas as pt  # noqa: E402  (frozen)
from p3_make_cases import load_tf_input  # noqa: E402  (frozen)
from p3_wb_collect_repaired_v1_2 import TOL_IN  # noqa: E402  (frozen, 1e-12 m)

P2 = ROOT / "results/phase4"
CASES = P2 / "cases"
OUT = P2 / "wb"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
W_SEED = 20260930
BUDGET = dict(global_levels=(128, 256, 512, 1024), n_local=512, max_local_rounds=12)
N_DRAWS = 1000
DRIVER_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def check_protocol():
    from p4_contracts import require_frozen
    return require_frozen()

def run_W(ti):
    eng = w.WEngineV12(ti, seed=W_SEED)
    rec = eng.run(**BUDGET)
    env = rec.get("envelope")
    checks = {}
    if env:
        Wm = {p: np.array(env[p]["W"]) for p in PAIRS}
        checks["W_finite_nonneg"] = bool(all(np.all(np.isfinite(Wm[p])) and np.all(Wm[p] >= -1e-15) for p in PAIRS))
        checks["W_pair_identity_maxabs_m"] = float(max(
            np.max(np.abs(Wm[PAIRS[1]] - 0.5 * Wm[PAIRS[0]])),
            np.max(np.abs(np.array(env[PAIRS[1]]["sup"]) + 0.5 * np.array(env[PAIRS[0]]["inf"]))),
            np.max(np.abs(np.array(env[PAIRS[1]]["inf"]) + 0.5 * np.array(env[PAIRS[0]]["sup"])))))
    rec["checks"] = checks
    rec["seed"] = W_SEED
    rec["budget"] = BUDGET
    return eng, rec


def run_reference(ti, seed):
    checks = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pt.fit_and_propagate(ti, seed=seed, n_draws=N_DRAWS)
        cons = pt.simulate_consistency(ti, fit)
    ml = fit.pop("_model")
    names = fit["param_names"]
    iw = [names.index(k) for k in ("well_A", "well_a", "well_b")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = ml.solver.get_parameter_sample(name=None, n=N_DRAWS, max_iter=50, seed=seed)
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
    fit.update(pcov_full=pcov, seed=seed, n_draws_requested=N_DRAWS, checks=checks,
               solver_success=bool(getattr(ml.solver, "result", None) is None or getattr(ml.solver.result, "success", True)))
    return fit, draws, names, Es


def truth_eval(eng, recW, case_id):
    with np.load(CASES / "truth" / f"{case_id}.npz", allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    theta = json.loads(str(tr["theta_true_json"]))
    E_true = {p: tr["E_true__" + p] for p in PAIRS}
    ev = {"case_id": case_id, "note": "evaluation only; read after W and reference records were written; never used by either"}
    if eng is not None and recW.get("W_status") == "ok":
        tc = w0.truth_compatibility(eng, recW, theta, E_true)
        x = np.array([theta["n"], np.log10(theta["theta"]), np.log10(theta["tau"]), np.log10(theta["a"]), np.log10(theta["b"])])
        _, _, X, _, t95 = eng.point(x)
        tau = recW["tolerance"]["tau"]
        lin = np.array([theta["A"], theta["eta_pump_true"], 0.0, theta["nat_gain"], theta["eta_nat_true"]])
        r0 = eng.yc - X @ lin
        c0 = float(r0.mean()); sse_full = float(np.sum((r0 - c0) ** 2))
        dp = eng.direct_point_interval(x, tau)
        tc.update(full_vector_sse_c0_profiled=sse_full, truth_full_vector_in_G=bool(sse_full <= tau and t95 <= w.T95_MAX),
                  A_true_in_direct_profile_interval=bool(dp["iv"].get("ok") and dp["iv"]["lo"] <= theta["A"] <= dp["iv"]["hi"]))
        env = recW["envelope"]
        tc["E_true_inside_TOL_IN"] = {p: [bool(env[p]["inf"][k] - TOL_IN <= E_true[p][k] <= env[p]["sup"][k] + TOL_IN) for k in range(30)] for p in PAIRS}
        ev["truth_compatibility"] = jsonable(tc)
    ev["E_true"] = {p: E_true[p].tolist() for p in PAIRS}
    ev["delta"] = {p: tr["delta__" + p].tolist() for p in PAIRS}
    ev["theta_true"] = theta
    return ev


def fresh(case_id, tf_sha, digest):
    m = OUT / "markers" / f"{case_id}.done"
    if not m.exists():
        return False
    r = json.loads(m.read_text())
    if r.get("tf_input_sha256") != tf_sha or r.get("protocol_sha256") != digest or r.get("driver_sha256") != DRIVER_SHA:
        return False
    if r.get("W_status") in (None, "error") or r.get("reference_status") != "ok" or not r.get("truth_eval_ok"):
        return False
    return all((ROOT / p).exists() and sha(ROOT / p) == h for p, h in r["outputs"].items())


def run_case(case_id, digest, seed, log):
    f = CASES / "tf_inputs" / f"{case_id}.npz"
    ti = load_tf_input(f)
    tf_sha = sha(f)
    if ti["protocol_sha256"] != digest:
        raise PermissionError(f"{case_id}: case protocol stamp differs from frozen protocol")
    res = dict(case_id=case_id, tf_input_sha256=tf_sha, protocol_sha256=digest, driver_sha256=DRIVER_SHA, meta=ti["meta"], W_seed=W_SEED, ref_seed=seed)
    t0 = time.time(); eng = None
    pW, pR, pD, pT = (OUT / "W" / f"{case_id}.json", OUT / "reference" / f"{case_id}.json", OUT / "reference" / f"{case_id}_draws.npz",
                      OUT / "truth_eval" / f"{case_id}.json")
    try:
        eng, rec = run_W(ti)
        rec["wall_s"] = time.time() - t0
        pW.write_text(json.dumps(jsonable(dict(res, **rec)), ensure_ascii=False))
        res.update(W_status=rec.get("W_status", "error"), flags=rec.get("flags", []), W_checks=rec["checks"], W_wall_s=rec["wall_s"])
    except Exception:
        res.update(W_status="error", flags=["W_error"])
        pW.write_text(json.dumps(jsonable(dict(res, error=traceback.format_exc()))))
        log(f"{case_id} W ERROR\n{traceback.format_exc()}")
    t1 = time.time()
    try:
        fit, draws, names, Es = run_reference(ti, seed)
        fit["wall_s"] = time.time() - t1
        np.savez_compressed(pD, draws=draws, param_names=np.array(names, dtype=str), **{f"E_draws__{p}": Es[p] for p in PAIRS})
        pR.write_text(json.dumps(jsonable(dict(res, **fit)), ensure_ascii=False))
        res.update(reference_status="ok", reference_checks=fit["checks"], reference_draws=fit["n_param_draws_accepted"], ref_wall_s=fit["wall_s"])
    except Exception:
        res["reference_status"] = "error"
        pR.write_text(json.dumps(jsonable(dict(res, error=traceback.format_exc()))))
        log(f"{case_id} REFERENCE ERROR\n{traceback.format_exc()}")
    try:
        recW = json.loads(pW.read_text())
        pT.write_text(json.dumps(jsonable(truth_eval(eng, recW, case_id))))
        res["truth_eval_ok"] = True
    except Exception:
        res["truth_eval_ok"] = False
        log(f"{case_id} TRUTH-EVAL ERROR\n{traceback.format_exc()}")
    res["outputs"] = {str(p.relative_to(ROOT)): sha(p) for p in (pW, pR, pD, pT) if p.exists()}
    res["total_wall_s"] = time.time() - t0
    (OUT / "markers" / f"{case_id}.done").write_text(json.dumps(jsonable(res), ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", choices=("shard_a", "shard_b"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("shard index outside nshards")
    digest = check_protocol()
    for d in ("W", "reference", "truth_eval", "markers", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    man = json.load(open(CASES / "derived_manifest.json"))
    cal = {int(r["realization"]): int(r["seed_pastas_param_sample"]) for r in json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))}
    ids = [(m["case_id"], cal[m["realization"]], m["tf_sha256"]) for m in man]
    if a.actor:
        import p4_partition as part
        allowed = part.load(a.actor)
        if set(allowed) - {x[0] for x in ids}:
            raise SystemExit("actor shard contains ids absent from the derived manifest")
        keep = set(allowed)
        ids = [x for x in ids if x[0] in keep]
    if a.only:
        ids = [x for x in ids if x[0] in set(a.only)]
    ids = ids[a.shard::a.nshards]
    tag = f"{a.actor}_proc{a.shard}" if a.actor else f"shard{a.shard}"
    logf = open(OUT / "logs" / f"{tag}.log", "a")

    def log(msg):
        logf.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n"); logf.flush()
    log(f"start {tag} {a.shard}/{a.nshards} n={len(ids)} protocol={digest[:16]} driver={DRIVER_SHA[:16]}")
    for i, (cid, seed, tfs) in enumerate(ids):
        if fresh(cid, tfs, digest):
            continue
        r = run_case(cid, digest, seed, log)
        log(f"{i + 1}/{len(ids)} {cid} W={r.get('W_status')} flags={','.join(r.get('flags', []))} ref={r.get('reference_status')} "
            f"truth={r.get('truth_eval_ok')} {r['total_wall_s']:.1f}s")
    log(f"{tag} done")


if __name__ == "__main__":
    main()
