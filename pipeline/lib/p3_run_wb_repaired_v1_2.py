"""D02 W repair v1.2 runner (B). Recomputes W ONLY, with lib/p3_wenvelope_repaired_v1_2.py, on the frozen cases.

- The original run (results/pilot/wb/) is read-only input: per case, the cached best fit / frozen tau is reused only
  if the case input hash, protocol hash and W seed match the original record; the repaired engine then reproduces
  the best-fit stage and records bit-identity, and the direct SSE at the cached centre must be <= the frozen tau
  (otherwise status cached_center_invalid; tau is never recalibrated).
- Reference fits/draws and TimesFM archives are NOT recomputed (inventoried in reuse_manifest.json).
- Truth is loaded only in the separate evaluation step after the W record is saved.
- --official requires results/pilot/repair_freeze.json whose hashes match the addendum, engine and this runner.

Usage: OMP_NUM_THREADS=1 ... python lib/p3_run_wb_repaired_v1_2.py --out results/pilot/wb_repaired_v1_2 --official --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_wenvelope as w0  # noqa: E402
import p3_wenvelope_repaired_v1_2 as w  # noqa: E402
from p3_make_cases import load_tf_input  # noqa: E402

PILOT = ROOT / "results/pilot"
CASES = PILOT / "cases"
OLD = PILOT / "wb"
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
W_SEED = 20260930


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


def check_freeze():
    fz = json.loads((PILOT / "repair_freeze.json").read_text())
    for rel, h in fz["hashes"].items():
        if sha(ROOT / rel) != h:
            raise PermissionError(f"repair freeze mismatch: {rel}")
    return fz


def truth_eval(eng, rec, case_id):
    with np.load(CASES / "truth" / f"{case_id}.npz", allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    theta = json.loads(str(tr["theta_true_json"]))
    E_true = {p: tr["E_true__" + p] for p in PAIRS}
    ev = {"case_id": case_id, "version": w.VERSION, "note": "evaluation only; not used by W acceptance. truth_nonlinear_in_G profiles the "
          "linear coefficients (compatibility), full_vector_in_G uses the true A, eta_pump, beta_nat, eta_nat with only c0 profiled."}
    tc = w0.truth_compatibility(eng, rec, theta, E_true)
    x = np.array([theta["n"], np.log10(theta["theta"]), np.log10(theta["tau"]), np.log10(theta["a"]), np.log10(theta["b"])])
    _, _, X, _, t95 = eng.point(x)
    tau = rec["tolerance"]["tau"]
    lin = np.array([theta["A"], theta["eta_pump_true"], 0.0, theta["nat_gain"], theta["eta_nat_true"]])
    r0 = eng.yc - X @ lin
    c0 = float(r0.mean()); sse_full = float(np.sum((r0 - c0) ** 2))
    dp = eng.direct_point_interval(x, tau)
    tc.update(full_vector_sse_c0_profiled=sse_full, truth_full_vector_in_G=bool(sse_full <= tau and t95 <= w.T95_MAX),
              A_true_in_direct_profile_interval=bool(dp["iv"].get("ok") and dp["iv"]["lo"] <= theta["A"] <= dp["iv"]["hi"]))
    ev["truth_compatibility"] = jsonable(tc)
    ev["E_true"] = {p: E_true[p].tolist() for p in PAIRS}
    ev["delta"] = {p: tr["delta__" + p].tolist() for p in PAIRS}
    return ev


def run_case(case_id, digest, out, log):
    f = CASES / "tf_inputs" / f"{case_id}.npz"
    ti = load_tf_input(f)
    old = json.loads((OLD / "W" / f"{case_id}.json").read_text())
    oldm = json.loads((OLD / "markers" / f"{case_id}.done").read_text())
    h = sha(f)
    inp = dict(tf_input_sha256=h, old_record_tf_input_sha256=old.get("tf_input_sha256"), old_marker_tf_input_sha256=oldm.get("tf_input_sha256"),
               old_record_protocol=old.get("protocol_sha256"), case_protocol=ti["protocol_sha256"], old_seed=old.get("seed"), W_seed=W_SEED,
               old_record_sha256=sha(OLD / "W" / f"{case_id}.json"), old_record_marker_hash=oldm["outputs"].get(f"wb/W/{case_id}.json"))
    inp["valid"] = bool(h == inp["old_record_tf_input_sha256"] == inp["old_marker_tf_input_sha256"] and ti["protocol_sha256"] == digest
                        == inp["old_record_protocol"] and inp["old_seed"] == W_SEED and inp["old_record_sha256"] == inp["old_record_marker_hash"]
                        and "tolerance" in old)
    res = {"case_id": case_id, "version": w.VERSION, "tf_input_sha256": h, "protocol_sha256": digest, "meta": ti["meta"], "input_identity": inp}
    t0 = time.time(); eng = None
    try:
        if not inp["valid"]:
            raise ValueError("cached-fit reuse rejected: input/protocol/seed/record identity mismatch")
        t = old["tolerance"]
        cached = dict(tau=t["tau"], sse_min=t["sse_min"], xbest=t["xbest"], tolerance_record=t)
        eng = w.WEngineV12(ti, seed=W_SEED)
        rec = eng.run(cached=cached)
        rec["wall_s"] = time.time() - t0; rec["seed"] = W_SEED
        env = rec.get("envelope")
        checks = {}
        if env:
            Wm = {p: np.array(env[p]["W"]) for p in PAIRS}
            checks["W_finite"] = bool(all(np.all(np.isfinite(Wm[p])) and np.all(Wm[p] >= -1e-15) for p in PAIRS))
            checks["W_pair_identity_maxabs_m"] = float(max(
                np.max(np.abs(Wm[PAIRS[1]] - 0.5 * Wm[PAIRS[0]])),
                np.max(np.abs(np.array(env[PAIRS[1]]["sup"]) + 0.5 * np.array(env[PAIRS[0]]["inf"]))),
                np.max(np.abs(np.array(env[PAIRS[1]]["inf"]) + 0.5 * np.array(env[PAIRS[0]]["sup"])))))
        rec["checks"] = checks
        (out / "W" / f"{case_id}.json").write_text(json.dumps(jsonable(dict(res, **rec)), ensure_ascii=False))
        res["W_status"] = rec.get("W_status", "error"); res["flags"] = rec.get("flags", []); res["checks"] = checks
    except Exception:
        res["W_status"] = "error"
        (out / "W" / f"{case_id}.json").write_text(json.dumps(jsonable(dict(res, error=traceback.format_exc(), flags=["W_error"]))))
        log(f"{case_id} W ERROR\n{traceback.format_exc()}")
    try:
        if eng is not None and res["W_status"] == "ok":
            recW = json.loads((out / "W" / f"{case_id}.json").read_text())
            (out / "truth_eval" / f"{case_id}.json").write_text(json.dumps(jsonable(truth_eval(eng, recW, case_id))))
    except Exception:
        log(f"{case_id} TRUTH-EVAL ERROR\n{traceback.format_exc()}")
    outs = [out / "W" / f"{case_id}.json", out / "truth_eval" / f"{case_id}.json"]
    res["outputs"] = {str(p.relative_to(PILOT)): (sha(p) if p.exists() else None) for p in outs}
    res["total_wall_s"] = time.time() - t0
    (out / "markers" / f"{case_id}.done").write_text(json.dumps(jsonable(res), ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    if a.official:
        if out.resolve() != (PILOT / "wb_repaired_v1_2").resolve():
            raise ValueError("official output directory must be results/pilot/wb_repaired_v1_2")
        check_freeze()
    elif out.resolve() == (PILOT / "wb_repaired_v1_2").resolve():
        raise ValueError("non-official runs may not write into the official directory")
    for d in ("W", "truth_eval", "markers", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    d0 = (PILOT / "protocol.sha256").read_text().split()[0]
    if sha(PILOT / "protocol.md") != d0:
        raise PermissionError("protocol.md does not match protocol.sha256")
    man = json.load(open(CASES / "observed_manifest.json"))
    ids = [m["case_id"] for m in man]
    if a.only:
        ids = [i for i in ids if i in set(a.only)]
    ids = ids[a.shard::a.nshards]
    logf = open(out / "logs" / f"shard{a.shard}.log", "a")

    def log(msg):
        logf.write(time.strftime("%H:%M:%S ") + msg + "\n"); logf.flush()
    log(f"start shard {a.shard}/{a.nshards} n={len(ids)} official={a.official} version={w.VERSION} protocol={d0[:16]}")
    for i, cid in enumerate(ids):
        if (out / "markers" / f"{cid}.done").exists():
            continue
        r = run_case(cid, d0, out, log)
        log(f"{i + 1}/{len(ids)} {cid} W={r.get('W_status')} flags={','.join(r.get('flags', []))} {r['total_wall_s']:.1f}s")
    log("shard done")


if __name__ == "__main__":
    main()
