"""FIXTURE smoke (B): the frozen v1.2 W engine (uniform enlarged budget) and the frozen Pastas reference load a
Phase 2 tf_input written by lib/p3_phase2_make_cases.py --fixture. Not official, not a score.
Writes results/phase2/B_merge/engine_smoke.json"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_wenvelope_repaired_v1_2 as w
import p3_pastas as ref
from p3_make_cases import load_tf_input
BUDGET = dict(global_levels=(128, 256, 512, 1024), n_local=512, max_local_rounds=12)
D = Path(sys.argv[1]); out = {}
for cid in ("r07_unconfined_T50_rho4_N18", "r04_confined_T500_rho0.25_N2"):
    ti = load_tf_input(D / "tf_inputs" / f"{cid}.npz")
    t0 = time.time(); rec = w.WEngineV12(ti, seed=20260930).run(**BUDGET)
    tW = time.time() - t0; t0 = time.time()
    fit = ref.fit_and_propagate(ti, seed=20261734, n_draws=200); fit.pop("_model", None)
    out[cid] = dict(W_status=rec.get("W_status"), flags=rec.get("flags"), W_runtime_s=tW, n_accepted=rec.get("n_accepted_total"),
                    local_rounds=len(rec.get("local", [])), witness_pass=rec.get("witness_check", {}).get("pass"),
                    ref_runtime_s=time.time() - t0, ref_draws=fit["n_param_draws_accepted"], ref_fit_rmse_m=fit["fit_rmse_m"],
                    ref_has_both_pairs=sorted(fit["pairs"]))
    print(cid, json.dumps(out[cid], default=str), flush=True)
(ROOT / "results/phase2/B_merge/engine_smoke.json").write_text(json.dumps(dict(note="FIXTURE smoke; not official", budget=BUDGET, rows=out), indent=1, default=str))
