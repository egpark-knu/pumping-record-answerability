"""Diagnostic driver for CANDIDATE v1.1 W engine (not frozen). Writes only results/pilot/wb_candidate_v1_1/."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))); sys.path.insert(0, str(ROOT / "lib"))
import p3_wenvelope_v1_1 as w
from p3_make_cases import load_tf_input
from p3_run_wb import jsonable, PAIRS
P = ROOT / "results/pilot"; OUT = P / "wb_candidate_v1_1"; (OUT / "W").mkdir(parents=True, exist_ok=True)
for cid in sys.argv[1:]:
    if (OUT / "W" / f"{cid}.json").exists():
        continue
    t0 = time.time(); ti = load_tf_input(P / "cases/tf_inputs" / f"{cid}.npz")
    eng = w.WEngine(ti, seed=20260930); rec = eng.run(); rec["wall_s"] = time.time() - t0
    with np.load(P / "cases/truth" / f"{cid}.npz", allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    if rec.get("envelope"):
        rec["truth_eval"] = jsonable(w.truth_compatibility(eng, rec, json.loads(str(tr["theta_true_json"])), {p: tr["E_true__" + p] for p in PAIRS}))
    (OUT / "W" / f"{cid}.json").write_text(json.dumps(jsonable(dict(case_id=cid, candidate="v1.1", **rec))))
    print(cid, rec["flags"], rec.get("converged"), rec.get("summary", {}).get("P1_continue_vs_stop", {}).get("W10"), round(rec["wall_s"], 1), flush=True)
