"""DIAGNOSTIC (not frozen, not a W result): does per-lead refinement change W(k) at leads the frozen rule
leaves unrefined? Uses the frozen engine's refine() unchanged. Start: union of frozen and candidate v1.1
envelopes (same tau, same best fit). For k = 10, 9, ..., 1 and 11..30: Powell from the previous lead's optima
(chain), first lead from the best fit; restart until the increment <= 5% (max 4). Writes
results/pilot/wb_candidate_v1_1/lead_refine/{case}.json."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))); sys.path.insert(0, str(ROOT / "lib"))
import p3_wenvelope as w
from p3_make_cases import load_tf_input
P = ROOT / "results/pilot"; OUT = P / "wb_candidate_v1_1/lead_refine"; OUT.mkdir(parents=True, exist_ok=True)
P1 = "P1_continue_vs_stop"
for cid in sys.argv[1:]:
    t0 = time.time()
    ti = load_tf_input(P / "cases/tf_inputs" / f"{cid}.npz"); eng = w.WEngine(ti, seed=20260930)
    c = json.loads((P / "wb_candidate_v1_1/W" / f"{cid}.json").read_text()); fz = json.loads((P / "wb/W" / f"{cid}.json").read_text())
    tau = c["tolerance"]["tau"]; xb = np.array(c["tolerance"]["xbest"])
    sup = np.array(c["envelope"][P1]["sup"]); inf = np.array(c["envelope"][P1]["inf"])
    if fz.get("envelope"):
        sup = np.maximum(sup, fz["envelope"][P1]["sup"]); inf = np.minimum(inf, fz["envelope"][P1]["inf"])
    U0 = sup - inf
    xs = {"sup": xb, "inf": xb}; incs = {}
    for k in list(range(10, 0, -1)) + list(range(11, 31)):
        if k == 11:
            xs = dict(xs10)
        seq = []
        for it in range(5):
            prevW = sup[k - 1] - inf[k - 1]
            ref = eng.refine(dict(sup=[sup[k - 1]] * 30, inf=[inf[k - 1]] * 30), tau, P1, k, starts=dict(sup=xs["sup"], inf=xs["inf"]))
            sup[k - 1] = max(sup[k - 1], ref["sup"]); inf[k - 1] = min(inf[k - 1], ref["inf"])
            xs = {"sup": ref["xsup"], "inf": ref["xinf"]}
            inc = (sup[k - 1] - inf[k - 1] - prevW) / max(prevW, 1e-12); seq.append(float(inc))
            if inc <= 0.05 and it >= 0:
                break
        incs[k] = seq
        if k == 10:
            xs10 = dict(xs)
    Wr = sup - inf
    out = dict(case_id=cid, tau=tau, union_W=U0.tolist(), refined_W=Wr.tolist(), rel_gain_by_lead=((Wr - U0) / np.maximum(U0, 1e-12)).tolist(),
               increments=incs, W10_union=float(U0[9]), W10_refined=float(Wr[9]), Wmax1to10_union=float(U0[:10].max()), Wmax1to10_refined=float(Wr[:10].max()),
               wall_s=time.time() - t0, note="diagnostic; not a W result")
    (OUT / f"{cid}.json").write_text(json.dumps(out))
    print(cid, "W10 %.4f->%.4f  Wmax %.4f->%.4f  maxlead_gain %.3f  %.0fs" % (U0[9], Wr[9], U0[:10].max(), Wr[:10].max(), np.max((Wr - U0) / np.maximum(U0, 1e-12)), time.time() - t0), flush=True)
