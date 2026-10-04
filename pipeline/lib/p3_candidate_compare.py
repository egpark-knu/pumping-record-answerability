"""Compare frozen W outputs with CANDIDATE v1.1 on the diagnostic subset (evidence for root decision)."""
import json
from pathlib import Path
P = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))) / "results/pilot"
P1 = "P1_continue_vs_stop"
rows = []
for f in sorted((P / "wb_candidate_v1_1/W").glob("*.json")):
    c = json.loads(f.read_text()); cid = c["case_id"]
    fz = json.loads((P / "wb/W" / f.name).read_text())
    te = json.loads((P / "wb/truth_eval" / f.name).read_text())
    def s(r):
        e = r.get("envelope", {}).get(P1)
        return (None, None) if not e else (e["W"][9], max(e["W"][:10]))
    fw, cw = s(fz), s(c)
    tcf = te.get("truth_compatibility"); tcc = c.get("truth_eval")
    rows.append(dict(case_id=cid, frozen_flags=fz.get("flags"), cand_flags=c.get("flags"), cand_converged=c.get("converged"),
                     frozen_W10=fw[0], cand_W10=cw[0], frozen_Wmax=fw[1], cand_Wmax=cw[1],
                     rel_change_W10=None if fw[0] in (None, 0) else (cw[0] - fw[0]) / fw[0],
                     frozen_Etrue_inside_1to10=None if not tcf else all(tcf["E_true_inside"][P1][:10]),
                     cand_Etrue_inside_1to10=None if not tcc else all(tcc["E_true_inside"][P1][:10]),
                     cand_truth_in_G=None if not tcc else tcc["truth_nonlinear_in_G"], cand_best_fit_seed_in_G=c.get("best_fit_seed_in_G"),
                     cand_n_accepted=c.get("n_accepted_total", {}).get("x1"), cand_wall_s=c.get("wall_s")))
json.dump(rows, open(P / "wb_candidate_v1_1/compare_frozen_vs_v1_1.json", "w"), indent=1)
for r in rows:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
