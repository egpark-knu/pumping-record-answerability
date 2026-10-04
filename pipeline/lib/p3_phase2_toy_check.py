"""D03 Phase 2 PREFLIGHT toy sanity check (B). FIXTURE synthetic rain only (lib/tests/p3_fixture.py): not KMA data,
not an official case, not a score. Uses the frozen v1.2 W engine unchanged, with its default (pilot) budget.

Candidate strata D1 (r = 200 m, a* = 20 d -> c = 2e5 / 2e4 / 200 d, Q = 100 m3/d); count_rule and matching are set
EXPLICITLY for the toy only (both matching options are built for invariant checks); this is not a selection.

Writes results/phase2/B_preflight/toy_check.json
Usage: OMP_NUM_THREADS=1 env/.venv_pilot/bin/python lib/p3_phase2_toy_check.py [--no-w]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "lib/tests"))
import p3_generator as g  # noqa: E402
import p3_wenvelope as w0  # noqa: E402
import p3_wenvelope_repaired_v1_2 as w  # noqa: E402
from p3_fixture import fixture_rain, fixture_realization  # noqa: E402
from p3_phase2_generator import PAIRS, Phase2Design, build_case  # noqa: E402
from p3_phase2_physmap import STORAGE, Stratum  # noqa: E402
from p3_phase2_preflight import jsonable  # noqa: E402

OUT = ROOT / "results/phase2/B_preflight/toy_check.json"


def d1_strata(r=200.0, astar=20.0, Q=100.0):
    return tuple(Stratum(t, S, T, astar / S, r, Q) for t, S in STORAGE.items() for T in (50.0, 500.0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-w", action="store_true"); a = ap.parse_args()
    rain = fixture_rain()
    real = fixture_realization(0)
    base = g.Design()
    bg = g.realization_background(real, rain, base)
    res = dict(note="FIXTURE synthetic rain; toy only; not official; count_rule/matching set explicitly for the toy", sigma_bg=bg["sigma_bg"], strata={})
    for rule in ("N_switches",):
        for st in d1_strata():
            srec = dict(cells={}, invariants={})
            cells = {}
            for m in ("M1", "M2"):
                d = Phase2Design(strata=d1_strata(), count_rule=rule, matching=m)
                for rho in d.rest_ratios:
                    for N in d.n_levels:
                        cells[(m, rho, N)] = build_case(bg, st, rho, N, d)
            inv = {}
            for m in ("M1", "M2"):
                V = {(rho, N): cells[(m, rho, N)]["derived"]["V"] for rho in (0.25, 1.0, 4.0) for N in (2, 6, 18)}
                qo = {(rho, N): cells[(m, rho, N)]["derived"]["qbar_out"] for rho in (0.25, 1.0, 4.0) for N in (2, 6, 18)}
                inv[m] = dict(V_equal_within_rho=all(np.ptp([V[(rho, N)] for N in (2, 6, 18)]) < 1e-8 * V[(rho, 2)] for rho in (0.25, 1.0, 4.0)),
                              mean_out_equal_within_rho=all(np.ptp([qo[(rho, N)] for N in (2, 6, 18)]) < 1e-10 * qo[(rho, 2)] for rho in (0.25, 1.0, 4.0)),
                              V_range_across_rho=[min(V.values()), max(V.values())], mean_out_range_across_rho=[min(qo.values()), max(qo.values())],
                              q0_range=[min(c["derived"]["q0"] for k, c in cells.items() if k[0] == m), max(c["derived"]["q0"] for k, c in cells.items() if k[0] == m)])
            c0 = cells[("M1", 1.0, 2)]
            inv["noise_rain_identical"] = all(np.array_equal(c["tf_input"]["rain"], c0["tf_input"]["rain"]) for c in cells.values())
            inv["pair2_identity_maxabs"] = max(float(np.max(np.abs(c["truth"]["E_true"][PAIRS[1]] + 0.5 * c["truth"]["E_true"][PAIRS[0]]))) for c in cells.values())
            inv["census"] = {f"rho{k[1]:g}_N{k[2]}": dict(c["derived"]["census"], rest_is_short=c["derived"]["rest_is_short_interval"])
                             for k, c in cells.items() if k[0] == "M1"}
            srec["invariants"] = inv
            srec["derived_example"] = {k: v for k, v in c0["derived"].items() if k != "census"}
            if not a.no_w:
                c = cells[("M1", 1.0, 6)]
                t0 = time.time()
                eng = w.WEngineV12(c["tf_input"], seed=20260930)
                rec = eng.run()
                th = c["truth"]["theta_true"]
                x = np.array([th["n"], np.log10(th["theta"]), np.log10(th["tau"]), np.log10(th["a"]), np.log10(th["b"])])
                sse_t = eng.point(x)[0]
                Et = c["truth"]["E_true"][PAIRS[0]]
                env = rec.get("envelope", {}).get(PAIRS[0])
                srec["toy_W"] = dict(case=c["case_id"], W_status=rec.get("W_status"), flags=rec.get("flags"), runtime_s=time.time() - t0,
                                     n_accepted=rec.get("n_accepted_total"), local_rounds=len(rec.get("local", [])),
                                     global_accepted={k: v.get("x1_n_accepted") for k, v in rec.get("global", {}).items()},
                                     refinement_increments=rec.get("refinement_increments"), summary=rec.get("summary", {}).get(PAIRS[0]),
                                     truth_nonlinear_in_G=bool(sse_t <= rec["tolerance"]["tau"]) if "tolerance" in rec else None,
                                     E_true_10_30=[float(Et[9]), float(Et[29])],
                                     E_true_inside_10_30=None if env is None else [bool(env["inf"][k] - 1e-12 <= Et[k] <= env["sup"][k] + 1e-12) for k in (9, 29)],
                                     witness_check=rec.get("witness_check"))
                print(st.sid, json.dumps(jsonable(srec["toy_W"]))[:600], flush=True)
            res["strata"][st.sid] = srec
    OUT.write_text(json.dumps(jsonable(res), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
