"""D03 Phase 2 PREFLIGHT toy: runtime and W sensitivity of the candidate enlarged budget vs the pilot budget.
FIXTURE rain only, candidate D1 strata, toy cells (rho=1, N=6 and rho=4, N=18). Not official, not a score.
Writes results/phase2/B_preflight/toy_budget.json"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "lib/tests"))
import p3_generator as g  # noqa: E402
import p3_wenvelope_repaired_v1_2 as w  # noqa: E402
from p3_fixture import fixture_rain, fixture_realization  # noqa: E402
from p3_phase2_generator import PAIRS, Phase2Design, build_case  # noqa: E402
from p3_phase2_preflight import jsonable  # noqa: E402
from p3_phase2_toy_check import d1_strata  # noqa: E402

BUDGETS = {"pilot": dict(global_levels=(128, 256, 512), n_local=256, max_local_rounds=8),
           "enlarged_candidate": dict(global_levels=(128, 256, 512, 1024), n_local=512, max_local_rounds=12)}
bg = g.realization_background(fixture_realization(0), fixture_rain(), g.Design())
d = Phase2Design(strata=d1_strata(), count_rule="N_switches", matching="M1")
out = []
for st in d.strata:
    for rho, N in ((1.0, 6), (4.0, 18)):
        c = build_case(bg, st, rho, N, d)
        for bn, kw in BUDGETS.items():
            t0 = time.time()
            rec = w.WEngineV12(c["tf_input"], seed=20260930).run(**kw)
            s = rec.get("summary", {}).get(PAIRS[0], {})
            out.append(dict(case=c["case_id"], budget=bn, W_status=rec.get("W_status"), flags=rec.get("flags"), runtime_s=time.time() - t0,
                            n_accepted=rec.get("n_accepted_total", {}).get("x1"), local_rounds=len(rec.get("local", [])), W10=s.get("W10"), W30=s.get("W30"),
                            Wmax1to10=s.get("Wmax1to10"), witness_pass=rec.get("witness_check", {}).get("pass")))
            print(json.dumps(jsonable(out[-1])), flush=True)
(ROOT / "results/phase2/B_preflight/toy_budget.json").write_text(json.dumps(jsonable(dict(note="FIXTURE toy; not official", budgets=BUDGETS, rows=out)), indent=1))
