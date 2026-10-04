"""Independent verification diagnostic for the frozen W envelope (self-imposed check; NOT part of W).

Motivation (observed in the run): the global 512^2 product design accepted 0 combinations in real cases,
so the frozen envelope is built by local boxes around the best fits. This check asks whether G(C) has
members far from those fits that extend E(10): broad starts (32 Sobol points over the whole 5-D box plus
the 12 declared Hantush (a,b) starts at the best-fit natural parameters) -> bounded SSE minimisation ->
if the end point is inside G(C) (same locked tau_C), push sup/inf E(10) with the frozen refine routine.
It never changes the stored W; it reports any extension so that root can decide.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_wenvelope as w  # noqa: E402
from p3_make_cases import load_tf_input  # noqa: E402

PILOT = ROOT / "results/pilot"
OUT = PILOT / "wb/diagnostics"
P1 = "P1_continue_vs_stop"
BOX = np.vstack([w.NAT_BOX, w.PUMP_BOX])
HSTARTS = [(a, b) for a in (10.0, 100.0, 1000.0) for b in (1e-3, 1e-2, 1e-1, 1.0)]


def check(cid):
    t0 = time.time()
    ti = load_tf_input(PILOT / "cases/tf_inputs" / f"{cid}.npz")
    Wr = json.loads((PILOT / "wb/W" / f"{cid}.json").read_text())
    eng = w.WEngine(ti, seed=w.W_SEED if hasattr(w, "W_SEED") else 20260930)
    tau = Wr["tolerance"]["tau"]
    xb = np.array(Wr["tolerance"]["xbest"])
    env = Wr["envelope"][P1]
    sup0, inf0 = env["sup"][9], env["inf"][9]
    starts = list(qmc.scale(qmc.Sobol(5, scramble=True, seed=777).random(32), BOX[:, 0], BOX[:, 1]))
    starts += [np.concatenate([xb[:3], [np.log10(a), np.log10(b)]]) for a, b in HSTARTS]
    sup, inf = sup0, inf0
    in_G, ends = 0, []
    for x0 in starts:
        def f(x):
            sse, _, _, _, t95 = eng.point(np.clip(x, BOX[:, 0], BOX[:, 1]))
            return sse if t95 <= w.T95_MAX else 1e12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(f, x0, method="Powell", bounds=BOX, options=dict(maxfev=500, xtol=1e-3, ftol=1e-10))
        x = np.clip(r.x, BOX[:, 0], BOX[:, 1])
        if r.fun > tau:
            continue
        in_G += 1
        ends.append(x.tolist())
        ref = eng.refine(dict(sup=[sup] * 30, inf=[inf] * 30), tau, P1, 10, starts=dict(sup=x, inf=x))
        sup, inf = max(sup, ref["sup"]), min(inf, ref["inf"])
    W0 = sup0 - inf0
    out = dict(case_id=cid, tau=tau, n_starts=len(starts), n_starts_reaching_G=in_G, W10_frozen=W0, sup10_frozen=sup0, inf10_frozen=inf0,
               sup10_check=sup, inf10_check=inf, W10_check=sup - inf, rel_extension=(sup - inf - W0) / W0 if W0 > 0 else None,
               ends_in_G=ends, wall_s=time.time() - t0,
               note="self-imposed verification diagnostic; stored W unchanged")
    (OUT / f"multistart_{cid}.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:]
    for cid in ids:
        if (OUT / f"multistart_{cid}.json").exists():
            continue
        o = check(cid)
        print(cid, o["n_starts_reaching_G"], round(o["W10_frozen"], 5), round(o["W10_check"], 5), o["rel_extension"], round(o["wall_s"], 1), flush=True)
