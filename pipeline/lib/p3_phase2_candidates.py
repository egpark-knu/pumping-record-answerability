"""D03 Phase 2 PREFLIGHT: per-stratum design tables for candidate physical designs (design quantities only).
Writes results/phase2/B_preflight/candidates.json and prints markdown rows. No case, no W, no score."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
from p3_phase2_physmap import STORAGE, Stratum, describe, unit_step  # noqa: E402
from p3_phase2_preflight import PUMP_BOX_FROZEN, jsonable  # noqa: E402
from p3_phase2_generator import stop_positions  # noqa: E402

CANDS = {"D1_r200_a20": (200.0, 20.0), "D2_r150_a10": (150.0, 10.0), "D3_r100_a20": (100.0, 20.0)}
Q = 100.0
sb = np.array([r["sigma_bg_m"] for r in json.load(open(ROOT / "results/phase2/B_preflight/sigma_bg.json"))["rows"]])
out = {}
for name, (r, astar) in CANDS.items():
    rows = []
    for t, S in STORAGE.items():
        for T in (50.0, 500.0):
            d = describe(Stratum(t, S, T, astar / S, r, Q))
            t95 = d["t95"]; u = unit_step(d["a"], d["b"], [10.0, 30.0])
            SR = Q * d["gain"] / sb
            cap = {}
            for f in (0.1, 0.25):
                ell = max(1, round(f * t95))
                try:
                    stop_positions(t95, 10 ** 6, ell, int(round(4 * t95)))
                except ValueError as e:
                    cap[f] = int(str(e).split("capacity ")[1].rstrip(")"))
            rows.append(dict(sid=d["sid"], c=d["c"], lam=d["lam"], a=d["a"], b=d["b"], pi_r=d["pi_r"], t50=d["t50"], t95=t95, gain=d["gain"],
                             s_inf_m=Q * d["gain"], E10_P1_m=Q * d["gain"] * u[0], E30_P1_m=Q * d["gain"] * u[1], k10_over_t95=10 / t95, k30_over_t95=30 / t95,
                             SR_min=float(SR.min()), SR_med=float(np.median(SR)), SR_max=float(SR.max()),
                             R_round={rho: int(round(rho * t95)) for rho in (0.25, 1.0, 4.0)}, R_ceil={rho: int(math.ceil(rho * t95)) for rho in (0.25, 1.0, 4.0)},
                             iso_stop_capacity=cap,
                             in_frozen_box=bool(PUMP_BOX_FROZEN[0][0] <= math.log10(d["a"]) <= PUMP_BOX_FROZEN[0][1] and PUMP_BOX_FROZEN[1][0] <= math.log10(d["b"]) <= PUMP_BOX_FROZEN[1][1])))
    out[name] = dict(r=r, astar=astar, Q=Q, c_rule="c = a*/S", rows=rows)
    print(f"\n### {name}  r={r:g} m, a*={astar:g} d, Q={Q:g} m3/d")
    print("| stratum | c (d) | λ (m) | b | π_r | t50 (d) | t95 (d) | gain (m per m³/d) | s∞ (m) | E_P1(10) (m) | E_P1(30) (m) | 10/t95 | SR min/med/max | R(ρ=.25/1/4), round | stop capacity f=.1/.25 | frozen box |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for x in rows:
        R = x["R_round"]
        print(f"| {x['sid']} | {x['c']:.3g} | {x['lam']:.0f} | {x['b']:.3g} | {x['pi_r']:.3g} | {x['t50']:.2f} | {x['t95']:.1f} | {x['gain']:.3g} | {x['s_inf_m']:.3f} | {x['E10_P1_m']:.3f} | {x['E30_P1_m']:.3f} | {x['k10_over_t95']:.2f} | {x['SR_min']:.2f}/{x['SR_med']:.2f}/{x['SR_max']:.2f} | {R[0.25]}/{R[1.0]}/{R[4.0]} | {x['iso_stop_capacity'].get(0.1)}/{x['iso_stop_capacity'].get(0.25)} | {'yes' if x['in_frozen_box'] else 'NO'} |")
(ROOT / "results/phase2/B_preflight/candidates.json").write_text(json.dumps(jsonable(out), indent=1))
