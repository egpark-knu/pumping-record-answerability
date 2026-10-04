"""D03 Phase 2 merge tests (B): production generator on FIXTURE rain with the actual pilot seeds, full-grid export,
origin-link negatives, metric masks, units. Not KMA heads, not official cases, not scores.

Writes results/phase2/B_merge/merge_tests.json and a 2-case sample export NPZ + a synthetic cell_metrics sample.
Usage: env/.venv_pilot/bin/python lib/tests/p3_phase2_merge_tests.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "lib/tests"))
import p3_generator as g  # noqa: E402
import p3_phase2_cases as pc  # noqa: E402
import p3_phase2_export as ex  # noqa: E402
import p3_phase2_metrics as mt  # noqa: E402
import p3_phase2_timesfm as A  # noqa: E402
from p3_phase2_physmap import unit_step  # noqa: E402

OUT = ROOT / "results/phase2/B_merge"
PLACEHOLDER = "0" * 64
RES = {}


def test(fn):
    t0 = time.time()
    try:
        info = fn() or {}
        RES[fn.__name__] = dict(passed=True, info=info, s=round(time.time() - t0, 2))
    except Exception as e:  # recorded, not hidden
        RES[fn.__name__] = dict(passed=False, error=repr(e), tb=traceback.format_exc()[-1500:])
    print(fn.__name__, RES[fn.__name__]["passed"], flush=True)
    return fn


CASES = list(pc.all_cases(fixture=True))


@test
def test_fixture_cases_use_production_schedules():
    reals = json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))
    by = {r["realization"]: g.base_schedule(r["seed_schedule"], g.Design()) for r in reals}
    n = 0
    for c in CASES:
        d = c["derived"]
        st = next(s for s in pc.strata() if s.sid == d["sid"])
        sc = pc.schedule(by[d["realization"]], pc.layer(st), d["rho_nominal"], d["N_nominal"])
        assert np.array_equal(sc["q"], c["tf_input"]["pumping_context"]), c["case_id"]
        n += 1
    assert n == 540 and len({c["case_id"] for c in CASES}) == 540
    return dict(n_cases=n)


@test
def test_case_invariants_heads_and_truth():
    by_real = {}
    pair2 = 0.0
    for c in CASES:
        by_real.setdefault(c["derived"]["realization"], []).append(c)
        E = c["truth"]["E_true"]
        pair2 = max(pair2, float(np.max(np.abs(E[pc.PAIRS[1]] + 0.5 * E[pc.PAIRS[0]]))))
        cen = c["derived"]["census"]
        assert cen["N_ON"] == cen["N_OFF"] == c["derived"]["N_nominal"] // 2 and cen["designated_rest_identity"]
        assert E[pc.PAIRS[0]][0] < 0 and np.all(np.isfinite(E[pc.PAIRS[0]]))
    for r, cs in by_real.items():
        assert all(np.array_equal(x["truth"]["eps"], cs[0]["truth"]["eps"]) for x in cs)
        assert all(np.array_equal(x["tf_input"]["rain"], cs[0]["tf_input"]["rain"]) for x in cs)
        assert all(np.array_equal(x["truth"]["h_nat"], cs[0]["truth"]["h_nat"]) for x in cs)
    assert pair2 < 1e-12
    return dict(pair2_identity_maxabs_m=pair2)


@test
def test_units_gain_and_E_true():
    """E_P1(k) = -q0 * gain * S_unit(k) (m, with q0 in m3/d and gain in m per m3/d); steady drawdown = Q gain."""
    worst = 0.0
    for c in CASES[:54]:
        d = c["derived"]
        u = unit_step(d["a_d"], d["b"], np.arange(0, pc.CTX + pc.HMAX + 1, dtype=float))[1:31]  # same grid as the generator
        want = -d["q0_m3d"] * d["gain_m_per_m3d"] * u
        worst = max(worst, float(np.max(np.abs(c["truth"]["E_true"][pc.PAIRS[0]] - want))))
    L = pc.layer(pc.strata()[0])
    s_long = float(pc.Q_NOM * L["d"]["gain"] * L["S"][-1])
    d0 = L["d"]
    cross_grid = float(np.max(np.abs(unit_step(d0["a"], d0["b"], np.arange(1, 31, dtype=float)) - L["S"][1:31])))
    assert worst < 1e-12
    return dict(E_true_vs_closed_form_maxabs_m=worst, steady_check_m=s_long, gain_confined_T50=L["d"]["gain"],
                frozen_kernel_cross_grid_step_diff_of_gain=cross_grid,
                note="closed form evaluated on the generator's own 0..1054 d grid; the frozen quadrature grid scales with max t, "
                     "so a 1..30 d evaluation differs by the recorded fraction of gain (pilot-known quadrature level)")


@test
def test_full_grid_export_H10_H30_and_tracks():
    info = {}
    arrs = {}
    for H in (10, 30):
        arr = ex.export_arrays(CASES, H, PLACEHOLDER, require_full_grid=True)
        arrs[H] = arr
        n = len(arr["query_id"])
        assert n == 2160 and len(set(arr["pair_id"])) == 1080 and len(set(arr["case_id"])) == 540 and len(set(arr["query_id"])) == 2160
        assert all(q.endswith(f"__H{H}") for q in arr["query_id"])
        assert set(arr) == A.INPUT_KEYS
        tr = {}
        for t in A.TRACKS:
            p = A.prepare_track(arr, H, t, require_full_grid=True)
            tr[t] = dict(contexts=list(p["contexts"].shape) if "contexts" in p else None, keys=sorted(p)[:8])
        h = hashlib.sha256()
        for k in ("head", "pumping", "rainfall"):
            h.update(np.ascontiguousarray(arr[k]).tobytes())
        info[f"H{H}"] = dict(rows=n, pairs=1080, cases=540, tracks=tr, numeric_sha256=h.hexdigest())
    # H10 inputs are the H30 prefix (independent requests later)
    assert np.array_equal(arrs[10]["pumping"], arrs[30]["pumping"][:, :pc.CTX + 10])
    assert np.array_equal(arrs[10]["rainfall"], arrs[30]["rainfall"][:, :pc.CTX + 10])
    assert np.array_equal(arrs[10]["head"], arrs[30]["head"])
    # sample NPZ (2 cases) for inspection
    idx = [i for i, c in enumerate(arrs[10]["case_id"]) if c in (CASES[0]["case_id"], CASES[-1]["case_id"])]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "export_sample_H10.npz", **{k: (v[idx] if k != "protocol_sha256" else v) for k, v in arrs[10].items()})
    meta0 = json.loads(str(arrs[10]["metadata_json"][0]))
    info["metadata_example"] = meta0
    return info


def _one_case_arrays(H=10):
    return ex.export_arrays([CASES[0]], H, PLACEHOLDER)


@test
def test_origin_link_negative_q80_future100():
    """Producer and array check reject observed last rate 80 with continue/current future 100;
    A's generic validator alone accepts it (why the producer enforces the D03 contract)."""
    bad = dict(CASES[0]); ti = dict(bad["tf_input"]); ti["pumping_context"] = ti["pumping_context"].copy()
    ti["pumping_context"][-1] = 80.0
    ti["future_Q"] = {"P1_continue_vs_stop_a": np.full(30, 100.0), "P1_continue_vs_stop_b": np.zeros(30),
                      "P2_current_vs_1p5x_a": np.full(30, 100.0), "P2_current_vs_1p5x_b": np.full(30, 150.0)}
    bad["tf_input"] = ti
    rej_case = False
    try:
        ex.assert_case_origin_link(bad, 10)
    except ex.OriginLinkError:
        rej_case = True
    arr = _one_case_arrays(10)
    arr = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arr.items()}
    arr["pumping"][:, pc.CTX - 1] = 80.0
    for i, s in enumerate(arr["schedule_id"]):
        pid = str(arr["pair_id"][i])
        arr["pumping"][i, pc.CTX:] = 100.0 if s == "a" else (0.0 if pid.endswith(pc.PAIRS[0]) else 150.0)
    rej_arr = False
    try:
        ex.assert_array_origin_link(arr)
    except ex.OriginLinkError:
        rej_arr = True
    A_generic_accepts = True
    try:
        A.validate(arr, 10)
    except ValueError:
        A_generic_accepts = False
    assert rej_case and rej_arr
    return dict(producer_rejects=rej_case, array_check_rejects=rej_arr, A_generic_validator_accepts=A_generic_accepts)


@test
def test_origin_link_other_negatives():
    out = {}
    for name, mutate in {
        "stop_nonzero": lambda a, i, p, s: a["pumping"].__setitem__((i, slice(pc.CTX, None)), 1.0) if (p.endswith(pc.PAIRS[0]) and s == "b") else None,
        "wrong_1p5x": lambda a, i, p, s: a["pumping"].__setitem__((i, slice(pc.CTX, None)), 1.4 * a["pumping"][i, pc.CTX - 1]) if (p.endswith(pc.PAIRS[1]) and s == "b") else None,
        "continue_off_by_ulp": lambda a, i, p, s: a["pumping"].__setitem__((i, slice(pc.CTX, None)), np.nextafter(a["pumping"][i, pc.CTX - 1], 1e9)) if (p.endswith(pc.PAIRS[0]) and s == "a") else None,
    }.items():
        arr = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in _one_case_arrays(10).items()}
        for i in range(len(arr["query_id"])):
            mutate(arr, i, str(arr["pair_id"][i]), str(arr["schedule_id"][i]))
        try:
            ex.assert_array_origin_link(arr); out[name] = False
        except ex.OriginLinkError:
            out[name] = True
    assert all(out.values())
    return dict(rejected=out)


@test
def test_metric_masks_and_ratio():
    truth = [1.0, -0.5, 0.0, float("nan"), float("inf"), 0.0005, 0.01, 0.2]
    tool = [2.0, 0.75, 0.3, 0.1, 0.1, 0.0, -0.02, float("nan")]
    rows = [mt.row_metrics(t, o, W=0.1, env_inf=-1, env_sup=1) for t, o in zip(truth, tool)]
    den = mt.denominators(rows)
    ratios = [r["effect_ratio"] for r in rows]
    assert den["n_truth_zero"] == 1 and den["n_truth_nonfinite"] == 2 and den["n_truth_nonzero"] == 5
    assert den["n_small_001"] == 1 and den["n_small_02"] == 2 and den["n_ratio_defined"] == 4
    assert ratios[:2] == [2.0, -1.5] and ratios[5] == 0.0 and ratios[6] == -2.0 and ratios[2] is None and ratios[7] is None
    assert rows[5]["sign_agreement"] is False  # tool exactly zero: valid ratio 0, sign disagreement
    assert rows[5]["small_001"] and rows[5]["ratio_defined"]  # small flag does not remove the row
    # W floor / invalid width
    assert mt.row_metrics(1, 1.1, W=1e-6)["error_over_W"] is None and mt.row_metrics(1, 1.1, W=2e-6)["error_over_W"] is not None
    neg = mt.row_metrics(1, 1.1, W=-0.01)
    assert neg["W_invalid"] and neg["W_negative"] and neg["error_over_W"] is None
    # inherited inclusion tolerance 1e-12 m
    sup, inf = 0.3, -0.2
    inc = [mt.row_metrics(1, sup + 5e-13, 1, inf, sup)["envelope_includes"], mt.row_metrics(1, sup + 5e-10, 1, inf, sup)["envelope_includes"],
           mt.row_metrics(1, inf - 5e-13, 1, inf, sup)["envelope_includes"], mt.row_metrics(1, inf - 5e-10, 1, inf, sup)["envelope_includes"]]
    assert inc == [True, False, True, False] and mt.TOL_IN == 1e-12
    return dict(denominators=den, ratios=ratios, inclusion_boundary=inc, TOL_IN=mt.TOL_IN)


@test
def test_cell_metrics_contract_mapping():
    rows = []
    c = CASES[0]
    d = c["derived"]
    for p in pc.PAIRS:
        for tr in ("raw", "exp10", "exp30", "exp90"):
            for k in (10, 30):
                Et = float(c["truth"]["E_true"][p][k - 1])
                rows.append(mt.cell_metrics_row(d, p, tr, "lead", k, 0.05, "ok", Et, 0.5 * Et, Et - 0.02, Et + 0.03, "synthetic_fixture"))
        rows.append(mt.cell_metrics_row(d, p, "raw", "max_days_1_10", None, 0.06, "ok", float(c["truth"]["E_true"][p][9]), 0.4 * float(c["truth"]["E_true"][p][9]), -1, 1, "synthetic_fixture"))
    mt.write_cell_metrics(rows, OUT / "cell_metrics_contract_sample_synthetic.csv")
    assert len(rows) == 18 and all(r["effect_ratio"] == 0.5 for r in rows if r["quantity"] == "lead")
    return dict(n_rows=len(rows), columns=mt.contract_columns(), note="synthetic mapping sample only; not a result")


def main():
    ok = all(v["passed"] for v in RES.values())
    rec = dict(status="FIXTURE rain + actual pilot seeds; production code path; not official", n_tests=len(RES), n_passed=sum(v["passed"] for v in RES.values()),
               all_passed=ok, tests=RES,
               code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                            ("lib/p3_phase2_cases.py", "lib/p3_phase2_export.py", "lib/p3_phase2_metrics.py", "lib/p3_phase2_timesfm.py",
                             "lib/timesfm_pilot_adapter.py", "lib/p3_wb_collect_repaired_v1_2.py", "lib/tests/p3_phase2_merge_tests.py")})
    (OUT / "merge_tests.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{rec['n_passed']}/{rec['n_tests']} passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
