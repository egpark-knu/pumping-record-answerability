"""Generate the 200 matched D02 Tier A cases after the protocol freeze (no scoring).

Usage (pilot):  python lib/p3_make_cases.py --out results/pilot/cases
Fixture dry run (synthetic rain, never pilot): python lib/p3_make_cases.py --fixture --out /tmp/p3_fixture_cases

Writes
  tool_inputs_H10.npz, tool_inputs_H30.npz   A adapter schema, 800 rows each (NO truth)
  tf_inputs/{case_id}.npz                     same-window observed record for W and the reference (NO truth)
  truth/{case_id}.npz                         evaluation only
  observed_manifest.json, truth_manifest.json, generation_checks.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_generator as g  # noqa: E402

PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
SITE_SHA = {"안동태화_충적": "20dddf7f15db", "산청산청_암반": "e95466d5ea3b", "남해남해_암반": "7733e1352ddc"}
TRUTH_KEYS = {"h_star_a", "h_nat", "h_pump_a", "eps", "E_true", "delta", "theta_true"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_tf(path: Path, case: dict, digest: str):
    ti = case["tf_input"]
    arrs = dict(head_context=ti["head_context"], rain=ti["rain"], pumping_context=ti["pumping_context"], dates=ti["dates"].astype(str),
                meta_json=np.array(json.dumps(ti["meta"], sort_keys=True, ensure_ascii=False)), case_id=np.array(case["case_id"]),
                protocol_sha256=np.array(digest))
    for k, v in ti["future_Q"].items():
        arrs["future_Q__" + k] = v
    np.savez_compressed(path, **arrs)


def load_tf_input(path) -> dict:
    """Reconstruct the tf_input dict used by p3_wenvelope.WEngine and p3_pastas.fit_and_propagate."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    return dict(case_id=str(d["case_id"]), head_context=d["head_context"], rain=d["rain"], pumping_context=d["pumping_context"],
                dates=d["dates"], meta=json.loads(str(d["meta_json"])), protocol_sha256=str(d["protocol_sha256"]),
                future_Q={k[len("future_Q__"):]: d[k] for k in d if k.startswith("future_Q__")})


def save_truth(path: Path, case: dict, digest: str):
    t = case["truth"]
    arrs = dict(case_id=np.array(case["case_id"]), h_star_a=t["h_star_a"], h_nat=t["h_nat"], h_pump_a=t["h_pump_a"], eps=t["eps"],
                theta_true_json=np.array(json.dumps({k: float(v) for k, v in t["theta_true"].items()}, sort_keys=True)),
                protocol_sha256=np.array(digest))
    for p in PAIRS:
        arrs["E_true__" + p] = t["E_true"][p]
        arrs["delta__" + p] = t["delta"][p]
    np.savez_compressed(path, **arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    d = g.Design()
    if a.fixture:
        sys.path.insert(0, str(ROOT / "lib/tests"))
        from p3_fixture import fixture_rain, fixture_realization
        digest = "f" * 64
        reals = [fixture_realization(r) for r in range(2)]
        rains = {"FIXTURE_SYNTHETIC": fixture_rain()}
    else:
        digest = g.require_frozen(ROOT / "results/pilot/protocol.md", ROOT / "results/pilot/protocol.sha256")
        reals = json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))
        rains = {s: g.load_rain(s, p) for s, p in SITE_SHA.items()}
    out = a.out
    (out / "tf_inputs").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    checks, obs_man, tru_man, cases_all = {}, [], [], []
    inv = dict(noise_identical=True, rain_identical=True, base_identical_outside_rest=True, rest_only_zero_days=True,
               rain_equals_source_slice=True, sr_definition=True, q0_positive=True)
    pair_id_max = 0.0
    for real in reals:
        rain = rains[real["site_stem"]]
        bg = g.realization_background(real, rain, d)
        src = rain.loc[bg["dates"][0]:bg["dates"][-1]].values
        inv["rain_equals_source_slice"] &= bool(np.array_equal(src, bg["P"]))
        cs = [g.build_case(bg, rho, sr, pir, d) for pir in d.pi_layers for sr in d.signal_ratios for rho in d.rest_ratios]
        for c in cs:
            tt = c["truth"]["theta_true"]; Qw = c["tf_input"]["pumping_context"]; R = tt["R_days"]
            outside = np.ones(g.CTX, bool); outside[g.REST_END - R:g.REST_END] = False
            inv["noise_identical"] &= bool(np.array_equal(c["truth"]["eps"], cs[0]["truth"]["eps"]))
            inv["rain_identical"] &= bool(np.array_equal(c["tf_input"]["rain"], cs[0]["tf_input"]["rain"]))
            inv["base_identical_outside_rest"] &= bool(np.array_equal(Qw[outside], (tt["Q_ref"] * bg["mult"])[outside]))
            inv["rest_only_zero_days"] &= bool((Qw == 0).sum() == R and np.all(Qw[~outside] == 0))
            inv["sr_definition"] &= bool(abs(tt["A"] * tt["Q_ref"] / bg["sigma_bg"] - c["meta"]["signal_ratio"]) < 1e-12)
            inv["q0_positive"] &= bool(tt["q0"] > 0)
            pair_id_max = max(pair_id_max, float(np.max(np.abs(c["truth"]["E_true"][PAIRS[1]] + 0.5 * c["truth"]["E_true"][PAIRS[0]]))))
            pt = out / "tf_inputs" / f"{c['case_id']}.npz"; save_tf(pt, c, digest)
            pr = out / "truth" / f"{c['case_id']}.npz"; save_truth(pr, c, digest)
            obs_man.append(dict(case_id=c["case_id"], realization=int(real["realization"]), site=real["site_stem"], origin_date=c["meta"]["origin_date"],
                                rest_ratio=c["meta"]["rest_ratio"], signal_ratio=c["meta"]["signal_ratio"], pi_layer=c["meta"]["pi_layer"],
                                tf_input=str(pt.relative_to(out)), sha256=sha(pt)))
            tru_man.append(dict(case_id=c["case_id"], truth=str(pr.relative_to(out)), sha256=sha(pr)))
            cases_all.append(c)
    checks["matched_invariants"] = inv
    checks["truth_pair2_identity_maxabs_m"] = pair_id_max
    # tool archives (A schema) + A's validator + reload checks
    spec = importlib.util.spec_from_file_location("adapter", ROOT / "lib/timesfm_pilot_adapter.py")
    ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)
    tool = {}
    for H in (10, 30):
        rows = [r for c in cases_all for r in g.tool_rows(c, H)]
        arr = g.tool_npz_arrays(rows, digest)
        path = out / f"tool_inputs_H{H}.npz"
        np.savez_compressed(path, **arr)
        with np.load(path, allow_pickle=False) as z:
            back = {k: z[k] for k in z.files}
        order, _ = ad.validate(back, H)
        leak = sorted(set(back) & TRUTH_KEYS)
        tool[H] = dict(path=path.name, sha256=sha(path), n_rows=len(rows), n_pairs=len(rows) // 2, n_cases=len(cases_all),
                       validator_ok=len(order) == len(rows), protocol_sha256_ok=str(back["protocol_sha256"]) == digest, truth_keys=leak,
                       shapes={k: list(back[k].shape) for k in ("head", "pumping", "rainfall")})
        # the tool head equals the tf_input head; tool future pumping equals the tf schedule
        tf0 = load_tf_input(out / "tf_inputs" / f"{cases_all[0]['case_id']}.npz")
        tool[H]["cross_consistent"] = bool(np.array_equal(back["head"][0], tf0["head_context"]) and
                                           np.array_equal(back["pumping"][0][g.CTX:], tf0["future_Q"][PAIRS[0] + "_a"][:H]))
    checks["tool_archives"] = tool
    # tf_input reload: no truth keys, same-window lengths
    bad = []
    for m in obs_man:
        ti = load_tf_input(out / m["tf_input"])
        if ti["head_context"].size != g.CTX or ti["rain"].size != g.CTX + g.HMAX or ti["pumping_context"].size != g.CTX or ti["protocol_sha256"] != digest:
            bad.append(m["case_id"])
        with np.load(out / m["tf_input"], allow_pickle=False) as z:
            if set(z.files) & TRUTH_KEYS:
                bad.append(m["case_id"] + ":truth_leak")
    checks["tf_inputs_ok"] = not bad
    checks["tf_inputs_bad"] = bad
    checks["counts"] = dict(cases=len(cases_all), unique_ids=len({c["case_id"] for c in cases_all}), realizations=len(reals),
                            expected_cases=len(reals) * 20)
    json.dump(obs_man, open(out / "observed_manifest.json", "w"), ensure_ascii=False, indent=1)
    json.dump(tru_man, open(out / "truth_manifest.json", "w"), ensure_ascii=False, indent=1)
    ok = (all(inv.values()) and pair_id_max < 1e-12 and all(v["validator_ok"] and v["protocol_sha256_ok"] and not v["truth_keys"] and v["cross_consistent"]
                                                           for v in tool.values()) and checks["tf_inputs_ok"] and checks["counts"]["cases"] == checks["counts"]["expected_cases"]
          and checks["counts"]["unique_ids"] == checks["counts"]["cases"])
    checks["all_ok"] = bool(ok)
    checks["runtime_s"] = time.time() - t0
    checks["protocol_sha256"] = digest
    checks["rain_files"] = {s: dict(sha256=r.attrs.get("sha256"), path=r.attrs.get("path")) for s, r in rains.items()} if not a.fixture else "FIXTURE"
    json.dump(checks, open(out / "generation_checks.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in checks.items() if k != "tool_archives"}, ensure_ascii=False, default=str)[:1500])
    print("ALL_OK" if ok else "CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
