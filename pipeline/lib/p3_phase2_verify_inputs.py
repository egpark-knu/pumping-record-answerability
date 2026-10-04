"""Independent verification of the OFFICIAL Phase 2 case bundle (B). Reads results/phase2/cases only; writes
results/phase2/cases/input_verification.json. No W, forecast, reference or score."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_generator as g  # noqa: E402
import p3_phase2_export as ex  # noqa: E402
from p3_make_cases import load_tf_input, TRUTH_KEYS  # noqa: E402
from p3_phase2_cases import SITE_SHA, CTX, HMAX, PAIRS  # noqa: E402

D = ROOT / "results/phase2/cases"
P2 = ROOT / "results/phase2"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    digest = (P2 / "protocol.sha256").read_text().split()[0]
    assert digest == sha(P2 / "protocol.md")
    man = json.load(open(D / "derived_manifest.json"))
    cal = {r["realization"]: r for r in json.load(open(ROOT / "results/pilot/phase0_checks/realization_calendar.json"))}
    rains = {s: g.load_rain(s, p) for s, p in SITE_SHA.items()}
    sig = {r["realization"]: r["sigma_bg_m"] for r in json.load(open(P2 / "B_preflight/sigma_bg.json"))["rows"]}
    sched = {r["case_id"]: r for r in csv.DictReader(open(P2 / "B_merge/schedule_cases.csv"))}
    chk = defaultdict(lambda: True)
    eps_ref, nat_ref = {}, {}
    per_group = defaultdict(list)
    for m in man:
        cid = m["case_id"]
        ti = load_tf_input(D / m["tf_input"])
        chk["tf_hash"] &= sha(D / m["tf_input"]) == m["tf_sha256"]
        chk["truth_hash"] &= sha(D / m["truth"]) == m["truth_sha256"]
        chk["protocol_stamp"] &= ti["protocol_sha256"] == digest
        with np.load(D / m["tf_input"], allow_pickle=False) as z:
            chk["tf_truth_free"] &= not (set(z.files) & TRUTH_KEYS)
        real = cal[m["realization"]]
        chk["real_KMA_site"] &= m["site"] == real["site_stem"] and ti["meta"]["rain_site"] == real["site_stem"]
        start = pd.Timestamp(real["context_start"])
        src = rains[real["site_stem"]].loc[start:start + pd.Timedelta(days=CTX + HMAX - 1)].values
        chk["rain_equals_KMA_slice"] &= np.array_equal(src, ti["rain"])
        chk["dates_match_calendar"] &= str(ti["dates"][0]) == real["context_start"] and str(ti["dates"][CTX]) == real["origin_first_forecast_day"]
        with np.load(D / m["truth"], allow_pickle=False) as z:
            eps, hnat = z["eps"], z["h_nat"]
            E1, E2 = z["E_true__" + PAIRS[0]], z["E_true__" + PAIRS[1]]
        r = m["realization"]
        if r not in eps_ref:
            eps_ref[r], nat_ref[r] = eps, hnat
        chk["noise_shared_in_realization"] &= np.array_equal(eps, eps_ref[r])
        chk["natural_shared_in_realization"] &= np.array_equal(hnat, nat_ref[r])
        chk["sigma_bg_equals_pilot_design"] &= m["sigma_bg_m"] == sig[r]
        chk["pair_identity_1e-12"] &= float(np.max(np.abs(E2 + 0.5 * E1))) < 1e-12
        s = sched[cid]
        chk["schedule_equals_production_check"] &= s["base_N_sha256"] == m["base_N_sha256"] and float(s["V_m3"]) == m["V_m3"] and float(s["q0_m3d"]) == m["q0_m3d"]
        cen = m["census"]
        chk["census"] &= cen["N_ON"] == cen["N_OFF"] == m["N_nominal"] // 2 and cen["designated_rest_identity"] and cen["tail_on"] and cen["tail_len"] >= m["t95_d"]
        chk["units"] &= m["units"] == dict(head="m", pumping="m3/d", rain="mm/d", T="m2/d", r="m", c="d", gain="m per m3/d", time="d")
        chk["origin_future"] &= bool(np.all(ti["future_Q"][PAIRS[0] + "_a"] == ti["pumping_context"][-1]))
        per_group[(m["sid"], m["rho_nominal"], m["realization"])].append((m["V_m3"], m["mean_outside_rest_m3d"]))
    rv = max((max(v[0] for v in x) - min(v[0] for v in x)) / max(v[0] for v in x) for x in per_group.values())
    rm = max((max(v[1] for v in x) - min(v[1] for v in x)) / max(v[1] for v in x) for x in per_group.values())
    tool = {}
    for H in (10, 30):
        with np.load(D / f"tool_inputs_H{H}.npz", allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
        ex.assert_array_origin_link(arr)
        ex.A.validate(arr, H, require_full_grid=True)
        metas = [json.loads(x) for x in arr["metadata_json"]]
        tool[H] = dict(rows=len(arr["query_id"]), pairs=len(set(arr["pair_id"])), cases=len(set(arr["case_id"])), protocol=str(arr["protocol_sha256"]) == digest,
                       truth_keys=sorted(set(arr) & TRUTH_KEYS), sha256=sha(D / f"tool_inputs_H{H}.npz"),
                       label_keys=sorted(metas[0]))
    rec = dict(protocol_sha256=digest, n_cases=len(man), checks=dict(chk), max_rel_V_spread_N=rv, max_rel_mean_spread_N=rm, tool=tool,
               rain_sha256={s: r.attrs["sha256"] for s, r in rains.items()})
    rec["all_ok"] = bool(all(chk.values()) and len(man) == 540 and rv < 1e-12 and rm < 1e-12
                         and all(v["rows"] == 2160 and v["pairs"] == 1080 and v["cases"] == 540 and v["protocol"] and not v["truth_keys"] for v in tool.values()))
    (D / "input_verification.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("all_ok", "checks", "max_rel_V_spread_N", "max_rel_mean_spread_N")}))


if __name__ == "__main__":
    main()
