"""Reuse inventory for the D02 W repair v1.2 (B): every reused input/archive is hashed and tied to its original record.
Writes results/pilot/wb_repaired_v1_2/reuse_manifest.json. No computation of scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PILOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1]))) / "results/pilot"
OLD = PILOT / "wb"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = json.load(open(PILOT / "cases/observed_manifest.json"))
    oldman = json.load(open(OLD / "wb_manifest.json"))
    tfm = json.load(open(PILOT / "A_timesfm_inference_manifest.json"))
    per_case = {}; bad = []
    for m in man:
        cid = m["case_id"]
        mk = json.loads((OLD / "markers" / f"{cid}.done").read_text())
        ref_json = OLD / "reference" / f"{cid}.json"; ref_npz = OLD / "reference" / f"{cid}_draws.npz"
        tf = PILOT / "cases/tf_inputs" / f"{cid}.npz"; tr = PILOT / "cases/truth" / f"{cid}.npz"
        oldW = OLD / "W" / f"{cid}.json"
        e = dict(tf_input_sha256=sha(tf), tf_input_marker_sha256=mk["tf_input_sha256"], truth_sha256=sha(tr),
                 reference_json_sha256=sha(ref_json), reference_json_marker_sha256=mk["outputs"].get(f"wb/reference/{cid}.json"),
                 reference_draws_sha256=sha(ref_npz), reference_draws_bytes=ref_npz.stat().st_size, reference_status=mk.get("reference_status"),
                 old_W_sha256=sha(oldW), old_W_marker_sha256=mk["outputs"].get(f"wb/W/{cid}.json"),
                 reference_checks={k: v for k, v in mk.get("checks", {}).items() if k.startswith("reference")})
        e["ok"] = bool(e["tf_input_sha256"] == e["tf_input_marker_sha256"] and e["reference_json_sha256"] == e["reference_json_marker_sha256"]
                       and e["old_W_sha256"] == e["old_W_marker_sha256"] and e["reference_status"] == "ok")
        if not e["ok"]:
            bad.append(cid)
        per_case[cid] = e
    tools = {}
    for H, rec in tfm["outputs"].items():
        p = Path(rec["path"])
        tools[f"H{H}"] = dict(path=str(p.relative_to(PILOT)), sha256=sha(p), manifest_sha256=rec["sha256"], match=sha(p) == rec["sha256"])
    rt = OLD / "tables/reference_table.csv"
    out = dict(
        purpose="Inputs and archives reused unchanged by the W repair v1.2; nothing listed here is recomputed.",
        protocol_sha256=(PILOT / "protocol.sha256").read_text().split()[0],
        reused=dict(cases="cases/tf_inputs + cases/truth (frozen protocol cases, 200)",
                    reference="wb/reference/{case}.json and {case}_draws.npz: structurally matched reference fits, paired E draws "
                              "(values reused; envelope-inclusion flags are recomputed against the v1.2 envelope)",
                    timesfm="tools/timesfm_H10.npz, tools/timesfm_H30.npz (A, unchanged)",
                    cached_fit="wb/W/{case}.json tolerance (sse_min, tau, xbest): reproduced bit-identically by the v1.2 engine before reuse; "
                               "tau is locked from the original record"),
        not_reused="wb/W envelopes/W values/flags, wb/truth_eval envelope-inclusion, reference_table inclusion columns (all depend on the flawed W)",
        n_cases=len(per_case), n_ok=len(per_case) - len(bad), not_ok=bad,
        old_reference_table=dict(path=str(rt.relative_to(PILOT)), sha256=sha(rt), wb_manifest_sha256=oldman["tables"]["reference_table.csv"]["sha256"]),
        old_wb_manifest_sha256=sha(OLD / "wb_manifest.json"),
        timesfm=tools, timesfm_manifest_sha256=sha(PILOT / "A_timesfm_inference_manifest.json"),
        cases_observed_manifest_sha256=sha(PILOT / "cases/observed_manifest.json"), cases_truth_manifest_sha256=sha(PILOT / "cases/truth_manifest.json"),
        per_case=per_case)
    (PILOT / "wb_repaired_v1_2").mkdir(exist_ok=True)
    (PILOT / "wb_repaired_v1_2/reuse_manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_cases", "n_ok", "not_ok", "old_reference_table", "timesfm")}, indent=1))


if __name__ == "__main__":
    main()
