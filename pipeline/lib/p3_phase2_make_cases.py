"""D03 Phase 2 official case bundle (B). Refuses to run officially unless results/phase2/protocol.md matches
protocol.sha256 (p3_phase2_cases.require_frozen_phase2). No W, forecast, reference or score here.

Official:  env/.venv_pilot/bin/python lib/p3_phase2_make_cases.py --out results/phase2/cases
Fixture :  env/.venv_pilot/bin/python lib/p3_phase2_make_cases.py --fixture --out /tmp/p3_phase2_fixture_cases

Writes tf_inputs/{case}.npz (truth-free, pilot layout), truth/{case}.npz (evaluation only), derived_manifest.json
(physical/derived per-case record incl. census, V, means, q0, SR, pi_r), tool_inputs_H10.npz / tool_inputs_H30.npz
(A schema via lib/p3_phase2_export.py, full-grid validated, origin link enforced), generation_checks.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_phase2_cases as pc  # noqa: E402
import p3_phase2_export as ex  # noqa: E402
from p3_make_cases import save_tf, save_truth, load_tf_input, TRUTH_KEYS  # noqa: E402  (frozen pilot serializers)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.fixture:
        if "/tmp/" not in str(a.out.resolve()) and "fixture" not in str(a.out):
            raise PermissionError("fixture output must be a temporary or fixture path")
        digest = "f" * 64
    else:
        digest = pc.require_frozen_phase2()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    (out / "tf_inputs").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    cases = list(pc.all_cases(fixture=a.fixture))
    man = []
    for c in cases:
        pt, pr = out / "tf_inputs" / f"{c['case_id']}.npz", out / "truth" / f"{c['case_id']}.npz"
        save_tf(pt, c, digest)
        save_truth(pr, c, digest)
        back = load_tf_input(pt)
        with np.load(pt, allow_pickle=False) as z:
            leak = sorted(set(z.files) & TRUTH_KEYS)
        assert not leak and np.array_equal(back["pumping_context"], c["tf_input"]["pumping_context"])
        d = {k: v for k, v in c["derived"].items()}
        d.update(tf_input=str(pt.relative_to(out)), tf_sha256=sha(pt), truth=str(pr.relative_to(out)), truth_sha256=sha(pr))
        man.append(d)
    tool = {}
    for H in (10, 30):
        arr = ex.export_arrays(cases, H, digest, require_full_grid=True)
        p = out / f"tool_inputs_H{H}.npz"
        np.savez_compressed(p, **arr)
        with np.load(p, allow_pickle=False) as z:
            back = {k: z[k] for k in z.files}
        ex.assert_array_origin_link(back)
        ex.A.validate(back, H, require_full_grid=True)
        tool[H] = dict(path=p.name, sha256=sha(p), rows=int(len(back["query_id"])), pairs=len(set(back["pair_id"])), cases=len(set(back["case_id"])))
    json.dump(man, open(out / "derived_manifest.json", "w"), ensure_ascii=False, indent=1, default=float)
    checks = dict(protocol_sha256=digest, fixture=a.fixture, n_cases=len(cases), unique_ids=len({c["case_id"] for c in cases}), tool=tool,
                  runtime_s=time.time() - t0, code_sha256={p: sha(ROOT / p) for p in ("lib/p3_phase2_cases.py", "lib/p3_phase2_export.py",
                                                                                     "lib/p3_phase2_make_cases.py", "lib/p3_phase2_physmap.py")})
    checks["all_ok"] = bool(checks["n_cases"] == checks["unique_ids"] == 540 and all(v["rows"] == 2160 and v["pairs"] == 1080 and v["cases"] == 540 for v in tool.values()))
    json.dump(checks, open(out / "generation_checks.json", "w"), indent=1)
    print(json.dumps(checks))
    return 0 if checks["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
