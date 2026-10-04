"""Build the figure-contract table cell_metrics.csv from ACTUAL outputs only (B): official W curves (results/phase2/wb),
truth evaluation, and A's TimesFM archives results/phase2/tools/timesfm_{raw,exp10,exp30,exp90}_H{10,30}.npz.

Refuses (no partial or invented values) unless all 8 tool archives exist, carry the frozen protocol hash, reference the
official tool_inputs_H{H}.npz by SHA-256 (key input_sha256), cover exactly the 1,080 official pair_ids, and have finite-typed
E_point [1080, H]. Expected A keys (pilot layout): pair_id [P], E_point [P, H], protocol_sha256, input_sha256; optional track.

Rows (FIGURE_INPUT_CONTRACT.json): 540 cases x 2 pairs x 4 tracks x 2 leads = 8,640 'lead' rows, plus 1,080 raw-track
'max_days_1_10' rows (W_m = max over days 1-10 of W; E_true_m/E_tool_m at the lead k* where that maximum is attained).
Tool contrasts are from the H10 request for lead 10 and from the H30 request for lead 30 (separate requests, protocol §7).
Also writes cell_denominators.json (all row-metric denominators per layer x pair x track x lead).

Usage: env/.venv_pilot/bin/python lib/p3_phase2_cell_metrics.py [--tools DIR] [--out DIR] [--provenance official_analysis]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "lib"))
import p3_phase2_metrics as mt  # noqa: E402

P2 = ROOT / "results/phase2"
TRACKS = ("raw", "exp10", "exp30", "exp90")
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_tools(tdir: Path, digest: str) -> dict:
    out = {}
    for tr in TRACKS:
        for H in (10, 30):
            p = tdir / f"timesfm_{tr}_H{H}.npz"
            if not p.exists():
                raise FileNotFoundError(f"missing actual tool archive {p}; cell_metrics is not written")
            with np.load(p, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            for k in ("pair_id", "E_point", "protocol_sha256", "input_sha256"):
                if k not in d:
                    raise KeyError(f"{p.name}: required key {k} absent")
            if str(d["protocol_sha256"]) != digest:
                raise ValueError(f"{p.name}: protocol hash differs")
            if str(d["input_sha256"]) != sha(P2 / f"cases/tool_inputs_H{H}.npz"):
                raise ValueError(f"{p.name}: not produced from the official tool_inputs_H{H}.npz")
            if "track" in d and str(d["track"]) != tr:
                raise ValueError(f"{p.name}: track label {d['track']} != {tr}")
            E = np.asarray(d["E_point"], float)
            if E.shape != (len(d["pair_id"]), H) or len(set(d["pair_id"].astype(str))) != 1080:
                raise ValueError(f"{p.name}: E_point shape {E.shape} / pairs {len(set(d['pair_id']))}")
            out[(tr, H)] = dict(zip(d["pair_id"].astype(str), E), _sha=sha(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", type=Path, default=P2 / "tools")
    ap.add_argument("--out", type=Path, default=P2 / "analysis")
    ap.add_argument("--provenance", default="official_analysis", choices=["official_analysis", "synthetic_fixture"])
    a = ap.parse_args()
    if a.provenance == "synthetic_fixture" and "fixture" not in str(a.out) and "/tmp/" not in str(a.out):
        raise PermissionError("synthetic output must go under a fixture or /tmp path")
    digest = (P2 / "protocol.sha256").read_text().split()[0]
    wbm = json.load(open(P2 / "wb/wb_manifest.json"))
    if not wbm["complete"]:
        raise RuntimeError("W/reference/truth collection is not complete")
    tools = load_tools(a.tools, digest)
    man = json.load(open(P2 / "cases/derived_manifest.json"))
    rows, groups = [], defaultdict(list)
    for m in man:
        cid = m["case_id"]
        W = json.loads((P2 / "wb/W" / f"{cid}.json").read_text())
        T = json.loads((P2 / "wb/truth_eval" / f"{cid}.json").read_text())
        for p in PAIRS:
            pid = f"{cid}__{p}"
            sup, inf = np.array(W["envelope"][p]["sup"]), np.array(W["envelope"][p]["inf"])
            Wc = sup - inf
            Et = np.array(T["E_true"][p])
            for tr in TRACKS:
                for k in (10, 30):
                    Eo = float(tools[(tr, k)][pid][k - 1])
                    r = mt.cell_metrics_row(m, p, tr, "lead", k, float(Wc[k - 1]), W["W_status"], float(Et[k - 1]), Eo, float(inf[k - 1]), float(sup[k - 1]), a.provenance)
                    rows.append(r)
                    groups[(m["sid"], p, tr, k)].append(mt.row_metrics(Et[k - 1], Eo, Wc[k - 1], inf[k - 1], sup[k - 1]))
            ks = int(np.argmax(Wc[:10]))
            Eo = float(tools[("raw", 10)][pid][ks])
            rows.append(mt.cell_metrics_row(m, p, "raw", "max_days_1_10", None, float(Wc[:10].max()), W["W_status"], float(Et[ks]), Eo, float(inf[ks]), float(sup[ks]), a.provenance))
    assert sum(r["quantity"] == "lead" for r in rows) == 8640 and sum(r["quantity"] == "max_days_1_10" for r in rows) == 1080
    a.out.mkdir(parents=True, exist_ok=True)
    mt.write_cell_metrics(rows, a.out / "cell_metrics.csv")
    den = {"|".join(map(str, k)): mt.denominators(v) for k, v in groups.items()}
    (a.out / "cell_denominators.json").write_text(json.dumps(dict(protocol_sha256=digest, provenance=a.provenance,
                                                                  tool_archives={f"{t}_H{h}": v["_sha"] for (t, h), v in tools.items()},
                                                                  wb_manifest_sha256=sha(P2 / "wb/wb_manifest.json"), denominators=den), indent=1))
    print(f"rows={len(rows)} -> {a.out / 'cell_metrics.csv'}")


if __name__ == "__main__":
    main()
