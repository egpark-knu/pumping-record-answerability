"""D03 Phase 2 exporter (B): case dicts from lib/p3_phase2_cases.py -> the A-validated tool NPZ schema
(lib/p3_phase2_timesfm.INPUT_KEYS). A's generic adapter/validator is used unchanged; the D03 scenario contract
(origin link) is enforced HERE, in the producer, because the generic validator accepts a positive constant future
that differs from the last observed rate (negative fixture in lib/tests/p3_phase2_export_tests.py).

Origin link (per case, per pair), with q_last = pumping_context[1023]:
  P1_continue_vs_stop : a = q_last (all H days), b = 0
  P2_current_vs_1p5x  : a = q_last,              b = 1.5 * q_last
Exact equality (np.array_equal) is required; any mismatch raises OriginLinkError.

Stable IDs: case_id  r{rr}_{storage}_T{T}_rho{rho}_N{N}
            pair_id  {case_id}__{pair}
            query_id {pair_id}__{a|b}__H{H}
Per horizon: 540 cases, 1,080 pairs, 2,160 queries. Metadata carries physical/derived LABELS only (never covariates).
"""
from __future__ import annotations

import json

import numpy as np

import p3_phase2_timesfm as A  # A's interface, imported unchanged
from p3_phase2_cases import CTX, PAIRS


class OriginLinkError(ValueError):
    pass


def forced_futures(q_last: float, H: int) -> dict:
    return {"P1_continue_vs_stop_a": np.full(H, q_last), "P1_continue_vs_stop_b": np.zeros(H),
            "P2_current_vs_1p5x_a": np.full(H, q_last), "P2_current_vs_1p5x_b": np.full(H, 1.5 * q_last)}


def assert_case_origin_link(case: dict, H: int = 30):
    ti = case["tf_input"]
    q_last = float(ti["pumping_context"][CTX - 1])
    if not q_last > 0:
        raise OriginLinkError(f"{case['case_id']}: last observed rate must be positive")
    want = forced_futures(q_last, H)
    for k, v in want.items():
        got = np.asarray(ti["future_Q"][k][:H], float)
        if got.shape != v.shape or not np.array_equal(got, v):
            raise OriginLinkError(f"{case['case_id']}: future {k} does not equal the origin-linked schedule (q_last={q_last})")


def tool_rows(case: dict, H: int) -> list[dict]:
    """Producer: builds the four queries of a case, forcing the future from the observed last rate."""
    assert_case_origin_link(case, H)
    ti = case["tf_input"]
    ctx = np.asarray(ti["pumping_context"], float)
    fut = forced_futures(float(ctx[CTX - 1]), H)
    rows = []
    for p in PAIRS:
        meta = dict(case["meta"], schedule_pair=p)
        mj = json.dumps(meta, sort_keys=True, ensure_ascii=False)
        pid = f"{case['case_id']}__{p}"
        for s in ("a", "b"):
            rows.append(dict(query_id=f"{pid}__{s}__H{H}", case_id=case["case_id"], pair_id=pid, schedule_id=s, metadata_json=mj,
                             head=np.asarray(ti["head_context"], float).copy(), pumping=np.concatenate([ctx, fut[f"{p}_{s}"]]),
                             rainfall=np.asarray(ti["rain"][:CTX + H], float).copy()))
    return rows


def assert_array_origin_link(arr: dict):
    """Independent array-level check on an exported NPZ dict (rows as serialized)."""
    n = len(arr["query_id"])
    by_pair = {}
    for i in range(n):
        by_pair.setdefault(str(arr["pair_id"][i]), {})[str(arr["schedule_id"][i])] = i
    for pid, ix in by_pair.items():
        a, b = ix["a"], ix["b"]
        pa, pb = np.asarray(arr["pumping"][a], float), np.asarray(arr["pumping"][b], float)
        q_last = pa[CTX - 1]
        if not np.array_equal(pa[:CTX], pb[:CTX]):
            raise OriginLinkError(f"{pid}: history differs between a and b")
        if not (q_last > 0 and np.all(pa[CTX:] == q_last)):
            raise OriginLinkError(f"{pid}: schedule a future must equal the last observed rate {q_last}")
        want_b = np.zeros_like(pa[CTX:]) if pid.endswith(PAIRS[0]) else 1.5 * pa[CTX:]
        if not pid.endswith(PAIRS) or not np.array_equal(pb[CTX:], want_b):
            raise OriginLinkError(f"{pid}: schedule b future violates stop / 1.5x contract")


def export_arrays(cases: list[dict], H: int, protocol_sha256: str, require_full_grid: bool = False) -> dict:
    rows = [r for c in cases for r in tool_rows(c, H)]
    out = {k: np.stack([r[k] for r in rows]) for k in ("head", "pumping", "rainfall")}
    for k in ("query_id", "case_id", "pair_id", "schedule_id", "metadata_json"):
        out[k] = np.array([r[k] for r in rows], dtype=str)
    out["protocol_sha256"] = np.array(protocol_sha256)
    assert_array_origin_link(out)
    A.validate(out, H, require_full_grid=require_full_grid)
    return out
