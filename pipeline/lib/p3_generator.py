"""Tier A case generator for the D02 pilot (Paper 3). Values follow
results/pilot/protocol_draft.md; nothing here is run on pilot cells until the
protocol hash is frozen (enforced by `require_frozen`).

Separation of information:
  tool_rows  -> TimesFM-3 adapter NPZ (A schema): 1024-d head, 1024+H stresses
  tf_input   -> W family and Pastas: the same 1024-d window + 30 future days only
  truth      -> evaluation only (h*, components, E_true, delta, true parameters)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from p3_kernels import hantush_block, natural_kernel, causal_conv

CTX, HMAX, GEN_WARMUP = 1024, 30, 0  # no real pre-context rain in truth (peer default, same-record rule)
REST_END = 874  # context day index at which pumping resumes after the rest
CL_DIR = Path(__import__("os").environ.get("KMA_CLIMATE_DIR", "climate"))


@dataclass(frozen=True)
class Design:
    rest_ratios: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    signal_ratios: tuple = (0.5, 2.0)
    pi_layers: tuple = (0.5, 5.0)
    t95: float = 60.0
    kernels: dict = field(default_factory=lambda: {
        0.5: dict(a=53.48475127106841, b=0.009348458918055505, T=50.0, S=1e-3, r=316.22776601683796),
        5.0: dict(a=32.77609031089729, b=0.15255022647828184, T=50.0, S=1e-3, r=1000.0)})
    f_recharge: float = 0.15
    nat_ranges: dict = field(default_factory=lambda: dict(n=(1.5, 3.0), theta=(5.0, 20.0), tau=(30.0, 120.0), S_eff=(0.03, 0.10)))
    phi: float = 0.8
    sigma_eps: float = 0.02
    base_multipliers: tuple = (0.8, 1.0, 1.2)
    seg_median: float = 20.0
    seg_sigma_ln: float = 0.6
    seg_bounds: tuple = (3, 90)
    delta_rel: float = 0.25


def a_true_gain(T, b):
    from scipy.special import k0
    return k0(2.0 * np.sqrt(b)) / (2.0 * np.pi * T)  # m per (m3/d)


def load_rain(stem: str, expected_sha_prefix: str | None = None) -> pd.Series:
    path = CL_DIR / f"{stem}_CL.txt"
    if not path.exists():  # NFD file names on macOS
        import unicodedata
        path = CL_DIR / (unicodedata.normalize("NFD", stem) + "_CL.txt")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha_prefix and not sha.startswith(expected_sha_prefix):
        raise ValueError(f"rain file hash mismatch for {stem}: {sha[:12]}")
    df = pd.read_csv(path, sep="\t", dtype={"Date": str})
    s = pd.Series(df["RAIN"].astype(float).values, index=pd.to_datetime(df["Date"], format="%Y%m%d"), name=stem)
    if s.isna().any():
        raise ValueError(f"{stem}: RAIN has missing values; eligibility rule violated")
    s.attrs["sha256"] = sha
    s.attrs["path"] = str(path)
    return s


def base_schedule(seed: int, d: Design) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.empty(CTX)
    i = 0
    while i < CTX:
        L = int(np.clip(round(d.seg_median * np.exp(d.seg_sigma_ln * rng.standard_normal())), *d.seg_bounds))
        m[i:i + L] = rng.choice(d.base_multipliers)
        i += L
    return m


def natural_draw(seed: int, d: Design) -> dict:
    rng = np.random.default_rng(seed)
    r = d.nat_ranges
    return dict(n=rng.uniform(*r["n"]), theta=rng.uniform(*r["theta"]), tau=rng.uniform(*r["tau"]), S_eff=rng.uniform(*r["S_eff"]))


def ar1_noise(seed: int, n: int, d: Design) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    e[0] = d.sigma_eps * rng.standard_normal()
    s = d.sigma_eps * np.sqrt(1 - d.phi ** 2)
    eta = s * rng.standard_normal(n)
    for i in range(1, n):
        e[i] = d.phi * e[i - 1] + eta[i]
    return e


def realization_background(real: dict, rain: pd.Series, d: Design) -> dict:
    """Everything shared by the 20 cells of a realization (rain, h_nat, noise, base multipliers)."""
    c0 = pd.Timestamp(real["context_start"])
    start = c0 - pd.Timedelta(days=GEN_WARMUP)
    end = c0 + pd.Timedelta(days=CTX + HMAX - 1)
    P = rain.loc[start:end].values
    if P.size != GEN_WARMUP + CTX + HMAX:
        raise ValueError("rain window incomplete for realization %s" % real["realization"])
    nat = natural_draw(real["seed_natural"], d)
    K, Kc = natural_kernel(nat["n"], nat["theta"], nat["tau"], P.size)
    gain = d.f_recharge * nat["tau"] / nat["S_eff"] / 1000.0  # m per (mm/d)
    Pbar = P[:CTX].mean()  # declared boundary: steady filter/reservoir under the CONTEXT-ONLY mean rain
    pre = Pbar * (1.0 - Kc[0, 1:])
    h_nat = gain * (np.convolve(P, K[0])[:P.size] + pre)
    eps = ar1_noise(real["seed_noise"], P.size, d)
    w = slice(GEN_WARMUP, GEN_WARMUP + CTX)
    sigma_bg = float(np.std(h_nat[w] + eps[w]))
    return dict(P=P, Pbar=Pbar, dates=pd.date_range(start, end, freq="D"), nat=nat, nat_gain=gain, h_nat=h_nat, eps=eps,
                sigma_bg=sigma_bg, mult=base_schedule(real["seed_schedule"], d), real=real)


def case_id(r, pir, sr, rho):
    return f"r{r:02d}_pi{pir:g}_sr{sr:g}_rho{rho:g}"


def build_case(bg: dict, rho: float, sr: float, pir: float, d: Design) -> dict:
    kp = d.kernels[pir]
    A = a_true_gain(kp["T"], kp["b"])
    Qref = sr * bg["sigma_bg"] / A
    R = int(round(rho * d.t95))
    Qw = Qref * bg["mult"].copy()
    Qw[REST_END - R:REST_END] = 0.0
    q0 = Qw[CTX - 1]
    assert q0 > 0 and (Qw == 0).sum() == R
    B, S = hantush_block(kp["a"], kp["b"], CTX + HMAX)
    B, S = B[0], S[0]
    fut = {"P1_continue_vs_stop": (np.full(HMAX, q0), np.zeros(HMAX)),
           "P2_current_vs_1p5x": (np.full(HMAX, q0), np.full(HMAX, 1.5 * q0))}

    def h_pump(Qfut):
        Q = np.concatenate([Qw, Qfut])
        # constant pre-context operation at Qref (generator design; unknown to W/Pastas)
        return A * (Qref * (1.0 - S[1:CTX + HMAX + 1]) + np.convolve(Q, B)[:CTX + HMAX])  # direct: history bit-identical

    w = slice(GEN_WARMUP, GEN_WARMUP + CTX + HMAX)
    h_nat = bg["h_nat"][w]
    eps = bg["eps"][w]
    hp_a = h_pump(fut["P1_continue_vs_stop"][0])
    h_star_a = h_nat - hp_a
    h_obs_ctx = (h_star_a + eps)[:CTX]
    E_true, delta, sims = {}, {}, {}
    for p, (qa, qb) in fut.items():
        ha = h_nat - h_pump(qa)
        hb = h_nat - h_pump(qb)
        assert np.array_equal(ha[:CTX], h_star_a[:CTX]) and np.array_equal(hb[:CTX], h_star_a[:CTX])
        E_true[p] = (ha - hb)[CTX:]
        delta[p] = np.maximum(d.delta_rel * np.abs(E_true[p]), d.sigma_eps)
        sims[p] = (ha, hb)
    real = bg["real"]
    cid = case_id(real["realization"], pir, sr, rho)
    meta = dict(origin_date=str(bg["dates"][GEN_WARMUP + CTX].date()), rain_site=real["site_stem"], realization=int(real["realization"]),
                rest_ratio=float(rho), signal_ratio=float(sr), pi_layer=float(pir), head_unit="m", pumping_unit="m3/d", rainfall_unit="mm/d",
                daily_alignment="head at end of day; stresses constant per day; lead 1 = first forecast day")
    P_ctx_fut = bg["P"][GEN_WARMUP:]
    tf_input = dict(case_id=cid, head_context=h_obs_ctx, rain=P_ctx_fut, pumping_context=Qw,
                    future_Q={f"{p}_{s}": q for p, (qa, qb) in fut.items() for s, q in (("a", qa), ("b", qb))},
                    dates=np.array([str(x.date()) for x in bg["dates"][GEN_WARMUP:]]), meta=meta)
    truth = dict(case_id=cid, h_star_a=h_star_a, h_nat=h_nat, h_pump_a=hp_a, eps=eps, E_true=E_true, delta=delta,
                 theta_true=dict(a=kp["a"], b=kp["b"], A=A, Q_ref=Qref, q0=q0, R_days=R, Q_start=float(Qw[0]),
                                 eta_pump_true=float(A * (Qref - Qw[0])), eta_nat_true=0.0, sigma_bg=bg["sigma_bg"], nat_gain=bg["nat_gain"], **bg["nat"],
                                 T=kp["T"], S=kp["S"], r=kp["r"]))
    return dict(case_id=cid, meta=meta, tf_input=tf_input, truth=truth, sims=sims)


def tool_rows(case: dict, horizon: int) -> list[dict]:
    """Rows in A's adapter schema (lib/timesfm_pilot_adapter.py validate())."""
    ti = case["tf_input"]
    rows = []
    for p in ("P1_continue_vs_stop", "P2_current_vs_1p5x"):
        meta = dict(case["meta"], schedule_pair=p)
        mj = json.dumps(meta, sort_keys=True, ensure_ascii=False)
        pid = f"{case['case_id']}__{p}"
        for s in ("a", "b"):
            rows.append(dict(query_id=f"{pid}__{s}__H{horizon}", case_id=case["case_id"], pair_id=pid, schedule_id=s, metadata_json=mj,
                             head=ti["head_context"].copy(),
                             pumping=np.concatenate([ti["pumping_context"], ti["future_Q"][f"{p}_{s}"][:horizon]]),
                             rainfall=ti["rain"][:CTX + horizon].copy()))
    return rows


def tool_npz_arrays(rows: list[dict], protocol_sha256: str) -> dict:
    out = {k: np.stack([r[k] for r in rows]) for k in ("head", "pumping", "rainfall")}
    for k in ("query_id", "case_id", "pair_id", "schedule_id", "metadata_json"):
        out[k] = np.array([r[k] for r in rows], dtype=str)
    out["protocol_sha256"] = np.array(protocol_sha256)
    return out


def require_frozen(protocol_path: Path, sha_path: Path) -> str:
    """Pilot generation gate: protocol.md must exist and match protocol.sha256."""
    if not protocol_path.exists() or not sha_path.exists():
        raise PermissionError("protocol not frozen: pilot-cell generation is not authorised")
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if sha_path.read_text().split()[0] != digest:
        raise PermissionError("protocol hash mismatch")
    return digest
