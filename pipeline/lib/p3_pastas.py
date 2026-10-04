"""Structurally matched transfer-function reference (Pastas 2.0.0 machinery) fitted on the
same 1024-day window. Presented as a reference with the declared response structure
(LATEST_STEERING 2026-09-30; phase0_peer_check.md correction 3), not as a competitor.

- natural: GammaReservoirP3 = daily gamma-CDF recharge blocks into the exact daily reservoir
  (same discrete kernel as generator and W family; parameters unknown and fitted)
- pumping: HantushExactP3 = exact Hantush-Jacob step (p3_kernels), numerically equal to
  Pastas Hantush(quad=True).quad_step (checked in tests) but vectorised
- initial state: the declared observed boundary via Pastas fill_before (rain: context mean
  = Pastas "mean"; well: first observed rate = "bfill"), plus free nuisance amplitudes
  eta_nat * exp(-(n+1)/tau) and eta_pump * (1 - S_H(n+1)) inside the stress models
  (StressModelIS), so the reference shares the W family's initial-state treatment
- ArNoiseModel; LeastSquares; declared 12-start multistart on (a, b); warm-up 3650 d
- paired E from JOINT parameter draws (get_parameter_sample(name=None)), pumping
  parameters extracted by name; Gaussian/local-covariance approximation stated.
No truth, hydraulic constant or transformed pumping covariate is supplied.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pastas as ps
from pandas import DataFrame, Series

from p3_kernels import hantush_unit_step, natural_kernel

CTX, HMAX = 1024, 30
WARMUP = 3650.0
FAMILY = dict(a=(1.0, 10 ** 3.5), b=(1e-4, 25.0), n=(0.5, 5.0), theta=(1.0, 100.0), tau=(5.0, 500.0))
HANTUSH_STARTS = [(a, b) for a in (10.0, 100.0, 1000.0) for b in (1e-3, 1e-2, 1e-1, 1.0)]  # declared, truth-free
logging.getLogger("pastas").setLevel(logging.ERROR)
TMAX_BIG = 1e5  # response length is then governed by Pastas' maxtmax (simulation length); no 0.999 truncation


class HantushExactP3(ps.Hantush):
    def get_tmax(self, p, cutoff=None):
        return TMAX_BIG

    def step(self, p, dt=1.0, cutoff=None, maxtmax=None, **kwargs):
        A, a, b = p
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        return A * hantush_unit_step(a, b, t)[0]


class GammaReservoirP3(ps.rfunc.RfuncBase):
    """Parameters A (gain), n, theta (d), tau (d). Step at integer days = A * cumulative discrete kernel."""

    @property
    def nparam(self):
        return 4

    def get_init_parameters(self, name):
        return DataFrame([(1.0 / self.gain_scale_factor, 0.0, 1e3 / self.gain_scale_factor, True, name),
                          (2.0, *FAMILY["n"], True, name), (10.0, *FAMILY["theta"], True, name), (60.0, *FAMILY["tau"], True, name)],
                         index=[name + "_A", name + "_n", name + "_theta", name + "_tau"], columns=["initial", "pmin", "pmax", "vary", "name"])

    def get_tmax(self, p, cutoff=None):
        return TMAX_BIG

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1.0, cutoff=None, maxtmax=None, **kwargs):
        A, n, theta, tau = p
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        m = np.round(t).astype(int)
        if np.ndim(dt) == 0 and dt != 1.0 or np.any(np.abs(t - m) > 1e-9):
            raise ValueError("daily model only: integer-day evaluation times")
        _, Kc = natural_kernel(n, theta, tau, int(m.max()))
        return A * Kc[0, m]

    def moment(self, p, order, method="discrete", dt=1.0):
        raise NotImplementedError

    @staticmethod
    def impulse(t, p):
        raise NotImplementedError


class StressModelIS(ps.StressModel):
    """StressModel plus one initial-state nuisance amplitude eta on a basis starting at the window start t0:
    kind 'exp' : eta * exp(-(n+1)/tau)   (tau = last rfunc parameter, GammaReservoirP3)
    kind 'tail': eta * (1 - S(n+1)/gain) (unit Hantush tail, HantushExactP3)
    Zero before t0 (warm-up is not fitted)."""

    def __init__(self, model, stress, rfunc, name, up=True, settings=None, kind="exp", t0=None, eta_bound=1.0):
        self._is_kind, self._t0, self._eta_bound = kind, pd.Timestamp(t0), float(eta_bound)
        super().__init__(model, stress, rfunc, name, up=up, settings=settings)

    def set_init_parameters(self):
        p = self.rfunc.get_init_parameters(self.name)
        p.loc[self.name + "_eta"] = (0.0, -self._eta_bound, self._eta_bound, True, self.name)
        self.parameters = p

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1.0):
        p = np.asarray(p, float)
        h = super().simulate(p[:-1], tmin=tmin, tmax=tmax, freq=freq, dt=dt)
        n = ((h.index - self._t0) / pd.Timedelta(days=1)).values
        m = n >= 0
        basis = np.zeros(h.size)
        if m.any():
            tt = n[m] + 1.0
            if self._is_kind == "exp":
                basis[m] = np.exp(-tt / p[-2])
            else:
                A, a, b = p[:3]
                basis[m] = 1.0 - hantush_unit_step(a, b, tt)[0]
        return Series(h.values + p[-1] * basis, index=h.index, name=self.name)


def well_settings():
    s = dict(ps.timeseries.settings["well"])
    s["fill_before"] = "bfill"  # declared boundary: first observed rate persists backwards
    return s


def _series(tf_input):
    dates = pd.to_datetime(tf_input["dates"][:CTX])
    h = Series(np.asarray(tf_input["head_context"], float), dates, name="head")
    P = Series(np.asarray(tf_input["rain"][:CTX], float), dates, name="rain")
    Q = Series(np.asarray(tf_input["pumping_context"], float), dates, name="Q")
    return h, P, Q


def build_model(h, P, Q):
    eb = 10.0 * float(np.ptp(h.values))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ml = ps.Model(h, name="tf_reference")
        StressModelIS(ml, P, GammaReservoirP3(), name="recharge", up=True, settings="prec", kind="exp", t0=h.index[0], eta_bound=eb)
        StressModelIS(ml, Q, HantushExactP3(), name="well", up=False, settings=well_settings(), kind="tail", t0=h.index[0], eta_bound=eb)
        ps.ArNoiseModel(ml)
        ps.solver.LeastSquares(ml)
    ml.set_parameter("well_a", pmin=FAMILY["a"][0], pmax=FAMILY["a"][1])
    ml.set_parameter("well_b", pmin=FAMILY["b"][0], pmax=FAMILY["b"][1])
    return ml


def multistart_solve(h, P, Q):
    best, log = None, []
    for a0, b0 in HANTUSH_STARTS:
        ml = build_model(h, P, Q)
        ml.set_parameter("well_a", initial=a0)
        ml.set_parameter("well_b", initial=b0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml.solve(report=False, warmup=WARMUP)
            obj = float(np.sum(ml.noise() ** 2))
        except Exception as err:  # recorded, not hidden
            log.append(dict(a0=a0, b0=b0, error=repr(err)))
            continue
        log.append(dict(a0=a0, b0=b0, obj=obj, a=float(ml.parameters.loc["well_a", "optimal"]), b=float(ml.parameters.loc["well_b", "optimal"])))
        if best is None or obj < best[0]:
            best = (obj, ml)
    if best is None:
        raise RuntimeError("all reference starts failed: " + str(log))
    return best[1], log


def unit_E(p_well, dQ):
    """E(k) for a well parameter vector [A, a, b] (Pastas sign: A < 0 for up=False): contribution difference."""
    A, a, b = p_well[:3]
    S = hantush_unit_step(a, b, np.arange(1, HMAX + 1, dtype=float))[0]
    B = np.diff(np.concatenate([[0.0], S]))
    return A * np.convolve(dQ, B)[:HMAX]


def fit_and_propagate(tf_input, seed, n_draws=1000, max_iter=50):
    h, P, Q = _series(tf_input)
    ml, starts = multistart_solve(h, P, Q)
    names = list(ml.parameters.index)
    iw = [names.index(k) for k in ("well_A", "well_a", "well_b")]
    p_opt = ml.parameters["optimal"].values
    fq = tf_input["future_Q"]
    pairs = sorted({k.rsplit("_", 1)[0] for k in fq})
    out = dict(pastas_version=ps.__version__, optimal_params=ml.parameters["optimal"].to_dict(), param_names=names,
               fit_rmse_m=float(np.sqrt(np.mean(ml.residuals() ** 2))), multistart=starts, pairs={})
    try:
        draws = ml.solver.get_parameter_sample(name=None, n=n_draws, max_iter=max_iter, seed=seed)
    except RuntimeError as err:
        draws = np.empty((0, len(names))); out["sample_error"] = repr(err)
    out["n_param_draws_accepted"] = int(draws.shape[0])
    out["draws_below_requested"] = bool(draws.shape[0] < n_draws)
    for p in pairs:
        dQ = np.asarray(fq[f"{p}_a"], float) - np.asarray(fq[f"{p}_b"], float)
        rec = dict(E_point=unit_E(p_opt[iw], dQ).tolist())
        if draws.shape[0]:
            Es = np.array([unit_E(d[iw], dQ) for d in draws])
            q10, q90 = np.quantile(Es, [0.1, 0.9], axis=0)
            rec.update(E_q10=q10.tolist(), E_q90=q90.tolist(), w_TF=(q90 - q10).tolist())
        out["pairs"][p] = rec
    out["_model"] = ml
    return out


def simulate_consistency(tf_input, fit):
    """E from unit_E equals the difference of two well-stress-model simulations whose Q is extended by
    each future schedule (same parameters incl. eta). Returns max |diff| (m)."""
    ml = fit["_model"]
    p_well = ml.get_parameters("well")
    h, P, Q = _series(tf_input)
    fut_idx = pd.date_range(Q.index[-1] + pd.Timedelta(days=1), periods=HMAX, freq="D")
    fq = tf_input["future_Q"]
    worst = 0.0
    for p, rec in fit["pairs"].items():
        sims = {}
        for s in ("a", "b"):
            Qx = pd.concat([Q, Series(np.asarray(fq[f"{p}_{s}"], float), fut_idx)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m2 = ps.Model(h, name="chk")
                sm = StressModelIS(m2, Qx, HantushExactP3(), name="well", up=False, settings=well_settings(), kind="tail",
                                   t0=h.index[0], eta_bound=1e9)
            sims[s] = sm.simulate(p_well, tmin=Q.index[0] - pd.Timedelta(days=WARMUP), tmax=fut_idx[-1]).loc[fut_idx].values
        worst = max(worst, float(np.max(np.abs((sims["a"] - sims["b"]) - np.asarray(rec["E_point"])))))
    return worst
