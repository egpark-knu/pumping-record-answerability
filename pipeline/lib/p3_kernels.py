"""Response kernels for the D02 Tier A pilot (Paper 3).

Exact Hantush leaky-well step (Hantush & Jacob 1955, time form) in the Pastas
(a, b) parameterisation, verified against scipy quadrature of W(u, r/B) in
results/pilot/phase0_checks/hantush_kernel_check.py:
    impulse theta(t) = A exp(-t/a - a b / t) / (2 t K0(2 sqrt b)),
    t_r = a b = r^2 S / (4T),  rho = r/lambda = 2 sqrt b,  gain = A.
Natural component: gamma recharge filter (block, CDF differences) into a
daily-exact linear reservoir.

Daily convention (shared by generator, W family and Pastas tool): stress on
day j is constant over that day; head is read at the end of day n; block
response B[m] = S(m+1) - S(m), m >= 0, S(0) = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import kve, gammainc

N_Z = 4000  # log-time nodes for cumulative integration


def _log_2k0(rho):
    # log(2 K0(rho)) without underflow: K0(x) = kve(0,x) * exp(-x)
    return np.log(2.0 * kve(0, rho)) - rho


def hantush_unit_step(a, b, t):
    """Unit-gain step response S(t)/A for arrays a, b (shape [M]) at times t (shape [T]).

    Integrates tau*theta(tau)/A over z = ln tau on a dense grid from ln(t_r) - 9
    (integrand < exp(-e^9) there) to ln(max t); exact to the quadrature error,
    checked in tests against scipy.integrate.quad of W(u, rho).
    """
    a = np.atleast_1d(np.asarray(a, float))[:, None]
    b = np.atleast_1d(np.asarray(b, float))[:, None]
    t = np.asarray(t, float)
    tr = a * b
    zlo = np.log(tr) - 9.0
    zhi = np.log(t.max()) + 1e-9
    zhi = np.maximum(zhi, zlo + 1.0)
    u = np.linspace(0.0, 1.0, N_Z)[None, :]
    z = zlo + (zhi - zlo) * u
    tau = np.exp(z)
    logf = -tau / a - tr / tau - _log_2k0(2.0 * np.sqrt(b))
    f = np.exp(logf)
    dz = (zhi - zlo) / (N_Z - 1)
    cum = np.concatenate([np.zeros((f.shape[0], 1)), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dz, axis=1)], axis=1)
    lt = np.log(np.maximum(t, 1e-300))
    out = np.empty((a.shape[0], t.size))
    for i in range(a.shape[0]):
        out[i] = np.interp(lt, z[i], cum[i], left=0.0, right=cum[i, -1])
    out[:, t <= 0] = 0.0
    return out


def hantush_t_frac(a, b, frac=0.95, tmax=20000.0):
    """Time at which the unit step reaches `frac` (NaN if beyond tmax)."""
    t = np.unique(np.concatenate([np.geomspace(1e-3, tmax, 6000)]))
    s = hantush_unit_step(a, b, t)
    res = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        k = np.searchsorted(s[i], frac)
        if 0 < k < t.size:
            res[i] = np.exp(np.interp(frac, s[i, k - 1:k + 1], np.log(t[k - 1:k + 1])))
    return res


def hantush_block(a, b, length):
    """Unit-gain daily block response B[m], m = 0..length-1, and step S(m), m = 0..length."""
    tt = np.arange(0, length + 1, dtype=float)
    S = hantush_unit_step(a, b, np.maximum(tt, 1e-12))
    S[:, 0] = 0.0
    return np.diff(S, axis=1), S


def natural_kernel(n_shape, theta, tau, length):
    """Unit-gain natural response K[m] (sum -> 1 as length -> inf) and its cumulative.

    Recharge from rain on day j: gamma block g[m] = F(m+1) - F(m).
    Reservoir: h[n] = e^{-1/tau} h[n-1] + (1 - e^{-1/tau}) R[n] (unit gain),
    i.e. impulse r[m] = (1 - e^{-1/tau}) e^{-m/tau}.
    Returns K [M, length] and cumulative Kc [M, length+1] (Kc[:,0]=0).
    """
    n_shape = np.atleast_1d(np.asarray(n_shape, float))[:, None]
    theta = np.atleast_1d(np.asarray(theta, float))[:, None]
    tau = np.atleast_1d(np.asarray(tau, float))[:, None]
    m = np.arange(length + 1, dtype=float)[None, :]
    F = gammainc(n_shape, m / theta)
    g = np.diff(F, axis=1)
    q = np.exp(-1.0 / tau)
    r = (1.0 - q) * q ** m[:, :length]
    L = 1 << int(np.ceil(np.log2(2 * length)))
    K = np.fft.irfft(np.fft.rfft(g, L) * np.fft.rfft(r, L), L)[:, :length]
    Kc = np.concatenate([np.zeros((K.shape[0], 1)), np.cumsum(K, axis=1)], axis=1)
    return K, Kc


def causal_conv(x, k):
    """y[n] = sum_{m=0}^{n} k[..., m] x[n-m], for x [T] and k [M, >=T]; returns [M, T]."""
    x = np.asarray(x, float)
    T = x.size
    k = np.atleast_2d(k)[:, :T]
    L = 1 << int(np.ceil(np.log2(2 * T)))
    return np.fft.irfft(np.fft.rfft(k, L) * np.fft.rfft(x, L)[None, :], L)[:, :T]
