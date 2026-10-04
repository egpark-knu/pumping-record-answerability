"""Phase 0 kernel-only check (no pilot cells). Verifies Pastas 2.0.0 Hantush vs exact
leaky-well integral and solves (a,b) for declared t95 and pi_r strata."""
import json, numpy as np, pastas as ps
from scipy.integrate import quad
from scipy.special import kv, k0
from scipy.optimize import brentq

def hantush_W(u, rho):
    # Hantush-Jacob (1955) leaky well function W(u, r/B) = int_u^inf exp(-y - rho^2/(4y))/y dy, via y = e^z
    f = lambda z: np.exp(-np.exp(z) - rho**2/(4*np.exp(z)))
    lo = np.log(u); hi = max(lo, np.log(max(rho, 1e-12))) + 60
    v, _ = quad(f, lo, min(hi, 8.0), limit=500)
    return v

def exact_step_frac(t, a, b):
    # Pastas mapping (derived here, verified numerically): a = cS, b = r^2/(4 lambda^2), t_r = r^2 S/(4T) = a b,
    # u = t_r / t, rho = r/lambda = 2 sqrt(b); s(t)/gain = W(u, rho) / (2 K0(rho))
    rho = 2*np.sqrt(b)
    return hantush_W(a*b/t, rho) / (2*k0(rho))

def t_frac(a, b, frac):
    return brentq(lambda t: exact_step_frac(t, a, b) - frac, 1e-6, 1e6)

out = {"pastas_version": ps.__version__}
# 1) identity: exact gain integral = 1
out["gain_check"] = [exact_step_frac(1e7, a, b) for a, b in [(10, 0.01), (100, 1.0), (500, 0.001)]]
# 2) pastas approx vs exact on a few kernels
H = ps.Hantush()
cmp = []
for a, b in [(20, 0.025), (40, 0.1), (100, 0.005), (15, 0.33)]:
    t = np.arange(1, 400)
    s_p = ps.Hantush.numpy_step(1.0, a, b, t)
    s_e = np.array([exact_step_frac(x, a, b) for x in t])
    cmp.append({"a": a, "b": b, "max_abs_err_frac_of_gain": float(np.max(np.abs(s_p - s_e)))})
out["pastas_numpy_step_vs_exact"] = cmp
# 3) design: t95 fixed, t_r = a*b = pi_r * dt
design = {}
for t95 in [60.0]:
    for pir in [0.5, 5.0]:
        g = lambda la: t_frac(10**la, pir/10**la, 0.95) - t95
        grid = np.linspace(-0.5, 3.5, 81)
        vals = [g(x) for x in grid]
        roots = [brentq(g, grid[i], grid[i+1]) for i in range(80) if np.sign(vals[i]) != np.sign(vals[i+1])]
        sols = []
        for la in roots:
            a = 10**la; b = pir/a
            sols.append({"a": a, "b": b, "t_r": a*b, "t50": t_frac(a, b, 0.5), "t95": t_frac(a, b, 0.95),
                         "t99": t_frac(a, b, 0.99), "frac_at_1d": exact_step_frac(1, a, b),
                         "frac_at_10d": exact_step_frac(10, a, b), "frac_at_30d": exact_step_frac(30, a, b),
                         "pastas_tmax95_approx": float(H.get_tmax([1.0, a, b], cutoff=0.95))})
        design[f"t95={t95},pi_r={pir}"] = sols
out["design"] = design
print(json.dumps(out, indent=1))
json.dump(out, open("results/pilot/phase0_checks/hantush_kernel_check.json", "w"), indent=1)
