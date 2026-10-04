"""FIXTURE ONLY: synthetic rain + fake realization for unit/API/physical checks.
Not KMA data, not a pilot cell, not a score."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_generator as g

def fixture_rain(seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", "2006-12-31", freq="D")
    doy = idx.dayofyear.values
    pwet = 0.15 + 0.25 * np.exp(-((doy - 200) / 40.0) ** 2)
    wet = rng.random(idx.size) < pwet
    amt = rng.gamma(0.7, 18.0, idx.size)
    s = pd.Series(np.where(wet, amt, 0.0).round(1), idx, name="FIXTURE_SYNTHETIC")
    return s

def fixture_realization(r=0):
    return dict(realization=r, site_stem="FIXTURE_SYNTHETIC", context_start="2003-01-01",
                seed_schedule=1000 + r, seed_natural=2000 + r, seed_noise=3000 + r, seed_pastas_param_sample=4000 + r)
