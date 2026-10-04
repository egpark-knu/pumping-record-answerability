"""API smoke test on a toy series (NOT a pilot cell, no KMA data, no scoring)."""
import numpy as np, pandas as pd, pastas as ps, json, inspect
rng = np.random.default_rng(0)
idx = pd.date_range("2000-01-01", periods=1500, freq="D")
P = pd.Series(rng.gamma(0.3, 10, len(idx)), idx)
Q = pd.Series(np.where((np.arange(len(idx)) % 300) < 200, 100.0, 0.0), idx)
h = (P.rolling(60, min_periods=1).mean()*0.3 - Q.rolling(30, min_periods=1).mean()*0.01 + rng.normal(0, 0.02, len(idx)))
ml = ps.Model(h.loc["2001-01-01":])
sig = {k: str(inspect.signature(v)) for k, v in {"Model": ps.Model, "StressModel": ps.StressModel, "ArNoiseModel": ps.ArNoiseModel}.items()}
ps.StressModel(ml, P, ps.Gamma(), name="recharge", settings="prec")
ps.StressModel(ml, Q, ps.Hantush(), name="well", up=False, settings="well")
ps.ArNoiseModel(ml)
ps.solver.LeastSquares(ml); ml.solve(report=False)
samp = ml.solver.get_parameter_sample(name="well", n=200, seed=1)
blk = ml.stressmodels["well"].rfunc.block(samp[0])
print(json.dumps({"signatures": sig, "params": ml.parameters.optimal.round(4).to_dict(), "well_sample_shape": list(samp.shape), "block_len": len(blk)}, indent=1, default=str))
