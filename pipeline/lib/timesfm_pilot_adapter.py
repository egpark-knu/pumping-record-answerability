"""Prepared D02 adapter. No inference before supplied frozen protocol hash matches.
Only head, raw pumping, raw rainfall reach the model. No truth or W inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
SNAPSHOT = Path(__import__("os").environ.get("TIMESFM_SNAPSHOT", "timesfm-3.0-pytorch-43046b85"))
API = dict(return_quantiles=True, use_symmetric_averaging=False,
           make_positive=False, sort_quantiles=True, use_znorm=False,
           padding_mode="none", univariate=False)


def validate(data, horizon):
    if horizon not in (10, 30):
        raise ValueError("Expected separate horizon 10 or 30 call")
    head = np.asarray(data["head"])
    if head.ndim != 2 or head.shape[1] != 1024 or len(head) == 0:
        raise ValueError("head must be nonempty [N,1024]")
    n = len(head)
    for key, shape in [("head", (n, 1024)), ("pumping", (n, 1024+horizon)),
                       ("rainfall", (n, 1024+horizon))]:
        a = np.asarray(data[key])
        if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key}: expected finite numeric {shape}")
        if not np.isfinite(a.astype(np.float32)).all():
            raise ValueError(f"{key} not representable as finite float32")
    if (data["rainfall"] < 0).any():
        raise ValueError("Raw rainfall cannot be negative")
    ids = {}
    for key in ("query_id", "case_id", "pair_id", "schedule_id", "metadata_json"):
        a = np.asarray(data[key])
        if a.shape != (n,) or a.dtype.kind not in "US":
            raise ValueError(f"{key} must be [N] Unicode/bytes strings, no pickled objects")
        ids[key] = a.astype(str)
    if len(set(ids["query_id"])) != n:
        raise ValueError("query_id must be unique")
    groups = {}
    for i, pair in enumerate(ids["pair_id"]):
        groups.setdefault(pair, []).append(i)
        meta = json.loads(ids["metadata_json"][i])
        for key in ("origin_date", "rain_site", "realization", "rest_ratio", "signal_ratio", "pi_layer", "schedule_pair", "head_unit", "pumping_unit", "rainfall_unit", "daily_alignment"):
            if key not in meta:
                raise ValueError(f"Missing metadata {key}")
    ordered = []
    for pair, ix in groups.items():
        if len(ix) != 2 or set(ids["schedule_id"][ix]) != {"a", "b"}:
            raise ValueError(f"{pair}: exactly schedules a and b required")
        a = next(i for i in ix if ids["schedule_id"][i] == "a")
        b = next(i for i in ix if ids["schedule_id"][i] == "b")
        if ids["case_id"][a] != ids["case_id"][b] or ids["metadata_json"][a] != ids["metadata_json"][b]:
            raise ValueError(f"{pair}: shared case/metadata required")
        if not np.array_equal(head[a], head[b]):
            raise ValueError(f"{pair}: history target differs")
        if not np.array_equal(data["pumping"][a, :1024], data["pumping"][b, :1024]):
            raise ValueError(f"{pair}: history pumping differs")
        if not np.array_equal(data["rainfall"][a], data["rainfall"][b]):
            raise ValueError(f"{pair}: history/future rainfall differs")
        ordered.extend([a, b])
    return np.asarray(ordered), ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--horizon", type=int, choices=[10, 30], required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--expected-protocol-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    digest = hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    if digest != a.expected_protocol_sha256 or len(digest) != 64:
        raise ValueError("Frozen protocol hash mismatch")
    if a.batch < 2 or a.batch % 2:
        raise ValueError("Batch must be positive even number >=2 to keep pairs together")
    if a.output.suffix != ".npz":
        raise ValueError("Output must have .npz suffix")
    if not a.output.resolve().is_relative_to((ROOT / "results/pilot").resolve()):
        raise ValueError("Output must stay under Paper3 results/pilot")
    with np.load(a.cases, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    order, ids = validate(data, a.horizon)
    if str(np.asarray(data["protocol_sha256"]).item()) != digest:
        raise ValueError("Case protocol hash mismatch")
    if a.validate_only:
        print(json.dumps({"validation": "passed", "queries": len(order), "protocol_sha256": digest}))
        return
    # This branch is prepared only; invoke in a separately assigned post-freeze turn.
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(SNAPSHOT),
        local_files_only=True, force_download=False, device=a.device,
        per_core_batch_size=a.batch))
    if list(model.config.quantiles) != [i/10 for i in range(1, 10)]:
        raise ValueError("Unexpected checkpoint quantile levels")
    q = np.empty((len(order), a.horizon, 9), dtype=np.float32)
    for start in range(0, len(order), a.batch):
        ix = order[start:start+a.batch]
        contexts = [data["head"][i].astype(np.float32) for i in ix]
        cov = [np.stack([data["pumping"][i], data["rainfall"][i]]).astype(np.float32) for i in ix]
        preds = list(model.predict_batch(contexts=contexts, horizon=a.horizon,
            past_future_covariates=cov, ts_ids=ids["query_id"][ix].tolist(), **API))
        if len(preds) != len(ix):
            raise ValueError("Incomplete model output")
        for offset, pred in enumerate(preds):
            values = np.asarray(pred.quantiles)
            if pred.ts_id != ids["query_id"][ix[offset]] or values.shape != (a.horizon,9) or not np.isfinite(values).all():
                raise ValueError("Invalid forecast output")
            q[start+offset] = values
    widths = q[:,:,8] - q[:,:,0]
    # Contrast of medians: neither a median nor a credible interval of joint E.
    e_point = q[0::2,:,4] - q[1::2,:,4]
    proxy_lo = q[0::2,:,0] - q[1::2,:,8]
    proxy_hi = q[0::2,:,8] - q[1::2,:,0]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, quantiles=q, marginal_widths=widths,
        E_point=e_point, E_width_proxy=proxy_hi-proxy_lo,
        E_proxy_low=proxy_lo, E_proxy_high=proxy_hi,
        quantile_levels=np.arange(1,10)/10, protocol_sha256=digest,
        input_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(),
        pair_id=ids["pair_id"][order][0::2], query_id=ids["query_id"][order],
        schedule_id=ids["schedule_id"][order], case_id=ids["case_id"][order],
        metadata_json=ids["metadata_json"][order], horizon=a.horizon,
        checkpoint_revision=SNAPSHOT.name, device=a.device, batch=a.batch,
        api_json=json.dumps(API, sort_keys=True))


if __name__ == "__main__":
    main()
