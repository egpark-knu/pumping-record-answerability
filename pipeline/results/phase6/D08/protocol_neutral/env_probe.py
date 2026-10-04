"""Validate original environments/import interfaces, without inference or fitting."""
import argparse
import importlib.metadata as md
import inspect
import json
import sys
from pathlib import Path
from d08_runtime import ROOT, HERE, gate, now, read, save


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--kind', choices=['science', 'model'], required=True)
    args = a.parse_args()
    digest = gate()
    contract = read(ROOT / 'results/phase5/FROZEN_EXECUTION_CONTRACT.json')[args.kind + '_python']
    assert sys.executable == contract['executable']
    versions = {pkg: md.version(pkg) for pkg in contract['packages']}
    for pkg, version in versions.items():
        assert version == contract['packages'][pkg]['version'], pkg
    if args.kind == 'science':
        import p4_cases
        import p5_distance_cases
        import p3_make_cases
        import p5_w_family
        import p4_run_wb
        family = p5_w_family.family_receipt(-6.0)
        assert family['effective']['consistent']
        assert p4_run_wb.BUDGET == dict(global_levels=(128, 256, 512, 1024), n_local=512, max_local_rounds=12)
        assert p4_run_wb.W_SEED == 20260930
        interfaces = {n: str(inspect.signature(f)) for n, f in [('physics', p4_cases._physics), ('geometry', p5_distance_cases.geometry), ('save_tf', p3_make_cases.save_tf), ('load_tf', p3_make_cases.load_tf_input), ('W_engine', p5_w_family.engine)]}
    else:
        import torch
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        from p3_phase2_timesfm_execution import checked_prediction
        assert torch.backends.mps.is_available()
        interfaces = dict(predict_batch=str(inspect.signature(TimesFM3Evaluator.predict_batch)), checked_prediction=str(inspect.signature(checked_prediction)), model_config=str(inspect.signature(ModelConfig)), mps_available=True)
    receipt = dict(protocol_sha256=digest, utc=now(), kind=args.kind, executable=sys.executable, versions=versions, interfaces=interfaces, model_instantiated=False, W_executed=False, scientific_data_scored=False)
    save(HERE / ('ENV_' + args.kind.upper() + '.json'), receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
