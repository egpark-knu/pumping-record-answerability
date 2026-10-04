"""D06 A execution driver: frozen D05 W / reference / truth-evaluation bodies on the isolated b >= 1e-6 family.

The case loop, W run, reference fit + joint-draw replay, truth evaluation, resume test and marker writer are the FROZEN
p4_run_wb functions, re-bound (identical bytecode) to a private globals dict in which only these names differ:
  w0, w, pt   -> isolated p5_w_family modules (Hantush log10 b lower floor -6; every other box/budget/algorithm frozen)
  CASES, OUT  -> results/phase5/cases, results/phase5/wb (or explicit fixture directories)
  DRIVER_SHA  -> SHA256 over this driver + p4_run_wb + p5_w_family (resume refuses a different driver)
  run_W / run_reference -> thin wrappers that call the frozen bodies and add a family_box receipt to the saved records.
Budgets are the frozen module values (W seed 20260930, global 128/256/512/1024, local 512, 12 rounds; reference 1000
joint draws, max_iter 50, calendar seed, 12 declared starts); they are not CLI options. Truth is read only after the W and
reference records are written. The frozen p4_run_wb module and the p3 modules are never mutated.

Production: requires the complete phase5 freeze, BLAS/OpenMP thread variables = 1, PYTHONDONTWRITEBYTECODE=1, and the actor shard
results/phase5/SHARD_<ACTOR>.json (1080 ids, actor_of(twin) rule). Use 8 single-thread processes per actor:
  for k in 0..7: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      env/.venv_pilot/bin/python lib/p5_run_wb.py --mode production --actor shard_b --shard k --nshards 8
Resume is automatic: a case is skipped only when its marker matches tf_input/protocol/driver hashes and all outputs.
Fixture mode (--mode fixture --cases-dir D --out-dir O) runs given case files with FIXTURE_DIGEST into scratch outputs
only; --fixture-budget may reduce the budget there for adapter tests and is refused in production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import p4_run_wb as frozen
import p5_contracts as K
import p5_w_family as wf

REBOUND = ('run_W', 'run_reference', 'truth_eval', 'fresh', 'run_case')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def driver_sha() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(frozen.__file__), Path(wf.__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()


def bind(cases_dir: Path, out_dir: Path, b_log10_min: float = K.B_LOG10_MIN_NEW, budget=None, n_draws=None) -> types.SimpleNamespace:
    """Frozen driver functions bound to the isolated family and the given directories (new globals; nothing mutated)."""
    ns = wf.load(b_log10_min)
    stamp = wf.family_stamp(ns)
    G = dict(vars(frozen))
    out_dir = Path(out_dir)
    # marker output paths are stored relative to ROOT: the project root in production, the scratch parent for fixtures
    root = K.ROOT if out_dir.resolve().is_relative_to(K.ROOT.resolve()) else out_dir.resolve().parent
    G.update(w0=ns.base, w=ns.repaired, pt=ns.pastas, CASES=Path(cases_dir), OUT=out_dir, P2=out_dir.parent, ROOT=root, DRIVER_SHA=driver_sha())
    if budget is not None:
        G['BUDGET'] = dict(budget)
    if n_draws is not None:
        G['N_DRAWS'] = int(n_draws)
    fn = {n: types.FunctionType(getattr(frozen, n).__code__, G, n, getattr(frozen, n).__defaults__, getattr(frozen, n).__closure__) for n in REBOUND}

    def run_W(ti):
        eng, rec = fn['run_W'](ti)
        rec['family_box'] = stamp
        return eng, rec

    def run_reference(ti, seed):
        fit, draws, names, Es = fn['run_reference'](ti, seed)
        fit['family_box'] = stamp
        return fit, draws, names, Es

    G.update(fn)
    G.update(run_W=run_W, run_reference=run_reference)
    return types.SimpleNamespace(G=G, run_case=G['run_case'], fresh=G['fresh'], family=ns, stamp=stamp, frozen_fn=fn)


def rebinding_receipt(b) -> dict:
    return dict({n: dict(same_code=b.frozen_fn[n].__code__ is getattr(frozen, n).__code__, globals_isolated=b.frozen_fn[n].__globals__ is b.G)
                 for n in REBOUND},
                w_is_isolated=b.G['w'] is b.family.repaired and b.G['w'] is not sys.modules.get('p3_wenvelope_repaired_v1_2'),
                pt_is_isolated=b.G['pt'] is b.family.pastas, frozen_module_globals_unchanged=frozen.w.__name__ == 'p3_wenvelope_repaired_v1_2'
                and frozen.pt.__name__ == 'p3_pastas' and frozen.CASES == K.P4 / 'cases', budget=b.G['BUDGET'], n_draws=b.G['N_DRAWS'],
                frozen_budget=frozen.BUDGET, frozen_n_draws=frozen.N_DRAWS)


def ref_seeds() -> dict:
    return {int(r['realization']): int(r['seed_pastas_param_sample'])
            for r in json.loads((K.ROOT / 'results/pilot/phase0_checks/realization_calendar.json').read_text())}


def load_shard(actor: str, base: Path = K.P5) -> list[str]:
    rec = json.loads((base / f'SHARD_{actor.upper()}.json').read_text())
    ids = rec['case_ids']
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != rec['case_ids_sha256'] or rec['n_cases'] != K.A_PER_ACTOR or len(set(ids)) != K.A_PER_ACTOR:
        raise ValueError('D06 shard manifest changed')
    if any(K.actor_of(c) != actor for c in ids) or set(ids) != set(K.shards()[actor]):
        raise ValueError('D06 shard rule mismatch')
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('production', 'fixture'))
    ap.add_argument('--actor', choices=K.ACTORS)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--cases-dir', type=Path)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--fixture-budget', type=json.loads, default=None)
    ap.add_argument('--fixture-draws', type=int, default=None)
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.nshards:
        raise SystemExit('shard index outside nshards')
    if a.mode == 'production':
        if a.cases_dir or a.out_dir or a.fixture_budget or a.fixture_draws:
            raise SystemExit('fixture directories/budgets are refused in production')
        if not a.actor:
            raise SystemExit('production requires --actor')
        bad = [v for v in THREAD_VARS + ('PYTHONDONTWRITEBYTECODE',) if os.environ.get(v) != '1']
        if bad:
            raise SystemExit('production requires single-thread BLAS/OpenMP and PYTHONDONTWRITEBYTECODE=1: ' + ','.join(bad))
        digest = K.require_frozen()
        cases_dir, out_dir = K.P5 / 'cases', K.P5 / 'wb'
        gen = json.loads((cases_dir / 'generation_checks.json').read_text())
        if not gen.get('all_ok') or gen.get('protocol_sha256') != digest:
            raise SystemExit('phase5 generation checks absent/failed/stale')
    else:
        if not a.cases_dir or not a.out_dir or a.out_dir.resolve().is_relative_to((K.P5 / 'wb').resolve()):
            raise SystemExit('fixture mode needs explicit --cases-dir/--out-dir outside results/phase5/wb')
        digest, cases_dir, out_dir = K.FIXTURE_DIGEST, a.cases_dir, a.out_dir
    b = bind(cases_dir, out_dir, budget=a.fixture_budget, n_draws=a.fixture_draws)
    for d in ('W', 'reference', 'truth_eval', 'markers', 'logs'):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    man = json.loads((cases_dir / 'derived_manifest.json').read_text())
    seeds = ref_seeds()
    ids = [(m['case_id'], seeds[int(m['realization'])], m['tf_sha256']) for m in man]
    if a.actor and a.mode == 'production':
        keep = set(load_shard(a.actor))
        if keep - {x[0] for x in ids}:
            raise SystemExit('actor shard contains ids absent from the derived manifest')
        ids = [x for x in ids if x[0] in keep]
    if a.only:
        ids = [x for x in ids if x[0] in set(a.only)]
    ids = ids[a.shard::a.nshards]
    tag = f'{a.actor}_proc{a.shard}' if a.actor else f'{a.mode}_shard{a.shard}'
    logf = open(out_dir / 'logs' / f'{tag}.log', 'a')

    def log(msg):
        logf.write(time.strftime('%Y-%m-%dT%H:%M:%S ') + msg + '\n')
        logf.flush()
    log(f"start {tag} mode={a.mode} n={len(ids)} protocol={digest[:16]} driver={b.G['DRIVER_SHA'][:16]} b_log10_min={b.family.b_log10_min} "
        f"budget={b.G['BUDGET']} draws={b.G['N_DRAWS']}")
    for i, (cid, seed, tfs) in enumerate(ids):
        if not cid.startswith('d06A_'):
            raise SystemExit('this driver runs only D06 A distance cases: ' + cid)
        if b.fresh(cid, tfs, digest):
            continue
        r = b.run_case(cid, digest, seed, log)
        log(f"{i + 1}/{len(ids)} {cid} W={r.get('W_status')} flags={','.join(r.get('flags', []))} ref={r.get('reference_status')} "
            f"truth={r.get('truth_eval_ok')} {r['total_wall_s']:.1f}s")
    if a.mode == 'production':
        K.require_frozen()
    log(f'{tag} done')


if __name__ == '__main__':
    main()
