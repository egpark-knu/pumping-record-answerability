"""Full computation of the paper, stage by stage.

    python run_pipeline.py --check                  checks only: freezes, imports, entrypoint startup (no computation)
    python run_pipeline.py --list                   every command of every stage, in order
    python run_pipeline.py --stage phase5           run one stage (commands in order; stops at the first failure)
    python run_pipeline.py --stage phase5 --from 4  resume a stage at its 4th command

Run from any directory. Commands run in this directory (PUMPING_RECORD_ROOT, default: the directory of this
file) with PUMPING_RECORD_ROOT exported. Science steps use the Python running this script; forecasting-engine
steps use ENGINE_PYTHON (default: the same Python). Engine steps need torch, the TimesFM 3.0 source on the
Python path (TIMESFM_SOURCE) and the checkpoint (TIMESFM_SNAPSHOT); the original engine runs used the Apple MPS
device. KMA_CLIMATE_DIR must point at data/climate of the repository.

Each stage checks its frozen protocol before computing (protocol.md must match protocol.sha256, and the pins in
the stage's freeze record must match the files). The freeze records of this release pin the public files.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('PUMPING_RECORD_ROOT', str(Path(__file__).resolve().parent))).resolve()
NSHARDS_WB = 8      # envelope-search shards (any shard count partitions the cases; 8 = original process count)
NSHARDS_C = 16      # linearized-width shards of phase5 (production recipe)
ACTORS = ('shard_a', 'shard_b')  # the two disjoint engine/envelope partitions of phase4 and phase5


def S(*args):
    return ('science', list(args))


def E(*args):
    return ('engine', list(args))


def shards(env, script, *args, n=NSHARDS_WB):
    return [(env, [script, *args, '--shard', str(k), '--nshards', str(n)]) for k in range(n)]


D08 = 'results/phase6/D08/protocol_neutral/d08_runtime.py'
C_ARGS = ('--cases', 'results/phase5/C_CASES.json', '--out', 'results/phase5/C')

STAGES = {
    'phase2': dict(
        title='pause-transition set (D03): cases, engine, envelope W and reference, cell metrics, analysis',
        steps=[S('lib/p3_phase2_make_cases.py', '--out', 'results/phase2/cases'),
               E('lib/p3_phase2_timesfm_execution.py'),
               *shards('science', 'lib/p3_phase2_run_wb.py'),
               S('lib/p3_phase2_collect_wb.py'),
               S('lib/p3_phase2_cell_metrics.py'),
               S('lib/p3_phase2_analysis.py')]),
    'phase3': dict(
        title='count-recency, pumping-scale and storage sets (D04): cases, envelope W, engine, normalization, analysis',
        steps=[S('lib/p3_phase3_make_cases.py'),
               *shards('science', 'lib/p3_phase3_run_wb.py'),
               S('lib/p3_phase3_collect_wb.py'),
               E('lib/p3_phase3_timesfm_execution.py'),
               E('lib/p3_phase3_normalization_run.py'),
               S('lib/p3_phase3_analysis.py')]),
    'phase4': dict(
        title='map-extension set and operating calendars at 200 m (D05): cases, envelope W, engine, statistics, tables',
        steps=[S('lib/p4_make_cases.py'),
               *[step for a in ACTORS for step in shards('science', 'lib/p4_run_wb.py', '--actor', a)],
               *[E('lib/p4_timesfm_execution.py', '--actor', a) for a in ACTORS],
               S('results/phase4/scripts/run_statistics.py'),
               S('lib/p4_map_supplement_figures.py'),
               S('lib/p4_calendar_contradiction_figures.py')]),
    'phase5': dict(
        title='operating calendars at 20 m, time-scale bundles, linearized widths and observation masks (D06)',
        steps=[S('lib/p5_distance_cases.py', '--mode', 'production', '--regen-parents'),
               *[step for a in ACTORS for step in shards('science', 'lib/p5_run_wb.py', '--mode', 'production', '--actor', a)],
               *[E('lib/p5_timesfm_execution.py', '--mode', 'production', '--actor', a) for a in ACTORS],
               S('lib/p5_collect.py', '--mode', 'production_A'),
               *shards('science', 'lib/p5_linearized.py', '--production', *C_ARGS, n=NSHARDS_C),
               S('lib/p5_linearized.py', '--merge', *C_ARGS),
               S('lib/p5_timescale.py', '--mode', 'production'),
               S('lib/p5_analysis.py', '--production'),
               S('lib/p5_figures.py', '--production')]),
    'phase6': dict(
        title='history intervention (D08): higher-rate, nominal-rate and zero-rate blocks on the W50 records',
        steps=[S(D08, 'preflight'), S(D08, 'selftest'),
               S(D08, 'A', '--execute'),
               S(D08, 'generate', '--execute'),
               E(D08, 'B', '--execute'),
               *[('science', [D08, 'W', '--execute', '--shard', str(k), '--nshards', '8']) for k in range(8)],
               S(D08, 'aggregate_B', '--execute'),
               S(D08, 'aggregate_W', '--execute')]),
    'phase8': dict(
        title='zero-effect (null) fits, effect detection, inclusion and aggregation (D10)',
        # stored null fits would be resumed by null_fit.py (and rejected, since they carry the original driver hash)
        set_aside=['results/phase8/calculations/null_fits'],
        steps=[S('results/phase8/scripts/prepare.py'),
               S('results/phase8/scripts/null_fit.py'),
               S('results/phase8/scripts/correct_null.py'),
               S('results/phase8/scripts/aggregate.py'),
               S('results/phase8/B/calculate_inclusion.py')]),
    'phase10': dict(
        title='matched-response-time storage comparison (Table S26)',
        steps=[S('results/phase10/calculations/additional2_calculations.py')]),
}
ORDER = list(STAGES)
STORED_ONLY = {
    'pilot': 'D02 pilot (method development; not an input of later stages). Procedure: results/pilot/protocol.md section 15; '
             'stored tables in the data bundle.',
    'phase6/engine': 'D07 engine robustness replay. Its driver also checks the run records of the original replay, which are '
                     'not part of this release; the stored outputs ENGINE_ROBUSTNESS_ROWS.csv and ENGINE_DISTRIBUTIONS.csv are in the data bundle.',
    'phase7': 'D09 supporting analyses of stored results. The scripts check the run records of the original analyses, which '
              'are not part of this release; their outputs are in the data bundle and phase8 reads them as stored inputs.',
}
FREEZE_RECORDS = ['results/pilot/protocol_freeze.json', 'results/phase2/protocol_freeze.json', 'results/phase3/protocol_freeze.json',
                  'results/phase4/protocol_freeze.json', 'results/phase4/PREPARE_OLD_HASHES.json', 'results/phase5/protocol_freeze.json',
                  'results/phase6/D08/protocol_neutral/FREEZE.json', 'results/phase6/D08/protocol_neutral/SOURCE_BASELINE.json',
                  'results/phase8/PROTOCOL_FREEZE_RECEIPT.json']
PROTOCOLS = {'pilot': 'results/pilot', 'phase2': 'results/phase2', 'phase3': 'results/phase3', 'phase4': 'results/phase4',
             'phase5': 'results/phase5', 'phase6': 'results/phase6/D08/protocol_neutral:results/phase6', 'phase8': 'results/phase8'}
BUNDLE_MARKERS = ['results/phase4/COHORT_MANIFESTS.json', 'results/phase4/bootstrap_draw_matrix.npy',
                  'results/phase6/D08/protocol_neutral/SOURCE_INDEX.json', 'results/phase7/D09/analysis/B02/B02_source_rows.csv']
CLIMATE = {'안동태화_충적_CL.txt': '20dddf7f15db', '산청산청_암반_CL.txt': 'e95466d5ea3b', '남해남해_암반_CL.txt': '7733e1352ddc'}
ENGINE_MODULES = {'torch', 'timesfm3', 'safetensors', 'huggingface_hub'}
HEX = re.compile(r'^[0-9a-f]{64}$')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def python_for(env):
    return os.environ.get('ENGINE_PYTHON', sys.executable) if env == 'engine' else sys.executable


def child_env():
    return dict(os.environ, PUMPING_RECORD_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1')


def commands(stage):
    return STAGES[stage]['steps']


def show(env, args):
    return ('[engine] ' if env == 'engine' else '') + ' '.join(['python', *args])


def do_list():
    print(f'# cwd: {ROOT}\n# environment: PUMPING_RECORD_ROOT, KMA_CLIMATE_DIR, TIMESFM_SNAPSHOT, TIMESFM_SOURCE; ENGINE_PYTHON for [engine] steps')
    for st in ORDER:
        print(f'\n## {st}: {STAGES[st]["title"]}')
        for p in STAGES[st].get('set_aside', []):
            print(f'#   before the first step, stored outputs in {p} are moved to stored/{p}')
        for i, (env, args) in enumerate(commands(st), 1):
            print(f'{i:3d}  {show(env, args)}')
    print('\n## stored outputs only (not rerun)')
    for k, v in STORED_ONLY.items():
        print(f'#   {k}: {v}')


def run_stage(stage, start):
    for p in STAGES[stage].get('set_aside', []):
        src, dst = ROOT / p, ROOT / 'stored' / p
        if src.exists() and any(src.iterdir()):
            if dst.exists():
                raise SystemExit(f'{dst} exists; remove it or move {src} aside yourself')
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            print(f'moved stored outputs {p} -> stored/{p}', flush=True)
    steps = commands(stage)
    for i, (env, args) in enumerate(steps, 1):
        if i < start:
            continue
        print(f'\n[{stage} {i}/{len(steps)}] {show(env, args)}', flush=True)
        t = time.time()
        rc = subprocess.call([python_for(env), *args], cwd=ROOT, env=child_env())
        print(f'[{stage} {i}/{len(steps)}] rc={rc} {time.time() - t:.0f} s', flush=True)
        if rc:
            raise SystemExit(f'stopped at {stage} step {i}; resume with --stage {stage} --from {i}')


# ---------------------------------------------------------------- checks
results = []


def record(name, ok, detail=''):
    results.append(dict(check=name, ok=bool(ok), detail=detail))
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  ({detail})' if detail else ''), flush=True)


def call(name, argv, timeout=900):
    t = time.time()
    p = subprocess.run(argv, cwd=ROOT, env=child_env(), capture_output=True, text=True, timeout=timeout)
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:] if (p.stdout + p.stderr).strip() else ['']
    record(name, p.returncode == 0, f'rc={p.returncode}, {time.time() - t:.1f} s; {tail[0][:160]}')
    return p.returncode


def pins(obj, out):
    """Every (project-relative path -> sha256) pin of a freeze record, at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and HEX.match(v) and ('/' in k or '.' in k) and not k.endswith('sha256'):
                out[k] = v
            else:
                pins(v, out)
    elif isinstance(obj, list):
        for v in obj:
            pins(v, out)
    return out


def imports_of(path):
    tree = ast.parse(Path(path).read_text(), str(path))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split('.')[0])
    return mods


def missing_imports(script, python):
    code = ('import importlib.util,json,sys\n'
            f'sys.path[:0]=[{str(ROOT / "lib")!r},{str((ROOT / script).parent)!r}]\n'
            f'mods={sorted(imports_of(ROOT / script))!r}\n'
            'print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))')
    p = subprocess.run([python, '-c', code], cwd=ROOT, env=child_env(), capture_output=True, text=True)
    return json.loads(p.stdout) if p.returncode == 0 else ['<python failed: ' + p.stderr.strip()[-120:] + '>']


def do_check(engine_python):
    print(f'root {ROOT}\npython {sys.version.split()[0]} ({sys.executable})')
    want = {}
    for line in (ROOT.parent / 'requirements-pipeline.txt').read_text().splitlines():
        m = re.match(r'^([A-Za-z0-9_.-]+)==(\S+)$', line.strip())
        if m:
            want[m.group(1)] = m.group(2)
    import importlib.metadata as md
    have = {}
    for k in want:
        try:
            have[k] = md.version(k)
        except md.PackageNotFoundError:
            have[k] = None
    record('science packages installed', all(have.values()), ', '.join(f'{k} {have[k]}' for k in want))
    diff = {k: (have[k], want[k]) for k in want if have[k] and have[k] != want[k]}
    print('INFO  versions differing from the recorded science environment: ' + (json.dumps(diff) if diff else 'none'))
    # environment and data
    clim = Path(os.environ.get('KMA_CLIMATE_DIR', 'climate'))
    record('KMA_CLIMATE_DIR holds the three climate tables', all((clim / n).is_file() and sha(clim / n).startswith(h) for n, h in CLIMATE.items()), str(clim))
    record('data bundle extracted into the pipeline directory', all((ROOT / m).is_file() for m in BUNDLE_MARKERS), ', '.join(m for m in BUNDLE_MARKERS if not (ROOT / m).is_file()) or 'markers present')
    snap = os.environ.get('TIMESFM_SNAPSHOT')
    if snap:
        ok = Path(snap, 'model.safetensors').is_file() and sha(Path(snap, 'model.safetensors')) == 'a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
        record('TIMESFM_SNAPSHOT is the TimesFM 3.0 checkpoint of the paper', ok, snap)
    else:
        print('INFO  TIMESFM_SNAPSHOT not set: engine steps cannot run here (science steps can)')
    src = os.environ.get('TIMESFM_SOURCE')
    if src:
        want = json.loads((ROOT / 'results/phase2/A_preflight_manifest.json').read_text())['source_hashes']
        bad = [k for k, v in want.items() if not (Path(src).parents[1] / k).is_file() or sha(Path(src).parents[1] / k) != v]
        record('TIMESFM_SOURCE is the TimesFM source of the paper', not bad, src + (f'; differs: {bad}' if bad else ''))
    else:
        print('INFO  TIMESFM_SOURCE not set: engine steps cannot run here (science steps can)')
    # freezes: protocol hashes and every pin of the public freeze records
    for st, dirs in PROTOCOLS.items():
        d = dirs.split(':')
        shafile, proto = ROOT / d[0] / 'protocol.sha256', ROOT / d[-1] / 'protocol.md'
        record(f'{st} protocol.md matches protocol.sha256', shafile.is_file() and proto.is_file() and shafile.read_text().split()[0] == sha(proto), shafile.read_text().split()[0][:12] if shafile.is_file() else 'missing')
    for rec in FREEZE_RECORDS:
        allpins = pins(json.loads((ROOT / rec).read_text()), {})
        bad = [k for k, v in allpins.items() if not (ROOT / k).is_file() or sha(ROOT / k) != v]
        record(f'pins of {rec}', not bad, f'{len(allpins)} pins' + (f'; mismatched/missing: {bad[:5]}' if bad else ''))
    # the freeze guards of the production code, each in its own process
    py = sys.executable
    lib = f'import sys; sys.path.insert(0, {str(ROOT / "lib")!r}); '
    call('guard pilot: p3_generator.require_frozen', [py, '-c', lib + 'import p3_generator as g; from pathlib import Path; print(g.require_frozen(Path("results/pilot/protocol.md"), Path("results/pilot/protocol.sha256")))'])
    call('guard phase2: p3_phase2_cases.require_frozen_phase2', [py, '-c', lib + 'import p3_phase2_cases as c; print(c.require_frozen_phase2())'])
    call('guard phase3: p3_phase3_run_wb.check_protocol', [py, '-c', lib + 'import p3_phase3_run_wb as r; print(r.check_protocol())'])
    call('guard phase4: p4_contracts.require_frozen', [py, '-c', lib + 'import p4_contracts as c; print(c.require_frozen())'])
    call('phase4 shard manifests: p4_partition.load', [py, '-c', lib + 'import p4_partition as p; print([len(p.load(a)) for a in ("shard_a", "shard_b")])'])
    call('guard phase5: p5_contracts.require_frozen', [py, '-c', lib + 'import p5_contracts as c; print(c.require_frozen())'])
    call('guard phase5 C: p5_linearized.require_frozen', [py, '-c', lib + 'import p5_linearized as c; print(c.require_frozen())'])
    call('guard phase6: d08_runtime preflight', [py, D08, 'preflight'])
    call('guard phase6: d08_runtime selftest', [py, D08, 'selftest'])
    # every listed script exists and compiles; its imports resolve (engine imports in the engine Python)
    scripts = sorted({args[0] for st in ORDER for _, args in commands(st)})
    record('all stage scripts present', all((ROOT / s).is_file() for s in scripts), f'{len(scripts)} scripts')
    bad = []
    for p in sorted(ROOT.rglob('*.py')):
        try:
            compile(p.read_text(), str(p), 'exec')
        except SyntaxError as e:
            bad.append(f'{p.relative_to(ROOT)}: {e}')
    record('all pipeline .py files compile', not bad, '; '.join(bad[:3]) or f'{sum(1 for _ in ROOT.rglob("*.py"))} files')
    engine_scripts = sorted({args[0] for st in ORDER for env, args in commands(st) if env == 'engine'})
    for s in scripts:
        miss = missing_imports(s, py)
        engine_only = [m for m in miss if m in ENGINE_MODULES]
        other = [m for m in miss if m not in ENGINE_MODULES]
        record(f'imports resolve: {s}', not other, ('engine modules for the engine environment: ' + ', '.join(engine_only)) if engine_only else '')
    if engine_python:
        for s in engine_scripts:
            miss = missing_imports(s, engine_python)
            record(f'imports resolve in ENGINE_PYTHON: {s}', not miss, ', '.join(miss))
    else:
        print('INFO  --engine-python not given: engine-step imports checked only for science modules')
    # entrypoint startup: argparse entrypoints answer --help (modules load, frozen settings import, no computation)
    for s in scripts:
        src = (ROOT / s).read_text()
        if 'argparse' in src and s not in engine_scripts:
            call(f'startup --help: {s}', [py, s, '--help'], timeout=300)
    n_fail = sum(not r['ok'] for r in results)
    print(f'\n{len(results) - n_fail} passed, {n_fail} failed')
    return n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true', help='freeze checks, imports and entrypoint startup; no computation')
    g.add_argument('--list', action='store_true', help='print every command of every stage')
    g.add_argument('--stage', choices=ORDER, help='run the commands of one stage in order')
    ap.add_argument('--from', dest='start', type=int, default=1, help='with --stage: first command to run (1-based)')
    ap.add_argument('--engine-python', default=os.environ.get('ENGINE_PYTHON'), help='with --check: also resolve engine-step imports in this Python')
    ap.add_argument('--json', type=Path, help='with --check: write the check results to this file')
    a = ap.parse_args()
    if a.list:
        return do_list()
    if a.stage:
        return run_stage(a.stage, a.start)
    n_fail = do_check(a.engine_python)
    if a.json:
        a.json.write_text(json.dumps(dict(root=str(ROOT), python=sys.version.split()[0], results=results), indent=1) + '\n')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
