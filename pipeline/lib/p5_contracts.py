"""D06 phase5 contracts: identities, A/B/C case universes and the complete-freeze production gate.

Execution modes are explicit. 'production' requires a complete results/phase5/protocol_freeze.json whose every pinned
file (including the transitive local import closure of the p5 runtime) still has its frozen SHA256, plus the unchanged
D05 freeze (p4_contracts.require_frozen). 'fixture' and 'prefreeze_input' never write official outputs, never stamp the
protocol digest and never score: they use FIXTURE_DIGEST, a 64-hex marker that can never equal a frozen protocol hash.

A identities: the D05 B calendar id with prefix d05B replaced by d06A (2160 = 3 forms x 12 months x 6 layers x 10
realizations). The actor of a new case is p4_partition.actor_of(twin id) (1080/1080, every form/month/layer cell 5/5).
B universe: D03 540 + D04 A240/B360/C90 + D05 A720 = 1950 (calendar cases excluded). C universe: 540+690+2880+2160 = 6270.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
LIB = ROOT / 'lib'
P2, P3, P4, P5 = (ROOT / 'results' / f'phase{i}' for i in (2, 3, 4, 5))
EVIDENCE = P5 / 'implementation_A_B'

A_GROUP = 'A_distance20'
A_COUNT = 2160
A_PER_ACTOR = 1080
R_NEW_M = 20.0
R_OLD_M = 200.0
B_LOG10_MIN_OLD = -4.0
B_LOG10_MIN_NEW = -6.0          # selected decision A: only the Hantush b lower floor changes, new r20 cases only
B_UNIVERSE_COUNT = 1950
C_UNIVERSE_COUNT = 6270
ACTORS = ('shard_a', 'shard_b')
PAIRS = ('P1_continue_vs_stop', 'P2_current_vs_1p5x')
MODES = ('production', 'fixture', 'prefreeze_input')
FREEZE_STATUS = 'complete_protocol_frozen'
FIXTURE_DIGEST = hashlib.sha256(b'D06-P5-FIXTURE-ONLY-NOT-A-PROTOCOL').hexdigest()
HEX64 = re.compile(r'[0-9a-f]{64}')

# p5 runtime entry points whose transitive local import closure must be pinned by the final freeze.
RUNTIME_ENTRIES = ('p5_contracts', 'p5_w_family', 'p5_distance_cases', 'p5_run_wb', 'p5_export', 'p5_timesfm_execution',
                   'p5_collect', 'p5_timescale')


# Files read at runtime that the AST import closure cannot see (exec-loaded frozen sources, fixed inputs). The HIGH freeze
# author adds the model package/checkpoint closure, climate/calendar inputs and old receipts on top of these.
EXTRA_PINS = ('lib/p3_wenvelope.py', 'lib/p3_wenvelope_repaired_v1_2.py', 'lib/p3_pastas.py', 'lib/p4_run_wb.py', 'lib/p4_export.py',
              'lib/p4_collect.py', 'lib/p4_statistics.py', 'results/phase4/protocol_freeze.json', 'results/phase4/statistics.json',
              'results/phase4/bootstrap_draw_matrix.npy', 'results/phase4/cases/derived_manifest.json',
              'results/phase2/cases/derived_manifest.json', 'results/phase3/cases/derived_manifest.json',
              'results/pilot/phase0_checks/realization_calendar.json')


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(path) -> str:
    p = Path(path)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# ------------------------------------------------------------------ transitive local import closure (AST, no execution)
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split('.')[0])
    return out


def local_import_closure(entries=RUNTIME_ENTRIES, lib=LIB) -> list[str]:
    """Project-relative paths of every local lib module reachable by absolute imports (including lazy imports inside
    functions) from the entry modules. Third-party/stdlib names have no file in lib/ and are not followed."""
    seen, todo = set(), list(entries)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        f = lib / f'{name}.py'
        if not f.is_file():
            alt = lib / 'tests' / f'{name}.py'
            if not alt.is_file():
                continue
            f = alt
        seen.add(name)
        todo.extend(_imports(f) - seen)
    files = []
    for name in sorted(seen):
        f = lib / f'{name}.py'
        files.append(rel(f if f.is_file() else lib / 'tests' / f'{name}.py'))
    return files


# ------------------------------------------------------------------ freeze gate
def require_frozen(base: Path = P5, *, verify_d05: bool = True, entries=RUNTIME_ENTRIES) -> str:
    """Production gate. Returns the frozen phase5 protocol digest or raises PermissionError."""
    f_path = base / 'protocol_freeze.json'
    if not f_path.is_file():
        raise PermissionError('phase5 protocol_freeze.json absent: production execution not authorised')
    f = json.loads(f_path.read_text())
    if f.get('status') != FREEZE_STATUS:
        raise PermissionError(f"phase5 freeze status {f.get('status')!r} is not {FREEZE_STATUS}")
    proto, side = base / 'protocol.md', base / 'protocol.sha256'
    if not proto.is_file() or not side.is_file():
        raise PermissionError('phase5 protocol.md / protocol.sha256 absent')
    digest = sha(proto)
    if digest != f.get('protocol_sha256') or digest != side.read_text().split()[0]:
        raise PermissionError('phase5 protocol hash mismatch')
    hashes = f.get('hashes') or {}
    if not hashes:
        raise PermissionError('phase5 freeze pins no files')
    bad = [p for p, h in hashes.items() if not isinstance(h, str) or not HEX64.fullmatch(h)]
    if bad:
        raise PermissionError('placeholder/non-SHA256 pins in phase5 freeze: ' + str(bad[:5]))
    missing = sorted((set(local_import_closure(entries)) | set(EXTRA_PINS)) - set(hashes))
    if missing:
        raise PermissionError('phase5 freeze omits runtime import-closure files: ' + str(missing[:10]))
    for p, h in hashes.items():
        q = ROOT / p if not Path(p).is_absolute() else Path(p)
        if not q.is_file() or sha(q) != h:
            raise PermissionError('phase5 frozen dependency changed or absent: ' + p)
    if verify_d05:
        import p4_contracts
        p4_contracts.require_frozen()
    if base == P5:
        import_audit(hashes)
    return digest


def resolve_mode(mode: str, base: Path = P5) -> str:
    """Digest to stamp: the frozen protocol digest in production, FIXTURE_DIGEST otherwise (never official)."""
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; choose one of {MODES}')
    return require_frozen(base) if mode == 'production' else FIXTURE_DIGEST


def official_digest(digest: str) -> bool:
    return digest != FIXTURE_DIGEST


# ------------------------------------------------------------------ A identities
def parent_ids() -> list[str]:
    import p4_contracts
    return p4_contracts.new_case_ids()[1]


def new_id(parent: str) -> str:
    if not parent.startswith('d05B_'):
        raise ValueError('A parent must be a D05 B calendar case')
    return 'd06A_' + parent[len('d05B_'):]


def twin_of(cid: str) -> str:
    if not cid.startswith('d06A_'):
        raise ValueError('not a D06 A distance case: ' + cid)
    return 'd05B_' + cid[len('d06A_'):]


def new_case_ids() -> list[str]:
    ids = [new_id(p) for p in parent_ids()]
    if len(ids) != A_COUNT or len(set(ids)) != A_COUNT:
        raise ValueError('D06 A identity count')
    return ids


def parse(cid: str) -> dict:
    import p4_partition
    k = p4_partition.parse(twin_of(cid))
    return dict(k, group=A_GROUP, twin_case_id=twin_of(cid))


def actor_of(cid: str) -> str:
    import p4_partition
    return p4_partition.actor_of(twin_of(cid))


def shards() -> dict:
    out = {a: [] for a in ACTORS}
    for cid in new_case_ids():
        out[actor_of(cid)].append(cid)
    if any(len(v) != A_PER_ACTOR for v in out.values()) or set(out['shard_a']) & set(out['shard_b']):
        raise ValueError('D06 A shards are not 1080/1080 disjoint')
    return out


def shard_balance(ids, zero_origin=None, variant=None) -> dict:
    from collections import Counter
    ks = [parse(c) for c in ids]
    cells = Counter((k['form'], k['month'], k['sid']) for k in ks)
    rep = dict(n=len(ids), by_realization=dict(Counter(k['realization'] for k in ks)), by_layer=dict(Counter(k['sid'] for k in ks)),
               by_form=dict(Counter(k['form'] for k in ks)), cell_min_max=[min(cells.values()), max(cells.values())], n_cells=len(cells))
    if zero_origin is not None:
        rep['zero_origin'] = sum(c in zero_origin for c in ids)
    if variant is not None:
        rep['water_curtain_variant'] = sum(c in variant for c in ids)
    return rep


# ------------------------------------------------------------------ B and C universes (input identities only)
def _manifest(phase_dir: Path) -> list[dict]:
    return json.loads((phase_dir / 'cases/derived_manifest.json').read_text())


def b_universe() -> list[dict]:
    """1950 noncalendar records with covariate sources (authoritative derived manifests); no outcome is read."""
    out = []
    for phase, d in (('phase2', P2), ('phase3', P3), ('phase4', P4)):
        for m in _manifest(d):
            grp = m.get('experiment_group', 'D03' if phase == 'phase2' else None)
            if phase == 'phase4' and grp != 'A':
                continue
            out.append(dict(source_phase=phase, case_id=m['case_id'], experiment_group=grp, realization=int(m['realization']), sid=m['sid'],
                            storage_type=m['storage_type'], R_days=float(m['R_days']), t95_d=float(m['t95_d']), SR=float(m['SR']),
                            pi_r=float(m['pi_r']), rho_nominal=m.get('rho_nominal'), N_nominal=m.get('N_nominal'), Q_scale=m.get('Q_scale', 1.0),
                            recency_label=m.get('recency_label'), d04c=bool(phase == 'phase3' and grp == 'C'),
                            tf_sha256=m['tf_sha256'], truth_sha256=m['truth_sha256']))
    keys = [(r['source_phase'], r['case_id']) for r in out]
    if len(out) != B_UNIVERSE_COUNT or len(set(keys)) != B_UNIVERSE_COUNT:
        raise ValueError(f'B universe {len(out)} != {B_UNIVERSE_COUNT}')
    return out


C_FIELDS = ('source_phase', 'case_id', 'experiment_group', 'twin_case_id', 'actor', 'realization', 'sid', 'storage_type', 'tf_input_path',
            'tf_sha256', 'truth_path', 'truth_sha256', 'W_path', 'W_tolerance_key', 'reference_path', 'T_m2_d', 'S', 'c_d', 'r_m', 'a_d', 'b',
            'gain_m_per_m3d', 't95_d', 'pi_r', 'family_bounds', 'q_origin_m3d', 'no_active_contrast', 'input_status')


def family_bounds(b_log10_min: float) -> dict:
    import numpy as np
    return dict(n=[0.5, 5.0], log10_theta=[0.0, 2.0], log10_tau=[float(np.log10(5.0)), float(np.log10(500.0))], log10_a=[0.0, 3.5],
                log10_b=[float(b_log10_min), float(np.log10(25.0))], t95_max_d=1000.0,
                linear='A in [0, Amax], eta_pump/eta_nat in [-eta_bound, eta_bound], beta_nat >= 0; Amax, eta_bound data-derived (WEngine, factor 10)',
                version='D06_new_r20_b1e-6' if b_log10_min == B_LOG10_MIN_NEW else 'D03_D05_original_b1e-4')


def _c_record(phase: str, m: dict, base: Path, twin=None, actor=None, b_min=B_LOG10_MIN_OLD, status='source_verified_by_manifest') -> dict:
    cid = m['case_id']
    return dict(source_phase=phase, case_id=cid, experiment_group=m.get('experiment_group', 'D03' if phase == 'phase2' else None), twin_case_id=twin,
                actor=actor if actor is not None else m.get('shard_actor'), realization=int(m['realization']), sid=m['sid'], storage_type=m['storage_type'],
                tf_input_path=rel(base / 'cases' / m['tf_input']) if m.get('tf_input') else rel(base / 'cases/tf_inputs' / f'{cid}.npz'),
                tf_sha256=m.get('tf_sha256'),
                truth_path=rel(base / 'cases' / m['truth']) if m.get('truth') else rel(base / 'cases/truth' / f'{cid}.npz'),
                truth_sha256=m.get('truth_sha256'), W_path=rel(base / 'wb/W' / f'{cid}.json'), W_tolerance_key='tolerance',
                reference_path=rel(base / 'wb/reference' / f'{cid}.json'), T_m2_d=float(m['T_m2_d']), S=float(m['storage_value']), c_d=float(m['c_d']),
                r_m=float(m['r_m']), a_d=float(m['a_d']), b=float(m['b']), gain_m_per_m3d=float(m['gain_m_per_m3d']), t95_d=float(m['t95_d']),
                pi_r=float(m['pi_r']), family_bounds=family_bounds(b_min), q_origin_m3d=m.get('q_origin_m3d', m.get('q0_m3d')),
                no_active_contrast=bool(m.get('no_active_contrast', False)), input_status=status)


def c_universe(new_manifest=None) -> list[dict]:
    """6270 C records (shared schema for the C worker). Old records reference their own immutable files/version. New
    records come from the generated phase5 manifest; before generation they carry input_status=pending_generation_after_freeze."""
    out = []
    for phase, d in (('phase2', P2), ('phase3', P3), ('phase4', P4)):
        out += [_c_record(phase, m, d) for m in _manifest(d)]
    if new_manifest is None:
        mf = P5 / 'cases/derived_manifest.json'
        new_manifest = json.loads(mf.read_text()) if mf.is_file() else None
    if new_manifest is not None:
        out += [_c_record('phase5', m, P5, twin=m['twin_case_id'], actor=m['shard_actor'], b_min=B_LOG10_MIN_NEW) for m in new_manifest]
    else:
        import p3_phase2_physmap as pm
        par = {m['case_id']: m for m in _manifest(P4)}
        for cid in new_case_ids():
            pm_ = par[twin_of(cid)]
            st = pm.Stratum(pm_['storage_type'], float(pm_['storage_value']), float(pm_['T_m2_d']), float(pm_['c_d']), R_NEW_M, 100.0)
            h = pm.hantush_params(st.T, st.S, st.c, st.r)
            out.append(dict(source_phase='phase5', case_id=cid, experiment_group=A_GROUP, twin_case_id=twin_of(cid), actor=actor_of(cid),
                            realization=int(pm_['realization']), sid=pm_['sid'], storage_type=pm_['storage_type'],
                            tf_input_path=rel(P5 / 'cases/tf_inputs' / f'{cid}.npz'), tf_sha256=None, truth_path=rel(P5 / 'cases/truth' / f'{cid}.npz'),
                            truth_sha256=None, W_path=rel(P5 / 'wb/W' / f'{cid}.json'), W_tolerance_key='tolerance',
                            reference_path=rel(P5 / 'wb/reference' / f'{cid}.json'), T_m2_d=st.T, S=st.S, c_d=st.c, r_m=st.r, a_d=h['a'], b=h['b'],
                            gain_m_per_m3d=h['gain'], t95_d=None, pi_r=h['pi_r'], family_bounds=family_bounds(B_LOG10_MIN_NEW),
                            q_origin_m3d=pm_['q_origin_m3d'], no_active_contrast=bool(pm_['no_active_contrast']),
                            input_status='pending_generation_after_freeze'))
    keys = [(r['source_phase'], r['case_id']) for r in out]
    if len(out) != C_UNIVERSE_COUNT or len(set(keys)) != C_UNIVERSE_COUNT:
        raise ValueError(f'C universe {len(out)} != {C_UNIVERSE_COUNT}')
    return out


# ------------------------------------------------------------------ audits and interface names (IMPLEMENTATION_INTERFACES draft)
def import_audit(hashes: dict | None = None) -> list[str]:
    """Every module currently loaded from lib/ must be pinned by the phase5 freeze (exec-loaded p5iso modules have no
    lib file and are covered by their frozen-source pins). Returns the audited paths; raises on an unpinned module."""
    import sys
    if hashes is None:
        hashes = json.loads((P5 / 'protocol_freeze.json').read_text())['hashes']
    seen = []
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if not f or not str(f).endswith('.py'):
            continue
        p = Path(f).resolve()
        if p.is_relative_to(LIB.resolve()):
            seen.append(rel(p))
    missing = sorted(set(seen) - set(hashes))
    if missing:
        raise PermissionError('loaded but unpinned lib modules: ' + str(missing[:10]))
    return sorted(seen)


def verify_old() -> int:
    """Unchanged D05 old-hash snapshot (and through it the immutable phase2/3/4 inputs, code and outputs)."""
    import p4_contracts
    return p4_contracts.verify_old()


a_case_ids = new_case_ids
parent_of = twin_of


def build_case_universe() -> dict:
    b = b_universe()
    return dict(B_CASES=b, D04C_SUBSET=[r for r in b if r['d04c']], C_CASES=c_universe())
