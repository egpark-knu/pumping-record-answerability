"""Isolated D06 W/reference family adapter: frozen sources, one allowlisted change (Hantush log10 b lower floor).

The frozen engine (p3_wenvelope, p3_wenvelope_repaired_v1_2) and the structurally matched reference (p3_pastas) bind the
family box as module globals (PUMP_BOX / BOX / FAMILY) read at call time by every global, local, refinement, direct
certification and reference pmin step. Patching those globals in the shared p3 modules would mutate the frozen modules
for every caller in the process, and a Sobol-only subclass would miss best_fit/refine/direct boxes. This adapter instead:

1. reads each frozen source file and refuses unless its SHA256 equals the D05 freeze pin (results/phase4/protocol_freeze.json);
2. applies exactly the allowlisted line replacements below (each old line must occur exactly once; nothing else changes);
3. compiles the result into NEW module objects (p5iso_*) registered under new names, so the frozen modules, their
   globals and every other caller stay untouched;
4. returns receipts: frozen SHA, patched-source SHA, unified line diff, and the effective bounds read back from the
   isolated modules.

With b_log10_min = -4.0 the patched sources are textually identical to the frozen box (only the isolated import names
differ); identity_regression() uses this to prove the adapter reproduces the frozen engine. Old cases never use this.
"""
from __future__ import annotations

import difflib
import importlib
import json
import sys
import types
from pathlib import Path

from p5_contracts import ROOT, P4, B_LOG10_MIN_OLD, B_LOG10_MIN_NEW, sha_bytes

SOURCES = {'wenvelope': 'lib/p3_wenvelope.py', 'repaired': 'lib/p3_wenvelope_repaired_v1_2.py', 'pastas': 'lib/p3_pastas.py'}
FROZEN_NAMES = {'wenvelope': 'p3_wenvelope', 'repaired': 'p3_wenvelope_repaired_v1_2', 'pastas': 'p3_pastas'}


def iso_name(key: str, b_log10_min: float) -> str:
    tag = f'{b_log10_min:+.1f}'.replace('+', 'p').replace('-', 'm').replace('.', '_')
    return f'p5iso_{key}_b{tag}'


def _edits(b_log10_min: float) -> dict:
    b = float(b_log10_min)
    if not (b <= B_LOG10_MIN_OLD and b in (B_LOG10_MIN_OLD, B_LOG10_MIN_NEW)):
        raise ValueError('only the frozen (-4) or selected D06 (-6) lower floor is allowlisted')
    wen = iso_name('wenvelope', b)
    return {
        'wenvelope': [("PUMP_BOX = np.array([[0.0, 3.5], [-4.0, np.log10(25.0)]])",
                       f"PUMP_BOX = np.array([[0.0, 3.5], [{b:.1f}, np.log10(25.0)]])")],
        'repaired': [("import p3_wenvelope as base", f"import {wen} as base"),
                     ("from p3_wenvelope import (CONV_TOL, EXCESS_MULTS, HMAX, IA, LEVELS, MAX_RESTARTS, NAT_BOX, NLIN, PUMP_BOX, T95_MAX)",
                      f"from {wen} import (CONV_TOL, EXCESS_MULTS, HMAX, IA, LEVELS, MAX_RESTARTS, NAT_BOX, NLIN, PUMP_BOX, T95_MAX)")],
        'pastas': [("FAMILY = dict(a=(1.0, 10 ** 3.5), b=(1e-4, 25.0), n=(0.5, 5.0), theta=(1.0, 100.0), tau=(5.0, 500.0))",
                    f"FAMILY = dict(a=(1.0, 10 ** 3.5), b=({10.0 ** b:g}, 25.0), n=(0.5, 5.0), theta=(1.0, 100.0), tau=(5.0, 500.0))")],
    }


def frozen_pins() -> dict:
    f = json.loads((P4 / 'protocol_freeze.json').read_text())
    return {k: f['hashes'][p] for k, p in SOURCES.items()}


def patched_sources(b_log10_min: float) -> dict:
    pins = frozen_pins()
    out = {}
    for key, path in SOURCES.items():
        raw = (ROOT / path).read_bytes()
        if sha_bytes(raw) != pins[key]:
            raise PermissionError(f'{path} differs from its D05 freeze pin')
        src = raw.decode()
        lines = src.splitlines(keepends=True)
        for old, new in _edits(b_log10_min)[key]:
            hits = [i for i, l in enumerate(lines) if l.rstrip('\n') == old]
            if len(hits) != 1:
                raise ValueError(f'allowlisted line occurs {len(hits)} times in {path}: {old}')
            lines[hits[0]] = new + '\n'
        new_src = ''.join(lines)
        diff = [l for l in difflib.unified_diff(src.splitlines(), new_src.splitlines(), lineterm='', n=0) if not l.startswith(('---', '+++', '@@'))]
        allowed = {'-' + o for o, n in _edits(b_log10_min)[key] if o != n} | {'+' + n for o, n in _edits(b_log10_min)[key] if o != n}
        if set(diff) != allowed:
            raise AssertionError(f'non-allowlisted difference in isolated {path}: {sorted(set(diff) ^ allowed)}')
        out[key] = dict(path=path, frozen_sha256=pins[key], source=new_src, patched_sha256=sha_bytes(new_src.encode()), diff=diff)
    return out


def load(b_log10_min: float = B_LOG10_MIN_NEW) -> types.SimpleNamespace:
    """Isolated module triple (base engine, repaired v1.2 engine, reference) for the given lower floor."""
    srcs = patched_sources(b_log10_min)
    mods = {}
    for key in ('wenvelope', 'repaired', 'pastas'):  # base before repaired: repaired imports the isolated base by name
        name = iso_name(key, b_log10_min)
        if name in sys.modules and getattr(sys.modules[name], '__p5_patched_sha256__', None) == srcs[key]['patched_sha256']:
            mods[key] = sys.modules[name]
            continue
        m = types.ModuleType(name)
        m.__file__ = f'<p5iso from {srcs[key]["path"]}>'
        m.__p5_patched_sha256__ = srcs[key]['patched_sha256']
        sys.modules[name] = m
        exec(compile(srcs[key]['source'], m.__file__, 'exec'), m.__dict__)
        mods[key] = m
    receipts = {k: {kk: v for kk, v in s.items() if kk != 'source'} for k, s in srcs.items()}
    ns = types.SimpleNamespace(base=mods['wenvelope'], repaired=mods['repaired'], pastas=mods['pastas'], b_log10_min=float(b_log10_min),
                               receipts=receipts)
    ns.effective = effective_bounds(ns)
    return ns


def effective_bounds(ns) -> dict:
    b, r, p = ns.base, ns.repaired, ns.pastas
    eff = dict(base_PUMP_BOX=b.PUMP_BOX.tolist(), repaired_PUMP_BOX=r.PUMP_BOX.tolist(), repaired_BOX=r.BOX.tolist(),
               pastas_FAMILY_b=list(p.FAMILY['b']), pastas_HANTUSH_STARTS=[list(x) for x in p.HANTUSH_STARTS],
               base_NAT_BOX=b.NAT_BOX.tolist(), T95_MAX=float(b.T95_MAX), P_FREE=int(b.P_FREE))
    want = float(ns.b_log10_min)
    eff['consistent'] = bool(b.PUMP_BOX[1, 0] == want and r.PUMP_BOX[1, 0] == want and r.BOX[4, 0] == want
                             and abs(p.FAMILY['b'][0] - 10.0 ** want) <= 1e-12 * 10.0 ** want
                             and r.WEngineV12.run.__globals__ is r.__dict__ and r.base is b
                             and b.WEngine.best_fit.__globals__ is b.__dict__)
    return eff


def frozen_module_state() -> dict:
    """Snapshot of the frozen modules' box globals; used to prove isolation (nothing is mutated or restored)."""
    out = {}
    for key, name in FROZEN_NAMES.items():
        m = importlib.import_module(name)
        if key == 'pastas':
            out[key] = dict(FAMILY=repr(m.FAMILY), HANTUSH_STARTS=repr(m.HANTUSH_STARTS))
        else:
            out[key] = dict(PUMP_BOX=m.PUMP_BOX.tolist(), BOX=getattr(m, 'BOX', None).tolist() if hasattr(m, 'BOX') else None)
    return out


def family_stamp(ns) -> dict:
    from p5_contracts import family_bounds
    return dict(family_version=family_bounds(ns.b_log10_min)['version'], bounds=family_bounds(ns.b_log10_min), b_log10_min=ns.b_log10_min, effective=ns.effective, sources=ns.receipts, adapter_sha256=sha_bytes(Path(__file__).read_bytes()))


# ------------------------------------------------------------------ interface names (IMPLEMENTATION_INTERFACES draft)
def family_receipt(b_log10_min: float = B_LOG10_MIN_NEW) -> dict:
    return family_stamp(load(b_log10_min))


def engine(tf_input, seed=20260930, b_log10_min: float = B_LOG10_MIN_NEW):
    return load(b_log10_min).repaired.WEngineV12(tf_input, seed=seed)


def reference(tf_input, seed, n_draws=1000, b_log10_min: float = B_LOG10_MIN_NEW):
    return load(b_log10_min).pastas.fit_and_propagate(tf_input, seed=seed, n_draws=n_draws)
