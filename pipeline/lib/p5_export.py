"""Observed-only D06 A export: the frozen D05 validator/exporter with one routing change for experiment_group A_distance20.

p4_export.declared_q_origin accepts only groups A/B and treats A as the artificial last-context-origin design, so the new
calendar cases would be rejected (or misrouted if relabelled A). Here the frozen functions validate, export_arrays,
subset and prepare_track are re-bound (identical bytecode, new globals dict) to a declared_q_origin that accepts ONLY
A_distance20 and re-derives the origin rate from the unchanged calendar rule (form, realization, origin date, observed
origin-day rain), exactly the D05 B branch. Metadata stay labels; no hydraulic quantity enters the model inputs.
The frozen p4_export module itself is not modified.
"""
from __future__ import annotations

import types

import numpy as np

import p4_export as frozen
from p5_contracts import A_GROUP

API, SNAPSHOT, TRACKS = frozen.API, frozen.SNAPSHOT, frozen.TRACKS
GROUPS = (A_GROUP,)


def declared_q_origin(meta, pumping_context, rain_origin):
    """Authoritative origin rate from observed/declared inputs only; raises on any inconsistent declaration."""
    if meta.get('experiment_group') not in GROUPS:
        raise ValueError('Unknown D06 experiment group')
    if 'q_origin_m3d' not in meta:
        raise ValueError('Missing declared origin rate')
    q = float(meta['q_origin_m3d'])
    last = float(pumping_context[1023])
    if not np.isfinite(q) or q < 0:
        raise ValueError('Invalid declared origin rate')
    from p4_cases import declared_origin_rate
    if meta.get('q_last_context_day_m3d') is not None and float(meta['q_last_context_day_m3d']) != last:
        raise ValueError('A_distance20 last-context declaration mismatch')
    if q != declared_origin_rate(meta['calendar_form'], meta['realization'], meta['origin_date'], float(rain_origin)):
        raise ValueError('A_distance20 origin rate differs from deterministic calendar')
    return q


_G = dict(vars(frozen))
_G['declared_q_origin'] = declared_q_origin
REBOUND = ('validate', 'export_arrays', 'subset', 'prepare_track')
for _name in REBOUND:
    _f = getattr(frozen, _name)
    _G[_name] = types.FunctionType(_f.__code__, _G, _name, _f.__defaults__, _f.__closure__)
validate, export_arrays, subset, prepare_track = (_G[n] for n in REBOUND)


def rebinding_receipt() -> dict:
    return {n: dict(same_code=_G[n].__code__ is getattr(frozen, n).__code__, globals_isolated=_G[n].__globals__ is not vars(frozen))
            for n in REBOUND}
