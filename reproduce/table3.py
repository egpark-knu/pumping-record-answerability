"""Table 3: signal-ratio crossings of detection frequency and magnitude resolution.

Crossings are recomputed from the archived fitted coefficients of the pooled
(no-layer) logistic detection model and the log-linear magnitude model; the
95% intervals are the archived 999-draw realization-cluster bootstrap endpoints.
No model is refitted.
"""
from __future__ import annotations

import math

from .common import DATA, STOP, read_csv, read_json

RHOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
OFFSET = 0.05   # pause ratio offset used in both fits: log(rho + 0.05)


def detection_crossing(coef: dict, probability: float, rho: float) -> float:
    logit = math.log(probability / (1 - probability))
    return math.exp((logit - coef['intercept'] - coef['log_rho_plus_offset'] * math.log(rho + OFFSET)) / coef['log_signal_ratio'])


def magnitude_crossing(coef: dict, rho: float) -> float:
    """Signal ratio at which W/|E_true| = 1 under log(W/|E|) = g0 + g_rho*log(rho+0.05) + g_SR*log(SR)."""
    return math.exp(-(coef['intercept'] + coef['log_rho_plus_offset'] * math.log(rho + OFFSET)) / coef['log_signal_ratio'])


def build() -> list[dict]:
    det = read_json(DATA / 'detection/detection_map_models.json')['primary']['no_layer']
    mag = read_json(DATA / 'detection/magnitude_models.json')
    rows = []
    for rho in RHOS:
        row = {'rho': rho}
        for p in (0.5, 0.9):
            node = det['crossings'][str(rho)][str(p)]
            value = detection_crossing(det['coefficients'], p, rho)
            assert math.isclose(value, node['signal_ratio'], rel_tol=1e-9), (rho, p, value, node['signal_ratio'])
            row[f'detection_p{p}'] = value
            row[f'detection_p{p}_low'] = node['bootstrap']['low']
            row[f'detection_p{p}_high'] = node['bootstrap']['high']
        for lead in (10, 30):
            model = mag[str(lead)][STOP]['no_layer']
            node = model['crossings'][str(rho)]
            value = magnitude_crossing(model['coefficients'], rho)
            assert math.isclose(value, node['signal_ratio'], rel_tol=1e-9), (rho, lead, value, node['signal_ratio'])
            row[f'magnitude{lead}'] = value
            row[f'magnitude{lead}_low'] = node['bootstrap']['low']
            row[f'magnitude{lead}_high'] = node['bootstrap']['high']
        rows.append(row)
    return rows


def check_against_archive(rows: list[dict]) -> dict:
    archived = {float(r['rho']): r for r in read_csv(DATA / 'detection/table3_crossings.csv')}
    worst = 0.0
    for r in rows:
        a = archived[r['rho']]
        for k in ('detection_p0.5', 'detection_p0.9', 'magnitude10', 'magnitude30'):
            worst = max(worst, abs(r[k] - float(a[k])) / abs(float(a[k])))
    return {'archived_table': 'data/detection/table3_crossings.csv', 'max_relative_difference': worst}


def markdown(rows: list[dict]) -> str:
    f = lambda r, k: f"{r[k]:.3f} ({r[k + '_low']:.3f}–{r[k + '_high']:.3f})"
    lines = ['| ρ | Detection P=0.5 | Detection P=0.9 | Magnitude 10 d | Magnitude 30 d |', '| --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['rho']:.2f} | {f(r, 'detection_p0.5')} | {f(r, 'detection_p0.9')} | {f(r, 'magnitude10')} | {f(r, 'magnitude30')} |")
    return '\n'.join(lines) + '\n'
