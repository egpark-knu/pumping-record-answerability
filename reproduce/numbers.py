"""Principal numbers of the manuscript, recomputed or read from the archived aggregates.

Each entry gives the value obtained here, the value printed in the manuscript,
the rounding used there, and the source file. Counts are recomputed from the
case-level summary table; model-based values are read from the archived fits.
"""
from __future__ import annotations

from .common import DATA, INCREASE, STOP, detected, read_csv, read_json
from .table3 import detection_crossing


def _close(value, printed, digits):
    return round(value, digits) == round(printed, digits)


def build() -> list[dict]:
    cases = read_csv(DATA / 'cases/case_level_summary.csv')
    det = read_json(DATA / 'detection/detection_map_models.json')
    coef = det['primary']['no_layer']['coefficients']
    out = []

    def add(name, value, printed, digits, source, scale=1.0):
        out.append(dict(name=name, value=value, manuscript=printed, rounding_digits=digits, source=source,
                        match=_close(value * scale, printed, digits)))

    # Answerability map (Section 3.3, Table 3, Abstract)
    mp = [r for r in cases if r['is_map'] == 'True' and r['pair_id'] == STOP and r['lead_day'] == '10']
    add('map records', len(mp), 1260, 0, 'cases/case_level_summary.csv')
    add('map records with detected effect', sum(detected(r) for r in mp), 861, 0, 'cases/case_level_summary.csv')
    add('signal coefficient b_S', coef['log_signal_ratio'], 3.922, 3, 'detection/detection_map_models.json')
    add('pause coefficient b_rho', coef['log_rho_plus_offset'], 0.664, 3, 'detection/detection_map_models.json')
    add('detection P=0.5 crossing, no pause (Abstract 0.24)', detection_crossing(coef, .5, 0.0), 0.241, 3, 'detection/detection_map_models.json')
    add('detection P=0.5 crossing, pause 4 response times (Abstract 0.11)', detection_crossing(coef, .5, 4.0), 0.114, 3, 'detection/detection_map_models.json')
    ex = det['exchange']
    add('long-pause / no-pause crossing ratio', ex['point']['SR4_over_SR0'], 0.475, 3, 'detection/detection_map_models.json')
    add('exchange ratio interval low', ex['bootstrap']['SR4_over_SR0']['low'], 0.395, 3, 'detection/detection_map_models.json')
    add('exchange ratio interval high', ex['bootstrap']['SR4_over_SR0']['high'], 0.547, 3, 'detection/detection_map_models.json')

    # Operating calendars (Section 3.3, Abstract 54% -> 86%)
    for dist, n_det, pct in (('calendar200m', 671, 54.0), ('calendar20m', 1073, 86.4)):
        rs = [r for r in cases if r['variant'] == 'original' and r['record_set'] == dist and r['pair_id'] == STOP
              and r['lead_day'] == '10' and r['active'] == 'True']
        k = sum(detected(r) for r in rs)
        add(f'{dist}: active decision days', len(rs), 1242, 0, 'cases/case_level_summary.csv')
        add(f'{dist}: detected', k, n_det, 0, 'cases/case_level_summary.csv')
        add(f'{dist}: detection percentage', 100 * k / len(rs), pct, 1, 'cases/case_level_summary.csv')

    # Forecasting engine (Section 3.5, Abstract 12% and 89%)
    base = next(b for b in read_json(DATA / 'intervention/baseline_ratios.json')['baselines']
                if b['pair'] == INCREASE and b['lead'] == 10 and b['variant'] == 'original')
    add('baseline median engine/true ratio, increase, 10 d (Abstract 12%)', base['ratio']['point'], 0.117, 3, 'intervention/baseline_ratios.json')
    summaries = read_json(DATA / 'intervention/intervention_summaries.json')['summaries']
    hi = next(s for s in summaries if s['pair'] == INCREASE and s['lead'] == 10 and s['variant'] == 'range')
    add('higher-rate history median ratio (Abstract 89%)', hi['ratio']['point'], 0.889, 3, 'intervention/intervention_summaries.json')
    groups = read_json(DATA / 'engine/rate_coverage_groups.json')['groups']
    g = {x['range_category']: x for x in groups if (x['record_set'], x['pair'], x['lead'], x['window'], x['signal_bin'])
         == ('calendar20m', INCREASE, 10, 'full1024', 'all_signal_bins')}
    add('within-range records (20 m)', g['within_range']['rows'], 126, 0, 'engine/rate_coverage_groups.json')
    add('within-range median ratio', g['within_range']['signed_engine_ratio']['point'], 0.471, 3, 'engine/rate_coverage_groups.json')
    add('outside-range records (20 m)', g['outside_range']['rows'], 1116, 0, 'engine/rate_coverage_groups.json')
    add('outside-range median ratio', g['outside_range']['signed_engine_ratio']['point'], 0.092, 3, 'engine/rate_coverage_groups.json')
    ops = read_csv(DATA / 'intervention/operation_paired_contrasts.csv')
    paddy = {r['metric']: float(r['median']) for r in ops if r['cohort'].startswith('paddy') and int(r['n']) == 168}
    add('paddy: higher-rate paired gain', paddy['range_paired_ratio_change'], 0.430, 3, 'intervention/operation_paired_contrasts.csv')
    add('paddy: nominal-control paired gain', paddy['control_paired_ratio_change'], 0.002, 3, 'intervention/operation_paired_contrasts.csv')

    # Matched-storage comparison (Section 3.2, Table S26)
    st = read_json(DATA / 'storage/matched_storage_comparison.json')['rows']
    med = sorted({round(r['median_paired_ratio'], 6) for r in st})
    add('matched-storage medians: minimum', min(med), 0.47, 2, 'storage/matched_storage_comparison.json')
    add('matched-storage medians: maximum', max(med), 0.84, 2, 'storage/matched_storage_comparison.json')
    add('matched-storage comparisons with 95% interval below one (of 12 rows)', sum(r['ci_high'] < 1 for r in st), 12, 0, 'storage/matched_storage_comparison.json')
    return out
