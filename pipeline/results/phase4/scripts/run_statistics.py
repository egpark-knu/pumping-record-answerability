"""Stage 4 statistics: assemble the record-level rows of the map and calendar cohorts and run the frozen statistics.

Calls the frozen API (p4_collect.assemble, p4_statistics.validate_inventory and run_statistics) exactly as the
original runs did, between two freeze checks, and writes the reference-fit diagnostics of the same run.
Writes results/phase4/{primary_rows.csv, reference_diagnostics.csv, REFERENCE_DIAGNOSTICS.json,
ANALYSIS_COVERAGE.json, statistics.json}.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[3])))
P = R / 'results/phase4'
sys.path.insert(0, str(R / 'lib'))
from p4_contracts import require_frozen  # noqa: E402
import p4_collect as collect  # noqa: E402
import p4_statistics as statistics  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, obj):
    (P / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    require_frozen()
    rows = collect.assemble(strict=True)
    records = json.loads((P / 'COHORT_MANIFESTS.json').read_text())['records']
    _, coverage = statistics.validate_inventory(rows, records, official=True)
    assert len(rows) == 16440, len(rows)
    collect.write(rows, P / 'primary_rows.csv')
    refs = {(r['source_phase'], r['case_id']): r for r in rows}
    assert len(refs) == 4110, len(refs)
    by_key = {(r['source_phase'], r['case_id'], r['pair_id'], r['lead_day']): r for r in rows}
    diags = []
    for (phase, cid), r in refs.items():
        p = R / r['reference_source_path']
        assert sha(p) == r['reference_sha256']
        j = json.loads(p.read_text())
        assert all(j['pairs'][pair]['E_point'][lead - 1] == by_key[phase, cid, pair, lead]['E_reference_m']
                   for pair in statistics.PAIRS for lead in statistics.LEADS)
        diags.append(dict(source_phase=phase, case_id=cid, reference_source_path=str(p.relative_to(R)), reference_sha256=sha(p),
                          solver_success=j.get('solver_success'), n_draws_requested=j.get('n_draws_requested'),
                          n_param_draws_accepted=j.get('n_param_draws_accepted'), draws_below_requested=j.get('draws_below_requested'),
                          reference_status=r['reference_status']))
    with (P / 'reference_diagnostics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(diags[0]))
        w.writeheader()
        w.writerows(diags)
    save('REFERENCE_DIAGNOSTICS.json', dict(n_cases=len(diags), by_phase=dict(Counter(d['source_phase'] for d in diags)),
                                            solver_false=[d for d in diags if d['solver_success'] is False],
                                            partial_draws=[d for d in diags if d['draws_below_requested']],
                                            minimum_accepted=min(d['n_param_draws_accepted'] for d in diags)))
    save('ANALYSIS_COVERAGE.json', dict(coverage=coverage, rows=len(rows), cases=len(refs),
                                        row_phase_counts=dict(Counter(r['source_phase'] for r in rows)),
                                        reference_status=dict(Counter(r['reference_status'] for r in rows)),
                                        primary_rows_sha256=sha(P / 'primary_rows.csv')))
    result = statistics.run_statistics(rows, cohort_manifest=records, reference_rows=None, official=True)
    assert result['bootstrap']['n'] == 999 and result['bootstrap']['seed'] == 20261001
    save('statistics.json', result)
    require_frozen()
    print(json.dumps(dict(rows=len(rows), cases=len(refs), coverage=coverage, statistics='results/phase4/statistics.json')))


if __name__ == '__main__':
    main()
