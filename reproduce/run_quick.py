"""Quick reproduction: Table 3, principal numbers and Figures 1-6 and S19 from the archived aggregates.

Usage:  python -m reproduce.run_quick [--out outputs] [--only figure1,figure3]
Writes outputs/table3.csv, outputs/table3.md, outputs/principal_numbers.json,
outputs/figures/<name>.png and .pdf, and outputs/quick_check.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy

from . import figures, numbers, table3
from .common import OUT, style


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--only', help='comma-separated figure names, e.g. figure1,figureS19')
    args = ap.parse_args(argv)
    out = args.out; (out / 'figures').mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); timing = {}

    rows = table3.build()
    with (out / 'table3.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / 'table3.md').write_text(table3.markdown(rows))
    t3 = table3.check_against_archive(rows); timing['table3_s'] = round(time.perf_counter() - t0, 2)

    t1 = time.perf_counter()
    nums = numbers.build()
    (out / 'principal_numbers.json').write_text(json.dumps(nums, indent=1) + '\n'); timing['numbers_s'] = round(time.perf_counter() - t1, 2)

    font = style(); made = {}
    names = args.only.split(',') if args.only else list(figures.FIGURES)
    for name in names:
        t = time.perf_counter()
        fig, facts = figures.FIGURES[name]()
        for ext, dpi in (('png', 300), ('pdf', 300)):
            fig.savefig(out / 'figures' / f'{name}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.16, facecolor='white')
        figures.plt.close(fig)
        made[name] = dict(facts=facts, seconds=round(time.perf_counter() - t, 2))

    ok = t3['max_relative_difference'] < 1e-9 and all(n['match'] for n in nums)
    report = dict(passed=ok, table3=t3, numbers_matched=sum(n['match'] for n in nums), numbers_total=len(nums),
                  mismatched=[n['name'] for n in nums if not n['match']], figures=made, font=font, timing=timing,
                  total_seconds=round(time.perf_counter() - t0, 2),
                  environment=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=numpy.__version__,
                                   matplotlib=matplotlib.__version__))
    (out / 'quick_check.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(f"Table 3 max relative difference vs archive: {t3['max_relative_difference']:.2e}")
    print(f"Principal numbers matching the manuscript: {report['numbers_matched']}/{report['numbers_total']}")
    print(f"Figures written: {', '.join(made)} -> {out / 'figures'}")
    print(f"Total {report['total_seconds']} s; {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
