"""Deterministic site/origin allocation for the 10 realizations (dates only; no series generated)."""
import json, datetime as dt
D0 = dt.date(2005, 1, 1); N = 7305; WARM, CTX, HMAX = 365, 1024, 30
lo, hi = WARM + CTX, N - HMAX           # origin index o: first forecast day index; context = [o-1024, o)
sites = [("안동태화_충적", "ASOS 136 Andong", 4), ("산청산청_암반", "ASOS 289 Sancheong", 3), ("남해남해_암반", "ASOS 295 Namhae", 3)]
rows = []; r = 0
for stem, stn, k in sites:
    for j in range(k):
        o = lo + round((j + 0.5) * (hi - lo) / k) + (91 * j if k == 4 else 0)  # quarter shift so Andong origins span seasons
        rows.append({"realization": r, "site_stem": stem, "kma_station": stn,
                     "warmup_start": str(D0 + dt.timedelta(o - CTX - WARM)), "context_start": str(D0 + dt.timedelta(o - CTX)),
                     "context_end": str(D0 + dt.timedelta(o - 1)), "origin_first_forecast_day": str(D0 + dt.timedelta(o)),
                     "horizon30_end": str(D0 + dt.timedelta(o + HMAX - 1)), "origin_index_in_CL_file": o,
                     "seed_schedule": 20260930 + 100*r + 1, "seed_natural": 20260930 + 100*r + 2, "seed_noise": 20260930 + 100*r + 3,
                     "seed_pastas_param_sample": 20260930 + 100*r + 4})
        r += 1
print(json.dumps(rows, ensure_ascii=False, indent=1))
json.dump(rows, open("results/pilot/phase0_checks/realization_calendar.json", "w"), ensure_ascii=False, indent=1)
