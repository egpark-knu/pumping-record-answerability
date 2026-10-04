"""D05 B operating-calendar preparation.

Deterministic daily pumping calendars for the phase-4 question. This module does not
freeze the phase-4 protocol, does not score W, and does not run a forecast model.

Daily off means a calendar day whose pumped volume is exactly zero. A within-day night
stop that only reduces that day's volume is not an off day and is not a pause.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "results" / "phase4"
CL_DIR = Path(__import__("os").environ.get("KMA_CLIMATE_DIR", "climate"))
PILOT_CALENDAR = ROOT / "results" / "pilot" / "phase0_checks" / "realization_calendar.json"

CONTEXT_DAYS = 1024
HORIZON_DAYS = 30
N_REALIZATIONS = 10
N_MONTHS = 12
Q_NOMINAL_M3D = 100.0
# Midpoint of the stated usual 14–15 h spray day. The sources do not name one hour.
CURTAIN_HOURS = 14.5
CURTAIN_DAILY_M3D = Q_NOMINAL_M3D * CURTAIN_HOURS / 24.0
# Any positive retained daily total. Fixed before scoring; not fit to an effect.
RAIN_OFF_THRESHOLD_MM = 0.0
# No daily-minimum series is in the local climate files. This probability is the
# authorized fallback and is self-imposed: no retained source states it.
WARM_NIGHT_OFF_PROBABILITY = 0.20
# Low April–October use on the variant. Self-imposed representation, not a metered rate.
OFFSEASON_FACTOR = 0.10
OFFSEASON_REALIZATIONS = (3, 6, 9)
FORMS = ("water_curtain", "paddy_irrigation", "domestic_continuous")
PAIRS = ("P1_continue_vs_stop", "P2_current_vs_1p5x")
IN_SEASON_MONTHS = (11, 12, 1, 2, 3)
PADDY_MONTHS = (5, 6, 7, 8)
ANCHOR_YEAR0 = 2010

# t95 is the one value per layer in results/phase2/cases/derived_manifest.json,
# copied through results/phase3/field_pause_sources.json. c follows the frozen D03 map.
LAYERS = (
    {"sid": "confined_T50", "storage_type": "confined", "S": 1e-4, "T": 50.0, "c": 2e5, "r": 200.0, "t95_d": 16.83218685945225},
    {"sid": "confined_T500", "storage_type": "confined", "S": 1e-4, "T": 500.0, "c": 2e5, "r": 200.0, "t95_d": 13.217981075671467},
    {"sid": "leaky_T50", "storage_type": "leaky", "S": 1e-3, "T": 50.0, "c": 2e4, "r": 200.0, "t95_d": 22.657549914651096},
    {"sid": "leaky_T500", "storage_type": "leaky", "S": 1e-3, "T": 500.0, "c": 2e4, "r": 200.0, "t95_d": 16.83218685945225},
    {"sid": "unconfined_T50", "storage_type": "unconfined", "S": 0.1, "T": 50.0, "c": 200.0, "r": 200.0, "t95_d": 57.74714073035453},
    {"sid": "unconfined_T500", "storage_type": "unconfined", "S": 0.1, "T": 500.0, "c": 200.0, "r": 200.0, "t95_d": 33.56820438729627},
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _u64(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def load_pilot_realizations() -> list[dict]:
    rows = json.loads(PILOT_CALENDAR.read_text())
    if len(rows) != N_REALIZATIONS or [r["realization"] for r in rows] != list(range(N_REALIZATIONS)):
        raise RuntimeError("pilot realization calendar is not realizations 0-9 in order")
    out = []
    for row in rows:
        out.append({
            "realization": int(row["realization"]),
            "site_stem": row["site_stem"],
            "kma_station": row["kma_station"],
            "anchor_year": ANCHOR_YEAR0 + int(row["realization"]),
            "seed_schedule": int(row["seed_schedule"]),
            "seed_natural": int(row["seed_natural"]),
            "seed_noise": int(row["seed_noise"]),
            "seed_pastas_param_sample": int(row["seed_pastas_param_sample"]),
            "offseason_use_variant": int(row["realization"]) in OFFSEASON_REALIZATIONS,
        })
    return out


def climate_path(stem: str) -> Path:
    path = CL_DIR / f"{stem}_CL.txt"
    if path.exists():
        return path
    alt = CL_DIR / (unicodedata.normalize("NFD", stem) + "_CL.txt")
    if alt.exists():
        return alt
    raise FileNotFoundError(stem)


def load_climate(stem: str) -> pd.DataFrame:
    path = climate_path(stem)
    df = pd.read_csv(path, sep="\t", dtype={"Date": str})
    if list(df.columns)[:3] != ["Date", "TEMP", "RAIN"]:
        raise RuntimeError(f"{stem}: expected Date TEMP RAIN, got {list(df.columns)}")
    df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    if df["date"].duplicated().any() or not df["date"].is_monotonic_increasing:
        raise RuntimeError(f"{stem}: dates are not a sorted unique daily index")
    gaps = df["date"].diff().dt.days.iloc[1:]
    if not bool((gaps == 1).all()):
        raise RuntimeError(f"{stem}: climate file is not a continuous daily series")
    df.attrs["sha256"] = sha256_file(path)
    df.attrs["path"] = str(path)
    df.attrs["columns"] = list(df.columns)
    return df


def rain_on(df: pd.DataFrame, day: date) -> float:
    hit = df.loc[df["date"] == pd.Timestamp(day), "RAIN"]
    if len(hit) != 1:
        raise RuntimeError(f"rain missing for {day}")
    value = float(hit.iloc[0])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite rain on {day}; not treated as dry or wet")
    return value


def warm_night(realization: int, day: date) -> bool:
    unit = _u64(f"d05-warm-v1|{realization}|{day.isoformat()}") / 2**64
    return unit < WARM_NIGHT_OFF_PROBABILITY


def maintenance_intervals(realization: int, year: int) -> list[tuple[date, int]]:
    """Two non-overlapping stops in a calendar year. Lengths are 1, 2, or 3 days."""
    lengths = (1 + (realization % 3), 1 + ((realization + 1) % 3))
    intervals = []
    for tag, length, base in (("a", lengths[0], 40), ("b", lengths[1], 220)):
        doy = base + (_u64(f"d05-maint-v1|{realization}|{year}|{tag}") % 80)
        start = date(year, 1, 1) + timedelta(days=int(doy))
        if (start + timedelta(days=length - 1)).year != year:
            start = date(year, 12, 31) - timedelta(days=length - 1)
        intervals.append((start, length))
    occupied = []
    for start, length in intervals:
        days = {start + timedelta(days=k) for k in range(length)}
        if occupied and days & set(occupied):
            raise RuntimeError("maintenance stops overlap")
        occupied.extend(days)
    return intervals


def maintenance_dayset(realization: int, years: range) -> set[date]:
    out = set()
    for year in years:
        for start, length in maintenance_intervals(realization, year):
            for k in range(length):
                out.add(start + timedelta(days=k))
    return out


def rate_on(form: str, day: date, realization: dict, rain_mm: float, maintenance: set[date]) -> float:
    if form == "water_curtain":
        if day.month in IN_SEASON_MONTHS:
            if warm_night(realization["realization"], day):
                return 0.0
            return CURTAIN_DAILY_M3D
        if realization["offseason_use_variant"]:
            return CURTAIN_DAILY_M3D * OFFSEASON_FACTOR
        return 0.0
    if form == "paddy_irrigation":
        if day.month not in PADDY_MONTHS:
            return 0.0
        if rain_mm > RAIN_OFF_THRESHOLD_MM:
            return 0.0
        return Q_NOMINAL_M3D
    if form == "domestic_continuous":
        if day in maintenance:
            return 0.0
        return Q_NOMINAL_M3D
    raise KeyError(form)


def span_for_year(year: int) -> tuple[date, date]:
    first = date(year, 1, 1)
    last_origin = date(year, 12, 1)
    return first - timedelta(days=CONTEXT_DAYS), last_origin + timedelta(days=HORIZON_DAYS - 1)


def iter_days(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def on_off_switches(q: np.ndarray) -> int:
    """Count changes between a zero day and a positive day. Positive-to-positive steps are not switches."""
    on = np.asarray(q, float) > 0.0
    return int(np.sum(on[1:] != on[:-1]))


def last_off_features(context_end: date, q: np.ndarray, t95_d: float) -> dict:
    """Last maximal zero run in the context. Distance is days after that run and before the origin.

    The origin is the day after context_end. A run that ends on context_end has distance 0.
    No zero day leaves the distance undefined; rho is then 0 because the last rest length is 0.
    """
    q = np.asarray(q, float)
    if q.size != CONTEXT_DAYS:
        raise RuntimeError("context length is not 1024")
    off = q == 0.0
    if not bool(off.any()):
        return {"last_off_length_days": 0, "last_off_distance_to_origin_days": None, "rho_last_off": 0.0}
    idx = np.flatnonzero(off)
    end_i = int(idx[-1])
    start_i = end_i
    while start_i > 0 and bool(off[start_i - 1]):
        start_i -= 1
    # Break the run if a later non-adjacent off was selected: end_i is the last off day,
    # so the run is contiguous back only while off. Days after end_i are positive.
    length = end_i - start_i + 1
    end_date = context_end - timedelta(days=CONTEXT_DAYS - 1 - end_i)
    origin = context_end + timedelta(days=1)
    distance = (origin - end_date).days - 1
    return {
        "last_off_length_days": int(length),
        "last_off_distance_to_origin_days": int(distance),
        "rho_last_off": float(length / t95_d),
    }


def future_schedules(q_origin: float) -> dict:
    """Hold the origin day's rate, including zero. Do not refill later days from the calendar."""
    q = float(q_origin)
    horizon = np.full(HORIZON_DAYS, q)
    return {
        "P1_continue_vs_stop": {"a_continue": horizon.copy(), "b_stop": np.zeros(HORIZON_DAYS)},
        "P2_current_vs_1p5x": {"a_current": horizon.copy(), "b_1p5x": np.full(HORIZON_DAYS, 1.5 * q)},
    }


def build_rate_series(form: str, realization: dict, climate: pd.DataFrame) -> pd.Series:
    start, end = span_for_year(realization["anchor_year"])
    years = range(start.year, end.year + 1)
    maintenance = maintenance_dayset(realization["realization"], years) if form == "domestic_continuous" else set()
    index = list(iter_days(start, end))
    values = [rate_on(form, day, realization, rain_on(climate, day), maintenance) for day in index]
    series = pd.Series(values, index=pd.to_datetime(index), name=form)
    return series


def slice_context(series: pd.Series, origin: date) -> tuple[date, np.ndarray, float]:
    context_end = origin - timedelta(days=1)
    context_start = origin - timedelta(days=CONTEXT_DAYS)
    window = series.loc[pd.Timestamp(context_start):pd.Timestamp(context_end)]
    if len(window) != CONTEXT_DAYS:
        raise RuntimeError(f"context for {origin} has {len(window)} days")
    q_origin = float(series.loc[pd.Timestamp(origin)])
    return context_end, window.to_numpy(float), q_origin


def case_features(form: str, realization: dict, layer: dict, origin: date, q: np.ndarray, q_origin: float, q_last: float) -> dict:
    rest = last_off_features(origin - timedelta(days=1), q, layer["t95_d"])
    zero = bool(q_origin == 0.0)
    return {
        "case_id": f"d05B_{form}_m{origin.month:02d}_r{realization['realization']:02d}_{layer['sid']}",
        "form": form,
        "origin_date": origin.isoformat(),
        "origin_month": int(origin.month),
        "anchor_year": int(realization["anchor_year"]),
        "realization": int(realization["realization"]),
        "site_stem": realization["site_stem"],
        "kma_station": realization["kma_station"],
        "sid": layer["sid"],
        "storage_type": layer["storage_type"],
        "t95_d": layer["t95_d"],
        "offseason_use_variant": bool(realization["offseason_use_variant"]) if form == "water_curtain" else False,
        "q_origin_m3d": q_origin,
        "q_last_context_day_m3d": q_last,
        "future_uses": "q_origin_including_zero",
        "n_on_off_switches": on_off_switches(q),
        "n_zero_days": int(np.sum(q == 0.0)),
        **rest,
        "zero_effect_case": zero,
        "relative_width_status": "undefined_because_origin_contrast_is_zero" if zero else "not_computed_here",
        "signal_ratio_status": "origin_rate_over_background_when_sigma_bg_is_attached",
    }


def expected_case_count() -> int:
    return len(FORMS) * N_MONTHS * len(LAYERS) * N_REALIZATIONS


def iter_cases(series_cache: dict, realizations: list[dict]):
    for realization in realizations:
        year = realization["anchor_year"]
        for form in FORMS:
            series = series_cache[(form, realization["realization"])]
            for month in range(1, 13):
                origin = date(year, month, 1)
                _end, q, q_origin = slice_context(series, origin)
                q_last = float(q[-1])
                for layer in LAYERS:
                    features = case_features(form, realization, layer, origin, q, q_origin, q_last)
                    yield features, q


def definition_self_check() -> None:
    q = np.array([1, 1, 0, 0, 5, 0, 2], float)
    if on_off_switches(q) != 4:
        raise RuntimeError("switch count treated a positive rate step as a pause or dropped a zero boundary")
    # Context ends 2010-04-30. Indices [-13:-10] are a 3-day rest; ten positive days remain before the origin.
    context = np.full(CONTEXT_DAYS, 1.0)
    context[-13:-10] = 0.0
    feat = last_off_features(date(2010, 4, 30), context, 10.0)
    if feat["last_off_length_days"] != 3 or feat["last_off_distance_to_origin_days"] != 10 or feat["rho_last_off"] != 0.3:
        raise RuntimeError(f"rest distance definition failed: {feat}")
    none = last_off_features(date(2010, 4, 30), np.ones(CONTEXT_DAYS), 10.0)
    if none["last_off_distance_to_origin_days"] is not None or none["rho_last_off"] != 0.0:
        raise RuntimeError("a context with no zero day stored a false distance")
    touching = np.ones(CONTEXT_DAYS)
    touching[-2:] = 0.0
    touch = last_off_features(date(2010, 4, 30), touching, 16.0)
    if touch["last_off_distance_to_origin_days"] != 0 or touch["last_off_length_days"] != 2:
        raise RuntimeError("a rest that reaches the origin was given a nonzero gap")
    fut = future_schedules(0.0)
    if fut["P1_continue_vs_stop"]["a_continue"].sum() != 0 or fut["P2_current_vs_1p5x"]["b_1p5x"].sum() != 0:
        raise RuntimeError("a zero origin rate was replaced by a positive continuation")
    fut = future_schedules(40.0)
    if not np.all(fut["P1_continue_vs_stop"]["b_stop"] == 0) or not np.all(fut["P2_current_vs_1p5x"]["b_1p5x"] == 60):
        raise RuntimeError("future pair does not hold the origin rate")


def _layer_gains() -> dict:
    from p3_phase2_physmap import hantush_params
    gains = {}
    for layer in LAYERS:
        h = hantush_params(layer["T"], layer["S"], layer["c"], layer["r"])
        gains[layer["sid"]] = h["gain"]
    return gains


def _sigma_bg(climate: pd.DataFrame, realization: dict, origin: date) -> float:
    import p3_generator as g
    context_start = origin - timedelta(days=CONTEXT_DAYS)
    rain = pd.Series(climate["RAIN"].to_numpy(float), index=pd.to_datetime(climate["date"]), name=realization["site_stem"])
    real = {
        "realization": realization["realization"],
        "context_start": context_start.isoformat(),
        "seed_natural": realization["seed_natural"],
        "seed_noise": realization["seed_noise"],
        "seed_schedule": realization["seed_schedule"],
    }
    bg = g.realization_background(real, rain, g.Design())
    return float(bg["sigma_bg"])


def audit() -> dict:
    definition_self_check()
    realizations = load_pilot_realizations()
    climates = {}
    climate_meta = []
    for stem in sorted({r["site_stem"] for r in realizations}):
        df = load_climate(stem)
        if "TEMP" not in df.columns:
            raise RuntimeError("TEMP column missing")
        if int(df["RAIN"].isna().sum()) != 0:
            raise RuntimeError(f"{stem}: retained RAIN has missing days")
        climates[stem] = df
        climate_meta.append({
            "site_stem": stem,
            "path": df.attrs["path"],
            "sha256": df.attrs["sha256"],
            "n_days": int(len(df)),
            "start": str(df["date"].iloc[0].date()),
            "end": str(df["date"].iloc[-1].date()),
            "columns": [c for c in df.columns if c != "date"],
            "rain_missing": int(df["RAIN"].isna().sum()),
            "temp_missing": int(df["TEMP"].isna().sum()),
            "temp_is_daily_minimum": False,
            "temp_note": "Column is named TEMP. No local file labels it as daily minimum temperature.",
        })
    series_cache = {}
    for realization in realizations:
        climate = climates[realization["site_stem"]]
        start, end = span_for_year(realization["anchor_year"])
        if pd.Timestamp(start) < climate["date"].iloc[0] or pd.Timestamp(end) > climate["date"].iloc[-1]:
            raise RuntimeError(f"anchor year {realization['anchor_year']} is outside {realization['site_stem']}")
        for form in FORMS:
            series_cache[(form, realization["realization"])] = build_rate_series(form, realization, climate)

    gains = _layer_gains()
    sigma = {}
    for realization in realizations:
        climate = climates[realization["site_stem"]]
        for month in range(1, 13):
            origin = date(realization["anchor_year"], month, 1)
            sigma[(realization["realization"], origin.isoformat())] = _sigma_bg(climate, realization, origin)

    rows = []
    for features, q in iter_cases(series_cache, realizations):
        key = (features["realization"], features["origin_date"])
        sig = sigma[key]
        gain = gains[features["sid"]]
        q0 = features["q_origin_m3d"]
        features = dict(features)
        features["sigma_bg_m"] = sig
        features["gain_m_per_m3d"] = gain
        features["signal_ratio_origin"] = float(gain * q0 / sig)
        features["signal_ratio_note"] = "gain * q_origin / sigma_bg. Zero when the origin rate is zero. Not W/|E|."
        if features["zero_effect_case"]:
            features["relative_width_status"] = "undefined_because_origin_contrast_is_zero"
        rows.append(features)
        if np.any(q < 0):
            raise RuntimeError("negative daily volume")
    if len(rows) != expected_case_count() or len({r["case_id"] for r in rows}) != expected_case_count():
        raise RuntimeError("case count is not the requested 2160 unique ids")

    # Night hours must not become extra zero days: an in-season non-warm day stays positive.
    sample_r = realizations[0]
    curtain = series_cache[("water_curtain", 0)]
    jan1 = date(sample_r["anchor_year"], 1, 1)
    if not warm_night(0, jan1):
        if float(curtain.loc[pd.Timestamp(jan1)]) != CURTAIN_DAILY_M3D:
            raise RuntimeError("a pumping night was stored as a zero day")
    # Variant versus stop in April.
    for realization in realizations:
        april = date(realization["anchor_year"], 4, 15)
        value = float(series_cache[("water_curtain", realization["realization"])].loc[pd.Timestamp(april)])
        if realization["offseason_use_variant"]:
            if value != CURTAIN_DAILY_M3D * OFFSEASON_FACTOR:
                raise RuntimeError("off-season variant is not the declared low rate")
        elif value != 0.0:
            raise RuntimeError("standard water-curtain April is not a full daily stop")
    # Irrigation follows retained rain.
    wet_checked = dry_checked = 0
    for realization in realizations:
        series = series_cache[("paddy_irrigation", realization["realization"])]
        climate = climates[realization["site_stem"]]
        for month in PADDY_MONTHS:
            for day in iter_days(date(realization["anchor_year"], month, 1), date(realization["anchor_year"], month, 28)):
                rain = rain_on(climate, day)
                q = float(series.loc[pd.Timestamp(day)])
                if rain > RAIN_OFF_THRESHOLD_MM:
                    if q != 0.0:
                        raise RuntimeError("a rain day stayed on")
                    wet_checked += 1
                else:
                    if q != Q_NOMINAL_M3D:
                        raise RuntimeError("a dry irrigation day was not at the nominal rate")
                    dry_checked += 1
                if wet_checked and dry_checked:
                    break
            if wet_checked and dry_checked:
                break
        if wet_checked and dry_checked:
            break
    if wet_checked == 0 or dry_checked == 0:
        raise RuntimeError("the retained May–August rain did not contain both a wet and a dry day")
    # Domestic stops are 1–3 days, two per year, and the only zeros.
    for realization in realizations:
        year = realization["anchor_year"]
        intervals = maintenance_intervals(realization["realization"], year)
        if len(intervals) != 2 or any(length < 1 or length > 3 for _s, length in intervals):
            raise RuntimeError("maintenance rule is not two stops of 1–3 days")
        series = series_cache[("domestic_continuous", realization["realization"])]
        year_q = series.loc[f"{year}-01-01":f"{year}-12-31"]
        if int(np.sum(year_q.to_numpy(float) == 0.0)) != sum(length for _s, length in intervals):
            raise RuntimeError("domestic zeros are not exactly the maintenance stops")

    zero_n = sum(r["zero_effect_case"] for r in rows)
    if zero_n == 0 or zero_n == len(rows):
        raise RuntimeError("zero-effect cases were dropped or every case was zero")
    variant_n = sum(r["offseason_use_variant"] for r in rows)
    # Re-slice once and compare one case byte for byte.
    again = next(iter_cases(series_cache, realizations))[0]["case_id"]
    if again != rows[0]["case_id"]:
        raise RuntimeError("case order is not reproducible")

    source_files = {
        "author instruction": PHASE4 / "author instruction",
        "field_pause_sources.json": ROOT / "results/phase3/field_pause_sources.json",
        "field_pause_access_supplement.md": ROOT / "results/phase3/field_pause_access_supplement.md",
        "relevant_operation_paragraphs.txt": ROOT / "results/phase3/sources/mdpi_w13010051/relevant_operation_paragraphs.txt",
        "miryang2022.pdf": ROOT / "results/phase3/sources/miryang2022.pdf",
        "watercurtain2022.pdf": ROOT / "results/phase3/sources/watercurtain2022.pdf",
        "realization_calendar.json": PILOT_CALENDAR,
        "p4_calendar.py": Path(__file__),
    }
    return {
        "component": "D05 B calendar and weather preparation",
        "turn_id": "run",
        "slot": "executor",
        "protocol_frozen": False,
        "models_scored": False,
        "W_scored": False,
        "n_cases": len(rows),
        "n_forms": len(FORMS),
        "n_origins_per_realization": N_MONTHS,
        "n_layers": len(LAYERS),
        "n_realizations": N_REALIZATIONS,
        "count_identity": "3 forms x 12 monthly origins x 6 layers x 10 realizations = 2160",
        "n_zero_effect_cases": zero_n,
        "n_water_curtain_offseason_variant_cases": variant_n,
        "zero_effect_cases_retained": True,
        "relative_width_not_imputed": True,
        "nightly_hours_not_mapped_to_daily_pause": True,
        "rain_days_checked_wet": wet_checked,
        "rain_days_checked_dry": dry_checked,
        "curtain_daily_m3d": CURTAIN_DAILY_M3D,
        "warm_night_off_probability": WARM_NIGHT_OFF_PROBABILITY,
        "warm_night_probability_source": "self-imposed fallback; tmin is absent; not taken from Kang's 21.1 percent trial reduction",
        "rain_off_threshold_mm": RAIN_OFF_THRESHOLD_MM,
        "rain_rule": "paddy May-August day is off when retained daily RAIN is strictly greater than 0 mm",
        "offseason_realizations": list(OFFSEASON_REALIZATIONS),
        "offseason_factor_of_in_season_daily_rate": OFFSEASON_FACTOR,
        "gains_m_per_m3d": gains,
        "sigma_bg_min_m": min(sigma.values()),
        "sigma_bg_max_m": max(sigma.values()),
        "climate": climate_meta,
        "sources_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "cases": rows,
    }


def write_audit(payload: dict) -> dict:
    cases = payload["cases"]
    summary = {k: v for k, v in payload.items() if k != "cases"}
    summary["case_id_sha256"] = hashlib.sha256("\n".join(r["case_id"] for r in cases).encode()).hexdigest()
    summary["feature_table_sha256"] = hashlib.sha256(
        json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (PHASE4 / "CALENDAR_INPUT_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (PHASE4 / "CALENDAR_CASES.json").write_text(json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n")
    prep = {
        "status": "preparation_only",
        "protocol_frozen": False,
        "parent_scope": "full D05 A/B/C remains with the parent plan",
        "generator": "lib/p4_calendar.py",
        "rules": summary,
    }
    # The case table is the auditable input. The preparation object points at it.
    prep["rules"].pop("cases", None)
    (PHASE4 / "CALENDAR_PREPARATION.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "lib"))
    summary = write_audit(audit())
    print(json.dumps({k: summary[k] for k in (
        "n_cases", "n_zero_effect_cases", "n_water_curtain_offseason_variant_cases",
        "curtain_daily_m3d", "sigma_bg_min_m", "sigma_bg_max_m", "case_id_sha256", "feature_table_sha256",
    )}, indent=2))
