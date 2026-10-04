# Data and code for: When can a pumping record answer 'what if we stop?'

Eungyu Park¹² (ORCID [0000-0002-2293-4686](https://orcid.org/0000-0002-2293-4686)), Jangwon Park¹², Taeyu Kim¹²

¹ School of Earth System Sciences, Kyungpook National University, Daegu 41566, Republic of Korea
² GeoAI Alignment, Inc., Daegu 41544, Republic of Korea

Corresponding author: Eungyu Park (egpark@knu.ac.kr)

## 1. Summary

Operating wells keep pumping and head records, and these records are increasingly asked what would happen if pumping stopped or increased. In twin experiments with six leaky-aquifer settings, measured daily rainfall and autocorrelated noise, we measured how much a record constrains the head difference between two future pumping schedules among all candidate responses that still fit it. A record answers a what-if question when the head change caused by its pumping changes is large against background variability at the observation point, and when those changes last about as long as the aquifer takes to respond. A pause four response times long roughly halves the signal ratio a record needs to detect a pumping effect. On operating calendars typical of water-curtain, paddy and domestic wells, observing at 20 m instead of 200 m raised detection from 54% to 86% of active decision days. A pretrained time-series forecaster used as a scenario engine barely responded to a pumping increase beyond the rates in its history, and one week at that rate restored the response. This repository contains the code and data needed to rebuild the paper's Table 3, figures and principal numbers, and the code and protocols of the full computation.

## 2. Repository structure

```
README.md, CITATION.cff, LICENSE (MIT, code), LICENSE-DATA.md (CC BY 4.0, data)
requirements.txt             quick reproduction (numpy, matplotlib, pillow)
requirements-pipeline.txt    full computation, science environment (versions of the original runs)
requirements-engine.txt      full computation, forecasting-engine environment
reproduce/                   quick reproduction (Table 3, principal numbers, figures)
data/                        aggregate and case-level tables used by reproduce/ (see data/SOURCES.json)
  cases/                     case_level_summary.csv: every record x question x lead
  detection/                 detection and magnitude models, Table 3 crossings, concordance
  calendars/                 calendar detection fractions and 200 m -> 20 m transitions
  engine/                    engine sign checks, signal-ratio bins, rate-coverage groups
  intervention/              history-intervention medians, intervals and paired contrasts
  masks/                     observation-mask width ratios and three representative pumping records
  figure1/                   fixed candidate curves of Figure 1
  scale/, linearized/        inputs of Figure S19
  storage/                   matched-response-time storage comparison (Table S26)
  SOURCES.json               SHA-256, size and origin of every data file
  statistics/                realization-cluster bootstrap draw matrix (999 x 10)
  climate/                   daily climate tables for KMA ASOS stations 136, 289, 295 (KOGL Type 1, see section 7)
pipeline/                    full computation: library code, stage scripts and protocols
  run_pipeline.py            checks, lists and runs the stages (section 5)
  lib/                       generator, response family and envelope W, statistics, linearization, engine drivers, figures
  results/<stage>/           stage scripts and the protocols fixed before outcomes were computed
```

The full computation keeps its original stage directory names because the code addresses files through them. The table maps them to the record sets and analyses of the paper. Labels such as `D03` or `d05B_` that appear in file names, case identifiers or data columns follow the same map.

| Stage directory | Internal label | Record sets and analyses |
| --- | --- | --- |
| `pilot` | D02 | pilot cases and response-kernel / envelope checks |
| `phase2` | D03 | pause–transition set |
| `phase3` | D04A, D04B, D04C | count–recency set, pumping-scale set, storage set |
| `phase4` | D05A, D05B | map-extension set; operating calendars at 200 m; answerability map and bootstrap |
| `phase5` | D06A, D06 B/C | operating calendars at 20 m; time-scale bundles; linearized widths and observation masks |
| `phase6` | D07, D08 | engine robustness; history intervention (higher-rate, nominal-rate and zero-rate blocks) |
| `phase7` | D09 | supporting analyses of stored results (paired contrasts by operation, lead discordance, calibration) |
| `phase8` | D10 | zero-effect (null) fits, effect detection, inclusion and aggregation |
| `phase10` | D12 | matched-response-time storage comparison (Table S26) |

Engine shards in the original runs were labelled `shard_a` and `shard_b`; they are disjoint, equal partitions of the cases.

## 3. Environment

The quick reproduction needs Python 3.12 and the packages pinned in `requirements.txt`:

```sh
python3.12 -m venv .venv && . .venv/bin/activate     # or: conda create -n par python=3.12
python -m pip install -r requirements.txt            # numpy 2.4.4, matplotlib 3.10.9, pillow 12.2.0
```

Figures use Times New Roman when it is installed and a serif font otherwise. The full computation used the versions in `requirements-pipeline.txt` (Python 3.11, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, Pastas 2.0.0) and, for the forecasting engine, `requirements-engine.txt` (TimesFM 3.0.2 with torch 2.14.0); see section 5.

## 4. Quick reproduction

From the repository root:

```sh
python -m reproduce.run_quick
```

This writes to `outputs/`:

- `table3.csv`, `table3.md`: Table 3. Detection and magnitude crossings are recomputed from the archived fitted coefficients; intervals are the archived 999-draw realization-cluster bootstrap endpoints. The script checks the crossings against `data/detection/table3_crossings.csv`.
- `principal_numbers.json`: 26 numbers quoted in the Abstract and Results (record counts, detection crossings, the pause exchange ratio 0.475, calendar detection 54.0% and 86.4%, engine ratios 0.117 and 0.889, range contrast 0.471 versus 0.092, paddy contrast 0.430 versus 0.002, matched-storage ratios), each compared with the value printed in the manuscript.
- `figures/figure1.png` … `figure6.png` and `figureS19.png` (with PDF copies): main Figures 1–6 and Supporting Information Figure S19. The manuscript has six main figures; Figure S19 is the former seventh figure (W against pumping scale and linearized width), moved to the Supporting Information before submission.
- `quick_check.json`: pass/fail summary, timing and package versions.

No fit, bootstrap, simulation or forecast is run. Tested on 2026-10-04 in a newly created conda environment (Python 3.12.15, `--no-default-packages`, user site-packages disabled) on an Apple-silicon Mac: installing the three pinned packages took 4 s from a warm pip cache, and `python -m reproduce.run_quick` finished in 8.2 s with Table 3 identical to the archived table (maximum relative difference 0), 26 of 26 principal numbers equal to the manuscript values, and all seven figures written.

## 5. Full reproduction

The full computation regenerates the synthetic records and recomputes envelopes, forecasts, reference fits, zero-effect fits and statistics. It was not rerun for this release. The procedure below was checked on 2026-10-04 up to the start of the computation: `python pipeline/run_pipeline.py --check` passed all its checks (freeze checks, imports, startup of the entrypoints, checkpoint and TimesFM source hashes) in a new Python 3.11 environment built from `requirements-pipeline.txt`, with the data bundle extracted as in step 2; engine-step imports were checked in the original engine environment.

**1. Environments.** From the repository root:

```sh
python3.11 -m venv .venv-science
.venv-science/bin/python -m pip install -r requirements-pipeline.txt

git clone https://github.com/google-research/timesfm.git
git -C timesfm checkout e31dadd84cb26bd5153fde6687502b8312e918fb
python3.11 -m venv .venv-engine
.venv-engine/bin/python -m pip install -r requirements-engine.txt
.venv-engine/bin/python -m pip install --no-deps -e ./timesfm
```

Download the TimesFM 3.0 checkpoint as in section 6. The original engine runs used the Apple MPS device (Apple-silicon Mac); the engine drivers of phases 3–6 check the checkpoint's weight and configuration hashes before loading it.

**2. Data bundle.** Download `pumping-record-answerability-data.zip` from the Zenodo record and extract it into `pipeline/`:

```sh
unzip pumping-record-answerability-data.zip
cp -R pumping-record-answerability-data/results pipeline/
```

`pipeline/results/<stage>/` then holds, next to the code and protocols, the stored stage outputs (record-level tables, cohort and case lists, engine and reference outputs, statistics, zero-effect fits) and the stage inputs that are not regenerated (cohort manifests, the 999 × 10 bootstrap draw matrix, the history-intervention record index, the stored supporting analyses). The synthetic case arrays and the per-case envelope searches are not archived; the stages regenerate them from the generator, the seeds in the protocols and the climate tables, and overwrite the stored outputs as they go. Keep an unmodified copy of the bundle to compare with.

**3. Locations used by the code:**

```sh
export PUMPING_RECORD_ROOT="$PWD/pipeline"
export KMA_CLIMATE_DIR="$PWD/data/climate"
export TIMESFM_SNAPSHOT="$HOME/.cache/huggingface/hub/models--google--timesfm-3.0-pytorch/snapshots/43046b85ec22d584a13f8098c2ed39c889e129c2"
export TIMESFM_SOURCE="$PWD/timesfm/src/timesfm3"
export ENGINE_PYTHON="$PWD/.venv-engine/bin/python"
```

`TIMESFM_SNAPSHOT` is the snapshot directory written by the download of section 6 (the path above is the default Hugging Face cache).

**4. Check, then run the stages in order:**

```sh
.venv-science/bin/python pipeline/run_pipeline.py --check     # freezes, imports, entrypoint startup; no computation
.venv-science/bin/python pipeline/run_pipeline.py --list      # every command, numbered
.venv-science/bin/python pipeline/run_pipeline.py --stage phase2
.venv-science/bin/python pipeline/run_pipeline.py --stage phase3
.venv-science/bin/python pipeline/run_pipeline.py --stage phase4
.venv-science/bin/python pipeline/run_pipeline.py --stage phase5
.venv-science/bin/python pipeline/run_pipeline.py --stage phase6
.venv-science/bin/python pipeline/run_pipeline.py --stage phase8
.venv-science/bin/python pipeline/run_pipeline.py --stage phase10
```

`--stage` runs the commands of a stage in `pipeline/`, science steps with the Python running the wrapper and engine steps with `ENGINE_PYTHON`, and stops at the first failure; `--stage NAME --from N` resumes at command N. The commands are (`k` = shard index; every shard of a loop must run, in any order or in parallel):

| Stage | Commands, in order (run in `pipeline/`; [E] = engine environment) |
| --- | --- |
| `phase2` | `lib/p3_phase2_make_cases.py --out results/phase2/cases`; [E] `lib/p3_phase2_timesfm_execution.py`; `lib/p3_phase2_run_wb.py --shard k --nshards 8` for k = 0–7; `lib/p3_phase2_collect_wb.py`; `lib/p3_phase2_cell_metrics.py`; `lib/p3_phase2_analysis.py` |
| `phase3` | `lib/p3_phase3_make_cases.py`; `lib/p3_phase3_run_wb.py --shard k --nshards 8` for k = 0–7; `lib/p3_phase3_collect_wb.py`; [E] `lib/p3_phase3_timesfm_execution.py`; [E] `lib/p3_phase3_normalization_run.py`; `lib/p3_phase3_analysis.py` |
| `phase4` | `lib/p4_make_cases.py`; `lib/p4_run_wb.py --actor A --shard k --nshards 8` for A = `shard_a`, `shard_b` and k = 0–7; [E] `lib/p4_timesfm_execution.py --actor A` for both A; `results/phase4/scripts/run_statistics.py`; `lib/p4_map_supplement_figures.py`; `lib/p4_calendar_contradiction_figures.py` |
| `phase5` | `lib/p5_distance_cases.py --mode production --regen-parents`; `lib/p5_run_wb.py --mode production --actor A --shard k --nshards 8` for both A and k = 0–7; [E] `lib/p5_timesfm_execution.py --mode production --actor A` for both A; `lib/p5_collect.py --mode production_A`; `lib/p5_linearized.py --production --cases results/phase5/C_CASES.json --out results/phase5/C --shard k --nshards 16` for k = 0–15; `lib/p5_linearized.py --merge --cases results/phase5/C_CASES.json --out results/phase5/C`; `lib/p5_timescale.py --mode production`; `lib/p5_analysis.py --production`; `lib/p5_figures.py --production` |
| `phase6` | `results/phase6/D08/protocol_neutral/d08_runtime.py` with, in order: `preflight`; `selftest`; `A --execute`; `generate --execute`; [E] `B --execute`; `W --execute --shard k --nshards 8` for k = 0–7; `aggregate_B --execute`; `aggregate_W --execute` |
| `phase8` | `results/phase8/scripts/prepare.py`; `results/phase8/scripts/null_fit.py`; `results/phase8/scripts/correct_null.py`; `results/phase8/scripts/aggregate.py`; `results/phase8/B/calculate_inclusion.py` |
| `phase10` | `results/phase10/calculations/additional2_calculations.py` |

Order matters: later stages read the cases and outputs of earlier stages (phase3 reads phase2 parents; the 20 m calendars are rebuilt from the 200 m calendars; the history intervention and the zero-effect fits read the envelopes of phases 2–5; Table S26 reads the phase3 cases and the phase8 tables). Before its first command, `--stage phase8` moves the stored zero-effect fits of the bundle to `pipeline/stored/`, because `null_fit.py` would otherwise resume from them. `prepare.py` repeats the phase8 freeze step of the original run: it writes the record manifest of the regenerated inputs, `protocol.md` (same text, new freeze time) and the pins that the later phase8 scripts check. Seeds, station windows, bounds, tolerances and bootstrap settings are fixed in the protocols under `pipeline/results/<stage>/protocol.md`.

Not rerun by `run_pipeline.py`; their stored outputs are in the data bundle: the pilot (`pilot`, method development, not an input of later stages; procedure in `results/pilot/protocol.md` section 15), the engine robustness replay (`results/phase6/engine`) and the supporting analyses (`results/phase7/D09`). The scripts of the last two also check run records of the original analyses that are not part of this release; phase8 reads their stored outputs as inputs.

Approximate computing time. The recorded durations of the original runs on one Apple-silicon workstation were about 1–2 h per stage for record generation, envelope search, engine inference and reference fits, with engine and envelope work split into two shards of up to eight processes each; the whole chain took roughly 10–15 h of wall-clock time. Envelope search and engine inference dominate.

Notes on the public edition. Paths are taken from the environment variables above instead of the authors' local directories, and executor labels and references to internal project instructions in code and protocols were neutralized; the scientific code and settings are otherwise unchanged. The `protocol.md` files are therefore public editions of the protocols fixed before outcomes were computed, and their checksums differ from the originals. The freeze records (`protocol.sha256`, `protocol_freeze.json` and the corresponding records of phases 4–8) and the hash constants in the code that pin protocols or input tables were rewritten to pin the public files, so the stage scripts run their original freeze checks unchanged against this release; pins on files that are not released (internal instructions, run records, bulk arrays that the stages regenerate) were removed. The original values and every replacement are listed in the build evidence of the release. Compare a rerun with the stored outputs of the data bundle and with `data/`. Forecasts depend on the exact checkpoint and runtime, so they need not reproduce bit for bit.

## 6. Forecasting engine

The engine is TimesFM 3.0 (Google Research). Its weights are not included here. Download them from the official model page at the revision used in the paper:

```sh
.venv-engine/bin/hf download google/timesfm-3.0-pytorch --revision 43046b85ec22d584a13f8098c2ed39c889e129c2   # engine environment of section 5
```

The weights are distributed under the TimesFM Non-Commercial License v1.0 (https://huggingface.co/google/timesfm-3.0-pytorch), which permits use for non-commercial testing, evaluation and research; read it before downloading. The TimesFM source code (https://github.com/google-research/timesfm) is Apache-2.0. Inference settings (patch sizes, normalization, detrending, quantile options, batch size) are listed in Text S14 of the Supporting Information and in `pipeline/lib/*timesfm*`. The engine code imports the TimesFM 3.0 source (`timesfm3`) of google-research/timesfm at commit `e31dadd84cb26bd5153fde6687502b8312e918fb` (installation in section 5).

## 7. Rainfall data

Daily climate tables for three Korea Meteorological Administration (KMA) ASOS stations are in `data/climate/`: Andong (station 136, `안동태화_충적_CL.txt`), Sancheong (289, `산청산청_암반_CL.txt`) and Namhae (295, `남해남해_암반_CL.txt`), 2005-01-01 to 2024-12-31, with columns Date, TEMP, RAIN, HUMID, HPA and WIND. The rainfall column drives the natural head; the other columns are not used. Source: Korea Meteorological Administration, ASOS daily data (기상청_지상(종관, ASOS) 일자료 조회서비스, https://www.data.go.kr/data/15059093/openapi.do), released under the Korea Open Government License Type 1 (공공누리 제1유형: attribution; commercial and non-commercial use and modification permitted). These tables are not covered by this repository's CC BY 4.0 data license; reuse them under KOGL Type 1 with attribution to KMA.

The record generator accepts only these exact tables (SHA-256 `20dddf7f…5128a`, `e95466d5…d9e30`, `7733e135…7c521`; full values in `data/SOURCES.json`; the generator checks the 12-character prefixes in `pipeline/lib/p3_phase2_cases.py`). A new download from the KMA portal (https://data.kma.go.kr, daily ASOS, stations 136, 289, 295) contains the same observations but a different file layout, so it will not match these hashes.

## 8. Citation and license

Please cite the paper and this archive (see `CITATION.cff`). The DOI 10.5281/zenodo.23131441 is reserved for this archive and resolves once the Zenodo record is published.

> Park, E., Park, J., & Kim, T. (2026). Data and code for: When can a pumping record answer 'what if we stop?' (Version v1.0.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23131441

- Code (`reproduce/`, `pipeline/`): MIT License (`LICENSE`).
- Data (`data/` except `data/climate/`, and the Zenodo data bundle): CC BY 4.0 (`LICENSE-DATA.md`).
- `data/climate/`: KMA data under KOGL Type 1 (section 7).
- TimesFM 3.0 weights: not included; TimesFM Non-Commercial License v1.0 (section 6).

Funding: National Research Foundation of Korea (NRF) grant funded by the Korea government (Ministry of Science and ICT, MSIT), Grant RS-2023-002772264.
