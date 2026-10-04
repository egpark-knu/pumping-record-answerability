"""Freeze the D03 Phase 2 protocol (B): protocol.sha256 + protocol_freeze.json (sources, decisions, code, evidence hashes).
Refuses if the production checks did not pass. Writes nothing else."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
P2 = ROOT / "results/phase2"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).relative_to(ROOT))


sc = json.load(open(P2 / "B_merge/schedule_check.json"))
mt = json.load(open(P2 / "B_merge/merge_tests.json"))
base = json.load(open(P2 / "B_preflight/base_hash_check.json"))
if not (sc["all_invariants_pass"] and all(v["detected"] for v in sc["negatives"].values()) and mt["all_passed"] and base["all_match"]):
    raise SystemExit("production checks not all passing: refusing to freeze")
if (P2 / "protocol.sha256").exists():
    raise SystemExit("protocol.sha256 already exists: a frozen protocol is never overwritten")
digest = sha(P2 / "protocol.md")
groups = {
    "authority": [P2 / "author instruction", ROOT / "direction/D03_record_features_and_engine_response.md", P2 / "LATEST_STEERING.md",
                  P2 / "ROOT_DESIGN_DECISION.json", P2 / "D03_design_debate_recovery.json", P2 / "SOURCE_PACKET.md", ROOT / "README.md"],
    "sources_C": [P2 / "sources/C_physical_followup.md", P2 / "sources/C_physical_followup_manifest.json", P2 / "sources/C_physical_sources.md",
                  P2 / "sources/source_manifest.json"] + sorted((P2 / "sources/primary/followup").glob("*")),
    "interface_A": [ROOT / "lib/p3_phase2_timesfm.py", ROOT / "lib/p3_phase2_timesfm_tests.py", ROOT / "lib/timesfm_pilot_adapter.py",
                    P2 / "A_preflight_manifest.json", P2 / "A_preflight_tests.log"],
    "figures_C": [ROOT / "lib/p3_phase2_figures.py", P2 / "FIGURE_INPUT_CONTRACT.json", P2 / "FIGURE_INPUT_CONTRACT.md"],
    "frozen_D02_code_imported": [ROOT / f"lib/{f}" for f in ("p3_kernels.py", "p3_generator.py", "p3_make_cases.py", "p3_wenvelope.py",
                                                             "p3_wenvelope_repaired_v1_2.py", "p3_wb_collect_repaired_v1_2.py", "p3_pastas.py")]
                                + [ROOT / "results/pilot/protocol.md", ROOT / "results/pilot/protocol_W_repair_addendum_v1_2.md", ROOT / "results/pilot/repair_freeze.json",
                                   ROOT / "results/pilot/phase0_checks/realization_calendar.json"],
    "phase2_code_B": sorted(p for p in (ROOT / "lib").glob("p3_phase2_*.py") if p.name not in ("p3_phase2_timesfm.py", "p3_phase2_timesfm_tests.py", "p3_phase2_figures.py"))
                     + [ROOT / "lib/tests/p3_phase2_merge_tests.py", ROOT / "lib/tests/p3_phase2_engine_smoke.py"],
    "evidence_B": [P2 / "B_merge/schedule_check.json", P2 / "B_merge/schedule_cases.csv", P2 / "B_merge/merge_tests.json", P2 / "B_merge/engine_smoke.json",
                   P2 / "B_merge/A_interface_tests_rerun.log", P2 / "B_merge/export_sample_H10.npz", P2 / "B_merge/cell_metrics_contract_sample_synthetic.csv",
                   P2 / "B_preflight/base_hash_check.json", P2 / "B_preflight/unit_fixtures.json", P2 / "B_preflight/sigma_bg.json",
                   P2 / "B_preflight/stratum_scan.json", P2 / "B_preflight/toy_budget.json", P2 / "B_phase0.md"],
}
rec = dict(protocol="results/phase2/protocol.md", protocol_sha256=digest, version="1.0",
           frozen_at=datetime.now().astimezone().isoformat(timespec="seconds"), frozen_by="executor (B)",
           official_cases_generated_before_freeze=False, official_scores_computed_before_freeze=False,
           pre_freeze_runs=["production schedules with actual seeds (no heads): B_merge/schedule_check.json",
                            "FIXTURE-rain production cases + full-grid export + negatives: B_merge/merge_tests.json (8/8)",
                            "A interface tests rerun unchanged: B_merge/A_interface_tests_rerun.log (10/10)",
                            "FIXTURE engine smoke (W v1.2 enlarged budget, Pastas): B_merge/engine_smoke.json",
                            "preflight toy runs on FIXTURE rain: B_preflight/toy_check.json, toy_budget.json"],
           checks=dict(schedule_invariants_540=sc["all_invariants_pass"], schedule_negatives_detected=all(v["detected"] for v in sc["negatives"].values()),
                       merge_tests=f"{mt['n_passed']}/{mt['n_tests']}", d02_base_hash_match=f"{base['n_checked'] - base['n_mismatch']}/{base['n_checked']}"),
           design=dict(r_m=200, Q_nominal_m3_d=100, a_star_d=20, c_rule="c=a*/S", T_m2_d=[50, 500], storage=[1e-4, 1e-3, 0.1],
                       rest_end_rule="e=1024-(floor(t95)+1)-18", rest_rule="R=round(rho*t95), rest=[e-R,e)",
                       prefix="ON e, OFF e+1,e+3,...,e+N-1 (1 d each)", compensation="C=[0,e-Rmax), kappa=1+sum(base0[OFF])/sum(base0[C])",
                       budget=dict(global_levels=[128, 256, 512, 1024], n_local=512, max_local_rounds=12, max_restarts=4, conv_tol=0.05, min_accepted=200, W_seed=20260930),
                       TOL_IN_m=1e-12, W_floor_m=1e-6, small_flags_m=[0.001, 0.02], tracks=["raw", "exp10", "exp30", "exp90"], horizons=[10, 30]),
           hashes={k: {rel(p): sha(p) for p in v if Path(p).is_file()} for k, v in groups.items()})
(P2 / "protocol.sha256").write_text(f"{digest}  protocol.md\n")
(P2 / "protocol_freeze.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
print(digest, rec["frozen_at"], {k: len(v) for k, v in rec["hashes"].items()})
