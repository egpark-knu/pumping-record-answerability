"""D03 Phase 2 preflight: verify the approved D02 base (protocol, v1.2 repair, code) by hash, read-only.

Writes results/phase2/B_preflight/base_hash_check.json. Mutates nothing under results/pilot or lib/p3_*.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
OUT = ROOT / "results/phase2/B_preflight/base_hash_check.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows, ok = [], True
    pf = json.load(open(ROOT / "results/pilot/protocol_freeze.json"))
    rf = json.load(open(ROOT / "results/pilot/repair_freeze.json"))
    named = {"case_schema": "results/pilot/case_schema.json", "realization_calendar": "results/pilot/phase0_checks/realization_calendar.json",
             "prep_checks": "results/pilot/prep_checks/prep_checks.json"}
    checks = [("protocol_freeze", "results/pilot/protocol.md", pf["protocol_sha256"])]
    for k, v in pf["hashes"].items():
        p = k if k.startswith("lib/") else named.get(k)
        if p:
            checks.append(("protocol_freeze", p, v))
    for k, v in rf["hashes"].items():
        if (ROOT / k).exists():
            checks.append(("repair_freeze", k, v))
    for src, p, want in checks:
        got = sha(ROOT / p)
        rows.append(dict(source=src, path=p, expected=want, actual=got, match=got == want))
        ok &= got == want
    for name, meta in pf["rain_sources"].items():
        got = sha(Path(meta["path"]))
        rows.append(dict(source="protocol_freeze.rain", path=meta["path"], expected=meta["sha256"], actual=got, match=got == meta["sha256"]))
        ok &= got == meta["sha256"]
    rec = dict(all_match=bool(ok), n_checked=len(rows), n_mismatch=sum(not r["match"] for r in rows),
               protocol_sha256_file=(ROOT / "results/pilot/protocol.sha256").read_text().split()[0], rows=rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}))


if __name__ == "__main__":
    main()
