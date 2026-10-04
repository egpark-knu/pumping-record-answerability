"""Poll-able status of the official Phase 2 W/reference/truth run: writes results/phase2/wb/status.json. Read-only on outputs."""
import json, time
from collections import Counter
from pathlib import Path
ROOT = Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])))
WB = ROOT / "results/phase2/wb"
man = json.load(open(ROOT / "results/phase2/cases/derived_manifest.json"))
ids = [m["case_id"] for m in man]
rows = {}
for cid in ids:
    p = WB / "markers" / f"{cid}.done"
    if p.exists():
        rows[cid] = json.loads(p.read_text())
st = dict(updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_expected=len(ids), n_done=len(rows),
          W_status=dict(Counter(r.get("W_status") for r in rows.values())), reference_status=dict(Counter(r.get("reference_status") for r in rows.values())),
          truth_eval_ok=sum(bool(r.get("truth_eval_ok")) for r in rows.values()),
          flags=dict(Counter(f for r in rows.values() for f in r.get("flags", []))),
          wall_s_median=(sorted(r["total_wall_s"] for r in rows.values())[len(rows) // 2] if rows else None),
          remaining=[c for c in ids if c not in rows][:20], n_remaining=len(ids) - len(rows))
(WB / "status.json").write_text(json.dumps(st, indent=1)); print(json.dumps(st))
