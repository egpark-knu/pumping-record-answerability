"""Freeze D04 B sources before official generation and scoring; keep inherited D03 hashes."""
import datetime,json
from pathlib import Path
import p3_phase3_cases as pc
ROOT=pc.ROOT;P3=pc.P3
if (P3/'protocol_freeze.json').exists():raise PermissionError('Already frozen; do not silently replace')
fx=json.loads((P3/'B_fixture/cases/generation_checks.json').read_text());assert fx['all_ok'] and fx['wrong_origin_rate_rejected']
d03=json.loads((pc.d03.P2/'protocol_freeze.json').read_text());hashes={p:h for group in d03['hashes'].values() for p,h in group.items()}
for p,h in hashes.items():assert pc.sha(ROOT/p)==h,p
for p in sorted((ROOT/'lib').glob('p3_phase3_*.py')):
    if p.name in ('p3_phase3_A_tests.py','p3_phase3_normalization.py') or 'timesfm' in p.name:continue
    hashes[str(p.relative_to(ROOT))]=pc.sha(p)
for name in ('protocol.md','author instruction','INPUT_CONTRACT.md'):
    p=P3/name;hashes[str(p.relative_to(ROOT))]=pc.sha(p)
# Snapshot every D03 artifact, not only algorithms. No inherited outputs are regenerated or rewritten.
snapshot={str(p.relative_to(ROOT)):pc.sha(p) for p in sorted(pc.d03.P2.rglob('*')) if p.is_file()}
(P3/'B_D03_READONLY_SNAPSHOT.json').write_text(json.dumps(snapshot,indent=1))
digest=pc.sha(P3/'protocol.md');record=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),role='executor',turn_id='run',protocol_sha256=digest,hashes=hashes,d03_snapshot_sha256=pc.sha(P3/'B_D03_READONLY_SNAPSHOT.json'),fixture_checks_sha256=pc.sha(P3/'B_fixture/cases/generation_checks.json'),n_d03_snapshot_files=len(snapshot))
(P3/'protocol.sha256').write_text(digest+'  protocol.md\n');(P3/'protocol_freeze.json').write_text(json.dumps(record,indent=1));print(json.dumps({k:v for k,v in record.items() if k!='hashes'}));pc.check_frozen()
