import os,json,subprocess,time,datetime,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
H=P.parent
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(d):
 q=P/'COMPONENT_W_PROGRESS.json'; t=q.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2));t.replace(q)
e=json.loads((H/'EXECUTION_PACKET.json').read_text()); w=e['commands']['W']
r=json.loads((P/'generated/GENERATION_RECEIPT.json').read_text())
assert r['n']==len(r['records'])==3726
assert len({(x['case_id'],x['variant']) for x in r['records']})==3726
assert r['protocol_sha256']==e['protocol_sha256']
for x in r['records']:
 assert hashlib.sha256(Path(x['tf_input_path']).read_bytes()).hexdigest()==x['tf_input_sha256']
started=now(); procs=[]; logs=[]
for i,argv in enumerate(w['parallel_argvs']):
 log=(P/f'COMPONENT_W_SHARD_{i}.log').open('w');logs.append(log)
 procs.append(subprocess.Popen(argv,env={**os.environ,**w['env']},cwd=w['cwd'],stdout=log,stderr=subprocess.STDOUT))
while True:
 codes=[p.poll() for p in procs]; files=list((P/'W').glob('*/*.json'))
 d=dict(turn_id='run',worker='executor',protocol_sha256=e['protocol_sha256'],started_utc=started,updated_utc=now(),shard_pids=[p.pid for p in procs],returncodes=codes,completed_variant_files=len(files),expected=150,commands=w)
 save(d); print(json.dumps({k:d[k] for k in ['updated_utc','returncodes','completed_variant_files']}),flush=True)
 if all(c is not None for c in codes): break
 time.sleep(30)
for log in logs:log.close()
if any(c!=0 for c in codes): raise SystemExit(1)
a=e['commands']['aggregate_W']
with (P/'COMPONENT_W_AGGREGATE.log').open('w') as log:
 code=subprocess.run(a['argv'],env={**os.environ,**a['env']},cwd=a['cwd'],stdout=log,stderr=subprocess.STDOUT).returncode
d['aggregate_returncode']=code;d['ended_utc']=now();save(d)
raise SystemExit(code)
