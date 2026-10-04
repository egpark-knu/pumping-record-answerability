"""Independent serialized-input/output audit; no scoring, reruns or mutation of frozen sources."""
import argparse,datetime,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import p3_phase3_cases as pc
import p3_phase3_export as ex
from p3_make_cases import load_tf_input,TRUTH_KEYS
ROOT=pc.ROOT;P3=pc.P3

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outputs',action='store_true');a=ap.parse_args();digest=pc.check_frozen()
    man=json.loads((P3/'cases/derived_manifest.json').read_text());checks=defaultdict(lambda:True);raw=[];calendar=json.loads((ROOT/'results/pilot/phase0_checks/realization_calendar.json').read_text());rains={s:pc.g.load_rain(s,h) for s,h in pc.d03.SITE_SHA.items()};cal={int(r['realization']):r for r in calendar}
    for m in man:
        tf=P3/'cases'/m['tf_input'];tr=P3/'cases'/m['truth'];ti=load_tf_input(tf)
        checks['input_hashes'] &= pc.sha(tf)==m['tf_sha256'] and pc.sha(tr)==m['truth_sha256']
        checks['protocol_stamp'] &= ti['protocol_sha256']==digest
        with np.load(tf,allow_pickle=False) as z:checks['input_truth_free'] &= not(set(z.files)&TRUTH_KEYS)
        rr=cal[m['realization']];src=rains[rr['site_stem']].loc[ti['dates'][0]:ti['dates'][-1]].to_numpy()
        checks['observed_rain_exact'] &= np.array_equal(src,ti['rain'])
        checks['calendar_exact'] &= str(ti['dates'][0])==rr['context_start'] and str(ti['dates'][1024])==rr['origin_first_forecast_day']
        with np.load(tr,allow_pickle=False) as z:
            truth=dict(h_nat=z['h_nat'],eps=z['eps'],E_true={p:z['E_true__'+p] for p in pc.PAIRS})
        raw.append(dict(case_id=m['case_id'],tf_input=ti,truth=truth,derived=m))
    from p3_phase3_make_cases import invariants
    inv=invariants(raw);checks.update(inv['invariants'])
    for H in (10,30):
        with np.load(P3/'cases'/f'tool_inputs_H{H}.npz',allow_pickle=False) as z:arr={k:z[k] for k in z.files}
        ex.validate(arr,H);checks[f'tool_H{H}_stamp']=str(arr['protocol_sha256'])==digest
        checks[f'tool_H{H}_serialized_crosslink']=True
        by={m['case_id']:c['tf_input'] for m,c in zip(man,raw)}
        for i,cid in enumerate(arr['case_id']):
            ti=by[str(cid)];checks[f'tool_H{H}_serialized_crosslink'] &= np.array_equal(arr['head'][i],ti['head_context']) and np.array_equal(arr['pumping'][i,:1024],ti['pumping_context']) and np.array_equal(arr['rainfall'][i],ti['rain'][:1024+H])
    snapshot=json.loads((P3/'B_D03_READONLY_SNAPSHOT.json').read_text());changed=[p for p,h in snapshot.items() if not(ROOT/p).exists() or pc.sha(ROOT/p)!=h];checks['D03_all_files_unchanged']=not changed
    rec=dict(at=datetime.datetime.now().astimezone().isoformat(),protocol_sha256=digest,scope='input_and_output' if a.outputs else 'input_only',n_cases=len(man),checks={k:bool(v) for k,v in checks.items()},D03_snapshot_files=len(snapshot),D03_changed=changed,case_counts=inv['case_counts'],matching_max_relative_spread=inv['matching_max_relative_spread'],truth_pair_identity_maxabs_m=inv['truth_pair_identity_maxabs_m'],audit_code_sha256=pc.sha(__file__))
    if a.outputs:
        records=[];errors=[];outsha={}
        for m in man:
            cid=m['case_id'];p=P3/'wb/markers'/f'{cid}.done'
            if not p.exists():errors.append(cid+':missing marker');continue
            mk=json.loads(p.read_text());records.append(mk)
            if mk['tf_input_sha256']!=m['tf_sha256'] or mk['protocol_sha256']!=digest or mk['driver_sha256']!=pc.sha(ROOT/'lib/p3_phase3_run_wb.py'):errors.append(cid+':provenance mismatch')
            if len(mk['outputs'])!=4:errors.append(cid+':not all4 outputs')
            for path,h in mk['outputs'].items():
                if not(ROOT/path).exists() or pc.sha(ROOT/path)!=h:errors.append(cid+':output hash mismatch')
                outsha[path]=h
            if mk.get('W_status')=='error' or mk.get('reference_status')!='ok' or not mk.get('truth_eval_ok'):errors.append(cid+':execution error')
            W=json.loads((P3/'wb/W'/f'{cid}.json').read_text());R=json.loads((P3/'wb/reference'/f'{cid}.json').read_text());T=json.loads((P3/'wb/truth_eval'/f'{cid}.json').read_text())
            if W.get('envelope'):
                for pair in pc.PAIRS:
                    env=W['envelope'][pair];inf=np.array(env['inf']);sup=np.array(env['sup']);width=np.array(env['W']);et=np.array(T['E_true'][pair]);er=np.array(R['pairs'][pair]['E_point'])
                    if not all(len(v)==30 and np.isfinite(v).all() for v in (inf,sup,width,et,er)) or not np.allclose(sup-inf,width,atol=1e-12,rtol=1e-12):errors.append(cid+':invalid lead curves')
            else:errors.append(cid+':no envelope')
        wb=json.loads((P3/'wb/wb_manifest.json').read_text());rec.update(output_errors=errors,n_markers=len(records),n_output_files_hashed=len(outsha),outputs_digest=hashlib.sha256(json.dumps(sorted(outsha.items())).encode()).hexdigest(),W_status=dict(Counter(m['W_status'] for m in records)),reference_status=dict(Counter(m['reference_status'] for m in records)),collector_complete=wb['complete'],collector_manifest_sha256=pc.sha(P3/'wb/wb_manifest.json'))
        rec['checks']['all690_outputs_provenance_complete']=len(records)==690 and len(outsha)==2760 and not errors and wb['complete']
    rec['all_ok']=all(rec['checks'].values());name='B_OUTPUT_VERIFICATION.json' if a.outputs else 'cases/input_verification.json';(P3/name).write_text(json.dumps(rec,indent=1));print(json.dumps(rec));return 0 if rec['all_ok'] else 1
if __name__=='__main__':sys.exit(main())
