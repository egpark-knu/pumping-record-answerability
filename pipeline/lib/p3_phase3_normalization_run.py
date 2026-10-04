"""All unique D03+D04 cases, exact final-context rolled covariate diagnostics."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from p3_phase3_normalization import capture_preprocessing
import p3_phase3_timesfm as prep
from p3_phase2_timesfm_execution import sha
R=Path(__import__("os").environ.get("PUMPING_RECORD_ROOT", str(Path(__file__).resolve().parents[1])));P=R/'results/phase3'

def main():
 start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert (P/'protocol_freeze.json').exists()
 rows=[];unique=set();max_roundoff=0.;sources={}
 for phase,count in [('phase2',540),('phase3',690)]:
  folder=R/'results'/phase;digest=sha(folder/'protocol.md');assert digest==(folder/'protocol.sha256').read_text().split()[0]
  for H in [10,30]:
   path=folder/f'cases/tool_inputs_H{H}.npz'
   with np.load(path,allow_pickle=False) as a:d={k:a[k] for k in a.files}
   assert len(set(d['case_id']))==count and str(d['protocol_sha256'])==digest
   for tr in prep.TRACKS:
    z=prep.prepare_track(d,H,tr);qraw={qid:q for qid,q in zip(d['query_id'],d['pumping'])}
    for startq in range(0,len(z['query_id']),128):
     stop=min(startq+128,len(z['query_id']));c=z['contexts'][startq:stop];cov=z['covariates'][startq:stop];s=capture_preprocessing(c,cov,H,'mps')
     assert np.all(s['n']==1024)
     for k in range(0,stop-startq,2):
      i=startq+k;cid=str(z['case_id'][i]);pid=str(z['pair_id'][i//2]);meta=json.loads(str(z['metadata_json'][i]));unique.add(cid)
      assert np.array_equal(s['std'][k],s['std'][k+1]) and np.array_equal(s['mean'][k],s['mean'][k+1])
      qa,qb=qraw[z['query_id'][i]],qraw[z['query_id'][i+1]];raw=float(qa[1024+H-1]-qb[1024+H-1]);feature=float(cov[k,0,-1]-cov[k+1,0,-1]);sigma=float(s['std'][k,1]);den=float(s['denominator'][k,1]);exact=float(s['normalized_future'][k,1,-1]-s['normalized_future'][k+1,1,-1]);div=feature/den
      max_roundoff=max(max_roundoff,abs(exact-div))
      row=dict(dataset='D03' if phase=='phase2' else 'D04',case_id=cid,pair_id=pid,schedule_pair=meta['schedule_pair'],track=tr,horizon=H,lead_day=H,raw_future_pumping_contrast=raw,feature_future_pumping_contrast=feature,normalized_future_change=exact,normalized_future_size=abs(exact),feature_contrast_over_denominator=div,raw_contrast_over_denominator=raw/den,
       normalized_future_change_mean=float(np.mean(s['normalized_future'][k,1]-s['normalized_future'][k+1,1])),pumping_cumulative_std=sigma,pumping_denominator=den,pumping_cumulative_mean=float(s['mean'][k,1]),pumping_raw_std=float(s['raw_context_std'][k,1]),pumping_detrended=int(s['detrended'][k,1]),pumping_safe_std_replacement=int(sigma<1e-6),head_cumulative_std=float(s['std'][k,0]),head_denominator=float(s['denominator'][k,0]),head_cumulative_mean=float(s['mean'][k,0]),head_raw_std=float(s['raw_context_std'][k,0]),head_detrended=int(s['detrended'][k,0]),rain_cumulative_std=float(s['std'][k,2]),count=1024,last_context_patch_index=31,head_channel_index=0,pumping_channel_index=1,rain_channel_index=2,context_left_padding=0,protocol_sha256=digest,input_sha256=sha(path),metadata_json=json.dumps(meta,sort_keys=True))
      rows.append(row)
    print(phase,tr,H,'normalization rows',len(rows),flush=True)
 sources[phase]=dict(protocol_sha256=sha(folder/'protocol.md'),H10=sha(folder/'cases/tool_inputs_H10.npz'),H30=sha(folder/'cases/tool_inputs_H30.npz'))
 assert len(unique)==1230 and len(rows)==19680 and len({(r['case_id'],r['pair_id'],r['track'],r['horizon']) for r in rows})==19680
 for r in rows:
  for v in r.values():
   if isinstance(v,(float,int)):assert np.isfinite(v)
 tmp=P/'normalization.csv.tmp'
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(P/'normalization.csv')
 schema=dict(task_id='run',rows=19680,unique_cases=1230,pairs=2,tracks=4,horizons=[10,30],key=['case_id','pair_id','track','horizon'],fields=list(rows[0]),main_x='normalized_future_size',signed_x='normalized_future_change',formula='actual installed _preprocess resblock rolled future input(a)-input(b) at lead H, pumping channel1, final context patch31; std after conditional detrend; raw std<1e-6 replaced with1',raw_numerator='physical raw Q_a(H)-Q_b(H), m3/d; feature numerator is raw or causal EWMA applied continuously',statistics='population cumulative unmasked patches0..31, count1024, float32 MPS source ops, head0 pumping1 rain2',scope='interpretation hypothesis diagnostic only, no causal claim',csv_sha256=sha(P/'normalization.csv'),driver_sha256=sha(__file__),helper_sha256=sha(R/'lib/p3_phase3_normalization.py'),inputs=sources,max_abs_float32_subtraction_difference_vs_feature_over_denominator=max_roundoff,elapsed_s=time.time()-start)
 (P/'NORMALIZATION_SCHEMA.json').write_text(json.dumps(schema,indent=2)+'\n')
 m=json.loads((P/'A_EXECUTION_MANIFEST.json').read_text());m.update(normalization_complete=True,normalization=schema);(P/'A_EXECUTION_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
 print('ALL_1230_NORMALIZATION_COMPLETE',flush=True)
if __name__=='__main__':main()
