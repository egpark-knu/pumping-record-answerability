"""D04 observed-only schema: frozen pilot shape/IDs and D03 origin checks, extended physical labels."""
import json
import numpy as np
import p3_phase2_export as d03
import timesfm_pilot_adapter as pilot

def export_arrays(cases,H,digest):
    rows=[r for c in cases for r in d03.tool_rows(c,H)]
    out={k:np.stack([r[k] for r in rows]) for k in ('head','pumping','rainfall')}
    out.update({k:np.array([r[k] for r in rows],dtype=str) for k in ('query_id','case_id','pair_id','schedule_id','metadata_json')})
    out['protocol_sha256']=np.array(digest);validate(out,H);return out

def validate(arr,H):
    if set(arr)!=d03.A.INPUT_KEYS:raise ValueError('Observed-only frozen schema required')
    order,ids=pilot.validate(arr,H);d03.assert_array_origin_link(arr)
    if len(order)!=2760 or len(set(arr['case_id']))!=690:raise ValueError('D04 grid must be 690 cases and 2760 queries')
    by={}
    for i,cid in enumerate(arr['case_id']):by.setdefault(str(cid),[]).append(i)
    grid=set()
    for cid,ix in by.items():
        if len(ix)!=4:raise ValueError('Four queries required per case')
        metas=[json.loads(str(arr['metadata_json'][i])) for i in ix]
        m=metas[0];grid.add((m['experiment_group'],m['realization'],m['storage_type'],m['T_m2_d'],m['rest_ratio'],m['transition_count'],m['Q_scale'],m['recency_label']))
        for field in ('head','rainfall'):
            if any(not np.array_equal(arr[field][i],arr[field][ix[0]]) for i in ix):raise ValueError('Paired observed input mismatch')
        if any(not np.array_equal(arr['pumping'][i,:1024],arr['pumping'][ix[0],:1024]) for i in ix):raise ValueError('Paired history mismatch')
        if any({k:v for k,v in x.items() if k!='schedule_pair'}!={k:v for k,v in m.items() if k!='schedule_pair'} for x in metas):raise ValueError('Metadata mismatch')
        if m['experiment_group']=='C' and m['storage_value'] not in (.0001,.001,.01):raise ValueError('C storage grid mismatch')
    if len(grid)!=690:raise ValueError('Duplicate physical design cell')
    return order,ids
