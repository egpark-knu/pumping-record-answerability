"""D12 additional 2: saved-bootstrap retrieval and stored-curve paired comparison.

Run with the science-environment Python (requirements-pipeline.txt). Writes only beside this script. No model imports,
physical response calls, fitting, optimization, or changes to input data.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from collections import Counter
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED, N_BOOT = 20261001, 999
PAIRS = ('P1_continue_vs_stop', 'P2_current_vs_1p5x')
RHOS = (0.25, 1.0, 4.0)
SOURCE_SHA = {}


def pin(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    old = SOURCE_SHA.setdefault(str(path), digest)
    assert old == digest, f'Input changed: {path}'
    return path


def readjson(path):
    return json.loads(pin(path).read_text())


def readcsv(path):
    with pin(path).open() as f:
        return list(csv.DictReader(f))


def writejson(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scalar_interpolate(x, xp, fp):
    """Independent two-point linear interpolation for the one verification pass."""
    j = int(np.searchsorted(xp, x, side='right'))
    if j == len(xp):
        assert x == xp[-1]
        return float(fp[-1])
    if j == 0:
        assert x == xp[0]
        return float(fp[0])
    t = (x - xp[j-1]) / (xp[j] - xp[j-1])
    return float(fp[j-1] + t * (fp[j] - fp[j-1]))


def rank_quantile(values, probability):
    """Independent sorted-rank check of numpy's default linear quantile."""
    v = sorted(float(x) for x in values)
    z = (len(v)-1) * probability
    a, b = math.floor(z), math.ceil(z)
    return v[a] + (z-a) * (v[b]-v[a])


def main():
    assert OUT == ROOT / 'results/phase10/calculations'
    for name in ('p4_statistics.py', 'p3_kernels.py', 'p3_phase2_physmap.py',
                 'p3_phase3_cases.py', 'p3_phase3_make_cases.py'):
        pin(ROOT / 'lib' / name)
    pin(ROOT / 'results/phase8/scripts/aggregate.py')
    # Public edition: the manuscript draft hashed here by the original run is not part of the release.

    matrix = np.load(pin(ROOT / 'results/phase4/bootstrap_draw_matrix.npy'), allow_pickle=False)
    assert matrix.shape == (999, 10) and np.all(matrix >= 0)
    assert np.issubdtype(matrix.dtype, np.integer) and np.all(matrix.sum(axis=1) == 10)
    rng = np.random.Generator(np.random.PCG64(SEED))
    canonical = np.array([np.bincount(rng.choice(10, size=10, replace=True), minlength=10)
                          for _ in range(N_BOOT)])
    assert np.array_equal(matrix, canonical)
    bootstrap = dict(seed=SEED, n=N_BOOT, cluster='realization', cluster_ids=list(range(10)),
                     rng='numpy.random.Generator(PCG64)', ci_type='percentile',
                     ci_percentiles=[2.5, 97.5], percentile_method='linear',
                     matrix_path=str(ROOT / 'results/phase4/bootstrap_draw_matrix.npy'),
                     matrix_sha256=SOURCE_SHA[str(ROOT / 'results/phase4/bootstrap_draw_matrix.npy')])

    joined = readcsv(ROOT / 'results/phase8/data/A_joined_rows.csv')
    old = readjson(ROOT / 'results/phase4/statistics.json')
    mag = readjson(ROOT / 'results/phase8/data/magnitude_models_UNCHANGED.json')
    current_table = readcsv(ROOT / 'results/phase8/data/Table3_D10.csv')
    assert mag == old['cohort_A']['models']['magnitude']
    assert old['bootstrap']['seed'] == SEED and old['bootstrap']['n'] == N_BOOT
    assert np.array_equal(matrix, np.asarray(old['bootstrap']['draw_matrix']))
    manifests = readjson(ROOT / 'results/phase4/COHORT_MANIFESTS.json')
    cohort_keys = {(r['source_phase'], r['case_id']) for r in manifests['map']}
    assert len(cohort_keys) == 1260
    members = {}
    for lead in (10, 30):
        rows = [r for r in joined if r['is_map'] == 'True' and r['pair_id'] == PAIRS[0]
                and int(r['lead_day']) == lead]
        assert len(rows) == 1260
        assert {(r['source_phase'], r['case_id']) for r in rows} == cohort_keys
        assert all(r['null_status'] == 'corrected_best_known_null_minimum_no_global_proof' for r in rows)
        assert all(float(r['SR']) > 0 and float(r['W_m']) > 0 and
                   math.isfinite(float(r['E_true_m'])) and abs(float(r['E_true_m'])) > 0 for r in rows)
        counts = Counter(int(r['realization']) for r in rows)
        assert dict(counts) == {i: 126 for i in range(10)}
        members[str(lead)] = [dict(source_phase=r['source_phase'], case_id=r['case_id'],
                                  realization=int(r['realization']), layer=r['layer_id'],
                                  rho=float(r['rho']), SR=float(r['SR']), W=float(r['W_m']),
                                  E_true=float(r['E_true_m']), current_detected=r['effect_detected'])
                              for r in rows]
    intervals = []
    for rho in (0., .1, .25, .5, 1., 2., 4.):
        current = next(r for r in current_table if float(r['rho']) == rho)
        for lead in (10, 30):
            model = mag[str(lead)][PAIRS[0]]['no_layer']
            node = model['crossings'][str(rho)]
            assert node['signal_ratio'] == float(current[f'magnitude{lead}'])
            assert node['status'] == 'ok' and node['reporting_status'] == 'finite_inside'
            assert node['bootstrap']['n_identified'] == 999 and node['bootstrap']['n_not_identified'] == 0
            assert node['bootstrap']['status_counts'] == {'ok': 999}
            c = model['coefficients']
            crossing = math.exp(-(c['intercept'] + c['log_rho_plus_offset'] * math.log(rho+.05))
                                / c['log_signal_ratio'])
            assert math.isclose(crossing, node['signal_ratio'], rel_tol=1e-14)
            intervals.append(dict(rho=rho, lead_day=lead, point=node['signal_ratio'],
                                  ci_low=node['bootstrap']['low'], ci_high=node['bootstrap']['high'],
                                  denominator_records=model['n'], clusters=10,
                                  records_per_cluster=126,
                                  bootstrap_weighted_denominator=[126*int(row.sum()) for row in matrix],
                                  identified_replicates=999, missing_replicates=0,
                                  extrapolated_replicates=node['bootstrap']['n_extrapolated'],
                                  source_selector=f'cohort_A.models.magnitude.{lead}.{PAIRS[0]}.no_layer.crossings.{rho}',
                                  saved_payload=node, newly_requested=rho in (.1,.25,.5,1.,2.)))
    table_result = dict(status='complete_by_retrieving_existing_identical_999_bootstrap',
                        bootstrap=bootstrap, definition='pooled no-layer log-linear magnitude contour W/abs(E_true)=1',
                        crossing_formula='exp(-(g0 + g_rho*log(rho+0.05))/g_SR)',
                        rows=intervals, membership_by_lead=members,
                        preserved_current_detection_columns=current_table,
                        missing_values=[], refits=0,
                        note='Saved bootstrap endpoints reused exactly. Raw bootstrap fitted coefficients/crossings are not serialized; no new fit was performed.')

    record_manifest = readjson(ROOT / 'results/phase8/data/record_manifest.json')
    derived = readjson(ROOT / 'results/phase3/cases/derived_manifest.json')
    c_derived = {r['case_id']:r for r in derived if r['experiment_group'] == 'C'}
    c_records = [r for r in record_manifest if r['source_phase']=='phase3' and r['case_id'] in c_derived]
    assert len(c_records) == len(c_derived) == 90
    curves, membership = {}, []
    weather = {}
    t95_by_s = {}
    for rec in c_records:
        d = c_derived[rec['case_id']]
        tf_path = pin(rec['tf_path']); w_path = pin(rec['w_path'])
        assert SOURCE_SHA[str(tf_path)] == rec['tf_sha256']
        assert SOURCE_SHA[str(w_path)] == rec['w_sha256']
        w = readjson(w_path)
        tr_path = pin(str(tf_path).replace('/tf_inputs/', '/truth/'))
        with np.load(tf_path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta_json']))
            rain = z['rain'].copy()
            dates = z['dates'].copy()
        with np.load(tr_path, allow_pickle=False) as z:
            theta = json.loads(str(z['theta_true_json']))
            hnat, eps = z['h_nat'].copy(), z['eps'].copy()
            E = {p:z[f'E_true__{p}'].copy() for p in PAIRS}
        s, rho, real = float(meta['storage_value']), float(meta['rest_ratio']), int(meta['realization'])
        key = (s, rho, real)
        assert key not in curves and rho in RHOS and real in range(10)
        assert w['case_id'] == rec['case_id'] and w['meta'] == meta
        assert theta['T'] == meta['T_m2_d'] == 50.
        assert theta['c'] == meta['c_d'] == 20000.
        assert theta['r'] == meta['r_m'] == 200.
        assert theta['S'] == s and math.isclose(theta['a'], 20000*s)
        assert theta['b'] == d['b'] == .01 and theta['A'] == d['gain_m_per_m3d']
        assert meta['t95_days'] == d['t95_d']
        t95_by_s.setdefault(s,meta['t95_days'])
        assert t95_by_s[s] == meta['t95_days']
        W = {p:np.asarray(w['envelope'][p]['W'], float) for p in PAIRS}
        for p in PAIRS:
            assert W[p].shape == E[p].shape == (30,)
            assert np.all(np.isfinite(W[p])) and np.all(W[p] > 0)
            assert np.all(np.isfinite(E[p])) and np.all(np.abs(E[p]) > 0)
            for lead in (10,30):
                row = next(r for r in joined if r['source_phase']=='phase3' and
                           r['case_id']==rec['case_id'] and r['pair_id']==p and int(r['lead_day'])==lead)
                assert W[p][lead-1] == float(row['W_m']) and E[p][lead-1] == float(row['E_true_m'])
        if real in weather:
            assert all(np.array_equal(a,b) for a,b in zip(weather[real],(rain,hnat,eps,dates)))
        else:
            weather[real] = (rain,hnat,eps,dates)
        membership.append(dict(case_id=rec['case_id'], realization=real, rho=rho, S=s,
                               T=theta['T'], c=theta['c'], r=theta['r'], a=theta['a'], b=theta['b'],
                               gain=theta['A'], t95=meta['t95_days'],
                               lead_domain=[1,30], normalized_lead_domain=[1/meta['t95_days'],30/meta['t95_days']],
                               rho_realized=d['rho_realized'], rest_days=d['R_days'],
                               linear_approximation=meta['linear_approximation'],
                               tf_path=str(tf_path), w_path=str(w_path), truth_path=str(tr_path)))
        curves[key] = dict(W=W, E=E, t95=meta['t95_days'], case_id=rec['case_id'])
    assert set(curves) == {(s,rho,i) for s in (1e-4,1e-3,1e-2) for rho in RHOS for i in range(10)}
    scales = [t95_by_s[1e-3]/t95_by_s[1e-4], t95_by_s[1e-2]/t95_by_s[1e-3]]
    assert all(math.isclose(x,10.,rel_tol=1e-10) for x in scales)
    assert len({r['gain'] for r in membership}) == 1
    results=[]
    interpolation_error=0.
    for small,large,requested in ((1e-4,1e-3,(.44,1.32)), (1e-3,1e-2,(.044,.13))):
        ts,tl=t95_by_s[small],t95_by_s[large]
        lower=max(requested[0],1/ts,1/tl)
        upper=min(requested[1],30/ts,30/tl)
        assert lower < upper
        # Conventional uniform shared normalized-lead grid, fixed before seeing ratios.
        grid=np.linspace(lower,upper,101)
        for rho in RHOS:
            for p in PAIRS:
                ratios=[]; detail=[]
                for real in range(10):
                    a,b=curves[(small,rho,real)],curves[(large,rho,real)]
                    xs,xl=np.arange(1,31)/ts,np.arange(1,31)/tl
                    assert grid[0]>=xs[0] and grid[-1]<=xs[-1] and grid[0]>=xl[0] and grid[-1]<=xl[-1]
                    ws,wl=np.interp(grid,xs,a['W'][p]),np.interp(grid,xl,b['W'][p])
                    es,el=np.interp(grid,xs,np.abs(a['E'][p])),np.interp(grid,xl,np.abs(b['E'][p]))
                    assert np.all(es>0) and np.all(el>0) and np.all(ws>0)
                    relative_small,relative_large=ws/es,wl/el
                    ratio=relative_large/relative_small
                    assert np.all(np.isfinite(ratio)) and np.all(ratio>0)
                    for j in (0,50,100):
                        for xp,fp,value in ((xs,a['W'][p],ws[j]),(xl,b['W'][p],wl[j]),
                                            (xs,np.abs(a['E'][p]),es[j]),(xl,np.abs(b['E'][p]),el[j])):
                            check=scalar_interpolate(grid[j],xp,fp)
                            interpolation_error=max(interpolation_error,abs(check-float(value)))
                            assert math.isclose(check,float(value),rel_tol=1e-13,abs_tol=1e-13)
                    ratios.append(ratio)
                    detail.append(dict(realization=real,small_case=a['case_id'],large_case=b['case_id'],
                                       interpolated_W_small=ws.tolist(),interpolated_W_large=wl.tolist(),
                                       interpolated_abs_E_small=es.tolist(),interpolated_abs_E_large=el.tolist(),
                                       paired_ratios=ratio.tolist(),within_realization_median=float(np.median(ratio))))
                ratios=np.array(ratios)
                point=float(np.median(ratios))
                reps=[]
                for counts in matrix:
                    selected=np.repeat(np.arange(10),counts)
                    reps.append(float(np.median(ratios[selected].ravel())))
                low,high=np.percentile(reps,[2.5,97.5],method='linear')
                assert math.isclose(point,rank_quantile(ratios.ravel(),.5),rel_tol=1e-14)
                assert math.isclose(low,rank_quantile(reps,.025),rel_tol=1e-14)
                assert math.isclose(high,rank_quantile(reps,.975),rel_tol=1e-14)
                similar=.8<=point<=1.25
                results.append(dict(S_small=small,S_large=large,rho=rho,pair=p,orientation='larger_S / smaller_S',
                                    requested_normalized_lead_interval=list(requested),
                                    actual_normalized_lead_interval=[lower,upper],shared_grid=grid.tolist(),
                                    source_small_lead_interval=[lower*ts,upper*ts],
                                    source_large_lead_interval=[lower*tl,upper*tl],
                                    median_paired_ratio=point,ci_low=float(low),ci_high=float(high),
                                    paired_realizations=10,grid_points_per_pair=101,paired_ratio_points=1010,
                                    cluster_denominator=10,bootstrap_replicates=N_BOOT,identified_replicates=N_BOOT,
                                    missing_replicates=0,bootstrap_statistics=reps,
                                    bootstrap_point_denominator=[1010]*N_BOOT,
                                    author_criterion_pass=similar,
                                    author_phrase='at matched lead in response times, relative ambiguity was similar' if similar else None,
                                    paired_details=detail))
    primary=[r for r in results if r['pair']==PAIRS[0]]
    passing=[r for r in primary if r['author_criterion_pass']]
    failing=[r for r in primary if not r['author_criterion_pass']]
    ranges={}
    for s in (1e-4,1e-3):
        subset=[r for r in primary if r['S_small']==s]
        ranges[str(s)]=[min(r['median_paired_ratio'] for r in subset),max(r['median_paired_ratio'] for r in subset)]
    # Writer text is assembled after all rho strata are shown, without favorable pooling.
    text1=(f"At matched lead in response times, the median paired relative-ambiguity ratios for larger versus smaller storage ranged from "
           f"{ranges[str(1e-4)][0]:.3f} to {ranges[str(1e-4)][1]:.3f} for S=10^-4 versus 10^-3 and from "
           f"{ranges[str(1e-3)][0]:.3f} to {ranges[str(1e-3)][1]:.3f} for S=10^-3 versus 10^-2.")
    if not failing:
        text2=('Across the tested pause ratios, at matched lead in response times, relative ambiguity was similar; '
               'rainfall and noise remained on their original daily time axes.')
    elif not passing:
        text2=('These ratios fell outside the author\'s similarity range and do not support the response-time principle; '
               'rainfall and noise remained on their original daily time axes.')
    else:
        pass_labels=', '.join(f"S={r['S_small']:g} versus {r['S_large']:g} at rho={r['rho']:g}" for r in passing)
        text1=(f"For {pass_labels}, at matched lead in response times, relative ambiguity was similar.")
        text2=(f"The other paired medians ranged from {min(r['median_paired_ratio'] for r in failing):.3f} "
               f"to {max(r['median_paired_ratio'] for r in failing):.3f} and did not support this interpretation, "
               'while rainfall and noise retained their daily timing.')
    storage_result=dict(status='complete',bootstrap=bootstrap,rows=results,membership=membership,
                        physical_facts=dict(T_m2_d=50,c_d=20000,r_m=200,b=.01,
                                            gain_m_per_m3d=membership[0]['gain'],
                                            a_by_S={str(s):20000*s for s in t95_by_s},
                                            t95_by_S={str(s):t for s,t in t95_by_s.items()},
                                            t95_adjacent_scale=scales,
                                            explanation='At fixed T,c,r, b and gain stay fixed, a=cS changes; the saved kernel expression depends on t/a at fixed b. Rain, natural heads and noise are identical daily arrays within realization, not time-scaled.'),
                        method=dict(pairing_key=['rho_nominal','realization','pair_id','matched_k_over_t95'],
                                    interpolation='piecewise linear interpolation of stored W and stored abs(E_true) separately before forming W/abs(E_true)',
                                    shared_grid='101 equally spaced k/t95 points over intersection of requested interval and both original lead1--30 domains; fixed before examining ratios',
                                    aggregation='median of paired (W_large/abs(E_large))/(W_small/abs(E_small)) over 10 realizations x 101 matched leads, separately for each rho and question',
                                    resampling='shared canonical999 realization-cluster multiplicities; repeat each realization whole, preserving its paired101-point vector',
                                    endpoint_policy='clip requested lower endpoints to available source domain; no extrapolation',
                                    criterion='point median in inclusive[0.8,1.25]; the CI is reported but not an additional author acceptance test'),
                        writer_two_sentences=[text1,text2],
                        no_global_favorable_pool=True,
                        residual_risks=['Conditional on ten selected station/noise realizations and sampled acceptable-model envelopes.',
                                        'Normalized-lead intervals contain only about the first1--3 days of the smaller-storage response; linear interpolation summarizes sparse source leads.',
                                        'Pairs share nominal rho, but integer-day rounding changes realized pause ratios and pumping histories; this is the saved experiment, not a pure time-scaled history.',
                                        'S=0.01 uses the saved linearized specific-yield response.'],
                        new_physical_response_calls=0,new_experiments=0,new_fits=0)

    table_lines=['| rho | Magnitude10d (95%CI) | Magnitude30d (95%CI) |', '| --- | --- | --- |']
    for rho in (0.,.1,.25,.5,1.,2.,4.):
        a=next(r for r in intervals if r['rho']==rho and r['lead_day']==10)
        b=next(r for r in intervals if r['rho']==rho and r['lead_day']==30)
        fmt=lambda r:f"{r['point']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})"
        table_lines.append(f"| {rho:.2f} | {fmt(a)} | {fmt(b)} |")
    table_text='\n'.join(table_lines)+'\n\nAll intervals: the same999 realization-cluster draws, seed20261001, percentile2.5--97.5. Each lead uses1260 map records,126 per realization. All999 replicates identified, zero extrapolated. Existing magnitude bootstrap values were already saved and are reused exactly. Detection columns and all point estimates remain unchanged. No missing-value caption is needed.\n'
    storage_lines=['| S_small | S_large | rho | question | paired median ratio | 95%CI | author similarity criterion |',
                   '| --- | --- | --- | --- | --- | --- | --- |']
    for r in results:
        storage_lines.append(f"| {r['S_small']:g} | {r['S_large']:g} | {r['rho']:g} | {r['pair']} | {r['median_paired_ratio']:.6f} | {r['ci_low']:.6f}-{r['ci_high']:.6f} | {'pass' if r['author_criterion_pass'] else 'outside; no principle-support claim'} |")
    storage_text=('\n'.join(storage_lines)+'\n\nPossible main text (stop question):\n\n'+' '.join(storage_result['writer_two_sentences'])+'\n\nMethods:\n\n'+
                  '\n'.join(f'- {k}: {v}' for k,v in storage_result['method'].items())+'\n\nAll comparisons use10 paired realizations and101 matched leads per realization. Both questions and every stored rho(0.25,1,4) are shown. Bootstrap intervals use the shared999 realization-cluster draws, seed20261001, linear percentile2.5--97.5; lead rows are not resampled independently.\n\nPhysical facts: '+json.dumps(storage_result['physical_facts'],indent=2)+'\n\n'+
                  '\n'.join(f"- S={r['S_small']:g} vs{r['S_large']:g}, rho={r['rho']:g}, {r['pair']}: requested k/t95={r['requested_normalized_lead_interval']}; actual={r['actual_normalized_lead_interval']}; physical leads small={r['source_small_lead_interval']}, large={r['source_large_lead_interval']}." for r in primary)+'\n\nResidual limitations:\n\n'+
                  '\n'.join('- '+x for x in storage_result['residual_risks'])+'\n\nComplete unrounded interpolated values, ratios, all999 bootstrap medians and case membership are in MATCHED_STORAGE_COMPARISON.json.\n')
    # One bounded input/output verification pass, no refit or additional physical calls.
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in SOURCE_SHA.items())
    assert unchanged
    writejson('TABLE3_INTERVALS.json',table_result)
    (OUT/'TABLE3_INTERVALS.md').write_text(table_text)
    writejson('MATCHED_STORAGE_COMPARISON.json',storage_result)
    (OUT/'MATCHED_STORAGE_COMPARISON.md').write_text(storage_text)
    written=['additional2_calculations.py','TABLE3_INTERVALS.json','TABLE3_INTERVALS.md',
             'MATCHED_STORAGE_COMPARISON.json','MATCHED_STORAGE_COMPARISON.md']
    for name in ('TABLE3_INTERVALS.json','MATCHED_STORAGE_COMPARISON.json'):
        json.loads((OUT/name).read_text())
    checks=dict(pair_uniqueness=True,complete_storage_grid_90=True,all_rho_and_both_questions=True,
                original_lead_bounds_only=True,expected_t95_factor10=True,positive_finite_denominators=True,
                weather_noise_identical_not_timescaled=True,bootstrap_unit_realization_not_lead_row=True,
                bootstrap_matrix_canonical=True,table3_magnitude_source_identical_current_phase8=True,
                table3_14_points_match_current=True,corrected_joined_rows_only=True,
                table3_magnitude_denominator1260_per_lead=True,table3_new_intervals10_available=True,
                calendar_detection_and_table3_detection_unchanged=True,input_shas_unchanged=unchanged,
                interpolation_spotchecks_independent=True,max_interpolation_abs_error=interpolation_error,
                paired_medians_and_CI_sorted_rank_check=True,output_json_reopened=True,
                verification_passes=1)
    writejson('ADDITIONAL2_RECEIPT.json',dict(turn_id='run',status='requested_calculation_components_complete',
                                           requested_components=['Table3 missing magnitude intervals','matched storage relative-ambiguity comparison'],
                                           scope='phase10/calculations only',source_sha256=SOURCE_SHA,
                                           output_sha256={str(OUT/n):hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in written},
                                           checks=checks,bootstrap=bootstrap,new_fits=0,new_optimization_calls=0,
                                           new_null_starts=0,new_experiments=0,new_physical_response_calls=0,
                                           figure1_replay_calls=0,manuscript_edits=0,
                                           table3_ci_provenance='Exact saved999 bootstrap endpoints; pre-correction detection labels never used. Magnitude source is explicitly unchanged by phase8.',
                                           table3_raw_bootstrap_fit_replicates_available=False,
                                           storage_bootstrap_statistics_saved=True,
                                           residual_risks=storage_result['residual_risks']))
    print(json.dumps(dict(status='complete',table3_new_interval_cells=10,storage_cases=90,
                          storage_rows=len(results),primary_rows=[{k:r[k] for k in ('S_small','S_large','rho','median_paired_ratio','ci_low','ci_high','author_criterion_pass')} for r in primary],
                          output_directory=str(OUT))))


if __name__=='__main__':
    main()
