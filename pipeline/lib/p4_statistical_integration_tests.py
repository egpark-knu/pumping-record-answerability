"""D05 design regressions. Synthetic data only, no scientific outcomes."""
import math
import unittest
import numpy as np
import p4_statistics as st
import p4_analysis as legacy


def row(**kw):
    r=dict(source_phase="phase4",case_id="a",realization=0,layer_id="confined_T50",storage_type="confined",SR=1.,SR_definition="outside_designated_rest",rho=0.,N_transitions=6,Q_scale=1.,experiment_group="A",track="raw",quantity="lead",pair_id=st.PAIRS[0],lead_day=10,env_inf_m=.2,env_sup_m=.4,W_m=.2,W_status="ok",E_true_m=.4,E_tool_m=.3,E_reference_m=.35,reference_status="ok")
    r.update(kw)
    return r


class DesignRegressions(unittest.TestCase):
    def test_steep_overlapping_sample_has_finite_coefficient_above_twenty(self):
        x=np.repeat([-.01,.01],10);y=np.r_[np.zeros(9),1,0,np.ones(9)]
        f=st.fit_logistic(y,np.c_[np.ones(20),x],column_names=["intercept","slope"])
        self.assertEqual(f["status"],"ok")
        self.assertAlmostEqual(f["coefficients"]["slope"],math.log(9)/.01,places=5)

    def test_complete_and_quasi_separation_are_distinct_and_have_no_coefficients(self):
        for x,y,want in [([-2,-1,1,2],[0,0,1,1],"complete_separation"),([-1,0,0,1],[0,0,1,1],"quasi_separation")]:
            f=st.fit_logistic(y,np.c_[np.ones(4),x])
            self.assertEqual(f["status"],want)
            self.assertIsNone(f["coefficients"])

    def test_invalid_supplied_sign_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"sign_determined"):
            st.adapt_row(row(sign_determined=0))

    def test_calendar_n6_never_enters_map(self):
        r=st.adapt_row(row(experiment_group="B",calendar_type="paddy_irrigation",origin_month=5,SR_definition="origin_rate"))
        self.assertFalse(st.classify_row(r)["cohort_A"])

    def test_unknown_origin_scale_is_not_imputed(self):
        r=row();r.pop("Q_scale")
        self.assertIsNone(st.adapt_row(r)["Q_scale"])

    def test_zero_nonfinite_and_reference_denominators_are_separate(self):
        rs=[st.adapt_row(row(case_id="opposite",E_tool_m=-.1)),st.adapt_row(row(case_id="zero",E_tool_m=0.)),st.adapt_row(row(case_id="missing",E_tool_m=float("nan"),E_reference_m=None,reference_status="error")),st.adapt_row(row(case_id="open",env_inf_m=-.1,env_sup_m=.1,E_true_m=0.,E_tool_m=-1.))]
        t=st.contradiction_table(rs,st.log_equal_bins([1.]*4)["edges"])
        self.assertEqual(t["tool_contradiction"]["numerator"],1)
        self.assertEqual(t["tool_disagreement_including_zero"]["numerator"],2)
        self.assertEqual(t["determined_total"],3)
        self.assertEqual(t["paired_tool_contradiction"]["denominator"],2)
        self.assertEqual(t["reference_contradiction"]["numerator"],0)
        self.assertEqual(t["undetermined_truth_error"]["denominator"],0)

    def test_reference_missing_is_not_optional(self):
        r=row();r.pop("E_reference_m");r.pop("reference_status")
        with self.assertRaisesRegex(ValueError,"reference"):
            st.join_reference_rows([r],[])

    def test_convex_hull_rejects_box_corner_and_handles_interpolation(self):
        support=st.Support(0,1,0,1,[0,1],points=[(0,math.log(.05)),(1,math.log(.05)),(0,math.log(1.05))])
        f=st.ok_fit({"intercept":-.8,"log_signal_ratio":1.,"log_rho_plus_offset":0.},support)
        c=st.crossing_from_fit(f,.5,1.,None)
        self.assertEqual(c["reporting_status"],"extrapolated")
        f=st.ok_fit({"intercept":-.1,"log_signal_ratio":1.,"log_rho_plus_offset":0.},support)
        c=st.crossing_from_fit(f,.5,.1,None)
        self.assertEqual(c["reporting_status"],"finite_inside")
        self.assertTrue(c["rho_interpolated"])

    def test_exponent_overflow_preserves_log_crossing(self):
        f=st.ok_fit({"intercept":-1000.,"log_signal_ratio":1.,"log_rho_plus_offset":0.},None)
        c=st.crossing_from_fit(f,.5,0.,None)
        self.assertIsNone(c["signal_ratio"])
        self.assertEqual(c["log_signal_ratio"],1000.)
        self.assertEqual(c["status"],"exponent_overflow")

    def test_no_active_contrast_is_not_active_failure(self):
        rs=[st.adapt_row(row(case_id="off",calendar_type="paddy_irrigation",origin_month=1,experiment_group="B",SR_definition="origin_rate",q_origin_m3d=0.,SR=0.,E_true_m=0.,W_m=0.,env_inf_m=0.,env_sup_m=0.)),st.adapt_row(row(case_id="on",calendar_type="paddy_irrigation",origin_month=1,experiment_group="B",SR_definition="origin_rate",q_origin_m3d=100.,W_m=0.,env_inf_m=.4,env_sup_m=.4))]
        cell=st.calendar_rates(rs)["cells"][0]
        self.assertEqual(cell["denominator"],2)
        self.assertEqual(cell["n_no_active_contrast"],1)
        self.assertEqual(cell["n_relative_width_undefined"],1)
        self.assertEqual(cell["sign_rate"]["rate"],.5)
        self.assertEqual(cell["active_sign_rate"]["rate"],1.)
        self.assertEqual(cell["active_relative_width_rate"]["rate"],1.)

    def test_shared_draws_keep_all_ten_clusters_with_reproducible_counts(self):
        universe,draws=st._draws([st.adapt_row(row())],999,20261001)
        self.assertEqual(universe,list(range(10)))
        self.assertEqual(len(draws),999)
        self.assertTrue(all(set(d)==set(range(10)) and sum(d.values())==10 for d in draws))
        self.assertEqual(st._draws([],999,20261001)[1],draws)

    def test_legacy_bootstrap_uses_same_canonical_ten_cluster_draw(self):
        idx=list(legacy.cluster_indices([0,1],draws=1))[0]
        self.assertGreater(len(idx),0)
        _,d=st._draws([],1,20261001)
        self.assertEqual(len(idx),d[0][0]+d[0][1])
        self.assertEqual(legacy.BOOTSTRAP_DRAWS,999)

    def test_rank_deficiency_and_nonconvergence_have_no_coefficients(self):
        f=st.fit_logistic([0,1,0,1],np.ones((4,2)))
        self.assertEqual(f["status"],"rank_deficient")
        f=st.fit_logistic([0,0,1,0,1,1],np.c_[np.ones(6),[-1,-1,-1,1,1,1]],max_iter=0)
        self.assertEqual(f["status"],"nonconvergence")
        self.assertIsNone(f["coefficients"])

    def test_magnitude_root_is_continuous_zero_target(self):
        f=st.ok_fit({"intercept":math.log(2),"log_signal_ratio":-1.,"log_rho_plus_offset":0.},None)
        c=st.magnitude_crossing_from_fit(f,0.,None)
        self.assertAlmostEqual(c["signal_ratio"],2.)
        self.assertEqual(c["target_log_ratio"],0.)

    def test_official_run_requires_explicit_membership(self):
        with self.assertRaisesRegex(ValueError,"cohort_manifest"):
            st.run_statistics([row()])


class IntegrationContracts(unittest.TestCase):
    def test_exact_manifest_and_missing_primary_row_detection(self):
        def entries(n,phase):
            out=[]
            for i in range(n):
                group=("B" if i<360 else "A" if i<600 else "C") if phase==3 else ("A" if i<720 else "B") if phase==4 else "D03"
                out.append(dict(case_id=f"p{phase}_{i}",realization=i%10,experiment_group=group,N_nominal=6 if phase!=2 or i<180 else 2))
            return out
        data=[entries(540,2),entries(690,3),entries(2880,4)]
        manifest=st.build_cohort_manifest(*data)
        self.assertEqual(manifest["counts"],{"A":1260,"B":2160,"C":4110})
        rs=[]
        for phase,items in enumerate(data,2):
            for item in items:
                for pair in st.PAIRS:
                    for lead in (10,30):
                        calendar=phase==4 and item["experiment_group"]=="B"
                        rs.append(st.adapt_row(row(source_phase=f"phase{phase}",case_id=item["case_id"],realization=item["realization"],experiment_group=item["experiment_group"],N_transitions=item["N_nominal"],pair_id=pair,lead_day=lead,Q_scale=.3 if phase==3 and item["experiment_group"]=="B" else 1.,calendar_type="water_curtain" if calendar else None,origin_month=1 if calendar else None,q_origin_m3d=100. if calendar else None,reference_source_path="synthetic_reference.json",reference_sha256="0"*64)))
        _,counts=st.validate_inventory(rs,manifest)
        self.assertEqual(counts,{"A":5040,"B":8640,"C":16440})
        with self.assertRaisesRegex(ValueError,"incomplete"):
            st.validate_inventory(rs[:-1],manifest)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            st.validate_inventory(rs+[rs[0]],manifest)

    def test_reference_file_join_uses_real_bytes_and_case_pair_lead_keys(self):
        import tempfile,json,hashlib
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix="p4_statistics_fixture_",dir=Path(__file__).parent) as d:
            p=Path(d)/"reference.json"
            payload=dict(case_id="a",pairs={st.PAIRS[0]:{"E_point":[.25]*30},st.PAIRS[1]:{"E_point":[-.125]*30}})
            p.write_text(json.dumps(payload))
            refs=st.reference_rows_from_files([dict(source_phase="phase4",case_id="a",reference_path=str(p))])
            self.assertEqual(len(refs),4)
            self.assertEqual(refs[0]["reference_sha256"],hashlib.sha256(p.read_bytes()).hexdigest())
            r=st.join_reference_rows([row()],refs)[0]
            self.assertEqual(r["E_reference_m"],.25)
            p.write_text(json.dumps(dict(payload,case_id="wrong")))
            bad=st.reference_rows_from_files([dict(source_phase="phase4",case_id="a",reference_path=str(p))])
            self.assertTrue(all(x["reference_status"]=="error" and x["E_reference_m"] is None for x in bad))

    def test_nonfinite_truth_excluded_from_log_magnitude_and_missing_bounds_not_fitted(self):
        a=st.adapt_row(row(E_true_m=float("nan")))
        self.assertFalse(st.classify_row(a)["magnitude_likelihood"])
        a=st.adapt_row(row(env_inf_m=None,env_sup_m=None))
        self.assertFalse(st.classify_row(a)["magnitude_likelihood"])

    def test_full_models_sensitivity_and_paired_layer_interval_are_serializable(self):
        import json
        rs=[]
        for rr in range(10):
            for layer in st.LAYER_ORDER:
                for rho in (0.,.5,2.):
                    for signal,events in ((.5,2),(1.,5),(2.,8)):
                        W=2./signal+.001*rr;yes=rr<events
                        lo=.1 if yes else -W/2
                        for pair in st.PAIRS:
                            for lead in (10,30):
                                rs.append(row(case_id=f"{rr}_{layer}_{rho}_{signal}",realization=rr,layer_id=layer,storage_type=layer.split("_")[0],rho=rho,SR=signal,env_inf_m=lo,env_sup_m=lo+W,W_m=W,E_true_m=1.,pair_id=pair,lead_day=lead))
        result=st.run_statistics(rs,n_bootstrap=3,official=False)
        self.assertIn("positive_rho_sensitivity",result["cohort_A"])
        block=result["cohort_A"]["models"]["sign"]["10"][st.PAIRS[0]]
        self.assertIn("bootstrap",block["layer_comparison"])
        self.assertEqual(block["no_layer"]["bootstrap_fit_status_counts"].get("ok"),3)
        point=block["no_layer"]["crossings"]["0.0"]["0.5"]
        self.assertIn("status_counts",point["bootstrap"])
        json.dumps(result,allow_nan=False)


class OfficialBoundaries(unittest.TestCase):
    def test_official_statistics_stops_at_unfrozen_protocol(self):
        from unittest.mock import patch
        with patch("p4_contracts.require_frozen",side_effect=PermissionError("synthetic unfrozen gate")):
            with self.assertRaisesRegex(PermissionError,"unfrozen gate"):
                st.run_statistics([row()],cohort_manifest={"records":[]})

    def test_custom_official_draw_matrix_cannot_change_seeded_experiment(self):
        _,draws=st._draws([],999,20261001)
        st.validate_draws(draws,official=True)
        draws[0]={i:1 for i in range(10)}
        with self.assertRaisesRegex(ValueError,"canonical"):
            st.validate_draws(draws,official=True)



if __name__=="__main__": unittest.main(verbosity=2)
