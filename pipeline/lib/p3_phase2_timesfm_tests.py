"""D03 interface fixtures only; no hydraulic simulation/model inference."""
import json
import unittest
import numpy as np
try:
    import p3_phase2_timesfm as mod
except ModuleNotFoundError:
    mod = None


def fixture(h=10):
    meta = dict(origin_date="2020-01-01", rain_site="fixture", realization=0,
                rest_ratio=1.0, signal_ratio=0.8, pi_layer=2.0,
                storage_type="confined", storage_value=1e-4, T_m2_d=50.0,
                r_m=100.0, Q_m3_d=100.0, transition_count=6, t95_days=60.0,
                head_unit="m", pumping_unit="m3/d", rainfall_unit="mm/d", daily_alignment=True)
    q=np.full((4,1024+h),100.0); q[:,900:920]=0
    q[1,1024:]=0; q[3,1024:]=150
    metas=[dict(meta,schedule_pair=p) for p in ("P1_continue_vs_stop",)*2+("P2_current_vs_1p5x",)*2]
    return dict(head=np.tile(np.linspace(-1,1,1024),(4,1)), pumping=q,
        rainfall=np.ones((4,1024+h)), query_id=np.array(["c:p1:a","c:p1:b","c:p2:a","c:p2:b"]),
        case_id=np.array(["c"]*4), pair_id=np.array(["c:p1"]*2+["c:p2"]*2),
        schedule_id=np.array(["a","b","a","b"]), metadata_json=np.array([json.dumps(m) for m in metas]),
        protocol_sha256=np.array("0"*64))


class Contract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(mod,"D03 filter/validation interface has not been implemented")

    def test_step_matches_closed_form_and_cannot_see_future(self):
        # Catches lookahead/symmetric convolution and wrong lag/initial condition.
        a=np.zeros((1,1100)); a[:,1024:]=100
        f=mod.exp_filter(a,10)
        self.assertTrue(np.array_equal(f[:,:1024],np.zeros((1,1024))))
        np.testing.assert_allclose(f[0,1024:1027],100*(1-np.exp(-np.arange(1,4)/10)),rtol=1e-13)
        a[:,1050:]=500
        np.testing.assert_array_equal(mod.exp_filter(a,10)[:,:1050],f[:,:1050])

    def test_history_state_continues_into_future(self):
        # Catches reset-to-zero at prediction origin.
        a=np.array([[100.,100.,0.,0.]])
        np.testing.assert_allclose(mod.exp_filter(a,30),[[100,100,100*np.exp(-1/30),100*np.exp(-2/30)]])

    def test_constant_and_zero_are_preserved_all_scales(self):
        for tau in (10,30,90):
            np.testing.assert_allclose(mod.exp_filter(np.full((2,1040),123.),tau),123.)
            np.testing.assert_array_equal(mod.exp_filter(np.zeros((2,1040)),tau),0.)

    def test_invalid_forcing_or_scale_is_rejected(self):
        for x,tau in [(np.array([[np.nan]]),10),(np.array([[-1.]]),30),(np.ones((1,2)),0),(np.ones((1,2)),60)]:
            with self.assertRaises(ValueError):mod.exp_filter(x,tau)

    def test_raw_and_all_three_tracks_preserve_pairs_and_shapes(self):
        for h in (10,30):
            d=fixture(h)
            for track in ("raw","exp10","exp30","exp90"):
                z=mod.prepare_track(d,h,track)
                self.assertEqual(z["covariates"].shape,(4,2,1024+h))
                np.testing.assert_array_equal(z["covariates"][0,:,:1024],z["covariates"][1,:,:1024])
                np.testing.assert_array_equal(z["covariates"][:,1,:],d["rainfall"])
                self.assertEqual(len(z["pair_id"]),2)
                if track=="raw":np.testing.assert_array_equal(z["covariates"][:,0,:],d["pumping"])
                else:self.assertGreater(z["covariates"][1,0,1024],0)

    def test_different_past_or_weather_is_rejected(self):
        for field,col in [("head",100),("pumping",100),("rainfall",1025)]:
            d=fixture();d[field][1,col]+=1
            with self.assertRaises(ValueError):mod.validate(d,10)

    def test_truth_payload_duplicate_ids_and_bad_shape_rejected(self):
        for change in ("truth","id","shape","horizon"):
            d=fixture()
            if change=="truth":d["theta_true"]=np.array([1.])
            if change=="id":d["query_id"][1]=d["query_id"][0]
            if change=="shape":d["head"]=d["head"][:,:1023]
            with self.assertRaises(ValueError):mod.validate(d,64 if change=="horizon" else 10)

    def test_missing_scale_or_physical_label_is_rejected(self):
        d=fixture()
        with self.assertRaises(ValueError):mod.prepare_track(d,10,"best_filter")
        metas=[json.loads(s) for s in d["metadata_json"]]
        for m in metas:del m["storage_type"]
        d["metadata_json"]=np.array([json.dumps(m) for m in metas])
        with self.assertRaises(ValueError):mod.validate(d,10)

    def test_full_grid_gate_rejects_fixture_as_official(self):
        with self.assertRaises(ValueError):mod.validate(fixture(),10,require_full_grid=True)

    def test_reordered_input_still_pairs_a_then_b(self):
        d=fixture();ix=[3,1,2,0]
        d={k:v[ix] if np.ndim(v)>0 and len(v)==4 else v for k,v in d.items()}
        z=mod.prepare_track(d,10,"raw")
        self.assertEqual(list(z["schedule_id"]),["a","b","a","b"])
        self.assertEqual(list(z["query_id"]),["c:p2:a","c:p2:b","c:p1:a","c:p1:b"])


if __name__=="__main__":unittest.main(verbosity=2)
