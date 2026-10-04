"""Catch raw-only std shortcut, wrong variance divisor, zero epsilon rule and stale data reuse."""
import unittest,importlib.util
from pathlib import Path
import numpy as np
class Diagnostic(unittest.TestCase):
 def module(self):
  path=Path(__file__).with_name('p3_phase3_normalization.py')
  self.assertTrue(path.exists(),'Exact diagnostic implementation missing')
  spec=importlib.util.spec_from_file_location('norm',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_population_running_statistics(self):
  m=self.module();head=np.tile(np.array([1,3],np.float32),512)[None,:]
  cov=np.stack([head+10,head+20],axis=1);cov=np.concatenate([cov,np.full((1,2,10),12,np.float32)],axis=2)
  got=m.capture_preprocessing(head,cov,10,'cpu')
  np.testing.assert_allclose(got['std'][0],[1,1,1],atol=2e-6)
  np.testing.assert_allclose(got['n'][0],[1024,1024,1024]);self.assertEqual(got['last_context_patch'],31)
 def test_conditional_detrending_and_safe_denominator(self):
  m=self.module();head=np.arange(1024,dtype=np.float32)[None,:];cov=np.stack([head,head+2],axis=1)
  cov=np.concatenate([cov,np.full((1,2,30),1024,np.float32)],axis=2)
  got=m.capture_preprocessing(head,cov,30,'cpu')
  self.assertTrue(np.max(got['std'])<1e-6);np.testing.assert_array_equal(got['denominator'],np.ones((1,3)))
  self.assertTrue(got['detrended'].all())
 def test_future_contrast_sign_and_nonfinite_rejection(self):
  m=self.module();head=np.ones((2,1024),np.float32);cov=np.ones((2,2,1034),np.float32)*10
  cov[0,0,1024:]=10;cov[1,0,1024:]=15
  got=m.capture_preprocessing(head,cov,10,'cpu');self.assertEqual(float(got['normalized_future'][0,1,-1]-got['normalized_future'][1,1,-1]),-5)
  head[0,0]=np.nan
  with self.assertRaises(ValueError):m.capture_preprocessing(head,cov,10,'cpu')
class Preparation(unittest.TestCase):
 def fixture(self):
  import json
  from p3_phase2_timesfm import PAIRS
  H=10;m=dict(origin_date='2020-01-01',rain_site='existing',realization=0,rest_ratio=1,signal_ratio=1,pi_layer=1,head_unit='m',pumping_unit='m3/d',rainfall_unit='mm/d',daily_alignment=True,storage_value=.01)
  return dict(head=np.zeros((4,1024)),pumping=np.vstack([np.full(1034,80.),np.r_[np.full(1024,80.),np.zeros(10)],np.full(1034,80.),np.r_[np.full(1024,80.),np.full(10,120.)]]),rainfall=np.zeros((4,1034)),query_id=np.array(['q1','q2','q3','q4']),case_id=np.array(['new']*4),pair_id=np.array(['p1','p1','p2','p2']),schedule_id=np.array(['a','b','a','b']),metadata_json=np.array([json.dumps(dict(m,schedule_pair=p)) for p in [PAIRS[0],PAIRS[0],PAIRS[1],PAIRS[1]]]),protocol_sha256=np.array('0'*64))
 def test_new_storage_label_and_exact_origin(self):
  import p3_phase3_timesfm as m
  d=self.fixture();z=m.prepare_track(d,10,'raw',['new']);self.assertEqual(z['covariates'].shape,(4,2,1034));self.assertEqual(z['pair_id'].tolist(),['p1','p2'])
  d['pumping'][0,1024:]=100
  with self.assertRaises(ValueError):m.prepare_track(d,10,'raw',['new'])
 def test_leakage_and_missing_case_rejected(self):
  import p3_phase3_timesfm as m
  d=self.fixture()
  with self.assertRaises(ValueError):m.prepare_track(d,10,'raw',['new','missing'])
  d['E_true']=np.ones(4)
  with self.assertRaises(ValueError):m.prepare_track(d,10,'raw')
if __name__=='__main__':unittest.main(verbosity=2)
