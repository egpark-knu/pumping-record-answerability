"""Execution-driver tests: output semantics and stale checkpoint rejection, no inference."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
try:
 import p3_phase2_timesfm_execution as d
except ModuleNotFoundError:
 d=None

class Execution(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(d,"Execution driver absent")
 def test_contrast_is_difference_of_medians_and_keeps_pair_ids(self):
  q=np.zeros((4,10,9),np.float32)
  for i in range(4):q[i]=np.arange(9,dtype=np.float32)+i*10
  z={'query_id':np.array(['p1:a','p1:b','p2:a','p2:b']),'pair_id':np.array(['p1','p2']),
     'case_id':np.array(['c']*4),'schedule_id':np.array(['a','b']*2),'metadata_json':np.array(['{}']*4)}
  out=d.pack_output(q,z,{'horizon':10,'track':'raw'})
  np.testing.assert_array_equal(out['E_point'],np.full((2,10),-10.))
  np.testing.assert_array_equal(out['E_width_proxy'],np.full((2,10),16.))
  self.assertEqual(list(out['pair_id']),['p1','p2'])
 def test_wrong_quantile_shape_nonfinite_and_id_is_rejected(self):
  with self.assertRaises(ValueError):d.checked_prediction(np.zeros((10,8)),'a','a',10)
  with self.assertRaises(ValueError):d.checked_prediction(np.full((10,9),np.nan),'a','a',10)
  with self.assertRaises(ValueError):d.checked_prediction(np.zeros((10,9)),'b','a',10)
 def test_checkpoint_requires_identical_fingerprint_ids_and_range(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'chunk.npz';ids=np.array(['a','b'])
   d.atomic_npz(p,dict(quantiles=np.zeros((2,10,9),np.float32),query_id=ids,start=0,stop=2,fingerprint='good'))
   self.assertEqual(d.load_chunk(p,'good',ids,0,2,10).shape,(2,10,9))
   for fp,qid,start,stop in [('bad',ids,0,2),('good',ids[::-1],0,2),('good',ids,2,4)]:
    with self.assertRaises(ValueError):d.load_chunk(p,fp,qid,start,stop,10)

if __name__=='__main__':unittest.main(verbosity=2)
