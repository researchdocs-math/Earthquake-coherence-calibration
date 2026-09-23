"""Invariants for quantile ranks, ties, weighted areas, and corruption repair."""
import unittest,numpy as np
from calibration import corrected_quantile,weighted_quantile,region_quantiles,alarm_metrics,region_grid
class CalibrationTests(unittest.TestCase):
 def test_rank(self):
  self.assertEqual(corrected_quantile(np.arange(39),.1),35)
  self.assertTrue(np.isinf(corrected_quantile(np.arange(19),.01)))
 def test_ties(self):
  self.assertEqual(corrected_quantile(np.ones(39)),1)
 def test_corruption(self):
  rng=np.random.default_rng(19)
  for _ in range(100):
   x=rng.normal(size=39);y=x.copy();y[rng.choice(39,2,replace=False)]=-1e6
   self.assertGreaterEqual(corrected_quantile(y,corruption=2),corrected_quantile(x))
 def test_weighted_quantile(self):
  x=np.array([1,2,2,3,4]);w=np.array([1,3,8,2,1]);q=weighted_quantile(x,w,.95)
  self.assertLessEqual(w[x>q].sum()/w.sum(),.05)
 def test_region_reduction(self):
  rng=np.random.default_rng(4);z=rng.normal(size=(3,20,20));r=region_grid((20,20));w=rng.uniform(.3,1,(20,20))
  for a in z:
   q=region_quantiles(a[None],r,w)[0];m=alarm_metrics(a[None],q,r,w)[0];self.assertEqual(m['any_exceed'],0)
if __name__=='__main__':unittest.main()
