"""Deterministic synthetic unit tests; fixtures are not experimental observations."""
import unittest,itertools,numpy as np
from run_experiment import cumulative_at,spectral_features,domains,FREQ,key,mel_features
from extract_features import spectra
class PipelineTests(unittest.TestCase):
 def test_constant_psd_integral(self):
  x=np.ones((2,len(FREQ)));e=np.array([[20.,21.,800.],[30.,30.5,1200.]])
  np.testing.assert_allclose(cumulative_at(x,e),e,atol=1e-9)
 def test_linear_psd_integral(self):
  x=np.tile(2*FREQ+3,(2,1));e=np.array([20.,21.1,55.3,800.]);np.testing.assert_allclose(cumulative_at(x,e),np.broadcast_to(e**2+3*e,(2,4)),rtol=1e-10)
 def test_nonnegative_narrow_bins(self):
  x=np.random.default_rng(0).uniform(0,1,(5,len(FREQ)));v=np.array([20,30,40,50,60]);f=spectral_features(x,v,1);self.assertTrue(np.isfinite(f).all());np.testing.assert_allclose(np.exp(f).sum(1),1,rtol=1e-10)
 def test_gain_invariance(self):
  x=np.random.default_rng(1).uniform(.1,1,(5,len(FREQ)));v=np.arange(20,70,10);np.testing.assert_allclose(spectral_features(x,v,1),spectral_features(x*100,v,1),atol=1e-10)
 def test_beta0_ignores_speed(self):
  x=np.random.default_rng(1).uniform(.1,1,(5,len(FREQ)));np.testing.assert_array_equal(spectral_features(x,np.full(5,20),0),spectral_features(x,np.full(5,60),0))
 def test_spatial_linear_process_fixture(self):
  v=np.array([20,30,40,50,60]);p=FREQ[None,:]/v[:,None]**2;x=spectral_features(p,v,1);np.testing.assert_allclose(x,np.broadcast_to(x[0],x.shape),atol=1e-10)
 def test_gain_feature_dimension(self):
  self.assertEqual(spectral_features(np.ones((2,len(FREQ))),np.array([20,60]),1,True).shape,(2,129))
 def test_mel_finite(self):
  x=mel_features(np.ones((2,len(FREQ))));self.assertEqual(x.shape,(2,128));np.testing.assert_allclose(np.exp(x).sum(1),1)
 def test_split_counts(self):
  labels=np.array(list(itertools.product(range(118),range(0,360,45),[20,30,40,50,60],[500,1000],[0,1])))
  for r in [0,1]:
   tr,te,d=domains(labels,r);self.assertEqual(int(tr.sum()),1416);self.assertEqual(int(te.sum()),17464);self.assertFalse(np.any(tr&te));self.assertEqual(sum((d=='speed')&te),1888);self.assertEqual(sum((d=='id')&te),1416);self.assertEqual(sum((d=='speed_direction_load')&te),1888)
 def test_tie_break(self):
  rows=[{'cv_macro_accuracy':.5,'C':1,'gamma_multiplier':g,'beta':b} for b in [1,0] for g in [4,.25,1]]
  self.assertEqual(sorted(rows,key=key)[0],{'cv_macro_accuracy':.5,'C':1,'gamma_multiplier':1,'beta':0})
 def test_timestamp_processing(self):
  t=np.arange(15000)/6000;a=np.c_[np.sin(2*np.pi*100*t),np.zeros(len(t)),np.ones(len(t))];p,q=spectra(np.c_[t,a],40);self.assertTrue(q['valid']);self.assertEqual(p.shape,(2,513));self.assertLess(abs(FREQ[p[0].argmax()]-100),6)
 def test_timestamp_gap_reject(self):
  t=np.arange(15000)/6000;t[7500:]+=.03;p,q=spectra(np.c_[t,np.zeros((len(t),3))],40);self.assertIsNone(p);self.assertFalse(q['valid'])
 def test_duplicate_times_averaged(self):
  t=np.arange(15000)/6000;a=np.c_[np.sin(2*np.pi*100*t),np.zeros(len(t)),np.ones(len(t))];arr=np.c_[t,a];arr=np.insert(arr,1000,arr[1000],axis=0);p,q=spectra(arr,40);self.assertTrue(q['valid']);self.assertEqual(q['duplicate_timestamps'],1)
 def test_negative_timestamp_reject(self):
  t=np.arange(15000)/6000;t[1000]=0;p,q=spectra(np.c_[t,np.zeros((len(t),3))],40);self.assertIsNone(p)
 def test_gap_crosses_window_edge(self):
  t=np.arange(18001)/6000;t=t[(t<.97)|(t>1.03)]
  p,q=spectra(np.c_[t,np.zeros((len(t),3))],40)
  self.assertIsNone(p);self.assertGreater(q['central_max_gap'],.02)
if __name__=='__main__':unittest.main(verbosity=2)
