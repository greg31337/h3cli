import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metal_native_bench import sol_policy_check

class Records(unittest.TestCase):
    def fixture(self):
        return {'evaluations':2,'blocks':2,'metal_options':{'sol-min-exact':.75,'sol-dense-steps':1,'sol-dense-sigma':.9,'sol-dense-layers':1},
                'sol_policy':[{'step':s,'block':b,'video_sigma':1 if s==1 else .5,'audio_sigma':1 if s==1 else .5,
                               'decision':'early-evaluation' if s==1 else 'early-layer' if b==0 else 'routed'} for s in (1,2) for b in (0,1)],
                'sol':[{'step':2,'block':1,'exact':80,'approximate':20,'protected':40,'local':10,'protected_rows':7}]}
    def test_dense_first_then_actual_routing(self):self.assertTrue(sol_policy_check(self.fixture()))
    def test_missing_or_duplicate_records(self):
        r=self.fixture();r['sol']=[];self.assertFalse(sol_policy_check(r))
        r=self.fixture();r['sol_policy'][-1]=copy.deepcopy(r['sol_policy'][0]);self.assertFalse(sol_policy_check(r))
    def test_noise_is_both_modalities(self):
        r=self.fixture();r['sol_policy'][-1]['audio_sigma']=.95;self.assertFalse(sol_policy_check(r))
        r['sol_policy'][-1]['decision']='high-noise';r['sol']=[];self.assertTrue(sol_policy_check(r))
    def test_no_false_dense_evidence(self):
        r=self.fixture();r['metal_options']['sol-min-exact']=1
        for x in r['sol_policy']:x['decision']='all-exact'
        self.assertFalse(sol_policy_check(r));r['sol']=[];self.assertTrue(sol_policy_check(r))
    def test_bad_counts_or_nonfinite_sigma(self):
        r=self.fixture();r['sol'][0]['protected']=100;self.assertFalse(sol_policy_check(r))
        r=self.fixture();r['sol_policy'][0]['video_sigma']=float('nan');self.assertFalse(sol_policy_check(r))

if __name__=='__main__':unittest.main()
