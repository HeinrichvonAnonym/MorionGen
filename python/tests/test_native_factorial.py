import sys
import unittest
import itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.native_analysis import factorial_block
from motiogen.native_analysis import root_dwell
from motiogen.native_analysis import wall_observation_recovery
import numpy as np

class FactorialTests(unittest.TestCase):
    def test_wall_hold_uses_timestamps(self):
        positions=np.zeros((3,3))
        self.assertFalse(wall_observation_recovery(positions,[5.,5.1,5.2],[0,0,0])['success'])
        result=wall_observation_recovery(positions,[5.,5.3,5.6],[0,0,0])
        self.assertTrue(result['success'])
        self.assertEqual(result['first_entry_after_request_seconds'],0.)
    def test_final_dwell_is_not_transient_arrival(self):
        joints=np.zeros((30,1,3));joints[15:,0,0]=1
        result=root_dwell(joints,10,[0,0,0],tail_seconds=1)
        self.assertEqual(result['tail_goal_occupancy_fraction'],0.)
        self.assertEqual(result['tail_stationary_at_goal_fraction'],0.)
    def rows(self):
        return [{'condition':{'a':a,'b':b},'values':{'error':10+2*a+3*b+5*a*b}} for a in (0,1) for b in (0,1)]
    def test_marginal_and_interaction(self):
        result=factorial_block(self.rows(),['a','b'])
        actual={tuple(r['factors']):r['delta'] for r in result['contrasts']}
        self.assertEqual(actual,{('a',):4.5,('b',):5.5,('a','b'):5.})
        simple={(r['factor'],tuple(r['fixed'].items())):r['delta'] for r in result['simple_contrasts']}
        self.assertEqual(simple,{('a',(('b',0),)):2.,('a',(('b',1),)):7.,
                                 ('b',(('a',0),)):3.,('b',(('a',1),)):8.})
    def test_incomplete_not_estimated(self):
        self.assertFalse(factorial_block(self.rows()[:-1],['a','b'])['complete'])
    def test_exact_null_does_not_depend_on_randomized_condition_order(self):
        rows=[{'condition':{'a':a,'b':b},'values':{'rate':1. if b else .02}}
              for a in (0,1) for b in (0,1)]
        for order in itertools.permutations(rows):
            effects={tuple(r['factors']):r['delta'] for r in factorial_block(order,['a','b'])['contrasts']}
            self.assertEqual(effects[('a',)],0.)
            self.assertEqual(effects[('a','b')],0.)
            self.assertEqual(effects[('b',)],.98)
    def test_real_small_effect_is_not_clamped_to_zero(self):
        rows=[{'condition':{'a':a},'values':{'value':a*1e-18}} for a in (0,1)]
        self.assertEqual(factorial_block(rows,['a'])['contrasts'][0]['delta'],1e-18)
    def test_simple_effects_average_to_marginal_in_three_factor_design(self):
        rows=[{'condition':{'g':g,'r':r,'i':i},
               'values':{'error':(1-g)*(10+2*r+3*i+5*r*i)}}
              for g in (0,1) for r in (0,1) for i in (0,1)]
        result=factorial_block(rows,['g','r','i'])
        self.assertEqual(len(result['simple_contrasts']),12)
        for factor in ['g','r','i']:
            simple=[c['delta'] for c in result['simple_contrasts'] if c['factor']==factor]
            marginal=next(c['delta'] for c in result['contrasts'] if c['factors']==[factor])
            self.assertAlmostEqual(np.mean(simple),marginal)
        grounded=[c['delta'] for c in result['simple_contrasts'] if c['factor']=='r' and c['fixed']['g']==1]
        self.assertEqual(grounded,[0.,0.])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):factorial_block(self.rows()+self.rows()[:1],['a','b'])
    def test_missing_not_zero(self):
        rows=self.rows();rows[0]['values']['error']=None
        result=factorial_block(rows,['a','b'])
        self.assertEqual(result['contrasts'],[])
        self.assertEqual(result['simple_contrasts'],[])
        self.assertEqual(result['excluded_metrics'],['error'])

if __name__=='__main__':unittest.main()
