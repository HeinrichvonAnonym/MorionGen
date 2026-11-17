import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.scoring import score
from motiogen.evalkit.k.strict_acceptance import assess
from motiogen.evalkit.k.head_scoring import compare
from motiogen.degradation.metrics import NAMES
from motiogen.degradation.positions import SOURCE


class StrictTests(unittest.TestCase):
    def setUp(self):
        self.policy=json.loads((ROOT/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text())
        self.t=np.arange(4)/10;p=np.tile(np.random.default_rng(12).normal(size=(len(NAMES),3)),(4,1,1))
        self.r=Motion(p,[SOURCE[n] for n in NAMES],self.t,'R');self.d=Motion(p,NAMES,self.t,'D')
        self.f=Motion(p,NAMES,self.t,'F')
    def run_score(self):return score(self.r,self.d,self.f,fixed_observation_support=True)
    def test_within_target_relative_regression_is_nonblocking(self):
        old=self.run_score();self.f.positions+=.02;new=self.run_score()
        result=assess(new,old,self.policy)
        self.assertEqual(result['status'],'passed')
        self.assertGreater(next(x for x in result['required'] if x['metric']=='positions_cm/left_wrist/world')['p95'],1)
    def test_missing_supported_frame_fails_coverage(self):
        self.f.positions[1,NAMES.index('head')]=np.nan
        result=assess(self.run_score(),None,self.policy)
        self.assertEqual(result['status'],'failed')
        row=next(x for x in result['required'] if x['metric']=='positions_cm/head/world')
        self.assertEqual(row['status'],'coverage_failed')
    def test_new_values_cannot_replace_input_missing_samples(self):
        self.d.positions[0,NAMES.index('head')]=np.nan
        self.f.positions[1,NAMES.index('head')]=np.nan
        a=self.run_score();f=a['comparisons']['F-R']['full']['positions_cm']['head_neck']['head']['world']
        self.assertEqual(f['n'],2)
        self.assertEqual(assess(a,None,self.policy)['status'],'failed')
    def test_source_jump_tolerance_keeps_full_error_and_coverage(self):
        self.f.positions[1]+=1
        s=score(self.r,self.d,self.f,np.array([False,True,False,False]),fixed_observation_support=True)
        result=assess(s,None,self.policy)
        self.assertEqual(result['status'],'passed')
        self.assertGreater(s['comparisons']['F-R']['full']['positions_cm']['hands']['left_wrist']['world']['p95'],5)
        self.f.positions[1]=np.nan
        s=score(self.r,self.d,self.f,np.array([False,True,False,False]),fixed_observation_support=True)
        self.assertEqual(assess(s,None,self.policy)['status'],'failed')
    def test_old_pairwise_scores_cannot_enter_strict_acceptance(self):
        with self.assertRaises(ValueError):assess(score(self.r,self.d,self.f),None,self.policy)
    def test_position_profile_relative_rotation_is_unobservable(self):
        self.r.names.append('Spine3');self.r.positions=np.concatenate([self.r.positions,self.r.positions[:,:1]],1)
        self.r.rotations=np.tile(np.eye(3),(4,len(self.r.names),1,1))
        result=compare(self.r,self.d,{},self.t,np.zeros(4,bool))
        for pair in ('D-R','F-D','F-R'):
            self.assertEqual(result['head_neck_rotation_comparisons'][pair]['full']['n'],0)


if __name__=='__main__':unittest.main()
