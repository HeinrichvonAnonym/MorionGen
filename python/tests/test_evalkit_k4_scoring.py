import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.scoring import score
from motiogen.degradation.metrics import NAMES,normalize,points
from motiogen.degradation.positions import SOURCE


class ScoringTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12)
        self.p=rng.normal(size=(len(NAMES),3))
        self.p=np.broadcast_to(self.p,(4,len(NAMES),3)).copy()
        self.t=np.arange(4)/10
        self.r=Motion(self.p,[SOURCE[n] for n in NAMES],self.t,'reference')
        self.d=Motion(self.p.copy(),NAMES,self.t,'degraded')

    def test_translation_is_fit_error_and_normalization_keeps_root(self):
        shift=np.array([.1,.2,.3]);f=Motion(self.p+shift,NAMES,self.t,'fitted')
        s=score(self.r,self.d,f)['comparisons']
        for pair in ('F-D','F-R'):
            wrist=s[pair]['full']['positions_cm']['hands']['left_wrist']
            self.assertAlmostEqual(wrist['world']['p95'],np.linalg.norm(shift)*100)
            self.assertAlmostEqual(wrist['common_length_world']['p95'],np.linalg.norm(shift)*100)
            self.assertLess(wrist['pelvis_relative']['p95'],1e-10)
        self.assertEqual(s['D-R']['full']['positions_cm']['hands']['left_wrist']['world']['p95'],0)

    def test_pure_length_change_removed_only_by_score_normalization(self):
        # Uniform scale around a fixed hip midpoint preserves directions.
        mid=(self.p[:,NAMES.index('left_hip')]+self.p[:,NAMES.index('right_hip')])/2
        f=Motion(mid[:,None]+1.2*(self.p-mid[:,None]),NAMES,self.t,'fitted')
        before=f.positions.copy();s=score(self.r,self.d,f)['comparisons']['F-R']['full']
        self.assertGreater(s['positions_cm']['hands']['left_wrist']['world']['p95'],1)
        self.assertLess(s['positions_cm']['hands']['left_wrist']['common_length_world']['p95'],1e-10)
        self.assertLess(s['directions_deg']['trunk_axis']['p95'],1e-5)
        np.testing.assert_array_equal(before,f.positions)

    def test_missing_head_and_temporal_tolerance_keep_denominators(self):
        p=self.p.copy();p[:,NAMES.index('head')]=np.nan
        f=Motion(p,NAMES,self.t,'fitted')
        s=score(self.r,self.d,f,np.array([False,True,True,False]))['comparisons']['F-R']
        metric=s['full']['directions_deg']['head_neck_axis']
        self.assertEqual(metric['denominator'],4);self.assertEqual(metric['n'],0)
        self.assertIsNone(metric['p95'])
        self.assertEqual(s['source_jump_tolerance']['directions_deg']['trunk_axis']['denominator'],2)
        self.assertEqual(s['continuous']['directions_deg']['trunk_axis']['denominator'],2)


if __name__=='__main__':unittest.main()
