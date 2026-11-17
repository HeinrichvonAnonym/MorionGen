import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.acceptance import compare


class AcceptanceTests(unittest.TestCase):
    def test_large_average_gain_cannot_hide_regional_regression(self):
        metric=lambda x:{'p95':x,'n':100,'denominator':100}
        old={'directions_deg':{'right_foot':metric(50.),'right_thigh':metric(11.)},
             'joint_angles_deg':{},'positions_cm':{},'temporal':{}}
        a={'comparisons':{p:{'full':copy.deepcopy(old)} for p in ('F-D','F-R')}}
        b=copy.deepcopy(a)
        for p in ('F-D','F-R'):
            b['comparisons'][p]['full']['directions_deg']['right_foot']['p95']=1.5
            b['comparisons'][p]['full']['directions_deg']['right_thigh']['p95']=17.
        result=compare(a,b)
        self.assertEqual(result['status'],'regression_blocked')
        self.assertEqual(len(result['regressions']),2)

    def test_lost_coverage_blocks_even_with_better_errors(self):
        v={'p95':10.,'n':100,'denominator':100}
        old={'directions_deg':{'trunk':v},'joint_angles_deg':{},'positions_cm':{},'temporal':{}}
        a={'comparisons':{p:{'full':copy.deepcopy(old)} for p in ('F-D','F-R')}};b=copy.deepcopy(a)
        for p in ('F-D','F-R'):b['comparisons'][p]['full']['directions_deg']['trunk'].update({'n':90,'p95':1.})
        self.assertEqual(compare(a,b)['status'],'regression_blocked')


if __name__=='__main__':unittest.main()
