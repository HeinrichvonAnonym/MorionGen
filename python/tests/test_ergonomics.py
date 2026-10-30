"""Rules and bounds, separate from any claim of human scoring validity."""
import unittest
from motiogen.ergonomics import component_scores,score,upper_arm_base,lower_arm_base,neck_base,trunk_base,wrist_base,_combine,tables

class ErgonomicsRules(unittest.TestCase):
    def test_boundaries(self):
        for boundary,below,above in [(-20,2,1),(20,1,2),(45,2,3),(90,3,4)]:
            self.assertEqual(upper_arm_base(boundary-1e-5),below)
            self.assertEqual(upper_arm_base(boundary+1e-5),above)
        for a,b in [(59.999,2),(60,1),(100,1),(100.001,2)]:self.assertEqual(lower_arm_base(a),b)
        self.assertEqual(neck_base(-.001,'RULA'),4)
        self.assertEqual(neck_base(10.001,'RULA'),2)
        self.assertEqual(neck_base(20.001,'RULA'),3)
        self.assertEqual(trunk_base(0,'RULA'),2)
        self.assertEqual(trunk_base(-30,'REBA'),3)
        for method in ['RULA','REBA']:
            self.assertEqual(wrist_base(15.001,method),wrist_base(-15.001,method))

    def test_unknown_not_neutral(self):
        for method,expected in [('RULA',[1,7]),('REBA',[1,15])]:
            out=score({},method)
            self.assertEqual(out['score_range'],expected)
            self.assertIsNone(out['total']);self.assertTrue(out['missing_inputs'])

    def test_selected_source_neck_trunk_wrist_boundaries(self):
        # Explicit expected bins from the selected source version, including
        # exact boundaries. These test rules, not human landmark validity.
        neck_cases={'RULA':[(-.001,4),(0,1),(9.999,1),(10,1),(10.001,2),(19.999,2),(20,2),(20.001,3)],
                    'REBA':[(-.001,2),(0,1),(19.999,1),(20,1),(20.001,2)]}
        trunk_cases={'RULA':[(0,2),(19.999,2),(20,2),(20.001,3),(59.999,3),(60,3),(60.001,4)],
                     'REBA':[(-20.001,3),(-20,2),(-19.999,2),(-.001,2),(0,1),(.001,2),(20,2),(20.001,3),(60,3),(60.001,4)]}
        for method,cases in neck_cases.items():
            for value,expected in cases:self.assertEqual(neck_base(value,method),expected,(method,value))
        for method,cases in trunk_cases.items():
            for value,expected in cases:self.assertEqual(trunk_base(value,method),expected,(method,value))
        for sign in [-1,1]:
            for value,rula,reba in [(0,1,1),(.001,2,1),(14.999,2,1),(15,2,1),(15.001,3,2)]:
                self.assertEqual(wrist_base(sign*value,'RULA'),rula)
                self.assertEqual(wrist_base(sign*value,'REBA'),reba)

    def test_position_angles_mirror_rotation_and_scale(self):
        import numpy as np
        from motiogen.posture_geometry import flexion
        # Two different limb configurations, neither a straight/degenerate arm.
        a=np.array([[0.,1.,0.],[.2,.8,.3]])
        b=np.zeros((2,3));c=np.array([[1.,0.,0.],[.6,-.3,.2]])
        expected=flexion(a,b,c)
        rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        for scale in [.7,1.,1.4]:
            for mirror in [-1.,1.]:
                transform=np.diag([mirror,1.,1.])@rotation
                translated=[v@transform*scale+np.array([2.,-3.,.4]) for v in [a,b,c]]
                np.testing.assert_allclose(flexion(*translated),expected,atol=1e-10)

    def test_reba_original_paper_worked_example(self):
        # Hignett/McAtamney 2000 worked example: A=8, B=4,
        # coupling +1 => B'=5; C=10, activity +1 => 11.
        c=dict(trunk=(5,),neck=(2,),legs=(3,),upper_arm=(3,),lower_arm=(2,),wrist=(1,),load=(0,),coupling=(1,),activity=(1,))
        result,a,b=_combine('REBA',tuple(sorted(c.items())))
        self.assertEqual((result,a,b),((11,),(8,),(4,)))

    def test_twist_sidebend_and_seated_support(self):
        c,_=component_scores(dict(trunk_flexion_deg=30,trunk_twisted=1,trunk_side_bent=1),'REBA')
        self.assertEqual(c['trunk'],(4,))
        c,_=component_scores(dict(trunk_flexion_deg=30,trunk_twisted=1,trunk_side_bent=1,trunk_supported_seated=1),'RULA')
        self.assertEqual(c['trunk'],(3,))
        c,_=component_scores(dict(knee_flexion_deg=90,seated=1,leg_support_score=1),'REBA')
        self.assertEqual(c['legs'],(1,))

    def test_bad_inputs_and_extension(self):
        with self.assertRaises(ValueError):score({'elbow_flexion_deg':float('nan')},'RULA')
        with self.assertRaises(ValueError):score({'load_score':4},'REBA')
        self.assertIn('trunk_extension_rule_not_specified',score({'trunk_flexion_deg':-15},'RULA')['missing_inputs'])

    def test_all_table_indices(self):
        # Exhaust each score combination against named nested lookup, using
        # a different traversal from the production flattened-index logic.
        t=tables()['REBA'];count=0
        for tr,row in enumerate(t['A'],1):
            for n in range(1,4):
                for leg in range(1,5):
                    for upper,brow in enumerate(t['B'],1):
                        for lower in range(1,3):
                            for wrist in range(1,4):
                                a=[row[i:i+4] for i in range(0,12,4)][n-1][leg-1]
                                b=[brow[:3],brow[3:]][lower-1][wrist-1]
                                c=dict(trunk=(tr,),neck=(n,),legs=(leg,),upper_arm=(upper,),lower_arm=(lower,),wrist=(wrist,),load=(0,),coupling=(0,),activity=(0,))
                                result,_,_=_combine('REBA',tuple(sorted(c.items())))
                                self.assertEqual(result,(t['C'][a-1][b-1],));count+=1
        self.assertEqual(count,2160)

if __name__=='__main__':unittest.main()
