"""Two-pass compatibility: temporal candidates must not alter the frozen first pass."""
import sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k import refinement

class Continuation(unittest.TestCase):
    def test_candidate_is_confined_to_second_pass(self):
        entry=refinement.refine
        initial=np.zeros((3,2));first=initial+1;second=initial+2
        settings={'rescue_after_baseline':True,'interior_rescue':True,'baseline_max_nfev':35,
                  'max_nfev':100,'continue_arm_branch':True,'accept_continued_arm':True}
        with patch.object(refinement,'refine',side_effect=[(first,{'pass':1}),(second,{'pass':2})]) as nested:
            result,diag=entry('unused',None,None,None,initial,settings)
        baseline=nested.call_args_list[0].args[-1];candidate=nested.call_args_list[1].args[-1]
        self.assertFalse(baseline['continue_arm_branch']);self.assertFalse(baseline['accept_continued_arm'])
        self.assertFalse(baseline['interior_rescue']);self.assertEqual(baseline['max_nfev'],35)
        self.assertTrue(candidate['continue_arm_branch']);self.assertTrue(candidate['accept_continued_arm'])
        self.assertFalse(candidate['rescue_after_baseline']);self.assertIs(nested.call_args_list[1].args[-2],first)
        self.assertIs(result,second);self.assertEqual(diag['baseline_pass'],{'pass':1})
        self.assertTrue(settings['rescue_after_baseline'])

if __name__=='__main__':unittest.main()
