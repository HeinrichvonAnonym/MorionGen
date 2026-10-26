import unittest
from motiogen.native_receipt_checks import intervention_checks


class NativeReceiptTests(unittest.TestCase):
    def test_primal_all_off_empty_counts_are_valid(self):
        row={'method':'primal','condition':{'ground_snap':0,'reprojection':0,'inertialization':0},
             'calls':{},'frames':600,'fps':30,'finite':True}
        self.assertTrue(intervention_checks(row,'.')['passed'])
        row['calls']=None
        self.assertFalse(intervention_checks(row,'.')['passed'])

    def test_primal_incorrect_branch_and_length_fail(self):
        row={'method':'primal','condition':{'ground_snap':1,'reprojection':1,'inertialization':1},
             'calls':{'initial_ground_snap':1,'ground_projection':43,'keypoint_reprojection':42,'inertialization':43},
             'frames':600,'fps':30,'finite':True}
        self.assertTrue(intervention_checks(row,'.')['passed'])
        row['calls']['inertialization']=0
        self.assertFalse(intervention_checks(row,'.')['passed'])
        row['calls']['inertialization']=43;row['frames']=602
        self.assertFalse(intervention_checks(row,'.')['passed'])

if __name__=='__main__':unittest.main()
