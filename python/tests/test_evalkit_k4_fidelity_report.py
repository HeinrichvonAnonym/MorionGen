import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'utils/motiogen/evalkit/reporting/fidelity.py';s=importlib.util.spec_from_file_location('report_fidelity',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class FidelityReport(unittest.TestCase):
    def test_good_native_pose_does_not_hide_inserted_failure(self):
        d={'metrics':{'position/wrist':{'expected':3,'passed':True},'interpolation_direction/arm':{'expected':6,'passed':False},'rotation/head':{'expected':0,'passed':None}}}
        r=m.quality_groups(d);self.assertEqual(len(r['observed_failures']),0);self.assertEqual(len(r['path_failures']),1);self.assertTrue(r['has_inserted_samples']);self.assertFalse(r['has_inserted_observations']);self.assertEqual(len(r['groups']['unsupported']),1)
    def test_native_clock_has_no_inserted_samples_even_with_anchor_rate(self):
        d={'metrics':{'position/wrist':{'expected':3,'passed':False},'interpolation_direction/arm':{'expected':0,'passed':None},'interpolation_rate/arm':{'expected':2,'passed':True}}}
        r=m.quality_groups(d);self.assertFalse(r['has_inserted_samples']);self.assertEqual(len(r['observed_failures']),1);self.assertEqual(r['path_failures'],[])
if __name__=='__main__':unittest.main()
