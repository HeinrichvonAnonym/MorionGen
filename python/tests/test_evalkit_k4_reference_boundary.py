import copy,importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('boundary',ROOT/'python/scripts/evalkit_k4_reference_boundary_audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BoundaryAudit(unittest.TestCase):
    def fixture(self):
        clock=dict(clock={'frames':150,'source_jump_frames':0},comparisons={'D-R':{'full':{'n':150,'p95':0.01}},'F-R':{'p95':20.}},head_neck_rotation_comparisons={'D-R':{'full':{'n':150,'p95':0.001}}})
        return dict(window='001',method='test',original_reference_file='/reference/001.npz',original_clock=copy.deepcopy(clock),**clock)

    def test_fit_changes_do_not_change_boundary(self):
        a=self.fixture();b=copy.deepcopy(a);b['comparisons']['F-R']['p95']=1.
        self.assertEqual(module.differences(module.boundary(a),module.boundary(b)),[])

    def test_counts_clock_rotation_and_reference_change_detected(self):
        a=self.fixture()
        for mutate in [lambda b:b['comparisons']['D-R']['full'].update(n=149),
                       lambda b:b['original_clock']['clock'].update(source_jump_frames=1),
                       lambda b:b['head_neck_rotation_comparisons']['D-R']['full'].update(p95=.2),
                       lambda b:b.update(original_reference_file='/other.npz')]:
            b=copy.deepcopy(a);mutate(b)
            self.assertTrue(module.differences(module.boundary(a),module.boundary(b)))


if __name__=='__main__':unittest.main()
