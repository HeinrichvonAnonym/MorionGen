import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.original_domain import assess
from motiogen.evalkit.k.diagnostic_domain import apply
from motiogen.evalkit.k.config import digest

class OriginalDomainTests(unittest.TestCase):
    def test_diagnostics_do_not_clip_or_edit_original_model(self):
        original=digest(biomech.MODEL);labels=['knee_angle_r','subtalar_angle_r'];q=np.array([[170.,60.],[120.,0.]])
        saved=q.copy();result=assess(biomech.MODEL,labels,q)
        np.testing.assert_array_equal(q,saved);self.assertEqual(digest(biomech.MODEL),original)
        self.assertEqual(result['coordinates']['knee_angle_r']['outside_range_frames'],1)
        self.assertEqual(result['coordinates']['subtalar_angle_r']['moved_from_original_lock_frames'],1)
        self.assertIn('no physiological',result['scope'])
    def test_authorized_copy_expansion_preserves_coupling_functions(self):
        import xml.etree.ElementTree as E
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as tmp:
            path=Path(tmp)/'copy.osim';m=osim.Model(str(biomech.MODEL));m.printToXML(str(path))
            before=E.parse(path)
            policy=json.loads((ROOT/'config/evalkit_k4_20261117/independent_v1/independent_knee180.json').read_text())['kinematic_domain']
            audit=apply(path,policy);after=E.parse(path)
            a=[E.tostring(x) for x in before.findall('.//SpatialTransform')];b=[E.tostring(x) for x in after.findall('.//SpatialTransform')]
            self.assertEqual(a,b);self.assertEqual(audit['policy']['scope'],'independent_kinematic')
            m=osim.Model(str(path));self.assertAlmostEqual(m.getCoordinateSet().get('knee_angle_r').getRangeMax(),np.pi)

if __name__=='__main__':unittest.main()
