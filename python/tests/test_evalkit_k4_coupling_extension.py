import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.diagnostic_domain import apply as domain
from motiogen.evalkit.k.coupling_extension import apply,hermite
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.degradation.metrics import angle

class CouplingExtensionTests(unittest.TestCase):
    def test_hermite_boundary_values_and_slopes(self):
        h=1e-6;f=lambda x:hermite(x,2.,3.,.3,.2,0.)
        self.assertAlmostEqual(f(2),.3);self.assertAlmostEqual(f(3),0)
        self.assertAlmostEqual((f(2+h)-f(2-h))/(2*h),.2,places=8)
        self.assertAlmostEqual((f(3+h)-f(3-h))/(2*h),0,places=8)
    def test_saved_fk_preserves_old_domain_and_reaches_deep_flexion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as temp:
            old=Path(temp)/'old.osim';new=Path(temp)/'new.osim'
            model=osim.Model(str(biomech.MODEL));model.printToXML(str(old))
            cfg=json.loads((ROOT/'config/evalkit_k4_20261117/independent_v1/independent_knee180.json').read_text())
            domain(old,cfg['kinematic_domain'])
            import shutil
            shutil.copyfile(old,new)
            receipt=apply(new,{'version':'knee_auxiliary_hermite_v2','boundary_deg':140.,'end_deg':180.,'grid_deg':.1})
            model=osim.Model(str(old));state=model.initSystem();cs=model.getCoordinateSet();labels=[cs.get(i).getName() for i in range(cs.getSize())]
            row=np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
            q=np.tile(row,(5,1))
            for side in ['l','r']:q[:,labels.index('knee_angle_'+side)]=[0,30,90,139.95,180]
            p,n=model_fk(old,labels,q,direct_couplers=True);f,_=model_fk(new,labels,q,direct_couplers=True)
            self.assertLess(np.max(abs(f[:4]-p[:4])),1e-6)
            for side in ['left','right']:
                hip,knee,ankle=[f[-1,n.index(side+'_'+name)] for name in ['hip','knee','ankle']]
                self.assertGreater(180-float(angle(hip-knee,ankle-knee)),175)
            self.assertLess(max(r['original_domain_max_error_native'] for r in receipt['functions'].values()),1e-5)

if __name__=='__main__':unittest.main()
