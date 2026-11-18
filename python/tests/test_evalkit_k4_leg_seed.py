from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]


def test_leg_direction_seed_recovers_known_fk_without_foot_or_reference():
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.leg_seed import LegDirectionSeeder
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    nominal=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    for side,s in [('left','l'),('right','r')]:
        seeder=LegDirectionSeeder(biomech.MODEL,labels,side);truth=nominal.copy()
        for name,value in [('hip_flexion',65),('hip_adduction',10),('hip_rotation',20),('knee_angle',105)]:
            truth[labels.index(name+'_'+s)]=value
        observed=seeder.points(truth)
        seeded,receipt=seeder.seed(nominal,observed)
        assert receipt['status']=='computed' and max(receipt['direction_error_deg'])<1e-5
        actual=seeder.points(seeded)
        np.testing.assert_allclose(actual-actual[0],observed-observed[0],atol=1e-7)
        unchanged=[i for i in range(len(labels)) if i not in seeder.cols+[seeder.knee_col]]
        np.testing.assert_array_equal(seeded[unchanged],nominal[unchanged])
        assert seeder.seed(nominal,np.array([[0,0,0],[0,-1,0],[0,-2,0]]))[1]['status']=='weak_knee_plane'
