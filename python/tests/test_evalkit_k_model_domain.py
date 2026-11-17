import unittest,tempfile
from pathlib import Path
import numpy as np
from motiogen.evalkit.k.model_domain import prepare
from motiogen.evalkit.k.config import ROOT,digest

class ModelDomainTests(unittest.TestCase):
    def test_copy_only_and_locked_column_realization(self):
        import opensim as osim
        import biomech
        osim.Logger.setLevelString('error')
        source=biomech.MODEL;before=digest(source);model=osim.Model(str(source));state=model.initSystem();cs=model.getCoordinateSet();labels=[cs.get(i).getName() for i in range(cs.getSize())];data=np.zeros((2,len(labels)));data[:,labels.index('subtalar_angle_l')]=23.
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            dest=Path(folder)/'probe.osim';realized,receipt=prepare(source,dest,labels,data,'ankle');probe=osim.Model(str(dest));ps=probe.initSystem()
            self.assertFalse(probe.getCoordinateSet().get('subtalar_angle_l').getLocked(ps))
            self.assertEqual(float(realized[:,labels.index('subtalar_angle_l')].max()),0.)
            self.assertEqual(float(data[:,labels.index('subtalar_angle_l')].max()),23.)
            self.assertAlmostEqual(np.degrees(probe.getCoordinateSet().get('knee_angle_l').getRangeMax()),140.)
            self.assertEqual(digest(source),before)
            self.assertIn('kinematic-domain diagnostic',receipt['scope'])
        with self.assertRaises(ValueError):prepare(source,source,labels,data,'ankle')

    def test_hip_chart_changes_only_flexion_and_preserves_full_turn_fk(self):
        import opensim as osim
        import biomech
        from motiogen.evalkit.k.model_domain import PROFILES
        osim.Logger.setLevelString('error')
        source=biomech.MODEL;before=digest(source)
        model=osim.Model(str(source));state=model.initSystem();cs=model.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        data=np.zeros((2,len(labels)))
        for name,value in [('hip_adduction_l',-60),('hip_rotation_l',-80),('knee_angle_l',160),('ankle_angle_l',-70)]:
            data[:,labels.index(name)]=value
        data[:,labels.index('hip_flexion_l')]=[-200,160]
        original_ranges=PROFILES['combined']['ranges']
        self.assertEqual({k:v for k,v in PROFILES['hip_flexion_chart']['ranges'].items() if k!='hip_flexion'},
                         {k:v for k,v in original_ranges.items() if k!='hip_flexion'})
        self.assertEqual(original_ranges['hip_flexion'],(-90,175))
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            path=Path(folder)/'chart.osim';q,_=prepare(source,path,labels,data,'hip_flexion_chart')
            probe=osim.Model(str(path));ps=probe.initSystem();points=[]
            for row in q:
                biomech._set(probe,ps,labels,row)
                points.append([biomech._child_origin(probe,ps,j) for j in ['hip_l','walker_knee_l','ankle_l','mtp_l']])
            np.testing.assert_allclose(points[0],points[1],atol=1e-12)
            self.assertEqual(digest(source),before)
