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
