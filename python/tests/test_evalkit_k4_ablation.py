import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]


class AblationTests(unittest.TestCase):
    def test_factorial_holds_numerical_processing_and_root_policy_constant(self):
        folder=ROOT/'config/evalkit_k4_20261117/ablation_v1'
        conditions=[json.loads((folder/f'g{g}_s{s}.json').read_text()) for g in (0,1) for s in (0,1)]
        controls=[{k:v for k,v in c.items() if k not in ('condition','independent_geometry','solver')} for c in conditions]
        self.assertTrue(all(c==controls[0] for c in controls))
        self.assertEqual(controls[0]['postprocess'],'native_preserving')
        self.assertEqual(controls[0]['constraint_policy'],'direct_couplers')

    def test_directed_interventions_change_only_declared_factors(self):
        folder=ROOT/'config/evalkit_k4_20261117/ablation_v1'
        base=json.loads((folder/'g1_s1.json').read_text())
        for label,pelvis,domain in [('g1_s1_pelvis',True,False),('g1_s1_subtalar',False,True),('g1_s1_pelvis_subtalar',True,True)]:
            c=json.loads((folder/(label+'.json')).read_text())
            self.assertEqual(c['refinement'].pop('include_pelvis_line',False),pelvis)
            self.assertEqual(bool(c.pop('diagnostic_domain',None)),domain)
            for key in ('condition','status'):c.pop(key)
            self.assertEqual(c,{k:v for k,v in base.items() if k not in ('condition','status')})

    def test_subtalar_intervention_keeps_asset_and_coordinate_ranges(self):
        import opensim as osim
        import biomech
        from motiogen.evalkit.k.diagnostic_domain import apply
        from motiogen.evalkit.k.config import digest
        original=digest(biomech.MODEL);scratch=ROOT/'develop_logs/test_runs/20261117_k4'
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            path=Path(temp)/'model.osim';model=osim.Model(str(biomech.MODEL));model.printToXML(str(path))
            before=model.getCoordinateSet().getSize();receipt=apply(path,'release_subtalar_original_range')
            modified=osim.Model(str(path))
            self.assertEqual(modified.getCoordinateSet().getSize(),before)
            for name,change in receipt['changes'].items():
                self.assertTrue(change['locked_before']);self.assertFalse(modified.getCoordinateSet().get(name).getDefaultLocked())
                self.assertEqual(change['range_native_before'],change['range_native_after'])
        self.assertEqual(digest(biomech.MODEL),original)

    def test_root_views_are_translation_only_and_input_bound(self):
        from evalkit_k4_views import derive
        base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
        scratch=ROOT/'develop_logs/test_runs/20261117_k4';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            out=Path(temp);source=base/'runtime_inputs/000/kimodo/candidate'
            derive(base/'common_v1/000/kimodo',source,'kimodo','000',out,True)
            with np.load(out/'before/000/kimodo/fk.npz') as z:a=z['positions']
            with np.load(out/'after/000/kimodo/fk.npz') as z:b=z['positions']
            delta=b-a
            np.testing.assert_allclose(delta, np.broadcast_to(delta[:,:1],delta.shape),atol=1e-7)
            receipt=json.loads((out/'receipts/000_kimodo.json').read_text())
            self.assertTrue(all(Path(p).is_relative_to(source) for p in receipt['input_hashes']))
            subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_ablation_score.py'),
                            '--fits',str(out/'before'),'--baseline',str(base/'common_v1'),
                            '--output',str(out/'scores'),'--window','000','--method','kimodo'],
                           check=True,capture_output=True)
            score=json.loads((out/'scores/000_kimodo.json').read_text())
            self.assertEqual(set(score['comparisons']),{'D-R','F-D','F-R'})
            self.assertLessEqual(score['paired_baseline_clock']['clock']['frames'],score['clock']['frames'])
            self.assertEqual(score['comparisons']['F-R']['full']['directions_deg']['head_neck_axis']['n'],0)


if __name__=='__main__':unittest.main()
