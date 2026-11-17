import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class PanelTests(unittest.TestCase):
    def test_selection_requires_complete_accounting_and_is_fixed(self):
        scratch=ROOT/'develop_logs/test_runs/20261117_k4';scratch.mkdir(parents=True,exist_ok=True)
        methods=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root=Path(temp);(root/'scores').mkdir();out=root/'panel.json'
            (root/'identity.json').write_text('{}')
            (root/'completion.json').write_text(json.dumps({'expected':656,'results':[]}))
            command=[sys.executable,str(ROOT/'python/scripts/evalkit_k4_panel.py'),'--baseline',str(root),'--output',str(out)]
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertFalse(out.exists())
            results=[]
            for i in range(82):
                for j,method in enumerate(methods):
                    key=f'{i:03d}';results.append([key,method,'completed'])
                    metric={'directions_deg':{'trunk':{'p95':i}},'joint_angles_deg':{},'positions_cm':{}}
                    for group in ('shoulders','hands','knees','feet'):
                        metric['positions_cm'][group]={'joint':{'world':{'p95':100-abs(i-80)+j*.01}}}
                    (root/'scores'/f'{key}_{method}.json').write_text(json.dumps({'comparisons':{'F-D':{'full':metric}}}))
            (root/'completion.json').write_text(json.dumps({'expected':656,'results':results}))
            subprocess.run(command,check=True,capture_output=True)
            rows=json.loads(out.read_text())['records']
            self.assertEqual(len(rows),18)
            self.assertIn(('074','kimodo'),{(r['window'],r['method']) for r in rows})
            self.assertIn(('017','primal'),{(r['window'],r['method']) for r in rows})
            original=out.read_bytes();subprocess.run(command,check=True,capture_output=True)
            self.assertEqual(out.read_bytes(),original)


if __name__=='__main__':unittest.main()
