import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from motiogen.adapters import Adapter,ResourceGate
from motiogen.config import ROOT,load_config
from motiogen.resource_review import inference_review
from motiogen.training import preflight_training


class GateTest(unittest.TestCase):
    def test_training_denied_before_loading(self):
        cfg=load_config()
        adapter=Adapter(cfg,'motionlcm',ROOT/'develop_logs/test_runs/tests',{'name':'reach','kind':'reach'},0)
        with self.assertRaisesRegex(ResourceGate,'No reviewed training'):
            preflight_training(adapter)

    def test_stale_resource_evidence(self):
        root=ROOT/'develop_logs/test_runs/tests'; root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            cfg=load_config(); cfg['_source']=str(Path(folder)/'baseline.json')
            entry={'motionlcm':{'reviewed':True,'source_basis':'test fixture','ram_bound_bytes':1,'gpu_bound_bytes':1,'max_frames':196,
                               'files':[{'path':str(Path(folder)/'missing.ckpt'),'sha256':'not-valid'}]}}
            (Path(folder)/'inference_evidence.json').write_text(json.dumps(entry))
            adapter=Adapter(cfg,'motionlcm',folder,{'name':'fixture','kind':'official','duration_seconds':1},0)
            with self.assertRaisesRegex(ResourceGate,'fingerprint is stale'):
                inference_review(adapter)


if __name__=='__main__':
    unittest.main()
