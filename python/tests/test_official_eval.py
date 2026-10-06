import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motiogen.adapters import MissingResources,Unsupported
from motiogen.config import ROOT,load_config
from motiogen.official_eval import evaluate_official,preflight
from motiogen.runtime import checksum


class OfficialGateTest(unittest.TestCase):
    def test_unavailable_native_evaluator(self):
        with self.assertRaises(Unsupported):
            preflight(load_config(),'ardy','absent.json')

    def test_three_demo_embeddings_cannot_produce_distribution_scores(self):
        root=ROOT/'develop_logs/test_runs/tests'; root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder=Path(folder)
            artifact=folder/'fixture.bin'; artifact.write_bytes(b'test-only fixture')
            archive=folder/'tiny.npz'
            np.savez(archive,generated_motion=np.zeros((3,512)),paired_text=np.zeros((3,512)),reference_motion=np.zeros((3,512)))
            protocol={'evaluator_id':'HumanML3D-T2M-finest','evaluator_checkpoint':str(artifact),'evaluator_sha256':checksum(artifact),
                      'normalization_file':str(artifact),'normalization_sha256':checksum(artifact),'embedding_archive':str(archive),'embedding_sha256':checksum(archive),
                      'reference_dataset':'synthetic unit fixture only','reference_split':'fixture','preprocess_source':'fixture','skeleton':'humanml22','fps':20}
            file=folder/'protocol.json'; file.write_text(json.dumps(protocol))
            with patch('motiogen.official_eval.budgets',return_value={'ram_bytes':8*1024**3}):
                with self.assertRaisesRegex(MissingResources,'Tiny demos are insufficient'):
                    evaluate_official(load_config(),'motionlcm',file,folder,0)


if __name__=='__main__':
    unittest.main()
