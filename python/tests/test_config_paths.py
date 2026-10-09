import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from motiogen.config import asset_root,environment_root,load_config,repo_root


class RelocatablePathTest(unittest.TestCase):
    def test_data_root_environment_override(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'MORIONGEN_DATA_ROOT':folder}):
            cfg=load_config()
            self.assertEqual(asset_root(cfg),Path(folder)/'outputs/assets')
            self.assertEqual(repo_root(cfg,'ardy'),Path(folder)/'baselines/ardy')
            self.assertEqual(environment_root(cfg,'ardy'),Path(folder)/'envs/motiogen-ardy')


if __name__=='__main__':
    unittest.main()
