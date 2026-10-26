import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from motiogen.weight_identity import snapshot,bind,verify_unchanged

class WeightIdentityTests(unittest.TestCase):
    def test_cache_and_post_run_change_check(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'assets').mkdir();weight=root/'weights.bin';weight.write_bytes(b'fixed pretrained weights')
            cfg={'data_root':tmp,'asset_root':'assets'}
            with patch('motiogen.weight_identity.dependencies',return_value=[weight]):
                first=bind(cfg,'test',root/'run')
                with patch('motiogen.weight_identity.checksum',side_effect=AssertionError('Should use unchanged stat cache')):
                    second=snapshot(cfg,'test')
                self.assertEqual(first['identity'],second['identity'])
                verify_unchanged(first)
                self.assertTrue(json.loads((root/'run/weight_binding.json').read_text())['after_run_unchanged'])
                weight.write_bytes(b'changed pretrained weights')
                with self.assertRaises(RuntimeError):verify_unchanged(first)
                changed=snapshot(cfg,'test')
                self.assertNotEqual(first['identity'],changed['identity'])

if __name__=='__main__':unittest.main()
