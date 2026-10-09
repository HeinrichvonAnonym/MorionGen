import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from motiogen.assets import is_archive,safe_extract
from motiogen.config import ROOT
from motiogen.transfer import proxy_file


class ResumeTest(unittest.TestCase):
    def response(self, status, headers, body):
        value=Mock(status_code=status,headers=headers)
        value.__enter__=Mock(return_value=value); value.__exit__=Mock(return_value=False)
        value.iter_content.return_value=iter([body])
        return value

    def check(self, response, content, expected_size=6):
        directory=ROOT/'develop_logs/test_runs/.cache'; directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path=Path(temporary)/'model'; path.with_name('model.part').write_bytes(b'abc')
            with patch('motiogen.transfer._get',return_value=response),patch('motiogen.transfer.policy',return_value={'disk_reserve_gib':0}):
                proxy_file('https://huggingface.co/public',path,expected_size)
            self.assertEqual(path.read_bytes(),content)
            return len(list(Path(temporary).glob('*.range_ignored_*')))

    def test_resume_validates_range(self):
        self.assertEqual(self.check(self.response(206,{'Content-Length':'3','Content-Range':'bytes 3-5/6'},b'def'),b'abcdef'),0)

    def test_full_response_never_appended_to_partial(self):
        self.assertEqual(self.check(self.response(200,{'Content-Length':'6'},b'abcdef'),b'abcdef'),1)

    def test_bad_range_preserves_partial(self):
        directory=ROOT/'develop_logs/test_runs/.cache'; directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path=Path(temporary)/'model'; partial=path.with_name('model.part'); partial.write_bytes(b'abc')
            response=self.response(206,{'Content-Length':'3','Content-Range':'bytes 0-2/6'},b'def')
            with patch('motiogen.transfer._get',return_value=response),patch('motiogen.transfer.policy',return_value={'disk_reserve_gib':0}):
                with self.assertRaisesRegex(RuntimeError,'Invalid resume'):
                    proxy_file('https://huggingface.co/public',path,6)
            self.assertEqual(partial.read_bytes(),b'abc'); self.assertFalse(path.exists())

    def test_tar_gz_logical_name_and_licensed_symlink_preservation(self):
        directory=ROOT/'develop_logs/test_runs/.cache'; directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            root=Path(temporary); archive=root/'download.part'; target=root/'assets'; target.mkdir()
            preserved='processed_data/smpl_all_models/smplx/SMPLX_MALE.npz'
            licensed=root/'licensed.npz'; licensed.write_bytes(b'licensed')
            link=target/preserved; link.parent.mkdir(parents=True); link.symlink_to(licensed)
            with tarfile.open(archive,'w:gz') as output:
                for name,payload in ((preserved,b'archive copy'),('processed_data/sample.txt',b'data')):
                    info=tarfile.TarInfo(name); info.size=len(payload)
                    output.addfile(info,io.BytesIO(payload))
            self.assertTrue(is_archive(archive,'processed_data.tar.gz'))
            safe_extract(archive,target,preserve_existing=(preserved,))
            self.assertTrue(link.is_symlink()); self.assertEqual(licensed.read_bytes(),b'licensed')
            self.assertEqual((target/'processed_data/sample.txt').read_bytes(),b'data')


if __name__=='__main__':
    unittest.main()
