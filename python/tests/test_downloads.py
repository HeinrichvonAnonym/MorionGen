import unittest
from unittest.mock import patch,Mock

from motiogen.downloads import DownloadPolicyError,direct_environment,resolve_asset_url,validate_asset_host


class TransportTest(unittest.TestCase):
    def test_proxy_removed(self):
        with patch.dict('os.environ',{'HTTPS_PROXY':'private','all_proxy':'private','KEEP':'value'},clear=True):
            self.assertEqual(direct_environment(),{'KEEP':'value'})

    def test_overseas_redirect_blocked_before_body_download(self):
        session=Mock(); session.__enter__=Mock(return_value=session); session.__exit__=Mock(return_value=False)
        session.head.return_value=Mock(status_code=302,headers={'Location':'https://cas-bridge.xethub.hf.co/file?Signature=private'})
        with patch('requests.Session',return_value=session):
            with self.assertRaisesRegex(DownloadPolicyError,'cas-bridge.xethub.hf.co') as error:
                resolve_asset_url('https://hf-mirror.com/a/resolve/b/c',{'asset_hosts':['hf-mirror.com'],'proxy_for_assets':False})
        self.assertNotIn('private',str(error.exception)); self.assertFalse(session.trust_env)
        self.assertEqual(session.head.call_count,1)

    def test_host_and_tls_checks(self):
        for url in ('http://hf-mirror.com/a','https://hf-mirror.com.evil/a','https://user:secret@hf-mirror.com/a'):
            with self.assertRaises(DownloadPolicyError):
                validate_asset_host(url)

    def test_official_proxy_redirect_allowed(self):
        session=Mock(); session.__enter__=Mock(return_value=session); session.__exit__=Mock(return_value=False)
        session.head.side_effect=[Mock(status_code=302,headers={'Location':'https://cas-bridge.xethub.hf.co/file?Signature=private'}),Mock(status_code=200)]
        settings={'asset_hosts':['huggingface.co','cas-bridge.xethub.hf.co'],'proxy_for_assets':True}
        with patch('requests.Session',return_value=session):
            self.assertTrue(resolve_asset_url('https://huggingface.co/model/resolve/main/file',settings).startswith('https://cas-bridge'))
        self.assertTrue(session.trust_env)


if __name__=='__main__':
    unittest.main()
