"""A reviewed low-memory cache profile must fail closed on a miss."""
import unittest
from unittest.mock import patch
from motiogen import local_llm, prompt_cache_review


class RequiredPromptCacheTests(unittest.TestCase):
    def test_miss_never_falls_back_to_large_encoder(self):
        cfg = {}
        scene = {'prompt': 'walk', 'updates': [{'prompt': 'turn'}]}

        def install_simulated_miss(config, baseline, out):
            fallback = local_llm.precompute_encoder
            local_llm.precompute_encoder = lambda *a, **kw: fallback(*a, **kw)

        with patch.object(local_llm, 'precompute_encoder') as large_encoder:
            with patch('motiogen.prompt_cache_review.check', return_value={'identity': 'exact'}), \
                 patch('motiogen.prompt_cache.install', side_effect=install_simulated_miss):
                prompt_cache_review.install_guard(cfg, 'ardy', scene, None)
                with self.assertRaisesRegex(RuntimeError, 'cannot instantiate'):
                    local_llm.precompute_encoder(cfg, 'ardy', ['walk', 'turn'])
                large_encoder.assert_not_called()

    def test_identity_rechecked_before_hit_use(self):
        cfg = {}

        def install_simulated_hit(config, baseline, out):
            local_llm.precompute_encoder = lambda *a, **kw: 'cached result'

        with patch.object(local_llm, 'precompute_encoder'), \
             patch('motiogen.prompt_cache_review.check', side_effect=[{'identity': 'exact'}, ValueError('source changed')]), \
             patch('motiogen.prompt_cache.install', side_effect=install_simulated_hit):
            prompt_cache_review.install_guard(cfg, 'ardy', {'prompt': 'walk'}, None)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                local_llm.precompute_encoder(cfg, 'ardy', ['walk'])


if __name__ == '__main__':
    unittest.main()
