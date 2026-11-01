import unittest

import torch

from motiogen import chois_guidance as cg


class Model:
    def __init__(self, steps=60):
        self.betas = torch.zeros(1); self.num_timesteps = steps; self.calls = []

    def p_sample(self, x, t, *args, **kwargs):
        self.calls.append(('plain', int(t[0]))); return x+torch.randn_like(x)

    def p_sample_guided_reconstruction_guidance(self, x, t, *args, **kwargs):
        self.calls.append(('guided', int(t[0]))); return x+torch.randn_like(x)


class ChoisGuidanceTest(unittest.TestCase):
    def test_options_default_to_native(self):
        self.assertEqual(cg.options({'guidance': 1}), {'terms': 'both', 'scale': 1., 'last_step': 10, 'direction': 'loss'})
        for bad in ({'terms': 'hands'}, {'direction': 'up'}, {'last_step': 1}, {'scale': -1}):
            with self.assertRaises(ValueError):
                cg.options(bad)

    def test_equal_norm_direction_keeps_norm_and_support(self):
        g = torch.zeros(2, 5, 4); g[0, 1:3] = torch.arange(8.).reshape(2, 4)+1
        gen = torch.Generator(); gen.manual_seed(3)
        direction, support = cg.equal_norm_direction(g, gen)
        self.assertAlmostEqual(float(direction.norm()), float(g.norm()), places=4)
        self.assertTrue(torch.all(direction[g == 0] == 0))
        self.assertAlmostEqual(support, 8/40)

    def test_guided_loop_steps_and_noise_match_native_order(self):
        for last in (10, 50):
            model = Model(); torch.manual_seed(0)
            out = cg.guided_loop(model, last)((1, 3), None, guidance_fn=lambda *a: 0)
            guided = [t for kind, t in model.calls if kind == 'guided']
            self.assertEqual(guided, list(range(last-1, 0, -1)))
            self.assertEqual([t for _, t in model.calls], list(range(59, -1, -1)))
            torch.manual_seed(0); x = torch.randn(1, 3)
            for _ in range(60):
                x = x+torch.randn_like(x)
            self.assertTrue(torch.equal(out, x))

    def test_no_guidance_function_is_plain_sampling(self):
        model = Model()
        cg.guided_loop(model, 50)((1, 3), None, guidance_fn=None)
        self.assertTrue(all(kind == 'plain' for kind, _ in model.calls))


if __name__ == '__main__':
    unittest.main()
