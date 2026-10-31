import unittest

import numpy as np
import torch

from motiogen.ardy_boundary import _exp_so3, _log_so3, apply, decay_weights


def rotations(rng, n, joints):
    return _exp_so3(torch.tensor(rng.normal(scale=.4, size=(n, joints, 3)), dtype=torch.float64))


class ArdyBoundaryTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.prev_root = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float64)
        self.prev_rot = rotations(rng, 4, 5)
        self.root = torch.tensor(rng.normal(size=(8, 3)), dtype=torch.float64)
        self.rot = rotations(rng, 8, 5)

    def test_weights_start_at_one_and_share_half_decay(self):
        fps = 20.
        inert = decay_weights('inertialization', 400, 400.).numpy()
        cross = decay_weights('crossfade', 400, 400.).numpy()
        half_inert = np.argmin(np.abs(inert-.5))/400.
        half_cross = np.argmin(np.abs(cross-.5))/400.
        self.assertAlmostEqual(half_inert, .168, places=2)
        self.assertAlmostEqual(half_inert, half_cross, places=2)
        self.assertEqual(float(decay_weights('inertialization', 8, fps)[0]), 1.)

    def test_so3_round_trip(self):
        back = _exp_so3(_log_so3(self.rot))
        self.assertLess(float((back-self.rot).abs().max()), 1e-9)

    def test_decay_starts_on_extrapolated_target(self):
        target_root = 2*self.prev_root[-1]-self.prev_root[-2]
        for treatment in ('inertialization', 'crossfade'):
            root, rot = apply(treatment, self.prev_root, self.prev_rot, self.root, self.rot, 20.)
            self.assertLess(float((root[0]-target_root).abs().max()), 1e-9)
            self.assertLess(float((root[-1]-self.root[-1]).norm()), float((self.root[0]-target_root).norm()))
            orth = rot@rot.transpose(-1, -2)
            self.assertLess(float((orth-torch.eye(3, dtype=orth.dtype)).abs().max()), 1e-9)

    def test_none_and_lowpass(self):
        root, rot = apply('none', self.prev_root, self.prev_rot, self.root, self.rot, 20.)
        self.assertTrue(torch.equal(root, self.root) and torch.equal(rot, self.rot))
        root, rot = apply('lowpass', self.prev_root, self.prev_rot, self.root, self.rot, 20.)
        self.assertEqual(tuple(root.shape), (8, 3))
        self.assertGreater(float(root.diff(dim=0).norm(dim=-1).mean()), 0.)
        self.assertLess(float(root.diff(dim=0).norm(dim=-1).mean()), float(self.root.diff(dim=0).norm(dim=-1).mean()))
        orth = rot@rot.transpose(-1, -2)
        self.assertLess(float((orth-torch.eye(3, dtype=orth.dtype)).abs().max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
