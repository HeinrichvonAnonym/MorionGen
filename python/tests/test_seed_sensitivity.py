import unittest
from motiogen.seed_sensitivity import seed_sensitivity


class SeedSensitivityTests(unittest.TestCase):
    def test_preserves_tasks_when_omitting_a_seed(self):
        result = seed_sensitivity([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 100],
                                  [0]*5 + [1]*5 + [2], [20,21,22,23,24]*2 + [20])
        self.assertEqual(result['balanced_tasks'], [0, 1])
        self.assertEqual(result['excluded_incomplete_tasks'], [2])
        self.assertEqual(result['balanced_mean_delta'], 7.)
        self.assertEqual([r['equal_task_mean_delta'] for r in result['per_seed']], [5,6,7,8,9])
        self.assertEqual([r['equal_task_mean_delta'] for r in result['leave_one_seed_out']], [7.5,7.25,7,6.75,6.5])

    def test_empty_and_incomplete_do_not_invent_zero(self):
        for values, tasks, seeds in [([], [], []), ([1.], [0], [20])]:
            result = seed_sensitivity(values, tasks, seeds)
            self.assertIsNone(result['balanced_mean_delta'])
            self.assertEqual(result['leave_one_seed_out'], [])

    def test_rejects_duplicate_nonfinite_and_wrong_seed(self):
        for values, tasks, seeds in [([1,2],[0,0],[20,20]), ([float('nan')],[0],[20]),
                                     ([1],[0],[25]), ([1],[],[])]:
            with self.assertRaises(ValueError): seed_sensitivity(values, tasks, seeds)

