import unittest
from motiogen.clock_audit import observed_hold_bounds


class ObservationBoundsTests(unittest.TestCase):
    def test_exact_timestamps_reduce_to_observed_streak(self):
        result=observed_hold_bounds([True,True,True],[5,5.25,5.5],[5,5.25,5.5])
        self.assertEqual(result,{'definite':True,'possible':True})

    def test_timing_interval_can_leave_result_ambiguous(self):
        self.assertEqual(observed_hold_bounds([True,True],[5,5.49],[5.02,5.52]),
                         {'definite':False,'possible':True})

    def test_outside_observation_breaks_hold(self):
        self.assertEqual(observed_hold_bounds([True,False,True],[5,5.5,6],[5,5.5,6]),
                         {'definite':False,'possible':False})

    def test_single_uncertain_observation_cannot_establish_two_states(self):
        # Even a broad interval is one timestamp, not two observations.
        self.assertEqual(observed_hold_bounds([True],[5],[6]),{'definite':False,'possible':False})

    def test_invalid_timestamp_interval_rejected(self):
        with self.assertRaises(ValueError):observed_hold_bounds([True],[6],[5])


if __name__=='__main__':unittest.main()
