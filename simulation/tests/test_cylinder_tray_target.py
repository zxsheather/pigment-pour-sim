import unittest

from cylinder_tray_vision import ConfirmedTrayTarget


def estimate(x):
    return dict(world_xy_m=[x, .2], method='blue_bearings')


class ConfirmedTrayTargetTest(unittest.TestCase):
    def test_stationary_target_survives_noisy_or_missing_rgb_estimates(self):
        initial = estimate(.6)
        tracker = ConfirmedTrayTarget(initial)
        for candidate in (.609, .609, .609, .650, None, .651, .610, .649):
            self.assertEqual(tracker.observe(None if candidate is None else estimate(candidate)), initial)

    def test_confirms_displacement_and_then_refines(self):
        initial = estimate(.6)
        tracker = ConfirmedTrayTarget(initial)
        self.assertEqual(tracker.observe(estimate(.652)), initial)
        self.assertEqual(tracker.observe(estimate(.650)), initial)
        self.assertEqual(tracker.observe(estimate(.654)), estimate(.654))
        self.assertEqual(tracker.observe(estimate(.659)), estimate(.654))
        self.assertEqual(tracker.observe(estimate(.660)), estimate(.654))
        self.assertEqual(tracker.observe(estimate(.661)), estimate(.661))
