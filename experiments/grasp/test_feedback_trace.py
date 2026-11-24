"""Regressions for interpreting height/contact interruptions, without MuJoCo."""
import unittest

from feedback_trace import summarize_trace


class FeedbackTraceTest(unittest.TestCase):
    def trace(self, contacts, heights, dt=.02):
        return [{"time_s": (i + 1) * dt, "contacts": contact, "lift_m": height}
                for i, (contact, height) in enumerate(zip(contacts, heights))]

    def test_height_dip_does_not_mean_fingers_lost_contact(self):
        result = summarize_trace(self.trace([2] * 8, [.06] * 5 + [.049] + [.06] * 2), .02)
        self.assertAlmostEqual(result["max_bilateral_contact_s"], .16)
        self.assertAlmostEqual(result["max_bilateral_lifted_s"], .10)
        self.assertEqual(result["first_sustained_hold_break"]["reasons"], ["height_below_5cm"])

    def test_contact_loss_is_distinct_from_height_loss(self):
        result = summarize_trace(self.trace([2] * 5 + [1, 0], [.06] * 7), .02)
        self.assertEqual(result["first_sustained_hold_break"]["reasons"], ["finger_contact_lost"])
        self.assertAlmostEqual(result["max_above_5cm_s"], .14)

    def test_terminal_hold_is_not_reported_as_a_drop(self):
        result = summarize_trace(self.trace([0] + [2] * 5, [.0] + [.06] * 5), .02)
        self.assertIsNone(result["first_sustained_hold_break"])
        self.assertEqual(result["held_breaks"], [])


if __name__ == "__main__":
    unittest.main()
