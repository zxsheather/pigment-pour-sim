"""Check that grip handoff requires sustained, opposing resistance."""
import unittest

from grasp_stages import GraspStages


class GraspStageSignals(unittest.TestCase):
    def test_empty_closed_gripper_does_not_trigger(self):
        stage = GraspStages()
        for _ in range(10):
            self.assertFalse(stage.observe(.1, [.2, .2], [-5., 5.], .048))
        self.assertEqual(stage.phase, "grasp")

    def test_open_command_and_one_sided_resistance_do_not_trigger(self):
        for requested, effort in [(4.5, [-5., 5.]), (.1, [-5., 0.])]:
            with self.subTest(requested=requested, effort=effort):
                stage = GraspStages()
                for _ in range(10):
                    self.assertFalse(stage.observe(requested, [2.8, 2.8], effort, .048))
                self.assertEqual(stage.phase, "grasp")

    def test_transient_resistance_is_not_accumulated_across_release(self):
        stage = GraspStages()
        for _ in range(10):
            stage.observe(.1, [2.8, 2.8], [-5., 5.], .048)
            stage.observe(4.5, [2.8, 2.8], [-5., 5.], .048)
        self.assertEqual(stage.phase, "grasp")

    def test_sustained_resistance_holds_model_close_command(self):
        stage = GraspStages()
        self.assertFalse(stage.observe(.1, [2.8, 2.8], [-5., 5.], .048))
        self.assertFalse(stage.observe(.1, [2.8, 2.8], [-5., 5.], .048))
        self.assertTrue(stage.observe(.1, [2.8, 2.8], [-5., 5.], .048))
        self.assertEqual(stage.phase, "lift")
        self.assertAlmostEqual(stage.gripper_command(4.5), .1)


if __name__ == "__main__":
    unittest.main()
