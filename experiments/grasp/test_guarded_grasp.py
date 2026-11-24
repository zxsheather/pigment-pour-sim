"""Regression cases for closing at the fingertips and unnecessary insertion."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from guarded_grasp import GuardConfig, GuardedTrace, bounded_pose, closing_geometry_valid, geometry_ready


def value(x=.132, y=0., z=-.005, angle=0., opening=4.5):
    return {"center_m": [x, y, z], "axis_error_deg": angle, "opening_min": opening}


def environment(center):
    pose = np.array([.132, 0., 0., 0., 0., 0., 4.5])
    return SimpleNamespace(ee=0, bottle=1, arm_base=2,
        d=SimpleNamespace(xmat=np.tile(np.eye(3).reshape(1, 9), (3, 1)),
            xpos=np.array([[0., 0., 0.], center, [0., 0., 0.]])),
        tcp_offset=np.array([.132, 0., 0.]), state_offsets={"right": np.zeros(3)},
        arm_pose=lambda side: pose.copy(), gripper_feedback=lambda: {"opening_motor_units": [4.5, 4.5]})


class GraspGateTests(unittest.TestCase):
    def test_open_alignment_refreshes_axis_after_bottle_settles(self):
        env = environment([.185, 0., -.005])
        trace = GuardedTrace({}, set(), np.array([0., 0., 1.]))
        trace.phase, trace.started_s = "align", 0.
        trace.approach_x = np.array([1., 0., 0.])
        trace.target_rotation = Rotation.from_euler("y", 8., degrees=True).as_matrix()
        raw = env.arm_pose("right"); raw[6] = 2.2
        trace.guarded_command(env, raw, 1, 0)
        np.testing.assert_allclose(trace.target_rotation[:, 2], [0., 0., 1.], atol=1e-12)
        self.assertEqual(trace.phase, "seat")

    def test_already_seated_hand_holds_position_before_closing(self):
        env = environment([.132, 0., -.005])
        trace = GuardedTrace({}, set(), np.array([0., 0., 1.]))
        raw = env.arm_pose("right"); raw[6] = 2.2
        for step in range(3):
            command = trace.guarded_command(env, raw, step, 0)
            np.testing.assert_array_equal(command[:6], env.arm_pose("right")[:6])
            self.assertEqual(command[6], 4.5)
        self.assertEqual(trace.phase, "close")
        command = trace.guarded_command(env, raw, 3, 0)
        self.assertEqual(command[6], 2.2)
        self.assertEqual([event["to"] for event in trace.events], ["ready", "close"])

    def test_tip_case_opens_and_retreats_instead_of_closing(self):
        env = environment([.163, .0004, -.024])
        trace = GuardedTrace({}, set(), np.array([0., 0., 1.]))
        raw = env.arm_pose("right"); raw[6] = 2.2
        command = trace.guarded_command(env, raw, 0, 0)
        self.assertEqual(trace.phase, "retreat")
        self.assertEqual(command[6], 4.5)
        self.assertLess(command[0], env.arm_pose("right")[0])
        self.assertFalse(trace.handoff_allowed(env))

    def test_original_tip_grasp_cannot_pass_preclosure_gate(self):
        self.assertFalse(geometry_ready(value(.16299, .0004, -.02423), GuardConfig()))

    def test_deep_alignment_does_not_require_more_insertion(self):
        for x in [.124, .132, .141]:
            self.assertTrue(geometry_ready(value(x), GuardConfig()))

    def test_each_alignment_channel_independently_blocks_closure(self):
        for case in [value(y=.012), value(z=-.025), value(angle=15), value(opening=3.0), value(x=.100)]:
            self.assertFalse(geometry_ready(case, GuardConfig()))

    def test_settling_tolerance_still_rejects_tip_grip(self):
        self.assertFalse(closing_geometry_valid(value(.163)))
        self.assertTrue(closing_geometry_valid(value(.132, z=-.018, opening=3.3)))

    def test_pose_corrections_have_physical_rate_limits(self):
        cfg = GuardConfig()
        actual = np.r_[np.zeros(6), 4.5]
        target = np.r_[[.1, -.2, .1], [.2, .1, -.3], 4.5]
        result = bounded_pose(actual, target, cfg)
        self.assertAlmostEqual(np.linalg.norm(result[:3]), cfg.max_translation_step_m)
        self.assertAlmostEqual(Rotation.from_euler("xyz", result[3:6]).magnitude(), cfg.max_rotation_step_rad)
        self.assertEqual(result[6], 4.5)


if __name__ == "__main__":
    unittest.main()
