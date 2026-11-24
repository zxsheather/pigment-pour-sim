"""Optional contract checks against an installed official SDK; no connection."""
import importlib.util
import inspect
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from deploy.sdk_bridge import SDKBridge, sdk_pose


@unittest.skipUnless(importlib.util.find_spec('x2robot'), 'Install the robot-matched official SDK for contract tests')
class OfficialSDKContractTests(unittest.TestCase):
    def test_actual_messages_and_right_only_dispatch(self):
        from x2robot.geometry_msgs import Point, Pose, PoseStamped, Quaternion
        from x2robot.sdk import ExecutionResult, GripperPosition
        robot = NS(right_arm=Mock(), right_gripper=Mock())
        robot.right_arm.set_end_pose.return_value = ExecutionResult(is_success=True)
        robot.right_gripper.set_position.return_value = ExecutionResult(is_success=True)
        pose = Pose(position=Point(x=.2, y=.1, z=.3), orientation=Quaternion(w=1.))
        matrix = sdk_pose(PoseStamped(pose=pose))
        SDKBridge(robot).write_right(matrix, .8)
        args, kwargs = robot.right_arm.set_end_pose.call_args
        self.assertIsInstance(args[0], Pose)
        self.assertEqual(kwargs, {'timeout': 1.})
        np.testing.assert_allclose(sdk_pose(args[0]), matrix)
        self.assertIsInstance(robot.right_gripper.set_position.call_args.args[0], GripperPosition)
        # Prove these messages can be serialized through the real generated API.
        np.testing.assert_allclose(sdk_pose(Pose.FromString(Pose.SerializeToString(args[0]))), matrix, atol=1e-6)

    def test_official_methods_support_deadlines(self):
        from x2robot import sdk
        methods = [('RightArmControllerStub', 'set_end_pose'), ('RightArmControllerStub', 'get_end_pose'),
                   ('RightGripperControllerStub', 'set_position'), ('RightGripperControllerStub', 'get_position'),
                   ('HeadCameraStub', 'get_rgb_image'), ('RightArmCameraStub', 'get_raw_image'),
                   ('HeadControllerStub', 'get_pose'), ('LiftControllerStub', 'get_lift_position'),
                   ('ChassisControllerStub', 'get_relative_position'), ('ChassisControllerStub', 'get_odometry')]
        for cls, method in methods:
            with self.subTest(cls=cls, method=method):
                self.assertIn('timeout', inspect.signature(getattr(getattr(sdk, cls), method)).parameters)


if __name__ == '__main__':
    unittest.main()
