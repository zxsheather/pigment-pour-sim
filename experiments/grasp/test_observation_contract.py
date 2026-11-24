"""Integration checks for the SDK observation contract and Cartesian executor.

The official Quanta X1 client reports head_pos as [pitch, yaw] and arm poses
as [x, y, z, roll, pitch, yaw, gripper]. These checks use measured simulator
state; they do not assume that passing establishes real-robot calibration.
"""
from pathlib import Path
import argparse
import tempfile
import unittest

import numpy as np

from grasp_walloss import GraspEnvironment
from grasp_stages import GraspStages


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=ROOT/"generated/scene.xml")
parser.add_argument("--config", type=Path, default=ROOT.parents[1]/"simulation/scene_config.json")
args, extra = parser.parse_known_args()


class ObservationContract(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.env = GraspEnvironment(args.scene, args.config, Path(self.folder.name), "blue", tcp="grasp")

    def tearDown(self):
        self.env.renderer.close()
        self.folder.cleanup()

    def test_head_state_is_measured_pitch_then_yaw(self):
        env = self.env
        state, _ = env.observation()
        expected = [float(env.d.qpos[env.m.joint(name).qposadr[0]])
                    for name in ["head_pitch_joint", "head_yaw_joint"]]
        np.testing.assert_allclose(state["head_pos"], expected, atol=1e-5,
                                   err_msg="SDK head_pos must be [pitch, yaw]")

    def test_small_cartesian_commands_reach_requested_positions(self):
        env = self.env
        for axis in range(3):
            with self.subTest(axis=axis):
                command = env.arm_pose("right").copy()
                command[axis] += .02
                for _ in range(20):
                    env.execute(command)
                np.testing.assert_allclose(env.arm_pose("right")[:3], command[:3], atol=.005)

    def test_open_close_commands_change_measured_opening(self):
        env = self.env
        command = env.arm_pose("right").copy()
        openings = []
        for gripper in [4.5, 0.5, 4.5]:
            command[6] = gripper
            for _ in range(20):
                env.execute(command)
            openings.append(float(env.arm_pose("right")[6]))
        self.assertLess(openings[1], openings[0] - 2.)
        self.assertGreater(openings[2], openings[1] + 2.)

    def test_free_closure_does_not_trigger_grasp_handoff(self):
        env = self.env
        stage = GraspStages()
        command = env.arm_pose("right").copy()
        command[6] = .1
        dt = round(env.step_time/env.m.opt.timestep)*env.m.opt.timestep
        for _ in range(20):
            feedback = env.execute(command)["gripper_feedback"]
            self.assertFalse(stage.observe(command[6], feedback["opening_motor_units"], feedback["effort_n"], dt))
        self.assertEqual(stage.phase, "grasp")


if __name__ == "__main__":
    unittest.main(argv=[__file__, *extra])
