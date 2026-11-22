import unittest

import numpy as np

from audit_cylinder_grasp_approach import verify_progress_prompts
from carry_env import CONTROL_DT
from cylinder_joint_actions import APERTURE_SCALE_M, JOINT_SERVO_STATE_WIDTH
from cylinder_progress_prompts import PROGRESS_PROMPTS, PROGRESS_PROTOCOL


class CartesianProgressAuditTest(unittest.TestCase):
    def test_query_on_rejected_tick_still_checks_current_prompt(self):
        initial_base = np.zeros(3)
        initial_command = np.zeros(9)
        progress_times = CONTROL_DT * np.arange(5)
        states = np.zeros((5, JOINT_SERVO_STATE_WIDTH), dtype=np.float32)
        states[:, 6] = .020 / APERTURE_SCALE_M
        states[:, 24] = -.003 / APERTURE_SCALE_M
        trajectory = dict(timestamp=progress_times[:-1],
                          progress_timestamp=progress_times,
                          progress_robot_state=states)
        episode = dict(action_space='cartesian', progress_protocol=PROGRESS_PROTOCOL,
                       progress_initial_joint_command=initial_command.tolist(),
                       prompt_changes=[dict(time_s=0., task_index=0),
                                       dict(time_s=float(progress_times[-1]), task_index=1)])
        observations = dict(timestamp=progress_times[[0, -1]],
                            instruction=np.asarray(PROGRESS_PROMPTS)[[0, 1]])

        result = verify_progress_prompts(initial_base, initial_command, episode, observations, trajectory)
        np.testing.assert_array_equal(result, [0, 1])

        observations['instruction'][1] = PROGRESS_PROMPTS[0]
        with self.assertRaises(AssertionError):
            verify_progress_prompts(initial_base, initial_command, episode, observations, trajectory)
