import math
import unittest

import torch

from simulation.cylinder_wrist_pose_loss import near_approach_world_wrist_loss


class BaseCommandResponseTest(unittest.TestCase):
    def test_matching_actions_have_no_loss_while_base_slows(self):
        response = {
            'previous_motion_affine': [[-.12, -.3], [.12, .3]],
            'step_limit': .015,
        }
        settings = {
            'base_action_affine': [[-.12, -.3], [.12, .3]],
            'base_control_dt_s': .05,
            'cue_affine': [[-3., -3., -3.], [3., 3., 3.]],
            'cue_near_scale_m': .04,
            'radius_m': .18,
            'weight': .2,
            'scale_m': .012,
            'approach_progress': {
                'min_distance_m': .06,
                'min_progress_m': .005,
                'weight': .4,
                'scale_m': .012,
                'base_command_response': response,
                'far': {'max_distance_m': .42, 'min_progress_m': .005, 'weight': .5, 'scale_m': .012},
            },
        }
        state = torch.zeros((1, 26), dtype=torch.float64)
        state[0, 23] = 1.
        state[0, 25] = -.6
        commands = torch.zeros((1, 8, 26), dtype=torch.float64)
        expected = torch.zeros((1, 8, 4, 3), dtype=torch.float64)
        expected[0, :, :, 0] = torch.linspace(.003, .024, 8, dtype=torch.float64)[:, None]
        valid = torch.tensor([[True] * 6 + [False] * 2])

        for gap in (.12, .30):
            state[0, 18] = math.asinh(gap / settings['cue_near_scale_m']) / 3.
            for use_global_response in (False, True):
                with self.subTest(gap=gap, use_global_response=use_global_response):
                    calibration = settings.copy()
                    if use_global_response:
                        calibration['base_command_response'] = response
                    exact = near_approach_world_wrist_loss(
                        expected, expected, commands, commands, state, calibration,
                        valid, None, measured_tip=expected.new_zeros((1, 3)))
                    torch.testing.assert_close(exact, torch.zeros_like(exact), atol=0., rtol=0.)

                    behind = expected.clone()
                    behind[..., 0] -= .006
                    behind.requires_grad_()
                    loss = near_approach_world_wrist_loss(
                        behind, expected, commands, commands, state, calibration,
                        valid, None, measured_tip=expected.new_zeros((1, 3)))
                    self.assertGreater(loss.sum().item(), 0.)
                    torch.testing.assert_close(loss[~valid], torch.zeros_like(loss[~valid]))
                    loss.sum().backward()
                    self.assertTrue(torch.isfinite(behind.grad).all())
                    self.assertGreater(behind.grad[valid].abs().sum().item(), 0.)
                    self.assertEqual(behind.grad[~valid].abs().sum().item(), 0.)
