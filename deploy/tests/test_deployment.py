import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.artifacts import ROOT, extract_overlay
from deploy.geometry import (from_xyz_quat, grip_from_sdk, grip_to_sdk,
                             model_from_sdk, sdk_from_model, transform)
from deploy.guards import check_unchanged, model_observation, proposed_step, validate_config
from deploy.robot import commission
from deploy.sdk_bridge import SDKBridge, image_rgb, sdk_pose


def configuration():
    c = json.loads((ROOT / 'deploy/robot.example.json').read_text())
    c['calibration_reviewed'] = True
    for side in ['left', 'right']:
        c[side].update(T_model_reference_from_sdk_reference=np.eye(4).tolist(),
                       T_sdk_tool_from_model_tcp=np.eye(4).tolist(),
                       gripper_model_per_sdk=2., gripper_model_offset=.25)
    c['right'].update(sdk_gripper_limits=[0., 4.5], workspace_model_xyz_min=[0., -.5, 0.],
                      workspace_model_xyz_max=[.7, .5, .8])
    c.update(lift_model_per_sdk=.001, lift_model_offset=0.,
             head_model_per_sdk=[np.pi / 180, -np.pi / 180], head_model_offset=[0., 0.])
    c['stationary_model_posture'].update(lift_m=.3, head_pitch_yaw_rad=[.55, 0.])
    return c


def feedback():
    now = time.monotonic()
    t = np.eye(4)
    t[:3, 3] = [.282, 0, .35]
    return {'left': {'T_sdk_reference_from_sdk_tool': t.copy(), 'gripper_sdk': 1.},
            'right': {'T_sdk_reference_from_sdk_tool': t.copy(), 'gripper_sdk': 1.},
            'head_pitch_yaw': np.array([.55 * 180 / np.pi, 0]), 'lift_sdk': 300.,
            'chassis_xy_yaw': np.zeros(3), 'chassis_velocity': np.zeros(3),
            'views': {'face_view': np.zeros((8, 8, 3), dtype=np.uint8),
                      'right_wrist_view': np.zeros((8, 8, 3), dtype=np.uint8)},
            'read_started_monotonic': now, 'read_finished_monotonic': now}


class GeometryTests(unittest.TestCase):
    def test_frame_and_nonzero_tcp_round_trip(self):
        arm = configuration()['right']
        a, b, pose = np.eye(4), np.eye(4), np.eye(4)
        a[:3, :3] = Rotation.from_euler('xyz', [.4, -.3, .2]).as_matrix()
        a[:3, 3] = [.1, -.4, .8]
        b[:3, :3] = Rotation.from_euler('xyz', [.1, .5, -.2]).as_matrix()
        b[:3, 3] = [.132, .01, -.03]
        pose[:3, :3] = Rotation.from_euler('xyz', [.2, -.1, .7]).as_matrix()
        pose[:3, 3] = [.2, .1, .3]
        arm['T_model_reference_from_sdk_reference'] = a
        arm['T_sdk_tool_from_model_tcp'] = b
        np.testing.assert_allclose(sdk_from_model(model_from_sdk(pose, arm), arm), pose, atol=1e-12)
        self.assertAlmostEqual(grip_to_sdk(grip_from_sdk(.8, arm), arm), .8)

    def test_reject_corrupt_rotation_and_quaternion(self):
        for q in [[0, 0, 0, 0], [np.nan, 0, 0, 1], [0, 0, 0, 2]]:
            with self.assertRaises(ValueError):
                from_xyz_quat([0, 0, 0] + q)
        reflection = np.eye(4)
        reflection[0, 0] = -1
        with self.assertRaises(ValueError):
            transform(reflection)

    def test_units_and_gripper_mapping_are_used(self):
        obs = model_observation(feedback(), configuration(), 'pick up blue bottle')
        np.testing.assert_allclose(obs['state']['head_pos'], [.55, 0], atol=1e-6)
        np.testing.assert_allclose(obs['state']['lift'], [.3], atol=1e-6)
        self.assertAlmostEqual(float(obs['state']['follow2_pos'][6]), 2.25)


class GuardTests(unittest.TestCase):
    def test_unfilled_profile_blocks_prediction(self):
        c = json.loads((ROOT / 'deploy/robot.example.json').read_text())
        with self.assertRaises(ValueError):
            validate_config(c)
        c['calibration_reviewed'] = True
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_limits_and_gripper_opt_in(self):
        raw, config = feedback(), configuration()
        action = [.4, .1, .4, 0., 0., .3, 4.]
        _, gripper, info = proposed_step(action, raw, config)
        self.assertIsNone(gripper)
        self.assertAlmostEqual(info['translation_step_m'], .002)
        self.assertAlmostEqual(info['rotation_step_rad'], np.pi / 180)
        _, gripper, _ = proposed_step(action, raw, config, allow_gripper=True)
        self.assertAlmostEqual(gripper, 1.025)

    def test_outside_workspace_nan_and_gripper_range(self):
        raw, config = feedback(), configuration()
        for action in [[2, 0, .3, 0, 0, 0, 2], [np.nan, 0, .3, 0, 0, 0, 2]]:
            with self.assertRaises(ValueError):
                proposed_step(action, raw, config)
        config['right']['sdk_gripper_limits'] = [0, .5]
        with self.assertRaises(ValueError):
            proposed_step([.3, 0, .3, 0, 0, 0, 4], raw, config, True)

    def test_stale_feedback_or_motion_blocks_writes(self):
        original, fresh, config = feedback(), feedback(), configuration()
        check_unchanged(original, fresh, config)
        for mutation in ['arm', 'base', 'speed', 'old', 'head', 'gripper', 'slow_read']:
            a, b = copy.deepcopy(original), copy.deepcopy(fresh)
            if mutation == 'arm': b['left']['T_sdk_reference_from_sdk_tool'][0, 3] += .01
            if mutation == 'base': b['chassis_xy_yaw'][0] += .01
            if mutation == 'speed': b['chassis_velocity'][0] = .1
            if mutation == 'old': a['read_started_monotonic'] -= 40
            if mutation == 'head': b['head_pitch_yaw'][0] += 10
            if mutation == 'gripper': b['right']['gripper_sdk'] += 1
            if mutation == 'slow_read': b['read_started_monotonic'] -= 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                check_unchanged(a, b, config)

    def test_preview_has_zero_writes_step_has_one_and_drift_blocks(self):
        action = np.tile([.3, 0, .35, 0, 0, 0, 2.5], (32, 1)).tolist()
        with patch('deploy.infer.predict', return_value={'follow2_pos': action}):
            for execute in [False, True]:
                bridge = Mock()
                bridge.read.side_effect = [feedback(), feedback()]
                _, result = commission(bridge, None, configuration(), 'pick up blue bottle', execute)
                self.assertEqual(bridge.write_right.call_count, int(execute))
                self.assertEqual(result['commands_sent'], int(execute))
            bridge = Mock()
            bad = feedback()
            bad['chassis_velocity'][0] = .2
            bridge.read.side_effect = [feedback(), bad]
            with self.assertRaises(ValueError):
                commission(bridge, None, configuration(), 'pick up blue bottle', True)
            bridge.write_right.assert_not_called()


class SDKTests(unittest.TestCase):
    def test_read_adapter_only_calls_getters_and_rgb_decodes(self):
        arm = NS(get_end_pose=Mock(return_value=[.2, 0, .3, 0, 0, 0, 1]))
        gripper = NS(get_position=Mock(return_value=NS(position=1.)))
        image = ImageFixture()
        robot = NS(left_arm=arm, right_arm=arm, left_gripper=gripper, right_gripper=gripper,
                   head=NS(get_pose=Mock(return_value=NS(pitch=.55, yaw=0))),
                   lift=NS(get_lift_position=Mock(return_value=NS(position=.3))),
                   chassis=NS(get_relative_position=Mock(return_value=NS(x=0, y=0, yaw=0)),
                              get_odometry=Mock(return_value=NS(twist=NS(twist=NS(linear=NS(x=0, y=0), angular=NS(z=0)))))),
                   head_camera=NS(get_rgb_image=Mock(return_value=image.message)),
                   right_arm_camera=NS(get_raw_image=Mock(return_value=image.message)))
        raw = SDKBridge(robot).read()
        np.testing.assert_array_equal(raw['views']['face_view'][0, 0], [255, 0, 0])
        arm.get_end_pose.assert_called_with(timeout=1.)

    def test_pose_stamped_and_bgr(self):
        pose = NS(pose=NS(position=NS(x=.1, y=.2, z=.3), orientation=NS(x=0, y=0, z=0, w=1)))
        np.testing.assert_allclose(sdk_pose(pose)[:3, 3], [.1, .2, .3])
        bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(image_rgb(bgr, 'BGR'), [[[255, 0, 0]]])

    def test_right_only_writes_check_execution_result(self):
        arm, gripper = Mock(), Mock()
        arm.set_end_pose.return_value = NS(is_success=True)
        gripper.set_position.return_value = NS(is_success=True)
        robot = NS(right_arm=arm, right_gripper=gripper)
        modules = {'x2robot.geometry_msgs': NS(Point=NS, Pose=NS, Quaternion=NS),
                   'x2robot.sdk': NS(GripperPosition=NS)}
        with patch.dict('sys.modules', modules):
            SDKBridge(robot).write_right(np.eye(4))
            self.assertEqual(arm.set_end_pose.call_count, 1)
            gripper.set_position.assert_not_called()
            arm.set_end_pose.return_value = NS(is_success=False)
            with self.assertRaises(RuntimeError):
                SDKBridge(robot).write_right(np.eye(4), 1.)
            gripper.set_position.assert_not_called()


class ImageFixture:
    def __init__(self):
        from PIL import Image
        image = Image.new('RGB', (8, 8), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        self.message = NS(data=buffer.getvalue())


class ArtifactTests(unittest.TestCase):
    def test_archive_cannot_escape_or_link(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for name, kind in [('../escape', tarfile.REGTYPE), ('allowed', tarfile.SYMTYPE)]:
                archive = temp / 'bad.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.linkname = '../escape'
                    tar.addfile(info)
                with self.assertRaises(ValueError):
                    extract_overlay(archive, temp / 'out', {'files': {'allowed': {'bytes': 0}}})
            self.assertFalse((temp / 'escape').exists())

    def test_valid_archive_checks_bytes_and_hash(self):
        payload = b'test checkpoint metadata'
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            archive = temp / 'good.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('config.json')
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            manifest = {'files': {'config.json': {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}
            extract_overlay(archive, temp / 'out', manifest)
            self.assertEqual((temp / 'out/config.json').read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
