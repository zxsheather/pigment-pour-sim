"""Conservative one-target commissioning, not a collision/force controller."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.geometry import (finite, from_model_pose, grip_from_sdk, grip_to_sdk,
                             model_from_sdk, pose_distance, sdk_from_model, to_model_pose, transform)


def validate_config(config):
    if config.get('calibration_reviewed') is not True:
        raise ValueError('Calibration must be measured and reviewed before model preview/step')
    for side in ['left', 'right']:
        arm = config[side]
        transform(arm['T_model_reference_from_sdk_reference'])
        transform(arm['T_sdk_tool_from_model_tcp'])
        grip_from_sdk(0, arm)
    right = config['right']
    lo = finite(right['workspace_model_xyz_min'], (3,))
    hi = finite(right['workspace_model_xyz_max'], (3,))
    if np.any(lo >= hi):
        raise ValueError('Invalid measured free workspace')
    g = finite(right['sdk_gripper_limits'], (2,))
    if g[0] >= g[1]:
        raise ValueError('Invalid SDK gripper range')
    lift_scale = finite([config['lift_model_per_sdk'], config['lift_model_offset']], (2,))[0]
    if abs(lift_scale) < 1e-8:
        raise ValueError('Invalid lift unit mapping')
    if np.any(abs(finite(config['head_model_per_sdk'], (2,))) < 1e-8):
        raise ValueError('Invalid head angle unit/sign mapping')
    finite(config['head_model_offset'], (2,))
    posture = config['stationary_model_posture']
    finite([posture['lift_m']], (1,))
    finite(posture['head_pitch_yaw_rad'], (2,))
    for name in ['lift_tolerance_m', 'head_tolerance_rad']:
        if not np.isfinite(posture[name]) or posture[name] <= 0:
            raise ValueError('Invalid posture tolerance')
    for name, value in config['limits'].items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid limit: {name}')
    if config['limits']['max_translation_step_m'] > .01 or config['limits']['max_rotation_step_rad'] > .05:
        raise ValueError('Commissioning bridge caps individual steps at 1 cm and 0.05 rad')


def lift_model(raw, config):
    return raw['lift_sdk'] * config['lift_model_per_sdk'] + config['lift_model_offset']


def head_model(raw, config):
    return finite(raw['head_pitch_yaw'], (2,)) * config['head_model_per_sdk'] + config['head_model_offset']


def stationary(raw, config):
    limits, posture = config['limits'], config['stationary_model_posture']
    if raw['read_finished_monotonic'] - raw['read_started_monotonic'] > limits['max_read_seconds']:
        raise ValueError('SDK read took too long')
    velocity = finite(raw['chassis_velocity'], (3,))
    if np.linalg.norm(velocity[:2]) > limits['max_chassis_speed_m_s'] or abs(velocity[2]) > limits['max_chassis_yaw_speed_rad_s']:
        raise ValueError('Chassis is moving')
    if abs(lift_model(raw, config) - posture['lift_m']) > posture['lift_tolerance_m']:
        raise ValueError('Lift differs from calibrated fixed posture')
    if np.any(abs(head_model(raw, config) - posture['head_pitch_yaw_rad']) > posture['head_tolerance_rad']):
        raise ValueError('Head differs from calibrated fixed posture')


def model_observation(raw, config, instruction):
    validate_config(config)
    stationary(raw, config)
    state = {}
    for side, name in [('left', 'follow1_pos'), ('right', 'follow2_pos')]:
        arm = config[side]
        state[name] = to_model_pose(model_from_sdk(raw[side]['T_sdk_reference_from_sdk_tool'], arm),
                                    grip_from_sdk(raw[side]['gripper_sdk'], arm)).astype(np.float32)
    state.update(lift=np.array([lift_model(raw, config)], dtype=np.float32),
                 head_pos=np.asarray(head_model(raw, config), dtype=np.float32),
                 velocity_decomposed=np.asarray(raw['chassis_velocity'], dtype=np.float32),
                 velocity_decomposed_odom=np.asarray(raw['chassis_velocity'], dtype=np.float32))
    return {'state': state, 'views': {k: v[None] for k, v in raw['views'].items()}, 'instruction': instruction}


def check_unchanged(original, fresh, config, now=None):
    now = time.monotonic() if now is None else now
    limits = config['limits']
    for raw in [original, fresh]:
        stationary(raw, config)
    if not 0 <= now - original['read_started_monotonic'] <= limits['max_inference_age_seconds']:
        raise ValueError('Prediction is too old')
    if not 0 <= now - fresh['read_started_monotonic'] <= limits['max_read_seconds'] + limits['max_dispatch_age_seconds']:
        raise ValueError('Fresh SDK feedback is too old')
    for side in ['left', 'right']:
        distance, angle = pose_distance(original[side]['T_sdk_reference_from_sdk_tool'], fresh[side]['T_sdk_reference_from_sdk_tool'])
        if distance > limits['max_arm_drift_m'] or angle > limits['max_arm_drift_rad']:
            raise ValueError(f'{side} arm moved during inference')
        delta = abs(grip_from_sdk(original[side]['gripper_sdk'], config[side]) - grip_from_sdk(fresh[side]['gripper_sdk'], config[side]))
        if delta > limits['max_gripper_drift_model']:
            raise ValueError(f'{side} gripper moved during inference')
    chassis_delta = finite(fresh['chassis_xy_yaw'], (3,)) - finite(original['chassis_xy_yaw'], (3,))
    yaw_delta = abs(np.arctan2(np.sin(chassis_delta[2]), np.cos(chassis_delta[2])))
    if np.linalg.norm(chassis_delta[:2]) > limits['max_chassis_drift_m'] or yaw_delta > limits['max_chassis_drift_rad']:
        raise ValueError('Chassis changed position during inference')
    if abs(lift_model(original, config) - lift_model(fresh, config)) > config['stationary_model_posture']['lift_tolerance_m']:
        raise ValueError('Lift moved during inference')
    if np.any(abs(head_model(fresh, config) - head_model(original, config)) > config['stationary_model_posture']['head_tolerance_rad']):
        raise ValueError('Head moved during inference')


def proposed_step(action, raw, config, allow_gripper=False):
    """Clip toward first predicted target. Reject positions outside calibrated box.

    Bounds apply to the model TCP and the whole predicted first target, not just
    the clipped position. This is not a swept-volume or IK collision check.
    """
    validate_config(config)
    arm, limits = config['right'], config['limits']
    action = finite(action, (7,))
    current = model_from_sdk(raw['right']['T_sdk_reference_from_sdk_tool'], arm)
    target = from_model_pose(action)
    lo, hi = np.asarray(arm['workspace_model_xyz_min']), np.asarray(arm['workspace_model_xyz_max'])
    for matrix in [current, target]:
        if np.any(matrix[:3, 3] < lo) or np.any(matrix[:3, 3] > hi):
            raise ValueError('Current or predicted TCP is outside measured free workspace')
    delta = target[:3, 3] - current[:3, 3]
    delta *= min(1., limits['max_translation_step_m'] / max(np.linalg.norm(delta), 1e-12))
    rotvec = Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).as_rotvec()
    rotvec *= min(1., limits['max_rotation_step_rad'] / max(np.linalg.norm(rotvec), 1e-12))
    step = current.copy()
    step[:3, 3] += delta
    step[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
    gripper_sdk = None
    if allow_gripper:
        current_grip = grip_from_sdk(raw['right']['gripper_sdk'], arm)
        model_grip = current_grip + np.clip(action[6] - current_grip, -limits['max_gripper_step_model'], limits['max_gripper_step_model'])
        gripper_sdk = grip_to_sdk(model_grip, arm)
        if not arm['sdk_gripper_limits'][0] <= gripper_sdk <= arm['sdk_gripper_limits'][1]:
            raise ValueError('Proposed gripper command outside calibrated SDK limits')
    return sdk_from_model(step, arm), gripper_sdk, {
        'predicted_model_pose': action.tolist(),
        'bounded_model_pose': to_model_pose(step, grip_from_sdk(raw['right']['gripper_sdk'], arm) if gripper_sdk is None else grip_from_sdk(gripper_sdk, arm)).tolist(),
        'translation_step_m': float(np.linalg.norm(delta)), 'rotation_step_rad': float(np.linalg.norm(rotvec)),
        'gripper_write_enabled': allow_gripper}
