"""Calibrate and verify wrist supervision against the fixed official MuJoCo robot."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from audit_gripper import audit
from carry_env import CONTROL_DT
from cylinder_joint_actions import ARM_SERVO_RATE_RAD_S, joint_action_to_targets, servo_increment_targets
from cylinder_wrist_pose_loss import (servo_arm_targets, servo_increment_wrist_pose_loss,
                                     wrist_markers, wrist_pose_auxiliary_loss)
from inspect_cylinder import CylinderEnv


def official_markers(env, data, orientation_offset):
    rotation = data.xmat[env.gripper_body].reshape(3, 3)
    tip = data.xpos[env.gripper_body] + rotation @ env.grasp_offset
    return np.vstack((tip, tip + orientation_offset * rotation.T))


def calibrate(env, norms, weight):
    data = mujoco.MjData(env.model)
    mujoco.mj_kinematics(env.model, data)
    base_rotation = data.xmat[env.robot_root].reshape(3, 3)
    base_position = data.xpos[env.robot_root]
    joint_ids = [env.model.joint(f'right_arm_joint{i}').id for i in range(1, 7)]
    if not np.all(env.model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('Expected six hinges on the official right-arm chain')
    orientation_offset = .06
    affine = lambda name: [norms[name][key][:6] for key in ('q01', 'q99')]
    calibration = dict(hinge_axes=(data.xaxis[joint_ids] @ base_rotation).tolist(),
                       hinge_anchors_m=((data.xanchor[joint_ids] - base_position) @ base_rotation).tolist(),
                       reference_joint_position_rad=data.qpos[env.arm_addresses].tolist(),
                       reference_markers_m=((official_markers(env, data, orientation_offset) - base_position)
                                            @ base_rotation).tolist(),
                       orientation_marker_offset_m=orientation_offset, position_scale_m=.10,
                       action_affine_rad=affine('action'), state_affine_rad=affine('observation.state'), weight=weight)
    return calibration, joint_ids


def verify(env, calibration, joint_ids):
    samples = np.random.default_rng(471).uniform(*env.model.jnt_range[joint_ids].T, size=(64, 6))
    data = mujoco.MjData(env.model)
    references = []
    for joints in samples:
        data.qpos[env.arm_addresses] = joints
        mujoco.mj_kinematics(env.model, data)
        base_rotation = data.xmat[env.robot_root].reshape(3, 3)
        reference = official_markers(env, data, calibration['orientation_marker_offset_m'])
        references.append((reference - data.xpos[env.robot_root]) @ base_rotation)
    position = wrist_markers(torch.from_numpy(samples), calibration).numpy()
    np.testing.assert_allclose(position, references, rtol=0., atol=1e-9)

    labels = torch.zeros((1, 8, 26), dtype=torch.float64)
    robot = torch.zeros((1, 1, 26), dtype=torch.float64)
    equal = wrist_pose_auxiliary_loss(labels.reshape(8, 26), labels, robot, calibration)
    if torch.count_nonzero(equal):
        raise ValueError('Wrist supervision must vanish at the demonstrated action')
    trial = labels.reshape(8, 26).clone()
    trial[0, 2] = .04
    trial.requires_grad_()
    loss = wrist_pose_auxiliary_loss(trial, labels, robot, calibration)
    loss.sum().backward()
    if (loss[0, 0] <= 0 or not torch.isfinite(trial.grad).all() or trial.grad[0, 2] <= 0
            or torch.count_nonzero(trial.grad[1:]) or torch.count_nonzero(trial.grad[:, 6:])):
        raise ValueError('Unexpected gradient from wrist supervision')
    return dict(verified_poses=len(samples), max_marker_error_m=float(np.abs(position - references).max()),
                verified_zero_loss_at_target=True, nonzero_arm_gradient=True, no_gripper_or_base_gradient=True)


def verify_servo_composition(env, calibration):
    rng = np.random.default_rng(417)
    low, high = np.asarray(calibration['joint_limits_rad'])
    anchor = rng.uniform(low, high, size=(16, 6))
    encoded = rng.uniform(-1.4, 1.4, size=(16, 8, 26))
    targets = servo_arm_targets(torch.from_numpy(encoded), torch.from_numpy(anchor), calibration).numpy()
    action_low, action_high = np.asarray(calibration['action_affine_rad'])
    for origin, chunk, target in zip(anchor, encoded, targets, strict=True):
        previous = np.r_[origin, 0., 0., 0.]
        for action, position in zip(chunk, target, strict=True):
            increments = np.r_[action_low + (action[:6] + 1) * (action_high - action_low) * .5, 0., 0., 0.]
            decoded = servo_increment_targets(increments, previous, CONTROL_DT)
            joints, *_ = joint_action_to_targets(env, decoded)
            np.testing.assert_allclose(position, joints, rtol=0., atol=1e-12)
            previous[:6] = joints

    labels = torch.zeros((1, 8, 26), dtype=torch.float64)
    proprio = torch.zeros((1, 1, 26), dtype=torch.float64)
    if torch.count_nonzero(servo_increment_wrist_pose_loss(labels.reshape(8, 26), labels, proprio, calibration)):
        raise ValueError('Servo wrist supervision must vanish at the demonstrated action')
    trial = labels.reshape(8, 26).clone()
    trial[0, 1] = .16
    trial[5:, :6] = .71
    trial.requires_grad_()
    mask = torch.tensor([[True] * 5 + [False] * 3])
    loss = servo_increment_wrist_pose_loss(trial, labels, proprio, calibration, flow_loss_mask=mask)
    loss.sum().backward()
    if (not (loss[:5, 0] > 0).all() or torch.count_nonzero(loss[5:])
            or not torch.isfinite(trial.grad).all() or trial.grad[0, 1] <= 0
            or torch.count_nonzero(trial.grad[5:]) or torch.count_nonzero(trial.grad[:, 6:])):
        raise ValueError('Unexpected servo wrist gradient or participation of padded frames')
    return dict(verified_servo_targets=int(targets.shape[0] * targets.shape[1]),
                uses_joint_limits_and_increment_rate=True, masked_suffix_zero=True,
                servo_zero_loss_at_target=True, nonzero_prefix_gradient=True,
                no_gripper_or_base_gradient=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-config', type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight', type=float, default=.5)
    parser.add_argument('--servo-increments', action='store_true',
                        help='Calibrate composed incremental actions with the wrist-target state layout.')
    args = parser.parse_args()
    if not np.isfinite(args.weight) or args.weight <= 0:
        parser.error('The wrist loss weight must be positive and finite')
    controller = json.loads(args.controller_config.read_text())
    stats = json.loads(args.norm_stats.read_text())['norm_stats']
    env = CylinderEnv(**controller['environment'])
    try:
        geometry = audit(env)
        calibration, ids = calibrate(env, stats, args.weight)
        verified = verify(env, calibration, ids)
        if args.servo_increments:
            low, high = env.model.jnt_range[ids].T
            limit = ARM_SERVO_RATE_RAD_S * CONTROL_DT
            np.testing.assert_allclose(calibration['action_affine_rad'], [[-limit]*6, [limit]*6])
            calibration.update(joint_limits_rad=[(low+.005).tolist(), (high-.005).tolist()],
                               arm_increment_limit_rad=limit, position_scale_m=.04,
                               tracking_affine_rad=[stats['observation.state'][key][19:21]
                                                    for key in ('q01', 'q99')])
            verified.update(verify_servo_composition(env, calibration))
        report = dict(kind='official_wrist_pose_loss_calibration', config=calibration, verification=verified,
                      normalization_sha256=hashlib.sha256(args.norm_stats.read_bytes()).hexdigest(),
                      geometry_audit_passed=geometry['passed'], geometry_urdf_sha256=geometry['urdf_sha256'],
                      servo_increments=args.servo_increments)
        with args.output.open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
        print(json.dumps(verified), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
