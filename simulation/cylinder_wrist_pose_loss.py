"""Measure joint prediction error at the right wrist and gripper tip using rigid FK."""
import torch


def wrist_markers(joints, calibration):
    """Transform the gripper tip and three orientation markers into the base frame."""
    axes = joints.new_tensor(calibration['hinge_axes'])
    anchors = joints.new_tensor(calibration['hinge_anchors_m'])
    angles = (joints - joints.new_tensor(calibration['reference_joint_position_rad'])).unbind(-1)
    points = joints.new_tensor(calibration['reference_markers_m']).expand(*joints.shape[:-1], 4, 3)
    # Each hinge rotation is about its axis at the reference pose. Applying
    # distal hinges first composes the serial chain about those fixed axes.
    for angle, axis, anchor in reversed(list(zip(angles, axes, anchors, strict=True))):
        delta = points - anchor
        c, s = angle.cos()[..., None, None], angle.sin()[..., None, None]
        along = (delta * axis).sum(dim=-1, keepdim=True) * axis
        points = anchor + delta * c + torch.linalg.cross(axis.expand_as(delta), delta) * s + along * (1 - c)
    return points


def denormalize_arm(values, affine):
    low, high = values.new_tensor(affine).unbind(0)
    return low + (values[..., :6] + 1) * (high - low) * .5


def wrist_pose_auxiliary_loss(predicted, labels, proprioception, calibration):
    """Return a per-element loss for the eight executed joint targets."""
    if (labels.ndim != 3 or labels.shape[1:] != (8, 26)
            or predicted.shape != (labels.shape[0] * 8, 26)
            or proprioception is None or proprioception.shape != (labels.shape[0], 1, 26)):
        raise ValueError('Wrist supervision requires matching eight-frame joint actions and 26D robot state')
    dtype = torch.float64 if predicted.dtype == torch.float64 else torch.float32
    origin = denormalize_arm(proprioception.to(dtype), calibration['state_affine_rad'])
    target = denormalize_arm(labels.to(dtype), calibration['action_affine_rad']) + origin
    forecast = denormalize_arm(predicted.reshape_as(labels).to(dtype), calibration['action_affine_rad']) + origin
    error = wrist_markers(forecast, calibration) - wrist_markers(target, calibration)
    tip = error[..., 0, :]
    orientation = error[..., 1:, :] - tip[..., None, :]
    squared_error = tip.square().sum(-1) + .5 * orientation.square().sum((-2, -1))
    loss = squared_error * (calibration['weight'] / calibration['position_scale_m'] ** 2)
    # Use one active joint-action slot so Wall's existing masks and reduction
    # include the pose loss once per executed frame.
    return torch.nn.functional.pad(loss.reshape(-1, 1), (0, 25))
