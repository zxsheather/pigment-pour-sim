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


def wrist_markers_with_base_motion(joints, base_velocity, calibration):
    """Place joint targets in the starting base frame after each commanded twist."""
    markers = wrist_markers(joints, calibration)
    forward, yaw_rate = base_velocity.unbind(-1)
    step = calibration['base_control_dt_s']
    turn = yaw_rate * step
    heading = turn.cumsum(dim=-1)
    mid_heading = heading - turn / 2
    distance = forward * step * torch.sinc(turn / (2 * torch.pi))
    tx = (distance * mid_heading.cos()).cumsum(dim=-1)
    ty = (distance * mid_heading.sin()).cumsum(dim=-1)
    x, y, z = markers.unbind(-1)
    cosine, sine = heading.cos()[..., None], heading.sin()[..., None]
    return torch.stack((cosine * x - sine * y + tx[..., None],
                        sine * x + cosine * y + ty[..., None], z), dim=-1)


def wrist_pose_with_base_motion_loss(predicted, labels, proprioception, calibration,
                                    dof_mask=None, flow_loss_mask=None):
    """Supervise wrist pose after nominal base motion over the executed prefix."""
    if (labels.ndim != 3 or labels.shape[1:] != (8, 26)
            or predicted.shape != (labels.shape[0] * 8, 26)
            or proprioception is None or proprioception.shape != (labels.shape[0], 1, 26)):
        raise ValueError('Coupled wrist supervision requires eight-frame joint actions and 26D robot state')
    dtype = torch.float64 if predicted.dtype == torch.float64 else torch.float32
    reference = labels.to(dtype)
    forecast = predicted.reshape_as(labels).to(dtype)
    origin = denormalize_arm(proprioception.to(dtype), calibration['state_affine_rad'])
    low, high = reference.new_tensor(calibration['base_action_affine']).unbind(0)
    demonstrated = wrist_markers_with_base_motion(
        denormalize_arm(reference, calibration['action_affine_rad']) + origin,
        low + (reference[..., 7:9] + 1) * (high - low) * .5, calibration)
    planned = wrist_markers_with_base_motion(
        denormalize_arm(forecast, calibration['action_affine_rad']) + origin,
        low + (forecast[..., 7:9] + 1) * (high - low) * .5, calibration)
    error = planned - demonstrated
    tip = error[..., 0, :]
    orientation = error[..., 1:, :] - tip[..., None, :]
    squared_error = tip.square().sum(-1) + .5 * orientation.square().sum((-2, -1))
    narrow = reference[..., 6] <= calibration['narrow_label_max']
    emphasis = 1 + narrow * (calibration['narrow_weight_multiplier'] - 1)
    valid = torch.ones_like(narrow)
    if dof_mask is not None:
        valid &= dof_mask.reshape_as(labels)[..., :9].bool().all(dim=-1)
    if flow_loss_mask is not None:
        valid &= flow_loss_mask.reshape_as(narrow).bool()
    # A later pose depends on the entire preceding base twist sequence.
    valid_prefix = valid.cumprod(dim=1).bool()
    loss = (squared_error * emphasis * (calibration['weight'] / calibration['position_scale_m'] ** 2)
            ).masked_fill(~valid_prefix, 0.)
    return torch.nn.functional.pad(loss.reshape(-1, 1), (0, 25))


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
