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
    return apply_base_motion(wrist_markers(joints, calibration), base_velocity,
                             calibration['base_control_dt_s'])


def apply_base_motion(markers, base_velocity, step):
    """Apply the commanded planar twist prefix to markers in the moving base frame."""
    forward, yaw_rate = base_velocity.unbind(-1)
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
    rate = calibration.get('arm_joint_rate_penalty')
    if rate is not None:
        lower, upper = reference.new_tensor(calibration['action_affine_rad']).unbind(0)
        normalized_limit = (2 * rate['rad_per_second'] * calibration['base_control_dt_s']) / (upper - lower)
        # A chunk has a common measured joint anchor, so differencing its
        # relative targets gives the displacement between adjacent setpoints.
        excess = (forecast[..., :6].diff(dim=1).abs() - normalized_limit).clamp_min(0.)
        adjacent_valid = valid[:, :-1] & valid[:, 1:]
        rate_loss = (rate['weight'] * excess.square().sum(-1)).masked_fill(~adjacent_valid, 0.)
        loss = loss + torch.nn.functional.pad(rate_loss, (1, 0))
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


def servo_arm_targets(actions, anchor, calibration):
    """Compose rate-limited increments with the fixed joint-limit margin."""
    lower, upper = actions.new_tensor(calibration['joint_limits_rad']).unbind(0)
    limit = calibration['arm_increment_limit_rad']
    deltas = denormalize_arm(actions, calibration['action_affine_rad']).clamp(-limit, limit)
    targets = []
    for delta in deltas.unbind(-2):
        anchor = (anchor + delta).clamp(lower, upper)
        targets.append(anchor)
    return torch.stack(targets, dim=-2)


def release_opening_targets(reference, state, calibration, dof_mask, flow_loss_mask):
    """Identify the teacher targets within a demonstrated widening prefix."""
    state_min, state_max = state.new_tensor(calibration['state_aperture_affine_m']).unbind()
    tracking_min, tracking_max = state.new_tensor(calibration['tracking_error_affine_m']).unbind()
    action_min, action_max = state.new_tensor(calibration['action_aperture_affine_m']).unbind()
    gap = state_min + (state[:, 6:7] + 1.) * (state_max-state_min) * .5
    tracking_error = tracking_min + (state[:, 24:25] + 1.) * (tracking_max-tracking_min) * .5
    previous_target = gap + tracking_error
    target = action_min + (reference[..., 6] + 1.) * (action_max-action_min) * .5

    valid = torch.ones_like(target, dtype=torch.bool)
    if dof_mask is not None:
        valid &= dof_mask.reshape_as(reference)[..., 6].bool()
    if flow_loss_mask is not None:
        valid &= flow_loss_mask.reshape_as(target).bool()
    valid &= state[:, 23:24] > calibration['cue_validity_threshold']
    steps = torch.cat((previous_target, target), dim=1).diff(dim=1)
    widening = (valid & (steps >= -calibration['allowed_target_drop_m'])).cumprod(dim=1).bool()
    gain = target - previous_target
    release = (widening & (gain >= calibration['minimum_opening_gain_m'])).any(dim=1, keepdim=True)
    frames = release & widening & (gain > calibration['allowed_target_drop_m'])
    return target, frames


def release_opening_loss(forecast, reference, state, calibration, dof_mask, flow_loss_mask):
    """Supervise the opening requested over a demonstrated widening prefix."""
    target, frames = release_opening_targets(reference, state, calibration, dof_mask, flow_loss_mask)
    action_min, action_max = state.new_tensor(calibration['action_aperture_affine_m']).unbind()
    requested = action_min + (forecast[..., 6] + 1.) * (action_max-action_min) * .5
    # Training labels already describe bounded servo targets. Compare them
    # with the undamped request so a closing prediction still gets a gradient.
    shortfall = (target - requested).clamp_min(0.) / calibration['scale_m']
    return (calibration['weight'] * shortfall.square()).masked_fill(~frames, 0.)


def near_approach_world_wrist_loss(planned, expected, forecast, reference, state, calibration,
                                  valid, dof_mask):
    """Supervise coordinated wrist and base outcomes close to a visible target."""
    cue_low, cue_high = state.new_tensor(calibration['cue_affine']).unbind(0)
    cue = cue_low + (state[:, (18, 21, 22)] + 1.) * (cue_high - cue_low) * .5
    # Convert the stored asinh cue back to a displacement in meters.
    displacement = cue.sinh() * calibration['cue_near_scale_m']
    near = (state[:, 23] > 0.) & (displacement.square().sum(-1) <= calibration['radius_m'] ** 2)
    if dof_mask is not None:
        valid = valid & dof_mask.reshape_as(reference)[..., 7:9].bool().all(dim=-1)
    active = near[:, None] & valid.cumprod(dim=1).bool()

    low, high = state.new_tensor(calibration['base_action_affine']).unbind(0)
    planned_motion = low + (forecast[..., 7:9] + 1.) * (high - low) * .5
    teacher_motion = low + (reference[..., 7:9] + 1.) * (high - low) * .5
    step = calibration['base_control_dt_s']
    error = apply_base_motion(planned, planned_motion, step) - apply_base_motion(expected, teacher_motion, step)
    tip = error[..., 0, :]
    orientation = error[..., 1:, :] - tip[..., None, :]
    squared_error = tip.square().sum(-1) + .5 * orientation.square().sum((-2, -1))
    return (squared_error * (calibration['weight'] / calibration['scale_m'] ** 2)).masked_fill(~active, 0.)


def servo_increment_wrist_pose_loss(predicted, labels, proprioception, calibration,
                                    dof_mask=None, flow_loss_mask=None):
    """Compare the FK of composed arm commands from a shared proprioceptive anchor."""
    if (labels.ndim != 3 or labels.shape[1:] != (8, 26)
            or predicted.shape != (labels.shape[0] * 8, 26)
            or proprioception is None or proprioception.shape != (labels.shape[0], 1, 26)):
        raise ValueError('Servo wrist supervision requires eight increments and a 26D wrist-target state')
    dtype = torch.float64 if predicted.dtype == torch.float64 else torch.float32
    state = proprioception[:, 0].to(dtype)
    anchor = denormalize_arm(state, calibration['state_affine_rad'])
    # The wrist-target layout preserves command error for joints 2 and 3.
    # Use measured positions for the four slots replaced by the visual cue.
    low, high = state.new_tensor(calibration['tracking_affine_rad']).unbind(0)
    tracking = low + (state[..., 19:21] + 1) * (high - low) * .5
    anchor = anchor + torch.nn.functional.pad(tracking, (1, 3))
    reference = labels.to(dtype)
    forecast = predicted.reshape_as(labels).to(dtype)
    expected = wrist_markers(servo_arm_targets(reference, anchor, calibration), calibration)
    planned = wrist_markers(servo_arm_targets(forecast, anchor, calibration), calibration)
    error = planned - expected
    tip = error[..., 0, :]
    orientation = error[..., 1:, :] - tip[..., None, :]
    squared_error = tip.square().sum(-1) + .5 * orientation.square().sum((-2, -1))
    valid = torch.ones_like(squared_error, dtype=torch.bool)
    if dof_mask is not None:
        valid &= dof_mask.reshape_as(labels)[..., :6].bool().all(dim=-1)
    if flow_loss_mask is not None:
        valid &= flow_loss_mask.reshape_as(valid).bool()
    valid_prefix = valid.cumprod(dim=1).bool()
    loss = squared_error * (calibration['weight'] / calibration['position_scale_m'] ** 2)
    closing = calibration.get('closing_lateral_alignment')
    if closing is not None:
        # The teacher's local y axis is the direction between the parallel
        # fingers. Apply the extra supervision only when its approach cue is
        # active and its commanded aperture enters the closing range.
        closing_frames = ((state[:, None, 23] > closing['cue_validity_threshold'])
                          & (reference[..., 6] <= closing['aperture_label_max']))
        if dof_mask is not None:
            closing_frames &= dof_mask.reshape_as(labels)[..., 6].bool()
        lateral_axis = (expected[..., 2, :] - expected[..., 0, :]) / calibration['orientation_marker_offset_m']
        lateral_error = (tip * lateral_axis).sum(dim=-1)
        alignment_loss = closing['weight'] * (lateral_error / closing['scale_m']).square()
        loss = loss + alignment_loss.masked_fill(~closing_frames, 0.)
    release_wrist = calibration.get('release_wrist')
    if release_wrist is not None:
        _, widening = release_opening_targets(reference, state, release_wrist, dof_mask, flow_loss_mask)
        release_loss = squared_error * (release_wrist['weight'] / release_wrist['scale_m'] ** 2)
        loss = loss + release_loss.masked_fill(~widening, 0.)
    world_wrist = calibration.get('near_approach_world_wrist')
    if world_wrist is not None:
        loss = loss + near_approach_world_wrist_loss(planned, expected, forecast, reference, state,
                                                   world_wrist, valid, dof_mask)
    loss = loss.masked_fill(~valid_prefix, 0.)
    padded = torch.nn.functional.pad(loss.reshape(-1, 1), (0, 25))
    release = calibration.get('release_opening')
    if release is not None:
        gap_loss = release_opening_loss(forecast, reference, state, release, dof_mask, flow_loss_mask)
        padded = padded + torch.nn.functional.pad(gap_loss.reshape(-1, 1), (6, 19))
    return padded
