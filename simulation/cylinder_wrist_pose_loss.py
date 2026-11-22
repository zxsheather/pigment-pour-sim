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


def slew_base_commands(requests, previous, lower, upper, step_limit):
    """Apply the planar actuator limits from the last observable base command."""
    commands = []
    for requested in requests.clamp(lower, upper).unbind(dim=-2):
        previous = previous + (requested - previous).clamp(-step_limit, step_limit)
        commands.append(previous)
    return torch.stack(commands, dim=-2)


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


def held_aperture_loss(forecast, reference, state, calibration, dof_mask, flow_loss_mask):
    """Penalize reopening during a demonstrated narrow hold after approach."""
    gap_min, gap_max = state.new_tensor(calibration['state_aperture_affine_m']).unbind()
    error_min, error_max = state.new_tensor(calibration['tracking_error_affine_m']).unbind()
    goal_min, goal_max = state.new_tensor(calibration['action_aperture_affine_m']).unbind()
    gap = gap_min + (state[:, 6:7] + 1.) * (gap_max - gap_min) * .5
    error = error_min + (state[:, 24:25] + 1.) * (error_max - error_min) * .5
    target = goal_min + (reference[..., 6] + 1.) * (goal_max - goal_min) * .5
    requested = goal_min + (forecast[..., 6] + 1.) * (goal_max - goal_min) * .5
    min_gap, max_gap = calibration['measured_aperture_range_m']
    anchor = ((state[:, 23:24] <= calibration['cue_validity_threshold'])
              & (gap >= min_gap) & (gap <= max_gap)
              & (error <= -calibration['minimum_closing_error_m'])
              & (gap + error <= calibration['max_previous_target_m']))
    retaining = target <= calibration['max_teacher_target_m']
    if dof_mask is not None:
        retaining &= dof_mask.reshape_as(reference)[..., 6].bool()
    if flow_loss_mask is not None:
        retaining &= flow_loss_mask.reshape_as(target).bool()
    frames = anchor & retaining.cumprod(dim=1).bool()
    excess = (requested - target).clamp_min(0.) / calibration['scale_m']
    loss = calibration['weight'] * excess.square()
    measured_limit = calibration.get('measured_gap_limit')
    if measured_limit is not None:
        # Preserve the demonstrated target when the measured aperture is narrower.
        limit = torch.maximum(target, gap - measured_limit['clearance_m'])
        over_gap = (requested - limit).clamp_min(0.) / calibration['scale_m']
        loss = loss + measured_limit['weight'] * over_gap.square()
    return loss.masked_fill(~frames, 0.)


def held_vertical_lift_loss(planned, expected, anchor_tip, reference, state, valid_prefix, dof_mask, settings):
    """Supervise vertical progress during a demonstrated held lift with a still base."""
    gap_low, gap_high = state.new_tensor(settings['state_aperture_affine_m']).unbind()
    error_low, error_high = state.new_tensor(settings['tracking_error_affine_m']).unbind()
    opening_low, opening_high = state.new_tensor(settings['action_aperture_affine_m']).unbind()
    base_low, base_high = state.new_tensor(settings['base_forward_affine_m_s']).unbind()
    yaw_low, yaw_high = state.new_tensor(settings['base_yaw_affine_rad_s']).unbind()
    gap = gap_low + (state[:, 6:7] + 1.) * (gap_high - gap_low) * .5
    error = error_low + (state[:, 24:25] + 1.) * (error_high - error_low) * .5
    opening = opening_low + (reference[..., 6] + 1.) * (opening_high - opening_low) * .5
    base_forward = base_low + (reference[..., 7] + 1.) * (base_high - base_low) * .5
    base_yaw = yaw_low + (reference[..., 8] + 1.) * (yaw_high - yaw_low) * .5

    low, high = settings['measured_aperture_range_m']
    eligible_phase = state[:, 23:24] <= settings['cue_validity_threshold']
    handoff = settings.get('approach_handoff')
    if handoff is not None:
        target_error = torch.linalg.vector_norm(wrist_target_displacement(state, handoff), dim=-1, keepdim=True)
        aligned_close = ((gap <= handoff['max_measured_aperture_m'])
                         & (gap + error <= handoff['max_commanded_aperture_m'])
                         & (target_error <= handoff['max_wrist_target_error_m']))
        eligible_phase = eligible_phase | aligned_close
    held = eligible_phase & (gap >= low) & (gap <= high) & (error <= -settings['minimum_preload_m'])
    retaining = ((opening <= settings['max_teacher_aperture_m'])
                 & (base_forward.abs() <= settings['max_base_speed_m_s'])
                 & (base_yaw.abs() <= settings['max_base_yaw_rad_s']))
    if dof_mask is not None:
        retaining &= dof_mask.reshape_as(reference)[..., 6:9].bool().all(dim=-1)
    prefix = valid_prefix & retaining.cumprod(dim=1).bool()
    expected_z = expected[..., 0, 2]
    planned_z = planned[..., 0, 2]
    upward_label = (expected_z - anchor_tip[:, 2:3] >= settings['minimum_teacher_rise_m']) & prefix
    frames = held & upward_label.any(dim=1, keepdim=True) & prefix
    shortfall = (expected_z - planned_z).clamp_min(0.) / settings['scale_m']
    return (settings['weight'] * shortfall.square()).masked_fill(~frames, 0.)


def wrist_target_displacement(state, calibration):
    cue_low, cue_high = state.new_tensor(calibration['cue_affine']).unbind(0)
    cue = cue_low + (state[:, (18, 21, 22)] + 1.) * (cue_high - cue_low) * .5
    # Convert the stored asinh cue back to a displacement in meters.
    return cue.sinh() * calibration['cue_near_scale_m']


def aligned_grasp_closing_loss(forecast, reference, state, valid, dof_mask, settings):
    """Supervise a demonstrated close from an aligned, open approach state."""
    cue = wrist_target_displacement(state, settings)
    tolerance = state.new_tensor(settings['max_position_error_m'])
    state_min, state_max = state.new_tensor(settings['state_aperture_affine_m']).unbind()
    measured = state_min + (state[:, 6] + 1.) * (state_max - state_min) * .5
    aligned = ((state[:, 23] > 0.) & (cue.abs() <= tolerance).all(dim=-1)
               & (measured >= settings['min_measured_aperture_m']))

    action_min, action_max = state.new_tensor(settings['action_aperture_affine_m']).unbind()
    target = action_min + (reference[..., 6] + 1.) * (action_max - action_min) * .5
    requested = action_min + (forecast[..., 6] + 1.) * (action_max - action_min) * .5
    if dof_mask is not None:
        valid = valid & dof_mask.reshape_as(reference)[..., 6:9].bool().all(dim=-1)
    active = aligned[:, None] & valid.cumprod(dim=1).bool() & (target <= settings['max_teacher_aperture_m'])
    excess = (requested - target).clamp_min(0.) / settings['scale_m']
    return (settings['weight'] * excess.square()).masked_fill(~active, 0.)


def wrist_target_is_near(state, calibration):
    displacement = wrist_target_displacement(state, calibration)
    return (state[:, 23] > 0.) & (displacement.square().sum(-1) <= calibration['radius_m'] ** 2)


def base_commands_from_observation(requests, state, low, high, response):
    state_low, state_high = state.new_tensor(response['previous_motion_affine']).unbind(0)
    # Forward command is recorded. The measured yaw rate estimates the preceding yaw command.
    previous = state_low + (state[:, (25, 8)] + 1.) * (state_high - state_low) * .5
    return slew_base_commands(requests, previous, low, high, response['step_limit'])


def approach_alignment_frame(state, world, settings):
    yaw_low, yaw_high = state.new_tensor(settings['state_yaw_affine']).unbind()
    yaw = yaw_low + (state[:, 17] + 1.) * (yaw_high - yaw_low) * .5
    # Markers are in the starting base frame. Convert horizontal projections
    # to the table axes so base yaw does not change the meaning of forward.
    forward_axis = torch.stack((yaw.sin(), yaw.cos()), dim=-1)
    sideways_axis = torch.stack((yaw.cos(), -yaw.sin()), dim=-1)
    cue = wrist_target_displacement(state, world)
    aperture_low, aperture_high = state.new_tensor(settings['state_aperture_affine_m']).unbind()
    aperture = aperture_low + (state[:, 6] + 1.) * (aperture_high - aperture_low) * .5
    return forward_axis, sideways_axis, cue, aperture


def high_approach_alignment_loss(planned, expected, measured_tip, state, valid_prefix, world, settings):
    """Match descent and restrain forward overshoot during a demonstrated alignment."""
    forward_axis, sideways_axis, cue, aperture = approach_alignment_frame(state, world, settings)
    forward_gap = (cue[:, :2] * forward_axis).sum(dim=-1)
    sideways_gap = (cue[:, :2] * sideways_axis).sum(dim=-1).abs()
    depth_low, depth_high = settings['depth_range_m']
    height_low, height_high = settings['height_range_m']
    height = -cue[:, 2]
    in_region = ((forward_gap >= depth_low) & (forward_gap <= depth_high)
                 & (height > height_low) & (height <= height_high))
    near_region = settings.get('near_region')
    if near_region is not None:
        near_depth_low, near_depth_high = near_region['depth_range_m']
        near_height_low, near_height_high = near_region['height_range_m']
        in_region = in_region | ((forward_gap >= near_depth_low) & (forward_gap <= near_depth_high)
                                 & (height > near_height_low) & (height <= near_height_high))
    eligible = ((state[:, 23] > 0.) & in_region & (sideways_gap <= settings['max_sideways_gap_m'])
                & (aperture >= settings['min_aperture_m']))

    teacher_shift = expected[..., 0, :] - measured_tip[:, None, :]
    teacher_depth = (teacher_shift[..., :2] * forward_axis[:, None, :]).sum(dim=-1)
    lowering = ((teacher_shift[..., 2] <= -settings['min_teacher_descent_m'])
                & (teacher_depth.abs() <= settings['max_teacher_depth_change_m']))
    error = planned[..., 0, :] - expected[..., 0, :]
    forward_excess = (error[..., :2] * forward_axis[:, None, :]).sum(dim=-1).clamp_min(0.)
    high_excess = error[..., 2].clamp_min(0.)
    loss = (forward_excess.square() + high_excess.square()) * (settings['weight'] / settings['scale_m'] ** 2)
    return loss.masked_fill(~(eligible[:, None] & lowering & valid_prefix), 0.)


def lateral_approach_alignment_loss(planned, expected, measured_tip, state, valid_prefix, world, settings):
    """Supervise a demonstrated rear clearance move or lateral alignment before advancing."""
    forward, sideways, cue, aperture = approach_alignment_frame(state, world, settings)
    depth = (cue[:, :2] * forward).sum(dim=-1)
    side = (cue[:, :2] * sideways).sum(dim=-1)
    height = -cue[:, 2]
    depth_min, depth_max = settings['depth_range_m']
    side_min, side_max = settings['sideways_range_m']
    height_min, height_max = settings['height_range_m']
    eligible = ((state[:, 23] > 0.) & (aperture >= settings['min_aperture_m'])
                & (depth >= depth_min) & (depth <= depth_max)
                & (side.abs() >= side_min) & (side.abs() <= side_max)
                & (height >= height_min) & (height <= height_max))

    teacher_shift = expected[..., 0, :] - measured_tip[:, None, :]
    teacher_depth = (teacher_shift[..., :2] * forward[:, None, :]).sum(dim=-1)
    teacher_side = (teacher_shift[..., :2] * sideways[:, None, :]).sum(dim=-1)
    direction = side.sign()[:, None]
    movement = settings['min_teacher_movement_m']
    drift = settings['max_teacher_drift_m']
    retreat = ((teacher_depth <= -movement) & (teacher_side.abs() <= drift)
               & (teacher_shift[..., 2].abs() <= drift))
    align = ((direction * teacher_side >= movement) & (teacher_depth.abs() <= drift))

    error = planned[..., 0, :] - expected[..., 0, :]
    forward_excess = (error[..., :2] * forward[:, None, :]).sum(dim=-1).clamp_min(0.)
    lateral_error = (error[..., :2] * sideways[:, None, :]).sum(dim=-1)
    lateral_shortfall = (-direction * lateral_error).clamp_min(0.)
    lateral_penalty = torch.where(retreat, lateral_error.square(), lateral_shortfall.square())
    loss = (forward_excess.square() + lateral_penalty) * (settings['weight'] / settings['scale_m'] ** 2)
    return loss.masked_fill(~(eligible[:, None] & (retreat | align) & valid_prefix), 0.)


def open_depth_alignment_loss(planned, expected, measured_tip, forecast, reference, state,
                              valid_prefix, teacher_motion, world, settings):
    """Keep the gripper open and match wrist alignment with a still base."""
    forward, sideways, cue, aperture = approach_alignment_frame(state, world, settings)
    depth = (cue[:, :2] * forward).sum(dim=-1)
    side = (cue[:, :2] * sideways).sum(dim=-1)
    offset = torch.stack((side, cue[:, 2]), dim=-1)
    initial_error = torch.linalg.vector_norm(offset, dim=-1)
    depth_low, depth_high = settings['depth_range_m']
    error_low, error_high = settings['plane_error_range_m']
    min_gap = settings['min_open_aperture_m']
    region = ((state[:, 23] > 0.) & (aperture >= min_gap)
              & (depth >= depth_low) & (depth <= depth_high)
              & (initial_error >= error_low) & (initial_error <= error_high))

    goal = measured_tip[:, None, :] + cue[:, None, :]
    residual = expected[..., 0, :] - goal
    expected_error = torch.linalg.vector_norm(torch.stack((
        (residual[..., :2] * sideways[:, None, :]).sum(dim=-1), residual[..., 2]), dim=-1), dim=-1)
    teacher_depth = ((expected[..., 0, :2] - measured_tip[:, None, :2])
                     * forward[:, None, :]).sum(dim=-1)
    low, high = state.new_tensor(settings['action_aperture_affine_m']).unbind()
    teacher_gap = low + (reference[..., 6] + 1.) * (high - low) * .5
    still = (teacher_motion.abs() <= state.new_tensor(settings['max_teacher_base_motion'])).all(dim=(-2, -1))
    eligible = (region & still & valid_prefix[:, -1] & (teacher_gap >= min_gap).all(dim=1)
                & (teacher_depth.abs() <= settings['max_teacher_depth_change_m']).all(dim=1)
                & (initial_error - expected_error[:, -1] >= settings['min_teacher_alignment_m']))

    residual = planned[..., 0, :] - goal
    planned_error = torch.linalg.vector_norm(torch.stack((
        (residual[..., :2] * sideways[:, None, :]).sum(dim=-1), residual[..., 2]), dim=-1), dim=-1)
    depth_error = ((planned[..., 0, :2] - expected[..., 0, :2]) * forward[:, None, :]).sum(dim=-1)
    requested_gap = low + (forecast[..., 6] + 1.) * (high - low) * .5
    shortfall = (planned_error - expected_error).clamp_min(0.)
    plane_loss = settings['weight'] * (depth_error.square() + shortfall.square()) / settings['scale_m'] ** 2
    gap_loss = settings['aperture_weight'] * ((min_gap - requested_gap).clamp_min(0.) / settings['aperture_scale_m']).square()
    return (plane_loss + gap_loss).masked_fill(~eligible[:, None], 0.)


def open_approach_advance_loss(planned, expected, measured_tip, forecast, reference, state,
                              valid_prefix, world, settings):
    """Match open-finger advance and, when demonstrated, lateral correction."""
    cue = wrist_target_displacement(state, world)
    depth_min, depth_max = settings['depth_range_m']
    state_low, state_high = state.new_tensor(settings['state_aperture_affine_m']).unbind()
    action_low, action_high = state.new_tensor(settings['action_aperture_affine_m']).unbind()
    measured_gap = state_low + (state[:, 6] + 1.) * (state_high - state_low) * .5
    teacher_gap = action_low + (reference[..., 6] + 1.) * (action_high - action_low) * .5
    requested_gap = action_low + (forecast[..., 6] + 1.) * (action_high - action_low) * .5
    min_gap = settings['min_open_aperture_m']
    min_measured_gap = settings.get('min_measured_aperture_m', min_gap)
    region = ((state[:, 23] > 0.) & (cue[:, 0] >= depth_min) & (cue[:, 0] <= depth_max)
              & (cue[:, 1].abs() <= settings['max_sideways_gap_m'])
              & (cue[:, 2].abs() <= settings['max_vertical_gap_m']) & (measured_gap >= min_measured_gap))

    teacher_shift = expected[..., 0, :] - measured_tip[:, None, :]
    difference = expected[..., 0, :] - planned[..., 0, :]
    lateral = settings.get('lateral_correction')
    if lateral is None:
        axis = torch.nn.functional.normalize(cue, dim=-1)
        teacher_progress = (teacher_shift * axis[:, None, :]).sum(dim=-1)
        deficit = (difference * axis[:, None, :]).sum(dim=-1)
        lateral_loss = 0.
    else:
        direction = cue[:, 1].sign()[:, None]
        teacher_side = direction * teacher_shift[..., 1]
        region &= ((cue[:, 1].abs() >= lateral['min_sideways_gap_m'])
                   & (teacher_side[:, -1] >= lateral['min_teacher_motion_m']))
        teacher_progress = teacher_shift[..., 0]
        deficit = difference[..., 0]
        lateral_error = (direction * difference[..., 1]).clamp_min(0.) / lateral['scale_m']
        lateral_loss = lateral['weight'] * lateral_error.square()
    eligible = (region & valid_prefix[:, -1] & (teacher_gap >= min_gap).all(dim=1)
                & (teacher_progress[:, -1] >= settings['min_teacher_progress_m']))
    progress_error = deficit.clamp_min(0.) / settings['progress_scale_m']
    gap_error = (min_gap - requested_gap).clamp_min(0.) / settings['aperture_scale_m']
    loss = (settings['progress_weight'] * progress_error.square()
            + settings['aperture_weight'] * gap_error.square() + lateral_loss)
    return loss.masked_fill(~(eligible[:, None] & valid_prefix), 0.)


def near_approach_world_wrist_loss(planned, expected, forecast, reference, state, calibration,
                                  valid, dof_mask, measured_tip=None):
    """Supervise coordinated wrist and base outcomes close to a visible target."""
    near = wrist_target_is_near(state, calibration)
    if dof_mask is not None:
        valid = valid & dof_mask.reshape_as(reference)[..., 7:9].bool().all(dim=-1)
    valid_prefix = valid.cumprod(dim=1).bool()
    active = near[:, None] & valid_prefix

    low, high = state.new_tensor(calibration['base_action_affine']).unbind(0)
    planned_motion = low + (forecast[..., 7:9] + 1.) * (high - low) * .5
    teacher_motion = low + (reference[..., 7:9] + 1.) * (high - low) * .5
    servo = calibration.get('base_command_response')
    if servo is not None:
        planned_motion = base_commands_from_observation(planned_motion, state, low, high, servo)
    step = calibration['base_control_dt_s']
    planned_world = apply_base_motion(planned, planned_motion, step)
    expected_world = apply_base_motion(expected, teacher_motion, step)
    error = planned_world - expected_world
    tip = error[..., 0, :]
    orientation = error[..., 1:, :] - tip[..., None, :]
    squared_error = tip.square().sum(-1) + .5 * orientation.square().sum((-2, -1))
    loss = squared_error * (calibration['weight'] / calibration['scale_m'] ** 2)
    progress = calibration.get('approach_progress')
    alignment = calibration.get('high_approach_alignment')
    lateral = calibration.get('lateral_approach_alignment')
    depth_alignment = calibration.get('open_depth_alignment')
    aligned_advance = calibration.get('aligned_open_advance')
    lateral_advance = calibration.get('open_lateral_advance')
    if servo is None and progress is not None:
        bounded = base_commands_from_observation(planned_motion, state, low, high,
                                                 progress['base_command_response'])
        planned_world = apply_base_motion(planned, bounded, step)
    far_loss = None
    if progress is not None:
        displacement = wrist_target_displacement(state, calibration)
        target = measured_tip + displacement
        start_distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        teacher_distance = torch.linalg.vector_norm(expected_world[..., 0, :] - target[:, None, :], dim=-1)
        planned_distance = torch.linalg.vector_norm(planned_world[..., 0, :] - target[:, None, :], dim=-1)
        teacher_progress = start_distance - teacher_distance
        advancing = (start_distance >= progress['min_distance_m']) & (teacher_progress >= progress['min_progress_m'])
        shortfall = (planned_distance - teacher_distance).clamp_min(0.)
        loss = loss + (progress['weight'] * (shortfall / progress['scale_m']).square()).masked_fill(~advancing, 0.)
        far = progress.get('far')
        if far is not None:
            far_start = ((state[:, 23:24] > 0.) & ~near[:, None]
                         & (start_distance <= far['max_distance_m']))
            far_valid = far_start & valid_prefix & (teacher_progress >= far['min_progress_m'])
            far_loss = (far['weight'] * (shortfall / far['scale_m']).square()).masked_fill(~far_valid, 0.)
    loss = loss.masked_fill(~active, 0.)
    if far_loss is not None:
        loss = loss + far_loss
    alignment_prefix = valid_prefix
    if dof_mask is not None and any(setting is not None for setting in
                                    (alignment, lateral, depth_alignment, aligned_advance, lateral_advance)):
        gripper_valid = dof_mask.reshape_as(reference)[..., 6].bool()
        alignment_prefix = (valid & gripper_valid).cumprod(dim=1).bool()
    if alignment is not None:
        loss = loss + high_approach_alignment_loss(planned_world, expected_world, measured_tip, state,
                                                  alignment_prefix, calibration, alignment)
    if lateral is not None:
        loss = loss + lateral_approach_alignment_loss(planned_world, expected_world, measured_tip, state,
                                                     alignment_prefix, calibration, lateral)
    if depth_alignment is not None:
        loss = loss + open_depth_alignment_loss(planned, expected, measured_tip, forecast, reference, state,
                                                 alignment_prefix, teacher_motion, calibration, depth_alignment)
    for advance in (aligned_advance, lateral_advance):
        if advance is not None:
            loss = loss + open_approach_advance_loss(planned_world, expected_world, measured_tip, forecast,
                                                     reference, state, alignment_prefix, calibration, advance)
    return loss


def near_grasp_roll_loss(planned, expected, state, valid_prefix, calibration, marker_offset):
    """Match the wrist roll relative to gravity for nearby approach observations."""
    # Components of the base up axis in the two wrist axes perpendicular to
    # its shaft. These components are invariant to base yaw and translation.
    planned_up = (planned[..., 2:, 2] - planned[..., :1, 2]) / marker_offset
    expected_up = (expected[..., 2:, 2] - expected[..., :1, 2]) / marker_offset
    planned_length = torch.linalg.vector_norm(planned_up, dim=-1, keepdim=True)
    expected_length = torch.linalg.vector_norm(expected_up, dim=-1, keepdim=True)
    # Roll becomes ambiguous when the shaft points vertically. Bound the
    # normalization gradient and require a well-defined teacher target.
    min_projection = .25
    error = (planned_up / planned_length.clamp_min(min_projection)
             - expected_up / expected_length.clamp_min(min_projection))
    active = (wrist_target_is_near(state, calibration)[:, None] & valid_prefix
              & (expected_length[..., 0] >= min_projection))
    return (calibration['weight'] * error.square().sum(-1)).masked_fill(~active, 0.)


def near_heading_recovery_loss(forecast, reference, state, valid, calibration):
    """Penalize turning farther away than a demonstrated heading recovery."""
    lower, upper = state.new_tensor(calibration['state_yaw_affine']).unbind()
    yaw = lower + (state[:, 17] + 1.) * (upper - lower) * .5
    delta = yaw - calibration['reset_yaw_rad']
    error = torch.atan2(delta.sin(), delta.cos())
    side = error.sign()[:, None]

    lower, upper = state.new_tensor(calibration['action_yaw_affine']).unbind()
    requested = lower + (forecast[..., 8:9] + 1.) * (upper - lower) * .5
    demonstrated = lower + (reference[..., 8] + 1.) * (upper - lower) * .5
    measured_lo, measured_hi = state.new_tensor(calibration['state_yaw_rate_affine']).unbind()
    previous = measured_lo + (state[:, 8:9] + 1.) * (measured_hi - measured_lo) * .5
    applied = slew_base_commands(requested, previous, lower, upper, calibration['command_step_limit'])[..., 0]

    returning = (side * demonstrated <= -calibration['minimum_return_rate_rad_s']) & valid
    nearby = wrist_target_is_near(state, calibration) & (error.abs() >= calibration['min_heading_error_rad'])
    active = nearby[:, None] & valid.cumprod(dim=1).bool() & returning.cumsum(dim=1).bool()
    excess = side * (applied - demonstrated).cumsum(dim=1) * calibration['control_dt_s']
    loss = calibration['weight'] * (excess.clamp_min(0.) / calibration['scale_rad']).square()
    return loss.masked_fill(~active, 0.)


def servo_increment_wrist_pose_loss(predicted, labels, proprioception, calibration,
                                    dof_mask=None, flow_loss_mask=None):
    """Compare the FK of composed arm commands from a shared proprioceptive anchor."""
    if (labels.ndim != 3 or labels.shape[1:] != (8, 26)
            or predicted.shape != (labels.shape[0] * 8, 26)
            or proprioception is None or proprioception.shape != (labels.shape[0], 1, 26)):
        raise ValueError('Servo wrist supervision requires eight increments and a 26D wrist-target state')
    dtype = torch.float64 if predicted.dtype == torch.float64 else torch.float32
    state = proprioception[:, 0].to(dtype)
    measured = denormalize_arm(state, calibration['state_affine_rad'])
    # The wrist-target layout preserves command error for joints 2 and 3.
    # Use measured positions for the four slots replaced by the visual cue.
    low, high = state.new_tensor(calibration['tracking_affine_rad']).unbind(0)
    tracking = low + (state[..., 19:21] + 1) * (high - low) * .5
    anchor = measured + torch.nn.functional.pad(tracking, (1, 3))
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
        measured_tip = None
        if any(world_wrist.get(key) is not None for key in (
                'approach_progress', 'high_approach_alignment', 'lateral_approach_alignment', 'open_depth_alignment',
                'aligned_open_advance', 'open_lateral_advance')):
            measured_tip = wrist_markers(measured, calibration)[..., 0, :]
        loss = loss + near_approach_world_wrist_loss(planned, expected, forecast, reference, state,
                                                   world_wrist, valid, dof_mask, measured_tip)
    roll = calibration.get('near_grasp_roll')
    if roll is not None:
        loss = loss + near_grasp_roll_loss(planned, expected, state, valid_prefix, roll,
                                          calibration['orientation_marker_offset_m'])
    lift = calibration.get('held_vertical_lift')
    if lift is not None:
        anchor_tip = wrist_markers(anchor, calibration)[..., 0, :]
        loss = loss + held_vertical_lift_loss(planned, expected, anchor_tip, reference, state, valid_prefix, dof_mask, lift)
    loss = loss.masked_fill(~valid_prefix, 0.)
    padded = torch.nn.functional.pad(loss.reshape(-1, 1), (0, 25))
    release = calibration.get('release_opening')
    if release is not None:
        gap_loss = release_opening_loss(forecast, reference, state, release, dof_mask, flow_loss_mask)
        padded = padded + torch.nn.functional.pad(gap_loss.reshape(-1, 1), (6, 19))
    retaining = calibration.get('held_aperture')
    if retaining is not None:
        gap_loss = held_aperture_loss(forecast, reference, state, retaining, dof_mask, flow_loss_mask)
        padded = padded + torch.nn.functional.pad(gap_loss.reshape(-1, 1), (6, 19))
    aligned_closing = calibration.get('aligned_grasp_closing')
    if aligned_closing is not None:
        gap_loss = aligned_grasp_closing_loss(forecast, reference, state, valid, dof_mask, aligned_closing)
        padded = padded + torch.nn.functional.pad(gap_loss.reshape(-1, 1), (6, 19))
    heading = calibration.get('near_heading_recovery')
    if heading is not None:
        yaw_valid = torch.ones_like(valid)
        if dof_mask is not None:
            yaw_valid &= dof_mask.reshape_as(labels)[..., 8].bool()
        if flow_loss_mask is not None:
            yaw_valid &= flow_loss_mask.reshape_as(valid).bool()
        heading_loss = near_heading_recovery_loss(forecast, reference, state, yaw_valid, heading)
        padded = padded + torch.nn.functional.pad(heading_loss.reshape(-1, 1), (8, 17))
    return padded
