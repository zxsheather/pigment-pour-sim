"""Robot-only joint representation for a prospective Wall cylinder policy.

The first nine state fields align with nine action fields. Wall's generic
relative-action loader subtracts state only for the first six joint targets;
gripper aperture and base velocities stay absolute. No object pose is used.
"""
import numpy as np


APERTURE_SCALE_M = .0975
JOINT_ACTION_WIDTH = 9
JOINT_STATE_WIDTH = 18
JOINT_SERVO_STATE_WIDTH = 26
JOINT_ACTION_LAYOUT = {
    'follow_right_arm_joint_position_relative': 6,
    'follow_right_gripper': 1,
    'velocity_decomposed': 2,
    'action_padding': 17,
}
JOINT_SERVO_INCREMENT = 'servo_increment'
JOINT_RELATIVE_POSITION = 'relative_position'
JOINT_INCREMENT_ACTION_KEY = 'follow_right_arm_servo_increment'
JOINT_INCREMENT_ACTION_LAYOUT = {
    JOINT_INCREMENT_ACTION_KEY: 6,
    **{name: width for name, width in JOINT_ACTION_LAYOUT.items() if name != 'follow_right_arm_joint_position_relative'},
}
ARM_SERVO_RATE_RAD_S = .65
JOINT_STATE_LAYOUT = {
    'follow_right_arm_joint_position': 6,
    'follow_right_gripper': 1,
    'velocity_decomposed': 2,
    'right_arm_joint_velocity': 6,
    'base_odometry': 3,
    'action_padding': 8,
}
JOINT_SERVO_STATE_LAYOUT = {
    **{name: width for name, width in JOINT_STATE_LAYOUT.items() if name != 'action_padding'},
    'right_arm_tracking_error': 6,
    'right_gripper_tracking_error': 1,
    'base_forward_command': 1,
}
JOINT_WRIST_CUE_STATE_LAYOUT = {
    **{name: width for name, width in JOINT_STATE_LAYOUT.items() if name != 'action_padding'},
    'initial_wrist_target_forward': 1,
    'right_joint_2_3_4_tracking_error': 3,
    'initial_wrist_target_left': 1,
    'initial_wrist_target_valid': 1,
    'right_gripper_tracking_error': 1,
    'base_forward_command': 1,
}
JOINT_WRIST_TARGET_STATE_LAYOUT = {
    **{name: width for name, width in JOINT_STATE_LAYOUT.items() if name != 'action_padding'},
    'wrist_target_forward': 1,
    'right_joint_2_3_tracking_error': 2,
    'wrist_target_left': 1,
    'wrist_target_up': 1,
    'wrist_target_valid': 1,
    'right_gripper_tracking_error': 1,
    'base_forward_command': 1,
}
JOINT_ACTION_NAMES = ([f'right_joint_position_{i}' for i in range(1, 7)]
                      + ['right_gripper_opening_fraction', 'base_vx', 'base_yaw_rate'])
JOINT_STATE_NAMES = (JOINT_ACTION_NAMES
                     + [f'right_joint_velocity_{i}' for i in range(1, 7)]
                     + ['base_odometry_x', 'base_odometry_y', 'base_odometry_yaw'])
JOINT_SERVO_STATE_NAMES = (JOINT_STATE_NAMES
                           + [f'right_joint_tracking_error_{i}' for i in range(1, 7)]
                           + ['right_gripper_tracking_error_fraction', 'previous_base_forward_command'])
WRIST_CUE_STATE_INDICES = (18, 22, 23)
WRIST_CUE_STATE_NAMES = ('initial_cylinder_forward_in_current_base_frame_m',
                         'initial_cylinder_left_in_current_base_frame_m', 'initial_rgb_detection_valid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def joint_action_representation(layout):
    items = list(layout.items())
    if items == list(JOINT_ACTION_LAYOUT.items()):
        return JOINT_RELATIVE_POSITION
    if items == list(JOINT_INCREMENT_ACTION_LAYOUT.items()):
        return JOINT_SERVO_INCREMENT
    return None


def joint_state_width(layout):
    items = list(layout.items())
    if items == list(JOINT_STATE_LAYOUT.items()):
        return JOINT_STATE_WIDTH
    require(items in (list(JOINT_SERVO_STATE_LAYOUT.items()), list(JOINT_WRIST_CUE_STATE_LAYOUT.items()),
                      list(JOINT_WRIST_TARGET_STATE_LAYOUT.items())),
            'Unsupported cylinder joint state layout')
    return JOINT_SERVO_STATE_WIDTH


def joint_observation(qpos, qvel, robot_state23, base_pose):
    qpos = np.asarray(qpos)
    qvel = np.asarray(qvel)
    robot = np.asarray(robot_state23)
    odometry = np.asarray(base_pose)
    require(qpos.shape == qvel.shape and qpos.shape[-1] == 6,
            'Expected matching six-joint positions and velocities')
    require(robot.shape == (*qpos.shape[:-1], 23)
            and odometry.shape == (*qpos.shape[:-1], 3), 'Robot state shape mismatch')
    state = np.concatenate([qpos, robot[..., 19:21], robot[..., 22:23],
                            qvel, odometry], axis=-1).astype(np.float32)
    require(np.isfinite(state).all(), 'Nonfinite robot observation')
    return state


def joint_command_action(commands):
    commands = np.asarray(commands)
    require(commands.shape[-1] == 9 and np.isfinite(commands).all(),
            'Expected bounded nine-dimensional joint commands')
    action = commands.copy().astype(np.float64)
    action[..., 6] *= 2 / APERTURE_SCALE_M
    return action.astype(np.float32)


def with_joint_servo_state(observation, previous_command):
    """Append tracked actuator state from the preceding applied control tick."""
    state = np.asarray(observation, dtype=np.float32)
    command = joint_command_action(previous_command)
    require(state.shape == (*command.shape[:-1], JOINT_STATE_WIDTH) and np.isfinite(state).all(),
            'Expected measured joint state and a matching previous command')
    tracking_error = command[..., :7] - state[..., :7]
    return np.concatenate((state, tracking_error, command[..., 7:8]), axis=-1)


def with_joint_wrist_cue_state(servo_state, initial_world_xy, *, origin_world_xy=None):
    """Project the displacement from the current base or a calibrated world origin."""
    state = np.asarray(servo_state, dtype=np.float32).copy()
    require(state.ndim > 0 and state.shape[-1] == JOINT_SERVO_STATE_WIDTH and np.isfinite(state).all(),
            'Expected the current measured servo state')
    if initial_world_xy is None:
        state[..., WRIST_CUE_STATE_INDICES] = 0.
        return state
    xy = np.asarray(initial_world_xy, dtype=np.float64)
    require(xy.shape == (2,) and np.isfinite(xy).all(), 'Invalid initial wrist location')
    pose = state[..., 15:18].astype(np.float64)
    origin = pose[..., :2]
    if origin_world_xy is not None:
        origin = np.asarray(origin_world_xy, dtype=np.float64)
        require(origin.shape == (2,) and np.isfinite(origin).all(), 'Invalid cue origin')
    dx, dy = np.moveaxis(xy - origin, -1, 0)
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    state[..., 18] = c * dx + s * dy
    state[..., 22] = -s * dx + c * dy
    state[..., 23] = 1.
    return state


def relative_joint_chunks(states, actions, horizon):
    states, actions = np.asarray(states), np.asarray(actions)
    require(states.shape in ((len(actions), JOINT_STATE_WIDTH), (len(actions), JOINT_SERVO_STATE_WIDTH))
            and actions.shape == (len(actions), JOINT_ACTION_WIDTH)
            and len(actions) > 0 and horizon > 0, 'Invalid joint demonstration')
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon),
                         len(actions) - 1)
    chunks = actions[indices].astype(np.float64)
    chunks[..., :6] -= states[:, None, :6]
    return chunks


def absolute_joint_chunk(actions, anchor_state):
    actions, state = np.asarray(actions), np.asarray(anchor_state)
    require(actions.ndim == 2 and actions.shape[1] in (9, 26)
            and state.shape in ((JOINT_STATE_WIDTH,), (JOINT_SERVO_STATE_WIDTH,))
            and np.isfinite(actions).all() and np.isfinite(state).all(),
            'Invalid joint action forecast or anchor state')
    result = actions[:, :JOINT_ACTION_WIDTH].astype(np.float64)
    result[:, :6] += state[:6]
    return result


def servo_increment_targets(actions, prior_commands, dt):
    """Decode each model increment from the most recently applied servo setpoint."""
    actions = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(prior_commands, dtype=np.float64)
    require(actions.ndim > 0 and actions.shape[-1] == JOINT_ACTION_WIDTH
            and actions.shape == previous.shape and np.isfinite(actions).all()
            and np.isfinite(previous).all() and np.isfinite(dt) and dt > 0, 'Invalid joint increment or prior command')
    limit = ARM_SERVO_RATE_RAD_S * dt
    targets = actions.copy()
    targets[..., :6] = previous[..., :6] + np.clip(actions[..., :6], -limit, limit)
    return targets


def robot_joint_observation(env, *, servo_state=False):
    observation = joint_observation(env.data.qpos[env.arm_addresses], env.data.qvel[env.arm_dofs],
                                    env.state_vector(), env.base_pose())
    return with_joint_servo_state(observation, bounded_joint_command(env)) if servo_state else observation


def joint_action_to_targets(env, action):
    """Apply the same official joint range margin used by Cartesian IK.

    Projection is recorded explicitly. The existing joint servo still applies
    its original rate limits and force caps; no object feedback is consulted.
    """
    action = np.asarray(action, dtype=float)
    require(action.shape == (JOINT_ACTION_WIDTH,) and np.isfinite(action).all(),
            'Invalid absolute joint action')
    limits = np.array([env.model.joint(f'right_arm_joint{i}').range for i in range(1, 7)])
    joints = np.clip(action[:6], limits[:, 0] + .005, limits[:, 1] - .005)
    projection = float(np.abs(joints - action[:6]).max())
    opening = float(np.clip(action[6], 0., 1.) * APERTURE_SCALE_M / 2)
    return (joints, opening, float(np.clip(action[7], -.12, .12)),
            float(np.clip(action[8], -.3, .3)), projection)


def bounded_joint_command(env):
    arm = [env.hold_targets[env.model.actuator(f'right_arm_joint{i}_servo').id]
           for i in range(1, 7)]
    fingers = [env.hold_targets[env.model.actuator(f'right_arm_gripper_{side}_joint_servo').id]
               for side in ('left', 'right')]
    return np.r_[arm, (fingers[0]-fingers[1])/2, env.applied_action].astype(np.float64)
