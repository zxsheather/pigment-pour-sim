"""Robot-only joint representation for a prospective Wall cylinder policy.

The first nine state fields align with nine action fields. Wall's generic
relative-action loader subtracts state only for the first six joint targets;
gripper aperture and base velocities stay absolute. No object pose is used.
"""
import numpy as np


APERTURE_SCALE_M = .0975
JOINT_ACTION_WIDTH = 9
JOINT_STATE_WIDTH = 18
JOINT_ACTION_LAYOUT = {
    'follow_right_arm_joint_position_relative': 6,
    'follow_right_gripper': 1,
    'velocity_decomposed': 2,
    'action_padding': 17,
}
JOINT_STATE_LAYOUT = {
    'follow_right_arm_joint_position': 6,
    'follow_right_gripper': 1,
    'velocity_decomposed': 2,
    'right_arm_joint_velocity': 6,
    'base_odometry': 3,
    'action_padding': 8,
}
JOINT_ACTION_NAMES = ([f'right_joint_position_{i}' for i in range(1, 7)]
                      + ['right_gripper_opening_fraction', 'base_vx', 'base_yaw_rate'])
JOINT_STATE_NAMES = (JOINT_ACTION_NAMES
                     + [f'right_joint_velocity_{i}' for i in range(1, 7)]
                     + ['base_odometry_x', 'base_odometry_y', 'base_odometry_yaw'])


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def relative_joint_chunks(states, actions, horizon):
    states, actions = np.asarray(states), np.asarray(actions)
    require(states.shape == (len(actions), JOINT_STATE_WIDTH)
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
            and state.shape == (JOINT_STATE_WIDTH,)
            and np.isfinite(actions).all() and np.isfinite(state).all(),
            'Invalid joint action forecast or anchor state')
    result = actions[:, :JOINT_ACTION_WIDTH].astype(np.float64)
    result[:, :6] += state[:6]
    return result


def robot_joint_observation(env):
    return joint_observation(env.data.qpos[env.arm_addresses], env.data.qvel[env.arm_dofs],
                             env.state_vector(), env.base_pose())


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
