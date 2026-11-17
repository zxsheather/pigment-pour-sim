"""Cartesian action conversion shared by demonstrations and Wall execution.

Uses Wall-X's first-two-rows rotation convention and R_rel = R_abs R_state^T.
No object position, task phase or desired liquid volume enters these helpers.
"""
import numpy as np
import mujoco


def rotation_from_rows(value):
    value=np.asarray(value,dtype=float).reshape(2,3)
    a=value[0];norm=float(np.linalg.norm(a))
    if norm<1e-7:raise ValueError("Degenerate model rotation")
    a=a/norm;b=value[1]-np.dot(value[1],a)*a;norm=float(np.linalg.norm(b))
    if norm<1e-7:raise ValueError("Degenerate model rotation")
    b=b/norm
    return np.stack([a,b,np.cross(a,b)])


def relative_to_absolute(actions,anchor_state):
    actions=np.asarray(actions,dtype=float)
    state=np.asarray(anchor_state,dtype=float)
    if actions.ndim!=2 or actions.shape[1] not in (23,26) or state.shape!=(23,):
        raise ValueError("Expected action chunk Hx23/26 and robot state 23")
    if not np.isfinite(actions).all() or not np.isfinite(state).all():raise ValueError("Nonfinite action/state")
    absolute=actions[:,:23].copy()
    for offset in (0,10):
        absolute[:,offset:offset+3]+=state[offset:offset+3]
        rotation=rotation_from_rows(state[offset+3:offset+9])
        for index,row in enumerate(actions):
            absolute[index,offset+3:offset+9]=(rotation_from_rows(row[offset+3:offset+9])@rotation)[:2].reshape(6)
    return absolute.astype(np.float32)


def commanded_cartesian_action(env,reference_qpos):
    """Forward-kinematics label of commanded joints, in the observation frame."""
    data=env.ik_data
    data.qpos[:]=reference_qpos
    for actuator,address in env.position_actuators.items():data.qpos[address]=env.hold_targets[actuator]
    mujoco.mj_kinematics(env.model,data)
    base_rotation=data.xmat[env.robot_root].reshape(3,3)
    parts=[]
    for side in ("left","right"):
        body=env.model.body(f"{side}_arm_gripper_base_link").id
        position=base_rotation.T@(data.xpos[body]-data.xpos[env.robot_root])
        rotation=base_rotation.T@data.xmat[body].reshape(3,3)
        opening=(data.qpos[env.model.joint(f"{side}_arm_gripper_left_joint").qposadr[0]]
                 -data.qpos[env.model.joint(f"{side}_arm_gripper_right_joint").qposadr[0]])/.0975
        parts.extend([position,rotation[:2].reshape(6),[opening]])
    parts.append([env.applied_action[0],0.,env.applied_action[1]])
    return np.concatenate(parts).astype(np.float32)


def cartesian_to_joint_target(env,action,*,max_ik_residual=.002):
    action=np.asarray(action,dtype=float)
    if action.shape!=(23,) or not np.isfinite(action).all():raise ValueError("Invalid absolute Cartesian action")
    base=env.data.xmat[env.robot_root].reshape(3,3)
    rotation=base@rotation_from_rows(action[13:19])
    origin=env.data.xpos[env.robot_root]+base@action[10:13]
    point=origin+rotation@env.grasp_offset
    joints,error=env.inverse_kinematics(point,rotation,initial=env.data.qpos[env.arm_addresses])
    env.last_cartesian_ik_residual=float(error)
    if error>max_ik_residual:raise ValueError(f"Unreachable model action:{error:.5f}")
    opening=float(np.clip(action[19]*.0975/2,0.,.04875))
    return joints,opening,float(np.clip(action[20],-.12,.12)),float(np.clip(action[22],-.3,.3))
