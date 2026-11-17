"""Validate Wall relative rotations and commanded Cartesian labels."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cylinder_actions import commanded_cartesian_action,relative_to_absolute,cartesian_to_joint_target
from inspect_cylinder import CylinderEnv


def validate():
    rng=np.random.default_rng(73);largest=0.
    for _ in range(32):
        state=np.zeros(23);absolute=rng.uniform(-.3,.3,(8,23));relative=absolute.copy()
        for offset in (0,10):
            state[offset:offset+3]=rng.uniform(-.4,.4,3)
            current=Rotation.random(random_state=rng).as_matrix()
            state[offset+3:offset+9]=current[:2].reshape(6)
            targets=Rotation.random(8,random_state=rng).as_matrix()
            absolute[:,offset+3:offset+9]=targets[:,:2].reshape(8,6)
            relative[:,offset:offset+3]=absolute[:,offset:offset+3]-state[offset:offset+3]
            relative[:,offset+3:offset+9]=(targets@current.T)[:,:2].reshape(8,6)
        restored=relative_to_absolute(relative,state)
        largest=max(largest,float(abs(restored-absolute).max()))
    assert largest<1e-6
    env=CylinderEnv(grasp_x=.115)
    try:
        env.reset_inspection()
        before=env.data.qpos.copy()
        for i in range(1,7):
            actuator=env.model.actuator(f"right_arm_joint{i}_servo").id
            env.hold_targets[actuator]+=float(rng.uniform(-.015,.015))
        action=commanded_cartesian_action(env,before)
        joints,opening,v,w=cartesian_to_joint_target(env,action)
        planned=np.array([env.hold_targets[env.model.actuator(f"right_arm_joint{i}_servo").id] for i in range(1,7)])
        data=env.ik_data;data.qpos[:]=before;data.qpos[env.arm_addresses]=joints;mujoco.mj_kinematics(env.model,data)
        p1=data.xpos[env.gripper_body].copy();r1=data.xmat[env.gripper_body].copy()
        data.qpos[env.arm_addresses]=planned;mujoco.mj_kinematics(env.model,data)
        position_error=float(np.linalg.norm(data.xpos[env.gripper_body]-p1))
        rotation_error=float(np.linalg.norm(data.xmat[env.gripper_body]-r1))
        assert position_error<1e-5 and rotation_error<1e-5
        assert abs(opening-.034)<1e-6 and v==0 and w==0
        return dict(passed=True,random_rotation_round_trips=32,maximum_round_trip_error=largest,
                    commanded_fk_position_error_m=position_error,commanded_fk_rotation_error=rotation_error,
                    action_dimension=23,rotation_convention="first two rows; R_abs = R_rel @ R_state")
    finally:env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error("Output exists")
    report=validate();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
