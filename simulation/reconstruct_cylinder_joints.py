"""Recover robot joint observations by replaying recorded bounded commands.

Checks both recorded robot state and cylinder pose before storing robot-only
joint positions/velocities. It does not create a training export or change the
current policy/action representation. A replay is not a learned task score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from inspect_cylinder import CylinderEnv


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Output exists')
    metadata=json.loads((a.episode/'metadata.json').read_text());r=metadata['result']
    require(r['success'] and r['grip_profile']=='official' and r['geometry_audit_passed'],'Expected successful official demonstration')
    require(r.get('grip_force_target_n') is None,'Probe currently supports recorded position servos only')
    controller=r['controller_parameters'];physics=r['contact_parameters']
    env=CylinderEnv(r['expected_volume_ml'],grasp_height=controller['grasp_height_m'],
        grasp_x=controller['grasp_x_m'],solver_impratio=physics['impratio'],
        physics_dt=physics['physics_timestep'],friction_scale=r['friction_scale'],mass_scale=r['mass_scale'])
    a.output.mkdir(parents=True);started=time.monotonic();positions=[];velocities=[]
    max_robot=max_object=max_quaternion=0.;failure=None
    try:
        env.reset_inspection(seed=r['seed'],jitter=r['position_jitter_m'],yaw_jitter=r['yaw_jitter_rad'])
        initial=a.episode/'starting_integration_state.npz'
        recovery=bool(r.get('initial_state_provenance'))
        if recovery:
            require(initial.is_file(),'Recovery requires its original integration state')
            with np.load(initial) as saved:
                mujoco.mj_setState(env.model,env.data,saved['state'],int(saved['state_spec']))
                env.hold_targets=dict(zip(saved['hold_actuator_ids'].tolist(),saved['hold_targets'].tolist()))
                env.applied_action[:]=saved['applied_base_action']
            mujoco.mj_forward(env.model,env.data);env.data.time=0.
        with np.load(a.episode/'trajectory.npz') as data:
            commands=data['joint_command'];require(commands.shape==(metadata['frames'],9),'Expected bounded9D command records')
            for index,command in enumerate(commands):
                robot=float(abs(env.state_vector()-data['state'][index]).max())
                obj=float(abs(env.data.xpos[env.cylinder_body]-data['cylinder_position'][index]).max())
                q=env.data.xquat[env.cylinder_body];reference=data['cylinder_quaternion'][index]
                quaternion=float(min(abs(q-reference).max(),abs(q+reference).max()))
                max_robot=max(max_robot,robot);max_object=max(max_object,obj);max_quaternion=max(max_quaternion,quaternion)
                if robot>1e-5 or obj>1e-6 or quaternion>1e-6:
                    failure=dict(frame=index,robot_state_max_difference=robot,
                                 object_position_max_difference=obj,object_quaternion_max_difference=quaternion)
                    break
                positions.append(env.data.qpos[env.arm_addresses].copy())
                velocities.append(env.data.qvel[env.arm_dofs].copy())
                env.step_joints(command[:6],opening=float(command[6]),forward=float(command[7]),yaw=float(command[8]))
            expected=len(commands)
        report=dict(kind='recorded_joint_observation_reconstruction',source=str(a.episode.resolve()),
            trajectory_sha256=hashlib.sha256((a.episode/'trajectory.npz').read_bytes()).hexdigest(),
            recovery_initial_state_restored=recovery,frames_expected=expected,frames_reconstructed=len(positions),
            robot_state23_max_difference=max_robot,object_position_max_difference=max_object,
            object_quaternion_max_difference=max_quaternion,passed=failure is None,failure=failure,
            elapsed_wall_seconds=time.monotonic()-started,rendering_used=False,
            new_model_training_started=False,current_training_or_action_representation_modified=False,
            task_success_measured=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if failure is None:
            np.savez_compressed(a.output/'robot_joint_observations.npz',
                                right_qpos=np.asarray(positions),right_qvel=np.asarray(velocities))
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        if failure is not None:raise SystemExit(1)
    finally:env.close()


if __name__=='__main__':main()
