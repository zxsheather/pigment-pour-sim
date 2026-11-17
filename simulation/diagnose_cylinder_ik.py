"""Replay the last default-policy prefix and diagnose its rejected IK target.

Uses recorded neural actions, not new model predictions. Multi-start searches
are diagnostics only; neither execution limits nor task gates are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from cylinder_actions import relative_to_absolute,cartesian_to_joint_target,rotation_from_rows
from inspect_cylinder import CylinderEnv


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--random-seeds',type=int,default=24)
    a=p.parse_args();require(not a.output.exists(),'Output exists')
    require(1<=a.random_seeds<=64,'Invalid seed count')
    result=json.loads((a.episode/'summary.json').read_text())
    aggregate=json.loads((a.episode.parent/'summary.json').read_text())
    require(result['failure_reason'].startswith('Unreachable model action:'),'Expected IK failure')
    require(not aggregate['action_filter_seconds'] and not aggregate['temporal_ensemble_predictions']
            and not aggregate.get('gripper_open_confirmations',0)
            and not any(aggregate['robot_servo_envelope'].values()),'Default execution required')
    case=result['case'];env=CylinderEnv(case['volume'],**aggregate['controller']['environment'],
        friction_scale=case.get('friction_scale',1.),mass_scale=case.get('mass_scale',1.))
    try:
        env.reset_inspection(seed=case['seed'],jitter=case.get('jitter',.004),yaw_jitter=case.get('yaw_jitter',0.))
        with np.load(a.episode/'recovery_physics.npz') as saved:
            mujoco.mj_setState(env.model,env.data,saved['full_state'][-1],int(saved['state_spec']))
            env.hold_targets=dict(zip(saved['hold_actuator_ids'].tolist(),saved['hold_targets'][-1].tolist()))
            env.applied_action[:]=saved['applied_base_action'][-1]
        mujoco.mj_forward(env.model,env.data)
        with np.load(a.episode/'model_observations.npz') as obs:
            anchor_time=float(obs['timestamp'][-1]);anchor=obs['state'][-1,:23]
        absolute=relative_to_absolute(np.load(a.episode/'predicted_chunks.npy')[-1],anchor)
        max_state_error=0.
        with np.load(a.episode/'trajectory.npz') as trajectory:
            indices=np.flatnonzero(trajectory['timestamp']>=anchor_time-1e-7)
            require(len(indices)<aggregate['execution_prefix'],'Expected a rejection within last prefix')
            for tick,index in enumerate(indices):
                max_state_error=max(max_state_error,float(abs(env.state_vector()-trajectory['state'][index]).max()))
                require(np.array_equal(absolute[tick],trajectory['action'][index]),'Recorded command mismatch')
                q,gap,forward,yaw=cartesian_to_joint_target(env,absolute[tick],max_ik_residual=aggregate['ik_max_residual'])
                env.step_joints(q,opening=gap,forward=forward,yaw=yaw)
        require(max_state_error<=1e-5,'Replayed robot state diverged from recorded prefix')
        action=absolute[len(indices)]
        try:cartesian_to_joint_target(env,action,max_ik_residual=aggregate['ik_max_residual'])
        except ValueError:pass
        else:raise ValueError('Rejected command unexpectedly accepted in replay')
        original_error=env.last_cartesian_ik_residual
        require(abs(original_error-result['ik_projection']['max_attempted_residual'])<1e-5,'Replayed IK failure differs')
        base=env.data.xmat[env.robot_root].reshape(3,3)
        rotation=base@rotation_from_rows(action[13:19])
        origin=env.data.xpos[env.robot_root]+base@action[10:13]
        point=origin+rotation@env.grasp_offset
        ids=[env.model.joint(f'right_arm_joint{i}').id for i in range(1,7)]
        require(all(env.model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE for i in ids),'Expected serial hinge arm')
        anchors=env.data.xanchor[ids]
        reach=float(np.linalg.norm(np.diff(anchors,axis=0),axis=1).sum()
                    +np.linalg.norm(env.data.xpos[env.gripper_body]-anchors[-1]))
        origin_excess=max(0.,float(np.linalg.norm(origin-anchors[0]))-reach)
        lower_bound=origin_excess/np.sqrt(1+(np.linalg.norm(env.grasp_offset)/.25)**2)
        data=env.ik_data;data.qpos[:]=env.data.qpos
        def residual(q):
            data.qpos[env.arm_addresses]=q;mujoco.mj_kinematics(env.model,data)
            current=data.xpos[env.gripper_body]+data.xmat[env.gripper_body].reshape(3,3)@env.grasp_offset
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,(rotation@data.xmat[env.gripper_body].reshape(3,3).T).reshape(-1))
            orientation=np.empty(3);mujoco.mju_quat2Vel(orientation,quat,1.)
            return np.r_[point-current,.25*orientation]
        ranges=np.array([env.model.joint(f'right_arm_joint{i}').range for i in range(1,7)])
        current=env.data.qpos[env.arm_addresses].copy();rng=np.random.default_rng(9041785);rows=[]
        for margin in [.005,0.]:
            low,high=ranges[:,0]+margin,ranges[:,1]-margin
            seeds=[current]+[current+rng.normal(0,.15,6) for _ in range(8)]
            seeds.extend(rng.uniform(low,high,(a.random_seeds,6)))
            best=None
            for seed in seeds:
                fit=least_squares(residual,np.clip(seed,low,high),bounds=(low,high),max_nfev=400,
                                  ftol=1e-10,xtol=1e-10,gtol=1e-10)
                if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
            row=dict(joint_limit_margin_rad=margin,seeds=len(seeds),best_weighted_residual=float(np.linalg.norm(best.fun)),
                position_error_m=float(np.linalg.norm(best.fun[:3])),rotation_error_rad=float(np.linalg.norm(best.fun[3:])/.25),
                best_joints_rad=best.x.tolist(),distance_to_bounds_rad=np.minimum(best.x-low,high-best.x).tolist())
            rows.append(row);print(json.dumps(row),flush=True)
        report=dict(kind='recorded_neural_target_ik_diagnosis',episode=str(a.episode.resolve()),
            source_failure_seconds=result['simulated_seconds'],replayed_prefix_steps=len(indices),
            replayed_robot_state_max_difference=max_state_error,reproduced_default_ik_residual=original_error,
            acceptance_weighted_residual=aggregate['ik_max_residual'],multi_start_results=rows,
            serial_arm_origin_reach_upper_bound_m=reach,target_origin_reach_excess_m=origin_excess,
            weighted_residual_geometric_lower_bound=lower_bound,
            geometrically_infeasible_at_original_acceptance=bool(lower_bound>aggregate['ik_max_residual']),
            lower_bound_derivation='Origin error <= point error + norm(grasp_offset)*rotation_angle; Cauchy-Schwarz with rotation weight0.25. Serial hinge-link lengths bound origin reach.',
            physics_replayed=True,new_policy_forward=False,task_success_measured=False,
            controller_or_acceptance_modified=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:env.close()


if __name__=='__main__':main()
