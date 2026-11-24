"""Replay recorded pre-grasp arm commands with selected base commands set to zero.

The experiment freezes the recorded arm targets even after the base moves to a
different pose. It measures physical sensitivity, without running a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_gripper import audit
from carry_env import CONTROL_DT
from cylinder_joint_actions import require, robot_joint_observation
from inspect_cylinder import CylinderEnv


BASE_REPLAYS = {
    'recorded': (1, 1),
    'zero_forward': (0, 1),
    'zero_yaw': (1, 0),
    'zero_base': (0, 0),
}


def replay(case, controller, commands, base_mask, states=None):
    env = CylinderEnv(case['volume'], **controller['environment'],
                      friction_scale=case.get('friction_scale', 1.),
                      mass_scale=case.get('mass_scale', 1.))
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0.))
        geometry = audit(env)
        require(geometry['passed'] and geometry['custom_pads'] == 0, 'Expected official-geometry replay')
        initial_pose = env.base_pose().copy()
        initial_cylinder = env.data.xpos[env.cylinder_body].copy()
        cylinder_dof = env.model.joint('cylinder_free').dofadr[0]
        nearest = None
        max_contacts = 0
        first_loaded = None
        collision = None
        for i, command in enumerate(commands):
            if states is not None:
                np.testing.assert_array_equal(robot_joint_observation(env), states[i])
            forward, yaw = command[7:] * base_mask
            _, metrics = env.step_joints(command[:6], opening=float(command[6]),
                                         forward=float(forward), yaw=float(yaw))
            t = float(env.data.time)
            grasp_distance = float(np.linalg.norm(env.grip_point_relative() - env.grasp_offset))
            if nearest is None or grasp_distance < nearest['grasp_distance_m']:
                nearest = dict(time_s=t, grasp_distance_m=grasp_distance,
                               base_delta_m=(env.base_pose()[:2] - initial_pose[:2]).tolist())
            max_contacts = max(max_contacts, metrics['grip_contacts'])
            if first_loaded is None and max_contacts:
                first_loaded = dict(frame=i + 1, time_s=t, loaded_fingers=max_contacts,
                    grasp_distance_m=grasp_distance, cylinder_position_change_m=float(np.linalg.norm(
                        env.data.xpos[env.cylinder_body] - initial_cylinder)),
                    cylinder_linear_speed_m_s=float(np.linalg.norm(env.data.qvel[cylinder_dof:cylinder_dof + 3])),
                    cylinder_tilt_rad=metrics['bottle_tilt_rad'], forbidden_contact=bool(metrics['forbidden_contacts']))
            if metrics['forbidden_contacts']:
                collision = dict(time_s=t, recorded_geom_names=metrics['forbidden_contacts'])
                break
            if states is None and max_contacts == 2:
                break
        return dict(steps=i + 1, end_time_s=t, nearest=nearest, collision=collision,
                    max_loaded_fingers=max_contacts, first_loaded=first_loaded,
                    final_base_delta_m=(env.base_pose()[:2] - initial_pose[:2]).tolist(),
                    final_base_yaw_delta_deg=float(np.degrees(env.base_pose()[2] - initial_pose[2])),
                    all_recorded_robot_states_exact=states is not None,
                    geometry_audit_passed=geometry['passed'])
    finally:
        env.close()


def measure(folder):
    summary = json.loads((folder / 'summary.json').read_text())
    require(summary['evaluation_complete'] and summary['action_space'] == 'joint'
            and summary['total'] == 1 and summary['successes'] == 0, 'Expected a complete failed joint rollout')
    result, = summary['episodes']
    require(result['geometry_audit']['passed'] and result['geometry_audit']['custom_pads'] == 0,
            'Unexpected source geometry')
    path = folder / 'episode_000' / 'trajectory.npz'
    with np.load(path, allow_pickle=False) as trace:
        commands, states, times = (trace[key] for key in ('joint_command', 'robot_joint_state', 'timestamp'))
        require(len(commands) > 0 and commands.shape == (len(times), 9)
                and states.shape == (len(times), 18) and np.isfinite(commands).all()
                and not trace['contacts'].any()
                and np.allclose(times, np.arange(len(times)) * CONTROL_DT, rtol=0, atol=1e-8),
                'Expected a finite pre-contact command trace from reset')
    variants = {name: replay(result['case'], summary['controller'], commands, mask,
                             states if name == 'recorded' else None)
                for name, mask in BASE_REPLAYS.items()}
    original = variants['recorded']
    reason = result['failure_reason'] or ''
    require(original['steps'] == len(commands) and abs(original['end_time_s'] - result['simulated_seconds']) < 1e-8
            and not original['max_loaded_fingers'], 'Recorded termination did not reproduce')
    if original['collision']:
        require(reason == 'collision:' + str(original['collision']['recorded_geom_names']),
                'Recorded collision did not reproduce')
    else:
        require(not reason.startswith('collision:'), 'Recorded collision missing')
    return dict(rollout=str(folder.resolve()), source_case=result['case'],
                trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), variants=variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new report path')
    report = dict(kind='recorded_pregrasp_base_axis_replay',
                  trials=[measure(folder) for folder in args.rollouts],
                  arm_and_gripper_targets_from_original_trace=True, learned_policy_inference=False,
                  teacher_label_generation=False, task_success_measured=False, independent_cases_read=False)
    with args.output.open('x') as file:
        file.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
