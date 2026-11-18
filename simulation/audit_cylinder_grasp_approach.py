"""Measure approach error on saved joint-policy rollouts and teacher traces."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from cylinder_joint_actions import APERTURE_SCALE_M, require, robot_joint_observation
from inspect_cylinder import CylinderEnv


def wrist_pose_breakdown(env, local_error, initial_base):
    rotation = env.data.xmat[env.gripper_body].reshape(3, 3)
    origin = env.data.xpos[env.gripper_body]
    current_tip = origin + rotation @ env.grasp_offset
    error_world = rotation @ local_error
    object_point = current_tip + error_world
    base = env.base_pose()
    angle = initial_base[2] - base[2]
    c, s = np.cos(angle), np.sin(angle)
    # Rigidly map the measured arm pose onto the reset base pose without a physics step.
    reset_base_tip = current_tip.copy()
    reset_base_tip[:2] = initial_base[:2] + np.array([[c, -s], [s, c]]) @ (current_tip[:2] - base[:2])
    # The official teacher approaches along world +y. Tool roll does not change this axis.
    nominal_shaft = np.array([0., 1., 0.])
    official_tip = origin + nominal_shaft * env.grasp_offset[0]
    return dict(error_world_m=error_world.tolist(),
                base_xy_delta_m=(base[:2] - initial_base[:2]).tolist(),
                base_yaw_delta_deg=float(-np.degrees(angle)),
                wrist_shaft_error_deg=float(np.degrees(np.arccos(np.clip(rotation[:, 0] @ nominal_shaft, -1., 1.)))),
                error_with_reset_base_rigid_pose_m=(object_point - reset_base_tip).tolist(),
                error_with_teacher_shaft_same_wrist_origin_m=(object_point - official_tip).tolist())


def measure_rollout(folder, *, pose_breakdown=False):
    summary = json.loads((folder / 'summary.json').read_text())
    require(summary['evaluation_complete'] and summary['total'] == 1, 'Expected one completed rollout')
    episode = summary['episodes'][0]
    require(episode['action_space'] == 'joint' and episode['geometry_audit']['passed'],
            'Expected an audited joint rollout')
    case = episode['case']
    env = CylinderEnv(case['volume'], friction_scale=case.get('friction_scale', 1),
                      mass_scale=case.get('mass_scale', 1), **summary['controller']['environment'])
    directory = folder / 'episode_000'
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0))
        initial_base = env.base_pose().copy()
        snapshots = []
        with (np.load(directory / 'recovery_physics.npz') as physics,
              np.load(directory / 'model_observations.npz') as observations,
              np.load(directory / 'trajectory.npz') as trajectory):
            require(len(physics['full_state']) == len(observations['state']) == len(observations['timestamp']),
                    'Snapshot and observation counts differ')
            spec = int(physics['state_spec'])
            for state, timestamp, observed in zip(physics['full_state'], observations['timestamp'],
                                                  observations['state']):
                mujoco.mj_setState(env.model, env.data, state, spec)
                mujoco.mj_forward(env.model, env.data)
                np.testing.assert_array_equal(robot_joint_observation(env), observed)
                require(abs(env.data.time - timestamp) < 1e-9, 'Snapshot clock mismatch')
                error = env.grip_point_relative() - env.grasp_offset
                row = dict(time_s=float(timestamp), grasp_distance_m=float(np.linalg.norm(error)),
                                      grasp_point_error_components_m=error.tolist(),
                                      aperture_m=float(observed[6] * APERTURE_SCALE_M),
                                      base_retreat_m=float(initial_base[1] - env.base_pose()[1]))
                if pose_breakdown:
                    row['pose_breakdown'] = wrist_pose_breakdown(env, error, initial_base)
                snapshots.append(row)
            maximum_contacts = int(trajectory['contacts'].max(initial=0))
        require(bool(snapshots), 'No rollout snapshots')
        return dict(rollout=str(folder.resolve()), snapshots=len(snapshots),
                    robot_observations_reproduced_exactly=True, maximum_recorded_loaded_contacts=maximum_contacts,
                    nearest=min(snapshots, key=lambda row: row['grasp_distance_m']), first=snapshots[0],
                    last=snapshots[-1], failure_reason=episode['failure_reason'])
    finally:
        env.close()


def close_references(dataset, volume):
    provenance = json.loads((dataset / 'demonstration_provenance.json').read_text())
    references = []
    for demo in provenance:
        result = demo['metadata']['result']
        if result['expected_volume_ml'] != volume or result.get('initial_state_provenance'):
            continue
        with np.load(Path(demo['source']) / 'trajectory.npz') as raw:
            offset = [result['controller_parameters']['grasp_x_m'], 0, 0]
            positions = raw['grip_point_relative'][raw['phase'] == 'close']
            require(len(positions) > 0, 'Full teacher demonstration has no close phase')
            distances = np.linalg.norm(positions - offset, axis=1)
        references.append(dict(seed=result['seed'], episode=demo['source'],
                               close_distance_m_min=float(distances.min()),
                               close_distance_m_max=float(distances.max()),
                               initial_close_distance_m=float(distances[0])))
    require(bool(references), 'No full training demonstration for the reference volume')
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', nargs='+', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference-volume', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pose-breakdown', action='store_true')
    args = parser.parse_args()
    report = dict(kind='recorded_joint_policy_geometric_approach_diagnostic',
                  metric='Distance from cylinder grasp point to the intended offset in the gripper frame',
                  reference='Recorded close phase of full training demonstrations, for descriptive comparison',
                  teacher_references=close_references(args.dataset, args.reference_volume),
                  policy_results=[measure_rollout(folder, pose_breakdown=args.pose_breakdown)
                                  for folder in args.rollouts],
                  new_policy_inference_run=False, independent_cases_read=False)
    if args.pose_breakdown:
        report['pose_breakdown_method'] = ('Measured world-space error and rigid-pose projections at the '
                                           'same snapshot. Neither projection advances physics or scores a policy.')
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
